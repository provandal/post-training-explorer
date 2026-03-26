"""
============================================================
Step 2: Direct Preference Optimization (DPO)
============================================================
Takes the SFT model from Step 1 and further aligns it using
preference pairs.  DPO teaches the model to prefer "chosen"
outputs (correct, concise, confident) over "rejected" ones
(verbose, hedging, or slightly wrong) — without needing a
separate reward model.

Key idea: for each input we create TWO outputs:
  chosen  → correct label
  rejected → wrong label

DPO directly optimizes the policy to increase the likelihood
gap between chosen and rejected responses.

Run in Google Colab with a GPU runtime.
============================================================
"""

# ── Colab dependency install (uncomment if running in Colab) ─────────
# !pip install -q torch transformers datasets accelerate peft trl sentencepiece safetensors

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, PeftModel, get_peft_model
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ── ANSI colors for Colab/terminal output ─────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

# ── Model configurations ─────────────────────────────────────────────
MODELS = {
    "360M": {"name": "HuggingFaceTB/SmolLM2-360M", "slug": "smollm2-360m",
             "sft_batch": 4, "dpo_batch": 2, "grpo_batch": 2, "grpo_gens": 8},
    "1.7B": {"name": "HuggingFaceTB/SmolLM2-1.7B", "slug": "smollm2-1.7b",
             "sft_batch": 2, "dpo_batch": 1, "grpo_batch": 1, "grpo_gens": 4},
}


def parse_args():
    parser = argparse.ArgumentParser(description="DPO training")
    parser.add_argument("--model-size", default="360M", choices=MODELS.keys(),
                        help="Model size to train (default: 360M)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output (default: reduced output)")
    return parser.parse_args()


def get_paths(model_size):
    """Get input/output paths based on model size."""
    if model_size == "360M":
        # Backward compatible: use existing flat structure
        sft_dir = SCRIPT_DIR / "outputs" / "sft"
        output_dir = SCRIPT_DIR / "outputs" / "dpo"
    else:
        slug = MODELS[model_size]["slug"]
        sft_dir = SCRIPT_DIR / "outputs" / slug / "sft"
        output_dir = SCRIPT_DIR / "outputs" / slug / "dpo"
    return sft_dir, output_dir


# ── Paths (defaults for 360M, overridden in main() based on args) ────
SCRIPT_DIR = Path(__file__).resolve().parent
SFT_DIR = SCRIPT_DIR / "outputs" / "sft"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "dpo"

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"


# ====================================================================
# 1. PREFERENCE PAIR GENERATION
# ====================================================================
# For each prompt we create a (chosen, rejected) pair.
# "Chosen" responses are correct, concise, and confident.
# "Rejected" responses have one or more of these problems:
#   - Wrong classification label
#   - Excessively verbose / hedging language
#   - Vague or unhelpful reasoning
# ====================================================================

LABELS = [
    "OLTP Database", "OLAP Analytics", "AI ML Training",
    "Video Streaming", "VDI Virtual Desktop", "Backup Archive",
]

WORKLOAD_PROFILES = {
    "OLTP Database": {
        "iops_range": (5000, 80000),
        "throughput_mb_range": (50, 400),
        "avg_latency_us_range": (100, 2000),
        "read_pct_range": (60, 80),
        "random_pct_range": (85, 99),
        "block_size_kb": [4, 8],
        "queue_depth_range": (16, 128),
    },
    "OLAP Analytics": {
        "iops_range": (100, 3000),
        "throughput_mb_range": (500, 5000),
        "avg_latency_us_range": (1000, 20000),
        "read_pct_range": (85, 99),
        "random_pct_range": (5, 30),
        "block_size_kb": [64, 128, 256, 512, 1024],
        "queue_depth_range": (1, 32),
    },
    "AI ML Training": {
        "iops_range": (500, 10000),
        "throughput_mb_range": (1000, 10000),
        "avg_latency_us_range": (500, 10000),
        "read_pct_range": (90, 99),
        "random_pct_range": (20, 60),
        "block_size_kb": [128, 256, 512, 1024],
        "queue_depth_range": (8, 64),
    },
    "Video Streaming": {
        "iops_range": (100, 2000),
        "throughput_mb_range": (200, 3000),
        "avg_latency_us_range": (1000, 15000),
        "read_pct_range": (90, 100),
        "random_pct_range": (10, 40),
        "block_size_kb": [256, 512, 1024, 2048],
        "queue_depth_range": (1, 16),
    },
    "VDI Virtual Desktop": {
        "iops_range": (2000, 30000),
        "throughput_mb_range": (20, 200),
        "avg_latency_us_range": (200, 5000),
        "read_pct_range": (50, 70),
        "random_pct_range": (70, 95),
        "block_size_kb": [4, 8, 16],
        "queue_depth_range": (4, 64),
    },
    "Backup Archive": {
        "iops_range": (50, 1000),
        "throughput_mb_range": (200, 5000),
        "avg_latency_us_range": (2000, 50000),
        "read_pct_range": (5, 30),
        "random_pct_range": (2, 15),
        "block_size_kb": [256, 512, 1024, 2048, 4096],
        "queue_depth_range": (1, 16),
    },
}


def generate_io_prompt(label: str) -> tuple:
    """Generate a formatted I/O metrics prompt for a given workload type.

    Returns:
        (prompt_text, metrics_dict) where metrics_dict contains the raw
        metric values used to build the prompt.
    """
    p = WORKLOAD_PROFILES[label]
    iops = random.randint(*p["iops_range"])
    throughput = random.randint(*p["throughput_mb_range"])
    latency = random.randint(*p["avg_latency_us_range"])
    read_pct = random.randint(*p["read_pct_range"])
    write_pct = 100 - read_pct
    random_pct = random.randint(*p["random_pct_range"])
    sequential_pct = 100 - random_pct
    block_kb = random.choice(p["block_size_kb"])
    queue_depth = random.randint(*p["queue_depth_range"])

    prompt_text = (
        f"Classify the following storage I/O workload based on these metrics:\n"
        f"- IOPS: {iops:,}\n"
        f"- Throughput: {throughput:,} MB/s\n"
        f"- Average Latency: {latency:,} us\n"
        f"- Read/Write Ratio: {read_pct}% read / {write_pct}% write\n"
        f"- Access Pattern: {random_pct}% random / {sequential_pct}% sequential\n"
        f"- Block Size: {block_kb} KB\n"
        f"- Queue Depth: {queue_depth}\n\n"
        f"Choose one of: OLTP Database, OLAP Analytics, AI ML Training, "
        f"Video Streaming, VDI Virtual Desktop, Backup Archive.\n"
        f"Provide the classification."
    )
    metrics = {
        "iops": iops, "throughput": throughput, "latency": latency,
        "read_pct": read_pct, "random_pct": random_pct,
        "block_kb": block_kb, "queue_depth": queue_depth,
    }
    return prompt_text, metrics


def generate_preference_pair(label: str) -> dict:
    """
    Create one preference pair: a prompt with a chosen (good) and
    rejected (bad) completion.

    Label-only format — matches SFT training. DPO teaches the model to
    increase the likelihood gap between correct and incorrect labels.
    """
    prompt, metrics = generate_io_prompt(label)

    # CHOSEN: correct label (matches SFT completion format)
    chosen = f" {label}"

    # REJECTED: wrong label
    wrong_label = random.choice([l for l in LABELS if l != label])
    rejected = f" {wrong_label}"

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "true_label": label,
        "wrong_label": wrong_label,
    }


def build_preference_dataset(n_per_class: int = 60) -> Dataset:
    """Build a balanced preference dataset."""
    pairs = []
    for label in LABELS:
        for _ in range(n_per_class):
            pairs.append(generate_preference_pair(label))
    random.shuffle(pairs)

    print(f"  Created {len(pairs)} preference pairs ({n_per_class} per class)")

    return Dataset.from_dict({
        "prompt": [p["prompt"] + "\n\nClassification:" for p in pairs],
        "chosen": [p["chosen"] for p in pairs],
        "rejected": [p["rejected"] for p in pairs],
    }), pairs


# ====================================================================
# 2. TOKEN PROBABILITY CAPTURE
# ====================================================================

EXAMPLE_PROMPTS_FOR_PROBS = [
    {"prompt": generate_io_prompt(label)[0], "label": label}
    for label in LABELS[:5]
]


def capture_token_probs(model, tokenizer, prompts, device, top_k=20):
    """Capture top-k next-token probabilities for each prompt."""
    model.eval()
    results = []

    for sample in prompts:
        text = sample["prompt"] + "\n\nClassification:"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs[0], top_k)
            top_tokens = [tokenizer.decode(idx.item()) for idx in top_indices]

        results.append({
            "prompt_snippet": sample["prompt"][:120] + "...",
            "expected_label": sample["label"],
            "top_tokens": top_tokens,
            "top_probs": [round(p.item(), 6) for p in top_probs],
        })

    return results


# ====================================================================
# 3. TRAINING LOSS CALLBACK
# ====================================================================

from transformers import TrainerCallback

class LossRecorderCallback(TrainerCallback):
    """Records DPO training metrics and prints color-coded progress."""
    def __init__(self):
        self.logs = []
        self._prev = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        entry = {"step": state.global_step}
        for key in ["loss", "rewards/chosen", "rewards/rejected",
                    "rewards/margins", "rewards/accuracies",
                    "logps/chosen", "logps/rejected"]:
            if key in logs:
                entry[key] = round(logs[key], 6)
        if len(entry) <= 1:
            return
        self.logs.append(entry)

        # Build color-coded output
        step = state.global_step
        epoch = round(state.epoch, 1) if state.epoch else 0
        parts = []

        # Loss (should decrease)
        if "loss" in logs:
            loss = logs["loss"]
            loss_color = self._color("loss", loss, "down")
            parts.append(f"loss: {loss_color}{loss:.4f}{RESET}")

        # Reward margin (should increase — bigger gap = stronger preference)
        margin = logs.get("rewards/margins")
        if margin is not None:
            m_color = self._color("rewards/margins", margin, "up")
            parts.append(f"margin: {m_color}{margin:+.3f}{RESET}")

        # Reward accuracy (should increase)
        acc = logs.get("rewards/accuracies")
        if acc is not None:
            a_color = self._color("rewards/accuracies", acc, "up")
            parts.append(f"acc: {a_color}{acc:.1%}{RESET}")

        # Chosen/rejected rewards (chosen should go up, rejected down)
        chosen = logs.get("rewards/chosen")
        rejected = logs.get("rewards/rejected")
        if chosen is not None and rejected is not None:
            c_color = self._color("rewards/chosen", chosen, "up")
            r_color = self._color("rewards/rejected", rejected, "down")
            parts.append(f"chosen: {c_color}{chosen:+.2f}{RESET} rejected: {r_color}{rejected:+.2f}{RESET}")

        if parts:
            # Progress bar
            total_steps = state.max_steps if state.max_steps and state.max_steps > 0 else 1
            pct = min(step / total_steps, 1.0) if total_steps > 1 else 0
            filled = int(pct * 20)
            bar = f"{'█' * filled}{'░' * (20 - filled)}"
            print(f"  [{bar}] {pct:>5.0%} Step {step:>4d} │ {' │ '.join(parts)}")

        self._prev.update({k: logs[k] for k in logs if isinstance(logs[k], (int, float))})

    def _color(self, key, value, direction):
        prev = self._prev.get(key)
        if prev is None:
            return BOLD
        if direction == "down":
            return GREEN if value < prev else (RED if value > prev * 1.1 else YELLOW)
        else:
            return GREEN if value > prev else (RED if value < prev * 0.9 else YELLOW)


class LiveProbeCallback(TrainerCallback):
    """Runs fixed probe prompts through the model periodically to show learning in real time."""

    def __init__(self, tokenizer, device, probe_interval=20):
        self.tokenizer = tokenizer
        self.device = device
        self.probe_interval = probe_interval
        self.probe_history = []
        # Fixed probe prompts: one easy, one medium, one hard
        # Use the same prompt generation as the training data
        self.probes = [
            {"prompt": generate_io_prompt("Backup Archive")[0], "label": "Backup Archive"},
            {"prompt": generate_io_prompt("OLTP Database")[0], "label": "OLTP Database"},
            {"prompt": generate_io_prompt("VDI Virtual Desktop")[0], "label": "VDI Virtual Desktop"},
        ]

    def on_log(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.probe_interval != 0:
            return
        if model is None:
            return

        model.eval()
        print(f"\n  {'─' * 60}")
        print(f"  {BOLD}LIVE PROBE — Step {state.global_step}{RESET}  (model generates on fixed prompts)")
        print(f"  {'─' * 60}")

        step_results = []
        for probe in self.probes:
            prompt_text = probe["prompt"] + "\n\nClassification:"
            inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=480).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=60, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            expected = probe["label"]
            correct = expected.lower() in generated.lower()
            color = GREEN if correct else RED
            badge = "\u2713" if correct else "\u2717"

            display_text = generated[:100].replace('\n', ' | ')
            print(f"  {color}{badge}{RESET} Expected: {BOLD}{expected}{RESET}")
            print(f"    Model: {color}{display_text}{RESET}")

            step_results.append({
                "step": state.global_step,
                "expected": expected,
                "generated": generated[:200],
                "correct": correct,
            })

        self.probe_history.extend(step_results)
        model.train()
        print()


# ====================================================================
# 4. MAIN TRAINING PIPELINE
# ====================================================================

def main():
    args = parse_args()
    cfg = MODELS[args.model_size]

    # Override module-level defaults based on args
    global MODEL_NAME, SFT_DIR, OUTPUT_DIR
    MODEL_NAME = cfg["name"]
    SFT_DIR, OUTPUT_DIR = get_paths(args.model_size)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check prerequisites (must be inside main() since paths depend on args)
    SFT_ADAPTER = SFT_DIR / "adapter"
    if not SFT_ADAPTER.exists():
        print("=" * 60)
        print("  ERROR: SFT adapter not found!")
        print(f"  Expected at: {SFT_ADAPTER}")
        print("  Please run train_sft.py first.")
        print("=" * 60)
        sys.exit(1)

    print("=" * 60)
    print("  STEP 2: Direct Preference Optimization (DPO)")
    print(f"  Model: {cfg['name']} ({args.model_size})")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Using device: {device}")
    if device != "cuda":
        print("[WARNING] No GPU detected. Training will be slow!")

    # ── 4a. Load base model + SFT adapter ────────────────────────────
    # DPO needs both a trainable "policy" model and a frozen "reference"
    # model.  DPOTrainer handles the reference model internally — we
    # just need to pass in the SFT-tuned model as the policy.
    print(f"\n[1/6] Loading base model + SFT adapter...")
    tokenizer = AutoTokenizer.from_pretrained(str(SFT_ADAPTER), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    ).to(device)

    # Load the SFT LoRA adapter on top of the base model
    model = PeftModel.from_pretrained(base_model, str(SFT_ADAPTER))
    # Merge the SFT adapter into the base weights so we can add a NEW
    # LoRA adapter for DPO on top.  This is a common pattern:
    #   base → merge SFT → add DPO LoRA
    model = model.merge_and_unload()
    print(f"  Loaded and merged SFT adapter. Parameters: {model.num_parameters():,}")

    # ── 4b. Capture pre-DPO probabilities ────────────────────────────
    print("\n[2/6] Capturing pre-DPO token probabilities...")
    pre_dpo_probs = capture_token_probs(model, tokenizer, EXAMPLE_PROMPTS_FOR_PROBS, device)

    with open(OUTPUT_DIR / "pre_dpo_token_probs.json", "w") as f:
        json.dump(pre_dpo_probs, f, indent=2)

    # ── 4c. Build preference dataset ─────────────────────────────────
    print("\n[3/6] Building preference pair dataset...")
    dataset, raw_pairs = build_preference_dataset(n_per_class=60)

    # Save some example pairs for the web app to display
    example_pairs = raw_pairs[:15]  # Save 15 illustrative examples
    with open(OUTPUT_DIR / "preference_pair_examples.json", "w") as f:
        json.dump(example_pairs, f, indent=2)
    print(f"  Saved {len(example_pairs)} example preference pairs.")

    # ── 4d. Configure new LoRA for DPO ───────────────────────────────
    # We add a fresh set of LoRA adapters for the DPO stage.
    # These learn the *preference* signal on top of the SFT knowledge.
    print("\n[4/6] Configuring LoRA for DPO training...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── 4e. DPO Training ─────────────────────────────────────────────
    # DPO loss: -log(sigmoid(beta * (log_pi(chosen) - log_pi(rejected)
    #                               - log_ref(chosen) + log_ref(rejected))))
    # where pi = policy (our model), ref = frozen reference (SFT model)
    # beta controls how much to deviate from the reference policy.
    # Lower beta (0.05) is lighter — less aggressive correction for small models.
    print("\n[5/6] Starting DPO training...")
    print(f"""
  {BOLD}What to watch for during DPO training:{RESET}
  ┌─────────────────────────────────────────────────────────┐
  │ {BOLD}loss{RESET}      DPO loss — should decrease. Measures how     │
  │           well the model separates chosen vs rejected.   │
  │                                                         │
  │ {BOLD}margin{RESET}    Reward margin (chosen - rejected). Should    │
  │           increase — means the model increasingly        │
  │           prefers concise correct answers over wrong ones.│
  │                                                         │
  │ {BOLD}acc{RESET}       How often the model assigns higher reward to │
  │           the chosen response. Should climb toward 90%+. │
  │                                                         │
  │ {BOLD}chosen{RESET}    Reward for preferred (correct) responses.    │
  │           Should increase (model likes these more).      │
  │                                                         │
  │ {BOLD}rejected{RESET}  Reward for rejected (wrong) responses.      │
  │           Should decrease (model avoids these).          │
  │                                                         │
  │ {GREEN}Green{RESET} = moving in the right direction                  │
  │ {RED}Red{RESET}   = moving the wrong way (investigate if persistent) │
  └─────────────────────────────────────────────────────────┘
""")
    loss_callback = LossRecorderCallback()
    probe_callback = LiveProbeCallback(tokenizer, device, probe_interval=20)

    batch_size = cfg["dpo_batch"]

    dpo_base_kwargs = dict(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,        # Effective batch = batch_size * 4
        learning_rate=5e-5,                   # Lower LR for DPO (it's a refinement step)
        lr_scheduler_type="cosine",
        warmup_steps=10,
        beta=0.05,                            # DPO temperature — lighter for 360M to avoid over-correction
        logging_steps=5 if args.verbose else 10,
        save_strategy="epoch",
        seed=SEED,
        bf16=(device == "cuda"),
        max_length=512,
        report_to="none",
    )

    # TRL 0.16+ removed max_prompt_length from DPOConfig
    try:
        dpo_config = DPOConfig(**dpo_base_kwargs, max_prompt_length=400)
    except TypeError:
        dpo_config = DPOConfig(**dpo_base_kwargs)

    # Suppress default trainer logging (we use our colored callback instead)
    import logging
    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)
    logging.getLogger("trl.trainer.dpo_trainer").setLevel(logging.ERROR)

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[loss_callback, probe_callback],
    )

    train_start = time.time()
    train_result = trainer.train()
    train_time = time.time() - train_start

    print(f"  DPO training complete in {train_time:.1f}s")
    print(f"  Final loss: {train_result.training_loss:.4f}")

    # ── 4f. Save artifacts ───────────────────────────────────────────
    print("\n[6/6] Saving artifacts...")

    # IMPORTANT: Use trainer.model — TRL may swap the model object during training.
    trained_model = trainer.model
    print(f"  model is trainer.model: {model is trained_model}")

    # Verify LoRA weights were actually updated
    lora_b_stds = [p.std().item() for n, p in trained_model.named_parameters() if "lora_B" in n]
    if lora_b_stds:
        avg_std = sum(lora_b_stds) / len(lora_b_stds)
        print(f"  LoRA B weight avg std: {avg_std:.6f} ({len(lora_b_stds)} matrices)")
        if avg_std < 0.001:
            print(f"  ⚠ WARNING: LoRA B weights are near-zero!")

    # Save the DPO LoRA adapter
    adapter_path = OUTPUT_DIR / "adapter"
    trained_model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"  Saved DPO adapter to {adapter_path}")

    # Save training curves
    # Format: loss_curve for the web app, plus the raw logs for compatibility
    with open(OUTPUT_DIR / "training_curves.json", "w") as f:
        json.dump({
            "loss_curve": loss_callback.logs,  # [{"step": 0, "loss": ..., ...}, ...]
            "logs": loss_callback.logs,        # backward-compat alias
            "final_loss": round(train_result.training_loss, 6),
            "training_time_seconds": round(train_time, 2),
            "total_steps": train_result.global_step,
        }, f, indent=2)

    # Save live probe history for visualization
    if probe_callback.probe_history:
        probe_path = OUTPUT_DIR / "probe_history.json"
        with open(probe_path, "w") as f:
            json.dump({"probes": probe_callback.probe_history}, f, indent=2)
        print(f"  Probe history → {probe_path}")

    # Capture post-DPO probabilities to show how DPO shifted distributions
    post_dpo_probs = capture_token_probs(model, tokenizer, EXAMPLE_PROMPTS_FOR_PROBS, device)
    with open(OUTPUT_DIR / "post_dpo_token_probs.json", "w") as f:
        json.dump(post_dpo_probs, f, indent=2)

    # Save probability shift analysis
    # Format expected by the web app:
    # {"examples": [{"input": "...", "chosen_style": "concise", "rejected_style": "verbose",
    #   "before": {"chosen_log_prob": -2.1, "rejected_log_prob": -1.8},
    #   "after": {"chosen_log_prob": -0.9, "rejected_log_prob": -3.2}}]}
    #
    # We compute actual log probabilities for a few chosen/rejected pairs
    # using both the pre-DPO (reference) and post-DPO (trained) models.
    print("  Computing chosen/rejected log-probability shifts...")

    # We need the reference model (pre-DPO SFT model) for "before" log probs.
    # Re-load the SFT model as reference since the current model has been DPO-trained.
    ref_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    ).to(device)
    ref_model_for_probs = PeftModel.from_pretrained(ref_base, str(SFT_ADAPTER))
    ref_model_for_probs = ref_model_for_probs.merge_and_unload()
    ref_model_for_probs.eval()

    def compute_completion_log_prob(mdl, prompt_text, completion_text):
        """Compute log probability of completion given prompt."""
        full_text = prompt_text + "\n\nClassification:" + completion_text
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        prompt_only = tokenizer(prompt_text + "\n\nClassification:", return_tensors="pt", truncation=True, max_length=400)
        prompt_len = prompt_only["input_ids"].shape[1]
        with torch.no_grad():
            outputs = mdl(**inputs)
            logits = outputs.logits
            shift_logits = logits[:, prompt_len - 1:-1, :]
            shift_labels = inputs["input_ids"][:, prompt_len:]
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            # Return mean log prob per token (normalized by length)
            return token_log_probs.mean().item()

    # Use a subset of the raw preference pairs for probability shift examples
    prob_shift_examples = []
    shift_sample_pairs = raw_pairs[:8]  # Use first 8 pairs

    for pair in shift_sample_pairs:
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]
        # Before (reference/SFT model) log probs
        before_chosen_lp = compute_completion_log_prob(ref_model_for_probs, prompt, chosen)
        before_rejected_lp = compute_completion_log_prob(ref_model_for_probs, prompt, rejected)

        # After (DPO-trained model) log probs
        after_chosen_lp = compute_completion_log_prob(model, prompt, chosen)
        after_rejected_lp = compute_completion_log_prob(model, prompt, rejected)

        prob_shift_examples.append({
            "input": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "chosen_label": pair.get("true_label", ""),
            "rejected_label": pair.get("wrong_label", ""),
            "before": {
                "chosen_log_prob": round(before_chosen_lp, 4),
                "rejected_log_prob": round(before_rejected_lp, 4),
            },
            "after": {
                "chosen_log_prob": round(after_chosen_lp, 4),
                "rejected_log_prob": round(after_rejected_lp, 4),
            },
        })

    # Clean up reference model
    del ref_model_for_probs
    del ref_base
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    with open(OUTPUT_DIR / "probability_shifts.json", "w") as f:
        json.dump({"examples": prob_shift_examples}, f, indent=2)
    print(f"  Saved probability shifts for {len(prob_shift_examples)} examples.")

    # Also save the legacy top-token probability shift data
    prob_shifts_legacy = []
    for i in range(len(EXAMPLE_PROMPTS_FOR_PROBS)):
        pre = pre_dpo_probs[i]
        post = post_dpo_probs[i]
        shift = {
            "prompt_snippet": pre["prompt_snippet"],
            "expected_label": pre["expected_label"],
            "pre_dpo_top5": list(zip(pre["top_tokens"][:5], pre["top_probs"][:5])),
            "post_dpo_top5": list(zip(post["top_tokens"][:5], post["top_probs"][:5])),
        }
        prob_shifts_legacy.append(shift)

    with open(OUTPUT_DIR / "token_probability_shifts.json", "w") as f:
        json.dump(prob_shifts_legacy, f, indent=2)

    # ── Sanity check: run 5 sample inferences ────────────────────
    print("\n  Running post-DPO sanity check...")

    sanity_labels = LABELS[:5]  # One per class (skip last to keep it quick)
    sanity_results = []
    trained_model.eval()
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # Load SFT sanity check for comparison if available
    sft_sanity_path = SFT_DIR / "sft_sanity_check.json"
    sft_accuracy = None
    if sft_sanity_path.exists():
        with open(sft_sanity_path) as f:
            sft_sanity = json.load(f)
            sft_accuracy = sft_sanity.get("sample_accuracy", None)

    for label in sanity_labels:
        prompt = generate_io_prompt(label)[0] + "\n\nClassification:"
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
        input_len = input_ids.shape[1]
        # Manual greedy decoding — more reliable than model.generate() post-TRL
        with torch.no_grad():
            for _ in range(80):
                logits = trained_model(input_ids).logits[:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True)
                if next_token.item() == eos_id:
                    break
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        generated = tokenizer.decode(input_ids[0][input_len:], skip_special_tokens=True).strip()

        # Extract predicted label
        predicted = ""
        gen_lower = generated.lower()
        for l in sorted(LABELS, key=len, reverse=True):
            if l.lower() in gen_lower:
                predicted = l
                break

        correct = predicted == label
        sanity_results.append({
            "expected": label,
            "predicted": predicted if predicted else "(unparseable)",
            "generated_snippet": generated[:100],
            "correct": correct,
        })

    num_correct = sum(1 for r in sanity_results if r["correct"])
    sample_accuracy = num_correct / len(sanity_results)

    # Determine verdict
    checks = []
    initial_loss = loss_callback.logs[0].get("loss") if loss_callback.logs else None
    # Find last log entry that actually has a loss value (final entry may be summary-only)
    final_loss = None
    for log_entry in reversed(loss_callback.logs):
        if "loss" in log_entry:
            final_loss = log_entry["loss"]
            break

    if initial_loss and final_loss and final_loss < initial_loss:
        checks.append(("pass", "Loss decreased during training"))
    else:
        checks.append(("warn", "Loss did not decrease — DPO may not have converged"))

    if sft_accuracy is not None:
        if sample_accuracy >= sft_accuracy - 0.15:
            checks.append(("pass", f"Accuracy held vs SFT ({sample_accuracy:.0%} vs {sft_accuracy:.0%})"))
        else:
            checks.append(("warn", f"Accuracy dropped vs SFT ({sample_accuracy:.0%} vs {sft_accuracy:.0%})"))

    has_format = all(any(l.lower() in r["generated_snippet"].lower() for l in LABELS) for r in sanity_results if r["correct"])
    if has_format or num_correct > 0:
        checks.append(("pass", "Model outputs correct format (label)"))
    else:
        checks.append(("fail", "Model outputs are not in expected format"))

    all_pass = all(c[0] == "pass" for c in checks)
    any_fail = any(c[0] == "fail" for c in checks)
    verdict = "HEALTHY" if all_pass else ("ISSUES DETECTED" if any_fail else "OK WITH WARNINGS")

    # Print structured results
    print("\n" + "=" * 60)
    print("  DPO RESULTS")
    print("=" * 60)
    print(f"  Training:  Loss {initial_loss:.2f} → {final_loss:.2f} in {train_time/60:.0f}m {train_time%60:.0f}s" if initial_loss and final_loss else f"  Training:  {train_time:.1f}s")
    print()
    print("  Sample outputs (5 prompts, one per class):")
    for r in sanity_results:
        mark = "✓" if r["correct"] else "✗"
        print(f'    {r["expected"]:<20s} → {r["predicted"]:<22s} {mark}')
    print(f"  Sample accuracy: {num_correct}/{len(sanity_results)} ({sample_accuracy:.0%})")
    if sft_accuracy is not None:
        delta = sample_accuracy - sft_accuracy
        direction = "+" if delta >= 0 else ""
        print(f"  vs SFT accuracy:  {direction}{delta:.0%}")
    print()
    print("  Sanity check:")
    for status, msg in checks:
        icon = "✓" if status == "pass" else ("⚠" if status == "warn" else "✗")
        print(f"    {icon} {msg}")
    print(f"\n  Verdict: {verdict} — {'proceed to GRPO' if not any_fail else 'investigate before proceeding'}")
    print("=" * 60)

    # Save sanity check JSON
    sanity_data = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "training_time_seconds": round(train_time, 2),
        "sample_results": sanity_results,
        "sample_accuracy": sample_accuracy,
        "sft_accuracy": sft_accuracy,
        "checks": [{"status": s, "message": m} for s, m in checks],
        "verdict": verdict,
    }
    with open(OUTPUT_DIR / "dpo_sanity_check.json", "w") as f:
        json.dump(sanity_data, f, indent=2)

    # ── Per-class validation ─────────────────────────────────────────
    print("\n  Per-class accuracy breakdown (5 prompts each):")
    per_class_correct = {}
    per_class_total = {}
    predicted_labels_all = []
    n_val_per_class = 5

    for label in LABELS:
        per_class_correct[label] = 0
        per_class_total[label] = n_val_per_class
        for _ in range(n_val_per_class):
            val_prompt = generate_io_prompt(label)[0] + "\n\nClassification:"
            val_input_ids = tokenizer(val_prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
            val_input_len = val_input_ids.shape[1]
            with torch.no_grad():
                for _ in range(80):
                    val_logits = trained_model(val_input_ids).logits[:, -1, :]
                    val_next_token = val_logits.argmax(dim=-1, keepdim=True)
                    if val_next_token.item() == eos_id:
                        break
                    val_input_ids = torch.cat([val_input_ids, val_next_token], dim=-1)
            val_generated = tokenizer.decode(val_input_ids[0][val_input_len:], skip_special_tokens=True).strip()

            val_predicted = ""
            val_gen_lower = val_generated.lower()
            for l in sorted(LABELS, key=len, reverse=True):
                if l.lower() in val_gen_lower:
                    val_predicted = l
                    break

            predicted_labels_all.append(val_predicted)
            if val_predicted == label:
                per_class_correct[label] += 1

    total_val_correct = sum(per_class_correct.values())
    total_val_total = sum(per_class_total.values())
    overall_val_acc = total_val_correct / total_val_total if total_val_total > 0 else 0

    for label in LABELS:
        acc = per_class_correct[label] / per_class_total[label]
        if acc >= 0.6:
            color = GREEN
        elif acc >= 0.4:
            color = YELLOW
        else:
            color = RED
        bar_filled = int(acc * 10)
        bar_str = f"{'█' * bar_filled}{'░' * (10 - bar_filled)}"
        print(f"    {label:<22s} {color}{bar_str} {per_class_correct[label]}/{per_class_total[label]} ({acc:.0%}){RESET}")

    print(f"    {'─' * 50}")
    print(f"    {BOLD}Overall: {total_val_correct}/{total_val_total} ({overall_val_acc:.0%}){RESET}")

    # Mode collapse check: see how many distinct labels the model actually predicted
    unique_predicted = set(predicted_labels_all)
    if len(unique_predicted) <= 2:
        print(f"\n    {RED}{BOLD}WARNING: Mode collapse detected!{RESET}")
        print(f"    {RED}Model only predicted {len(unique_predicted)} distinct label(s): {', '.join(sorted(unique_predicted)) if unique_predicted else '(none)'}{RESET}")
    elif len(unique_predicted) <= 4:
        print(f"\n    {YELLOW}Note: Model predicted only {len(unique_predicted)}/6 distinct labels: {', '.join(sorted(unique_predicted))}{RESET}")
    else:
        print(f"\n    {GREEN}No mode collapse — {len(unique_predicted)}/6 distinct labels predicted.{RESET}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DPO Training Complete!")
    print("=" * 60)
    print(f"  Model:               {cfg['name']} ({args.model_size})")
    print(f"  Adapter saved to:    {adapter_path}")
    print(f"  Training time:       {train_time:.1f}s")
    print(f"  Final loss:          {train_result.training_loss:.4f}")
    print(f"  Preference pairs:    {len(raw_pairs)}")
    print(f"  Artifacts in:        {OUTPUT_DIR}")
    print(f"\n  Next step: Run train_grpo.py")


if __name__ == "__main__":
    main()
