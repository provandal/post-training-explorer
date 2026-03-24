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
  chosen  → correct classification, concise reasoning
  rejected → wrong class, excessive hedging, or verbose

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

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SFT_DIR = SCRIPT_DIR / "outputs" / "sft"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "dpo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

# ── Check prerequisites ─────────────────────────────────────────────
SFT_ADAPTER = SFT_DIR / "adapter"
if not SFT_ADAPTER.exists():
    print("=" * 60)
    print("  ERROR: SFT adapter not found!")
    print(f"  Expected at: {SFT_ADAPTER}")
    print("  Please run train_sft.py first.")
    print("=" * 60)
    sys.exit(1)


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

# Good (concise, correct) reasons for each workload
GOOD_REASONS = {
    "OLTP Database": [
        "High random IOPS with small block sizes indicate transactional database operations.",
        "Small block random reads dominate, consistent with index lookups and row fetches.",
    ],
    "OLAP Analytics": [
        "Large sequential reads with high throughput indicate analytical table scans.",
        "Predominantly sequential read pattern with large blocks suggests data warehouse queries.",
    ],
    "AI ML Training": [
        "High throughput reads with mixed access patterns suggest training data pipeline loading.",
        "Large block reads with very high throughput indicate GPU training data ingestion.",
    ],
    "Video Streaming": [
        "Steady sequential reads with large block sizes indicate media streaming operations.",
        "Consistent high-throughput sequential reads suggest video file delivery.",
    ],
    "VDI Virtual Desktop": [
        "Mixed read/write random I/O with small blocks indicates virtual desktop user activity.",
        "Balanced read-write ratio with small random I/O suggests concurrent desktop sessions.",
    ],
    "Backup Archive": [
        "Large sequential writes with high throughput indicate backup data ingestion.",
        "Write-dominant sequential pattern with large block sizes suggests archival operations.",
    ],
}

# Bad reasons — hedging, verbose, or vague
BAD_REASON_TEMPLATES = [
    "It could possibly be {label}, but I'm not entirely sure. The metrics seem to maybe suggest something along those lines, though there are other possibilities to consider. The IOPS and throughput values are somewhat consistent with what you might expect, but further analysis would be needed to confirm this assessment definitively.",
    "Well, looking at these numbers, I think it might be {label}? The data is a bit ambiguous and could point to several different workload types. Without more context about the specific environment and additional metrics, it's hard to say with complete confidence.",
    "Based on my analysis of the provided metrics, taking into account the various factors including but not limited to the IOPS values, throughput measurements, latency characteristics, read/write distribution, and access patterns, I would tentatively suggest that this could potentially be classified as {label}, although other interpretations are certainly possible.",
]


def generate_io_prompt(label: str) -> str:
    """Generate a formatted I/O metrics prompt for a given workload type."""
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

    return (
        f"Classify the following storage I/O workload based on these metrics:\n"
        f"- IOPS: {iops:,}\n"
        f"- Throughput: {throughput:,} MB/s\n"
        f"- Average Latency: {latency:,} us\n"
        f"- Read/Write Ratio: {read_pct}% read / {write_pct}% write\n"
        f"- Access Pattern: {random_pct}% random / {sequential_pct}% sequential\n"
        f"- Block Size: {block_kb} KB\n"
        f"- Queue Depth: {queue_depth}\n\n"
        f"Provide the workload classification and a brief reason."
    )


def generate_preference_pair(label: str) -> dict:
    """
    Create one preference pair: a prompt with a chosen (good) and
    rejected (bad) completion.

    Rejection strategies (randomly selected):
    1. Wrong label with confident reasoning
    2. Correct label but excessively verbose/hedging
    3. Wrong label with hedging
    """
    prompt = generate_io_prompt(label)

    # CHOSEN: correct, concise, confident
    chosen = f"Classification: {label}\nReason: {random.choice(GOOD_REASONS[label])}"

    # REJECTED: pick a strategy
    strategy = random.choice(["wrong_label", "verbose_correct", "wrong_and_hedging"])

    if strategy == "wrong_label":
        # Pick a wrong label and give a confident (but wrong) answer
        wrong_label = random.choice([l for l in LABELS if l != label])
        wrong_reason = random.choice(GOOD_REASONS[wrong_label])
        rejected = f"Classification: {wrong_label}\nReason: {wrong_reason}"
    elif strategy == "verbose_correct":
        # Correct label but excessively verbose / hedging
        template = random.choice(BAD_REASON_TEMPLATES)
        rejected = f"Classification: {label}\nReason: {template.format(label=label)}"
    else:  # wrong_and_hedging
        wrong_label = random.choice([l for l in LABELS if l != label])
        template = random.choice(BAD_REASON_TEMPLATES)
        rejected = f"Classification: {wrong_label}\nReason: {template.format(label=wrong_label)}"

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "true_label": label,
        "rejection_strategy": strategy,
    }


def build_preference_dataset(n_per_class: int = 50) -> Dataset:
    """Build a balanced preference dataset."""
    pairs = []
    for label in LABELS:
        for _ in range(n_per_class):
            pairs.append(generate_preference_pair(label))
    random.shuffle(pairs)

    print(f"  Created {len(pairs)} preference pairs ({n_per_class} per class)")

    return Dataset.from_dict({
        "prompt": [p["prompt"] for p in pairs],
        "chosen": [p["chosen"] for p in pairs],
        "rejected": [p["rejected"] for p in pairs],
    }), pairs


# ====================================================================
# 2. TOKEN PROBABILITY CAPTURE
# ====================================================================

EXAMPLE_PROMPTS_FOR_PROBS = [
    {"prompt": generate_io_prompt(label), "label": label}
    for label in LABELS[:5]
]


def capture_token_probs(model, tokenizer, prompts, device, top_k=20):
    """Capture top-k next-token probabilities for each prompt."""
    model.eval()
    results = []

    for sample in prompts:
        text = sample["prompt"] + "\n\n"
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
    """Records training metrics at each logging step."""
    def __init__(self):
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            entry = {"step": state.global_step}
            # DPO logs several useful metrics
            for key in ["loss", "rewards/chosen", "rewards/rejected",
                        "rewards/margins", "rewards/accuracies",
                        "logps/chosen", "logps/rejected"]:
                if key in logs:
                    entry[key] = round(logs[key], 6)
            if len(entry) > 1:  # More than just 'step'
                self.logs.append(entry)


# ====================================================================
# 4. MAIN TRAINING PIPELINE
# ====================================================================

def main():
    print("=" * 60)
    print("  STEP 2: Direct Preference Optimization (DPO)")
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
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
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
    dataset, raw_pairs = build_preference_dataset(n_per_class=50)

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
        lora_alpha=32,
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
    print("\n[5/6] Starting DPO training...")
    loss_callback = LossRecorderCallback()

    dpo_base_kwargs = dict(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,        # Effective batch = 8
        learning_rate=5e-5,                   # Lower LR for DPO (it's a refinement step)
        lr_scheduler_type="cosine",
        warmup_steps=10,
        beta=0.1,                             # DPO temperature parameter
        logging_steps=5,
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

    trainer = DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[loss_callback],
    )

    train_start = time.time()
    train_result = trainer.train()
    train_time = time.time() - train_start

    print(f"  DPO training complete in {train_time:.1f}s")
    print(f"  Final loss: {train_result.training_loss:.4f}")

    # ── 4f. Save artifacts ───────────────────────────────────────────
    print("\n[6/6] Saving artifacts...")

    # Save the DPO LoRA adapter
    adapter_path = OUTPUT_DIR / "adapter"
    model.save_pretrained(str(adapter_path))
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
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    ).to(device)
    ref_model_for_probs = PeftModel.from_pretrained(ref_base, str(SFT_ADAPTER))
    ref_model_for_probs = ref_model_for_probs.merge_and_unload()
    ref_model_for_probs.eval()

    def compute_completion_log_prob(mdl, prompt_text, completion_text):
        """Compute log probability of completion given prompt."""
        full_text = prompt_text + "\n\n" + completion_text
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        prompt_only = tokenizer(prompt_text + "\n\n", return_tensors="pt", truncation=True, max_length=400)
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
        strategy = pair["rejection_strategy"]

        # Determine style labels
        chosen_style = "concise"
        if strategy == "wrong_label":
            rejected_style = "wrong_label"
        elif strategy == "verbose_correct":
            rejected_style = "verbose"
        else:
            rejected_style = "wrong_and_verbose"

        # Before (reference/SFT model) log probs
        before_chosen_lp = compute_completion_log_prob(ref_model_for_probs, prompt, chosen)
        before_rejected_lp = compute_completion_log_prob(ref_model_for_probs, prompt, rejected)

        # After (DPO-trained model) log probs
        after_chosen_lp = compute_completion_log_prob(model, prompt, chosen)
        after_rejected_lp = compute_completion_log_prob(model, prompt, rejected)

        prob_shift_examples.append({
            "input": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "chosen_style": chosen_style,
            "rejected_style": rejected_style,
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

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DPO Training Complete!")
    print("=" * 60)
    print(f"  Adapter saved to:    {adapter_path}")
    print(f"  Training time:       {train_time:.1f}s")
    print(f"  Final loss:          {train_result.training_loss:.4f}")
    print(f"  Preference pairs:    {len(raw_pairs)}")
    print(f"  Artifacts in:        {OUTPUT_DIR}")
    print(f"\n  Next step: Run train_grpo.py")


if __name__ == "__main__":
    main()
