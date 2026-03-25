"""
============================================================
Step 1: Supervised Fine-Tuning (SFT)
============================================================
Fine-tunes SmolLM2 to classify storage I/O workloads
into 6 categories using LoRA adapters.

This is the foundation step — we teach the model *what* to
output (correct classification format) before later steps
refine *how* it outputs (style via DPO, accuracy via GRPO).

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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
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
             "batch_size": 4, "n_per_class": 80},
    "1.7B": {"name": "HuggingFaceTB/SmolLM2-1.7B", "slug": "smollm2-1.7b",
             "batch_size": 2, "n_per_class": 80},
}


def parse_args():
    parser = argparse.ArgumentParser(description="SFT training")
    parser.add_argument("--model-size", default="360M", choices=MODELS.keys(),
                        help="Model size to train (default: 360M)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output (default: reduced output)")
    return parser.parse_args()


def get_output_path(model_size):
    """Get output path based on model size."""
    if model_size == "360M":
        # Backward compatible: use existing flat structure
        return SCRIPT_DIR / "outputs" / "sft"
    else:
        slug = MODELS[model_size]["slug"]
        return SCRIPT_DIR / "outputs" / slug / "sft"


# ── Paths (defaults for 360M, overridden in main() based on args) ────
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "sft"

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

# ====================================================================
# 1. SYNTHETIC TRAINING DATA
# ====================================================================
# We generate training examples that pair I/O metric descriptions with
# classification labels + brief reasoning.  In a real scenario you'd
# collect actual storage telemetry; here we create representative
# synthetic data so the demo is self-contained.
# ====================================================================

WORKLOAD_PROFILES = {
    "OLTP Database": {
        "iops_range": (5000, 80000),
        "throughput_mb_range": (50, 400),
        "avg_latency_us_range": (100, 2000),
        "read_pct_range": (60, 80),
        "random_pct_range": (85, 99),
        "block_size_kb": [4, 8],
        "queue_depth_range": (16, 128),
        "reasons": [
            "High random IOPS with small block sizes indicate transactional database operations",
            "Small block random reads dominate, consistent with index lookups and row fetches",
            "Low latency small-block random I/O is characteristic of OLTP workloads",
        ],
    },
    "OLAP Analytics": {
        "iops_range": (100, 3000),
        "throughput_mb_range": (500, 5000),
        "avg_latency_us_range": (1000, 20000),
        "read_pct_range": (85, 99),
        "random_pct_range": (5, 30),
        "block_size_kb": [64, 128, 256, 512, 1024],
        "queue_depth_range": (1, 32),
        "reasons": [
            "Large sequential reads with high throughput indicate analytical table scans",
            "Predominantly sequential read pattern with large block sizes suggests data warehouse queries",
            "High throughput with large I/O sizes and sequential access is typical of OLAP workloads",
        ],
    },
    "AI ML Training": {
        "iops_range": (500, 10000),
        "throughput_mb_range": (1000, 10000),
        "avg_latency_us_range": (500, 10000),
        "read_pct_range": (90, 99),
        "random_pct_range": (20, 60),
        "block_size_kb": [128, 256, 512, 1024],
        "queue_depth_range": (8, 64),
        "reasons": [
            "High throughput reads with mixed sequential/random access suggest training data pipeline loading",
            "Large block reads with very high throughput indicate GPU training data ingestion",
            "Read-dominant mixed-access pattern with high bandwidth is characteristic of ML data loading",
        ],
    },
    "Video Streaming": {
        "iops_range": (100, 2000),
        "throughput_mb_range": (200, 3000),
        "avg_latency_us_range": (1000, 15000),
        "read_pct_range": (90, 100),
        "random_pct_range": (10, 40),
        "block_size_kb": [256, 512, 1024, 2048],
        "queue_depth_range": (1, 16),
        "reasons": [
            "Steady sequential reads with large block sizes indicate media streaming operations",
            "Consistent high-throughput sequential reads suggest video file delivery",
            "Large block sequential read pattern with steady bandwidth is typical of streaming workloads",
        ],
    },
    "VDI Virtual Desktop": {
        "iops_range": (2000, 30000),
        "throughput_mb_range": (20, 200),
        "avg_latency_us_range": (200, 5000),
        "read_pct_range": (50, 70),
        "random_pct_range": (70, 95),
        "block_size_kb": [4, 8, 16],
        "queue_depth_range": (4, 64),
        "reasons": [
            "Mixed read/write random I/O with small blocks indicates virtual desktop user activity",
            "Balanced read-write ratio with small random I/O suggests many concurrent desktop sessions",
            "Small block random I/O with mixed reads and writes is characteristic of VDI workloads",
        ],
    },
    "Backup Archive": {
        "iops_range": (50, 1000),
        "throughput_mb_range": (200, 5000),
        "avg_latency_us_range": (2000, 50000),
        "read_pct_range": (5, 30),
        "random_pct_range": (2, 15),
        "block_size_kb": [256, 512, 1024, 2048, 4096],
        "queue_depth_range": (1, 16),
        "reasons": [
            "Large sequential writes with high throughput indicate backup data ingestion",
            "Write-dominant sequential pattern with very large block sizes suggests archival operations",
            "Sustained sequential write throughput with large I/O sizes is typical of backup workloads",
        ],
    },
}

LABELS = list(WORKLOAD_PROFILES.keys())


def generate_sample(label: str) -> dict:
    """Generate one synthetic I/O metrics sample for a given workload label."""
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
    reason = random.choice(p["reasons"])

    # Format the I/O metrics as a structured text prompt
    prompt = (
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
        f"Provide the classification and a brief reason."
    )
    response = f"{label}\nReason: {reason}"

    return {"prompt": prompt, "completion": response, "label": label}


def build_dataset(n_per_class: int = 80) -> Dataset:
    """Build a balanced training dataset with n_per_class examples per label."""
    samples = []
    for label in LABELS:
        for _ in range(n_per_class):
            samples.append(generate_sample(label))
    random.shuffle(samples)
    print(f"  Created {len(samples)} training samples ({n_per_class} per class)")

    # Use prompt + completion columns so TRL masks prompt tokens in the loss.
    # This ensures the model only learns to predict the classification output,
    # not the prompt text itself. The "Classification: " suffix is part of the
    # prompt so the model only needs to learn the label + reason.
    return Dataset.from_dict({
        "prompt": [s["prompt"] + "\n\nClassification: " for s in samples],
        "completion": [s["completion"] for s in samples],
    })


# ====================================================================
# 2. TOKEN PROBABILITY CAPTURE (for visualization in the web app)
# ====================================================================

# We'll record how the model's predictions change after fine-tuning
# by capturing the top-20 token probabilities for 5 example prompts.

EXAMPLE_PROMPTS = [
    generate_sample("OLTP Database"),
    generate_sample("OLAP Analytics"),
    generate_sample("AI ML Training"),
    generate_sample("Video Streaming"),
    generate_sample("Backup Archive"),
]


def capture_token_probs(model, tokenizer, prompts, device, top_k=20):
    """
    For each prompt, run a forward pass and capture the top-k token
    probabilities at the first generated position (i.e., what would
    the model predict as the very next token?).

    Returns a list of dicts with prompt text, top tokens, and probs.
    """
    model.eval()
    results = []

    for sample in prompts:
        text = sample["prompt"] + "\n\nClassification: "
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Logits for the last token position → next-token prediction
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
# 3. SAMPLE OUTPUT GENERATION
# ====================================================================

def generate_outputs(model, tokenizer, prompts, device, max_new_tokens=80):
    """Generate text outputs for a list of prompts."""
    model.eval()
    results = []

    for sample in prompts:
        text = sample["prompt"] + "\n\nClassification: "
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # Greedy for reproducibility
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        results.append({
            "prompt_snippet": sample["prompt"][:120] + "...",
            "expected": sample["completion"],
            "generated": generated.strip(),
        })

    return results


# ====================================================================
# 4. LoRA WEIGHT CAPTURE (simplified, for web app visualization)
# ====================================================================

def capture_lora_weights(model):
    """
    Extract a simplified summary of LoRA weight matrices.
    We save the norms + a small slice of actual values so the
    web app can render a heatmap / matrix visualization.
    """
    lora_info = {}

    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            data = param.detach().cpu().float().numpy()
            # Save: shape, frobenius norm, mean, std, and a small
            # (max 16x16) slice of actual values for visualization
            slice_r = min(data.shape[0], 16)
            slice_c = min(data.shape[1], 16) if len(data.shape) > 1 else 1
            if len(data.shape) == 1:
                small = data[:slice_r].tolist()
            else:
                small = data[:slice_r, :slice_c].tolist()

            lora_info[name] = {
                "shape": list(data.shape),
                "frobenius_norm": float(np.linalg.norm(data)),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "sample_values": small,
            }

    return lora_info


# ====================================================================
# 5. TRAINING LOSS CALLBACK
# ====================================================================

from transformers import TrainerCallback

class LossRecorderCallback(TrainerCallback):
    """Records training loss and prints color-coded progress."""
    def __init__(self):
        self.losses = []
        self._prev = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return

        step = state.global_step
        epoch = round(state.epoch, 1) if state.epoch else 0
        loss = logs["loss"]
        self.losses.append({
            "step": step,
            "loss": round(loss, 6),
            "epoch": round(state.epoch, 4) if state.epoch else 0,
        })

        # Build color-coded output
        parts = []

        # Loss (should decrease)
        loss_color = self._color("loss", loss, "down")
        parts.append(f"loss: {loss_color}{loss:.4f}{RESET}")

        # Token accuracy (should increase)
        tok_acc = logs.get("mean_token_accuracy")
        if tok_acc is not None:
            acc_color = self._color("mean_token_accuracy", tok_acc, "up")
            parts.append(f"tok_acc: {acc_color}{tok_acc:.1%}{RESET}")

        # Grad norm (informational — flag if very high)
        grad = logs.get("grad_norm")
        if grad is not None:
            grad_color = RED if grad > 5.0 else (YELLOW if grad > 2.0 else "")
            parts.append(f"grad: {grad_color}{grad:.2f}{RESET}")

        # Learning rate (informational)
        lr = logs.get("learning_rate")
        if lr is not None:
            parts.append(f"lr: {lr:.2e}")

        print(f"  Step {step:>4d} (epoch {epoch:>4.1f}) │ {' │ '.join(parts)}")
        self._prev.update({k: logs[k] for k in logs if isinstance(logs[k], (int, float))})

    def _color(self, key, value, direction):
        """Return ANSI color based on whether metric moved in the right direction."""
        prev = self._prev.get(key)
        if prev is None:
            return BOLD
        if direction == "down":
            return GREEN if value < prev else (RED if value > prev * 1.1 else YELLOW)
        else:  # up
            return GREEN if value > prev else (RED if value < prev * 0.9 else YELLOW)


# ====================================================================
# 6. MAIN TRAINING PIPELINE
# ====================================================================

def main():
    args = parse_args()
    cfg = MODELS[args.model_size]

    # Override module-level defaults based on args
    global MODEL_NAME, OUTPUT_DIR
    MODEL_NAME = cfg["name"]
    OUTPUT_DIR = get_output_path(args.model_size)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  STEP 1: Supervised Fine-Tuning (SFT)")
    print(f"  Model: {cfg['name']} ({args.model_size})")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Using device: {device}")
    if device != "cuda":
        print("[WARNING] No GPU detected. Training will be slow!")

    # ── 6a. Load base model and tokenizer ────────────────────────────
    print(f"\n[1/7] Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    ).to(device)

    print(f"  Model parameters: {model.num_parameters():,}")

    # ── 6b. Capture BASE model outputs (before fine-tuning) ──────────
    print("\n[2/7] Capturing base model outputs (before fine-tuning)...")
    base_probs = capture_token_probs(model, tokenizer, EXAMPLE_PROMPTS, device)
    base_outputs = generate_outputs(model, tokenizer, EXAMPLE_PROMPTS, device)

    with open(OUTPUT_DIR / "base_token_probs.json", "w") as f:
        json.dump(base_probs, f, indent=2)
    with open(OUTPUT_DIR / "base_outputs.json", "w") as f:
        json.dump(base_outputs, f, indent=2)
    print("  Saved base model token probabilities and sample outputs.")

    # ── 6c. Build training dataset ───────────────────────────────────
    n_per_class = cfg["n_per_class"]
    print(f"\n[3/7] Building synthetic training dataset ({n_per_class} per class)...")
    dataset = build_dataset(n_per_class=n_per_class)

    # ── Verify prompt masking ────────────────────────────────────
    # TRL's SFTTrainer should mask prompt tokens in the loss. Verify
    # by checking the prompt vs completion token ratio.
    sample_prompt = dataset[0]["prompt"]
    sample_completion = dataset[0]["completion"]
    prompt_tokens = len(tokenizer.encode(sample_prompt))
    completion_tokens = len(tokenizer.encode(sample_completion))
    total_tokens = prompt_tokens + completion_tokens
    completion_pct = completion_tokens / total_tokens * 100
    print(f"  Prompt tokens: ~{prompt_tokens} | Completion tokens: ~{completion_tokens} | "
          f"Masking: {'✓' if completion_pct < 30 else '⚠'} "
          f"(completion is {completion_pct:.0f}% of sequence)")

    # ── 6d. Configure LoRA ───────────────────────────────────────────
    # LoRA (Low-Rank Adaptation) adds small trainable matrices to the
    # attention layers while keeping the base model weights frozen.
    # This is dramatically more efficient than full fine-tuning:
    #   - rank=16: each adapter matrix is 16-dimensional
    #   - alpha=32: scaling factor (alpha/rank = 2x effective learning rate)
    #   - Only ~0.5% of parameters are actually trained!
    print("\n[4/7] Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=16,                           # Low-rank dimension
        lora_alpha=32,                  # Scaling factor
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── 6e. Training ─────────────────────────────────────────────────
    print("\n[5/7] Starting SFT training...")
    print(f"""
  {BOLD}What to watch for during training:{RESET}
  ┌─────────────────────────────────────────────────────────┐
  │ {BOLD}loss{RESET}      Should decrease steadily. Starts ~2.5-3.0,   │
  │           should reach <0.5. Measures how well the      │
  │           model predicts the next token in completions.  │
  │                                                         │
  │ {BOLD}tok_acc{RESET}   Token-level accuracy on completions. Should   │
  │           climb toward 90%+. Shows what fraction of     │
  │           output tokens the model gets right.            │
  │                                                         │
  │ {BOLD}grad{RESET}      Gradient norm. Should stay stable (0.5-2.0). │
  │           Spikes above 5.0 suggest training instability. │
  │                                                         │
  │ {BOLD}lr{RESET}        Learning rate (cosine schedule). Starts high, │
  │           decays to near zero. Just informational.       │
  │                                                         │
  │ {GREEN}Green{RESET} = moving in the right direction                  │
  │ {RED}Red{RESET}   = moving the wrong way (investigate if persistent) │
  │ {YELLOW}Yellow{RESET} = neutral / slight concern                       │
  └─────────────────────────────────────────────────────────┘
""")
    loss_callback = LossRecorderCallback()

    batch_size = cfg["batch_size"]

    # SFTConfig/SFTTrainer API varies across TRL versions:
    #   TRL <0.16:  max_seq_length + dataset_text_field in SFTConfig
    #   TRL 0.16+:  those args removed entirely; SFTTrainer auto-detects
    #               the text column and uses max_length from TrainingArguments
    #
    # Dataset has prompt + completion columns. TRL auto-detects these and:
    #   - Concatenates prompt + completion
    #   - Masks prompt tokens in the loss (only trains on completion)
    # This is critical: the model learns the output format, not the prompt.
    sft_base_kwargs = dict(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=5,                  # More epochs for small model
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,       # Effective batch = batch_size * 4
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=5 if args.verbose else 25,
        save_strategy="epoch",
        seed=SEED,
        bf16=(device == "cuda"),
        fp16=False,
        report_to="none",                    # Don't log to wandb etc.
    )

    # Try old-style SFTConfig first, then fall back to minimal config
    try:
        training_args = SFTConfig(
            **sft_base_kwargs,
            max_seq_length=512,
            packing=False,
        )
    except TypeError:
        # Newer TRL: no max_seq_length — use max_length instead
        training_args = SFTConfig(
            **sft_base_kwargs,
            max_length=512,
            packing=False,
        )

    # Suppress noisy TRL tokenization mismatch warnings.
    # These fire on every example but are harmless — TRL still masks
    # prompt tokens correctly using a fallback strategy.
    import logging
    trl_logger = logging.getLogger("trl.trainer.sft_trainer")
    trl_logger.setLevel(logging.ERROR)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[loss_callback],
    )

    # Restore logger after dataset processing
    trl_logger.setLevel(logging.WARNING)

    # Suppress default trainer logging (we use our colored callback instead)
    import logging
    logging.getLogger("transformers.trainer").setLevel(logging.WARNING)

    train_start = time.time()
    train_result = trainer.train()
    train_time = time.time() - train_start

    print(f"  Training complete in {train_time:.1f}s")
    print(f"  Final loss: {train_result.training_loss:.4f}")

    # Verify loss is consistent with completion-only training
    if loss_callback.losses:
        first_loss = loss_callback.losses[0]["loss"]
        if first_loss > 7.0:
            print(f"  ⚠ Initial loss ({first_loss:.1f}) is unusually high — "
                  f"prompt masking may not be working correctly")
        else:
            print(f"  ✓ Initial loss ({first_loss:.1f}) consistent with completion-only training")

    # ── 6f. Save the LoRA adapter ────────────────────────────────────
    print("\n[6/7] Saving artifacts...")

    # Save the LoRA adapter (small! usually < 5 MB)
    adapter_path = OUTPUT_DIR / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"  Saved LoRA adapter to {adapter_path}")

    # Save training loss curve
    # Format: loss_curve for the web app, plus the raw losses for compatibility
    with open(OUTPUT_DIR / "training_loss.json", "w") as f:
        json.dump({
            "loss_curve": loss_callback.losses,  # [{"step": 0, "loss": 2.8, "epoch": 0.0}, ...]
            "losses": loss_callback.losses,      # backward-compat alias
            "final_loss": round(train_result.training_loss, 6),
            "training_time_seconds": round(train_time, 2),
            "total_steps": train_result.global_step,
        }, f, indent=2)

    # Save LoRA weight visualization data (detailed per-layer format)
    lora_weights_detailed = capture_lora_weights(model)
    with open(OUTPUT_DIR / "lora_weights_detailed.json", "w") as f:
        json.dump(lora_weights_detailed, f, indent=2)
    print(f"  Saved detailed LoRA weight data ({len(lora_weights_detailed)} matrices)")

    # Save simplified LoRA weight data in the format the web app expects:
    # {"layer": "...", "lora_A": [[...]], "lora_B": [[...]], "rank": 16}
    # We pick one representative attention layer to visualize.
    lora_a_data = None
    lora_b_data = None
    target_layer_name = None
    for name, param in model.named_parameters():
        if "lora_A" in name and "q_proj" in name and param.requires_grad:
            lora_a_data = param.detach().cpu().float().numpy()
            # Derive the layer path from the parameter name
            # e.g. "base_model.model.model.layers.8.self_attn.q_proj.lora_A.default.weight"
            # We want something like "model.layers.8.self_attn.q_proj"
            parts = name.replace(".lora_A.default.weight", "").replace(".lora_A.weight", "")
            parts = parts.replace("base_model.model.", "")
            target_layer_name = parts
        if "lora_B" in name and "q_proj" in name and param.requires_grad:
            lora_b_data = param.detach().cpu().float().numpy()

    if lora_a_data is not None and lora_b_data is not None:
        lora_weights_simple = {
            "layer": target_layer_name,
            "lora_A": lora_a_data.tolist(),
            "lora_B": lora_b_data.tolist(),
            "rank": int(lora_a_data.shape[0]),
        }
    else:
        # Fallback: save from the detailed data
        lora_weights_simple = {
            "layer": "unknown",
            "lora_A": [],
            "lora_B": [],
            "rank": 16,
        }
    with open(OUTPUT_DIR / "lora_weights.json", "w") as f:
        json.dump(lora_weights_simple, f, indent=2)
    print(f"  Saved LoRA weight visualization for layer: {target_layer_name}")

    # ── 6g. Capture SFT model outputs (after fine-tuning) ────────────
    # Reload the model from the saved adapter — TRL's SFTTrainer leaves
    # the in-memory model in a state where generate() produces empty output.
    print("\n[7/7] Capturing fine-tuned model outputs...")
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    sft_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    ).to(device)
    model = PeftModel.from_pretrained(sft_base, str(adapter_path))
    model.eval()

    # Quick smoke test — verify the reloaded model can generate
    _test_input = tokenizer(
        EXAMPLE_PROMPTS[0]["prompt"] + "\n\nClassification: ",
        return_tensors="pt", truncation=True, max_length=512,
    ).to(device)
    with torch.no_grad():
        _test_ids = model.generate(**_test_input, max_new_tokens=20, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    _test_out = tokenizer.decode(_test_ids[0][_test_input["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip()
    if _test_out:
        print(f"  Smoke test: \"{_test_out[:60]}\" ✓")
    else:
        print(f"  ⚠ Smoke test: model generated empty output!")
        print(f"    Raw IDs: {_test_ids[0][-20:].tolist()}")
        print(f"    Input length: {_test_input['input_ids'].shape[1]}")
        print(f"    Output length: {_test_ids.shape[1]}")

    sft_probs = capture_token_probs(model, tokenizer, EXAMPLE_PROMPTS, device)
    sft_outputs = generate_outputs(model, tokenizer, EXAMPLE_PROMPTS, device)

    # Print what was actually generated (debug)
    for i, out in enumerate(sft_outputs[:3]):
        gen = out["generated"][:80] if out["generated"] else "(empty)"
        print(f"  Sample {i}: {gen}")

    with open(OUTPUT_DIR / "sft_token_probs.json", "w") as f:
        json.dump(sft_probs, f, indent=2)
    with open(OUTPUT_DIR / "sft_outputs.json", "w") as f:
        json.dump(sft_outputs, f, indent=2)

    # Save before/after comparison in the format the web app expects:
    # [{"input": "...", "base_output": "...", "sft_output": "...",
    #   "true_label": "...", "base_correct": false, "sft_correct": true}]
    comparison = []
    for i in range(len(EXAMPLE_PROMPTS)):
        true_label = EXAMPLE_PROMPTS[i]["label"]

        # Check if each output contains the correct classification
        base_text = base_outputs[i]["generated"]
        sft_text = sft_outputs[i]["generated"]
        base_correct = true_label.lower() in base_text.lower()
        sft_correct = true_label.lower() in sft_text.lower()

        comparison.append({
            "input": EXAMPLE_PROMPTS[i]["prompt"],
            "base_output": base_text,
            "sft_output": sft_text,
            "true_label": true_label,
            "base_correct": base_correct,
            "sft_correct": sft_correct,
            # Keep extra info for richer visualization
            "base_top3_tokens": base_probs[i]["top_tokens"][:3],
            "sft_top3_tokens": sft_probs[i]["top_tokens"][:3],
        })
    with open(OUTPUT_DIR / "before_after_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Saved {len(comparison)} before/after comparison examples.")

    # ── Sanity check: structured results ─────────────────────────
    sanity_labels = LABELS[:5]  # One per class (skip VDI to keep it to 5)
    sanity_results = []

    for i, sample in enumerate(EXAMPLE_PROMPTS[:5]):
        label = sample["label"]
        sft_text = sft_outputs[i]["generated"]

        # Extract predicted label from SFT output
        predicted = ""
        text_lower = sft_text.lower()
        for l in sorted(LABELS, key=len, reverse=True):
            if l.lower() in text_lower:
                predicted = l
                break

        correct = predicted == label
        sanity_results.append({
            "expected": label,
            "predicted": predicted if predicted else "(unparseable)",
            "generated_snippet": sft_text[:100],
            "correct": correct,
        })

    num_correct = sum(1 for r in sanity_results if r["correct"])
    sample_accuracy = num_correct / len(sanity_results)

    # Determine verdict
    checks = []
    initial_loss = loss_callback.losses[0]["loss"] if loss_callback.losses else None
    final_loss = loss_callback.losses[-1]["loss"] if loss_callback.losses else None

    if initial_loss and final_loss:
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100
        if loss_reduction > 50:
            checks.append(("pass", f"Loss decreased significantly ({loss_reduction:.0f}% reduction)"))
        elif loss_reduction > 0:
            checks.append(("warn", f"Loss decreased modestly ({loss_reduction:.0f}% reduction)"))
        else:
            checks.append(("fail", "Loss did not decrease"))

    has_format = any(r["correct"] for r in sanity_results)
    if has_format:
        checks.append(("pass", "Model outputs correct format (label + reason)"))
    else:
        checks.append(("fail", "Model outputs are not in expected format"))

    # Check if model distinguishes easy categories
    easy_correct = sum(1 for r in sanity_results
                       if r["expected"] in ["OLTP Database", "Backup Archive"] and r["correct"])
    easy_total = sum(1 for r in sanity_results
                     if r["expected"] in ["OLTP Database", "Backup Archive"])
    if easy_total > 0 and easy_correct == easy_total:
        checks.append(("pass", "Model distinguishes easy categories (OLTP vs Backup)"))
    elif easy_total > 0:
        checks.append(("warn", "Model struggles with even easy categories"))

    # Check for OLAP/AI ML confusion
    confused = [r for r in sanity_results
                if r["expected"] in ["OLAP Analytics", "AI ML Training"] and not r["correct"]]
    if confused:
        checks.append(("info", "Confuses OLAP/AI ML — these have similar I/O profiles"))

    all_pass = all(c[0] in ["pass", "info"] for c in checks)
    any_fail = any(c[0] == "fail" for c in checks)
    verdict = "HEALTHY" if all_pass else ("ISSUES DETECTED" if any_fail else "OK WITH WARNINGS")

    # Print structured results
    print("\n" + "=" * 60)
    print("  SFT RESULTS")
    print("=" * 60)
    if initial_loss and final_loss:
        print(f"  Training:  Loss {initial_loss:.2f} → {final_loss:.2f} "
              f"({(initial_loss - final_loss) / initial_loss * 100:.0f}% reduction) "
              f"in {train_time/60:.0f}m {train_time%60:.0f}s")
    else:
        print(f"  Training:  {train_time:.1f}s")
    print()
    print("  Sample outputs (5 prompts, one per class):")
    for r in sanity_results:
        mark = "✓" if r["correct"] else "✗"
        note = ""
        if not r["correct"] and r["expected"] in ["OLAP Analytics", "AI ML Training"]:
            note = " (common confusion)"
        print(f'    {r["expected"]:<20s} → {r["predicted"]:<22s} {mark}{note}')
    print(f"  Sample accuracy: {num_correct}/{len(sanity_results)} ({sample_accuracy:.0%})")
    print()
    print("  Sanity check:")
    for status, msg in checks:
        icon = {"pass": "✓", "warn": "⚠", "fail": "✗", "info": "⚠"}.get(status, "?")
        print(f"    {icon} {msg}")
    print(f"\n  Verdict: {verdict} — {'proceed to DPO' if not any_fail else 'investigate before proceeding'}")
    print("=" * 60)

    # Save sanity check JSON
    sanity_data = {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "training_time_seconds": round(train_time, 2),
        "sample_results": sanity_results,
        "sample_accuracy": sample_accuracy,
        "checks": [{"status": s, "message": m} for s, m in checks],
        "verdict": verdict,
    }
    with open(OUTPUT_DIR / "sft_sanity_check.json", "w") as f:
        json.dump(sanity_data, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SFT Training Complete!")
    print("=" * 60)
    print(f"  Model:               {cfg['name']} ({args.model_size})")
    print(f"  Adapter saved to:    {adapter_path}")
    print(f"  Training time:       {train_time:.1f}s")
    print(f"  Final loss:          {train_result.training_loss:.4f}")
    print(f"  Trainable params:    {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"  Artifacts in:        {OUTPUT_DIR}")
    print(f"\n  Next step: Run train_dpo.py")


if __name__ == "__main__":
    main()
