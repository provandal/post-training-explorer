"""
============================================================
Step 4: Export Artifacts for Web App
============================================================
Loads each model variant (base, SFT, DPO, GRPO) and runs
inference on a standard set of 20 test prompts. Produces a
single precomputed_results.json that the web app can load
to show real model outputs without needing a GPU at runtime.

This is the "cooking show reveal" — everything is pre-baked
so the web demo can show genuine model behavior.

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
from peft import PeftModel

# ── Reproducibility ──────────────────────────────────────────────────
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SFT_ADAPTER = SCRIPT_DIR / "outputs" / "sft" / "adapter"
DPO_ADAPTER = SCRIPT_DIR / "outputs" / "dpo" / "adapter"
GRPO_ADAPTER = SCRIPT_DIR / "outputs" / "grpo" / "adapter"

# The web app's public data directory
# Adjust this path if your project structure differs
WEBAPP_DATA_DIR = SCRIPT_DIR.parent / "app" / "public" / "data"
WEBAPP_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Also save to the scripts outputs directory
EXPORT_DIR = SCRIPT_DIR / "outputs" / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

# ── Check prerequisites ─────────────────────────────────────────────
missing = []
for name, path in [("SFT", SFT_ADAPTER), ("DPO", DPO_ADAPTER), ("GRPO", GRPO_ADAPTER)]:
    if not path.exists():
        missing.append(f"  - {name} adapter: {path}")

if missing:
    print("=" * 60)
    print("  WARNING: Some adapters are missing!")
    for m in missing:
        print(m)
    print("\n  Will export results for available models only.")
    print("  Run the missing training scripts first for complete results.")
    print("=" * 60)


# ====================================================================
# 1. TEST PROMPTS
# ====================================================================
# A fixed set of 20 test prompts (balanced across categories) that
# every model variant will be evaluated on. Using a fixed set ensures
# fair comparison.
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


def generate_test_prompt(label: str, idx: int) -> dict:
    """Generate a deterministic test prompt using seed = SEED + idx."""
    rng = random.Random(SEED + idx)
    p = WORKLOAD_PROFILES[label]
    iops = rng.randint(*p["iops_range"])
    throughput = rng.randint(*p["throughput_mb_range"])
    latency = rng.randint(*p["avg_latency_us_range"])
    read_pct = rng.randint(*p["read_pct_range"])
    write_pct = 100 - read_pct
    random_pct = rng.randint(*p["random_pct_range"])
    sequential_pct = 100 - random_pct
    block_kb = rng.choice(p["block_size_kb"])
    queue_depth = rng.randint(*p["queue_depth_range"])

    prompt = (
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

    return {
        "id": idx,
        "prompt": prompt,
        "true_label": label,
        "metrics": {
            "iops": iops,
            "throughput_mb": throughput,
            "avg_latency_us": latency,
            "read_pct": read_pct,
            "write_pct": write_pct,
            "random_pct": random_pct,
            "sequential_pct": sequential_pct,
            "block_size_kb": block_kb,
            "queue_depth": queue_depth,
        },
    }


def build_test_set() -> list[dict]:
    """
    Build 20 test prompts: ~3-4 per category, deterministically seeded.
    We do 4 for the first 2 categories and 3 for the remaining 4 to
    reach exactly 20.
    """
    test_prompts = []
    idx = 0
    counts = [4, 4, 3, 3, 3, 3]  # 4+4+3+3+3+3 = 20

    for label, count in zip(LABELS, counts):
        for _ in range(count):
            test_prompts.append(generate_test_prompt(label, idx))
            idx += 1

    return test_prompts


# ====================================================================
# 2. INFERENCE + PROBABILITY CAPTURE
# ====================================================================

def run_inference(model, tokenizer, prompt_text: str, device: str,
                  max_new_tokens: int = 80, top_k_probs: int = 20):
    """
    Run inference on a single prompt and capture:
      - Generated text
      - Top-k token probabilities at 3 key positions:
        1. First generated token (initial prediction)
        2. Token after "Classification:" (the label token)
        3. Token after "Reason:" (start of reasoning)
      - Generation time
    """
    model.eval()
    full_prompt = prompt_text + "\n\n"

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    # ── Generate ─────────────────────────────────────────────────────
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                  # Greedy for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,               # Get logits at each step
        )
    gen_time = time.time() - start_time

    # Decode generated text
    generated_ids = output_ids.sequences[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # ── Capture token probabilities at key positions ─────────────────
    scores = output_ids.scores  # List of logit tensors, one per generated token
    prob_snapshots = []

    # Positions to capture: first token, and tokens at key semantic points
    positions_to_capture = [0]  # Always capture position 0

    # Find position of "Classification" and "Reason" tokens in the output
    generated_token_strs = [tokenizer.decode(tid) for tid in generated_ids]
    for pos, tok_str in enumerate(generated_token_strs):
        if "classif" in tok_str.lower() or "Classification" in tok_str:
            # Capture the token AFTER "Classification:"
            if pos + 1 < len(scores):
                positions_to_capture.append(pos + 1)
        if "reason" in tok_str.lower() or "Reason" in tok_str:
            if pos + 1 < len(scores):
                positions_to_capture.append(pos + 1)

    # Also capture the midpoint of generation
    mid = len(scores) // 2
    if mid not in positions_to_capture and mid < len(scores):
        positions_to_capture.append(mid)

    # Deduplicate and sort
    positions_to_capture = sorted(set(positions_to_capture))[:4]

    for pos in positions_to_capture:
        if pos >= len(scores):
            continue
        logits = scores[pos]
        probs = torch.softmax(logits[0], dim=-1)
        top_probs, top_indices = torch.topk(probs, min(top_k_probs, len(probs)))

        prob_snapshots.append({
            "position": pos,
            "actual_token": tokenizer.decode(generated_ids[pos].item()) if pos < len(generated_ids) else "",
            "top_tokens": [tokenizer.decode(idx.item()) for idx in top_indices],
            "top_probs": [round(p.item(), 6) for p in top_probs],
        })

    return {
        "generated_text": generated_text,
        "generation_time_ms": round(gen_time * 1000, 2),
        "num_tokens_generated": len(generated_ids),
        "token_probabilities": prob_snapshots,
    }


# ====================================================================
# 3. MODEL LOADING HELPERS
# ====================================================================

def load_base_model(device, dtype):
    """Load the unmodified base model."""
    print(f"  Loading base model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=dtype,
    ).to(device)
    return model, tokenizer


def load_sft_model(base_model, device):
    """Load base + SFT adapter."""
    print(f"  Loading SFT adapter from {SFT_ADAPTER}")
    model = PeftModel.from_pretrained(base_model, str(SFT_ADAPTER))
    return model


def load_dpo_model(device, dtype):
    """Load base + SFT (merged) + DPO adapter."""
    print(f"  Loading DPO model (base + SFT merged + DPO adapter)")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=dtype,
    ).to(device)
    # First merge SFT
    sft_model = PeftModel.from_pretrained(base, str(SFT_ADAPTER))
    merged = sft_model.merge_and_unload()
    # Then add DPO adapter
    model = PeftModel.from_pretrained(merged, str(DPO_ADAPTER))
    return model


def load_grpo_model(device, dtype):
    """Load base + SFT (merged) + GRPO adapter."""
    print(f"  Loading GRPO model (base + SFT merged + GRPO adapter)")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=dtype,
    ).to(device)
    # First merge SFT
    sft_model = PeftModel.from_pretrained(base, str(SFT_ADAPTER))
    merged = sft_model.merge_and_unload()
    # Then add GRPO adapter
    model = PeftModel.from_pretrained(merged, str(GRPO_ADAPTER))
    return model


# ====================================================================
# 4. RESOURCE UTILIZATION
# ====================================================================

def get_resource_info():
    """Collect resource utilization info for the summary."""
    info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2) if torch.cuda.is_available() else 0,
    }

    # Check adapter checkpoint sizes
    for name, path in [("sft", SFT_ADAPTER), ("dpo", DPO_ADAPTER), ("grpo", GRPO_ADAPTER)]:
        if path.exists():
            total_size = sum(
                f.stat().st_size for f in path.rglob("*") if f.is_file()
            )
            info[f"{name}_checkpoint_size_mb"] = round(total_size / 1e6, 2)
        else:
            info[f"{name}_checkpoint_size_mb"] = None

    # Load training time info from each stage
    for name in ["sft", "dpo", "grpo"]:
        curves_file = SCRIPT_DIR / "outputs" / name / "training_curves.json"
        loss_file = SCRIPT_DIR / "outputs" / name / "training_loss.json"
        for f in [curves_file, loss_file]:
            if f.exists():
                with open(f) as fh:
                    data = json.load(fh)
                    if "training_time_seconds" in data:
                        info[f"{name}_training_time_seconds"] = data["training_time_seconds"]
                break

    return info


# ====================================================================
# 5. MAIN EXPORT PIPELINE
# ====================================================================

def main():
    print("=" * 60)
    print("  STEP 4: Export Artifacts for Web App")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"\n[INFO] Using device: {device}")

    # ── Build test set ───────────────────────────────────────────────
    print("\n[1/4] Building standardized test set (20 prompts)...")
    test_prompts = build_test_set()
    print(f"  Created {len(test_prompts)} test prompts across {len(LABELS)} categories")

    # ── Define which models to evaluate ──────────────────────────────
    model_variants = ["base"]
    if SFT_ADAPTER.exists():
        model_variants.append("sft")
    if DPO_ADAPTER.exists():
        model_variants.append("dpo")
    if GRPO_ADAPTER.exists():
        model_variants.append("grpo")

    print(f"\n[2/4] Will evaluate models: {model_variants}")

    # ── Run inference for each model ─────────────────────────────────
    results = {
        "metadata": {
            "base_model": MODEL_NAME,
            "num_test_prompts": len(test_prompts),
            "model_variants": model_variants,
            "categories": LABELS,
            "export_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": SEED,
        },
        "test_prompts": test_prompts,
        "model_results": {},
    }

    print(f"\n[3/4] Running inference...")

    for variant in model_variants:
        print(f"\n  --- {variant.upper()} model ---")

        # Load the appropriate model
        if variant == "base":
            model, tokenizer = load_base_model(device, dtype)
        elif variant == "sft":
            base_model, tokenizer = load_base_model(device, dtype)
            model = load_sft_model(base_model, device)
        elif variant == "dpo":
            model = load_dpo_model(device, dtype)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        elif variant == "grpo":
            model = load_grpo_model(device, dtype)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        # Run inference on all test prompts
        variant_results = []
        total_time = 0
        correct = 0

        for i, test in enumerate(test_prompts):
            result = run_inference(model, tokenizer, test["prompt"], device)
            variant_results.append(result)
            total_time += result["generation_time_ms"]

            # Check if classification is correct
            import re
            match = re.search(r"Classification:\s*(.+?)(?:\n|$)",
                              result["generated_text"], re.IGNORECASE)
            if match:
                predicted = match.group(1).strip()
                for label in LABELS:
                    if label.lower() in predicted.lower():
                        if label == test["true_label"]:
                            correct += 1
                        break

            if (i + 1) % 5 == 0:
                print(f"    Completed {i + 1}/{len(test_prompts)} prompts")

        accuracy = correct / len(test_prompts) if test_prompts else 0
        results["model_results"][variant] = {
            "outputs": variant_results,
            "summary": {
                "total_generation_time_ms": round(total_time, 2),
                "avg_generation_time_ms": round(total_time / len(test_prompts), 2),
                "accuracy": round(accuracy, 4),
                "correct": correct,
                "total": len(test_prompts),
            },
        }

        print(f"  {variant.upper()}: accuracy={accuracy:.1%} ({correct}/{len(test_prompts)}), "
              f"avg_time={total_time/len(test_prompts):.1f}ms")

        # Free GPU memory before loading next model
        del model
        if variant == "sft":
            del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Collect training artifacts from each stage ───────────────────
    print("\n[4/4] Collecting training artifacts and resource info...")

    # Load training curves from each step
    training_data = {}
    for name in ["sft", "dpo", "grpo"]:
        for fname in ["training_loss.json", "training_curves.json"]:
            fpath = SCRIPT_DIR / "outputs" / name / fname
            if fpath.exists():
                with open(fpath) as f:
                    training_data[name] = json.load(f)
                break

    results["training_data"] = training_data

    # Load preference pair examples (DPO)
    pref_examples = SCRIPT_DIR / "outputs" / "dpo" / "preference_pair_examples.json"
    if pref_examples.exists():
        with open(pref_examples) as f:
            results["dpo_preference_examples"] = json.load(f)

    # Load generation logs (GRPO)
    gen_logs = SCRIPT_DIR / "outputs" / "grpo" / "generation_logs.json"
    if gen_logs.exists():
        with open(gen_logs) as f:
            results["grpo_generation_logs"] = json.load(f)

    # Load LoRA weight visualization data
    lora_weights = SCRIPT_DIR / "outputs" / "sft" / "lora_weights.json"
    if lora_weights.exists():
        with open(lora_weights) as f:
            results["lora_weight_visualization"] = json.load(f)

    # Load before/after comparison from SFT
    comparison = SCRIPT_DIR / "outputs" / "sft" / "before_after_comparison.json"
    if comparison.exists():
        with open(comparison) as f:
            results["sft_before_after"] = json.load(f)

    # Load probability shift data from DPO
    prob_shifts = SCRIPT_DIR / "outputs" / "dpo" / "probability_shifts.json"
    if prob_shifts.exists():
        with open(prob_shifts) as f:
            results["dpo_probability_shifts"] = json.load(f)

    # Load GRPO group statistics
    grpo_stats = SCRIPT_DIR / "outputs" / "grpo" / "group_statistics.json"
    if grpo_stats.exists():
        with open(grpo_stats) as f:
            results["grpo_group_statistics"] = json.load(f)

    # Resource utilization
    results["resource_utilization"] = get_resource_info()

    # ── Save precomputed_results.json ────────────────────────────────
    output_path = WEBAPP_DATA_DIR / "precomputed_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved precomputed_results.json to {output_path}")
    size_mb = output_path.stat().st_size / 1e6
    print(f"  File size: {size_mb:.2f} MB")

    # Also save a copy in the export directory
    export_copy = EXPORT_DIR / "precomputed_results.json"
    with open(export_copy, "w") as f:
        json.dump(results, f, indent=2)

    # Save a standalone resource summary
    resource_summary = results["resource_utilization"]
    resource_summary["model_accuracies"] = {
        variant: results["model_results"][variant]["summary"]["accuracy"]
        for variant in model_variants
    }
    resource_summary["model_avg_gen_times_ms"] = {
        variant: results["model_results"][variant]["summary"]["avg_generation_time_ms"]
        for variant in model_variants
    }

    with open(EXPORT_DIR / "resource_summary.json", "w") as f:
        json.dump(resource_summary, f, indent=2)
    with open(WEBAPP_DATA_DIR / "resource_summary.json", "w") as f:
        json.dump(resource_summary, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Export Complete!")
    print("=" * 60)
    print(f"  Models evaluated:    {', '.join(model_variants)}")
    print(f"  Test prompts:        {len(test_prompts)}")
    print(f"  Results file:        {output_path}")
    print(f"  Results size:        {size_mb:.2f} MB")
    print(f"\n  Accuracy summary:")
    for variant in model_variants:
        s = results["model_results"][variant]["summary"]
        print(f"    {variant:>6s}: {s['accuracy']:.1%} ({s['correct']}/{s['total']})")
    print(f"\n  The web app can now load precomputed_results.json!")


if __name__ == "__main__":
    main()
