"""
============================================================
Step 5: Convert Models to ONNX for Browser Inference
============================================================
Merges trained adapters into the base model and converts to
ONNX format for client-side inference via @huggingface/transformers.

Two models are exported:
  1. Base model (untrained) — shows what SmolLM2-360M does without training
  2. GRPO model (SFT + GRPO merged) — shows the fully trained result

Both are pushed to HuggingFace Hub so the web app can download
them directly in the browser.

Run in Google Colab with a GPU runtime after training is complete.
============================================================
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import HfApi, create_repo

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SFT_ADAPTER = SCRIPT_DIR / "outputs" / "sft" / "adapter"
GRPO_ADAPTER = SCRIPT_DIR / "outputs" / "grpo" / "adapter"
ONNX_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "onnx"

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

# ── HuggingFace Hub settings ────────────────────────────────────────
# These will be set based on the authenticated user
HF_BASE_REPO_SUFFIX = "smollm2-360m-storage-io-base-onnx"
HF_GRPO_REPO_SUFFIX = "smollm2-360m-storage-io-grpo-onnx"


def get_hf_username():
    """Get the authenticated HuggingFace username."""
    api = HfApi()
    user_info = api.whoami()
    return user_info["name"]


def save_base_model(output_dir: Path):
    """Save the unmodified base model for ONNX export."""
    print(f"\n[1/6] Saving base model to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32,
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Base model saved to {output_dir}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def merge_grpo_model(output_dir: Path):
    """Merge SFT + GRPO adapters into base and save for ONNX export."""
    print(f"\n[2/6] Merging SFT + GRPO adapters → {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check prerequisites
    if not SFT_ADAPTER.exists():
        print(f"  ERROR: SFT adapter not found at {SFT_ADAPTER}")
        print("  Run train_sft.py first.")
        sys.exit(1)
    if not GRPO_ADAPTER.exists():
        print(f"  ERROR: GRPO adapter not found at {GRPO_ADAPTER}")
        print("  Run train_grpo.py first.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base → merge SFT → merge GRPO
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=torch.float32,
    )

    print("  Merging SFT adapter...")
    sft_model = PeftModel.from_pretrained(base, str(SFT_ADAPTER))
    merged = sft_model.merge_and_unload()

    print("  Merging GRPO adapter...")
    grpo_model = PeftModel.from_pretrained(merged, str(GRPO_ADAPTER))
    final = grpo_model.merge_and_unload()

    final.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Merged GRPO model saved to {output_dir}")

    del final, grpo_model, merged, sft_model, base
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def convert_to_onnx(model_dir: Path, onnx_dir: Path, label: str):
    """Convert a saved model to ONNX using optimum-cli."""
    print(f"\n[{'3' if 'base' in label else '4'}/6] Converting {label} to ONNX → {onnx_dir}")

    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)

    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", str(model_dir),
        "--task", "text-generation-with-past",
        str(onnx_dir),
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        print(f"  ERROR: ONNX conversion failed for {label}")
        sys.exit(1)

    print(f"  ONNX conversion complete for {label}")

    # Report output size
    total_size = sum(f.stat().st_size for f in onnx_dir.rglob("*") if f.is_file())
    print(f"  Output size: {total_size / 1e6:.1f} MB")


def push_to_hub(onnx_dir: Path, repo_id: str, label: str):
    """Push ONNX model directory to HuggingFace Hub."""
    step = "5" if "base" in label else "6"
    print(f"\n[{step}/6] Pushing {label} to HuggingFace Hub → {repo_id}")

    api = HfApi()

    # Create repo if it doesn't exist
    create_repo(repo_id, exist_ok=True, repo_type="model")

    # Upload the entire ONNX directory
    api.upload_folder(
        folder_path=str(onnx_dir),
        repo_id=repo_id,
        commit_message=f"Upload {label} ONNX model for browser inference",
    )

    print(f"  Pushed to https://huggingface.co/{repo_id}")


def main():
    print("=" * 60)
    print("  STEP 5: Convert Models to ONNX for Browser Inference")
    print("=" * 60)

    # ── Verify HuggingFace authentication ────────────────────────────
    try:
        username = get_hf_username()
        print(f"\n  Authenticated as: {username}")
    except Exception as e:
        print(f"\n  ERROR: Not authenticated with HuggingFace Hub")
        print(f"  Run: huggingface-cli login")
        print(f"  Or set the HF_TOKEN environment variable")
        sys.exit(1)

    base_repo = f"{username}/{HF_BASE_REPO_SUFFIX}"
    grpo_repo = f"{username}/{HF_GRPO_REPO_SUFFIX}"

    print(f"  Base model repo: {base_repo}")
    print(f"  GRPO model repo: {grpo_repo}")

    # ── Prepare model directories ────────────────────────────────────
    base_model_dir = ONNX_OUTPUT_DIR / "base_model"
    grpo_model_dir = ONNX_OUTPUT_DIR / "grpo_model"
    base_onnx_dir = ONNX_OUTPUT_DIR / "base_onnx"
    grpo_onnx_dir = ONNX_OUTPUT_DIR / "grpo_onnx"

    # Step 1: Save base model
    save_base_model(base_model_dir)

    # Step 2: Merge and save GRPO model
    merge_grpo_model(grpo_model_dir)

    # Step 3: Convert base to ONNX
    convert_to_onnx(base_model_dir, base_onnx_dir, "base model")

    # Step 4: Convert GRPO to ONNX
    convert_to_onnx(grpo_model_dir, grpo_onnx_dir, "GRPO model")

    # Step 5: Push base ONNX to Hub
    push_to_hub(base_onnx_dir, base_repo, "base model")

    # Step 6: Push GRPO ONNX to Hub
    push_to_hub(grpo_onnx_dir, grpo_repo, "GRPO model")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ONNX Conversion Complete!")
    print("=" * 60)
    print(f"\n  Base ONNX model: https://huggingface.co/{base_repo}")
    print(f"  GRPO ONNX model: https://huggingface.co/{grpo_repo}")
    print(f"\n  Update inference.js MODEL_CONFIGS with these repo IDs:")
    print(f"    base: '{base_repo}'")
    print(f"    grpo: '{grpo_repo}'")
    print(f"\n  The web app's LiveInferencePanel will download these")
    print(f"  models directly in the browser for client-side inference.")


if __name__ == "__main__":
    main()
