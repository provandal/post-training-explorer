# Post-Training Pipeline Scripts

Standalone Python scripts that implement the full post-training pipeline for fine-tuning SmolLM2-360M on a storage I/O workload classification task. These are the same scripts that power the Colab notebooks — runnable directly if you prefer a script-based workflow.

## Pipeline Order

Run these scripts in sequence. Each script's output feeds into the next.

| # | Script | Purpose | Input | Output | Runtime |
|---|--------|---------|-------|--------|---------|
| 1 | `generate_dataset.py` | Generate synthetic workload profiles | Workload profile definitions (built-in) | `dataset/` with train/eval JSONL splits | ~10s, CPU |
| 2 | `train_sft.py` | Supervised fine-tuning | `dataset/` + base model from HF Hub | SFT adapter in `models/sft/` | ~12 min, T4 GPU |
| 3 | `train_dpo.py` | Direct Preference Optimization | `dataset/` + SFT adapter | DPO adapter in `models/dpo/` | ~15 min, T4 GPU |
| 4 | `train_grpo.py` | Group Relative Policy Optimization | `dataset/` + SFT adapter | GRPO adapter in `models/grpo/` | ~35 min, T4 GPU |
| 5 | `export_artifacts.py` | Merge adapter weights into base model | Any adapter from step 2-4 | Merged model in `models/merged/` | ~2 min, CPU |
| 6 | `convert_to_onnx.py` | Export to ONNX for browser inference | Merged model from step 5 | ONNX model + tokenizer in `models/onnx/` | ~3 min, CPU |

## Prerequisites

```bash
pip install -r requirements.txt
```

A Colab T4 GPU (free tier) is sufficient for all training scripts. Dataset generation, export, and ONNX conversion run on CPU.

## Quick Start

```bash
# 1. Generate training data
python generate_dataset.py

# 2. Fine-tune with SFT
python train_sft.py

# 3. (Optional) Train DPO on top of SFT
python train_dpo.py

# 4. (Optional) Train GRPO on top of SFT
python train_grpo.py

# 5. Merge adapter into base model
python export_artifacts.py

# 6. Convert to ONNX for browser deployment
python convert_to_onnx.py
```

Steps 3 and 4 are independent — both build on the SFT adapter from step 2. You can run either or both.

## Other Scripts

| Script | Purpose |
|--------|---------|
| `ai_advisor.py` | AINOS Lite — an AI advisor agent for storage infrastructure. Separate from the training pipeline; uses the fine-tuned model for workload classification as part of a larger advisory system. |

## Related

- **Colab notebooks**: `notebooks/` — interactive versions of this pipeline with inline explanations
- **Web app**: `app/` — guided tour and live browser inference using the ONNX model produced by this pipeline
