"""
============================================================
Step 3: Group Relative Policy Optimization (GRPO)
============================================================
Takes the SFT model and optimizes it using GRPO — a
reinforcement learning approach where:

1. For each prompt, generate K completions (K=8)
2. Score each completion with a reward function
3. Compute advantages RELATIVE to the group mean
4. Update the policy to favor higher-advantage completions

Key difference from DPO:
  - DPO uses pre-generated preference pairs (offline)
  - GRPO generates completions on-the-fly and scores them
    with a reward function (online RL)

Our reward function is simple: binary classification accuracy.
  correct classification = 1.0
  incorrect = 0.0

This lets the model explore and discover that being concise
and correct is the winning strategy.

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
import re
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, PeftModel, get_peft_model
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
             "sft_batch": 4, "grpo_batch": 2, "grpo_gens": 8},
    "1.7B": {"name": "HuggingFaceTB/SmolLM2-1.7B", "slug": "smollm2-1.7b",
             "sft_batch": 2, "grpo_batch": 1, "grpo_gens": 4},
}


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training")
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
        output_dir = SCRIPT_DIR / "outputs" / "grpo"
    else:
        slug = MODELS[model_size]["slug"]
        sft_dir = SCRIPT_DIR / "outputs" / slug / "sft"
        output_dir = SCRIPT_DIR / "outputs" / slug / "grpo"
    return sft_dir, output_dir


# ── Paths (defaults for 360M, overridden in main() based on args) ────
SCRIPT_DIR = Path(__file__).resolve().parent
SFT_DIR = SCRIPT_DIR / "outputs" / "sft"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "grpo"

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"

# ====================================================================
# 1. DATA + REWARD FUNCTION
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
        f"Choose one of: OLTP Database, OLAP Analytics, AI ML Training, "
        f"Video Streaming, VDI Virtual Desktop, Backup Archive.\n"
        f"Provide the classification."
    )


# Keyword shortcuts: if the model generates a near-miss label like
# "VDI Virtual Machine Delivery" or "Backup Video Archive", we can
# still identify the intended category by matching distinguishing keywords.
# Ordered from most-specific to least-specific to avoid false matches.
LABEL_KEYWORDS = {
    "OLTP Database":       ["oltp", "transactional", "database"],
    "OLAP Analytics":      ["olap", "analytics", "analytical", "warehouse", "data warehouse"],
    "AI ML Training":      ["ai ml", "ml training", "ai training", "machine learning",
                            "gpu training", "training data", "model training"],
    "Video Streaming":     ["video streaming", "streaming", "media streaming", "video"],
    "VDI Virtual Desktop": ["vdi", "virtual desktop", "desktop"],
    "Backup Archive":      ["backup", "archive", "archival"],
}


def extract_classification(text: str) -> str:
    """
    Parse the model's output to extract the predicted classification label.
    Returns the label string if found, or empty string if parsing fails.

    Robust extraction strategy:
    1. Try to match "Classification: <label>" — exact label match
    2. Try keyword matching on the Classification line (handles near-miss labels
       like "VDI Virtual Machine Delivery" → matches "vdi" → VDI Virtual Desktop)
    3. Fall back to matching any known label in the full text
    4. Fall back to keyword matching on the full text
    5. Log when extraction fails for debugging
    """
    text_lower = text.lower()

    # Strategy 1: Try to match "Classification: <label>" with exact label match
    match = re.search(r"Classification:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        predicted = match.group(1).strip().lower()
        # Exact or substring match against known labels
        for label in LABELS:
            if label.lower() in predicted or predicted in label.lower():
                return label
        # Strategy 2: Keyword match on the Classification line
        for label, keywords in LABEL_KEYWORDS.items():
            for kw in keywords:
                if kw in predicted:
                    return label

    # Strategy 3: Fall back to matching any known label in the full text
    labels_by_length = sorted(LABELS, key=len, reverse=True)
    for label in labels_by_length:
        if label.lower() in text_lower:
            return label

    # Strategy 4: Keyword match on the full text
    for label, keywords in LABEL_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                return label

    # No match — count failures silently (printed in summary at end)
    return ""


# Module-level counters for debug logging
_reward_debug_counter = 0
_REWARD_DEBUG_LIMIT = 10
_extract_fail_count = 0


def reward_fn(completions: list[str], true_label: str) -> list[float]:
    """
    Binary reward function for classification accuracy.
      1.0 -> correct classification
      0.0 -> incorrect or unparseable

    This is intentionally simple — more complex reward functions could
    give partial credit for close answers, reward conciseness, etc.

    Debug logging: prints the first few raw completions to stdout so the
    user can see what the model is actually generating during GRPO.
    """
    global _reward_debug_counter, _extract_fail_count
    rewards = []
    for completion in completions:
        predicted = extract_classification(completion)
        if not predicted:
            _extract_fail_count += 1
        reward = 1.0 if predicted == true_label else 0.0
        rewards.append(reward)

        # Debug: print first N completions to stdout for visibility
        if _reward_debug_counter < _REWARD_DEBUG_LIMIT:
            snippet = completion[:120].replace("\n", " | ")
            print(f"  [GRPO DEBUG #{_reward_debug_counter}] true={true_label}, "
                  f"predicted={predicted or '(empty)'}, reward={reward:.0f}, "
                  f"text='{snippet}...'")
            _reward_debug_counter += 1

    return rewards


def build_prompt_dataset(n_per_class: int = 30) -> tuple[Dataset, dict]:
    """Build a dataset of prompts (no completions — GRPO generates those).

    Each prompt ends with "\\n\\nClassification:" so the model starts
    generating the label immediately (matching SFT training format).
    """
    prompts = []
    labels = []
    for label in LABELS:
        for _ in range(n_per_class):
            prompts.append(generate_io_prompt(label))
            labels.append(label)

    # Shuffle together
    combined = list(zip(prompts, labels))
    random.shuffle(combined)
    prompts, labels = zip(*combined)

    # Build a lookup from prompt (with suffix) to true label for the reward function
    PROMPT_SUFFIX = "\n\nClassification:"
    full_prompts = [p + PROMPT_SUFFIX for p in prompts]
    prompt_to_label = dict(zip(full_prompts, labels))

    print(f"  Created {len(prompts)} prompts ({n_per_class} per class)")

    dataset = Dataset.from_dict({"prompt": list(full_prompts)})
    return dataset, prompt_to_label


# ====================================================================
# 2. TRY TRL'S GRPOTrainer, FALL BACK TO MANUAL IMPLEMENTATION
# ====================================================================
# TRL added GRPOTrainer in v0.12+. If the installed version doesn't
# have it, we implement a simplified GRPO training loop manually.
# ====================================================================

def check_grpo_available():
    """Check if TRL has GRPOTrainer."""
    try:
        from trl import GRPOTrainer, GRPOConfig
        return True
    except ImportError:
        return False


# ====================================================================
# 3. MANUAL GRPO IMPLEMENTATION (fallback)
# ====================================================================
# This is a simplified but complete implementation of GRPO that's
# easier to understand than the full TRL version.  Great for
# educational purposes!
# ====================================================================

class SimpleGRPO:
    """
    Simplified Group Relative Policy Optimization.

    For each training step:
    1. Sample a batch of prompts
    2. For each prompt, generate K completions
    3. Score each completion with the reward function
    4. Compute group-relative advantages:
       advantage_i = (reward_i - group_mean) / group_std
    5. Update the policy using the REINFORCE-style gradient:
       loss = -sum(advantage_i * log_prob(completion_i))
       (with KL penalty to stay near the reference policy)

    Single-model architecture: uses adapter enable/disable for
    reference log probs instead of loading a separate reference model.
    When use_adapter_ref=True, the model's LoRA adapters are disabled
    to compute reference log probs (i.e., the merged SFT base serves
    as the reference).
    """

    def __init__(self, model, tokenizer, device,
                 ref_model=None, use_adapter_ref=False,
                 num_generations=8, lr=1e-5, kl_coeff=0.05, max_new_tokens=80):
        self.model = model
        self.ref_model = ref_model
        self.use_adapter_ref = use_adapter_ref
        self.tokenizer = tokenizer
        self.device = device
        self.num_generations = num_generations
        self.kl_coeff = kl_coeff
        self.max_new_tokens = max_new_tokens

        # Only optimize trainable (LoRA) parameters
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )

        # Tracking
        self.logs = []
        self.generation_logs = []

    def generate_completions(self, prompt_text: str) -> list[str]:
        """Generate K completions for a single prompt using sampling.

        Expects prompt_text to already end with "\\n\\nClassification:"
        so the model starts generating the label immediately.
        """
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=400
        ).to(self.device)

        completions = []
        self.model.eval()  # Use eval mode for generation
        with torch.no_grad():
            for _ in range(self.num_generations):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.85,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                # Prepend "Classification: " since it was part of the prompt
                completions.append("Classification: " + generated.strip())

        return completions

    def compute_log_probs(self, model, prompt_text: str, completion_text: str) -> torch.Tensor:
        """
        Compute log probability of a completion given a prompt.
        Returns the sum of log probs over completion tokens.
        prompt_text should already end with "\\n\\nClassification:".
        """
        # Strip any "Classification: " prefix from completion to avoid duplication
        clean_completion = completion_text.replace("Classification: ", "", 1).replace("Classification:", "", 1)
        # Ensure leading space for clean BPE boundary with "Classification:" suffix
        if clean_completion and not clean_completion.startswith(" "):
            clean_completion = " " + clean_completion
        full_text = prompt_text + clean_completion
        inputs = self.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        prompt_only = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=400
        )
        prompt_len = prompt_only["input_ids"].shape[1]

        outputs = model(**inputs)
        logits = outputs.logits

        # Get log probs for the completion tokens only
        # Shift logits and labels: logits[t] predicts token[t+1]
        shift_logits = logits[:, prompt_len - 1:-1, :]
        shift_labels = inputs["input_ids"][:, prompt_len:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        return token_log_probs.sum()

    def _compute_ref_log_probs(self, prompt_text: str, completion_text: str) -> torch.Tensor:
        """Compute reference log probs using either a separate ref model or adapter disable."""
        if self.use_adapter_ref:
            # Single-model path: disable LoRA adapters to get reference (SFT base) log probs
            with torch.no_grad():
                with self.model.disable_adapter():
                    return self.compute_log_probs(self.model, prompt_text, completion_text)
        else:
            # Two-model path: use the separate frozen reference model
            with torch.no_grad():
                return self.compute_log_probs(self.ref_model, prompt_text, completion_text)

    def train_step(self, prompts: list[str], true_labels: list[str], step_num: int):
        """
        One GRPO training step over a batch of prompts.

        Returns dict with metrics for logging.
        """
        total_loss = 0.0
        total_reward = 0.0
        total_correct = 0
        total_completions = 0
        step_generation_log = []

        self.model.train()

        for prompt, true_label in zip(prompts, true_labels):
            # Step 1: Generate K completions
            completions = self.generate_completions(prompt)

            # Step 2: Score each completion
            rewards = reward_fn(completions, true_label)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

            # Step 3: Compute group-relative advantages
            # This is the "Group Relative" part of GRPO:
            # Instead of using an absolute baseline, we normalize
            # advantages within the group of K completions.
            group_mean = rewards_tensor.mean()
            group_std = rewards_tensor.std()
            if group_std < 1e-8:
                # All rewards are the same — no signal to learn from
                advantages = torch.zeros_like(rewards_tensor)
            else:
                advantages = (rewards_tensor - group_mean) / (group_std + 1e-8)

            # Log this generation group (for the web app to visualize)
            if len(self.generation_logs) < 25:  # Save up to 25 examples
                gen_log_entry = {
                    "step": step_num,
                    "prompt_snippet": prompt[:120] + "...",
                    "true_label": true_label,
                    "completions": [],
                    "group_mean_reward": round(group_mean.item(), 4),
                    "group_std_reward": round(group_std.item(), 4),
                }
                for j, (comp, rew, adv) in enumerate(zip(completions, rewards, advantages)):
                    predicted = extract_classification(comp)
                    gen_log_entry["completions"].append({
                        "text": comp[:200],  # Truncate for storage
                        "predicted_label": predicted,
                        "reward": round(rew, 4),
                        "advantage": round(adv.item(), 4),
                    })
                self.generation_logs.append(gen_log_entry)

            # Step 4: Compute GRPO loss
            # loss = -sum(advantage_i * log_pi(completion_i))
            #        + kl_coeff * KL(pi || ref)
            prompt_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            for completion, advantage in zip(completions, advantages):
                if abs(advantage.item()) < 1e-8:
                    continue  # Skip zero-advantage completions

                # Log prob under current policy
                policy_log_prob = self.compute_log_probs(self.model, prompt, completion)

                # Log prob under reference (frozen SFT model) for KL penalty
                ref_log_prob = self._compute_ref_log_probs(prompt, completion)

                # KL divergence estimate (per-sample)
                kl = policy_log_prob - ref_log_prob

                # GRPO objective: maximize advantage-weighted log prob, minus KL penalty
                # We negate because optimizers minimize
                sample_loss = -(advantage.item() * policy_log_prob) + self.kl_coeff * kl
                prompt_loss = prompt_loss + sample_loss

            # Accumulate metrics
            total_loss += prompt_loss.item()
            total_reward += sum(rewards)
            total_correct += sum(1 for r in rewards if r > 0)
            total_completions += len(completions)

            # Backward pass
            if prompt_loss.requires_grad:
                prompt_loss.backward()

        # Optimizer step (after accumulating gradients from all prompts in batch)
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Metrics
        metrics = {
            "step": step_num,
            "loss": round(total_loss / max(len(prompts), 1), 6),
            "mean_reward": round(total_reward / max(total_completions, 1), 4),
            "accuracy": round(total_correct / max(total_completions, 1), 4),
            "total_correct": total_correct,
            "total_completions": total_completions,
        }
        self.logs.append(metrics)
        return metrics


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
    print("  STEP 3: Group Relative Policy Optimization (GRPO)")
    print(f"  Model: {cfg['name']} ({args.model_size})")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Using device: {device}")
    if device != "cuda":
        print("[WARNING] No GPU detected. Training will be VERY slow!")

    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    # ── Determine single-model vs two-model architecture ─────────────
    # For 1.7B, loading two full model copies won't fit on a T4 (16 GB).
    # Instead, we load one base model, merge SFT, add LoRA for GRPO,
    # and use disable_adapter() to get reference log probs.
    use_single_model = (args.model_size != "360M")

    # ── Load base model + SFT adapter ────────────────────────────────
    print(f"\n[1/5] Loading base model + SFT adapter...")
    tokenizer = AutoTokenizer.from_pretrained(str(SFT_ADAPTER), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Policy model (will be trained)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, dtype=dtype,
    ).to(device)
    policy_model = PeftModel.from_pretrained(base_model, str(SFT_ADAPTER))
    policy_model = policy_model.merge_and_unload()

    ref_model = None
    if not use_single_model:
        # Two-model path (360M): load a separate frozen reference copy
        ref_base = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, trust_remote_code=True, dtype=dtype,
        ).to(device)
        ref_model = PeftModel.from_pretrained(ref_base, str(SFT_ADAPTER))
        ref_model = ref_model.merge_and_unload()
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    else:
        print("  [INFO] Using single-model architecture (adapter disable for reference)")

    print(f"  Policy model parameters: {policy_model.num_parameters():,}")

    # ── Check if TRL's GRPOTrainer is available ──────────────────────
    use_trl_grpo = check_grpo_available()

    if use_trl_grpo:
        print("\n[INFO] TRL GRPOTrainer is available — using it.")
    else:
        print("\n[INFO] TRL GRPOTrainer not available — using simplified implementation.")

    # ── Build prompt dataset ─────────────────────────────────────────
    print("\n[2/5] Building prompt dataset...")
    dataset, prompt_to_label = build_prompt_dataset(n_per_class=30)

    # ── Add LoRA adapters to the policy model ────────────────────────
    print("\n[3/5] Adding LoRA adapters for GRPO...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy_model = get_peft_model(policy_model, lora_config)
    trainable, total = policy_model.get_nb_trainable_parameters()
    print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Training ─────────────────────────────────────────────────────
    print("\n[4/5] Starting GRPO training...")
    print(f"""
  {BOLD}What to watch for during GRPO training:{RESET}
  ┌─────────────────────────────────────────────────────────┐
  │ {BOLD}loss{RESET}      Policy gradient loss. Should generally       │
  │           decrease, but can be noisy step-to-step.       │
  │                                                         │
  │ {BOLD}reward{RESET}    Mean reward across generated completions.    │
  │           Should increase — means more correct answers.  │
  │           1.0 = perfect, 0.0 = all wrong.               │
  │                                                         │
  │ {BOLD}accuracy{RESET}  Fraction of completions that got the right   │
  │           label. THE key metric. Should climb upward.    │
  │           If it plateaus, model has hit capacity limit.  │
  │                                                         │
  │ {BOLD}kl{RESET}        KL divergence from reference model. Should   │
  │           stay small (<0.5). High KL = model drifting    │
  │           too far from SFT starting point.               │
  │                                                         │
  │ {GREEN}Green{RESET} = moving in the right direction                  │
  │ {RED}Red{RESET}   = moving the wrong way (investigate if persistent) │
  └─────────────────────────────────────────────────────────┘
""")
    train_start = time.time()

    # Reset debug counter for this run
    global _reward_debug_counter
    _reward_debug_counter = 0

    global _REWARD_DEBUG_LIMIT
    _REWARD_DEBUG_LIMIT = 10 if args.verbose else 0

    batch_size = cfg["grpo_batch"]
    num_generations = cfg["grpo_gens"]

    if use_trl_grpo:
        # ── TRL GRPOTrainer path ─────────────────────────────────────
        from trl import GRPOTrainer, GRPOConfig
        from transformers import TrainerCallback

        class GRPOLossCallback(TrainerCallback):
            def __init__(self):
                self.logs = []
                self._prev = {}
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                entry = {"step": state.global_step}
                for key in ["loss", "reward", "reward_std", "kl",
                            "clip_ratio", "entropy",
                            "frac_reward_zero_std"]:
                    if key in logs:
                        entry[key] = round(logs[key], 6)
                # For binary reward (1=correct, 0=wrong), reward mean = accuracy
                if "reward" in entry:
                    entry["mean_reward"] = entry["reward"]
                    entry["accuracy"] = entry["reward"]
                if len(entry) > 1:
                    self.logs.append(entry)

                # Color-coded output
                step = state.global_step
                epoch = round(state.epoch, 1) if state.epoch else 0
                parts = []

                if "loss" in logs:
                    loss = logs["loss"]
                    l_color = self._color("loss", loss, "down")
                    parts.append(f"loss: {l_color}{loss:.4f}{RESET}")

                if "reward" in logs:
                    reward = logs["reward"]
                    r_color = self._color("reward", reward, "up")
                    parts.append(f"reward: {r_color}{reward:.3f}{RESET}")

                if "kl" in logs:
                    kl = logs["kl"]
                    kl_color = RED if kl > 1.0 else (YELLOW if kl > 0.5 else "")
                    parts.append(f"kl: {kl_color}{kl:.3f}{RESET}")

                if parts:
                    # Progress bar
                    total_steps = state.max_steps if state.max_steps and state.max_steps > 0 else 1
                    pct = min(step / total_steps, 1.0) if total_steps > 1 else 0
                    filled = int(pct * 20)
                    bar_str = f"{'█' * filled}{'░' * (20 - filled)}"
                    print(f"    [{bar_str}] {pct:>5.0%} Step {step:>4d} │ {' │ '.join(parts)}")

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
                self.probes = [
                    {"prompt": generate_io_prompt("Backup Archive") + "\n\nClassification:", "label": "Backup Archive"},
                    {"prompt": generate_io_prompt("OLTP Database") + "\n\nClassification:", "label": "OLTP Database"},
                    {"prompt": generate_io_prompt("VDI Virtual Desktop") + "\n\nClassification:", "label": "VDI Virtual Desktop"},
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
                    inputs = self.tokenizer(probe["prompt"], return_tensors="pt", truncation=True, max_length=480).to(self.device)

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

        grpo_callback = GRPOLossCallback()
        probe_callback = LiveProbeCallback(tokenizer, device, probe_interval=20)

        # Build reward function compatible with TRL's interface
        def trl_reward_fn(completions, **kwargs):
            """
            TRL GRPOTrainer passes completions as a list of strings.
            We need to extract the prompt info from kwargs to score them.
            """
            prompts = kwargs.get("prompts", kwargs.get("prompt", []))
            rewards = []
            for prompt_text, completion_text in zip(prompts, completions):
                true_label = prompt_to_label.get(prompt_text, "")
                if true_label:
                    predicted = extract_classification(completion_text)
                    rewards.append(1.0 if predicted == true_label else 0.0)
                else:
                    rewards.append(0.0)
            return rewards

        # TRL GRPOTrainer requires generation_batch_size divisible by num_generations.
        # Use num_generations=4 for TRL path (manual path uses the full cfg value).
        trl_num_gens = min(num_generations, 4)
        trl_batch_size = max(batch_size, trl_num_gens)  # Must be >= num_generations

        grpo_base_kwargs = dict(
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            num_train_epochs=2,
            per_device_train_batch_size=trl_batch_size,
            gradient_accumulation_steps=2,
            learning_rate=1e-5,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            num_generations=trl_num_gens,
            temperature=0.5,       # Lower temp = stay closer to SFT policy
            logging_steps=5 if args.verbose else 10,
            save_strategy="epoch",
            seed=SEED,
            bf16=(device == "cuda"),
            max_completion_length=80,
            report_to="none",
        )

        # Build config, dropping any args the installed TRL version doesn't recognize
        try:
            grpo_config = GRPOConfig(**grpo_base_kwargs)
        except TypeError as e:
            # Remove unrecognized kwargs and retry
            bad_arg = str(e).split("'")[1] if "'" in str(e) else None
            if bad_arg and bad_arg in grpo_base_kwargs:
                del grpo_base_kwargs[bad_arg]
                grpo_config = GRPOConfig(**grpo_base_kwargs)
            else:
                raise

        # Suppress default trainer logging, noisy TRL warnings, and deprecation warnings
        import logging as _logging
        import warnings
        _logging.getLogger("transformers.trainer").setLevel(_logging.WARNING)
        _logging.getLogger("trl.trainer.grpo_trainer").setLevel(_logging.ERROR)
        _logging.getLogger("trl.trainer.sft_trainer").setLevel(_logging.ERROR)
        warnings.filterwarnings("ignore", message=".*generation_config.*deprecated.*")

        trainer = GRPOTrainer(
            model=policy_model,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_funcs=trl_reward_fn,
            callbacks=[grpo_callback, probe_callback],
        )

        train_result = trainer.train()
        train_time = time.time() - train_start
        training_logs = grpo_callback.logs
        generation_logs = []  # TRL doesn't expose individual generations easily

        # Save adapter
        adapter_path = OUTPUT_DIR / "adapter"
        trainer.model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

    else:
        # ── Manual GRPO path ─────────────────────────────────────────
        grpo = SimpleGRPO(
            model=policy_model,
            tokenizer=tokenizer,
            device=device,
            ref_model=ref_model,
            use_adapter_ref=use_single_model,
            num_generations=num_generations,
            lr=1e-5,
            kl_coeff=0.05,
            max_new_tokens=80,
        )

        prompts = dataset["prompt"]
        num_epochs = 2
        step = 0
        import math
        total_steps_manual = num_epochs * math.ceil(len(prompts) / batch_size)

        for epoch in range(num_epochs):
            print(f"\n  Epoch {epoch + 1}/{num_epochs}")
            # Shuffle prompts each epoch
            indices = list(range(len(prompts)))
            random.shuffle(indices)

            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                batch_prompts = [prompts[i] for i in batch_indices]
                batch_labels = [prompt_to_label[p] for p in batch_prompts]

                metrics = grpo.train_step(batch_prompts, batch_labels, step)
                step += 1

                if step % (5 if args.verbose else 8) == 0:
                    # Color-code based on direction
                    loss_val = metrics['loss']
                    reward_val = metrics['mean_reward']
                    acc_val = metrics['accuracy']

                    if step > (5 if args.verbose else 20):
                        prev_step = grpo.logs[-2] if len(grpo.logs) > 1 else None
                    else:
                        prev_step = None

                    if prev_step:
                        l_color = GREEN if loss_val < prev_step['loss'] else (RED if loss_val > prev_step['loss'] * 1.1 else YELLOW)
                        r_color = GREEN if reward_val > prev_step.get('mean_reward', 0) else (RED if reward_val < prev_step.get('mean_reward', 0) * 0.9 else YELLOW)
                        a_color = GREEN if acc_val > prev_step.get('accuracy', 0) else (RED if acc_val < prev_step.get('accuracy', 0) * 0.9 else YELLOW)
                    else:
                        l_color = r_color = a_color = BOLD

                    # Progress bar
                    pct_m = min(step / total_steps_manual, 1.0) if total_steps_manual > 1 else 0
                    filled_m = int(pct_m * 20)
                    bar_m = f"{'█' * filled_m}{'░' * (20 - filled_m)}"
                    print(f"    [{bar_m}] {pct_m:>5.0%} Step {step:>4d}: loss={l_color}{loss_val:.4f}{RESET} │ "
                          f"reward={r_color}{reward_val:.3f}{RESET} │ "
                          f"accuracy={a_color}{acc_val:.3f}{RESET}")

                    # Live probe (manual GRPO path)
                    if step % 20 == 0:
                        policy_model.eval()
                        print(f"\n    {'─' * 56}")
                        print(f"    {BOLD}LIVE PROBE — Step {step}{RESET}")
                        print(f"    {'─' * 56}")
                        probe_prompts_manual = [
                            {"prompt": generate_io_prompt("Backup Archive") + "\n\nClassification:", "label": "Backup Archive"},
                            {"prompt": generate_io_prompt("OLTP Database") + "\n\nClassification:", "label": "OLTP Database"},
                            {"prompt": generate_io_prompt("VDI Virtual Desktop") + "\n\nClassification:", "label": "VDI Virtual Desktop"},
                        ]
                        for probe in probe_prompts_manual:
                            inp = tokenizer(probe["prompt"], return_tensors="pt", truncation=True, max_length=480).to(device)
                            with torch.no_grad():
                                out = policy_model.generate(**inp, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                            gen_text = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
                            expected = probe["label"]
                            is_correct = expected.lower() in gen_text.lower()
                            c = GREEN if is_correct else RED
                            b = "\u2713" if is_correct else "\u2717"
                            print(f"    {c}{b}{RESET} Expected: {BOLD}{expected}{RESET}")
                            print(f"      Model: {c}{gen_text[:100].replace(chr(10), ' | ')}{RESET}")
                        policy_model.train()
                        print()

        train_time = time.time() - train_start
        training_logs = grpo.logs
        generation_logs = grpo.generation_logs

        # Save adapter
        adapter_path = OUTPUT_DIR / "adapter"
        policy_model.save_pretrained(str(adapter_path))
        tokenizer.save_pretrained(str(adapter_path))

    print(f"\n  GRPO training complete in {train_time:.1f}s")

    # ── Save artifacts ───────────────────────────────────────────────
    print("\n[5/5] Saving artifacts...")

    print(f"  Saved GRPO adapter to {adapter_path}")

    # Save training curves
    with open(OUTPUT_DIR / "training_curves.json", "w") as f:
        json.dump({
            "loss_curve": training_logs,  # [{"step": 0, "loss": ..., ...}, ...]
            "logs": training_logs,        # backward-compat alias
            "training_time_seconds": round(train_time, 2),
            "total_steps": len(training_logs),
            "used_trl_grpo": use_trl_grpo,
        }, f, indent=2)

    # Save live probe history for visualization
    probe_history = []
    if use_trl_grpo and hasattr(probe_callback, 'probe_history'):
        probe_history = probe_callback.probe_history
    if probe_history:
        probe_path = OUTPUT_DIR / "probe_history.json"
        with open(probe_path, "w") as f:
            json.dump({"probes": probe_history}, f, indent=2)
        print(f"  Probe history → {probe_path}")

    # Save generation logs in the format the web app expects:
    # {"examples": [{"input": "...", "true_label": "...",
    #   "generations": [{"text": "...", "reward": 1.0, "correct": true}]}]}
    def format_generation_logs(raw_logs):
        """Convert internal generation log format to web app expected format."""
        formatted = []
        for entry in raw_logs:
            example = {
                "input": entry.get("prompt_snippet", ""),
                "true_label": entry.get("true_label", ""),
                "generations": [],
            }
            for comp in entry.get("completions", []):
                predicted = comp.get("predicted_label", "")
                t_label = entry.get("true_label", "")
                example["generations"].append({
                    "text": comp.get("text", ""),
                    "reward": comp.get("reward", 0.0),
                    "correct": predicted == t_label and predicted != "",
                })
            formatted.append(example)
        return formatted

    if generation_logs:
        formatted_gen_logs = format_generation_logs(generation_logs)
        with open(OUTPUT_DIR / "generation_logs.json", "w") as f:
            json.dump({"examples": formatted_gen_logs}, f, indent=2)
        print(f"  Saved {len(formatted_gen_logs)} generation log entries "
              f"(each with {num_generations} completions + scores)")
    else:
        # If using TRL's GRPOTrainer, generate some example logs post-hoc
        print("  Generating example generation logs post-training...")
        example_logs_raw = []
        policy_model.eval()

        for i, label in enumerate(LABELS):
            if i >= 4:  # Save at least 4 examples
                break
            prompt = generate_io_prompt(label)
            completions = []
            with torch.no_grad():
                for _ in range(num_generations):
                    text = prompt + "\n\nClassification:"
                    inputs = tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=400).to(device)
                    output_ids = policy_model.generate(
                        **inputs,
                        max_new_tokens=80,
                        do_sample=True,
                        temperature=0.5,
                        top_p=0.85,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    generated = tokenizer.decode(
                        output_ids[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    completions.append("Classification: " + generated.strip())

            rewards = reward_fn(completions, label)

            entry_raw = {
                "prompt_snippet": prompt[:120] + "...",
                "true_label": label,
                "completions": [],
            }
            for comp, rew in zip(completions, rewards):
                predicted = extract_classification(comp)
                entry_raw["completions"].append({
                    "text": comp[:200],
                    "predicted_label": predicted,
                    "reward": round(rew, 4),
                })
            example_logs_raw.append(entry_raw)

        formatted_gen_logs = format_generation_logs(example_logs_raw)
        with open(OUTPUT_DIR / "generation_logs.json", "w") as f:
            json.dump({"examples": formatted_gen_logs}, f, indent=2)
        print(f"  Saved {len(formatted_gen_logs)} post-training generation examples.")
        generation_logs = example_logs_raw

    # Save group statistics over training in the format the web app expects:
    # {"accuracy_curve": [{"step": 0, "accuracy": 0.45}, ...],
    #  "reward_curve": [{"step": 0, "mean_reward": 0.3}, ...],
    #  "training_time_seconds": 2100}
    if training_logs:
        accuracy_curve = [
            {"step": l["step"], "accuracy": l.get("accuracy", 0)}
            for l in training_logs
        ]
        reward_curve = [
            {"step": l["step"], "mean_reward": l.get("mean_reward", l.get("reward", 0))}
            for l in training_logs
        ]

        stats_summary = {
            "accuracy_curve": accuracy_curve,
            "reward_curve": reward_curve,
            "training_time_seconds": round(train_time, 2),
            "total_steps": len(training_logs),
            "initial_accuracy": training_logs[0].get("accuracy", 0),
            "final_accuracy": training_logs[-1].get("accuracy", 0),
            "initial_reward": training_logs[0].get("mean_reward", 0),
            "final_reward": training_logs[-1].get("mean_reward", 0),
        }
        with open(OUTPUT_DIR / "group_statistics.json", "w") as f:
            json.dump(stats_summary, f, indent=2)

    # ── Sanity check: run 5 sample inferences ────────────────────
    print("\n  Running post-GRPO sanity check...")

    sanity_labels = LABELS[:5]
    sanity_results = []
    policy_model.eval()

    # Load SFT sanity check for comparison
    sft_sanity_path = SFT_DIR / "sft_sanity_check.json"
    sft_accuracy = None
    if sft_sanity_path.exists():
        with open(sft_sanity_path) as f:
            sft_sanity = json.load(f)
            sft_accuracy = sft_sanity.get("sample_accuracy", None)

    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    for label in sanity_labels:
        prompt = generate_io_prompt(label) + "\n\nClassification:"
        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(device)
        input_len = input_ids.shape[1]
        # Manual greedy decoding — more reliable than model.generate() post-TRL
        with torch.no_grad():
            for _ in range(80):
                logits = policy_model(input_ids).logits[:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True)
                if next_token.item() == eos_id:
                    break
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        generated = tokenizer.decode(input_ids[0][input_len:], skip_special_tokens=True).strip()

        predicted = extract_classification(generated)
        correct = predicted == label
        sanity_results.append({
            "expected": label,
            "predicted": predicted if predicted else "(unparseable)",
            "generated_snippet": generated[:100],
            "correct": correct,
        })

    num_correct = sum(1 for r in sanity_results if r["correct"])
    sample_accuracy = num_correct / len(sanity_results)

    # Get accuracy trajectory from training logs
    initial_accuracy = training_logs[0].get("accuracy", 0) if training_logs else None
    mid_idx = len(training_logs) // 2 if training_logs else 0
    mid_accuracy = training_logs[mid_idx].get("accuracy", 0) if training_logs and mid_idx > 0 else None
    final_accuracy = training_logs[-1].get("accuracy", 0) if training_logs else None

    # Find best and worst from generation logs
    best_gen = None
    worst_gen = None
    if generation_logs:
        for entry in generation_logs:
            for comp in entry.get("completions", []):
                if comp.get("reward", 0) > 0 and best_gen is None:
                    best_gen = {"label": entry["true_label"], "text": comp["text"][:120]}
                if comp.get("reward", 0) == 0 and worst_gen is None:
                    worst_gen = {"label": entry["true_label"], "text": comp["text"][:120]}
                if best_gen and worst_gen:
                    break
            if best_gen and worst_gen:
                break

    # Determine verdict
    checks = []
    if initial_accuracy is not None and final_accuracy is not None:
        if final_accuracy > initial_accuracy:
            checks.append(("pass", f"Accuracy improved during training ({initial_accuracy:.0%} → {final_accuracy:.0%})"))
        elif final_accuracy == initial_accuracy:
            checks.append(("warn", f"Accuracy plateaued ({final_accuracy:.0%}) — model may have hit capacity"))
        else:
            checks.append(("warn", f"Accuracy decreased ({initial_accuracy:.0%} → {final_accuracy:.0%})"))

    if sft_accuracy is not None:
        if sample_accuracy >= sft_accuracy:
            checks.append(("pass", f"GRPO improved over SFT ({sample_accuracy:.0%} vs {sft_accuracy:.0%})"))
        else:
            checks.append(("warn", f"GRPO below SFT ({sample_accuracy:.0%} vs {sft_accuracy:.0%})"))

    any_fail = any(c[0] == "fail" for c in checks)
    all_pass = all(c[0] == "pass" for c in checks)
    verdict = "HEALTHY" if all_pass else ("ISSUES DETECTED" if any_fail else "OK WITH WARNINGS")

    # Print structured results
    print("\n" + "=" * 60)
    print("  GRPO RESULTS")
    print("=" * 60)
    print(f"  Training:  {train_time/60:.0f}m {train_time%60:.0f}s")
    if initial_accuracy is not None:
        trajectory = f"{initial_accuracy:.0%}"
        if mid_accuracy is not None:
            trajectory += f" → {mid_accuracy:.0%}"
        if final_accuracy is not None:
            trajectory += f" → {final_accuracy:.0%}"
        print(f"  Accuracy trajectory: {trajectory}")
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

    # ── Per-class accuracy breakdown ──────────────────────────────────
    print(f"\n  {BOLD}Per-class accuracy breakdown:{RESET}")
    per_class_correct = {label: 0 for label in LABELS}
    per_class_total = {label: 0 for label in LABELS}
    per_class_predictions = {label: [] for label in LABELS}

    val_prompts = []
    for label in LABELS:
        for _ in range(5):
            val_prompts.append({"prompt": generate_io_prompt(label), "label": label})

    for sample in val_prompts:
        prompt_text = sample["prompt"] + "\n\nClassification:"
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            output_ids = policy_model.generate(
                **inputs, max_new_tokens=60, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = extract_classification("Classification: " + generated.strip())
        true_label = sample["label"]
        per_class_total[true_label] += 1
        per_class_correct[true_label] += 1 if predicted == true_label else 0
        per_class_predictions[true_label].append(predicted or "(unparseable)")

    total_correct = sum(per_class_correct.values())
    total_n = sum(per_class_total.values())
    print(f"  Overall: {total_correct}/{total_n} ({100*total_correct/total_n:.0f}%)")
    for label in LABELS:
        n = per_class_total[label]
        c = per_class_correct[label]
        preds = per_class_predictions[label]
        color = GREEN if c >= 3 else (YELLOW if c >= 1 else RED)
        print(f"    {color}{label:25s}: {c}/{n}  predictions: {preds}{RESET}")

    from collections import Counter
    all_preds = [p for preds in per_class_predictions.values() for p in preds]
    pred_counts = Counter(all_preds)
    most_common = pred_counts.most_common(1)[0]
    if most_common[1] > len(all_preds) * 0.5:
        print(f"\n  {RED}{BOLD}⚠ MODE COLLAPSE DETECTED: '{most_common[0]}' predicted {most_common[1]}/{len(all_preds)} times{RESET}")

    if best_gen:
        print(f"\n  Best output:  [{best_gen['label']}] {best_gen['text']}")
    if worst_gen:
        print(f"  Worst output: [{worst_gen['label']}] {worst_gen['text']}")

    print()
    print("  Sanity check:")
    for status, msg in checks:
        icon = "✓" if status == "pass" else ("⚠" if status == "warn" else "✗")
        print(f"    {icon} {msg}")
    print(f"\n  Verdict: {verdict} — {'proceed to Export' if not any_fail else 'investigate'}")
    print("=" * 60)

    # Save sanity check JSON
    sanity_data = {
        "training_time_seconds": round(train_time, 2),
        "initial_accuracy": initial_accuracy,
        "mid_accuracy": mid_accuracy,
        "final_accuracy": final_accuracy,
        "sample_results": sanity_results,
        "sample_accuracy": sample_accuracy,
        "sft_accuracy": sft_accuracy,
        "best_generation": best_gen,
        "worst_generation": worst_gen,
        "checks": [{"status": s, "message": m} for s, m in checks],
        "verdict": verdict,
    }
    with open(OUTPUT_DIR / "grpo_sanity_check.json", "w") as f:
        json.dump(sanity_data, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GRPO Training Complete!")
    print("=" * 60)
    print(f"  Model:               {cfg['name']} ({args.model_size})")
    print(f"  Adapter saved to:    {adapter_path}")
    print(f"  Training time:       {train_time:.1f}s")
    if training_logs:
        final = training_logs[-1]
        print(f"  Final accuracy:      {final.get('accuracy', 'N/A')}")
        print(f"  Final mean reward:   {final.get('mean_reward', final.get('reward', 'N/A'))}")
    print(f"  Generation logs:     {len(generation_logs) if generation_logs else 'post-hoc'}")
    print(f"  Artifacts in:        {OUTPUT_DIR}")
    print(f"\n  Next step: Run export_artifacts.py")


if __name__ == "__main__":
    main()
