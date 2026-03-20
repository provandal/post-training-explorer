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

# ── Paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SFT_DIR = SCRIPT_DIR / "outputs" / "sft"
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "grpo"
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
        f"Provide the workload classification and a brief reason."
    )


def extract_classification(text: str) -> str:
    """
    Parse the model's output to extract the predicted classification label.
    Returns the label string if found, or empty string if parsing fails.
    """
    # Try to match "Classification: <label>"
    match = re.search(r"Classification:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if match:
        predicted = match.group(1).strip()
        # Fuzzy-match against known labels
        for label in LABELS:
            if label.lower() in predicted.lower() or predicted.lower() in label.lower():
                return label
    return ""


def reward_fn(completions: list[str], true_label: str) -> list[float]:
    """
    Binary reward function for classification accuracy.
      1.0 → correct classification
      0.0 → incorrect or unparseable

    This is intentionally simple — more complex reward functions could
    give partial credit for close answers, reward conciseness, etc.
    """
    rewards = []
    for completion in completions:
        predicted = extract_classification(completion)
        rewards.append(1.0 if predicted == true_label else 0.0)
    return rewards


def build_prompt_dataset(n_per_class: int = 30) -> tuple[Dataset, dict]:
    """Build a dataset of prompts (no completions — GRPO generates those)."""
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

    # Build a lookup from prompt to true label for the reward function
    prompt_to_label = dict(zip(prompts, labels))

    print(f"  Created {len(prompts)} prompts ({n_per_class} per class)")

    dataset = Dataset.from_dict({"prompt": list(prompts)})
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
    """

    def __init__(self, model, ref_model, tokenizer, device,
                 num_generations=8, lr=1e-5, kl_coeff=0.05, max_new_tokens=80):
        self.model = model
        self.ref_model = ref_model
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
        """Generate K completions for a single prompt using sampling."""
        full_text = prompt_text + "\n\n"
        inputs = self.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=400
        ).to(self.device)

        completions = []
        self.model.eval()  # Use eval mode for generation
        with torch.no_grad():
            for _ in range(self.num_generations):
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated = self.tokenizer.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                completions.append(generated.strip())

        return completions

    def compute_log_probs(self, model, prompt_text: str, completion_text: str) -> torch.Tensor:
        """
        Compute log probability of a completion given a prompt.
        Returns the sum of log probs over completion tokens.
        """
        full_text = prompt_text + "\n\n" + completion_text
        inputs = self.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        prompt_only = self.tokenizer(
            prompt_text + "\n\n", return_tensors="pt", truncation=True, max_length=400
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
                with torch.no_grad():
                    ref_log_prob = self.compute_log_probs(self.ref_model, prompt, completion)

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
    print("=" * 60)
    print("  STEP 3: Group Relative Policy Optimization (GRPO)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[INFO] Using device: {device}")
    if device != "cuda":
        print("[WARNING] No GPU detected. Training will be VERY slow!")

    # ── Load base model + SFT adapter ────────────────────────────────
    print(f"\n[1/5] Loading base model + SFT adapter...")
    tokenizer = AutoTokenizer.from_pretrained(str(SFT_ADAPTER), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device == "cpu" else torch.bfloat16

    # Policy model (will be trained)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=dtype,
    ).to(device)
    policy_model = PeftModel.from_pretrained(base_model, str(SFT_ADAPTER))
    policy_model = policy_model.merge_and_unload()

    # Reference model (frozen copy of SFT model for KL penalty)
    ref_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, torch_dtype=dtype,
    ).to(device)
    ref_model = PeftModel.from_pretrained(ref_base, str(SFT_ADAPTER))
    ref_model = ref_model.merge_and_unload()
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

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
        lora_alpha=32,
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
    train_start = time.time()

    if use_trl_grpo:
        # ── TRL GRPOTrainer path ─────────────────────────────────────
        from trl import GRPOTrainer, GRPOConfig
        from transformers import TrainerCallback

        class GRPOLossCallback(TrainerCallback):
            def __init__(self):
                self.logs = []
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    entry = {"step": state.global_step}
                    for key in ["loss", "reward", "reward_std", "kl",
                                "clip_ratio", "entropy"]:
                        if key in logs:
                            entry[key] = round(logs[key], 6)
                    if len(entry) > 1:
                        self.logs.append(entry)

        grpo_callback = GRPOLossCallback()

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

        grpo_config = GRPOConfig(
            output_dir=str(OUTPUT_DIR / "checkpoints"),
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=1e-5,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            num_generations=8,
            logging_steps=5,
            save_strategy="epoch",
            seed=SEED,
            bf16=(device == "cuda"),
            max_completion_length=80,
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=policy_model,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            reward_funcs=trl_reward_fn,
            callbacks=[grpo_callback],
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
            ref_model=ref_model,
            tokenizer=tokenizer,
            device=device,
            num_generations=8,
            lr=1e-5,
            kl_coeff=0.05,
            max_new_tokens=80,
        )

        prompts = dataset["prompt"]
        batch_size = 2
        num_epochs = 2
        step = 0

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

                if step % 5 == 0:
                    print(f"    Step {step}: loss={metrics['loss']:.4f}, "
                          f"reward={metrics['mean_reward']:.3f}, "
                          f"accuracy={metrics['accuracy']:.3f}")

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
            "logs": training_logs,
            "training_time_seconds": round(train_time, 2),
            "total_steps": len(training_logs),
            "used_trl_grpo": use_trl_grpo,
        }, f, indent=2)

    # Save generation logs (the rich data showing all 8 generations + scores)
    if generation_logs:
        with open(OUTPUT_DIR / "generation_logs.json", "w") as f:
            json.dump(generation_logs, f, indent=2)
        print(f"  Saved {len(generation_logs)} generation log entries "
              f"(each with {8} completions + scores + advantages)")
    else:
        # If using TRL's GRPOTrainer, generate some example logs post-hoc
        print("  Generating example generation logs post-training...")
        example_logs = []
        policy_model.eval()

        for i, label in enumerate(LABELS):
            if i >= 4:  # Save at least 4 examples
                break
            prompt = generate_io_prompt(label)
            completions = []
            with torch.no_grad():
                for _ in range(8):
                    text = prompt + "\n\n"
                    inputs = tokenizer(text, return_tensors="pt",
                                       truncation=True, max_length=400).to(device)
                    output_ids = policy_model.generate(
                        **inputs,
                        max_new_tokens=80,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.9,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    generated = tokenizer.decode(
                        output_ids[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    )
                    completions.append(generated.strip())

            rewards = reward_fn(completions, label)
            rewards_t = torch.tensor(rewards)
            mean_r = rewards_t.mean().item()
            std_r = rewards_t.std().item()
            advantages = ((rewards_t - mean_r) / (std_r + 1e-8)).tolist() if std_r > 1e-8 else [0.0] * len(rewards)

            entry = {
                "step": "post-training",
                "prompt_snippet": prompt[:120] + "...",
                "true_label": label,
                "completions": [],
                "group_mean_reward": round(mean_r, 4),
                "group_std_reward": round(std_r, 4),
            }
            for comp, rew, adv in zip(completions, rewards, advantages):
                predicted = extract_classification(comp)
                entry["completions"].append({
                    "text": comp[:200],
                    "predicted_label": predicted,
                    "reward": round(rew, 4),
                    "advantage": round(adv, 4),
                })
            example_logs.append(entry)

        with open(OUTPUT_DIR / "generation_logs.json", "w") as f:
            json.dump(example_logs, f, indent=2)
        print(f"  Saved {len(example_logs)} post-training generation examples.")

    # Save group statistics over training
    if training_logs:
        stats_summary = {
            "total_steps": len(training_logs),
            "initial_accuracy": training_logs[0].get("accuracy", 0),
            "final_accuracy": training_logs[-1].get("accuracy", 0),
            "initial_reward": training_logs[0].get("mean_reward", 0),
            "final_reward": training_logs[-1].get("mean_reward", 0),
            "reward_trajectory": [
                {"step": l["step"], "reward": l.get("mean_reward", l.get("reward", 0))}
                for l in training_logs
            ],
        }
        with open(OUTPUT_DIR / "group_statistics.json", "w") as f:
            json.dump(stats_summary, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GRPO Training Complete!")
    print("=" * 60)
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
