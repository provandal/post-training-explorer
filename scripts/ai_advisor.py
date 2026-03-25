"""
============================================================
AI Research Advisor (AINOS Lite)
============================================================
Analyzes training results and proposes next experiments
using Claude as a research strategist.

This demonstrates the core AINOS concept:
  AI proposes -> Human approves -> System executes

Usage:
  python scripts/ai_advisor.py
  python scripts/ai_advisor.py --model-size 1.7B
  python scripts/ai_advisor.py --dry-run  # Skip Claude API call, show what would be sent

Requires:
  pip install anthropic rich
  ANTHROPIC_API_KEY environment variable
============================================================
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

MODELS = {
    "360M": {"name": "HuggingFaceTB/SmolLM2-360M", "slug": "smollm2-360m"},
    "1.7B": {"name": "HuggingFaceTB/SmolLM2-1.7B", "slug": "smollm2-1.7b"},
}


def collect_all_results():
    """Scan outputs/ for all training results across model sizes."""
    results = {}

    for size, cfg in MODELS.items():
        size_results = {"model": cfg["name"], "techniques": {}}

        # Check both legacy paths (360M) and slug-based paths
        if size == "360M":
            base_dirs = [SCRIPT_DIR / "outputs"]
        else:
            base_dirs = [SCRIPT_DIR / "outputs" / cfg["slug"]]

        for base_dir in base_dirs:
            for technique in ["sft", "dpo", "grpo"]:
                adapter_path = base_dir / technique / "adapter"
                if not adapter_path.exists():
                    continue

                tech_info = {"adapter_exists": True}

                # Load training curves
                for fname in ["training_loss.json", "training_curves.json"]:
                    curves_path = base_dir / technique / fname
                    if curves_path.exists():
                        with open(curves_path) as f:
                            data = json.load(f)
                        tech_info["training_time_seconds"] = data.get("training_time_seconds")
                        tech_info["final_loss"] = data.get("final_loss")
                        if "loss_curve" in data and data["loss_curve"]:
                            last = data["loss_curve"][-1]
                            tech_info["final_step_loss"] = last.get("loss")
                        break

                # Load accuracy from group_statistics (GRPO)
                stats_path = base_dir / technique / "group_statistics.json"
                if stats_path.exists():
                    with open(stats_path) as f:
                        stats = json.load(f)
                    tech_info["final_accuracy"] = stats.get("final_accuracy")
                    tech_info["initial_accuracy"] = stats.get("initial_accuracy")

                size_results["techniques"][technique] = tech_info

        # Load export results if available
        for export_path in [
            SCRIPT_DIR / "outputs" / "export" / "precomputed_results.json",
            SCRIPT_DIR.parent / "app" / "public" / "data" / "precomputed_results.json",
        ]:
            if export_path.exists():
                with open(export_path) as f:
                    export_data = json.load(f)
                for variant in ["base", "sft", "dpo", "grpo"]:
                    summary = export_data.get("model_results", {}).get(variant, {}).get("summary")
                    if summary:
                        size_results["techniques"].setdefault(variant, {})
                        size_results["techniques"][variant]["eval_accuracy"] = summary.get("accuracy")
                        size_results["techniques"][variant]["eval_correct"] = summary.get("correct")
                        size_results["techniques"][variant]["eval_total"] = summary.get("total")
                break

        if size_results["techniques"]:
            results[size] = size_results

    return results


def build_prompt(results):
    """Build the structured prompt for Claude."""
    lines = [
        "You are an AI research advisor analyzing post-training experiments on storage I/O workload classification.",
        "The task is classifying storage I/O patterns into 6 categories: OLTP Database, OLAP Analytics, AI ML Training, Video Streaming, VDI Virtual Desktop, Backup Archive.",
        "",
        "Here are the current experiment results:",
        "",
    ]

    for size, data in results.items():
        lines.append(f"## Model: SmolLM2-{size} ({data['model']})")
        for technique, info in sorted(data["techniques"].items()):
            acc = info.get("eval_accuracy")
            acc_str = f"{acc:.0%}" if acc is not None else "not evaluated"
            correct = info.get("eval_correct", "?")
            total = info.get("eval_total", "?")
            time_s = info.get("training_time_seconds")
            time_str = f", training time: {time_s/60:.0f}m" if time_s else ""
            loss = info.get("final_loss") or info.get("final_step_loss")
            loss_str = f", final loss: {loss:.4f}" if loss else ""
            lines.append(f"- {technique.upper()}: {acc_str} ({correct}/{total}){time_str}{loss_str}")
        lines.append("")

    lines.extend([
        "Available compute: Google Colab T4 (16GB VRAM, free), Colab A100 (40GB, $0/Colab Pro), AWS ml.g5.xlarge ($1.20/hr)",
        "",
        "Based on these results, what should I try next to improve accuracy?",
        "",
        "Respond with EXACTLY this JSON structure:",
        '```json',
        '{',
        '  "analysis": "2-3 sentence analysis of current results",',
        '  "recommendation": "1-2 sentence recommendation",',
        '  "experiment": {',
        '    "description": "What to run",',
        '    "model_size": "360M or 1.7B",',
        '    "technique": "sft, dpo, or grpo",',
        '    "key_changes": ["list of specific changes"],',
        '    "estimated_time_minutes": 30,',
        '    "estimated_cost": "$0 (Colab T4)"',
        '  },',
        '  "rationale": "Why this experiment, referencing the data"',
        '}',
        '```',
    ])

    return "\n".join(lines)


def call_strategist(prompt_text):
    """Call Claude API with the structured prompt."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("\n[ERROR] anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n[ERROR] ANTHROPIC_API_KEY environment variable not set.")
        print("  Get your key at: https://console.anthropic.com/")
        print("  Then: export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt_text}],
    )

    response_text = message.content[0].text

    # Extract JSON from response
    import re
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))

    # Try parsing the whole response as JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return {"raw_response": response_text, "analysis": response_text[:500]}


def display_recommendation(recommendation, results):
    """Display the recommendation with Rich formatting."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
        console = Console()
    except ImportError:
        # Fallback to plain text
        print("\n" + "=" * 60)
        print("  AI RESEARCH ADVISOR")
        print("=" * 60)
        print(f"\n  Analysis: {recommendation.get('analysis', 'N/A')}")
        print(f"\n  Recommendation: {recommendation.get('recommendation', 'N/A')}")
        exp = recommendation.get("experiment", {})
        print(f"\n  Proposed experiment:")
        print(f"    {exp.get('description', 'N/A')}")
        print(f"    Model: {exp.get('model_size', 'N/A')}")
        print(f"    Technique: {exp.get('technique', 'N/A')}")
        print(f"    Est. time: {exp.get('estimated_time_minutes', 'N/A')} min")
        print(f"    Est. cost: {exp.get('estimated_cost', 'N/A')}")
        print(f"\n  Rationale: {recommendation.get('rationale', 'N/A')}")
        return

    # Current best accuracy
    best_acc = 0
    best_label = "none"
    for size, data in results.items():
        for tech, info in data["techniques"].items():
            acc = info.get("eval_accuracy", 0) or 0
            if acc > best_acc:
                best_acc = acc
                best_label = f"SmolLM2-{size} + {tech.upper()}"

    console.print()
    console.print(Panel(
        f"[bold cyan]AI RESEARCH ADVISOR[/bold cyan] — Storage I/O Classification",
        subtitle=f"Current best: {best_label} at {best_acc:.0%} accuracy",
        box=box.DOUBLE,
        border_style="cyan",
    ))

    # Analysis
    console.print(f"\n[bold]Analysis:[/bold] {recommendation.get('analysis', 'N/A')}")

    # Recommendation
    console.print(f"\n[bold green]Recommendation:[/bold green] {recommendation.get('recommendation', 'N/A')}")

    # Experiment details
    exp = recommendation.get("experiment", {})
    if exp:
        table = Table(title="Proposed Experiment", box=box.SIMPLE, title_style="bold cyan")
        table.add_column("Parameter", style="bold")
        table.add_column("Value")
        table.add_row("Description", exp.get("description", "N/A"))
        table.add_row("Model", exp.get("model_size", "N/A"))
        table.add_row("Technique", exp.get("technique", "N/A"))
        if exp.get("key_changes"):
            table.add_row("Key Changes", "\n".join(f"• {c}" for c in exp["key_changes"]))
        table.add_row("Est. Time", f"{exp.get('estimated_time_minutes', '?')} min")
        table.add_row("Est. Cost", exp.get("estimated_cost", "N/A"))
        console.print(table)

    # Rationale
    console.print(f"\n[dim]Rationale: {recommendation.get('rationale', 'N/A')}[/dim]")


def get_approval():
    """Ask the user whether to run the proposed experiment."""
    print()
    response = input("Run this experiment? [y/N/skip] ").strip().lower()
    return response in ("y", "yes")


def run_experiment(experiment):
    """Run the proposed experiment as a subprocess."""
    technique = experiment.get("technique", "").lower()
    model_size = experiment.get("model_size", "360M")

    script_map = {
        "sft": "train_sft.py",
        "dpo": "train_dpo.py",
        "grpo": "train_grpo.py",
    }

    script = script_map.get(technique)
    if not script:
        print(f"[WARNING] Unknown technique: {technique}. Cannot run automatically.")
        return

    cmd = [sys.executable, str(SCRIPT_DIR / script), "--model-size", model_size]
    print(f"\n[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="AI Research Advisor (AINOS Lite)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the prompt that would be sent to Claude without calling the API")
    args = parser.parse_args()

    print("=" * 60)
    print("  AI Research Advisor (AINOS Lite)")
    print("=" * 60)

    # 1. Gather all results
    print("\n[1/3] Scanning experiment results...")
    results = collect_all_results()

    if not results:
        print("\n[WARNING] No training results found in scripts/outputs/")
        print("  Run at least train_sft.py first, then export_artifacts.py")
        sys.exit(1)

    for size, data in results.items():
        print(f"\n  SmolLM2-{size}:")
        for tech, info in sorted(data["techniques"].items()):
            acc = info.get("eval_accuracy")
            acc_str = f"{acc:.0%}" if acc is not None else "N/A"
            print(f"    {tech.upper():>6s}: accuracy={acc_str}")

    # 2. Build prompt and call Claude (or dry-run)
    prompt_text = build_prompt(results)

    if args.dry_run:
        print("\n[DRY RUN] Would send this prompt to Claude:")
        print("-" * 40)
        print(prompt_text)
        print("-" * 40)
        print(f"\nPrompt length: {len(prompt_text)} characters")
        return

    print("\n[2/3] Consulting Claude...")
    recommendation = call_strategist(prompt_text)

    # 3. Display recommendation
    print("\n[3/3] Advisor recommendation:")
    display_recommendation(recommendation, results)

    # 4. Optional: approve and run
    experiment = recommendation.get("experiment")
    if experiment:
        if get_approval():
            run_experiment(experiment)
        else:
            print("\nSkipped. You can run the experiment manually later.")


if __name__ == "__main__":
    main()
