#!/usr/bin/env python3
"""
generate_dataset.py - Synthetic Storage I/O Workload Classification Dataset Generator

Generates ~2000 realistic storage workload entries across six categories,
split into train / validation / test sets with Gaussian-distributed features
and deliberate edge cases that blur category boundaries.

Output directory: ../app/public/data/
"""

import json
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "app" / "public" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Workload profile definitions
# Each tuple: (min, max) -- Gaussian center = midpoint, sigma ~ (max-min)/6
# so ~99.7 % of samples land inside the range before clamping.
# ---------------------------------------------------------------------------
PROFILES = {
    "OLTP Database": {
        "iops":           (30000, 80000),
        "latency_ms":     (0.1, 0.8),
        "block_size_kb":  (4, 8),
        "read_pct":       (60, 80),
        "sequential_pct": (10, 25),
        "queue_depth":    (16, 64),
    },
    "OLAP Analytics": {
        "iops":           (5000, 15000),
        "latency_ms":     (2.0, 8.0),
        "block_size_kb":  (64, 256),
        "read_pct":       (85, 95),
        "sequential_pct": (70, 90),
        "queue_depth":    (4, 16),
    },
    "AI ML Training": {
        "iops":           (10000, 30000),
        "latency_ms":     (0.5, 3.0),
        "block_size_kb":  (128, 1024),
        "read_pct":       (90, 99),
        "sequential_pct": (85, 95),
        "queue_depth":    (32, 128),
    },
    "Video Streaming": {
        "iops":           (1000, 5000),
        "latency_ms":     (5.0, 15.0),
        "block_size_kb":  (256, 1024),
        "read_pct":       (95, 99),
        "sequential_pct": (92, 99),
        "queue_depth":    (1, 8),
    },
    "VDI Virtual Desktop": {
        "iops":           (20000, 60000),
        "latency_ms":     (0.2, 0.9),
        "block_size_kb":  (4, 16),
        "read_pct":       (50, 70),
        "sequential_pct": (15, 30),
        "queue_depth":    (32, 128),
    },
    "Backup Archive": {
        "iops":           (2000, 8000),
        "latency_ms":     (10.0, 50.0),
        "block_size_kb":  (256, 4096),
        "read_pct":       (5, 15),
        "sequential_pct": (88, 98),
        "queue_depth":    (1, 4),
    },
}

# Preferred block sizes to snap to (powers of two are realistic)
BLOCK_SNAP_VALUES = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Brief explanation templates per category
EXPLANATIONS = {
    "OLTP Database": (
        "High IOPS with very low latency and small block sizes indicate a "
        "transactional database workload with mixed random read/write I/O."
    ),
    "OLAP Analytics": (
        "Moderate IOPS with larger block sizes and predominantly sequential "
        "reads are characteristic of analytical query processing."
    ),
    "AI ML Training": (
        "High-throughput sequential reads with large block sizes and deep "
        "queue depths are typical of machine-learning training data pipelines."
    ),
    "Video Streaming": (
        "Low IOPS with large sequential reads, high read ratio, and shallow "
        "queue depth match a media streaming workload pattern."
    ),
    "VDI Virtual Desktop": (
        "High IOPS with small blocks, low latency, and a balanced read/write "
        "mix are hallmarks of virtual desktop infrastructure."
    ),
    "Backup Archive": (
        "Low IOPS with very large block sizes, high write ratio, and elevated "
        "latency indicate a backup or archival storage workload."
    ),
}


# ---------------------------------------------------------------------------
# Helper: Gaussian sample clamped to [lo, hi], centered at midpoint
# ---------------------------------------------------------------------------
def _gauss(lo: float, hi: float, n: int, *, sigma_factor: float = 6.0) -> np.ndarray:
    mu = (lo + hi) / 2.0
    sigma = (hi - lo) / sigma_factor
    samples = rng.normal(mu, sigma, size=n)
    return np.clip(samples, lo, hi)


def _snap_block(value: float, lo: int, hi: int) -> int:
    """Snap a continuous block-size sample to the nearest power-of-two value
    that falls within the allowed range."""
    candidates = [b for b in BLOCK_SNAP_VALUES if lo <= b <= hi]
    if not candidates:
        # Fallback: just round to nearest candidate overall
        candidates = BLOCK_SNAP_VALUES
    return int(min(candidates, key=lambda c: abs(c - value)))


# ---------------------------------------------------------------------------
# Generate entries for one workload category
# ---------------------------------------------------------------------------
def generate_category(label: str, n: int, *, edge_fraction: float = 0.10) -> list[dict]:
    profile = PROFILES[label]
    entries: list[dict] = []

    n_normal = int(n * (1.0 - edge_fraction))
    n_edge = n - n_normal

    for batch, count, sigma_f in [("normal", n_normal, 6.0), ("edge", n_edge, 3.0)]:
        iops_raw        = _gauss(*profile["iops"], count, sigma_factor=sigma_f)
        latency_raw     = _gauss(*profile["latency_ms"], count, sigma_factor=sigma_f)
        block_raw       = _gauss(*profile["block_size_kb"], count, sigma_factor=sigma_f)
        read_pct_raw    = _gauss(*profile["read_pct"], count, sigma_factor=sigma_f)
        seq_pct_raw     = _gauss(*profile["sequential_pct"], count, sigma_factor=sigma_f)
        qdepth_raw      = _gauss(*profile["queue_depth"], count, sigma_factor=sigma_f)

        # For edge cases, intentionally push some features slightly outside
        # the "comfortable" zone so the model must learn nuance.
        if batch == "edge":
            # Shift latency up by 0-30 % of range
            lat_range = profile["latency_ms"][1] - profile["latency_ms"][0]
            latency_raw += rng.uniform(0, 0.30 * lat_range, size=count)

            # Shift IOPS by +/- 10 % of range
            iops_range = profile["iops"][1] - profile["iops"][0]
            iops_raw += rng.uniform(-0.10 * iops_range, 0.10 * iops_range, size=count)

            # Nudge sequential_pct +/- 5 points
            seq_pct_raw += rng.uniform(-5, 5, size=count)

        for i in range(count):
            iops = int(round(np.clip(iops_raw[i], 100, 200000)))
            latency = round(float(np.clip(latency_raw[i], 0.01, 200.0)), 2)
            block = _snap_block(block_raw[i], *profile["block_size_kb"])
            read_pct = int(round(np.clip(read_pct_raw[i], 1, 99)))
            write_pct = 100 - read_pct
            seq_pct = int(round(np.clip(seq_pct_raw[i], 0, 100)))
            qdepth = int(round(np.clip(qdepth_raw[i], 1, 256)))

            rw_str = f"{read_pct}/{write_pct}"

            formatted_input = (
                f"IOPS: {iops} | Latency: {latency}ms | Block Size: {block}K | "
                f"Read/Write: {rw_str} | Sequential: {seq_pct}% | Queue Depth: {qdepth}"
            )
            formatted_output = f"Classification: {label} - {EXPLANATIONS[label]}"

            entries.append({
                "iops": iops,
                "latency_ms": latency,
                "block_size_kb": block,
                "read_pct": read_pct,
                "write_pct": write_pct,
                "sequential_pct": seq_pct,
                "queue_depth": qdepth,
                "label": label,
                "formatted_input": formatted_input,
                "formatted_output": formatted_output,
            })

    return entries


# ---------------------------------------------------------------------------
# Compute per-category statistics
# ---------------------------------------------------------------------------
def compute_stats(entries: list[dict]) -> dict:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_label[e["label"]].append(e)

    stats: dict = {}
    numeric_keys = ["iops", "latency_ms", "block_size_kb", "read_pct",
                    "write_pct", "sequential_pct", "queue_depth"]

    for label, items in sorted(by_label.items()):
        cat_stats: dict = {"count": len(items)}
        for key in numeric_keys:
            vals = np.array([item[key] for item in items], dtype=float)
            cat_stats[key] = {
                "min": round(float(vals.min()), 2),
                "max": round(float(vals.max()), 2),
                "mean": round(float(vals.mean()), 2),
                "std": round(float(vals.std()), 2),
                "median": round(float(np.median(vals)), 2),
            }
        stats[label] = cat_stats

    stats["_total"] = {"count": len(entries)}
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Target counts per category (total ~2000)
    # Slightly more OLTP & VDI to reflect real-world prevalence
    CATEGORY_COUNTS = {
        "OLTP Database":       380,
        "OLAP Analytics":      310,
        "AI ML Training":      330,
        "Video Streaming":     310,
        "VDI Virtual Desktop": 360,
        "Backup Archive":      310,
    }
    total_target = sum(CATEGORY_COUNTS.values())  # 2000

    # Generate all entries
    all_entries: list[dict] = []
    for label, count in CATEGORY_COUNTS.items():
        all_entries.extend(generate_category(label, count, edge_fraction=0.10))

    # Shuffle deterministically
    indices = rng.permutation(len(all_entries))
    all_entries = [all_entries[i] for i in indices]

    # Split: 1400 train / 300 val / 300 test
    train = all_entries[:1400]
    val   = all_entries[1400:1700]
    test  = all_entries[1700:]

    # Write datasets
    for name, data in [("dataset_train.json", train),
                       ("dataset_val.json", val),
                       ("dataset_test.json", test)]:
        path = OUTPUT_DIR / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  Wrote {path}  ({len(data)} entries)")

    # Write stats
    all_stats = compute_stats(all_entries)
    all_stats["_splits"] = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"  Wrote {stats_path}")

    # --------------- Summary ---------------
    print("\n" + "=" * 68)
    print("  DATASET GENERATION SUMMARY")
    print("=" * 68)
    print(f"  Total entries : {len(all_entries)}")
    print(f"  Train / Val / Test : {len(train)} / {len(val)} / {len(test)}")
    print("-" * 68)

    for label in sorted(CATEGORY_COUNTS.keys()):
        s = all_stats[label]
        print(f"\n  [{label}]  count={s['count']}")
        for feat in ["iops", "latency_ms", "block_size_kb", "read_pct",
                      "sequential_pct", "queue_depth"]:
            fs = s[feat]
            print(f"    {feat:18s}  mean={fs['mean']:>10.2f}  "
                  f"std={fs['std']:>8.2f}  "
                  f"range=[{fs['min']}, {fs['max']}]")

    print("\n" + "=" * 68)
    print("  Done. Files written to:", OUTPUT_DIR)
    print("=" * 68)


if __name__ == "__main__":
    main()
