#!/usr/bin/env python3
"""
Explore step-level class balance in the PRM training data.

Reports:
  - Overall pos/neg counts and fractions
  - Per-trajectory neg step distribution (how many rows have 0/1/2/... neg steps)
  - Neg step position distribution (where in the trajectory do neg steps appear)
  - Effective gradient weight ratio at various NEG_LOSS_WEIGHT settings
  - Recommended NEG_LOSS_WEIGHT range

Usage (on server):
    python scripts/verifier/explore_class_balance.py --dataset raw_phase1_phase2 --mode allsteps
    python scripts/verifier/explore_class_balance.py --dataset raw_phase1_phase2 --mode firsterr
    python scripts/verifier/explore_class_balance.py --dataset raw_phase2 --mode firsterr
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from openai_prm_raw import (
    DEFAULT_RAW_DATA_DIR,
    build_raw_phase1_phase2_dataset,
    build_raw_phase2_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["raw_phase2", "raw_phase1_phase2"],
        default="raw_phase1_phase2",
    )
    parser.add_argument(
        "--mode",
        choices=["firsterr", "allsteps"],
        default="allsteps",
    )
    parser.add_argument("--raw-data-dir", default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def analyze_split(dataset, split_name: str) -> None:
    total_rows = len(dataset)
    total_steps = 0
    total_pos = 0
    total_neg = 0

    neg_per_row: list[int] = []
    steps_per_row: list[int] = []
    neg_positions: list[float] = []  # normalized position [0, 1]
    neg_at_last_step = 0

    for row in dataset:
        labels = row["labels"]
        n_steps = len(labels)
        n_neg = sum(1 for lab in labels if not lab)
        n_pos = n_steps - n_neg

        total_steps += n_steps
        total_pos += n_pos
        total_neg += n_neg
        neg_per_row.append(n_neg)
        steps_per_row.append(n_steps)

        for i, lab in enumerate(labels):
            if not lab:
                neg_positions.append(i / max(n_steps - 1, 1))
                if i == n_steps - 1:
                    neg_at_last_step += 1

    neg_frac = total_neg / total_steps if total_steps else 0
    pos_frac = 1 - neg_frac

    print(f"\n{'='*60}")
    print(f"Split: {split_name}")
    print(f"{'='*60}")
    print(f"  rows           = {total_rows:>10,}")
    print(f"  total_steps    = {total_steps:>10,}")
    print(f"  pos_steps      = {total_pos:>10,}  ({pos_frac:.4f})")
    print(f"  neg_steps      = {total_neg:>10,}  ({neg_frac:.4f})")
    print(f"  pos:neg ratio  = {pos_frac/neg_frac:.1f}:1" if neg_frac > 0 else "  pos:neg ratio  = inf:1")

    # Per-row neg step distribution
    neg_counts = Counter(neg_per_row)
    print(f"\n  Per-row neg step distribution:")
    print(f"    rows with 0 neg steps: {neg_counts.get(0, 0):>8,}  ({neg_counts.get(0, 0)/total_rows:.4f})")
    print(f"    rows with 1 neg step:  {neg_counts.get(1, 0):>8,}  ({neg_counts.get(1, 0)/total_rows:.4f})")
    for k in sorted(neg_counts.keys()):
        if k <= 1:
            continue
        print(f"    rows with {k} neg steps: {neg_counts[k]:>8,}  ({neg_counts[k]/total_rows:.4f})")

    # Steps per row stats
    if steps_per_row:
        avg_steps = sum(steps_per_row) / len(steps_per_row)
        max_steps = max(steps_per_row)
        print(f"\n  Steps per row: avg={avg_steps:.1f}, max={max_steps}")

    # Neg position distribution
    if neg_positions:
        buckets = [0] * 10
        for p in neg_positions:
            idx = min(int(p * 10), 9)
            buckets[idx] += 1
        print(f"\n  Neg step position distribution (0=start, 1=end):")
        for i, count in enumerate(buckets):
            bar = "#" * int(count / max(buckets) * 40)
            print(f"    [{i*10:>3d}%-{(i+1)*10:>3d}%]: {count:>8,}  {bar}")
        print(f"    neg at last step: {neg_at_last_step:>8,}  ({neg_at_last_step/total_neg:.4f})" if total_neg else "")

    # Effective gradient weight analysis
    if neg_frac > 0:
        print(f"\n  NEG_LOSS_WEIGHT analysis (without focal):")
        print(f"    {'negw':>6s}  {'neg_grad_share':>14s}  {'effective_ratio':>15s}  note")
        for negw in [1, 2, 3, 5, 8, 10, 12, 15]:
            neg_share = negw * neg_frac / (pos_frac + negw * neg_frac)
            eff_ratio = neg_share / (1 - neg_share) if neg_share < 1 else float("inf")
            note = ""
            if 0.25 <= neg_share <= 0.40:
                note = "<-- sweet spot"
            elif neg_share > 0.5:
                note = "<-- neg-dominated"
            print(f"    {negw:>6d}  {neg_share:>14.4f}  {eff_ratio:>15.2f}:1  {note}")
        print(f"\n  Target: neg_grad_share ≈ 0.30-0.35 for balanced learning")
        target_negw = 0.30 * pos_frac / (neg_frac * (1 - 0.30))
        print(f"  → Recommended negw ≈ {target_negw:.1f} (for 30% neg gradient share)")
        target_negw_35 = 0.35 * pos_frac / (neg_frac * (1 - 0.35))
        print(f"  → Recommended negw ≈ {target_negw_35:.1f} (for 35% neg gradient share)")


def main() -> None:
    args = parse_args()
    stop_at_first = args.mode == "firsterr"

    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode} (stop_at_first_negative={stop_at_first})")
    print(f"Raw data dir: {args.raw_data_dir}")

    if args.dataset == "raw_phase2":
        ds = build_raw_phase2_dataset(
            raw_data_dir=args.raw_data_dir,
            neutral_policy="nonnegative",
            stop_at_first_negative=stop_at_first,
        )
    else:
        ds = build_raw_phase1_phase2_dataset(
            raw_data_dir=args.raw_data_dir,
            neutral_policy="nonnegative",
            stop_at_first_negative=stop_at_first,
        )

    analyze_split(ds[args.split], args.split)


if __name__ == "__main__":
    main()
