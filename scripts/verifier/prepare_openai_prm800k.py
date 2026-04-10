#!/usr/bin/env python3
"""
Materialize local stepwise datasets from raw OpenAI PRM800K JSONL files.
"""

from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from openai_prm_raw import (
    DEFAULT_RAW_DATA_DIR,
    build_raw_phase1_phase2_dataset,
    build_raw_phase2_dataset,
    phase1_phase2_cache_dir,
    phase2_cache_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare raw OpenAI PRM800K for token-PRM training.")
    parser.add_argument(
        "--dataset-source",
        default="raw_phase2",
        choices=["raw_phase2", "raw_phase1_phase2"],
        help="Which raw PRM800K view to materialize.",
    )
    parser.add_argument("--raw-data-dir", default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument(
        "--neutral-policy",
        default="nonnegative",
        choices=["nonnegative", "positive_only"],
        help="How to map rating=0 when converting {-1,0,+1} to binary labels.",
    )
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="Disable first-error-only truncation when building the dataset view.",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_source == "raw_phase2":
        cache_dir = args.cache_dir or phase2_cache_dir(
            neutral_policy=args.neutral_policy,
            stop_at_first_negative=not args.all_steps,
        )
        dataset = build_raw_phase2_dataset(
            raw_data_dir=args.raw_data_dir,
            cache_dir=cache_dir,
            force_rebuild=args.force_rebuild,
            neutral_policy=args.neutral_policy,
            stop_at_first_negative=not args.all_steps,
        )
    elif args.dataset_source == "raw_phase1_phase2":
        cache_dir = args.cache_dir or phase1_phase2_cache_dir(
            neutral_policy=args.neutral_policy,
            stop_at_first_negative=not args.all_steps,
        )
        dataset = build_raw_phase1_phase2_dataset(
            raw_data_dir=args.raw_data_dir,
            cache_dir=cache_dir,
            force_rebuild=args.force_rebuild,
            neutral_policy=args.neutral_policy,
            stop_at_first_negative=not args.all_steps,
        )
    else:
        raise ValueError(f"Unsupported dataset source: {args.dataset_source}")

    print(f"Saved processed dataset to: {cache_dir}")
    for split in ("train", "test"):
        hf_split = dataset[split]
        n_rows = len(hf_split)
        n_steps = sum(len(row["labels"]) for row in hf_split)
        n_neg = sum(sum(1 for label in row["labels"] if not label) for row in hf_split)
        n_pos = n_steps - n_neg
        n_neg_rows = sum(
            1
            for row in hf_split
            if len(row["labels"]) > 0 and (not bool(row["labels"][-1]))
        )
        print(
            f"{split}: rows={n_rows:,} steps={n_steps:,} "
            f"pos={n_pos:,} neg={n_neg:,} neg_step_frac={(n_neg / max(n_steps, 1)):.4f} "
            f"neg_row_frac={(n_neg_rows / max(n_rows, 1)):.4f} neg_rows={n_neg_rows:,}"
        )


if __name__ == "__main__":
    main()
