#!/usr/bin/env python3
"""
Check exact problem overlap between a MATH eval parquet and raw PRM800K files.

PRM800K uses its own train/test problem split. If an evaluation file uses the
original full MATH test split, many problems can overlap with PRM800K training
unless the eval set is restricted to OpenAI's held-out math_splits/test subset.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

import datasets


DEFAULT_MATH_PARQUET = "/root/autodl-tmp/prm_grpo/data/trl_math/test.parquet"
DEFAULT_PRM800K_ROOT = "/root/autodl-tmp/prm_grpo/prm800k_raw/prm800k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check exact MATH/PRM800K problem overlap.")
    parser.add_argument("--math-parquet", default=DEFAULT_MATH_PARQUET, help="TRL-style MATH parquet path.")
    parser.add_argument("--prm800k-root", default=DEFAULT_PRM800K_ROOT, help="Raw PRM800K repository root.")
    parser.add_argument("--sample-limit", type=int, default=10, help="How many overlap examples to print.")
    parser.add_argument("--write-overlaps-jsonl", default="", help="Optional path to write overlap examples.")
    return parser.parse_args()


def normalize_problem(text: Any) -> str:
    return " ".join(str(text or "").split())


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                yield json.loads(line)


def raw_prm_problem(row: dict[str, Any]) -> str:
    question = row.get("question", {})
    if isinstance(question, dict):
        return normalize_problem(question.get("problem", ""))
    return normalize_problem(row.get("problem", ""))


def raw_math_split_problem(row: dict[str, Any]) -> str:
    if "problem" in row:
        return normalize_problem(row["problem"])
    question = row.get("question", {})
    if isinstance(question, dict):
        return normalize_problem(question.get("problem", ""))
    return normalize_problem(question)


def load_eval_problems(path: str) -> list[dict[str, Any]]:
    dataset = datasets.load_dataset("parquet", data_files=path, split="train")
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(dataset):
        problem = normalize_problem(row.get("problem", ""))
        rows.append(
            {
                "eval_row_idx": idx,
                "dataset_idx": row.get("index", idx),
                "problem": problem,
                "gold_answer": row.get("gold_answer", ""),
            }
        )
    return rows


def collect_prm800k_sources(root: str) -> dict[str, dict[str, list[int]]]:
    sources: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    data_dir = os.path.join(root, "data")
    for name in ("phase1_train", "phase1_test", "phase2_train", "phase2_test"):
        path = os.path.join(data_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        for idx, row in enumerate(read_jsonl(path)):
            problem = raw_prm_problem(row)
            if problem:
                sources[problem][name].append(idx)

    math_splits_dir = os.path.join(root, "math_splits")
    for name in ("train", "test"):
        path = os.path.join(math_splits_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            continue
        source_name = f"math_splits_{name}"
        for idx, row in enumerate(read_jsonl(path)):
            problem = raw_math_split_problem(row)
            if problem:
                sources[problem][source_name].append(idx)

    return sources


def source_count(overlaps: list[dict[str, Any]], source: str) -> int:
    return sum(1 for row in overlaps if source in row["sources"])


def main() -> None:
    args = parse_args()
    eval_rows = load_eval_problems(args.math_parquet)
    source_map = collect_prm800k_sources(args.prm800k_root)

    overlaps: list[dict[str, Any]] = []
    for row in eval_rows:
        sources = dict(source_map.get(row["problem"], {}))
        if sources:
            overlaps.append({**row, "sources": sources})

    total = len(eval_rows)
    print("=" * 80)
    print("MATH / PRM800K Exact Problem Overlap")
    print("=" * 80)
    print(f"math_parquet       : {args.math_parquet}")
    print(f"prm800k_root       : {args.prm800k_root}")
    print(f"eval_problem_count : {total}")
    print(f"overlap_count      : {len(overlaps)}")
    print(f"overlap_frac       : {len(overlaps) / total if total else 0.0:.4f}")

    sources = [
        "phase1_train",
        "phase1_test",
        "phase2_train",
        "phase2_test",
        "math_splits_train",
        "math_splits_test",
    ]
    for source in sources:
        print(f"{source:18}: {source_count(overlaps, source)}")

    prm_train_sources = {"phase1_train", "phase2_train", "math_splits_train"}
    prm_test_sources = {"phase1_test", "phase2_test", "math_splits_test"}
    train_overlap = sum(1 for row in overlaps if prm_train_sources.intersection(row["sources"]))
    test_overlap = sum(1 for row in overlaps if prm_test_sources.intersection(row["sources"]))
    print(f"any_prm_train_or_math_train_overlap: {train_overlap}")
    print(f"any_prm_test_or_math_test_overlap  : {test_overlap}")

    print("\nOverlap examples")
    print("-" * 80)
    for row in overlaps[: args.sample_limit]:
        print(
            f"[eval_row={row['eval_row_idx']} dataset_idx={row['dataset_idx']}] "
            f"sources={sorted(row['sources'].keys())} gold={row['gold_answer']!r}"
        )
        print(row["problem"][:500] + (" ..." if len(row["problem"]) > 500 else ""))
        print()

    if args.write_overlaps_jsonl:
        with open(args.write_overlaps_jsonl, "w", encoding="utf-8") as writer:
            for row in overlaps:
                writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved overlaps: {args.write_overlaps_jsonl}")


if __name__ == "__main__":
    main()
