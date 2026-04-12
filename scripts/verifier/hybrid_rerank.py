#!/usr/bin/env python3
"""
Hybrid reranker: majority vote picks the answer bucket, token-PRM picks the
best trajectory inside that bucket.

Reads the jsonl written by eval_prm_best_of_n.py (min aggregation variant) and
reports accuracy alongside the reference baselines.

Usage:
    python scripts/verifier/hybrid_rerank.py \
        --jsonl /root/autodl-tmp/prm_grpo/outputs/prm_best_of_n/math_test_100_best_of_16_tokenprm_min.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Majority+PRM hybrid reranker evaluator.")
    parser.add_argument("--jsonl", required=True, help="Path produced by eval_prm_best_of_n.py.")
    parser.add_argument(
        "--tiebreak",
        choices=["prm_sum", "first"],
        default="prm_sum",
        help="How to break ties between equally-sized answer buckets.",
    )
    return parser.parse_args()


def load_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                records.append(json.loads(line))
    return records


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_correct(sample: dict[str, Any]) -> bool:
    return as_float(sample.get("math_correct")) > 0.0


def score_of(sample: dict[str, Any]) -> float:
    return as_float(sample.get("prm_score"), float("-inf"))


def hybrid_select(sampled: list[dict[str, Any]], tiebreak: str) -> int:
    """Return the index into `sampled` chosen by the hybrid rule."""
    if not sampled:
        return -1

    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(sampled):
        answer = str(sample.get("boxed_answer", "")).strip()
        if not answer:
            continue
        buckets[answer].append(idx)

    if not buckets:
        # No extractable answers: fall back to plain PRM argmax.
        return max(range(len(sampled)), key=lambda i: score_of(sampled[i]))

    max_size = max(len(indices) for indices in buckets.values())
    top_buckets = [answer for answer, indices in buckets.items() if len(indices) == max_size]

    if len(top_buckets) == 1:
        winning_answer = top_buckets[0]
    elif tiebreak == "prm_sum":
        winning_answer = max(
            top_buckets,
            key=lambda ans: sum(score_of(sampled[i]) for i in buckets[ans]),
        )
    else:
        winning_answer = top_buckets[0]

    winning_indices = buckets[winning_answer]
    return max(winning_indices, key=lambda i: score_of(sampled[i]))


def evaluate(records: list[dict[str, Any]], tiebreak: str) -> dict[str, Any]:
    hybrid_correct: list[float] = []
    greedy_correct: list[float] = []
    majority_correct: list[float] = []
    prm_min_correct: list[float] = []
    oracle_correct: list[float] = []

    bucket_sizes: list[int] = []
    bucket_counts: list[int] = []
    single_bucket_rows = 0
    hybrid_differs_from_prm_min = 0
    hybrid_differs_from_majority_pick = 0

    for row in records:
        sampled = row.get("sampled", [])
        if not sampled:
            continue

        hybrid_idx = hybrid_select(sampled, tiebreak)
        if hybrid_idx < 0:
            continue

        hybrid_correct.append(float(is_correct(sampled[hybrid_idx])))
        greedy_correct.append(as_float(row.get("greedy_correct")))
        majority_correct.append(as_float(row.get("majority_correct")))
        prm_min_correct.append(as_float(row.get("prm_best_correct")))
        oracle_correct.append(as_float(row.get("sample_oracle_correct")))

        buckets: dict[str, int] = defaultdict(int)
        for sample in sampled:
            answer = str(sample.get("boxed_answer", "")).strip()
            if answer:
                buckets[answer] += 1
        bucket_counts.append(len(buckets))
        if buckets:
            bucket_sizes.append(max(buckets.values()))
            if len(buckets) == 1:
                single_bucket_rows += 1

        prm_min_idx = int(row.get("prm_best_index", -1))
        if 0 <= prm_min_idx < len(sampled) and prm_min_idx != hybrid_idx:
            hybrid_differs_from_prm_min += 1

    n = len(hybrid_correct)
    return {
        "num_examples": n,
        "hybrid_accuracy": mean(hybrid_correct) if hybrid_correct else float("nan"),
        "greedy_accuracy": mean(greedy_correct) if greedy_correct else float("nan"),
        "majority_accuracy": mean(majority_correct) if majority_correct else float("nan"),
        "prm_min_accuracy": mean(prm_min_correct) if prm_min_correct else float("nan"),
        "oracle_accuracy": mean(oracle_correct) if oracle_correct else float("nan"),
        "avg_bucket_count": mean(bucket_counts) if bucket_counts else float("nan"),
        "avg_max_bucket_size": mean(bucket_sizes) if bucket_sizes else float("nan"),
        "single_bucket_rows": single_bucket_rows,
        "hybrid_differs_from_prm_min_rows": hybrid_differs_from_prm_min,
    }


def main() -> None:
    args = parse_args()
    records = load_records(args.jsonl)
    stats = evaluate(records, tiebreak=args.tiebreak)

    print(f"jsonl                       = {args.jsonl}")
    print(f"tiebreak                    = {args.tiebreak}")
    print(f"num_examples                = {stats['num_examples']}")
    print()
    print(f"greedy_accuracy             = {stats['greedy_accuracy']:.4f}")
    print(f"prm_min_accuracy            = {stats['prm_min_accuracy']:.4f}")
    print(f"majority_accuracy           = {stats['majority_accuracy']:.4f}")
    print(f"hybrid_accuracy             = {stats['hybrid_accuracy']:.4f}   <-- majority bucket + PRM tiebreak")
    print(f"oracle_accuracy             = {stats['oracle_accuracy']:.4f}")
    print()
    print(f"avg_bucket_count_per_row    = {stats['avg_bucket_count']:.2f}")
    print(f"avg_max_bucket_size         = {stats['avg_max_bucket_size']:.2f}")
    print(f"rows_with_single_bucket     = {stats['single_bucket_rows']}")
    print(f"hybrid!=prm_min rows        = {stats['hybrid_differs_from_prm_min_rows']}")


if __name__ == "__main__":
    main()
