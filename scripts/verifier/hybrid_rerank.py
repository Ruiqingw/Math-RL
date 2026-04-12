#!/usr/bin/env python3
"""
Hybrid reranker sweeps for a best-of-N jsonl produced by eval_prm_best_of_n.py
(min aggregation variant).

Methods reported:
    - baseline_hybrid : majority bucket, PRM tiebreak inside bucket (identical
      accuracy to plain majority by construction, included for sanity).
    - filter_majority : drop bottom-k by PRM score, then majority vote on the
      survivors. Swept over several k values.
    - weighted_majority : each candidate's vote is weighted by its PRM score
      (score is assumed non-negative; negatives are clipped to 0).

Usage:
    python scripts/verifier/hybrid_rerank.py \
        --jsonl /root/autodl-tmp/prm_grpo/outputs/prm_best_of_n/math_test_100_best_of_16_tokenprm_min.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from statistics import mean
from typing import Any, Callable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Majority+PRM hybrid reranker evaluator.")
    parser.add_argument("--jsonl", required=True, help="Path produced by eval_prm_best_of_n.py.")
    parser.add_argument(
        "--filter-ks",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8],
        help="List of k values for filter_majority (drop bottom-k PRM scores).",
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
    return as_float(sample.get("prm_score"), 0.0)


def answer_of(sample: dict[str, Any]) -> str:
    return str(sample.get("boxed_answer", "")).strip()


def majority_answer(samples: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for sample in samples:
        answer = answer_of(sample)
        if answer:
            counts[answer] += 1
    if not counts:
        return ""
    best_count = max(counts.values())
    for answer in (answer_of(s) for s in samples):
        if answer and counts[answer] == best_count:
            return answer
    return ""


def first_correct_with_answer(samples: list[dict[str, Any]], answer: str) -> int:
    for idx, sample in enumerate(samples):
        if answer_of(sample) == answer:
            return idx
    return -1


def select_baseline_hybrid(sampled: list[dict[str, Any]]) -> int:
    """Majority bucket, then PRM argmax inside that bucket."""
    if not sampled:
        return -1
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(sampled):
        answer = answer_of(sample)
        if answer:
            buckets[answer].append(idx)
    if not buckets:
        return max(range(len(sampled)), key=lambda i: score_of(sampled[i]))
    max_size = max(len(indices) for indices in buckets.values())
    top = [ans for ans, idx_list in buckets.items() if len(idx_list) == max_size]
    if len(top) == 1:
        winning = top[0]
    else:
        winning = max(top, key=lambda ans: sum(score_of(sampled[i]) for i in buckets[ans]))
    return max(buckets[winning], key=lambda i: score_of(sampled[i]))


def select_filter_majority(sampled: list[dict[str, Any]], k: int) -> int:
    """Drop bottom-k by PRM score, then majority vote on the rest."""
    if not sampled:
        return -1
    if k <= 0 or k >= len(sampled):
        survivors_idx = list(range(len(sampled)))
    else:
        ranked = sorted(range(len(sampled)), key=lambda i: score_of(sampled[i]), reverse=True)
        survivors_idx = ranked[: len(sampled) - k]

    survivors = [sampled[i] for i in survivors_idx]
    winning_answer = majority_answer(survivors)
    if not winning_answer:
        return max(survivors_idx, key=lambda i: score_of(sampled[i]))

    candidates = [i for i in survivors_idx if answer_of(sampled[i]) == winning_answer]
    if not candidates:
        return survivors_idx[0]
    return max(candidates, key=lambda i: score_of(sampled[i]))


def select_weighted_majority(sampled: list[dict[str, Any]]) -> int:
    """Each vote is weighted by the candidate's PRM score (non-negative)."""
    if not sampled:
        return -1
    weights: dict[str, float] = defaultdict(float)
    members: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(sampled):
        answer = answer_of(sample)
        if not answer:
            continue
        weight = max(score_of(sample), 0.0)
        weights[answer] += weight
        members[answer].append(idx)
    if not weights:
        return max(range(len(sampled)), key=lambda i: score_of(sampled[i]))
    winning_answer = max(weights.items(), key=lambda kv: kv[1])[0]
    return max(members[winning_answer], key=lambda i: score_of(sampled[i]))


def evaluate_method(
    records: list[dict[str, Any]],
    selector: Callable[[list[dict[str, Any]]], int],
) -> tuple[float, int, int]:
    correct_flags: list[float] = []
    differs_from_majority = 0
    differs_from_prm_min = 0
    for row in records:
        sampled = row.get("sampled", [])
        if not sampled:
            continue
        idx = selector(sampled)
        if idx < 0:
            continue
        correct_flags.append(float(is_correct(sampled[idx])))

        row_majority_answer = str(row.get("majority_answer", "")).strip()
        if row_majority_answer and answer_of(sampled[idx]) != row_majority_answer:
            differs_from_majority += 1

        prm_min_idx = int(row.get("prm_best_index", -1))
        if 0 <= prm_min_idx < len(sampled) and prm_min_idx != idx:
            differs_from_prm_min += 1

    accuracy = mean(correct_flags) if correct_flags else float("nan")
    return accuracy, differs_from_majority, differs_from_prm_min


def baseline_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    greedy = [as_float(r.get("greedy_correct")) for r in records if r.get("sampled")]
    majority = [as_float(r.get("majority_correct")) for r in records if r.get("sampled")]
    prm_min = [as_float(r.get("prm_best_correct")) for r in records if r.get("sampled")]
    oracle = [as_float(r.get("sample_oracle_correct")) for r in records if r.get("sampled")]
    return {
        "greedy_accuracy": mean(greedy) if greedy else float("nan"),
        "majority_accuracy": mean(majority) if majority else float("nan"),
        "prm_min_accuracy": mean(prm_min) if prm_min else float("nan"),
        "oracle_accuracy": mean(oracle) if oracle else float("nan"),
    }


def diagnostic_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    bucket_counts: list[int] = []
    bucket_sizes: list[int] = []
    single_bucket_rows = 0
    for row in records:
        sampled = row.get("sampled", [])
        if not sampled:
            continue
        counts: dict[str, int] = defaultdict(int)
        for s in sampled:
            ans = answer_of(s)
            if ans:
                counts[ans] += 1
        bucket_counts.append(len(counts))
        if counts:
            bucket_sizes.append(max(counts.values()))
            if len(counts) == 1:
                single_bucket_rows += 1
    return {
        "avg_bucket_count": mean(bucket_counts) if bucket_counts else float("nan"),
        "avg_max_bucket_size": mean(bucket_sizes) if bucket_sizes else float("nan"),
        "single_bucket_rows": float(single_bucket_rows),
    }


def main() -> None:
    args = parse_args()
    records = load_records(args.jsonl)
    n_rows = sum(1 for r in records if r.get("sampled"))

    base = baseline_stats(records)
    diag = diagnostic_stats(records)

    print(f"jsonl                       = {args.jsonl}")
    print(f"num_examples                = {n_rows}")
    print()
    print("Reference baselines:")
    print(f"  greedy                    = {base['greedy_accuracy']:.4f}")
    print(f"  prm_min (in-jsonl)        = {base['prm_min_accuracy']:.4f}")
    print(f"  majority                  = {base['majority_accuracy']:.4f}")
    print(f"  oracle                    = {base['oracle_accuracy']:.4f}")
    print()
    print("Hybrid methods:")

    hybrid_acc, hybrid_dm, hybrid_dp = evaluate_method(records, select_baseline_hybrid)
    print(
        f"  baseline_hybrid           = {hybrid_acc:.4f}  "
        f"(diff_vs_majority={hybrid_dm} rows, diff_vs_prm_min={hybrid_dp} rows)"
    )

    for k in args.filter_ks:
        acc, dm, dp = evaluate_method(records, lambda s, _k=k: select_filter_majority(s, _k))
        print(
            f"  filter_majority k={k:<2d}      = {acc:.4f}  "
            f"(diff_vs_majority={dm} rows, diff_vs_prm_min={dp} rows)"
        )

    wacc, wdm, wdp = evaluate_method(records, select_weighted_majority)
    print(
        f"  weighted_majority         = {wacc:.4f}  "
        f"(diff_vs_majority={wdm} rows, diff_vs_prm_min={wdp} rows)"
    )

    print()
    print("Diagnostics:")
    print(f"  avg_bucket_count_per_row  = {diag['avg_bucket_count']:.2f}")
    print(f"  avg_max_bucket_size       = {diag['avg_max_bucket_size']:.2f}")
    print(f"  single_bucket_rows        = {int(diag['single_bucket_rows'])}")


if __name__ == "__main__":
    main()
