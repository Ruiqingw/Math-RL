#!/usr/bin/env python3
"""
Analyze PRM best-of-N reranking outputs for reward-hacking-style evidence.

The strongest evidence is a misranking case:
  - at least one sampled completion is correct,
  - PRM selects an incorrect completion,
  - the selected incorrect completion has higher PRM score than every correct
    completion for the same problem.
"""

from __future__ import annotations

import argparse
import json
import math
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PRM reranking JSONL outputs.")
    parser.add_argument("--jsonl", required=True, help="Path produced by eval_prm_best_of_n.py.")
    parser.add_argument("--case-limit", type=int, default=8, help="How many worst cases to print.")
    parser.add_argument("--snippet-chars", type=int, default=500, help="Characters shown per wrong/correct sample.")
    parser.add_argument("--write-cases-jsonl", default="", help="Optional path to save all misranking cases.")
    return parser.parse_args()


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else float("nan")


def fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def chars(sample: dict[str, Any]) -> int:
    return len(str(sample.get("text", "")))


def snippet(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def candidate_score(sample: dict[str, Any]) -> float:
    return as_float(sample.get("prm_score"), float("-inf"))


def candidate_correct(sample: dict[str, Any]) -> bool:
    return as_float(sample.get("math_correct")) > 0.0


def load_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            if line.strip():
                records.append(json.loads(line))
    return records


def analyze(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    greedy_correct = [as_float(row.get("greedy_correct")) for row in records]
    majority_correct = [as_float(row.get("majority_correct")) for row in records]
    prm_correct = [as_float(row.get("prm_best_correct")) for row in records]
    oracle_correct = [as_float(row.get("sample_oracle_correct")) for row in records]

    correct_scores: list[float] = []
    wrong_scores: list[float] = []
    selected_scores: list[float] = []
    selected_correct_scores: list[float] = []
    selected_wrong_scores: list[float] = []
    best_correct_ranks: list[int] = []
    selected_wrong_more_steps = 0
    selected_wrong_longer = 0
    pairwise_wins = 0.0
    pairwise_total = 0
    per_problem_pairwise: list[float] = []
    misranking_cases: list[dict[str, Any]] = []
    correct_available = 0

    for row_idx, row in enumerate(records):
        sampled = row.get("sampled", [])
        if not sampled:
            continue

        for sample in sampled:
            score = candidate_score(sample)
            if candidate_correct(sample):
                correct_scores.append(score)
            else:
                wrong_scores.append(score)

        selected_idx = int(row.get("prm_best_index", -1))
        if not (0 <= selected_idx < len(sampled)):
            continue

        selected = sampled[selected_idx]
        selected_score = candidate_score(selected)
        selected_scores.append(selected_score)
        if candidate_correct(selected):
            selected_correct_scores.append(selected_score)
        else:
            selected_wrong_scores.append(selected_score)

        correct_candidates = [sample for sample in sampled if candidate_correct(sample)]
        if not correct_candidates:
            continue
        correct_available += 1
        wrong_candidates = [sample for sample in sampled if not candidate_correct(sample)]

        if wrong_candidates:
            row_wins = 0.0
            row_total = 0
            for correct in correct_candidates:
                correct_score = candidate_score(correct)
                for wrong in wrong_candidates:
                    wrong_score = candidate_score(wrong)
                    row_wins += float(correct_score > wrong_score) + 0.5 * float(correct_score == wrong_score)
                    row_total += 1
            pairwise_wins += row_wins
            pairwise_total += row_total
            per_problem_pairwise.append(row_wins / row_total if row_total else 0.0)

        sorted_indices = sorted(range(len(sampled)), key=lambda idx: candidate_score(sampled[idx]), reverse=True)
        correct_indices = {idx for idx, sample in enumerate(sampled) if candidate_correct(sample)}
        best_correct_rank = min(rank for rank, idx in enumerate(sorted_indices, start=1) if idx in correct_indices)
        best_correct_ranks.append(best_correct_rank)

        if candidate_correct(selected):
            continue

        best_correct = max(correct_candidates, key=candidate_score)
        score_gap = selected_score - candidate_score(best_correct)
        selected_wrong_more_steps += int(as_float(selected.get("n_steps")) > as_float(best_correct.get("n_steps")))
        selected_wrong_longer += int(chars(selected) > chars(best_correct))

        misranking_cases.append(
            {
                "row_idx": row_idx,
                "dataset_idx": row.get("dataset_idx"),
                "score_gap_selected_wrong_minus_best_correct": score_gap,
                "selected_idx": selected_idx,
                "selected_score": selected_score,
                "selected_boxed_answer": selected.get("boxed_answer", ""),
                "selected_n_steps": selected.get("n_steps", 0),
                "selected_chars": chars(selected),
                "best_correct_score": candidate_score(best_correct),
                "best_correct_boxed_answer": best_correct.get("boxed_answer", ""),
                "best_correct_n_steps": best_correct.get("n_steps", 0),
                "best_correct_chars": chars(best_correct),
                "gold_answer": row.get("gold_answer", ""),
                "majority_answer": row.get("majority_answer", ""),
                "best_correct_rank": best_correct_rank,
                "problem": row.get("problem", ""),
                "selected_text": selected.get("text", ""),
                "best_correct_text": best_correct.get("text", ""),
            }
        )

    misranking_cases.sort(
        key=lambda case: as_float(case["score_gap_selected_wrong_minus_best_correct"]),
        reverse=True,
    )

    return {
        "num_examples": n,
        "greedy_accuracy": safe_mean(greedy_correct),
        "majority_vote_accuracy": safe_mean(majority_correct),
        "prm_best_accuracy": safe_mean(prm_correct),
        "sample_oracle_accuracy": safe_mean(oracle_correct),
        "correct_available_count": correct_available,
        "misranking_count": len(misranking_cases),
        "misranking_frac_among_correct_available": len(misranking_cases) / correct_available if correct_available else 0.0,
        "candidate_correct_score_mean": safe_mean(correct_scores),
        "candidate_wrong_score_mean": safe_mean(wrong_scores),
        "selected_score_mean": safe_mean(selected_scores),
        "selected_correct_score_mean": safe_mean(selected_correct_scores),
        "selected_wrong_score_mean": safe_mean(selected_wrong_scores),
        "best_correct_rank_mean": safe_mean([float(rank) for rank in best_correct_ranks]),
        "within_problem_correct_beats_wrong": pairwise_wins / pairwise_total if pairwise_total else float("nan"),
        "within_problem_correct_beats_wrong_mean_by_problem": safe_mean(per_problem_pairwise),
        "within_problem_pairwise_problem_count": len(per_problem_pairwise),
        "within_problem_pairwise_below_random_frac": (
            sum(1 for value in per_problem_pairwise if value < 0.5) / len(per_problem_pairwise)
            if per_problem_pairwise
            else float("nan")
        ),
        "selected_wrong_more_steps_frac": (
            selected_wrong_more_steps / len(misranking_cases) if misranking_cases else 0.0
        ),
        "selected_wrong_longer_frac": selected_wrong_longer / len(misranking_cases) if misranking_cases else 0.0,
        "misranking_cases": misranking_cases,
    }


def print_report(result: dict[str, Any], case_limit: int, snippet_chars: int) -> None:
    print("=" * 80)
    print("PRM Reranking Diagnostics")
    print("=" * 80)
    print(f"num_examples                         : {result['num_examples']}")
    print(f"greedy_accuracy                      : {fmt(result['greedy_accuracy'])}")
    print(f"majority_vote_accuracy               : {fmt(result['majority_vote_accuracy'])}")
    print(f"prm_best_accuracy                    : {fmt(result['prm_best_accuracy'])}")
    print(f"sample_oracle_accuracy               : {fmt(result['sample_oracle_accuracy'])}")
    print(f"correct_available_count              : {result['correct_available_count']}")
    print(f"misranking_count                     : {result['misranking_count']}")
    print(
        "misranking_frac_among_correct_available: "
        f"{fmt(result['misranking_frac_among_correct_available'])}"
    )
    print(f"candidate_correct_score_mean         : {fmt(result['candidate_correct_score_mean'])}")
    print(f"candidate_wrong_score_mean           : {fmt(result['candidate_wrong_score_mean'])}")
    print(f"selected_correct_score_mean          : {fmt(result['selected_correct_score_mean'])}")
    print(f"selected_wrong_score_mean            : {fmt(result['selected_wrong_score_mean'])}")
    print(f"best_correct_rank_mean               : {fmt(result['best_correct_rank_mean'])}")
    print(f"within_problem_correct_beats_wrong   : {fmt(result['within_problem_correct_beats_wrong'])}")
    print(
        "within_problem_correct_beats_wrong_mean_by_problem: "
        f"{fmt(result['within_problem_correct_beats_wrong_mean_by_problem'])}"
    )
    print(f"within_problem_pairwise_problem_count: {result['within_problem_pairwise_problem_count']}")
    print(
        "within_problem_pairwise_below_random_frac: "
        f"{fmt(result['within_problem_pairwise_below_random_frac'])}"
    )
    print(f"selected_wrong_more_steps_frac       : {fmt(result['selected_wrong_more_steps_frac'])}")
    print(f"selected_wrong_longer_frac           : {fmt(result['selected_wrong_longer_frac'])}")

    print("\nWorst misranking cases")
    print("-" * 80)
    for case in result["misranking_cases"][:case_limit]:
        print(
            f"[row={case['row_idx']} dataset_idx={case['dataset_idx']}] "
            f"gap={case['score_gap_selected_wrong_minus_best_correct']:.4f} "
            f"rank_correct={case['best_correct_rank']} "
            f"gold={case['gold_answer']!r} selected={case['selected_boxed_answer']!r} "
            f"correct={case['best_correct_boxed_answer']!r}"
        )
        print(
            f"  selected: score={case['selected_score']:.4f} "
            f"steps={case['selected_n_steps']} chars={case['selected_chars']}"
        )
        print(f"  selected_text: {snippet(case['selected_text'], snippet_chars)}")
        print(
            f"  best_correct: score={case['best_correct_score']:.4f} "
            f"steps={case['best_correct_n_steps']} chars={case['best_correct_chars']}"
        )
        print(f"  correct_text:  {snippet(case['best_correct_text'], snippet_chars)}")
        print()


def main() -> None:
    args = parse_args()
    records = load_records(args.jsonl)
    result = analyze(records)
    print_report(result, args.case_limit, args.snippet_chars)

    if args.write_cases_jsonl:
        with open(args.write_cases_jsonl, "w", encoding="utf-8") as writer:
            for case in result["misranking_cases"]:
                writer.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"Saved misranking cases: {args.write_cases_jsonl}")


if __name__ == "__main__":
    main()
