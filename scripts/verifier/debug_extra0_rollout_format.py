#!/usr/bin/env python3
"""Show how rollout text is parsed and formatted for extra0 PRM scoring."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Iterable


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from qwen_extra0_prm import format_solution_with_extra0  # noqa: E402
from step_splitter import split_into_steps  # noqa: E402


SYNTHETIC_PROBLEM = "Solve for x: x + 1 = 3."
SYNTHETIC_COMPLETION = (
    "1. Subtract 1 from both sides.\n\n"
    "2. This gives x = 2.\n\n"
    "Therefore the answer is \\boxed{2}."
)


def _candidate_text(row: dict[str, Any], source: str, sample_index: int) -> str:
    if source == "greedy":
        return str(row.get("greedy_text", "") or "")

    sampled = row.get("sampled", [])
    if not isinstance(sampled, list) or not sampled:
        return ""

    if source == "prm_best":
        best_idx = row.get("prm_best_index", sample_index)
        try:
            sample_index = int(best_idx)
        except (TypeError, ValueError):
            sample_index = 0

    if sample_index < 0 or sample_index >= len(sampled):
        sample_index = 0
    sample = sampled[sample_index]
    if isinstance(sample, dict):
        return str(sample.get("text", "") or "")
    return str(sample or "")


def iter_examples(args: argparse.Namespace) -> Iterable[tuple[str, str]]:
    if args.jsonl:
        with open(args.jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                problem = str(row.get("problem", "") or row.get("prompt", "") or "")
                completion = _candidate_text(row, args.source, args.sample_index)
                yield problem, completion
        return

    problem = args.problem or SYNTHETIC_PROBLEM
    completion = args.completion or SYNTHETIC_COMPLETION
    yield problem, completion


def build_debug_record(problem: str, completion: str) -> dict[str, Any]:
    steps = [step for step in split_into_steps(completion) if step.strip()]
    extra0_input = format_solution_with_extra0(problem, steps) if problem and steps else ""
    return {
        "problem": problem,
        "raw_rollout_text": completion,
        "parsed_step_count": len(steps),
        "parsed_steps": steps,
        "extra0_scoring_input": extra0_input,
    }


def format_markdown(record: dict[str, Any], idx: int) -> str:
    parsed_steps = "\n".join(
        f"[{step_idx}] {step}" for step_idx, step in enumerate(record["parsed_steps"], start=1)
    )
    return (
        f"## Example {idx}\n\n"
        "### Problem\n\n"
        f"{record['problem']}\n\n"
        "### Raw Rollout Text\n\n"
        "```text\n"
        f"{record['raw_rollout_text']}\n"
        "```\n\n"
        f"### Parsed Steps ({record['parsed_step_count']})\n\n"
        "```text\n"
        f"{parsed_steps}\n"
        "```\n\n"
        "### Extra0 Scoring Input\n\n"
        "```text\n"
        f"{record['extra0_scoring_input']}\n"
        "```\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="", help="Problem text for a single manual example.")
    parser.add_argument("--completion", default="", help="Raw rollout completion for a single manual example.")
    parser.add_argument("--jsonl", default="", help="Optional eval_prm_best_of_n JSONL to sample examples from.")
    parser.add_argument(
        "--source",
        default="sampled",
        choices=["sampled", "greedy", "prm_best"],
        help="Which completion source to read when --jsonl is used.",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index for --source sampled.")
    parser.add_argument("--limit", type=int, default=3, help="Maximum examples to print.")
    parser.add_argument("--format", choices=["markdown", "jsonl"], default="markdown")
    parser.add_argument("--output", default="", help="Optional output path. Defaults to stdout.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs: list[str] = []
    for idx, (problem, completion) in enumerate(iter_examples(args), start=1):
        if idx > args.limit:
            break
        record = build_debug_record(problem, completion)
        if args.format == "jsonl":
            outputs.append(json.dumps(record, ensure_ascii=False))
        else:
            outputs.append(format_markdown(record, idx))

    text = "\n".join(outputs)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
            if text:
                f.write("\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
