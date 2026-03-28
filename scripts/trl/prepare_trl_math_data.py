#!/usr/bin/env python3
"""
Prepare MATH-lighteval data for TRL GRPO training.

This keeps a lightweight schema that is easier to consume from custom reward
functions and TRL trainers:

- prompt: plain text prompt fed to the policy
- problem: raw problem statement
- gold_answer: extracted final boxed answer
- solution: raw reference solution
- split / index / data_source / ability: metadata

It can either:
1. Load the original HF dataset directly, or
2. Read the existing verl parquet files and flatten them into a TRL-friendly
   schema without changing answer extraction logic.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import datasets

from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
DEFAULT_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def extract_solution(solution_str: str) -> str:
    return remove_boxed(last_boxed_only_string(solution_str))


def build_prompt(problem: str, instruction: str) -> str:
    return f"{problem} {instruction}".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TRL-friendly MATH parquet data.")
    parser.add_argument(
        "--source",
        choices=["hf", "verl_parquet"],
        default="verl_parquet",
        help="Whether to build from the HF dataset directly or flatten existing verl parquet files.",
    )
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Optional local raw dataset path when --source=hf.",
    )
    parser.add_argument(
        "--verl_data_dir",
        default="/root/autodl-tmp/prm_grpo/data/verl_math",
        help="Directory containing existing verl train.parquet/test.parquet when --source=verl_parquet.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="/root/autodl-tmp/prm_grpo/data/trl_math",
        help="Directory where TRL parquet files will be written.",
    )
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Instruction suffix appended to each math problem.",
    )
    return parser.parse_args()


def process_hf_example(example: dict[str, Any], idx: int, split: str, instruction: str) -> dict[str, Any]:
    problem = example["problem"]
    solution = example["solution"]
    gold_answer = extract_solution(solution)
    return {
        "prompt": build_prompt(problem, instruction),
        "problem": problem,
        "gold_answer": gold_answer,
        "solution": solution,
        "split": split,
        "index": idx,
        "data_source": DATA_SOURCE,
        "ability": "math",
    }


def process_verl_example(example: dict[str, Any]) -> dict[str, Any]:
    prompt = example["prompt"]
    if isinstance(prompt, list) and prompt:
        prompt_text = prompt[0].get("content", "")
    else:
        prompt_text = ""

    extra_info = example.get("extra_info", {}) or {}
    reward_model = example.get("reward_model", {}) or {}

    return {
        "prompt": prompt_text,
        "problem": extra_info.get("question", ""),
        "gold_answer": reward_model.get("ground_truth", ""),
        "solution": extra_info.get("answer", ""),
        "split": extra_info.get("split", ""),
        "index": extra_info.get("index", -1),
        "data_source": example.get("data_source", DATA_SOURCE),
        "ability": example.get("ability", "math"),
    }


def load_from_hf(args: argparse.Namespace) -> tuple[datasets.Dataset, datasets.Dataset]:
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(DATA_SOURCE)

    train_dataset = dataset["train"].map(
        lambda ex, idx: process_hf_example(ex, idx, "train", args.instruction),
        with_indices=True,
    )
    test_dataset = dataset["test"].map(
        lambda ex, idx: process_hf_example(ex, idx, "test", args.instruction),
        with_indices=True,
    )
    return train_dataset, test_dataset


def load_from_verl_parquet(args: argparse.Namespace) -> tuple[datasets.Dataset, datasets.Dataset]:
    train_path = os.path.join(args.verl_data_dir, "train.parquet")
    test_path = os.path.join(args.verl_data_dir, "test.parquet")
    train_dataset = datasets.load_dataset("parquet", data_files=train_path, split="train")
    test_dataset = datasets.load_dataset("parquet", data_files=test_path, split="train")

    train_dataset = train_dataset.map(process_verl_example)
    test_dataset = test_dataset.map(process_verl_example)
    return train_dataset, test_dataset


def main() -> None:
    args = parse_args()

    if args.source == "hf":
        train_dataset, test_dataset = load_from_hf(args)
    else:
        train_dataset, test_dataset = load_from_verl_parquet(args)

    os.makedirs(args.local_save_dir, exist_ok=True)
    train_path = os.path.join(args.local_save_dir, "train.parquet")
    test_path = os.path.join(args.local_save_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    with open(os.path.join(args.local_save_dir, "train_example.json"), "w", encoding="utf-8") as f:
        json.dump(train_dataset[0], f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.local_save_dir, "test_example.json"), "w", encoding="utf-8") as f:
        json.dump(test_dataset[0], f, indent=2, ensure_ascii=False)

    print(f"Saved TRL train parquet: {train_path}")
    print(f"Saved TRL test parquet:  {test_path}")


if __name__ == "__main__":
    main()
