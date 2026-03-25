#!/usr/bin/env python3
"""
Prepare MATH-lighteval data for verl.

Compared with verl's stock preprocessing script, this variant keeps the raw
problem text in extra_info so later custom reward functions can access it.
That is useful for verifier-guided RL, where reward computation needs the
original question in addition to the generated solution.
"""

import argparse
import json
import os

import datasets

from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str: str) -> str:
    return remove_boxed(last_boxed_only_string(solution_str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare verl MATH parquet data with question metadata.")
    parser.add_argument("--local_dataset_path", default=None, help="Optional local raw dataset path.")
    parser.add_argument(
        "--local_save_dir",
        default="/root/autodl-tmp/prm_grpo/data/verl_math",
        help="Directory where parquet files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_source = "DigitalLearningGmbH/MATH-lighteval"

    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["problem"]
            answer_raw = example["solution"]
            solution = extract_solution(answer_raw)

            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": f"{question_raw} {instruction_following}"}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question_raw,
                    "answer": answer_raw,
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    os.makedirs(args.local_save_dir, exist_ok=True)
    train_path = os.path.join(args.local_save_dir, "train.parquet")
    test_path = os.path.join(args.local_save_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    with open(os.path.join(args.local_save_dir, "train_example.json"), "w", encoding="utf-8") as f:
        json.dump(train_dataset[0], f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.local_save_dir, "test_example.json"), "w", encoding="utf-8") as f:
        json.dump(test_dataset[0], f, indent=2, ensure_ascii=False)

    print(f"Saved train parquet: {train_path}")
    print(f"Saved test parquet:  {test_path}")


if __name__ == "__main__":
    main()
