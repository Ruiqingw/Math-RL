"""
Utilities for loading raw OpenAI PRM800K data into the stepwise format used by
our verifier/token-PRM trainers.

The raw repository stores one full labeled solution per JSONL line. This module
reconstructs the chosen trajectory for each labeled solution and emits rows of:

    {
        "prompt": <problem text>,
        "completions": [step_1, step_2, ...],
        "labels": [bool, bool, ...],
    }

Design choices for the current token-PRM line:
- default to raw phase 2 only
- default to first-error-only truncation
- treat rating >= 0 as non-error (positive) and rating < 0 as error (negative)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from datasets import DatasetDict, load_dataset, load_from_disk


DEFAULT_RAW_DATA_DIR = "/root/autodl-tmp/prm_grpo/prm800k_raw/prm800k/data"
DEFAULT_CACHE_ROOT = "/root/autodl-tmp/prm_grpo/datasets"


def phase2_cache_dir(
    *,
    cache_root: str = DEFAULT_CACHE_ROOT,
    neutral_policy: str = "nonnegative",
    stop_at_first_negative: bool = True,
) -> str:
    neutral_tag = "nonneg" if neutral_policy == "nonnegative" else "posonly"
    prefix_tag = "firsterr" if stop_at_first_negative else "allsteps"
    return os.path.join(
        cache_root,
        f"prm800k_openai_phase2_stepwise_{neutral_tag}_{prefix_tag}",
    )


DEFAULT_PHASE2_CACHE_DIR = phase2_cache_dir()


def rating_to_binary_label(rating: int, neutral_policy: str = "nonnegative") -> bool:
    """
    Map OpenAI PRM800K ratings {-1, 0, +1} to the binary supervision used by
    the current verifier/token-PRM trainers.
    """
    rating = int(rating)
    if neutral_policy == "nonnegative":
        return rating >= 0
    if neutral_policy == "positive_only":
        return rating == 1
    raise ValueError(f"Unknown neutral policy: {neutral_policy}")


def _extract_chosen_step(step: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    completions = step.get("completions") or []
    chosen_completion = step.get("chosen_completion")
    human_completion = step.get("human_completion")

    if chosen_completion is not None and completions:
        if 0 <= chosen_completion < len(completions):
            chosen = completions[chosen_completion]
            rating = chosen.get("rating")
            if rating is None:
                return None
            return chosen["text"], int(rating)
        return None

    if human_completion is not None:
        if isinstance(human_completion, dict):
            text = human_completion["text"]
            raw_rating = human_completion.get("rating", 1)
            if raw_rating is None:
                return None
            rating = int(raw_rating)
        else:
            text = str(human_completion)
            rating = 1
        return text, rating

    if len(completions) == 1:
        chosen = completions[0]
        rating = chosen.get("rating")
        if rating is None:
            return None
        return chosen["text"], int(rating)

    return None


def process_phase2_example(
    example: Dict[str, Any],
    *,
    neutral_policy: str = "nonnegative",
    stop_at_first_negative: bool = True,
) -> Optional[Dict[str, Any]]:
    prompt = example["question"]["problem"]
    labeled_steps = example["label"]["steps"]

    completions: List[str] = []
    labels: List[bool] = []

    for step in labeled_steps:
        chosen = _extract_chosen_step(step)
        if chosen is None:
            break

        text, rating = chosen
        if not text:
            break

        label = rating_to_binary_label(rating, neutral_policy=neutral_policy)
        completions.append(text)
        labels.append(label)

        if stop_at_first_negative and not label:
            break

    if not completions:
        return None

    return {
        "prompt": prompt,
        "completions": completions,
        "labels": labels,
    }


def process_phase2_batch(
    examples: Dict[str, List[Any]],
    *,
    neutral_policy: str = "nonnegative",
    stop_at_first_negative: bool = True,
) -> Dict[str, List[Any]]:
    outputs: List[Dict[str, Any]] = []
    batch_size = len(examples["label"])

    for idx in range(batch_size):
        example = {key: value[idx] for key, value in examples.items()}
        row = process_phase2_example(
            example,
            neutral_policy=neutral_policy,
            stop_at_first_negative=stop_at_first_negative,
        )
        if row is not None:
            outputs.append(row)

    if not outputs:
        return {"prompt": [], "completions": [], "labels": []}

    return {key: [row[key] for row in outputs] for key in ("prompt", "completions", "labels")}


def build_raw_phase2_dataset(
    *,
    raw_data_dir: str = DEFAULT_RAW_DATA_DIR,
    cache_dir: Optional[str] = None,
    force_rebuild: bool = False,
    neutral_policy: str = "nonnegative",
    stop_at_first_negative: bool = True,
) -> DatasetDict:
    if cache_dir is None:
        cache_dir = phase2_cache_dir(
            neutral_policy=neutral_policy,
            stop_at_first_negative=stop_at_first_negative,
        )

    if cache_dir and os.path.isdir(cache_dir) and not force_rebuild:
        return load_from_disk(cache_dir)

    data_files = {
        "train": os.path.join(raw_data_dir, "phase2_train.jsonl"),
        "test": os.path.join(raw_data_dir, "phase2_test.jsonl"),
    }
    for split_name, path in data_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing raw PRM800K {split_name} split: {path}\n"
                "Expected a clone of https://github.com/openai/prm800k under "
                f"{os.path.dirname(os.path.dirname(raw_data_dir))}"
            )

    raw_dataset = load_dataset("json", data_files=data_files)
    processed = raw_dataset.map(
        lambda batch: process_phase2_batch(
            batch,
            neutral_policy=neutral_policy,
            stop_at_first_negative=stop_at_first_negative,
        ),
        batched=True,
        batch_size=16,
        remove_columns=raw_dataset["train"].column_names,
        desc="Processing raw OpenAI PRM800K phase2",
    )

    if cache_dir:
        os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
        processed.save_to_disk(cache_dir)

    return processed
