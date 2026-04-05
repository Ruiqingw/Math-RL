"""
Utilities for loading raw OpenAI PRM800K data into the stepwise format used by
our verifier/token-PRM trainers.

The raw repository stores one full labeled solution per JSONL line. We expand
each labeled solution into one or more stepwise rows:

    {
        "prompt": <problem text>,
        "completions": [step_1, step_2, ...],
        "labels": [bool, bool, ...],
    }

For phase 2 this is important: when `chosen_completion` is null at the first
mistake, the listed completions are still labeled alternatives and should be
emitted as terminal branches rather than discarded.
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


def _truncate_to_first_negative(
    completions: List[str],
    labels: List[bool],
    *,
    stop_at_first_negative: bool,
) -> Tuple[List[str], List[bool]]:
    if not stop_at_first_negative:
        return list(completions), list(labels)

    for idx, label in enumerate(labels):
        if not label:
            return list(completions[: idx + 1]), list(labels[: idx + 1])
    return list(completions), list(labels)


def _truncate_row(
    prompt: str,
    completions: List[str],
    labels: List[bool],
    *,
    stop_at_first_negative: bool,
) -> Optional[Dict[str, Any]]:
    completions, labels = _truncate_to_first_negative(
        completions,
        labels,
        stop_at_first_negative=stop_at_first_negative,
    )
    if not completions:
        return None
    return {
        "prompt": prompt,
        "completions": completions,
        "labels": labels,
    }


def process_phase2_example_rows(
    example: Dict[str, Any],
    *,
    neutral_policy: str = "nonnegative",
    stop_at_first_negative: bool = True,
) -> List[Dict[str, Any]]:
    prompt = example["question"]["problem"]
    labeled_steps = example["label"]["steps"]

    outputs: List[Dict[str, Any]] = []
    previous_completions: List[str] = []
    previous_labels: List[bool] = []

    for step in labeled_steps:
        completions = step.get("completions") or []
        human_completion = step.get("human_completion")
        chosen_completion = step.get("chosen_completion")

        if not completions and human_completion is None:
            break

        for completion_idx, completion in enumerate(completions):
            if completion_idx == chosen_completion:
                continue
            rating = completion.get("rating")
            if rating is None:
                continue
            content = completion.get("text")
            if not content:
                continue
            label = rating_to_binary_label(int(rating), neutral_policy=neutral_policy)
            row = _truncate_row(
                prompt,
                previous_completions[:] + [content],
                previous_labels[:] + [label],
                stop_at_first_negative=stop_at_first_negative,
            )
            if row is not None:
                outputs.append(row)

        if chosen_completion is not None and completions:
            if not (0 <= chosen_completion < len(completions)):
                break
            chosen = completions[chosen_completion]
            rating = chosen.get("rating")
            content = chosen.get("text")
            if rating is None or not content:
                break
            label = rating_to_binary_label(int(rating), neutral_policy=neutral_policy)
            previous_completions.append(content)
            previous_labels.append(label)
            if stop_at_first_negative and not label:
                break
            continue

        if human_completion is not None:
            if isinstance(human_completion, dict):
                content = human_completion.get("text")
                rating = human_completion.get("rating", 1)
            else:
                content = str(human_completion)
                rating = 1
            if rating is None or not content:
                break
            label = rating_to_binary_label(int(rating), neutral_policy=neutral_policy)
            previous_completions.append(content)
            previous_labels.append(label)
            if stop_at_first_negative and not label:
                break
            continue

        if len(completions) == 1:
            only_completion = completions[0]
            rating = only_completion.get("rating")
            content = only_completion.get("text")
            if rating is None or not content:
                break
            label = rating_to_binary_label(int(rating), neutral_policy=neutral_policy)
            previous_completions.append(content)
            previous_labels.append(label)
            if stop_at_first_negative and not label:
                break
            continue

        if chosen_completion is None:
            break

    final_row = _truncate_row(
        prompt,
        previous_completions,
        previous_labels,
        stop_at_first_negative=stop_at_first_negative,
    )
    if final_row is not None:
        outputs.append(final_row)
    return outputs


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
        rows = process_phase2_example_rows(
            example,
            neutral_policy=neutral_policy,
            stop_at_first_negative=stop_at_first_negative,
        )
        outputs.extend(rows)

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
