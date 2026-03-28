#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from verl.utils.reward_score.math_reward import compute_score, last_boxed_only_string, remove_boxed


def extract_boxed_answer(text: str) -> str:
    if not text:
        return ""
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return ""
    return remove_boxed(boxed)


def normalize_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        if not completion:
            return ""
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def math_boxed_reward(prompts, completions, gold_answer, **kwargs):
    rewards = []
    for completion, answer in zip(completions, gold_answer):
        completion_text = normalize_completion(completion)
        rewards.append(float(compute_score(completion_text, ground_truth=answer)))
    return rewards

