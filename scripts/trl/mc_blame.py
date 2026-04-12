#!/usr/bin/env python3
"""
Monte Carlo blame assignment via greedy completion from step prefixes.

Uses vLLM for fast batched inference. Instead of binary search (sequential),
generates completions from ALL step prefixes in one batched vLLM call,
then finds the earliest step where recovery fails.

Reward formula:
  correct rollout  → reward = 0.0 (math_boxed_reward provides the +1)
  wrong rollout    → reward = -beta * (total_steps - blame_step) / total_steps
"""

from __future__ import annotations

import os
import sys
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERIFIER_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "verifier")
if VERIFIER_DIR not in sys.path:
    sys.path.insert(0, VERIFIER_DIR)

from step_splitter import split_into_steps  # noqa: E402


_CACHED_BLAME_LLM: dict[str, Any] = {"key": None, "llm": None}


def _load_blame_llm(model_path: str, gpu_memory_utilization: float = 0.15):
    if _CACHED_BLAME_LLM["key"] == model_path:
        return _CACHED_BLAME_LLM["llm"]

    from vllm import LLM

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        max_model_len=2048,
        trust_remote_code=True,
    )
    _CACHED_BLAME_LLM["key"] = model_path
    _CACHED_BLAME_LLM["llm"] = llm
    return llm


def check_answer(text: str, gold_answer: str) -> bool:
    from verl.utils.reward_score.math_reward import compute_score
    try:
        return compute_score(text, ground_truth=gold_answer) > 0.0
    except Exception:
        return False


def blame_reward(
    blame_step: int,
    total_steps: int,
    beta: float = 0.5,
) -> float:
    if total_steps <= 1:
        return -beta
    return -beta * (total_steps - blame_step) / total_steps


def compute_blame_rewards_batch(
    prompts: list[str],
    completions: list[str],
    gold_answers: list[str],
    base_correct: list[bool],
    model_path: str,
    beta: float = 0.5,
    max_new_tokens: int = 512,
    gpu_memory_utilization: float = 0.15,
) -> list[float]:
    """Compute blame-based rewards for a batch using vLLM batched inference.

    All wrong rollouts' step prefixes are collected and generated in ONE
    vLLM call for maximum throughput.
    """
    from vllm import SamplingParams

    wrong_indices: list[int] = []
    wrong_steps: list[list[str]] = []
    wrong_prompts: list[str] = []
    wrong_answers: list[str] = []

    for i, (prompt, completion, gold, correct) in enumerate(
        zip(prompts, completions, gold_answers, base_correct)
    ):
        if not correct:
            steps = split_into_steps(completion)
            wrong_indices.append(i)
            wrong_steps.append(steps)
            wrong_prompts.append(prompt)
            wrong_answers.append(gold)

    rewards = [0.0] * len(prompts)

    if not wrong_indices:
        return rewards

    all_prefixes: list[str] = []
    prefix_map: list[tuple[int, int]] = []  # (wrong_idx_in_list, step_k)

    for wi, (prompt, steps) in enumerate(zip(wrong_prompts, wrong_steps)):
        n = len(steps)
        if n == 0:
            orig_idx = wrong_indices[wi]
            rewards[orig_idx] = -beta
            continue
        for k in range(n + 1):
            if k == 0:
                prefix = prompt
            else:
                prefix = prompt + "\n\n" + "\n\n".join(steps[:k])
            all_prefixes.append(prefix)
            prefix_map.append((wi, k))

    if not all_prefixes:
        return rewards

    llm = _load_blame_llm(model_path, gpu_memory_utilization)
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        top_p=1.0,
    )
    outputs = llm.generate(all_prefixes, sampling_params)

    results: dict[int, dict[int, bool]] = {}
    for idx, ((wi, k), output) in enumerate(zip(prefix_map, outputs)):
        completion_text = output.outputs[0].text
        full_text = all_prefixes[idx] + completion_text
        correct = check_answer(full_text, wrong_answers[wi])
        results.setdefault(wi, {})[k] = correct

    for wi in range(len(wrong_indices)):
        steps = wrong_steps[wi]
        n = len(steps)
        if n == 0:
            continue

        step_results = results.get(wi, {})
        blame = n
        for k in range(n + 1):
            if not step_results.get(k, False):
                blame = k + 1
                break

        blame = min(blame, n)
        orig_idx = wrong_indices[wi]
        rewards[orig_idx] = blame_reward(blame, n, beta=beta)

    return rewards
