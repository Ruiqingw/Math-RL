#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from typing import Any

from verl.utils.reward_score.math_reward import compute_score, last_boxed_only_string, remove_boxed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERIFIER_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "verifier")
if VERIFIER_DIR not in sys.path:
    sys.path.insert(0, VERIFIER_DIR)

from reward_fn import PRMClassifier, score_steps  # noqa: E402
from step_splitter import split_into_steps  # noqa: E402


_CACHED_VERIFIER: dict[str, Any] = {
    "key": None,
    "model": None,
    "tokenizer": None,
}


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


def _load_verifier(verifier_model_path: str, verifier_device: str):
    from transformers import AutoTokenizer

    cache_key = f"{verifier_model_path}::{verifier_device}"
    if _CACHED_VERIFIER["key"] == cache_key:
        return _CACHED_VERIFIER["model"], _CACHED_VERIFIER["tokenizer"]

    tokenizer = AutoTokenizer.from_pretrained(verifier_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PRMClassifier.from_pretrained(verifier_model_path, device=verifier_device)
    _CACHED_VERIFIER["key"] = cache_key
    _CACHED_VERIFIER["model"] = model
    _CACHED_VERIFIER["tokenizer"] = tokenizer
    return model, tokenizer


def verifier_shaping_reward(
    prompts,
    completions,
    problem,
    verifier_model_path,
    verifier_device="cpu",
    verifier_max_length=1536,
    verifier_batch_size=1,
    verifier_beta=0.1,
    verifier_delta=0.05,
    verifier_threshold=0.5,
    **kwargs,
):
    model, tokenizer = _load_verifier(verifier_model_path, verifier_device)
    rewards = []

    for completion, problem_text in zip(completions, problem):
        completion_text = normalize_completion(completion)
        steps = split_into_steps(completion_text)
        if not steps or not problem_text:
            rewards.append(0.0)
            continue

        step_scores = score_steps(
            problem=problem_text,
            steps=steps,
            model=model,
            tokenizer=tokenizer,
            device=verifier_device,
            max_length=verifier_max_length,
            batch_size=verifier_batch_size,
        )

        r_avg_step = sum(step_scores) / len(step_scores)
        r_first_error = 0.0
        for idx, score in enumerate(step_scores):
            if score < verifier_threshold:
                r_first_error = 1.0 - (idx / len(step_scores))
                break

        shaping = verifier_beta * r_avg_step - verifier_delta * r_first_error
        rewards.append(float(shaping))

    return rewards


class VerifierShapingReward:
    """Returns *only* the verifier shaping signal (no base correctness).

    Designed to be used alongside ``math_boxed_reward`` as a second reward
    function so that TRL logs each component separately:
      - ``rewards/math_boxed_reward/mean`` = accuracy (0/1)
      - ``rewards/verifier_shaping_reward/mean`` = shaping signal
    """

    def __init__(
        self,
        verifier_model_path: str,
        verifier_device: str = "cpu",
        verifier_max_length: int = 1536,
        verifier_batch_size: int = 1,
        verifier_beta: float = 0.1,
        verifier_delta: float = 0.05,
        verifier_threshold: float = 0.5,
    ):
        self.verifier_model_path = verifier_model_path
        self.verifier_device = verifier_device
        self.verifier_max_length = verifier_max_length
        self.verifier_batch_size = verifier_batch_size
        self.verifier_beta = verifier_beta
        self.verifier_delta = verifier_delta
        self.verifier_threshold = verifier_threshold
        self.__name__ = "verifier_shaping_reward"

    def __call__(self, prompts, completions, problem, **kwargs):
        return verifier_shaping_reward(
            prompts=prompts,
            completions=completions,
            problem=problem,
            verifier_model_path=self.verifier_model_path,
            verifier_device=self.verifier_device,
            verifier_max_length=self.verifier_max_length,
            verifier_batch_size=self.verifier_batch_size,
            verifier_beta=self.verifier_beta,
            verifier_delta=self.verifier_delta,
            verifier_threshold=self.verifier_threshold,
            **kwargs,
        )


# Keep MathVerifierReward for backward compatibility, but prefer using
# [math_boxed_reward, VerifierShapingReward(...)] as two separate reward_funcs.
class MathVerifierReward:
    def __init__(
        self,
        verifier_model_path: str,
        verifier_device: str = "cpu",
        verifier_max_length: int = 1536,
        verifier_batch_size: int = 1,
        verifier_beta: float = 0.1,
        verifier_delta: float = 0.05,
        verifier_threshold: float = 0.5,
    ):
        self.verifier_model_path = verifier_model_path
        self.verifier_device = verifier_device
        self.verifier_max_length = verifier_max_length
        self.verifier_batch_size = verifier_batch_size
        self.verifier_beta = verifier_beta
        self.verifier_delta = verifier_delta
        self.verifier_threshold = verifier_threshold
        self.__name__ = "math_verifier_reward"

    def __call__(self, prompts, completions, gold_answer, problem, **kwargs):
        base_rewards = math_boxed_reward(
            prompts=prompts,
            completions=completions,
            gold_answer=gold_answer,
            **kwargs,
        )
        shaping_rewards = verifier_shaping_reward(
            prompts=prompts,
            completions=completions,
            problem=problem,
            verifier_model_path=self.verifier_model_path,
            verifier_device=self.verifier_device,
            verifier_max_length=self.verifier_max_length,
            verifier_batch_size=self.verifier_batch_size,
            verifier_beta=self.verifier_beta,
            verifier_delta=self.verifier_delta,
            verifier_threshold=self.verifier_threshold,
            **kwargs,
        )
        return [float(base + shaping) for base, shaping in zip(base_rewards, shaping_rewards)]
