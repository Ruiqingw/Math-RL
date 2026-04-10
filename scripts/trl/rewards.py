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

from reward_fn import PRMClassifier, score_steps as score_steps_classifier  # noqa: E402
from step_splitter import split_into_steps  # noqa: E402
from token_reward_fn import (  # noqa: E402
    load_model_bundle as load_token_prm_bundle,
    score_steps as score_steps_token_prm,
)


_CACHED_VERIFIER: dict[str, Any] = {
    "key": None,
    "backend": None,
    "model": None,
    "tokenizer": None,
    "label_tokens": None,
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


def _group_key(prompt: Any, problem_text: Any, answer: Any) -> tuple[str, str, str]:
    return (str(prompt), str(problem_text), str(answer))


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
        return (
            _CACHED_VERIFIER["backend"],
            _CACHED_VERIFIER["model"],
            _CACHED_VERIFIER["tokenizer"],
            _CACHED_VERIFIER["label_tokens"],
        )

    cls_head_path = os.path.join(verifier_model_path, "cls_head.pt")
    if os.path.exists(cls_head_path):
        tokenizer = AutoTokenizer.from_pretrained(verifier_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "left"

        model = PRMClassifier.from_pretrained(verifier_model_path, device=verifier_device)
        backend = "classifier"
        label_tokens = None
    else:
        device_map = None if verifier_device == "cpu" else "auto"
        model, tokenizer, label_tokens = load_token_prm_bundle(
            verifier_model_path,
            device_map=device_map,
        )
        backend = "token_prm"

    _CACHED_VERIFIER["key"] = cache_key
    _CACHED_VERIFIER["backend"] = backend
    _CACHED_VERIFIER["model"] = model
    _CACHED_VERIFIER["tokenizer"] = tokenizer
    _CACHED_VERIFIER["label_tokens"] = label_tokens
    return backend, model, tokenizer, label_tokens


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
    verifier_tiebreak_only=False,
    gold_answer=None,
    **kwargs,
):
    backend, model, tokenizer, label_tokens = _load_verifier(
        verifier_model_path,
        verifier_device,
    )
    log_metric = kwargs.get("log_metric")
    rewards = []
    base_correct_flags = []
    min_step_scores = []
    min_penalties = []
    if gold_answer is None:
        gold_answer = [None] * len(completions)

    for prompt, completion, problem_text, answer in zip(prompts, completions, problem, gold_answer):
        completion_text = normalize_completion(completion)
        steps = split_into_steps(completion_text)
        if not steps or not problem_text:
            rewards.append(0.0)
            base_correct_flags.append(False)
            continue

        base_correct = False
        if answer is not None:
            try:
                base_correct = bool(compute_score(completion_text, ground_truth=answer))
            except Exception:
                base_correct = False

        if backend == "classifier":
            step_scores = score_steps_classifier(
                problem=problem_text,
                steps=steps,
                model=model,
                tokenizer=tokenizer,
                device=verifier_device,
                max_length=verifier_max_length,
                batch_size=verifier_batch_size,
            )
        else:
            step_scores = score_steps_token_prm(
                problem=problem_text,
                steps=steps,
                model=model,
                tokenizer=tokenizer,
                label_tokens=label_tokens,
                device=verifier_device,
                max_length=verifier_max_length,
                batch_size=verifier_batch_size,
            )

        # Use a weakest-link signal. Offline best-of-N diagnostics showed that
        # min aggregation preserves PRM usefulness better than averaging, which
        # can dilute a single bad step across many fluent-looking steps.
        min_step_score = min(step_scores)
        min_centered_score = min_step_score - verifier_threshold
        min_penalty = max(verifier_threshold - min_step_score, 0.0)
        min_step_scores.append(float(min_step_score))
        min_penalties.append(float(min_penalty))

        # Conservative PRM usage:
        # - Correct final answers keep the clean 0/1 boxed reward only.
        # - Incorrect final answers receive only a non-positive min-step penalty.
        if base_correct:
            shaping = 0.0
        else:
            shaping = verifier_beta * min(min_centered_score, 0.0)
        rewards.append(float(shaping))
        base_correct_flags.append(base_correct)

    if rewards:
        gated_rewards = rewards[:]
        total_groups = 0
        all_wrong_groups = 0
        active_tiebreak_groups = 0
        start = 0
        while start < len(gated_rewards):
            group_id = _group_key(prompts[start], problem[start], gold_answer[start])
            end = start + 1
            while end < len(gated_rewards) and _group_key(prompts[end], problem[end], gold_answer[end]) == group_id:
                end += 1

            total_groups += 1
            group_has_tie = end - start > 1
            group_all_wrong = not any(base_correct_flags[start:end])
            if group_all_wrong:
                all_wrong_groups += 1
            if verifier_tiebreak_only:
                # Only let the verifier break ties when every sampled completion
                # for this prompt is still wrong under the main answer-level reward.
                if group_has_tie and group_all_wrong:
                    active_tiebreak_groups += 1
                else:
                    for idx in range(start, end):
                        gated_rewards[idx] = 0.0
            start = end

        if log_metric is not None and total_groups > 0:
            log_metric("math_boxed_reward/all_wrong_group_count", float(all_wrong_groups))
            log_metric("math_boxed_reward/all_wrong_group_frac", float(all_wrong_groups / total_groups))
            if min_step_scores:
                log_metric(
                    "verifier_shaping_reward/min_step_score_mean",
                    float(sum(min_step_scores) / len(min_step_scores)),
                )
                log_metric(
                    "verifier_shaping_reward/min_penalty_mean",
                    float(sum(min_penalties) / len(min_penalties)),
                )
            if verifier_tiebreak_only:
                log_metric("verifier_shaping_reward/gate_active_group_count", float(active_tiebreak_groups))
                log_metric(
                    "verifier_shaping_reward/gate_active_group_frac",
                    float(active_tiebreak_groups / total_groups),
                )
        if verifier_tiebreak_only:
            rewards = gated_rewards

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
        verifier_tiebreak_only: bool = False,
    ):
        self.verifier_model_path = verifier_model_path
        self.verifier_device = verifier_device
        self.verifier_max_length = verifier_max_length
        self.verifier_batch_size = verifier_batch_size
        self.verifier_beta = verifier_beta
        self.verifier_delta = verifier_delta
        self.verifier_threshold = verifier_threshold
        self.verifier_tiebreak_only = verifier_tiebreak_only
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
            verifier_tiebreak_only=self.verifier_tiebreak_only,
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
