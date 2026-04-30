#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
import logging
from typing import Any

try:
    from verl.utils.reward_score.math_reward import compute_score, last_boxed_only_string, remove_boxed
except ModuleNotFoundError:
    try:
        from math_verify import parse as math_verify_parse
        from math_verify import verify as math_verify_verify
    except ModuleNotFoundError:  # pragma: no cover
        math_verify_parse = None
        math_verify_verify = None

    def last_boxed_only_string(text: str) -> str | None:
        start = str(text).rfind("\\boxed")
        if start < 0:
            return None
        brace_start = text.find("{", start)
        if brace_start < 0:
            return None
        depth = 0
        for idx in range(brace_start, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    def remove_boxed(boxed: str) -> str:
        text = str(boxed).strip()
        prefix = "\\boxed{"
        if text.startswith(prefix) and text.endswith("}"):
            return text[len(prefix) : -1]
        return text

    def compute_score(solution_str: str, ground_truth: str) -> float:
        if math_verify_parse is not None and math_verify_verify is not None:
            gold = math_verify_parse(str(ground_truth))
            pred = math_verify_parse(str(solution_str))
            if math_verify_verify(gold, pred):
                return 1.0
            boxed = last_boxed_only_string(solution_str)
            if boxed is not None and math_verify_verify(gold, math_verify_parse(remove_boxed(boxed))):
                return 1.0
            return 0.0

        boxed = last_boxed_only_string(solution_str)
        pred_answer = remove_boxed(boxed) if boxed is not None else str(solution_str).strip()
        return 1.0 if pred_answer.strip() == str(ground_truth).strip() else 0.0

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
from qwen_extra0_prm import (  # noqa: E402
    load_extra0_prm,
    score_steps as score_steps_extra0,
)


logger = logging.getLogger(__name__)
VERIFIER_BACKEND_IDS = {
    "classifier": 1.0,
    "token_prm": 2.0,
    "extra0_token_cls": 3.0,
}


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


def _load_json_file(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as reader:
        return json.load(reader)


def _detect_verifier_backend(verifier_model_path: str) -> str:
    if os.path.exists(os.path.join(verifier_model_path, "cls_head.pt")):
        return "classifier"

    adapter_config_path = os.path.join(verifier_model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        adapter_config = _load_json_file(adapter_config_path)
        if str(adapter_config.get("task_type", "")).upper() == "TOKEN_CLS":
            return "extra0_token_cls"

    config_path = os.path.join(verifier_model_path, "config.json")
    if os.path.exists(config_path):
        config = _load_json_file(config_path)
        architectures = [str(item) for item in config.get("architectures", [])]
        if any("TokenClassification" in architecture for architecture in architectures):
            return "extra0_token_cls"
        if int(config.get("num_labels", 0) or 0) == 2 and config.get("id2label") is not None:
            return "extra0_token_cls"

    return "token_prm"


def _load_verifier(verifier_model_path: str, verifier_device: str, verifier_backend: str = "auto"):
    from transformers import AutoTokenizer

    requested_backend = verifier_backend
    backend = _detect_verifier_backend(verifier_model_path) if verifier_backend == "auto" else verifier_backend
    if backend not in VERIFIER_BACKEND_IDS:
        raise ValueError(f"Unsupported verifier backend: {verifier_backend}")

    cache_key = f"{verifier_model_path}::{verifier_device}::{backend}"
    if _CACHED_VERIFIER["key"] == cache_key:
        return (
            _CACHED_VERIFIER["backend"],
            _CACHED_VERIFIER["model"],
            _CACHED_VERIFIER["tokenizer"],
            _CACHED_VERIFIER["label_tokens"],
        )

    cls_head_path = os.path.join(verifier_model_path, "cls_head.pt")
    logger.info("Loading verifier backend=%s requested=%s path=%s device=%s", backend, requested_backend, verifier_model_path, verifier_device)

    if backend == "classifier":
        tokenizer = AutoTokenizer.from_pretrained(verifier_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "left"

        if not os.path.exists(cls_head_path):
            raise FileNotFoundError(f"Classification-head verifier selected but cls_head.pt was not found: {cls_head_path}")
        model = PRMClassifier.from_pretrained(verifier_model_path, device=verifier_device)
        label_tokens = None
    elif backend == "extra0_token_cls":
        device_map = None if verifier_device == "cpu" else "auto"
        model, tokenizer, _ = load_extra0_prm(
            verifier_model_path,
            device_map=device_map,
        )
        label_tokens = None
    else:
        device_map = None if verifier_device == "cpu" else "auto"
        model, tokenizer, label_tokens = load_token_prm_bundle(
            verifier_model_path,
            device_map=device_map,
        )

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
    verifier_backend="auto",
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
    if not 0.0 <= float(verifier_beta) <= 1.0:
        raise ValueError(f"verifier_beta must be in [0, 1] so wrong answers cannot exceed correct answers, got {verifier_beta}")
    backend, model, tokenizer, label_tokens = _load_verifier(
        verifier_model_path,
        verifier_device,
        verifier_backend,
    )
    log_metric = kwargs.get("log_metric")
    rewards = []
    base_correct_flags = []
    min_step_scores = []
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
        elif backend == "extra0_token_cls":
            step_scores = score_steps_extra0(
                problem=problem_text,
                steps=steps,
                model=model,
                tokenizer=tokenizer,
                device=verifier_device,
                max_length=verifier_max_length,
                require_all_steps=False,
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

        if not step_scores:
            rewards.append(0.0)
            base_correct_flags.append(base_correct)
            continue

        # Use a weakest-link signal. Offline best-of-N diagnostics showed that
        # min aggregation preserves PRM usefulness better than averaging, which
        # can dilute a single bad step across many fluent-looking steps.
        min_step_score = min(step_scores)
        min_step_scores.append(float(min_step_score))

        # Min-form shaping:
        # - Correct final answers keep the clean 0/1 boxed reward only.
        # - Incorrect final answers receive min_step_score directly as shaping.
        # Used alongside math_boxed_reward, this yields:
        # reward = base_correct + beta * (1 - base_correct) * prm_score.
        if base_correct:
            shaping = 0.0
        else:
            shaping = verifier_beta * min_step_score
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
            log_metric("verifier_shaping_reward/backend_id", VERIFIER_BACKEND_IDS[backend])
            log_metric(
                "verifier_shaping_reward/backend_is_extra0_token_cls",
                1.0 if backend == "extra0_token_cls" else 0.0,
            )
            log_metric("math_boxed_reward/all_wrong_group_count", float(all_wrong_groups))
            log_metric("math_boxed_reward/all_wrong_group_frac", float(all_wrong_groups / total_groups))
            if min_step_scores:
                log_metric(
                    "verifier_shaping_reward/min_step_score_mean",
                    float(sum(min_step_scores) / len(min_step_scores)),
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
        verifier_backend: str = "auto",
        verifier_device: str = "cpu",
        verifier_max_length: int = 1536,
        verifier_batch_size: int = 1,
        verifier_beta: float = 0.1,
        verifier_delta: float = 0.05,
        verifier_threshold: float = 0.5,
        verifier_tiebreak_only: bool = False,
    ):
        self.verifier_model_path = verifier_model_path
        self.verifier_backend = verifier_backend
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
            verifier_backend=self.verifier_backend,
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
        verifier_backend: str = "auto",
        verifier_device: str = "cpu",
        verifier_max_length: int = 1536,
        verifier_batch_size: int = 1,
        verifier_beta: float = 0.1,
        verifier_delta: float = 0.05,
        verifier_threshold: float = 0.5,
    ):
        self.verifier_model_path = verifier_model_path
        self.verifier_backend = verifier_backend
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
            verifier_backend=self.verifier_backend,
            verifier_device=self.verifier_device,
            verifier_max_length=self.verifier_max_length,
            verifier_batch_size=self.verifier_batch_size,
            verifier_beta=self.verifier_beta,
            verifier_delta=self.verifier_delta,
            verifier_threshold=self.verifier_threshold,
            **kwargs,
        )
        return [float(base + shaping) for base, shaping in zip(base_rewards, shaping_rewards)]


class MCBlameReward:
    """Monte Carlo blame-based step-level reward using vLLM batched inference.

    All wrong rollouts' step prefixes are generated in one batched vLLM call.
    Correct rollouts get 0.0 (the +1 comes from math_boxed_reward).
    """

    def __init__(
        self,
        model_path: str,
        beta: float = 0.5,
        max_new_tokens: int = 512,
        gpu_memory_utilization: float = 0.15,
    ):
        self.model_path = model_path
        self.beta = beta
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.__name__ = "mc_blame_reward"

    def __call__(self, prompts, completions, gold_answer, **kwargs):
        from scripts.trl.mc_blame import compute_blame_rewards_batch

        completion_texts = [normalize_completion(c) for c in completions]
        base_correct = []
        for ct, ans in zip(completion_texts, gold_answer):
            try:
                base_correct.append(bool(compute_score(ct, ground_truth=ans)))
            except Exception:
                base_correct.append(False)

        return compute_blame_rewards_batch(
            prompts=prompts,
            completions=completion_texts,
            gold_answers=gold_answer,
            base_correct=base_correct,
            model_path=self.model_path,
            beta=self.beta,
            max_new_tokens=self.max_new_tokens,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
