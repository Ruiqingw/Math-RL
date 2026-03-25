#!/usr/bin/env python3
"""
Custom verl reward function: built-in MATH outcome reward + verifier shaping.

Designed for a clean A/B comparison:
  - baseline GRPO uses verl's ordinary MATH rule reward
  - verifier GRPO keeps the same base reward, then adds a small shaping term

This avoids changing the main reward target while testing whether the verifier
provides useful auxiliary signal.
"""

import os
import sys
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from verl.utils.reward_score.math_reward import compute_score as math_rule_score


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
VERIFIER_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "verifier")
if VERIFIER_DIR not in sys.path:
    sys.path.insert(0, VERIFIER_DIR)

from reward_fn import PRMClassifier, score_steps  # noqa: E402
from step_splitter import split_into_steps, extract_boxed_answer  # noqa: E402


_CACHED_VERIFIER: Dict[str, Any] = {
    "key": None,
    "model": None,
    "tokenizer": None,
}


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    value = _env_str(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = _env_str(name)
    return float(value) if value is not None else default


def _resolve_verifier_config(
    verifier_model_path: Optional[str],
    verifier_device: Optional[str],
    verifier_max_length: Optional[int],
    verifier_batch_size: Optional[int],
    beta: Optional[float],
    delta: Optional[float],
    correct_threshold: Optional[float],
) -> Dict[str, Any]:
    return {
        "verifier_model_path": verifier_model_path or _env_str("VERIFIER_MODEL_PATH"),
        "verifier_device": verifier_device or _env_str("VERIFIER_DEVICE", "cuda"),
        "verifier_max_length": verifier_max_length if verifier_max_length is not None else _env_int("VERIFIER_MAX_LENGTH", 1024),
        "verifier_batch_size": verifier_batch_size if verifier_batch_size is not None else _env_int("VERIFIER_BATCH_SIZE", 4),
        "beta": beta if beta is not None else _env_float("VERIFIER_BETA", 0.1),
        "delta": delta if delta is not None else _env_float("VERIFIER_DELTA", 0.05),
        "correct_threshold": correct_threshold if correct_threshold is not None else _env_float("VERIFIER_THRESHOLD", 0.5),
    }


def _get_problem(extra_info: Optional[dict]) -> Optional[str]:
    if not extra_info:
        return None
    for key in ("question", "problem", "question_raw"):
        value = extra_info.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _load_verifier(verifier_model_path: str, verifier_device: str):
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


def _verifier_shaping(
    problem: str,
    solution_str: str,
    verifier_model_path: str,
    verifier_device: str = "cuda",
    verifier_max_length: int = 1024,
    verifier_batch_size: int = 4,
    beta: float = 0.1,
    delta: float = 0.05,
    correct_threshold: float = 0.5,
) -> Dict[str, Any]:
    steps = split_into_steps(solution_str)
    if not steps:
        return {
            "shaping_score": 0.0,
            "r_avg_step": 0.0,
            "r_first_error": 0.0,
            "n_steps": 0,
            "first_bad_idx": -1,
        }

    model, tokenizer = _load_verifier(verifier_model_path, verifier_device)
    step_scores = score_steps(
        problem=problem,
        steps=steps,
        model=model,
        tokenizer=tokenizer,
        device=verifier_device,
        max_length=verifier_max_length,
        batch_size=verifier_batch_size,
    )

    r_avg_step = sum(step_scores) / len(step_scores)

    first_bad_idx = -1
    r_first_error = 0.0
    for idx, score in enumerate(step_scores):
        if score < correct_threshold:
            first_bad_idx = idx
            r_first_error = 1.0 - (idx / len(step_scores))
            break

    shaping_score = beta * r_avg_step - delta * r_first_error
    return {
        "shaping_score": float(shaping_score),
        "r_avg_step": float(r_avg_step),
        "r_first_error": float(r_first_error),
        "n_steps": int(len(step_scores)),
        "first_bad_idx": int(first_bad_idx),
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    verifier_model_path: Optional[str] = None,
    verifier_device: Optional[str] = None,
    verifier_max_length: Optional[int] = None,
    verifier_batch_size: Optional[int] = None,
    beta: Optional[float] = None,
    delta: Optional[float] = None,
    correct_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    verifier_cfg = _resolve_verifier_config(
        verifier_model_path=verifier_model_path,
        verifier_device=verifier_device,
        verifier_max_length=verifier_max_length,
        verifier_batch_size=verifier_batch_size,
        beta=beta,
        delta=delta,
        correct_threshold=correct_threshold,
    )

    base_score = float(math_rule_score(solution_str, ground_truth))
    predicted_answer = extract_boxed_answer(solution_str)
    problem = _get_problem(extra_info)

    if not verifier_cfg["verifier_model_path"] or not problem:
        return {
            "score": base_score,
            "base_score": base_score,
            "shaping_score": 0.0,
            "r_avg_step": 0.0,
            "r_first_error": 0.0,
            "n_steps": 0,
            "first_bad_idx": -1,
            "verifier_used": 0.0,
            "predicted_answer": predicted_answer,
        }

    shaping = _verifier_shaping(
        problem=problem,
        solution_str=solution_str,
        verifier_model_path=verifier_cfg["verifier_model_path"],
        verifier_device=verifier_cfg["verifier_device"],
        verifier_max_length=verifier_cfg["verifier_max_length"],
        verifier_batch_size=verifier_cfg["verifier_batch_size"],
        beta=verifier_cfg["beta"],
        delta=verifier_cfg["delta"],
        correct_threshold=verifier_cfg["correct_threshold"],
    )

    return {
        "score": float(base_score + shaping["shaping_score"]),
        "base_score": base_score,
        "shaping_score": shaping["shaping_score"],
        "r_avg_step": shaping["r_avg_step"],
        "r_first_error": shaping["r_first_error"],
        "n_steps": shaping["n_steps"],
        "first_bad_idx": shaping["first_bad_idx"],
        "verifier_used": 1.0,
        "predicted_answer": predicted_answer,
    }


def compute_score_batched(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[str],
    extra_infos: List[Optional[dict]],
    verifier_model_path: Optional[str] = None,
    verifier_device: Optional[str] = None,
    verifier_max_length: Optional[int] = None,
    verifier_batch_size: Optional[int] = None,
    beta: Optional[float] = None,
    delta: Optional[float] = None,
    correct_threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    outputs = []
    for data_source, solution_str, ground_truth, extra_info in zip(
        data_sources, solution_strs, ground_truths, extra_infos, strict=True
    ):
        outputs.append(
            compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                verifier_model_path=verifier_model_path,
                verifier_device=verifier_device,
                verifier_max_length=verifier_max_length,
                verifier_batch_size=verifier_batch_size,
                beta=beta,
                delta=delta,
                correct_threshold=correct_threshold,
            )
        )
    return outputs
