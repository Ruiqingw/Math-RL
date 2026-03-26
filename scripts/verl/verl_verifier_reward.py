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
import time
import traceback
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

_DEBUG_CALL_COUNT = 0


def _debug_enabled() -> bool:
    value = os.environ.get("VERIFIER_DEBUG", "0").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _debug_log(message: str) -> None:
    if not _debug_enabled():
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    pid = os.getpid()
    line = f"[verl_verifier_reward {ts} pid={pid}] {message}"
    print(line, file=sys.stderr, flush=True)
    debug_path = os.environ.get("VERIFIER_DEBUG_LOG")
    if debug_path:
        try:
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


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
        _debug_log(f"reusing cached verifier for key={cache_key}")
        return _CACHED_VERIFIER["model"], _CACHED_VERIFIER["tokenizer"]

    t0 = time.time()
    _debug_log(
        f"loading verifier start path={verifier_model_path} device={verifier_device}"
    )
    tokenizer = AutoTokenizer.from_pretrained(verifier_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PRMClassifier.from_pretrained(verifier_model_path, device=verifier_device)

    _CACHED_VERIFIER["key"] = cache_key
    _CACHED_VERIFIER["model"] = model
    _CACHED_VERIFIER["tokenizer"] = tokenizer
    _debug_log(
        f"loading verifier done path={verifier_model_path} device={verifier_device} elapsed={time.time() - t0:.2f}s"
    )
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
    t0 = time.time()
    steps = split_into_steps(solution_str)
    _debug_log(
        f"verifier_shaping start steps={len(steps)} device={verifier_device} max_length={verifier_max_length} batch_size={verifier_batch_size}"
    )
    if not steps:
        _debug_log("verifier_shaping early-exit: no steps parsed")
        return {
            "shaping_score": 0.0,
            "r_avg_step": 0.0,
            "r_first_error": 0.0,
            "n_steps": 0,
            "first_bad_idx": -1,
        }

    model, tokenizer = _load_verifier(verifier_model_path, verifier_device)
    score_t0 = time.time()
    _debug_log(f"score_steps start n_steps={len(steps)}")
    step_scores = score_steps(
        problem=problem,
        steps=steps,
        model=model,
        tokenizer=tokenizer,
        device=verifier_device,
        max_length=verifier_max_length,
        batch_size=verifier_batch_size,
    )
    _debug_log(
        f"score_steps done n_steps={len(steps)} elapsed={time.time() - score_t0:.2f}s"
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
    _debug_log(
        "verifier_shaping done "
        f"n_steps={len(step_scores)} first_bad_idx={first_bad_idx} "
        f"r_avg_step={r_avg_step:.4f} r_first_error={r_first_error:.4f} "
        f"elapsed={time.time() - t0:.2f}s"
    )
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
    global _DEBUG_CALL_COUNT
    _DEBUG_CALL_COUNT += 1
    t0 = time.time()
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
    _debug_log(
        f"compute_score start call={_DEBUG_CALL_COUNT} data_source={data_source} problem_present={bool(problem)} solution_chars={len(solution_str)}"
    )

    if not verifier_cfg["verifier_model_path"] or not problem:
        _debug_log(
            f"compute_score fallback call={_DEBUG_CALL_COUNT} verifier_path_present={bool(verifier_cfg['verifier_model_path'])} problem_present={bool(problem)}"
        )
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

    try:
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
    except Exception as e:
        _debug_log(
            f"compute_score exception call={_DEBUG_CALL_COUNT} type={type(e).__name__} msg={e}\n{traceback.format_exc()}"
        )
        raise

    result = {
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
    _debug_log(
        f"compute_score done call={_DEBUG_CALL_COUNT} score={result['score']:.4f} base={base_score:.4f} shaping={shaping['shaping_score']:.4f} elapsed={time.time() - t0:.2f}s"
    )
    return result


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
