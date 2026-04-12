#!/usr/bin/env python3
"""
Monte Carlo blame assignment via greedy completion from step prefixes.

Core logic:
  For a wrong rollout with N steps, binary-search for the earliest step k
  such that greedy-completing from prefix[:k] still gets the wrong answer.
  That step is the "blame step" — the first point where the reasoning
  derailed beyond recovery.

Also provides the reward formula:
  correct rollout  → reward = 1.0
  wrong rollout    → reward = -beta * (total_steps - blame_step) / total_steps
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERIFIER_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "verifier")
if VERIFIER_DIR not in sys.path:
    sys.path.insert(0, VERIFIER_DIR)

from step_splitter import split_into_steps  # noqa: E402


_CACHED_BLAME_MODEL: dict[str, Any] = {
    "key": None,
    "model": None,
    "tokenizer": None,
}


def _load_blame_model(model_path: str, device: str = "cuda"):
    cache_key = f"{model_path}::{device}"
    if _CACHED_BLAME_MODEL["key"] == cache_key:
        return _CACHED_BLAME_MODEL["model"], _CACHED_BLAME_MODEL["tokenizer"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    _CACHED_BLAME_MODEL["key"] = cache_key
    _CACHED_BLAME_MODEL["model"] = model
    _CACHED_BLAME_MODEL["tokenizer"] = tokenizer
    return model, tokenizer


def greedy_complete(
    model,
    tokenizer,
    prefix: str,
    max_new_tokens: int = 512,
) -> str:
    inputs = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def check_answer(text: str, gold_answer: str) -> bool:
    from verl.utils.reward_score.math_reward import compute_score
    try:
        return compute_score(text, ground_truth=gold_answer) > 0.0
    except Exception:
        return False


def find_blame_step(
    model,
    tokenizer,
    prompt: str,
    steps: list[str],
    gold_answer: str,
    max_new_tokens: int = 512,
) -> int:
    """Binary search for the earliest step from which greedy completion fails.

    Returns the 1-indexed blame step (1 = first step is already wrong).
    If greedy completion from the full prompt (no steps) also fails,
    returns 1.  If completion from all steps somehow succeeds, returns
    len(steps) (last step blamed by default).
    """
    n = len(steps)
    if n == 0:
        return 1

    def _can_recover(k: int) -> bool:
        """Check if greedy completing from prefix of first k steps gets it right."""
        if k == 0:
            prefix = prompt
        else:
            prefix = prompt + "\n\n" + "\n\n".join(steps[:k])
        completion = greedy_complete(model, tokenizer, prefix, max_new_tokens)
        full_text = prefix + completion
        return check_answer(full_text, gold_answer)

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_recover(mid):
            lo = mid + 1
        else:
            hi = mid

    blame = lo + 1
    return min(blame, n)


def blame_reward(
    blame_step: int,
    total_steps: int,
    beta: float = 0.5,
) -> float:
    if total_steps <= 1:
        return -beta
    return -beta * (total_steps - blame_step) / total_steps


def compute_blame_rewards(
    prompts: list[str],
    completions: list[str],
    gold_answers: list[str],
    base_correct: list[bool],
    model_path: str,
    device: str = "cuda",
    beta: float = 0.5,
    max_new_tokens: int = 512,
) -> tuple[list[float], list[dict]]:
    """Compute blame-based rewards for a batch.

    Returns:
        rewards: list of floats (1.0 for correct, negative for wrong)
        diagnostics: list of dicts with blame details for each wrong rollout
    """
    model, tokenizer = _load_blame_model(model_path, device)

    rewards: list[float] = []
    diagnostics: list[dict] = []

    for i, (prompt, completion, gold, correct) in enumerate(
        zip(prompts, completions, gold_answers, base_correct)
    ):
        if correct:
            rewards.append(1.0)
            diagnostics.append({"correct": True})
            continue

        steps = split_into_steps(completion)
        n_steps = len(steps)

        blame = find_blame_step(
            model, tokenizer, prompt, steps, gold,
            max_new_tokens=max_new_tokens,
        )
        reward = blame_reward(blame, n_steps, beta=beta)
        rewards.append(reward)
        diagnostics.append({
            "correct": False,
            "n_steps": n_steps,
            "blame_step": blame,
            "reward": reward,
            "blame_frac": blame / n_steps if n_steps > 0 else 0.0,
        })

    return rewards, diagnostics
