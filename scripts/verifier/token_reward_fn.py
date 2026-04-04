"""
Reward/scoring helpers for a token-prediction PRM.

This mirrors the existing classifier-based reward_fn, but the verifier score is
computed from the LM's next-token probability over the {positive, negative}
label tokens.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from step_splitter import extract_boxed_answer, split_into_steps
from token_prm import LabelTokenPair, load_token_prm, score_step_positive_probs


@torch.no_grad()
def score_steps(
    problem: str,
    steps: List[str],
    model,
    tokenizer,
    label_tokens: LabelTokenPair,
    *,
    device: str = "cuda",
    max_length: int = 1536,
    batch_size: int = 4,
) -> List[float]:
    return score_step_positive_probs(
        problem,
        steps,
        model,
        tokenizer,
        label_tokens,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )


def compute_reward(
    problem: str,
    solution: str,
    gold_answer: str,
    model,
    tokenizer,
    label_tokens: LabelTokenPair,
    *,
    device: str = "cuda",
    alpha: float = 1.0,
    beta: float = 0.3,
    delta: float = 0.1,
    correct_threshold: float = 0.5,
    max_length: int = 1536,
) -> Tuple[float, Dict]:
    predicted = extract_boxed_answer(solution)
    r_final = 1.0 if (predicted and predicted == gold_answer.strip()) else -1.0

    steps = split_into_steps(solution)
    if not steps:
        return r_final, {
            "r_final": r_final,
            "r_avg_step": 0.0,
            "r_first_error": 0.0,
            "step_scores": [],
            "n_steps": 0,
            "predicted_answer": predicted,
        }

    step_scores = score_steps(
        problem,
        steps,
        model,
        tokenizer,
        label_tokens,
        device=device,
        max_length=max_length,
    )
    r_avg_step = sum(step_scores) / len(step_scores)

    first_bad_idx = None
    for idx, score in enumerate(step_scores):
        if score < correct_threshold:
            first_bad_idx = idx
            break

    if first_bad_idx is not None:
        r_first_error = 1.0 - (first_bad_idx / len(steps))
    else:
        r_first_error = 0.0

    total = alpha * r_final + beta * r_avg_step - delta * r_first_error
    return total, {
        "r_final": r_final,
        "r_avg_step": r_avg_step,
        "r_first_error": r_first_error,
        "step_scores": step_scores,
        "n_steps": len(steps),
        "predicted_answer": predicted,
    }


def load_model_bundle(
    model_path: str,
    *,
    device_map: Optional[str] = "auto",
) -> Tuple[object, object, LabelTokenPair]:
    return load_token_prm(model_path, device_map=device_map)
