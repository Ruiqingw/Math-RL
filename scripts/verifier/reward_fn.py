"""
reward_fn.py — Composite reward for Verifier-Guided GRPO.

Reward formula:
    R = alpha * R_final + beta * R_avg_step - delta * R_first_error

Where:
    R_final     = +1 if boxed answer matches gold, -1 otherwise
    R_avg_step  = mean of per-step verifier scores in [0, 1]
    R_first_error = 1 - (first_bad_step_idx / n_steps), so earlier errors
                    give a larger penalty (0 if all steps are good)

Verifier scoring uses a classification head approach:
    - Load base CausalLM + Linear(hidden_size, 2) classification head
    - For each step, extract hidden state at last prompt token position
    - Score = softmax(cls_logits)[0]  (probability of positive class)
"""

import os
import sys
import re
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from step_splitter import split_into_steps, extract_boxed_answer
from verifier_prompt import format_verifier_prompt


# — Classification Head (must match train_verifier.py) ————————————

class PRMClassifier(nn.Module):
    """Inference-only wrapper: base LM + classification head."""

    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.score = nn.Linear(hidden_size, 2, bias=False)
        self.config = base_model.config

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cuda", **kwargs):
        """
        Load a trained PRMClassifier from a directory containing:
          - Base model files (config.json, model.safetensors, etc.)
          - cls_head.pt (classification head state dict)
        """
        import os
        if device == "cuda":
            hf_device_map = {"": 0}
        elif device.startswith("cuda:"):
            hf_device_map = {"": int(device.split(":", 1)[1])}
        elif device in {"auto", "balanced", "balanced_low_0", "sequential"}:
            hf_device_map = device
        else:
            hf_device_map = {"": device}

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=hf_device_map,
        )
        hidden_size = base_model.config.hidden_size
        model = cls(base_model, hidden_size)

        # Load classification head
        head_path = os.path.join(model_path, "cls_head.pt")
        if os.path.exists(head_path):
            map_location = device if device not in {"auto", "balanced", "balanced_low_0", "sequential"} else "cpu"
            head_state = torch.load(head_path, map_location=map_location)
            model.score.load_state_dict(head_state)
        else:
            raise FileNotFoundError(f"Classification head not found: {head_path}")

        if device in {"auto", "balanced", "balanced_low_0", "sequential"}:
            score_device = next(base_model.parameters()).device
        else:
            score_device = torch.device(device)
        model.score = model.score.to(device=score_device, dtype=torch.bfloat16)
        model.eval()
        return model


# — Per-step verifier scoring ————————————————————————

@torch.no_grad()
def score_steps(
    problem: str,
    steps: List[str],
    model: PRMClassifier,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
    max_length: int = 1024,
    batch_size: int = 4,
) -> List[float]:
    """
    Score each step using the verifier (classification head method).

    For step k, builds a prompt with prior steps separated from the current
    step, feeds it to the model, extracts the hidden state at the last prompt
    token, and passes it through the classification head.

    Score = softmax(cls_logits)[0]  (probability of positive class) in (0, 1).

    Args:
        problem:     The math problem string.
        steps:       List of step strings (already split).
        model:       PRMClassifier with base model + classification head.
        tokenizer:   Corresponding tokenizer.
        device:      'cuda' or 'cpu'.
        max_length:  Max token length for verifier input.
        batch_size:  Number of steps to score in parallel.

    Returns:
        List of float scores in (0, 1), one per step.
    """
    prompts = [
        format_verifier_prompt(problem, steps[: k + 1])
        for k in range(len(steps))
    ]

    all_scores = []

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]

        # Tokenize batch — prompts only, no answer token appended
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        # Forward through base model to get hidden states
        outputs = model.base_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (B, L, H)

        # Last real token position for each sample in the batch
        last_token_indices = enc["attention_mask"].sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)

        # Extract hidden state at last prompt token
        last_hidden = hidden_states[batch_idx, last_token_indices]  # (B, H)

        # Classification head
        cls_logits = model.score(last_hidden)  # (B, 2)
        probs = torch.softmax(cls_logits, dim=-1)

        # Score = probability of positive class (index 0)
        for i in range(len(batch_prompts)):
            score = probs[i, 0].item()
            all_scores.append(score)

    return all_scores


# — Composite reward ——————————————————————————————

def compute_reward(
    problem: str,
    solution: str,
    gold_answer: str,
    model: PRMClassifier,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
    alpha: float = 1.0,
    beta: float = 0.3,
    delta: float = 0.1,
    correct_threshold: float = 0.5,
) -> Tuple[float, Dict]:
    """
    Compute the composite reward for one (problem, solution) pair.

    Args:
        problem:           The math problem text.
        solution:          The model-generated solution text.
        gold_answer:       The reference answer string (from \\boxed{}).
        model, tokenizer:  Trained verifier model.
        device:            Inference device.
        alpha, beta, delta: Reward weight hyperparameters.
        correct_threshold: Verifier score below this = step is "wrong".

    Returns:
        (total_reward, info_dict)
    """
    # — 1. Final answer reward ————————————————————
    predicted = extract_boxed_answer(solution)
    r_final  = 1.0 if (predicted and predicted == gold_answer.strip()) else -1.0

    # — 2. Step-level verifier scores ——————————————
    steps = split_into_steps(solution)
    if not steps:
        return r_final, {
            "r_final": r_final, "r_avg_step": 0.0, "r_first_error": 0.0,
            "step_scores": [], "n_steps": 0, "predicted_answer": predicted,
        }

    step_scores = score_steps(problem, steps, model, tokenizer, device)

    # — 3. Avg step reward ————————————————————————
    r_avg_step = sum(step_scores) / len(step_scores)

    # — 4. First-error penalty ————————————————————
    first_bad_idx = None
    for idx, sc in enumerate(step_scores):
        if sc < correct_threshold:
            first_bad_idx = idx
            break

    if first_bad_idx is not None:
        r_first_error = 1.0 - (first_bad_idx / len(steps))
    else:
        r_first_error = 0.0   # no error found -> no penalty

    # — 4. Composite ——————————————————————————————
    total = alpha * r_final + beta * r_avg_step - delta * r_first_error

    info = {
        "r_final":          r_final,
        "r_avg_step":       r_avg_step,
        "r_first_error":    r_first_error,
        "step_scores":      step_scores,
        "n_steps":          len(steps),
        "predicted_answer": predicted,
    }
    return total, info


# — Batch reward (for GRPO rollouts) —————————————————

def compute_rewards_batch(
    problems: List[str],
    solutions: List[str],
    gold_answers: List[str],
    model: PRMClassifier,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
    **kwargs,
) -> Tuple[List[float], List[Dict]]:
    """
    Compute rewards for a batch of (problem, solution, gold_answer) triples.
    Used during GRPO rollout evaluation.

    Returns:
        rewards: List[float]
        infos:   List[Dict]
    """
    rewards, infos = [], []
    for problem, solution, gold in zip(problems, solutions, gold_answers):
        r, info = compute_reward(
            problem, solution, gold, model, tokenizer, device, **kwargs
        )
        rewards.append(r)
        infos.append(info)
    return rewards, infos
