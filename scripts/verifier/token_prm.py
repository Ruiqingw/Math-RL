"""
Utilities for an OpenAI-style token-prediction PRM.

This approximates the paper's setup more closely than the classifier-head
verifier:

- no extra classification head
- the LM predicts a single supervision token at the end of the verifier prompt
- optional first-error-only supervision to mimic the PRM paper's prefix labels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier_prompt import format_verifier_prompt


LABEL_TOKEN_CANDIDATES: Sequence[Tuple[str, str]] = (
    ("+", "-"),
    (" +", " -"),
    (" correct", " incorrect"),
    (" Yes", " No"),
    (" positive", " negative"),
)


@dataclass(frozen=True)
class LabelTokenPair:
    positive_text: str
    negative_text: str
    positive_id: int
    negative_id: int


def select_label_token_pair(
    tokenizer: AutoTokenizer,
    candidates: Sequence[Tuple[str, str]] = LABEL_TOKEN_CANDIDATES,
) -> LabelTokenPair:
    """
    Find a stable single-token positive/negative label pair for the current tokenizer.
    """
    for positive_text, negative_text in candidates:
        positive_ids = tokenizer.encode(positive_text, add_special_tokens=False)
        negative_ids = tokenizer.encode(negative_text, add_special_tokens=False)
        if len(positive_ids) == 1 and len(negative_ids) == 1 and positive_ids[0] != negative_ids[0]:
            return LabelTokenPair(
                positive_text=positive_text,
                negative_text=negative_text,
                positive_id=positive_ids[0],
                negative_id=negative_ids[0],
            )

    tried = ", ".join(f"{pos!r}/{neg!r}" for pos, neg in candidates)
    raise ValueError(
        "Could not find a single-token positive/negative label pair. "
        f"Tried: {tried}"
    )


def _truncate_to_first_negative(
    steps: Sequence[str],
    labels: Sequence[Any],
    stop_at_first_negative: bool,
) -> Tuple[List[str], List[Any]]:
    if not stop_at_first_negative:
        return list(steps), list(labels)

    for idx, label in enumerate(labels):
        if not bool(label):
            return list(steps[: idx + 1]), list(labels[: idx + 1])
    return list(steps), list(labels)


class TokenPRMDataset(Dataset):
    """
    Convert PRM step data into token-prediction rows.

    Each example is a verifier prompt whose final supervision token is the
    model target:
        positive step -> positive label token
        negative step -> negative label token
    """

    def __init__(
        self,
        hf_split,
        tokenizer: AutoTokenizer,
        label_tokens: LabelTokenPair,
        max_length: int,
        max_rows: Optional[int] = None,
        stop_at_first_negative: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.label_tokens = label_tokens
        self.max_length = max_length
        self.stop_at_first_negative = stop_at_first_negative
        self.examples: List[Dict[str, Any]] = []
        self.sample_labels: List[int] = []  # 0=pos, 1=neg

        n_rows = 0
        for row in hf_split:
            n_rows += 1
            problem = row["prompt"]
            steps = row["completions"]
            labels = row["labels"]
            steps, labels = _truncate_to_first_negative(steps, labels, stop_at_first_negative)

            for k, label in enumerate(labels):
                is_positive = bool(label)
                prompt_text = format_verifier_prompt(problem, steps[: k + 1])
                target_token_id = (
                    label_tokens.positive_id if is_positive else label_tokens.negative_id
                )
                class_label = 0 if is_positive else 1
                self.examples.append(
                    {
                        "problem": problem,
                        "prompt": prompt_text,
                        "current_step": steps[k],
                        "n_steps_in_prompt": k + 1,
                        "label": class_label,
                        "target_token_id": target_token_id,
                    }
                )
                self.sample_labels.append(class_label)

            if max_rows and n_rows >= max_rows:
                break

        self.n_pos = sum(1 for label in self.sample_labels if label == 0)
        self.n_neg = len(self.sample_labels) - self.n_pos

    def __len__(self) -> int:
        return len(self.examples)

    def __getitems__(self, indices):
        return [self.__getitem__(i) for i in indices]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt_ids = self.tokenizer.encode(
            ex["prompt"],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length - 1,
        )
        prompt_ids = prompt_ids[-(self.max_length - 1) :]

        input_ids = prompt_ids + [ex["target_token_id"]]
        labels = [-100] * len(prompt_ids) + [ex["target_token_id"]]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def prompt_debug_row(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        raw_prompt_ids = self.tokenizer.encode(
            ex["prompt"],
            add_special_tokens=True,
            truncation=False,
        )
        truncated_prompt_ids = self.tokenizer.encode(
            ex["prompt"],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length - 1,
        )
        prompt_len = len(truncated_prompt_ids)
        raw_prompt_len = len(raw_prompt_ids)
        model_input_ids = truncated_prompt_ids + [ex["target_token_id"]]
        return {
            "dataset_idx": idx,
            "problem": ex["problem"],
            "prompt_text": ex["prompt"],
            "current_step": ex["current_step"],
            "label": ex["label"],
            "label_name": "positive" if ex["label"] == 0 else "negative",
            "n_steps_in_prompt": ex["n_steps_in_prompt"],
            "prompt_len_tokens": prompt_len,
            "raw_prompt_len_tokens": raw_prompt_len,
            "dropped_tokens": max(raw_prompt_len - prompt_len, 0),
            "truncated": raw_prompt_len > (self.max_length - 1),
            "model_input_text": self.tokenizer.decode(model_input_ids, skip_special_tokens=True),
        }

    def sampled_prompt_stats(self, sample_size: int, seed: int) -> Dict[str, float]:
        n_examples = len(self.examples)
        if n_examples == 0:
            return {
                "prompt_len_mean": 0.0,
                "prompt_len_p95": 0.0,
                "truncation_rate": 0.0,
                "sample_size": 0,
            }

        sample_size = min(sample_size, n_examples)
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_examples, size=sample_size, replace=False).tolist()
        lengths = []
        truncated = 0
        for idx in indices:
            row = self.prompt_debug_row(int(idx))
            lengths.append(row["raw_prompt_len_tokens"])
            truncated += int(row["truncated"])

        lengths_np = np.asarray(lengths, dtype=np.float32)
        return {
            "prompt_len_mean": float(lengths_np.mean()),
            "prompt_len_p95": float(np.percentile(lengths_np, 95)),
            "truncation_rate": float(truncated / sample_size),
            "sample_size": sample_size,
        }


class PadCollator:
    """Right-pad token-PRM batches."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [feature["input_ids"] for feature in features]
        labels_list = [feature["labels"] for feature in features]
        max_len = max(ids.size(0) for ids in input_ids_list)

        padded_ids = []
        padded_labels = []
        attention_mask = []
        for input_ids, labels in zip(input_ids_list, labels_list):
            pad_len = max_len - input_ids.size(0)
            padded_ids.append(F.pad(input_ids, (0, pad_len), value=self.pad_token_id))
            padded_labels.append(F.pad(labels, (0, pad_len), value=-100))
            attention_mask.append(F.pad(torch.ones_like(input_ids), (0, pad_len), value=0))

        return {
            "input_ids": torch.stack(padded_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(attention_mask),
        }


def target_positions_from_labels(labels: torch.Tensor) -> torch.Tensor:
    target_mask = labels != -100
    if target_mask.ndim != 2:
        raise ValueError(f"Expected 2D labels tensor, got shape={tuple(labels.shape)}")
    if not torch.all(target_mask.sum(dim=1) == 1):
        raise ValueError("Each example must contain exactly one supervised token")
    return target_mask.to(torch.int64).argmax(dim=1)


def binary_classes_from_labels(labels: torch.Tensor, label_tokens: LabelTokenPair) -> torch.Tensor:
    target_positions = target_positions_from_labels(labels)
    batch_idx = torch.arange(labels.size(0), device=labels.device)
    target_ids = labels[batch_idx, target_positions]
    return torch.where(
        target_ids == label_tokens.positive_id,
        torch.zeros_like(target_ids),
        torch.ones_like(target_ids),
    )


def pair_logits_from_causal_lm_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_tokens: LabelTokenPair,
) -> torch.Tensor:
    """
    Extract [positive, negative] logits for the single supervision token.
    """
    target_positions = target_positions_from_labels(labels)
    context_positions = torch.clamp(target_positions - 1, min=0)
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    next_token_logits = logits[batch_idx, context_positions]
    return torch.stack(
        [
            next_token_logits[:, label_tokens.positive_id],
            next_token_logits[:, label_tokens.negative_id],
        ],
        dim=-1,
    )


def load_token_prm(
    model_name_or_path: str,
    *,
    device_map: Optional[str] = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, LabelTokenPair]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    label_tokens = select_label_token_pair(tokenizer)

    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    model.eval()
    return model, tokenizer, label_tokens


@torch.no_grad()
def score_step_positive_probs(
    problem: str,
    steps: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    label_tokens: LabelTokenPair,
    *,
    device: str = "cuda",
    max_length: int = 1536,
    batch_size: int = 4,
) -> List[float]:
    """
    Score each step with the token-prediction PRM.

    The positive score is computed by taking the next-token logits at the end of
    the verifier prompt and normalizing over the {positive, negative} label tokens.
    """
    prompts = [format_verifier_prompt(problem, steps[: k + 1]) for k in range(len(steps))]
    positive_probs: List[float] = []

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        last_token_indices = enc["attention_mask"].sum(dim=1) - 1
        batch_idx = torch.arange(enc["input_ids"].size(0), device=enc["input_ids"].device)
        next_token_logits = outputs.logits[batch_idx, last_token_indices]
        pair_logits = torch.stack(
            [
                next_token_logits[:, label_tokens.positive_id],
                next_token_logits[:, label_tokens.negative_id],
            ],
            dim=-1,
        )
        probs = torch.softmax(pair_logits, dim=-1)[:, 0]
        positive_probs.extend(probs.float().cpu().tolist())

    return positive_probs
