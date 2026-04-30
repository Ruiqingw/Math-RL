"""
Shared utilities for Qwen-style ``<extra_0>`` token-classification PRMs.

The main protocol is:

- append ``<extra_0>`` after each reasoning step
- train ``AutoModelForTokenClassification(num_labels=2)``
- compute loss only at the inserted ``<extra_0>`` token positions
- interpret class 1 as the positive/correct-step probability
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizerBase


EXTRA0_TOKEN = "<extra_0>"
IGNORE_INDEX = -100
NEGATIVE_LABEL = 0
POSITIVE_LABEL = 1


@dataclass(frozen=True)
class Extra0Encoding:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    extra0_positions: List[int]
    expected_extra0_count: int
    kept_extra0_count: int
    dropped_label_count: int
    truncated: bool

    def as_tensors(self) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.labels, dtype=torch.long),
        }


def normalize_step_label(label: Any) -> int:
    """
    Convert a raw step label into the token-classification class id.

    Boolean PRM800K-style labels map to ``1`` for positive and ``0`` for
    negative. Integer labels are accepted when already encoded as ``0`` or ``1``.
    """
    if isinstance(label, bool):
        return POSITIVE_LABEL if label else NEGATIVE_LABEL
    if isinstance(label, int) and label in (NEGATIVE_LABEL, POSITIVE_LABEL):
        return int(label)
    raise ValueError(f"Expected a bool or 0/1 step label, got {label!r}")


def resolve_extra0_token_id(
    tokenizer: PreTrainedTokenizerBase,
    extra0_token: str = EXTRA0_TOKEN,
) -> int:
    """
    Resolve the tokenizer id for ``<extra_0>`` and verify it is a single token.
    """
    token_id = tokenizer.convert_tokens_to_ids(extra0_token)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or token_id == unk_token_id:
        encoded = tokenizer.encode(extra0_token, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] != unk_token_id:
            token_id = int(encoded[0])
        else:
            raise ValueError(
                f"Tokenizer does not contain {extra0_token!r} as a single token. "
                f"Got encoded ids: {encoded}"
            )

    encoded = tokenizer.encode(extra0_token, add_special_tokens=False)
    if len(encoded) != 1 or int(encoded[0]) != int(token_id):
        raise ValueError(
            f"{extra0_token!r} must encode to exactly [{token_id}], got {encoded}"
        )
    return int(token_id)


def clean_step_text(step: str) -> str:
    text = str(step).strip()
    if not text:
        raise ValueError("Reasoning steps must be non-empty strings")
    return text


def format_solution_with_extra0(
    problem: str,
    steps: Sequence[str],
    *,
    extra0_token: str = EXTRA0_TOKEN,
) -> str:
    """
    Format one problem and full step list with ``<extra_0>`` after each step.
    """
    problem_text = str(problem).strip()
    if not problem_text:
        raise ValueError("Problem text must be non-empty")
    if not steps:
        raise ValueError("At least one reasoning step is required")

    step_lines = [f"{clean_step_text(step)} {extra0_token}" for step in steps]
    return problem_text + "\n\n" + "\n".join(step_lines)


def _extra0_positions(input_ids: Sequence[int], extra0_token_id: int) -> List[int]:
    return [idx for idx, token_id in enumerate(input_ids) if int(token_id) == extra0_token_id]


def _labels_for_kept_positions(
    labels: Sequence[Any],
    kept_count: int,
    *,
    truncation_side: str,
) -> List[int]:
    class_labels = [normalize_step_label(label) for label in labels]
    if kept_count > len(class_labels):
        raise ValueError(
            f"Found {kept_count} <extra_0> positions but only {len(class_labels)} labels"
        )
    if kept_count == len(class_labels):
        return class_labels
    if truncation_side == "left":
        return class_labels[-kept_count:] if kept_count else []
    return class_labels[:kept_count]


def build_extra0_token_classification_encoding(
    tokenizer: PreTrainedTokenizerBase,
    problem: str,
    steps: Sequence[str],
    step_labels: Sequence[Any],
    *,
    max_length: int = 1536,
    extra0_token: str = EXTRA0_TOKEN,
    extra0_token_id: Optional[int] = None,
) -> Extra0Encoding:
    """
    Tokenize one full solution and supervise only inserted ``<extra_0>`` tokens.
    """
    if len(steps) != len(step_labels):
        raise ValueError(f"steps/labels length mismatch: {len(steps)} vs {len(step_labels)}")

    token_id = (
        resolve_extra0_token_id(tokenizer, extra0_token)
        if extra0_token_id is None
        else int(extra0_token_id)
    )
    text = format_solution_with_extra0(problem, steps, extra0_token=extra0_token)
    raw_ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    input_ids = [int(token_id_) for token_id_ in encoding["input_ids"]]
    attention_mask = [int(mask) for mask in encoding.get("attention_mask", [1] * len(input_ids))]
    positions = _extra0_positions(input_ids, token_id)
    kept_labels = _labels_for_kept_positions(
        step_labels,
        len(positions),
        truncation_side=getattr(tokenizer, "truncation_side", "right"),
    )

    labels = [IGNORE_INDEX] * len(input_ids)
    for position, class_label in zip(positions, kept_labels):
        labels[position] = class_label

    expected_count = len(step_labels)
    kept_count = len(positions)
    return Extra0Encoding(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        extra0_positions=positions,
        expected_extra0_count=expected_count,
        kept_extra0_count=kept_count,
        dropped_label_count=max(expected_count - kept_count, 0),
        truncated=len(raw_ids) > len(input_ids),
    )


class Extra0PadCollator:
    """Right-pad extra0 token-classification examples."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [feature["input_ids"] for feature in features]
        attention_mask_list = [feature["attention_mask"] for feature in features]
        labels_list = [feature["labels"] for feature in features]
        max_len = max(input_ids.size(0) for input_ids in input_ids_list)

        padded_ids = []
        padded_attention = []
        padded_labels = []
        for input_ids, attention_mask, labels in zip(
            input_ids_list,
            attention_mask_list,
            labels_list,
        ):
            pad_len = max_len - input_ids.size(0)
            padded_ids.append(F.pad(input_ids, (0, pad_len), value=self.pad_token_id))
            padded_attention.append(F.pad(attention_mask, (0, pad_len), value=0))
            padded_labels.append(F.pad(labels, (0, pad_len), value=IGNORE_INDEX))

        return {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(padded_attention),
            "labels": torch.stack(padded_labels),
        }


def extra0_positions_from_labels(labels: torch.Tensor) -> List[List[int]]:
    if labels.ndim != 2:
        raise ValueError(f"Expected a 2D labels tensor, got shape={tuple(labels.shape)}")
    positions_by_row: List[List[int]] = []
    for row in labels:
        positions_by_row.append((row != IGNORE_INDEX).nonzero(as_tuple=False).flatten().tolist())
    return positions_by_row


def load_extra0_prm(
    model_name_or_path: str,
    *,
    device_map: Optional[str] = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[AutoModelForTokenClassification, AutoTokenizer, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    extra0_token_id = resolve_extra0_token_id(tokenizer)

    kwargs: Dict[str, Any] = {
        "num_labels": 2,
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if device_map is not None:
        kwargs["device_map"] = device_map
    model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, **kwargs)
    model.eval()
    return model, tokenizer, extra0_token_id


def _infer_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def score_steps(
    problem: str,
    steps: Sequence[str],
    model: AutoModelForTokenClassification,
    tokenizer: PreTrainedTokenizerBase,
    *,
    device: Optional[str | torch.device] = None,
    max_length: int = 1536,
    extra0_token: str = EXTRA0_TOKEN,
    require_all_steps: bool = True,
) -> List[float]:
    """
    Return positive probabilities at all kept ``<extra_0>`` positions.
    """
    if not steps:
        return []

    extra0_token_id = resolve_extra0_token_id(tokenizer, extra0_token)
    dummy_labels = [POSITIVE_LABEL] * len(steps)
    encoding = build_extra0_token_classification_encoding(
        tokenizer,
        problem,
        steps,
        dummy_labels,
        max_length=max_length,
        extra0_token=extra0_token,
        extra0_token_id=extra0_token_id,
    )
    if require_all_steps and encoding.kept_extra0_count != len(steps):
        raise ValueError(
            "Truncation dropped <extra_0> positions during scoring: "
            f"kept={encoding.kept_extra0_count}, expected={len(steps)}"
        )

    target_device = torch.device(device) if device is not None else _infer_model_device(model)
    input_ids = torch.tensor([encoding.input_ids], dtype=torch.long, device=target_device)
    attention_mask = torch.tensor(
        [encoding.attention_mask],
        dtype=torch.long,
        device=target_device,
    )
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    step_logits = outputs.logits[0, encoding.extra0_positions, :]
    return torch.softmax(step_logits, dim=-1)[:, POSITIVE_LABEL].float().cpu().tolist()

