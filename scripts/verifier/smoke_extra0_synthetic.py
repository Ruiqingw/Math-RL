#!/usr/bin/env python3
"""Tiny synthetic smoke test for the extra0 token-classification protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.models.qwen2 import Qwen2Config


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from qwen_extra0_prm import (  # noqa: E402
    IGNORE_INDEX,
    build_extra0_token_classification_encoding,
    ensure_extra0_token,
    resize_token_embeddings_for_extra0,
    score_steps,
)


DEFAULT_TOKENIZER_PATH = "models/Qwen2.5-Math-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def build_tiny_qwen2_token_classifier(vocab_size: int, pad_token_id: int | None) -> AutoModelForTokenClassification:
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        num_labels=2,
        pad_token_id=pad_token_id,
        tie_word_embeddings=False,
    )
    return AutoModelForTokenClassification.from_config(config)


def main() -> None:
    args = parse_args()
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    raw_extra0_id = tokenizer.convert_tokens_to_ids("<extra_0>")
    extra0_was_added = raw_extra0_id is None or raw_extra0_id == tokenizer.unk_token_id
    extra0_token_id = ensure_extra0_token(tokenizer)

    problem = "Solve for x: x + 1 = 3."
    steps = [
        "1. Subtract 1 from both sides.",
        "2. This gives x = 2.",
        "Therefore the answer is \\boxed{2}.",
    ]
    step_labels = [1, 1, 1]
    encoding = build_extra0_token_classification_encoding(
        tokenizer,
        problem,
        steps,
        step_labels,
        max_length=args.max_length,
        extra0_token_id=extra0_token_id,
    )

    label_positions = [idx for idx, label in enumerate(encoding.labels) if label != IGNORE_INDEX]
    labels_at_positions = [encoding.labels[idx] for idx in encoding.extra0_positions]
    token_ids_at_positions = [encoding.input_ids[idx] for idx in encoding.extra0_positions]

    assert extra0_token_id != tokenizer.unk_token_id
    assert encoding.extra0_positions == label_positions
    assert labels_at_positions == step_labels
    assert all(token_id == extra0_token_id for token_id in token_ids_at_positions)

    model = build_tiny_qwen2_token_classifier(len(tokenizer), tokenizer.pad_token_id)
    resize_token_embeddings_for_extra0(model, tokenizer, extra0_token_id)
    model.eval()
    scores = score_steps(
        problem,
        steps,
        model,
        tokenizer,
        device="cpu",
        max_length=args.max_length,
        require_all_steps=True,
    )
    assert len(scores) == len(steps)

    print(
        json.dumps(
            {
                "tokenizer_path": args.tokenizer_path,
                "extra0_token_id": extra0_token_id,
                "extra0_was_added": extra0_was_added,
                "step_count": len(steps),
                "extra0_position_count": len(encoding.extra0_positions),
                "label_positions_match": encoding.extra0_positions == label_positions,
                "score_count": len(scores),
                "scores": scores,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
