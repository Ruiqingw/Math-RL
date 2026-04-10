#!/usr/bin/env python3
"""
Sweep decision thresholds for the token-prediction PRM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from token_prm import (
    PadCollator,
    TokenPRMDataset,
    binary_classes_from_labels,
    load_token_prm,
    pair_logits_from_causal_lm_logits,
)
from openai_prm_raw import (
    DEFAULT_RAW_DATA_DIR,
    build_raw_phase1_phase2_dataset,
    build_raw_phase2_dataset,
    phase1_phase2_cache_dir,
    phase2_cache_dir,
)


DEFAULT_MAX_LENGTH = 1536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune thresholds for a token-prediction PRM.")
    parser.add_argument("--model-path", required=True, help="Checkpoint directory.")
    parser.add_argument(
        "--dataset-source",
        default="raw_phase1_phase2",
        choices=["raw_phase2", "raw_phase1_phase2"],
        help="Which dataset view to evaluate on.",
    )
    parser.add_argument("--raw-data-dir", default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--cache-dir", default="")
    parser.add_argument(
        "--neutral-policy",
        default="nonnegative",
        choices=["nonnegative", "positive_only"],
        help="How to map rating=0 when converting {-1,0,+1} to binary labels.",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--min-threshold", type=float, default=0.0)
    parser.add_argument("--max-threshold", type=float, default=1.0)
    parser.add_argument("--num-thresholds", type=int, default=201)
    parser.add_argument(
        "--metric",
        default="balanced_accuracy",
        choices=["balanced_accuracy", "neg_f1", "accuracy", "neg_recall"],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", default="")
    parser.add_argument(
        "--all-steps",
        action="store_true",
        help="Disable first-error-only supervision reconstruction for the eval view.",
    )
    return parser.parse_args()


def load_eval_dataset(
    tokenizer,
    label_tokens,
    dataset_source: str,
    raw_data_dir: str,
    cache_dir: str,
    neutral_policy: str,
    split: str,
    max_length: int,
    all_steps: bool,
):
    if dataset_source == "raw_phase2":
        effective_cache_dir = cache_dir or phase2_cache_dir(
            neutral_policy=neutral_policy,
            stop_at_first_negative=not all_steps,
        )
        dataset = build_raw_phase2_dataset(
            raw_data_dir=raw_data_dir,
            cache_dir=effective_cache_dir,
            neutral_policy=neutral_policy,
            stop_at_first_negative=not all_steps,
        )
    elif dataset_source == "raw_phase1_phase2":
        effective_cache_dir = cache_dir or phase1_phase2_cache_dir(
            neutral_policy=neutral_policy,
            stop_at_first_negative=not all_steps,
        )
        dataset = build_raw_phase1_phase2_dataset(
            raw_data_dir=raw_data_dir,
            cache_dir=effective_cache_dir,
            neutral_policy=neutral_policy,
            stop_at_first_negative=not all_steps,
        )
    else:
        raise ValueError(f"Unsupported dataset source: {dataset_source}")
    hf_split = dataset[split]
    return TokenPRMDataset(
        hf_split,
        tokenizer,
        label_tokens,
        max_length=max_length,
        stop_at_first_negative=not all_steps,
    )


@torch.no_grad()
def collect_eval_scores(dataloader, model, label_tokens, tensor_device: torch.device) -> Dict[str, np.ndarray]:
    all_pos_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(tensor_device)
        attention_mask = batch["attention_mask"].to(tensor_device)
        labels = batch["labels"].to(tensor_device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pair_logits = pair_logits_from_causal_lm_logits(outputs.logits, labels, label_tokens)
        pos_probs = torch.softmax(pair_logits, dim=-1)[:, 0]
        true_cls = binary_classes_from_labels(labels, label_tokens)

        all_pos_probs.append(pos_probs.float().cpu())
        all_labels.append(true_cls.cpu())

    return {
        "pos_probs": torch.cat(all_pos_probs).numpy(),
        "labels": torch.cat(all_labels).numpy(),
    }


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def metrics_at_threshold(pos_probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    pred_cls = np.where(pos_probs >= threshold, 0, 1)

    pos_mask = labels == 0
    neg_mask = labels == 1
    pred_neg_mask = pred_cls == 1

    accuracy = float((pred_cls == labels).mean())
    pos_recall = safe_div(((pred_cls == 0) & pos_mask).sum(), pos_mask.sum())
    neg_recall = safe_div(((pred_cls == 1) & neg_mask).sum(), neg_mask.sum())
    balanced_accuracy = 0.5 * (pos_recall + neg_recall)
    neg_precision = safe_div(((pred_cls == 1) & neg_mask).sum(), pred_neg_mask.sum())
    neg_f1 = safe_div(2.0 * neg_precision * neg_recall, neg_precision + neg_recall)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "pos_recall": pos_recall,
        "neg_recall": neg_recall,
        "balanced_accuracy": balanced_accuracy,
        "neg_precision": neg_precision,
        "neg_f1": neg_f1,
        "pred_neg_fraction": float(pred_neg_mask.mean()),
    }


def select_best(metrics: List[Dict[str, float]], metric_name: str) -> Dict[str, float]:
    return max(metrics, key=lambda row: (row[metric_name], -abs(row["threshold"] - 0.5)))


def main() -> None:
    args = parse_args()

    load_device_map = None if args.device == "cpu" else "auto"
    model, tokenizer, label_tokens = load_token_prm(
        args.model_path,
        device_map=load_device_map,
    )
    tensor_device = torch.device(args.device)

    print(f"Loading token-PRM checkpoint from: {args.model_path}")
    print(f"Using label tokens: +={label_tokens.positive_text!r} -={label_tokens.negative_text!r}")

    eval_ds = load_eval_dataset(
        tokenizer=tokenizer,
        label_tokens=label_tokens,
        dataset_source=args.dataset_source,
        raw_data_dir=args.raw_data_dir,
        cache_dir=args.cache_dir,
        neutral_policy=args.neutral_policy,
        split=args.split,
        max_length=args.max_length,
        all_steps=args.all_steps,
    )
    dataloader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PadCollator(tokenizer.pad_token_id),
    )

    outputs = collect_eval_scores(dataloader, model, label_tokens, tensor_device)
    pos_probs = outputs["pos_probs"]
    labels = outputs["labels"]

    thresholds = np.linspace(args.min_threshold, args.max_threshold, args.num_thresholds)
    metrics = [metrics_at_threshold(pos_probs, labels, float(th)) for th in thresholds]
    best = select_best(metrics, args.metric)

    natural_neg_fraction = float((labels == 1).mean())
    print(f"\nEval examples: {len(labels):,}")
    print(f"Natural negative fraction: {natural_neg_fraction:.4f}")
    print(f"Selection metric: {args.metric}")
    print(f"\nRecommended threshold: {best['threshold']:.4f}")
    print(
        "Metrics at recommended threshold: "
        f"accuracy={best['accuracy']:.4f} "
        f"pos_recall={best['pos_recall']:.4f} "
        f"neg_recall={best['neg_recall']:.4f} "
        f"balanced_accuracy={best['balanced_accuracy']:.4f} "
        f"neg_precision={best['neg_precision']:.4f} "
        f"neg_f1={best['neg_f1']:.4f}"
    )

    ranked = sorted(metrics, key=lambda row: (row[args.metric], -abs(row["threshold"] - 0.5)), reverse=True)
    print(f"\nTop {min(args.top_k, len(ranked))} thresholds by {args.metric}:")
    for row in ranked[: args.top_k]:
        print(
            f"  th={row['threshold']:.4f} "
            f"bal_acc={row['balanced_accuracy']:.4f} "
            f"neg_f1={row['neg_f1']:.4f} "
            f"acc={row['accuracy']:.4f} "
            f"pos_recall={row['pos_recall']:.4f} "
            f"neg_recall={row['neg_recall']:.4f} "
            f"pred_neg={row['pred_neg_fraction']:.4f}"
        )

    if args.output_json:
        payload = {
            "model_path": args.model_path,
            "split": args.split,
            "selection_metric": args.metric,
            "recommended": best,
            "natural_negative_fraction": natural_neg_fraction,
            "results": metrics,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved full sweep to: {args.output_json}")

    print(
        "\nUse this threshold by passing "
        f"`VERIFIER_THRESHOLD={best['threshold']:.4f}` into the downstream reward pipeline."
    )


if __name__ == "__main__":
    main()
