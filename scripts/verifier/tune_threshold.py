#!/usr/bin/env python3
"""
Sweep verifier thresholds on the evaluation split and recommend one.

Primary recommendation uses balanced accuracy, which is a safer default than
raw accuracy for highly imbalanced step-verification labels.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from reward_fn import PRMClassifier
from train_verifier import MAX_LENGTH, PadCollator, VerifierDataset


DEFAULT_DATASET_GLOB = "/root/autodl-tmp/prm_grpo/datasets/prm800k/trl-lib___prm800k/default/0.0.0/*/"
DEFAULT_MODEL_PATH = "/root/autodl-tmp/prm_grpo/verifier_cls/checkpoint-2000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune verifier decision threshold.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Checkpoint directory.")
    parser.add_argument(
        "--dataset-glob",
        default=DEFAULT_DATASET_GLOB,
        help="Glob pointing to cached PRM800K arrow shard directory.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Which cached split to evaluate.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for batched tensors.")
    parser.add_argument("--batch-size", type=int, default=8, help="Eval batch size.")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Prompt max length.")
    parser.add_argument("--min-threshold", type=float, default=0.0, help="Sweep start.")
    parser.add_argument("--max-threshold", type=float, default=1.0, help="Sweep end.")
    parser.add_argument("--num-thresholds", type=int, default=201, help="Number of thresholds to sweep.")
    parser.add_argument(
        "--metric",
        default="balanced_accuracy",
        choices=["balanced_accuracy", "neg_f1", "accuracy", "neg_recall"],
        help="Metric used for the primary recommendation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many top thresholds to print.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save the full sweep results as JSON.",
    )
    return parser.parse_args()


def load_eval_dataset(tokenizer, dataset_glob: str, split: str, max_length: int) -> VerifierDataset:
    matches = glob.glob(dataset_glob)
    if not matches:
        raise FileNotFoundError(f"No cached dataset directory matched: {dataset_glob}")

    arrow_dir = matches[0]
    split_path = os.path.join(arrow_dir, f"prm800k-{split}.arrow")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Cached split not found: {split_path}")

    hf_split = HFDataset.from_file(split_path)
    return VerifierDataset(hf_split, tokenizer, max_length=max_length)


@torch.no_grad()
def collect_eval_scores(
    dataloader: DataLoader,
    model: PRMClassifier,
    tensor_device: torch.device,
) -> Dict[str, np.ndarray]:
    all_pos_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(tensor_device)
        attention_mask = batch["attention_mask"].to(tensor_device)

        outputs = model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_idx, last_token_indices]

        cls_logits = model.score(last_hidden)
        pos_probs = torch.softmax(cls_logits, dim=-1)[:, 0]

        all_pos_probs.append(pos_probs.float().cpu())
        all_labels.append(batch["labels"].cpu())

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
    return max(
        metrics,
        key=lambda row: (row[metric_name], -abs(row["threshold"] - 0.5)),
    )


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_device = "cpu" if args.device == "cpu" else "cuda"
    tensor_device = torch.device(args.device)

    print(f"Loading verifier checkpoint from: {args.model_path}")
    model = PRMClassifier.from_pretrained(args.model_path, device=load_device)
    model.eval()

    print(f"Loading PRM800K cached split: {args.split}")
    eval_ds = load_eval_dataset(
        tokenizer=tokenizer,
        dataset_glob=args.dataset_glob,
        split=args.split,
        max_length=args.max_length,
    )
    dataloader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PadCollator(tokenizer.pad_token_id),
    )

    print("Collecting verifier probabilities...")
    outputs = collect_eval_scores(dataloader, model, tensor_device)
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
        f"`correct_threshold={best['threshold']:.4f}` into compute_reward()."
    )


if __name__ == "__main__":
    main()
