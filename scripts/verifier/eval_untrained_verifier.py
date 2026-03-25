#!/usr/bin/env python3
"""
Evaluate an untrained verifier baseline on cached PRM800K data.

This script mirrors the current verifier pipeline:
  - same verifier prompts
  - same tokenization / max length
  - same "last prompt token -> 2-way classification head" logic

Unlike train_verifier.py, the classification head here is randomly initialized
and never trained. That gives a useful sanity-check baseline for how much signal
the trained verifier is adding beyond chance / bias effects.
"""

import argparse
import glob
import os
import random
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train_verifier import MAX_LENGTH, MODEL_NAME, PadCollator, VerifierDataset


DEFAULT_DATASET_GLOB = "/root/autodl-tmp/prm_grpo/datasets/prm800k/trl-lib___prm800k/default/0.0.0/*/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an untrained verifier baseline.")
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Base model path used to produce verifier hidden states.",
    )
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
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds for untrained classifier heads.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of raw PRM800K rows before step expansion.",
    )
    return parser.parse_args()


def load_eval_dataset(tokenizer, dataset_glob: str, split: str, max_length: int, max_rows: int | None) -> VerifierDataset:
    matches = glob.glob(dataset_glob)
    if not matches:
        raise FileNotFoundError(f"No cached dataset directory matched: {dataset_glob}")

    arrow_dir = matches[0]
    split_path = os.path.join(arrow_dir, f"prm800k-{split}.arrow")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Cached split not found: {split_path}")

    hf_split = HFDataset.from_file(split_path)
    return VerifierDataset(hf_split, tokenizer, max_length=max_length, max_rows=max_rows)


def build_random_head(hidden_size: int, device: torch.device, dtype: torch.dtype, seed: int) -> nn.Linear:
    torch.manual_seed(seed)
    head = nn.Linear(hidden_size, 2, bias=False)
    nn.init.normal_(head.weight, std=0.02)
    head = head.to(device=device, dtype=dtype)
    head.eval()
    return head


def init_metric_state() -> Dict[str, float]:
    return {
        "n_total": 0,
        "n_correct": 0,
        "n_pos": 0,
        "n_neg": 0,
        "n_pos_correct": 0,
        "n_neg_correct": 0,
        "n_pred_pos": 0,
        "n_pred_neg": 0,
    }


def finalize_metrics(state: Dict[str, float]) -> Dict[str, float]:
    n_total = max(int(state["n_total"]), 1)
    n_pos = int(state["n_pos"])
    n_neg = int(state["n_neg"])
    n_pred_neg = int(state["n_pred_neg"])

    accuracy = state["n_correct"] / n_total
    pos_accuracy = state["n_pos_correct"] / n_pos if n_pos else 0.0
    neg_accuracy = state["n_neg_correct"] / n_neg if n_neg else 0.0
    balanced_accuracy = 0.5 * (pos_accuracy + neg_accuracy)
    neg_precision = state["n_neg_correct"] / n_pred_neg if n_pred_neg else 0.0
    neg_f1 = (
        2.0 * neg_precision * neg_accuracy / (neg_precision + neg_accuracy)
        if (neg_precision + neg_accuracy) > 0
        else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "pos_accuracy": float(pos_accuracy),
        "neg_accuracy": float(neg_accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "neg_precision": float(neg_precision),
        "neg_f1": float(neg_f1),
        "pred_neg_fraction": float(state["n_pred_neg"] / n_total),
    }


@torch.no_grad()
def evaluate_untrained_heads(
    dataloader: DataLoader,
    base_model,
    score_heads: Dict[int, nn.Linear],
    tensor_device: torch.device,
) -> Dict[int, Dict[str, float]]:
    metric_states = {seed: init_metric_state() for seed in score_heads}

    for batch in dataloader:
        input_ids = batch["input_ids"].to(tensor_device)
        attention_mask = batch["attention_mask"].to(tensor_device)
        labels = batch["labels"].cpu().numpy()

        outputs = base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_idx, last_token_indices]

        labels_pos = labels == 0
        labels_neg = labels == 1

        for seed, score_head in score_heads.items():
            logits = score_head(last_hidden)
            pred_cls = torch.argmax(logits, dim=-1).cpu().numpy()

            state = metric_states[seed]
            state["n_total"] += len(labels)
            state["n_correct"] += int((pred_cls == labels).sum())
            state["n_pos"] += int(labels_pos.sum())
            state["n_neg"] += int(labels_neg.sum())
            state["n_pos_correct"] += int(((pred_cls == 0) & labels_pos).sum())
            state["n_neg_correct"] += int(((pred_cls == 1) & labels_neg).sum())
            state["n_pred_pos"] += int((pred_cls == 0).sum())
            state["n_pred_neg"] += int((pred_cls == 1).sum())

    return {seed: finalize_metrics(state) for seed, state in metric_states.items()}


def summarize_metric(metric_name: str, per_seed: Dict[int, Dict[str, float]]) -> str:
    values = np.array([metrics[metric_name] for metrics in per_seed.values()], dtype=np.float64)
    return f"{values.mean():.4f} ± {values.std(ddof=0):.4f}"


def main() -> None:
    args = parse_args()

    random.seed(0)
    np.random.seed(0)

    print(f"Loading tokenizer from: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading cached PRM800K split: {args.split}")
    eval_ds = load_eval_dataset(
        tokenizer=tokenizer,
        dataset_glob=args.dataset_glob,
        split=args.split,
        max_length=args.max_length,
        max_rows=args.max_rows,
    )
    dataloader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PadCollator(tokenizer.pad_token_id),
    )

    print(f"Loading base model from: {args.model_name}")
    load_device = "cpu" if args.device == "cpu" else "auto"
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=load_device,
    )
    base_model.eval()

    tensor_device = torch.device(args.device)
    hidden_size = base_model.config.hidden_size
    head_device = next(base_model.parameters()).device
    head_dtype = next(base_model.parameters()).dtype

    print(f"Building random classifier heads for seeds: {args.seeds}")
    score_heads = {
        seed: build_random_head(hidden_size, head_device, head_dtype, seed)
        for seed in args.seeds
    }

    print("Running untrained verifier eval...")
    per_seed = evaluate_untrained_heads(dataloader, base_model, score_heads, tensor_device)

    natural_neg_fraction = float((np.array(eval_ds.sample_labels) == 1).mean()) if eval_ds.sample_labels else 0.0
    print(f"\nEval examples: {len(eval_ds):,}")
    print(f"Natural negative fraction: {natural_neg_fraction:.4f}")
    print(f"Max length: {args.max_length}")
    print("\nPer-seed metrics:")
    for seed in args.seeds:
        metrics = per_seed[seed]
        print(
            f"  seed={seed:<4d} "
            f"acc={metrics['accuracy']:.4f} "
            f"pos_acc={metrics['pos_accuracy']:.4f} "
            f"neg_acc={metrics['neg_accuracy']:.4f} "
            f"bal_acc={metrics['balanced_accuracy']:.4f} "
            f"neg_prec={metrics['neg_precision']:.4f} "
            f"neg_f1={metrics['neg_f1']:.4f} "
            f"pred_neg={metrics['pred_neg_fraction']:.4f}"
        )

    print("\nAverage over seeds:")
    for metric_name in [
        "accuracy",
        "pos_accuracy",
        "neg_accuracy",
        "balanced_accuracy",
        "neg_precision",
        "neg_f1",
        "pred_neg_fraction",
    ]:
        print(f"  {metric_name}: {summarize_metric(metric_name, per_seed)}")


if __name__ == "__main__":
    main()
