#!/usr/bin/env python3
"""
Train an OpenAI-style token-prediction PRM.

Current default data path:
- raw OpenAI PRM800K phase 2
- reconstructed chosen trajectories
- first-error-only truncation
- rating >= 0 treated as non-error
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from token_prm import (
    LabelTokenPair,
    PadCollator,
    TokenPRMDataset,
    binary_classes_from_labels,
    pair_logits_from_causal_lm_logits,
    select_label_token_pair,
)
from openai_prm_raw import (
    DEFAULT_RAW_DATA_DIR,
    build_raw_phase2_dataset,
    phase2_cache_dir,
)

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/prm_grpo/datasets/prm800k")


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


MODEL_NAME = "/root/autodl-tmp/prm_grpo/models/Qwen2.5-Math-1.5B"
OUTPUT_ROOT = "/root/autodl-tmp/prm_grpo/token_prm_runs"
DATASET_SOURCE = "raw_phase2"
RAW_DATA_DIR = DEFAULT_RAW_DATA_DIR
RAW_NEUTRAL_POLICY = "nonnegative"
MAX_LENGTH = 1536
STOP_AT_FIRST_NEGATIVE = True
FREEZE_BASE_MODEL = False
WANDB_PROJECT = "math_rl_token_prm"
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 4
EVAL_ROW_FRACTION = 0.125
NEG_LOSS_WEIGHT = 10.0
FOCAL_GAMMA = 2.0
MAX_STEPS = 20000
WARMUP_RATIO = 0.01


def training_mode_tag(freeze_base_model: bool) -> str:
    return "headonly" if freeze_base_model else "fullft"


def dataset_tag(
    dataset_source: str,
    *,
    neutral_policy: str,
    stop_at_first_negative: bool,
) -> str:
    if dataset_source == "raw_phase2":
        neutral_tag = "nonneg" if neutral_policy == "nonnegative" else "posonly"
        prefix_tag = "firsterr" if stop_at_first_negative else "allsteps"
        return f"phase2raw-{neutral_tag}-{prefix_tag}"
    raise ValueError(f"Unknown dataset source: {dataset_source}")


class TokenPRMTrainer(Trainer):
    """Trainer with compact eval outputs and small checkpoints."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)

        pair_logits = pair_logits_from_causal_lm_logits(
            outputs.logits,
            labels,
            self._label_tokens,
        )
        true_cls = binary_classes_from_labels(labels, self._label_tokens)
        class_weights = torch.tensor(
            [1.0, self._neg_loss_weight],
            device=pair_logits.device,
            dtype=pair_logits.dtype,
        )
        log_probs = torch.nn.functional.log_softmax(pair_logits, dim=-1)
        log_pt = log_probs.gather(1, true_cls.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        sample_weights = class_weights[true_cls]
        focal_factor = (1.0 - pt).pow(self._focal_gamma)
        loss = -(sample_weights * focal_factor * log_pt).mean()
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"]

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            pair_logits = pair_logits_from_causal_lm_logits(
                outputs.logits,
                labels,
                self._label_tokens,
            ).detach()
            true_cls = binary_classes_from_labels(labels, self._label_tokens).detach()

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, pair_logits, true_cls)

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        unwrapped = self.model.module if hasattr(self.model, "module") else self.model
        unwrapped.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        logger.info("Saved token-PRM model to %s", output_dir)

    def _save_optimizer_and_scheduler(self, output_dir):
        logger.info("Skipping optimizer/scheduler save for token-PRM checkpoint: %s", output_dir)


def compute_metrics(eval_pred):
    pair_logits, labels = eval_pred
    pred_cls = np.argmax(pair_logits, axis=1)
    true_cls = np.asarray(labels).reshape(-1)
    probs = torch.softmax(torch.tensor(pair_logits), dim=-1).numpy()
    neg_probs = probs[:, 1]

    accuracy = float((pred_cls == true_cls).mean())
    pos_mask = true_cls == 0
    neg_mask = true_cls == 1
    pos_accuracy = float((pred_cls[pos_mask] == 0).mean()) if pos_mask.any() else 0.0
    neg_accuracy = float((pred_cls[neg_mask] == 1).mean()) if neg_mask.any() else 0.0
    balanced_accuracy = 0.5 * (pos_accuracy + neg_accuracy)
    pred_neg_fraction = float((pred_cls == 1).mean())

    neg_auroc = 0.5
    if pos_mask.any() and neg_mask.any():
        pos_scores = neg_probs[pos_mask]
        neg_scores = neg_probs[neg_mask]
        wins = sum((n > pos_scores).sum() + 0.5 * (n == pos_scores).sum() for n in neg_scores)
        neg_auroc = float(wins / (len(neg_scores) * len(pos_scores)))

    neg_average_precision = 0.0
    if neg_mask.any():
        order = np.argsort(-neg_probs)
        sorted_labels = true_cls[order] == 1
        tp = np.cumsum(sorted_labels)
        precision = tp / np.arange(1, len(sorted_labels) + 1)
        neg_average_precision = float((precision[sorted_labels]).sum() / max(sorted_labels.sum(), 1))

    thresholds = np.unique(neg_probs)
    best_balanced_accuracy = balanced_accuracy
    best_balanced_accuracy_threshold = 0.5
    for threshold in thresholds:
        threshold_pred = (neg_probs >= threshold).astype(np.int64)
        threshold_pos_accuracy = (
            float((threshold_pred[pos_mask] == 0).mean()) if pos_mask.any() else 0.0
        )
        threshold_neg_accuracy = (
            float((threshold_pred[neg_mask] == 1).mean()) if neg_mask.any() else 0.0
        )
        threshold_balanced_accuracy = 0.5 * (threshold_pos_accuracy + threshold_neg_accuracy)
        if threshold_balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = float(threshold_balanced_accuracy)
            best_balanced_accuracy_threshold = float(threshold)

    return {
        "accuracy": accuracy,
        "pos_accuracy": pos_accuracy,
        "neg_accuracy": neg_accuracy,
        "balanced_accuracy": float(balanced_accuracy),
        "pred_neg_fraction": pred_neg_fraction,
        "neg_auroc": neg_auroc,
        "neg_average_precision": neg_average_precision,
        "best_balanced_accuracy": float(best_balanced_accuracy),
        "best_balanced_accuracy_threshold": float(best_balanced_accuracy_threshold),
    }


def load_training_dataset():
    if DATASET_SOURCE == "raw_phase2":
        logger.info(
            "Loading token-PRM dataset from raw OpenAI PRM800K phase2: raw_dir=%s cache_dir=%s neutral_policy=%s stop_at_first_negative=%s",
            RAW_DATA_DIR,
            phase2_cache_dir(
                neutral_policy=RAW_NEUTRAL_POLICY,
                stop_at_first_negative=STOP_AT_FIRST_NEGATIVE,
            ),
            RAW_NEUTRAL_POLICY,
            STOP_AT_FIRST_NEGATIVE,
        )
        return build_raw_phase2_dataset(
            raw_data_dir=RAW_DATA_DIR,
            neutral_policy=RAW_NEUTRAL_POLICY,
            stop_at_first_negative=STOP_AT_FIRST_NEGATIVE,
        )
    raise ValueError(f"Unsupported dataset source: {DATASET_SOURCE}")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    label_tokens = select_label_token_pair(tokenizer)

    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    run_name = (
        f"prm-token-{training_mode_tag(FREEZE_BASE_MODEL)}-"
        f"{dataset_tag(DATASET_SOURCE, neutral_policy=RAW_NEUTRAL_POLICY, stop_at_first_negative=STOP_AT_FIRST_NEGATIVE)}-"
        f"eval{int(EVAL_ROW_FRACTION * 100)}-"
        f"negw{str(NEG_LOSS_WEIGHT).replace('.', 'p')}-"
        f"focalg{str(FOCAL_GAMMA).replace('.', 'p')}-qwen25-math-1.5b"
    )
    output_dir = os.path.join(OUTPUT_ROOT, run_name)
    logger.info(
        "Token-PRM config: project=%s run_name=%s output_dir=%s dataset_source=%s label_tokens=(%r,%r) neg_loss_weight=%.2f focal_gamma=%.2f no_rebalance=true",
        os.environ["WANDB_PROJECT"],
        run_name,
        output_dir,
        DATASET_SOURCE,
        label_tokens.positive_text,
        label_tokens.negative_text,
        NEG_LOSS_WEIGHT,
        FOCAL_GAMMA,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if FREEZE_BASE_MODEL:
        for param in base_model.parameters():
            param.requires_grad = False
        logger.info("Training mode: head-only token PRM (base model frozen)")
    else:
        logger.info("Training mode: full fine-tuning token PRM")

    ds = load_training_dataset()
    eval_max_rows = max(1, int(len(ds["test"]) * EVAL_ROW_FRACTION))
    train_ds = TokenPRMDataset(
        ds["train"],
        tokenizer,
        label_tokens,
        max_length=MAX_LENGTH,
        stop_at_first_negative=STOP_AT_FIRST_NEGATIVE,
    )
    eval_ds = TokenPRMDataset(
        ds["test"],
        tokenizer,
        label_tokens,
        max_length=MAX_LENGTH,
        max_rows=eval_max_rows,
        stop_at_first_negative=STOP_AT_FIRST_NEGATIVE,
    )
    logger.info(
        "Token-PRM dataset stats: train(pos=%s neg=%s) eval(pos=%s neg=%s, max_rows=%s, row_fraction=%.2f)",
        f"{train_ds.n_pos:,}",
        f"{train_ds.n_neg:,}",
        f"{eval_ds.n_pos:,}",
        f"{eval_ds.n_neg:,}",
        f"{eval_max_rows:,}",
        EVAL_ROW_FRACTION,
    )
    samples_per_optimizer_step = PER_DEVICE_TRAIN_BATCH_SIZE * 8
    full_run_steps = int(np.ceil((len(train_ds) * 3) / samples_per_optimizer_step))
    logger.info(
        "Token-PRM step budget: train_examples=%s samples_per_optimizer_step=%s full_run_steps=%s capped_max_steps=%s warmup_ratio=%.3f",
        f"{len(train_ds):,}",
        f"{samples_per_optimizer_step:,}",
        f"{full_run_steps:,}",
        f"{MAX_STEPS:,}",
        WARMUP_RATIO,
    )

    if wandb is not None and wandb.run is not None:
        wandb.run.summary["label_tokens/positive_text"] = label_tokens.positive_text
        wandb.run.summary["label_tokens/negative_text"] = label_tokens.negative_text
        wandb.run.summary["label_tokens/positive_id"] = label_tokens.positive_id
        wandb.run.summary["label_tokens/negative_id"] = label_tokens.negative_id
        wandb.run.summary["training/neg_loss_weight"] = NEG_LOSS_WEIGHT
        wandb.run.summary["training/focal_gamma"] = FOCAL_GAMMA
        wandb.run.summary["training/rebalance"] = "none"
        wandb.run.summary["training/max_steps"] = MAX_STEPS
        wandb.run.summary["training/warmup_ratio"] = WARMUP_RATIO
        wandb.run.summary["training/full_run_steps_uncapped"] = full_run_steps

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=run_name,
        metric_for_best_model="eval_best_balanced_accuracy",
        greater_is_better=True,
    )

    trainer = TokenPRMTrainer(
        model=base_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PadCollator(tokenizer.pad_token_id),
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.001,
            )
        ],
        compute_metrics=compute_metrics,
    )
    trainer._label_tokens = label_tokens
    trainer._neg_loss_weight = NEG_LOSS_WEIGHT
    trainer._focal_gamma = FOCAL_GAMMA

    logger.info("Starting token-prediction PRM training...")
    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    logger.info("Saving final token-PRM model to %s", final_dir)
    trainer.save_model(final_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
