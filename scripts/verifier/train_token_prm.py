#!/usr/bin/env python3
"""
Train an OpenAI-style token-prediction PRM.

Current default data path:
- raw OpenAI PRM800K phase 1 + phase 2
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
from token_reward_fn import score_steps as score_steps_token_prm
from step_splitter import split_into_steps
from openai_prm_raw import (
    DEFAULT_RAW_DATA_DIR,
    build_raw_phase1_phase2_dataset,
    build_raw_phase2_dataset,
    phase1_phase2_cache_dir,
    phase2_cache_dir,
)
from eval_prm_best_of_n import (
    aggregate_step_scores,
    load_reused_generations,
    majority_vote_answer,
    safe_math_score,
    score_majority_answer,
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
DATASET_SOURCE = "raw_phase1_phase2"
RAW_DATA_DIR = DEFAULT_RAW_DATA_DIR
RAW_NEUTRAL_POLICY = "nonnegative"
MAX_LENGTH = 1536
STOP_AT_FIRST_NEGATIVE = True
FREEZE_BASE_MODEL = False
WANDB_PROJECT = "math_rl_token_prm"
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 4
EVAL_ROW_FRACTION = 0.125
NEG_LOSS_WEIGHT = 3.5
FOCAL_GAMMA = 2.0
MAX_STEPS = 20000
WARMUP_RATIO = 0.01
BEST_OF_N_REUSE_JSONL = (
    "/root/autodl-tmp/prm_grpo/outputs/prm_best_of_n/math_test_100_best_of_16.jsonl"
)
BEST_OF_N_EVAL_MAX_SAMPLES = 100
BEST_OF_N_PRM_AGGREGATION = "min"
BEST_OF_N_VERIFIER_MAX_LENGTH = 1024
BEST_OF_N_VERIFIER_BATCH_SIZE = 8


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
    if dataset_source == "raw_phase1_phase2":
        neutral_tag = "nonneg" if neutral_policy == "nonnegative" else "posonly"
        prefix_tag = "firsterr" if stop_at_first_negative else "allsteps"
        return f"phase1phase2raw-{neutral_tag}-{prefix_tag}"
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
        tok = getattr(self, "processing_class", None) or self.tokenizer
        if tok is not None:
            tok.save_pretrained(output_dir)
        logger.info("Saved token-PRM model to %s", output_dir)

    def _save_optimizer_and_scheduler(self, output_dir):
        logger.info("Skipping optimizer/scheduler save for token-PRM checkpoint: %s", output_dir)

    def _evaluate_best_of_n_metric(self, metric_key_prefix: str) -> Dict[str, float]:
        examples = getattr(self, "_best_of_n_examples", None)
        sampled_completions = getattr(self, "_best_of_n_sampled_completions", None)
        if not examples or not sampled_completions:
            return {}

        model = self.model.module if hasattr(self.model, "module") else self.model
        was_training = model.training
        device = str(self.args.device)
        best_correct = []

        model.eval()
        try:
            for example, completion_group in zip(examples, sampled_completions):
                if not completion_group:
                    best_correct.append(0.0)
                    continue
                scored_group = []
                for text in completion_group:
                    steps = split_into_steps(text)
                    step_scores = score_steps_token_prm(
                        example.problem,
                        steps,
                        model,
                        getattr(self, "processing_class", None) or self.tokenizer,
                        self._label_tokens,
                        device=device,
                        max_length=self._best_of_n_verifier_max_length,
                        batch_size=self._best_of_n_verifier_batch_size,
                    )
                    scored_group.append(
                        aggregate_step_scores(step_scores, self._best_of_n_prm_aggregation)
                    )

                best_idx = max(range(len(scored_group)), key=lambda idx: scored_group[idx])
                best_text = completion_group[best_idx]
                best_correct.append(safe_math_score(best_text, example.gold_answer))
        finally:
            if was_training:
                model.train()

        best_of_n_accuracy = float(np.mean(best_correct)) if best_correct else 0.0
        return {
            f"{metric_key_prefix}_prm_best_of_16_accuracy": best_of_n_accuracy,
            f"{metric_key_prefix}_prm_best_of_16_vs_greedy_gap": (
                best_of_n_accuracy - self._best_of_n_reference_metrics["greedy_accuracy"]
            ),
            f"{metric_key_prefix}_prm_best_of_16_vs_majority_gap": (
                best_of_n_accuracy - self._best_of_n_reference_metrics["majority_vote_accuracy"]
            ),
        }

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if not self.is_world_process_zero():
            return metrics

        extra_metrics = self._evaluate_best_of_n_metric(metric_key_prefix)
        if extra_metrics:
            logger.info(
                "Fixed best-of-N eval: prm_best_of_16_accuracy=%.4f (greedy=%.4f majority=%.4f oracle=%.4f)",
                extra_metrics[f"{metric_key_prefix}_prm_best_of_16_accuracy"],
                self._best_of_n_reference_metrics["greedy_accuracy"],
                self._best_of_n_reference_metrics["majority_vote_accuracy"],
                self._best_of_n_reference_metrics["sample_oracle_accuracy"],
            )
            metrics.update(extra_metrics)
            self.log(extra_metrics)
        return metrics


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


def load_fixed_best_of_n_eval() -> Optional[dict]:
    if not BEST_OF_N_REUSE_JSONL or not os.path.exists(BEST_OF_N_REUSE_JSONL):
        logger.warning(
            "Fixed best-of-N eval JSONL was not found, skipping rerank metric: %s",
            BEST_OF_N_REUSE_JSONL,
        )
        return None

    examples, greedy_texts, sampled_completions = load_reused_generations(
        BEST_OF_N_REUSE_JSONL,
        BEST_OF_N_EVAL_MAX_SAMPLES,
    )

    greedy_correct = []
    majority_correct = []
    oracle_correct = []
    for row_idx, example in enumerate(examples):
        completion_group = sampled_completions[row_idx]
        sample_scores = [safe_math_score(text, example.gold_answer) for text in completion_group]
        majority_answer, _ = majority_vote_answer(completion_group)
        greedy_correct.append(safe_math_score(greedy_texts[row_idx], example.gold_answer))
        majority_correct.append(score_majority_answer(majority_answer, example.gold_answer))
        oracle_correct.append(max(sample_scores) if sample_scores else 0.0)

    reference_metrics = {
        "greedy_accuracy": float(np.mean(greedy_correct)) if greedy_correct else 0.0,
        "majority_vote_accuracy": float(np.mean(majority_correct)) if majority_correct else 0.0,
        "sample_oracle_accuracy": float(np.mean(oracle_correct)) if oracle_correct else 0.0,
    }
    logger.info(
        "Loaded fixed best-of-N eval set: jsonl=%s examples=%s greedy=%.4f majority=%.4f oracle=%.4f aggregation=%s",
        BEST_OF_N_REUSE_JSONL,
        f"{len(examples):,}",
        reference_metrics["greedy_accuracy"],
        reference_metrics["majority_vote_accuracy"],
        reference_metrics["sample_oracle_accuracy"],
        BEST_OF_N_PRM_AGGREGATION,
    )
    return {
        "examples": examples,
        "sampled_completions": sampled_completions,
        "reference_metrics": reference_metrics,
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
    if DATASET_SOURCE == "raw_phase1_phase2":
        logger.info(
            "Loading token-PRM dataset from raw OpenAI PRM800K phase1+phase2: raw_dir=%s cache_dir=%s neutral_policy=%s stop_at_first_negative=%s",
            RAW_DATA_DIR,
            phase1_phase2_cache_dir(
                neutral_policy=RAW_NEUTRAL_POLICY,
                stop_at_first_negative=STOP_AT_FIRST_NEGATIVE,
            ),
            RAW_NEUTRAL_POLICY,
            STOP_AT_FIRST_NEGATIVE,
        )
        return build_raw_phase1_phase2_dataset(
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
    fixed_best_of_n_eval = load_fixed_best_of_n_eval()
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
        if fixed_best_of_n_eval is not None:
            wandb.run.summary["best_of_n/reference_greedy_accuracy"] = fixed_best_of_n_eval["reference_metrics"]["greedy_accuracy"]
            wandb.run.summary["best_of_n/reference_majority_vote_accuracy"] = fixed_best_of_n_eval["reference_metrics"]["majority_vote_accuracy"]
            wandb.run.summary["best_of_n/reference_sample_oracle_accuracy"] = fixed_best_of_n_eval["reference_metrics"]["sample_oracle_accuracy"]

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
    trainer._best_of_n_prm_aggregation = BEST_OF_N_PRM_AGGREGATION
    trainer._best_of_n_verifier_max_length = BEST_OF_N_VERIFIER_MAX_LENGTH
    trainer._best_of_n_verifier_batch_size = BEST_OF_N_VERIFIER_BATCH_SIZE
    trainer._best_of_n_examples = fixed_best_of_n_eval["examples"] if fixed_best_of_n_eval is not None else None
    trainer._best_of_n_sampled_completions = (
        fixed_best_of_n_eval["sampled_completions"] if fixed_best_of_n_eval is not None else None
    )
    trainer._best_of_n_reference_metrics = (
        fixed_best_of_n_eval["reference_metrics"]
        if fixed_best_of_n_eval is not None
        else {
            "greedy_accuracy": 0.0,
            "majority_vote_accuracy": 0.0,
            "sample_oracle_accuracy": 0.0,
        }
    )

    logger.info("Starting token-prediction PRM training...")
    trainer.train()

    final_dir = os.path.join(output_dir, "final")
    logger.info("Saving final token-PRM model to %s", final_dir)
    trainer.save_model(final_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
