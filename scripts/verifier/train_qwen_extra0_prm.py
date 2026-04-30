#!/usr/bin/env python3
"""
Train a Qwen-style ``<extra_0>`` token-classification PRM.

Default mainline:

- base model: models/Qwen2.5-Math-7B-Instruct
- tuning: LoRA
- data: raw OpenAI PRM800K phase1+phase2, all steps
- labels: rating >= 0 -> positive
- loss: only at inserted ``<extra_0>`` positions
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from openai_prm_raw import (  # noqa: E402
    DEFAULT_RAW_DATA_DIR,
    build_raw_phase1_phase2_dataset,
    build_raw_phase2_dataset,
    phase1_phase2_cache_dir,
    phase2_cache_dir,
)
from qwen_extra0_prm import (  # noqa: E402
    EXTRA0_TOKEN,
    IGNORE_INDEX,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    Extra0PadCollator,
    build_extra0_token_classification_encoding,
    resolve_extra0_token_id,
    score_steps as score_steps_extra0,
)
from step_splitter import split_into_steps  # noqa: E402

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


DEFAULT_MODEL_PATH = "models/Qwen2.5-Math-7B-Instruct"
DEFAULT_OUTPUT_ROOT = "token_prm_runs"
DEFAULT_WANDB_PROJECT = "math_rl_extra0_prm"
DEFAULT_BEST_OF_N_REUSE_JSONL = (
    "/root/autodl-tmp/prm_grpo/outputs/prm_best_of_n/"
    "math_test_100_best_of_16.jsonl"
)


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def str2bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--dataset-source", choices=["raw_phase1_phase2", "raw_phase2"], default="raw_phase1_phase2")
    parser.add_argument("--raw-data-dir", default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--dataset-cache-root", default="/root/autodl-tmp/prm_grpo/datasets")
    parser.add_argument("--force-rebuild-dataset", action="store_true")
    parser.add_argument("--neutral-policy", choices=["nonnegative", "positive_only"], default="nonnegative")
    parser.add_argument("--stop-at-first-negative", type=str2bool, default=False)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--eval-row-fraction", type=float, default=0.125)
    parser.add_argument("--eval-max-rows", type=int, default=None)

    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--bf16", type=str2bool, default=True)
    parser.add_argument("--gradient-checkpointing", type=str2bool, default=True)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=1)

    parser.add_argument("--use-lora", type=str2bool, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--lora-modules-to-save", default="score,classifier")

    parser.add_argument("--neg-loss-weight", type=float, default=5.0)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--rebalance-mode", choices=["none", "sampler"], default="none")
    parser.add_argument("--sampler-target-neg-frac", type=float, default=0.20)

    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--best-of-n-reuse-jsonl", default=DEFAULT_BEST_OF_N_REUSE_JSONL)
    parser.add_argument("--best-of-n-eval-max-samples", type=int, default=100)
    parser.add_argument("--best-of-n-prm-aggregation", choices=["mean_log", "sum_log", "mean", "min"], default="min")
    parser.add_argument("--best-of-n-verifier-max-length", type=int, default=1024)
    return parser.parse_args()


def dataset_tag(args: argparse.Namespace) -> str:
    source = "phase1phase2raw" if args.dataset_source == "raw_phase1_phase2" else "phase2raw"
    neutral = "nonneg" if args.neutral_policy == "nonnegative" else "posonly"
    steps = "firsterr" if args.stop_at_first_negative else "allsteps"
    return f"{source}-{neutral}-{steps}"


def run_name_from_args(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    tuning = "lora" if args.use_lora else "full"
    negw = str(args.neg_loss_weight).replace(".", "p")
    gamma = str(args.focal_gamma).replace(".", "p")
    return f"extra0-prm-{tuning}-{dataset_tag(args)}-negw{negw}-focalg{gamma}"


def aggregate_step_scores(step_scores: Sequence[float], mode: str) -> float:
    if not step_scores:
        return float("-inf")
    if mode == "mean":
        return float(sum(step_scores) / len(step_scores))
    if mode == "min":
        return float(min(step_scores))

    clamped = [min(max(float(score), 1e-6), 1.0) for score in step_scores]
    log_scores = [math.log(score) for score in clamped]
    if mode == "sum_log":
        return float(sum(log_scores))
    if mode == "mean_log":
        return float(sum(log_scores) / len(log_scores))
    raise ValueError(f"Unsupported PRM aggregation: {mode}")


def load_fixed_best_of_n_eval(args: argparse.Namespace) -> Optional[dict]:
    path = args.best_of_n_reuse_jsonl
    if not path:
        logger.warning("Fixed best-of-N eval JSONL is disabled.")
        return None
    if not os.path.exists(path):
        logger.warning("Fixed best-of-N eval JSONL was not found, skipping rerank metrics: %s", path)
        return None

    examples: List[Dict[str, Any]] = []
    greedy_correct: List[float] = []
    majority_correct: List[float] = []
    oracle_correct: List[float] = []
    num_generations: List[int] = []

    with open(path, "r", encoding="utf-8") as reader:
        for line_idx, line in enumerate(reader):
            if args.best_of_n_eval_max_samples is not None and line_idx >= args.best_of_n_eval_max_samples:
                break
            row = json.loads(line)
            sampled = row.get("sampled", [])
            candidates = [
                {
                    "text": str(sample.get("text", "")),
                    "math_correct": float(sample.get("math_correct", 0.0) or 0.0),
                }
                for sample in sampled
            ]
            if not candidates:
                continue
            examples.append(
                {
                    "problem": str(row.get("problem", "")),
                    "gold_answer": str(row.get("gold_answer", "")),
                    "candidates": candidates,
                }
            )
            greedy_correct.append(float(row.get("greedy_correct", 0.0) or 0.0))
            majority_correct.append(float(row.get("majority_correct", 0.0) or 0.0))
            oracle_correct.append(float(row.get("sample_oracle_correct", max(candidate["math_correct"] for candidate in candidates)) or 0.0))
            num_generations.append(len(candidates))

    if not examples:
        logger.warning("Fixed best-of-N eval JSONL had no usable candidate rows: %s", path)
        return None

    reference_metrics = {
        "best_of_n/reference_greedy_accuracy": float(np.mean(greedy_correct)) if greedy_correct else 0.0,
        "best_of_n/reference_majority_vote_accuracy": float(np.mean(majority_correct)) if majority_correct else 0.0,
        "best_of_n/reference_sample_oracle_accuracy": float(np.mean(oracle_correct)) if oracle_correct else 0.0,
    }
    logger.info(
        "Loaded fixed best-of-N eval set: jsonl=%s examples=%s mean_candidates=%.1f greedy=%.4f majority=%.4f oracle=%.4f aggregation=%s",
        path,
        f"{len(examples):,}",
        float(np.mean(num_generations)) if num_generations else 0.0,
        reference_metrics["best_of_n/reference_greedy_accuracy"],
        reference_metrics["best_of_n/reference_majority_vote_accuracy"],
        reference_metrics["best_of_n/reference_sample_oracle_accuracy"],
        args.best_of_n_prm_aggregation,
    )
    return {
        "examples": examples,
        "reference_metrics": reference_metrics,
    }


def load_training_dataset(args: argparse.Namespace):
    if args.dataset_source == "raw_phase2":
        cache_dir = phase2_cache_dir(
            cache_root=args.dataset_cache_root,
            neutral_policy=args.neutral_policy,
            stop_at_first_negative=args.stop_at_first_negative,
        )
        logger.info(
            "Loading extra0 PRM dataset: source=raw_phase2 raw_dir=%s cache_dir=%s neutral_policy=%s stop_at_first_negative=%s",
            args.raw_data_dir,
            cache_dir,
            args.neutral_policy,
            args.stop_at_first_negative,
        )
        return build_raw_phase2_dataset(
            raw_data_dir=args.raw_data_dir,
            cache_dir=cache_dir,
            force_rebuild=args.force_rebuild_dataset,
            neutral_policy=args.neutral_policy,
            stop_at_first_negative=args.stop_at_first_negative,
        )

    cache_dir = phase1_phase2_cache_dir(
        cache_root=args.dataset_cache_root,
        neutral_policy=args.neutral_policy,
        stop_at_first_negative=args.stop_at_first_negative,
    )
    logger.info(
        "Loading extra0 PRM dataset: source=raw_phase1_phase2 raw_dir=%s cache_dir=%s neutral_policy=%s stop_at_first_negative=%s",
        args.raw_data_dir,
        cache_dir,
        args.neutral_policy,
        args.stop_at_first_negative,
    )
    return build_raw_phase1_phase2_dataset(
        raw_data_dir=args.raw_data_dir,
        cache_dir=cache_dir,
        force_rebuild=args.force_rebuild_dataset,
        neutral_policy=args.neutral_policy,
        stop_at_first_negative=args.stop_at_first_negative,
    )


@dataclass
class Extra0DatasetStats:
    row_count: int
    supervised_count: int
    pos_count: int
    neg_count: int
    expected_extra0_count: int
    kept_extra0_count: int
    dropped_label_count: int
    truncated_row_count: int

    @property
    def natural_neg_frac(self) -> float:
        if self.supervised_count == 0:
            return 0.0
        return self.neg_count / self.supervised_count

    @property
    def extra0_positions_mean(self) -> float:
        if self.row_count == 0:
            return 0.0
        return self.kept_extra0_count / self.row_count

    @property
    def truncation_rate(self) -> float:
        if self.row_count == 0:
            return 0.0
        return self.truncated_row_count / self.row_count

    @property
    def dropped_label_frac(self) -> float:
        if self.expected_extra0_count == 0:
            return 0.0
        return self.dropped_label_count / self.expected_extra0_count

    def as_metrics(self, prefix: str) -> Dict[str, float]:
        return {
            f"extra0/{prefix}_pos_count": float(self.pos_count),
            f"extra0/{prefix}_neg_count": float(self.neg_count),
            f"extra0/{prefix}_natural_neg_frac": self.natural_neg_frac,
            f"extra0/{prefix}_extra0_positions_mean": self.extra0_positions_mean,
            f"extra0/{prefix}_truncation_rate": self.truncation_rate,
        }


class Extra0PRMDataset(Dataset):
    def __init__(
        self,
        hf_split,
        tokenizer,
        *,
        max_length: int,
        max_rows: Optional[int] = None,
    ) -> None:
        self.examples: List[Dict[str, torch.Tensor]] = []
        self.sample_labels: List[int] = []
        self.extra0_token_id = resolve_extra0_token_id(tokenizer, EXTRA0_TOKEN)

        pos_count = 0
        neg_count = 0
        expected_extra0_count = 0
        kept_extra0_count = 0
        dropped_label_count = 0
        truncated_row_count = 0
        row_count = 0

        for row in hf_split:
            if max_rows is not None and row_count >= max_rows:
                break
            steps = row["completions"]
            labels = row["labels"]
            if not steps:
                continue

            encoding = build_extra0_token_classification_encoding(
                tokenizer,
                row["prompt"],
                steps,
                labels,
                max_length=max_length,
                extra0_token_id=self.extra0_token_id,
            )
            if encoding.kept_extra0_count == 0:
                continue

            tensor_row = encoding.as_tensors()
            self.examples.append(tensor_row)
            row_count += 1

            supervised = [label for label in encoding.labels if label != IGNORE_INDEX]
            row_pos_count = sum(1 for label in supervised if label == POSITIVE_LABEL)
            row_neg_count = sum(1 for label in supervised if label == NEGATIVE_LABEL)
            self.sample_labels.append(1 if row_neg_count > 0 else 0)
            pos_count += row_pos_count
            neg_count += row_neg_count
            expected_extra0_count += encoding.expected_extra0_count
            kept_extra0_count += encoding.kept_extra0_count
            dropped_label_count += encoding.dropped_label_count
            truncated_row_count += int(encoding.truncated)

        self.stats = Extra0DatasetStats(
            row_count=row_count,
            supervised_count=pos_count + neg_count,
            pos_count=pos_count,
            neg_count=neg_count,
            expected_extra0_count=expected_extra0_count,
            kept_extra0_count=kept_extra0_count,
            dropped_label_count=dropped_label_count,
            truncated_row_count=truncated_row_count,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitems__(self, indices):
        return [self.__getitem__(idx) for idx in indices]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]


class Extra0PRMTrainer(Trainer):
    """Trainer that computes loss only at ``<extra_0>`` positions."""

    def get_train_dataloader(self) -> DataLoader:
        if getattr(self, "_rebalance_mode", "none") != "sampler":
            return super().get_train_dataloader()

        train_dataset = self.train_dataset
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        sample_labels = getattr(train_dataset, "sample_labels", None)
        if not sample_labels:
            logger.warning("Sampler rebalance requested but train_dataset has no sample_labels; using default dataloader.")
            return super().get_train_dataloader()

        n_pos_only = sum(1 for label in sample_labels if label == 0)
        n_has_neg = sum(1 for label in sample_labels if label == 1)
        if n_pos_only == 0 or n_has_neg == 0:
            logger.warning(
                "Sampler rebalance fallback: only one row class present (pos_only=%s has_neg=%s). "
                "Using default dataloader.",
                n_pos_only,
                n_has_neg,
            )
            return super().get_train_dataloader()

        target_neg_frac = float(getattr(self, "_sampler_target_neg_frac", 0.20))
        if not 0.0 < target_neg_frac < 1.0:
            raise ValueError(f"--sampler-target-neg-frac must be between 0 and 1, got {target_neg_frac}")

        natural_neg_row_frac = n_has_neg / (n_pos_only + n_has_neg)
        weight_pos_only = 1.0
        weight_has_neg = target_neg_frac * n_pos_only / ((1.0 - target_neg_frac) * n_has_neg)
        sample_weights = [weight_has_neg if label == 1 else weight_pos_only for label in sample_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        logger.info(
            "Extra0 sampler rebalance ablation: pos_only_rows=%s has_neg_rows=%s natural_has_neg_frac=%.4f "
            "target_has_neg_frac=%.4f weights(pos_only=%.4f, has_neg=%.4f)",
            f"{n_pos_only:,}",
            f"{n_has_neg:,}",
            natural_neg_row_frac,
            target_neg_frac,
            weight_pos_only,
            weight_has_neg,
        )

        batch_size = getattr(self, "_train_batch_size", self.args.per_device_train_batch_size)
        dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            persistent_workers=self.args.dataloader_persistent_workers,
        )
        return self.accelerator.prepare(dataloader)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
        )
        mask = labels != IGNORE_INDEX
        if not torch.any(mask):
            loss = outputs.logits.sum() * 0.0
            return (loss, outputs) if return_outputs else loss

        step_logits = outputs.logits[mask]
        true_cls = labels[mask]
        class_weights = torch.tensor(
            [self._neg_loss_weight, 1.0],
            device=step_logits.device,
            dtype=step_logits.dtype,
        )
        ce_loss = F.cross_entropy(step_logits, true_cls, weight=class_weights, reduction="none")
        if self._focal_gamma > 0:
            log_probs = F.log_softmax(step_logits, dim=-1)
            pt = log_probs.gather(1, true_cls.unsqueeze(1)).squeeze(1).exp()
            ce_loss = ((1.0 - pt).pow(self._focal_gamma)) * ce_loss
        loss = ce_loss.mean()
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"]
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            mask = labels != IGNORE_INDEX
            step_logits = outputs.logits[mask].detach()
            true_cls = labels[mask].detach()

        if prediction_loss_only:
            return (loss, None, None)
        return (loss, step_logits, true_cls)

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        unwrapped = self.model.module if hasattr(self.model, "module") else self.model
        unwrapped.save_pretrained(output_dir)
        tok = getattr(self, "processing_class", None) or self.tokenizer
        if tok is not None:
            tok.save_pretrained(output_dir)
        logger.info("Saved extra0 PRM model to %s", output_dir)

    def _save_optimizer_and_scheduler(self, output_dir):
        logger.info("Skipping optimizer/scheduler save for small extra0 checkpoint: %s", output_dir)

    def _prune_checkpoints_to_best_only(self) -> None:
        output_dir = self.args.output_dir
        if not output_dir or not os.path.isdir(output_dir):
            return

        best_checkpoint = self.state.best_model_checkpoint
        for name in os.listdir(output_dir):
            checkpoint_dir = os.path.join(output_dir, name)
            if not (os.path.isdir(checkpoint_dir) and name.startswith("checkpoint-")):
                continue
            if best_checkpoint and os.path.abspath(checkpoint_dir) == os.path.abspath(best_checkpoint):
                continue
            logger.info("Removing non-best extra0 PRM checkpoint: %s", checkpoint_dir)
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

    def _save_checkpoint(self, *args, **kwargs):
        super()._save_checkpoint(*args, **kwargs)
        self._prune_checkpoints_to_best_only()

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        aliased_logs = dict(logs)
        metric_aliases = {
            "loss": "extra0/train_loss",
            "train_loss": "extra0/train_loss",
            "eval_loss": "extra0/eval_loss",
            "eval_accuracy": "extra0/eval_accuracy",
            "eval_pos_accuracy": "extra0/eval_pos_accuracy",
            "eval_neg_accuracy": "extra0/eval_neg_accuracy",
            "eval_balanced_accuracy": "extra0/eval_balanced_accuracy",
            "eval_pred_neg_fraction": "extra0/eval_pred_neg_fraction",
            "eval_neg_auroc": "extra0/eval_neg_auroc",
            "eval_neg_average_precision": "extra0/eval_neg_average_precision",
        }
        for source_key, target_key in metric_aliases.items():
            if source_key in aliased_logs and target_key not in aliased_logs:
                aliased_logs[target_key] = aliased_logs[source_key]
        return super().log(aliased_logs, *args, **kwargs)

    def _evaluate_best_of_n_metric(self) -> Dict[str, float]:
        best_of_n_eval = getattr(self, "_best_of_n_eval", None)
        if not best_of_n_eval:
            return {}

        model = self.model.module if hasattr(self.model, "module") else self.model
        tokenizer = getattr(self, "processing_class", None) or self.tokenizer
        was_training = model.training
        device = str(self.args.device)

        selected_correct: List[float] = []
        correct_candidate_scores: List[float] = []
        wrong_candidate_scores: List[float] = []
        solvable_group_count = 0
        misranking_count = 0

        model.eval()
        try:
            for example in best_of_n_eval["examples"]:
                candidates = example["candidates"]
                scored_candidates: List[float] = []
                candidate_correctness = [float(candidate["math_correct"]) for candidate in candidates]
                for candidate in candidates:
                    steps = split_into_steps(candidate["text"])
                    step_scores = score_steps_extra0(
                        example["problem"],
                        steps,
                        model,
                        tokenizer,
                        device=device,
                        max_length=self._best_of_n_verifier_max_length,
                        require_all_steps=False,
                    )
                    score = aggregate_step_scores(step_scores, self._best_of_n_prm_aggregation)
                    scored_candidates.append(score)
                    if math.isfinite(score):
                        if candidate["math_correct"] >= 0.5:
                            correct_candidate_scores.append(score)
                        else:
                            wrong_candidate_scores.append(score)

                best_idx = max(range(len(scored_candidates)), key=lambda idx: scored_candidates[idx])
                best_correct = candidate_correctness[best_idx]
                selected_correct.append(best_correct)
                if any(correctness >= 0.5 for correctness in candidate_correctness):
                    solvable_group_count += 1
                    if best_correct < 0.5:
                        misranking_count += 1
        finally:
            if was_training:
                model.train()

        prm_best_acc = float(np.mean(selected_correct)) if selected_correct else 0.0
        ref = best_of_n_eval["reference_metrics"]
        vs_greedy_gap = prm_best_acc - ref["best_of_n/reference_greedy_accuracy"]
        vs_majority_gap = prm_best_acc - ref["best_of_n/reference_majority_vote_accuracy"]
        misranking_frac = float(misranking_count / solvable_group_count) if solvable_group_count else 0.0
        correct_score_mean = float(np.mean(correct_candidate_scores)) if correct_candidate_scores else 0.0
        wrong_score_mean = float(np.mean(wrong_candidate_scores)) if wrong_candidate_scores else 0.0
        return {
            "best_of_n/prm_best_of_16_accuracy": prm_best_acc,
            "best_of_n/vs_greedy_gap": vs_greedy_gap,
            "best_of_n/vs_majority_gap": vs_majority_gap,
            "best_of_n/misranking_frac": misranking_frac,
            "best_of_n/candidate_correct_score_mean": correct_score_mean,
            "best_of_n/candidate_wrong_score_mean": wrong_score_mean,
            "eval_best_of_n_prm_best_of_16_accuracy": prm_best_acc,
            "eval_best_of_n_vs_greedy_gap": vs_greedy_gap,
            "eval_best_of_n_vs_majority_gap": vs_majority_gap,
        }

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if not self.is_world_process_zero():
            return metrics

        best_of_n_metrics = self._evaluate_best_of_n_metric()
        if best_of_n_metrics:
            logger.info(
                "Fixed best-of-N eval: prm_best_of_16_accuracy=%.4f vs_greedy=%.4f vs_majority=%.4f misranking=%.4f",
                best_of_n_metrics["best_of_n/prm_best_of_16_accuracy"],
                best_of_n_metrics["best_of_n/vs_greedy_gap"],
                best_of_n_metrics["best_of_n/vs_majority_gap"],
                best_of_n_metrics["best_of_n/misranking_frac"],
            )
            metrics.update(best_of_n_metrics)
            self.log(best_of_n_metrics)
        return metrics


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.asarray(logits)
    true_cls = np.asarray(labels).reshape(-1)
    if logits.size == 0 or true_cls.size == 0:
        return {
            "accuracy": 0.0,
            "pos_accuracy": 0.0,
            "neg_accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "pred_neg_fraction": 0.0,
            "neg_auroc": 0.5,
            "neg_average_precision": 0.0,
        }

    pred_cls = np.argmax(logits, axis=-1)
    probs = _softmax_np(logits)
    neg_probs = probs[:, NEGATIVE_LABEL]

    accuracy = float((pred_cls == true_cls).mean())
    pos_mask = true_cls == POSITIVE_LABEL
    neg_mask = true_cls == NEGATIVE_LABEL
    pos_accuracy = float((pred_cls[pos_mask] == POSITIVE_LABEL).mean()) if pos_mask.any() else 0.0
    neg_accuracy = float((pred_cls[neg_mask] == NEGATIVE_LABEL).mean()) if neg_mask.any() else 0.0
    balanced_accuracy = 0.5 * (pos_accuracy + neg_accuracy)
    pred_neg_fraction = float((pred_cls == NEGATIVE_LABEL).mean())

    neg_auroc = 0.5
    if pos_mask.any() and neg_mask.any():
        pos_scores = neg_probs[pos_mask]
        neg_scores = neg_probs[neg_mask]
        wins = sum((neg_score > pos_scores).sum() + 0.5 * (neg_score == pos_scores).sum() for neg_score in neg_scores)
        neg_auroc = float(wins / (len(neg_scores) * len(pos_scores)))

    neg_average_precision = 0.0
    if neg_mask.any():
        order = np.argsort(-neg_probs)
        sorted_labels = true_cls[order] == NEGATIVE_LABEL
        tp = np.cumsum(sorted_labels)
        precision = tp / np.arange(1, len(sorted_labels) + 1)
        neg_average_precision = float((precision[sorted_labels]).sum() / max(sorted_labels.sum(), 1))

    return {
        "accuracy": accuracy,
        "pos_accuracy": pos_accuracy,
        "neg_accuracy": neg_accuracy,
        "balanced_accuracy": float(balanced_accuracy),
        "pred_neg_fraction": pred_neg_fraction,
        "neg_auroc": neg_auroc,
        "neg_average_precision": neg_average_precision,
    }


def log_dataset_stats(train_ds: Extra0PRMDataset, eval_ds: Extra0PRMDataset, args: argparse.Namespace) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics.update(train_ds.stats.as_metrics("train"))
    metrics.update(eval_ds.stats.as_metrics("eval"))
    total_expected = train_ds.stats.expected_extra0_count + eval_ds.stats.expected_extra0_count
    total_dropped = train_ds.stats.dropped_label_count + eval_ds.stats.dropped_label_count
    metrics["extra0/dropped_label_frac"] = (
        float(total_dropped / total_expected) if total_expected else 0.0
    )
    metrics["extra0/neg_loss_weight"] = float(args.neg_loss_weight)
    metrics["extra0/focal_gamma"] = float(args.focal_gamma)
    metrics["extra0/rebalance_mode"] = 1.0 if args.rebalance_mode == "sampler" else 0.0
    metrics["extra0/sampler_target_neg_frac"] = float(args.sampler_target_neg_frac)
    metrics["extra0/effective_neg_weight_share"] = effective_neg_weight_share(
        train_ds.stats.pos_count,
        train_ds.stats.neg_count,
        args.neg_loss_weight,
    )
    metrics["extra0/lora_r"] = float(args.lora_r if args.use_lora else 0)
    metrics["extra0/lora_alpha"] = float(args.lora_alpha if args.use_lora else 0)
    metrics["extra0/save_total_limit"] = float(args.save_total_limit)

    logger.info(
        "Extra0 dataset stats: train_rows=%s train_pos=%s train_neg=%s train_extra0_mean=%.2f train_trunc=%.4f "
        "eval_rows=%s eval_pos=%s eval_neg=%s eval_extra0_mean=%.2f eval_trunc=%.4f dropped_label_frac=%.6f",
        f"{train_ds.stats.row_count:,}",
        f"{train_ds.stats.pos_count:,}",
        f"{train_ds.stats.neg_count:,}",
        train_ds.stats.extra0_positions_mean,
        train_ds.stats.truncation_rate,
        f"{eval_ds.stats.row_count:,}",
        f"{eval_ds.stats.pos_count:,}",
        f"{eval_ds.stats.neg_count:,}",
        eval_ds.stats.extra0_positions_mean,
        eval_ds.stats.truncation_rate,
        metrics["extra0/dropped_label_frac"],
    )
    return metrics


def effective_neg_weight_share(pos_count: int, neg_count: int, neg_loss_weight: float) -> float:
    weighted_neg = neg_count * neg_loss_weight
    total = pos_count + weighted_neg
    return float(weighted_neg / total) if total else 0.0


def update_wandb_contract_summary(
    args: argparse.Namespace,
    training_args: TrainingArguments,
    *,
    extra0_token_id: int,
    fixed_best_of_n_eval: Optional[dict],
) -> None:
    if wandb is None or wandb.run is None:
        return

    summary = {
        "extra0/base_model": args.model_path,
        "extra0/tuning_mode": "lora" if args.use_lora else "full",
        "extra0/lora_r": args.lora_r if args.use_lora else 0,
        "extra0/lora_alpha": args.lora_alpha if args.use_lora else 0,
        "extra0/rebalance_mode": args.rebalance_mode,
        "extra0/sampler_target_neg_frac": args.sampler_target_neg_frac,
        "extra0/save_total_limit": args.save_total_limit,
        "extra0/extra0_token_id": extra0_token_id,
        "extra0/best_metric_name": training_args.metric_for_best_model,
    }
    if fixed_best_of_n_eval is not None:
        summary.update(fixed_best_of_n_eval["reference_metrics"])
    for key, value in summary.items():
        wandb.run.summary[key] = value


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    modules_to_save = split_csv(args.lora_modules_to_save)
    return LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=split_csv(args.lora_target_modules),
        modules_to_save=modules_to_save or None,
    )


def prepare_model(args: argparse.Namespace):
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_path,
        num_labels=2,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    model.config.use_cache = False
    if args.use_lora:
        model = get_peft_model(model, build_lora_config(args))
        model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.rebalance_mode == "sampler":
        logger.info("Sampler rebalance enabled as an explicit ablation; default mainline remains rebalance_mode=none.")

    run_name = run_name_from_args(args)
    output_dir = os.path.join(args.output_root, run_name)
    logger.info(
        "Extra0 PRM config: run_name=%s model=%s output_dir=%s dataset=%s neutral_policy=%s stop_at_first_negative=%s tuning=%s",
        run_name,
        args.model_path,
        output_dir,
        args.dataset_source,
        args.neutral_policy,
        args.stop_at_first_negative,
        "lora" if args.use_lora else "full",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    extra0_token_id = resolve_extra0_token_id(tokenizer)
    logger.info("Resolved %s token id: %s", EXTRA0_TOKEN, extra0_token_id)

    model = prepare_model(args)
    ds = load_training_dataset(args)

    eval_max_rows = args.eval_max_rows
    if eval_max_rows is None:
        eval_max_rows = max(1, int(math.ceil(len(ds["test"]) * args.eval_row_fraction)))
    train_ds = Extra0PRMDataset(
        ds["train"],
        tokenizer,
        max_length=args.max_length,
        max_rows=args.max_train_rows,
    )
    eval_ds = Extra0PRMDataset(
        ds["test"],
        tokenizer,
        max_length=args.max_length,
        max_rows=eval_max_rows,
    )
    if len(train_ds) == 0 or len(eval_ds) == 0:
        raise ValueError(f"Empty train/eval dataset after encoding: train={len(train_ds)} eval={len(eval_ds)}")

    data_metrics = log_dataset_stats(train_ds, eval_ds, args)
    fixed_best_of_n_eval = load_fixed_best_of_n_eval(args)
    metric_for_best_model = (
        "eval_best_of_n_prm_best_of_16_accuracy"
        if fixed_best_of_n_eval is not None
        else "eval_balanced_accuracy"
    )
    if fixed_best_of_n_eval is None:
        logger.warning(
            "Falling back to eval_balanced_accuracy for checkpoint selection because fixed best-of-N eval data is unavailable."
        )
    else:
        logger.info("Using fixed best-of-N reranking accuracy for checkpoint selection: %s", metric_for_best_model)
    samples_per_optimizer_step = args.per_device_train_batch_size * args.gradient_accumulation_steps
    uncapped_steps = int(math.ceil((len(train_ds) * args.num_train_epochs) / samples_per_optimizer_step))
    logger.info(
        "Extra0 PRM step budget: train_rows=%s samples_per_optimizer_step=%s uncapped_steps=%s max_steps=%s",
        f"{len(train_ds):,}",
        f"{samples_per_optimizer_step:,}",
        f"{uncapped_steps:,}",
        f"{args.max_steps:,}",
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        fp16=False,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to=args.report_to,
        run_name=run_name,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        seed=args.seed,
    )

    trainer = Extra0PRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=Extra0PadCollator(tokenizer.pad_token_id),
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.001,
            )
        ],
        compute_metrics=compute_metrics,
    )
    trainer._neg_loss_weight = args.neg_loss_weight
    trainer._focal_gamma = args.focal_gamma
    trainer._rebalance_mode = args.rebalance_mode
    trainer._sampler_target_neg_frac = args.sampler_target_neg_frac
    trainer._best_of_n_eval = fixed_best_of_n_eval
    trainer._best_of_n_prm_aggregation = args.best_of_n_prm_aggregation
    trainer._best_of_n_verifier_max_length = args.best_of_n_verifier_max_length
    trainer.log(data_metrics)
    if fixed_best_of_n_eval is not None:
        trainer.log(fixed_best_of_n_eval["reference_metrics"])
    update_wandb_contract_summary(
        args,
        training_args,
        extra0_token_id=extra0_token_id,
        fixed_best_of_n_eval=fixed_best_of_n_eval,
    )

    logger.info("Starting extra0 PRM training...")
    trainer.train()
    trainer._prune_checkpoints_to_best_only()
    logger.info(
        "Training complete. best_checkpoint=%s save_total_limit=%s",
        trainer.state.best_model_checkpoint,
        args.save_total_limit,
    )
    if wandb is not None and wandb.run is not None:
        if trainer.state.best_model_checkpoint:
            wandb.run.summary["extra0/best_checkpoint"] = trainer.state.best_model_checkpoint
        wandb.run.summary["extra0/best_metric_name"] = training_args.metric_for_best_model
    logger.info("Done.")


if __name__ == "__main__":
    main()
