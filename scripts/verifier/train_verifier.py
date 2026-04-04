#!/usr/bin/env python3
"""
train_verifier.py — Fine-tune Qwen2.5-Math-1.5B on PRM800K as a step verifier.

Training approach (classification head):
  We add a Linear(hidden_size → 2) head on top of the base LM's hidden states.
  For each verifier prompt, we extract the hidden state at the final prompt token
  and classify the current step as positive or negative.

  This is a pure binary classification setup. We do not append "+" / "-"
  answer tokens anymore; each training row carries a direct class label 0/1.

v3: Soft-balanced sampling via WeightedRandomSampler to fix all-positive
collapse while staying closer to the real PRM800K class distribution.

Dataset: PRM800K (trl-lib format)
For an example with N steps we emit N training rows (one per step).
"""

import os
import sys
import logging
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    HfArgumentParser,
)
from datasets import Dataset as HFDataset, DatasetDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from verifier_prompt import format_verifier_prompt

try:
    import wandb
except ImportError:  # pragma: no cover - optional at local edit time
    wandb = None

# — Environment ————————————————————————————————————
os.environ.setdefault("HF_ENDPOINT",      "https://hf-mirror.com")
os.environ.setdefault("DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", "/root/autodl-tmp/prm_grpo/datasets/prm800k")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# — Defaults ——————————————————————————————————————
MODEL_NAME   = "/root/autodl-tmp/prm_grpo/models/Qwen2.5-Math-1.5B"
DATASET_NAME = "trl-lib/prm800k"
OUTPUT_DIR   = "/root/autodl-tmp/prm_grpo/verifier_cls"
MAX_LENGTH   = 1536
TARGET_NEGATIVE_FRACTION = 0.20
WANDB_DEBUG_TABLE_ROWS = 24
WANDB_STATS_SAMPLE_SIZE = 256


# — Classification Head Wrapper ———————————————————————

class PRMClassifier(nn.Module):
    """
    Wraps a causal LM with a binary classification head.

    For each sample, extracts the hidden state at the last non-padding prompt
    token and passes it through a linear layer to produce 2-class logits
    (positive vs negative step).
    """

    def __init__(self, base_model, hidden_size: int):
        super().__init__()
        self.base_model = base_model
        self.score = nn.Linear(hidden_size, 2, bias=False)
        nn.init.normal_(self.score.weight, std=0.02)
        # HF Trainer introspects these
        self.config = base_model.config

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (B, L, H)

        if attention_mask is not None:
            last_token_pos = attention_mask.sum(dim=1) - 1
        else:
            last_token_pos = torch.full(
                (hidden_states.size(0),),
                hidden_states.size(1) - 1,
                device=hidden_states.device,
                dtype=torch.long,
            )
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)

        last_hidden = hidden_states[batch_idx, last_token_pos]  # (B, H)

        cls_logits = self.score(last_hidden)  # (B, 2)

        # Return a simple object with .logits attribute
        return type("ClfOutput", (), {"logits": cls_logits})()

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()


# — Dataset ——————————————————————————————————————

class VerifierDataset(Dataset):
    """
    Converts PRM800K rows into per-step training examples.

    Each item:
        input_ids : verifier prompt tokens
        labels    : scalar class label (0=positive, 1=negative)
    """

    def __init__(
        self,
        hf_split,
        tokenizer: AutoTokenizer,
        max_length: int = MAX_LENGTH,
        max_rows: Optional[int] = None,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length

        self.examples: List[Dict[str, Any]] = []
        self.sample_labels: List[int] = []  # 0=pos, 1=neg for sampler
        n_rows = 0
        for row in hf_split:
            n_rows += 1
            problem = row["prompt"]
            # trl-lib format: completions=List[str], labels=List[bool]
            steps   = row["completions"]
            ratings = row["labels"]

            for k in range(len(steps)):
                label = ratings[k]
                class_label = 0 if label == 1 else 1
                prompt_text = format_verifier_prompt(problem, steps[: k + 1])
                self.examples.append({
                    "problem": problem,
                    "prompt": prompt_text,
                    "current_step": steps[k],
                    "n_steps_in_prompt": k + 1,
                    "label": class_label,
                })
                self.sample_labels.append(class_label)
            if max_rows and n_rows >= max_rows:
                break

        # Count pos/neg for dynamic class weighting
        self.n_pos = sum(1 for l in self.sample_labels if l == 0)
        self.n_neg = len(self.sample_labels) - self.n_pos
        logger.info(
            f"Built {len(self.examples):,} training examples from {n_rows:,} PRM800K rows"
            f" (pos={self.n_pos:,}, neg={self.n_neg:,}, ratio={self.n_pos/max(self.n_neg,1):.1f}:1)"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitems__(self, indices):
        return [self.__getitem__(i) for i in indices]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]

        prompt_ids = self.tokenizer.encode(
            ex["prompt"],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
        )

        return {
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
            "labels":    torch.tensor(ex["label"], dtype=torch.long),
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
            max_length=self.max_length,
            truncation=True,
        )
        raw_prompt_len = len(raw_prompt_ids)
        prompt_len = len(truncated_prompt_ids)
        model_input_text = self.tokenizer.decode(
            truncated_prompt_ids,
            skip_special_tokens=True,
        )
        return {
            "dataset_idx": idx,
            "problem": ex["problem"],
            "prompt_text": ex["prompt"],
            "model_input_text": model_input_text,
            "current_step": ex["current_step"],
            "label": ex["label"],
            "label_name": "positive" if ex["label"] == 0 else "negative",
            "n_steps_in_prompt": ex["n_steps_in_prompt"],
            "prompt_len_tokens": prompt_len,
            "raw_prompt_len_tokens": raw_prompt_len,
            "dropped_tokens": max(raw_prompt_len - prompt_len, 0),
            "truncated": raw_prompt_len > self.max_length,
            "has_current_step_marker": "[Current step]" in model_input_text,
            "has_answer_suffix": "Answer:" in model_input_text,
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
        rng = random.Random(seed)
        indices = rng.sample(range(n_examples), sample_size)
        lengths = []
        truncated = 0
        for idx in indices:
            row = self.prompt_debug_row(idx)
            lengths.append(row["raw_prompt_len_tokens"])
            truncated += int(row["truncated"])

        lengths_np = np.array(lengths, dtype=np.float32)
        return {
            "prompt_len_mean": float(lengths_np.mean()),
            "prompt_len_p95": float(np.percentile(lengths_np, 95)),
            "truncation_rate": float(truncated / sample_size),
            "sample_size": sample_size,
        }


# — Collator ——————————————————————————————————————

class PadCollator:
    """Right-pad sequences to the longest in the batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [f["input_ids"] for f in features]
        labels_list    = [f["labels"]    for f in features]
        max_len = max(t.size(0) for t in input_ids_list)

        padded_ids, attn_list = [], []
        for ids in input_ids_list:
            pad_len = max_len - ids.size(0)
            padded_ids.append(F.pad(ids, (0, pad_len), value=self.pad_token_id))
            attn_list.append(F.pad(torch.ones_like(ids), (0, pad_len), value=0))

        return {
            "input_ids":      torch.stack(padded_ids),
            "labels":         torch.stack(labels_list),
            "attention_mask": torch.stack(attn_list),
        }


# — Trainer & Metrics ————————————————————————————————

class ClassificationTrainer(Trainer):
    """Trainer that uses the classification head output for loss computation."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        cls_logits = outputs.logits  # (batch, 2) from classification head

        # Weighted CE — class weights computed from data distribution
        class_weights = torch.tensor(
            self._class_weights, device=cls_logits.device, dtype=cls_logits.dtype
        )
        loss = F.cross_entropy(cls_logits, labels, weight=class_weights)

        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        """Save base model + classification head separately."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Handle DataParallel / DeepSpeed wrappers
        unwrapped = self.model
        if hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        unwrapped.base_model.save_pretrained(output_dir)
        # Save classification head
        torch.save(unwrapped.score.state_dict(), os.path.join(output_dir, "cls_head.pt"))
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved model + cls_head to {output_dir}")


    def get_train_dataloader(self) -> DataLoader:
        """
        Override to use WeightedRandomSampler for soft-balanced class sampling.
        Instead of forcing a 1:1 class mix, we target a gentler negative-step
        fraction so training stays closer to the real data distribution.
        """
        train_dataset = self.train_dataset
        sample_labels = train_dataset.sample_labels
        n_pos = sum(1 for l in sample_labels if l == 0)
        n_neg = sum(1 for l in sample_labels if l == 1)

        if n_pos == 0 or n_neg == 0:
            logger.warning(
                "Sampler fallback: only one class present (pos=%s, neg=%s). "
                "Using shuffled dataloader without reweighting.",
                n_pos,
                n_neg,
            )
            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=self.args.dataloader_drop_last,
            )

        natural_neg_fraction = n_neg / (n_pos + n_neg)
        target_neg_fraction = getattr(self, "_target_negative_fraction", TARGET_NEGATIVE_FRACTION)

        weight_pos = 1.0
        weight_neg = (
            target_neg_fraction * n_pos /
            ((1.0 - target_neg_fraction) * n_neg)
        )
        sample_weights = [weight_neg if l == 1 else weight_pos for l in sample_labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
        )
        logger.info(
            "Soft-balanced sampler: pos=%s neg=%s natural_neg=%.4f "
            "target_neg=%.4f weights(pos=%.4f, neg=%.4f)",
            f"{n_pos:,}",
            f"{n_neg:,}",
            natural_neg_fraction,
            target_neg_fraction,
            weight_pos,
            weight_neg,
        )
        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override default prediction_step because HF Trainer does outputs[1:] on
        the model output, but our ClfOutput is not subscriptable.
        We manually run compute_loss(return_outputs=True) and extract .logits.
        """
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")

        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            # Wrap in tuple so preprocess_logits_for_metrics receives (B, 2) after unwrap
            logits = (outputs.logits.detach(),)

        if prediction_loss_only:
            return (loss, None, None)

        return (loss, logits, labels)


class VerifierWandbDebugCallback(TrainerCallback):
    """Log prompt-format debug tables and sampled prediction tables to W&B."""

    def __init__(
        self,
        train_dataset: VerifierDataset,
        eval_dataset: VerifierDataset,
        data_collator: PadCollator,
        debug_rows: int = WANDB_DEBUG_TABLE_ROWS,
        stats_sample_size: int = WANDB_STATS_SAMPLE_SIZE,
        seed: int = 42,
    ):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.debug_rows = debug_rows
        self.stats_sample_size = stats_sample_size
        self.seed = seed
        self.train_indices = self._sample_indices(len(train_dataset), debug_rows, seed)
        self.eval_indices = self._sample_indices(len(eval_dataset), debug_rows, seed + 1)

    @staticmethod
    def _sample_indices(dataset_len: int, n_rows: int, seed: int) -> List[int]:
        if dataset_len == 0:
            return []
        rng = random.Random(seed)
        count = min(dataset_len, n_rows)
        return sorted(rng.sample(range(dataset_len), count))

    @staticmethod
    def _wandb_run():
        if wandb is None:
            return None
        return wandb.run

    def _prompt_table(self, dataset: VerifierDataset, indices: List[int], split: str):
        run = self._wandb_run()
        if run is None:
            return None

        columns = [
            "split",
            "dataset_idx",
            "label_name",
            "n_steps_in_prompt",
            "prompt_len_tokens",
            "raw_prompt_len_tokens",
            "dropped_tokens",
            "truncated",
            "has_current_step_marker",
            "has_answer_suffix",
            "problem",
            "current_step",
            "model_input_text",
            "prompt_text",
        ]
        table = wandb.Table(columns=columns)
        for idx in indices:
            row = dataset.prompt_debug_row(idx)
            table.add_data(
                split,
                row["dataset_idx"],
                row["label_name"],
                row["n_steps_in_prompt"],
                row["prompt_len_tokens"],
                row["raw_prompt_len_tokens"],
                row["dropped_tokens"],
                row["truncated"],
                row["has_current_step_marker"],
                row["has_answer_suffix"],
                row["problem"],
                row["current_step"],
                row["model_input_text"],
                row["prompt_text"],
            )
        return table

    def _prediction_table(self, model, state) -> Optional["wandb.Table"]:
        run = self._wandb_run()
        if run is None or not self.eval_indices:
            return None

        unwrapped = model.module if hasattr(model, "module") else model
        features = [self.eval_dataset[idx] for idx in self.eval_indices]
        batch = self.data_collator(features)
        model_device = next(unwrapped.parameters()).device
        input_ids = batch["input_ids"].to(model_device)
        attention_mask = batch["attention_mask"].to(model_device)

        was_training = unwrapped.training
        unwrapped.eval()
        with torch.no_grad():
            outputs = unwrapped(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            probs = torch.softmax(outputs.logits, dim=-1)[:, 0].float().cpu().numpy()
        if was_training:
            unwrapped.train()

        pred_cls = np.where(probs >= 0.5, 0, 1)

        columns = [
            "global_step",
            "dataset_idx",
            "label_name",
            "pred_label_name",
            "pos_prob",
            "correct",
            "n_steps_in_prompt",
            "prompt_len_tokens",
            "raw_prompt_len_tokens",
            "dropped_tokens",
            "truncated",
            "has_current_step_marker",
            "has_answer_suffix",
            "problem",
            "current_step",
            "model_input_text",
            "prompt_text",
        ]
        table = wandb.Table(columns=columns)
        for idx, pos_prob, pred_label in zip(self.eval_indices, probs, pred_cls):
            row = self.eval_dataset.prompt_debug_row(idx)
            gold_label = row["label"]
            table.add_data(
                state.global_step,
                row["dataset_idx"],
                row["label_name"],
                "positive" if pred_label == 0 else "negative",
                float(pos_prob),
                bool(pred_label == gold_label),
                row["n_steps_in_prompt"],
                row["prompt_len_tokens"],
                row["raw_prompt_len_tokens"],
                row["dropped_tokens"],
                row["truncated"],
                row["has_current_step_marker"],
                row["has_answer_suffix"],
                row["problem"],
                row["current_step"],
                row["model_input_text"],
                row["prompt_text"],
            )
        return table

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        run = self._wandb_run()
        if run is None:
            return

        train_stats = self.train_dataset.sampled_prompt_stats(self.stats_sample_size, self.seed + 10)
        eval_stats = self.eval_dataset.sampled_prompt_stats(self.stats_sample_size, self.seed + 20)

        run.summary["prompt_debug/train_sample_size"] = train_stats["sample_size"]
        run.summary["prompt_debug/eval_sample_size"] = eval_stats["sample_size"]

        payload = {
            "prompt_debug/train_examples": self._prompt_table(self.train_dataset, self.train_indices, "train"),
            "prompt_debug/eval_examples": self._prompt_table(self.eval_dataset, self.eval_indices, "eval"),
            "prompt_debug/train_prompt_len_mean": train_stats["prompt_len_mean"],
            "prompt_debug/train_prompt_len_p95": train_stats["prompt_len_p95"],
            "prompt_debug/train_truncation_rate": train_stats["truncation_rate"],
            "prompt_debug/eval_prompt_len_mean": eval_stats["prompt_len_mean"],
            "prompt_debug/eval_prompt_len_p95": eval_stats["prompt_len_p95"],
            "prompt_debug/eval_truncation_rate": eval_stats["truncation_rate"],
        }
        wandb.log(payload, step=state.global_step)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        run = self._wandb_run()
        if run is None or model is None:
            return

        prediction_table = self._prediction_table(model, state)
        if prediction_table is not None:
            wandb.log({"prompt_debug/eval_predictions": prediction_table}, step=state.global_step)


def preprocess_logits_for_metrics(logits, labels):
    """
    prediction_step returns logits as a 1-tuple: (tensor_B_2,).
    Unwrap so compute_metrics receives a plain (N, 2) array.
    """
    if isinstance(logits, (tuple, list)):
        return logits[0]
    return logits


def compute_metrics(eval_pred):
    """Accuracy + per-class stats on binary classification."""
    predictions, labels = eval_pred    # predictions: (n, 2); labels: (n,)

    # predicted class index (0 = positive, 1 = negative)
    pred_cls = np.argmax(predictions, axis=1)
    true_cls = np.asarray(labels).reshape(-1)

    acc     = (pred_cls == true_cls).mean()
    pos_acc = (pred_cls[true_cls == 0] == 0).mean() if (true_cls == 0).any() else 0.0
    neg_acc = (pred_cls[true_cls == 1] == 1).mean() if (true_cls == 1).any() else 0.0
    balanced_acc = 0.5 * (pos_acc + neg_acc)
    return {
        "accuracy":          float(acc),
        "pos_accuracy":      float(pos_acc),      # recall on correct steps
        "neg_accuracy":      float(neg_acc),      # recall on incorrect steps
        "balanced_accuracy": float(balanced_acc),
    }


# — Main ——————————————————————————————————————

def main():
    logger.info(f"Loading model & tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Keep the tail of the verifier prompt, where [Current step] and "Answer:"
    # live, when long problem/context forces truncation.
    tokenizer.truncation_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",        # fills available GPUs automatically
    )

    # Wrap with classification head
    hidden_size = base_model.config.hidden_size  # 1536 for Qwen2.5-Math-1.5B
    model = PRMClassifier(base_model, hidden_size)
    # Move classification head to same device/dtype as base model
    first_device = next(base_model.parameters()).device
    model.score = model.score.to(device=first_device, dtype=torch.bfloat16)
    logger.info(f"PRMClassifier: hidden_size={hidden_size}, score on {first_device}")

    # Load from cached arrow files (bypass HuggingFace Hub)
    import glob
    arrow_dir = glob.glob("/root/autodl-tmp/prm_grpo/datasets/prm800k/trl-lib___prm800k/default/0.0.0/*/")[0]
    logger.info(f"Loading dataset from arrow cache: {arrow_dir}")
    ds = DatasetDict({
        "train": HFDataset.from_file(arrow_dir + "prm800k-train.arrow"),
        "test":  HFDataset.from_file(arrow_dir + "prm800k-test.arrow"),
    })

    train_ds = VerifierDataset(ds["train"], tokenizer)
    eval_ds  = VerifierDataset(ds["test"],  tokenizer)
    collator = PadCollator(tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir          = OUTPUT_DIR,
        num_train_epochs     = 3,
        per_device_train_batch_size = 4,
        per_device_eval_batch_size  = 8,
        gradient_accumulation_steps = 8,
        learning_rate        = 1e-5,
        warmup_ratio         = 0.05,
        weight_decay         = 0.01,
        fp16                 = False,
        bf16                 = True,
        logging_steps        = 50,
        eval_strategy        = "steps",
        eval_steps           = 500,
        save_strategy        = "steps",
        save_steps           = 500,
        save_total_limit     = 2,
        load_best_model_at_end = True,
        dataloader_num_workers = 2,
        remove_unused_columns  = False,
        report_to            = "wandb",
        run_name             = "prm-cls-softneg40-qwen25-math-1.5b",
        metric_for_best_model  = "eval_balanced_accuracy",
        greater_is_better      = True,
    )

    # Compute class weights from actual data distribution
    pos_weight = 1.0
    neg_weight = 1.0  # balanced sampling handles imbalance; no extra loss weight needed
    logger.info(f"Class weights: pos={pos_weight:.2f}, neg={neg_weight:.2f} (balanced sampling active)")

    trainer = ClassificationTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = eval_ds,
        data_collator   = collator,
        tokenizer       = tokenizer,
        callbacks       = [
            EarlyStoppingCallback(
                early_stopping_patience=5,
                early_stopping_threshold=0.001,
            ),
            VerifierWandbDebugCallback(
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                data_collator=collator,
            ),
        ],
        compute_metrics                = compute_metrics,
        preprocess_logits_for_metrics  = preprocess_logits_for_metrics,
    )

    trainer._class_weights = [pos_weight, neg_weight]
    trainer._target_negative_fraction = TARGET_NEGATIVE_FRACTION
    logger.info("Starting verifier training (classification head)...")
    trainer.train()

    save_path = os.path.join(OUTPUT_DIR, "final")
    logger.info(f"Saving final model to {save_path}")
    trainer.save_model(save_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
