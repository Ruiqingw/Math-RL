#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from scripts.trl.rewards import MathVerifierReward, math_boxed_reward


DEFAULT_MODEL_PATH = "/root/autodl-tmp/prm_grpo/models/Qwen2.5-Math-1.5B"
DEFAULT_DATA_DIR = "/root/autodl-tmp/prm_grpo/data/trl_math"
DEFAULT_WANDB_PROJECT = "math_rl_trl"
DEFAULT_VERIFIER_MODEL_PATH = "/root/autodl-tmp/prm_grpo/verifier_cls/checkpoint-6000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TRL GRPO verifier-shaped run on MATH.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default="/root/autodl-tmp/prm_grpo/outputs/trl_grpo_math_verifier")
    parser.add_argument("--run-name", default="trl-grpo-math-verifier")
    parser.add_argument("--train-max-samples", type=int, default=2000)
    parser.add_argument("--eval-max-samples", type=int, default=50)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=1536)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-mode", default="colocate", choices=["colocate", "server"])
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--report-to", default="wandb")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--log-completions-samples", type=int, default=8)
    parser.add_argument("--verifier-model-path", default=DEFAULT_VERIFIER_MODEL_PATH)
    parser.add_argument("--verifier-device", default="cuda")
    parser.add_argument("--verifier-max-length", type=int, default=1536)
    parser.add_argument("--verifier-batch-size", type=int, default=1)
    parser.add_argument("--verifier-beta", type=float, default=0.1)
    parser.add_argument("--verifier-delta", type=float, default=0.05)
    parser.add_argument("--verifier-threshold", type=float, default=0.5)
    return parser.parse_args()


class WandbVerifierEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, num_samples: int, max_new_tokens: int):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        try:
            import wandb
        except Exception:
            return

        if model is None or wandb.run is None or len(self.eval_dataset) == 0:
            return

        model.eval()
        table = wandb.Table(columns=["prompt", "completion", "gold_answer", "accuracy"])

        total_correct = 0.0
        total_seen = 0
        num_table_rows = min(self.num_samples, len(self.eval_dataset))

        for idx in range(len(self.eval_dataset)):
            sample = self.eval_dataset[idx]
            prompt = sample["prompt"]
            gold_answer = sample["gold_answer"]
            encoded = self.tokenizer(prompt, return_tensors="pt")
            encoded = {k: v.to(model.device) for k, v in encoded.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generated_ids = outputs[0][encoded["input_ids"].shape[1] :]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            accuracy = math_boxed_reward([prompt], [generated_text], [gold_answer])[0]
            total_correct += accuracy
            total_seen += 1

            if idx < num_table_rows:
                table.add_data(prompt, generated_text, gold_answer, accuracy)

        metrics = {
            "eval/accuracy": total_correct / max(total_seen, 1),
            "completions": table,
        }
        composite_metric = None
        logs = kwargs.get("metrics") or {}
        for key in ("eval_rewards/MathVerifierReward/mean", "eval_reward", "eval/reward"):
            if key in logs:
                composite_metric = logs[key]
                break
        if composite_metric is not None:
            metrics["eval/composite_reward"] = composite_metric
        wandb.log(metrics, step=state.global_step)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    train_path = os.path.join(args.data_dir, "train.parquet")
    test_path = os.path.join(args.data_dir, "test.parquet")

    train_dataset = load_dataset("parquet", data_files=train_path, split="train")
    eval_dataset = load_dataset("parquet", data_files=test_path, split="train")

    if args.train_max_samples is not None:
        train_dataset = train_dataset.select(range(min(args.train_max_samples, len(train_dataset))))
    if args.eval_max_samples is not None:
        eval_dataset = eval_dataset.select(range(min(args.eval_max_samples, len(eval_dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        bf16=args.bf16,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        report_to=args.report_to,
        log_completions=True,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        seed=args.seed,
    )

    reward_func = MathVerifierReward(
        verifier_model_path=args.verifier_model_path,
        verifier_device=args.verifier_device,
        verifier_max_length=args.verifier_max_length,
        verifier_batch_size=args.verifier_batch_size,
        verifier_beta=args.verifier_beta,
        verifier_delta=args.verifier_delta,
        verifier_threshold=args.verifier_threshold,
    )

    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[
            WandbVerifierEvalCallback(
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                num_samples=args.log_completions_samples,
                max_new_tokens=args.max_completion_length,
            )
        ],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
