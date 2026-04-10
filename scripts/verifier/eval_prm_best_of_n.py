#!/usr/bin/env python3
"""
Offline PRM reranking evaluation on a small MATH test subset.

This compares three base-model inference strategies on the same prompts:

1. Greedy decoding at T=0.
2. Majority voting over N sampled completions at T=0.8.
3. PRM best-of-N: score the same N sampled completions with a token PRM and
   take the highest-scoring completion.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any

import datasets
import torch
from tqdm import tqdm

from verl.utils.reward_score.math_reward import compute_score, last_boxed_only_string, remove_boxed


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from step_splitter import split_into_steps  # noqa: E402
from token_reward_fn import load_model_bundle, score_steps  # noqa: E402


DEFAULT_MODEL_PATH = "/root/autodl-tmp/prm_grpo/models/Qwen2.5-Math-1.5B"
DEFAULT_VERIFIER_MODEL_PATH = (
    "/root/autodl-tmp/prm_grpo/token_prm_runs/"
    "prm-token-fullft-phase2raw-nonneg-firsterr-eval12-negw10p0-focalg2p0-qwen25-math-1.5b/"
    "final"
)
DEFAULT_DATA_DIR = "/root/autodl-tmp/prm_grpo/data/trl_math"
DEFAULT_OUTPUT_JSONL = (
    "/root/autodl-tmp/prm_grpo/outputs/prm_best_of_n/"
    "math_test_100_best_of_16.jsonl"
)
DEFAULT_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


@dataclass
class EvalExample:
    idx: int
    prompt: str
    problem: str
    gold_answer: str
    solution: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base greedy, majority vote, and token-PRM best-of-N on MATH."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Base model path.")
    parser.add_argument(
        "--verifier-model-path",
        default=DEFAULT_VERIFIER_MODEL_PATH,
        help="Token PRM checkpoint path.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing TRL-style train.parquet/test.parquet.",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split.")
    parser.add_argument("--max-samples", type=int, default=100, help="Number of MATH problems.")
    parser.add_argument("--num-generations", type=int, default=16, help="Sampled completions per prompt.")
    parser.add_argument("--sample-temperature", type=float, default=0.8, help="Temperature for sampled completions.")
    parser.add_argument("--sample-top-p", type=float, default=1.0, help="Top-p for sampled completions.")
    parser.add_argument("--greedy-temperature", type=float, default=0.0, help="Temperature for greedy baseline.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Base model max new tokens.")
    parser.add_argument("--generation-batch-size", type=int, default=16, help="vLLM prompt batch size.")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.55, help="vLLM GPU memory fraction.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="vLLM dtype.")
    parser.add_argument(
        "--verifier-device",
        default="cuda",
        help="Device used by the token PRM, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument("--verifier-max-length", type=int, default=1024, help="PRM prompt max length.")
    parser.add_argument("--verifier-batch-size", type=int, default=8, help="PRM step scoring batch size.")
    parser.add_argument(
        "--prm-aggregation",
        default="mean_log",
        choices=["mean_log", "sum_log", "mean", "min"],
        help="How to aggregate step-level PRM positive probabilities into one completion score.",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the split before taking max-samples.")
    parser.add_argument("--seed", type=int, default=817, help="Seed used for dataset shuffling and vLLM sampling.")
    parser.add_argument(
        "--instruction",
        default=DEFAULT_INSTRUCTION,
        help="Instruction suffix if the parquet row has no prebuilt prompt.",
    )
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL, help="Per-problem output path.")
    return parser.parse_args()


def safe_math_score(solution_text: str, gold_answer: str) -> float:
    try:
        return float(compute_score(solution_text or "", ground_truth=gold_answer))
    except Exception:
        return 0.0


def extract_boxed_answer(text: str) -> str:
    if not text:
        return ""
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return ""
    try:
        return str(remove_boxed(boxed)).strip()
    except Exception:
        return ""


def build_prompt(problem: str, instruction: str) -> str:
    return f"{problem} {instruction}".strip()


def load_examples(args: argparse.Namespace) -> list[EvalExample]:
    parquet_path = os.path.join(args.data_dir, f"{args.split}.parquet")
    dataset = datasets.load_dataset("parquet", data_files=parquet_path, split="train")
    if args.shuffle:
        dataset = dataset.shuffle(seed=args.seed)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    examples: list[EvalExample] = []
    for local_idx, row in enumerate(dataset):
        problem = str(row.get("problem", "") or "")
        prompt = str(row.get("prompt", "") or "").strip() or build_prompt(problem, args.instruction)
        examples.append(
            EvalExample(
                idx=int(row.get("index", local_idx) if row.get("index", local_idx) is not None else local_idx),
                prompt=prompt,
                problem=problem,
                gold_answer=str(row.get("gold_answer", "") or ""),
                solution=str(row.get("solution", "") or ""),
            )
        )
    return examples


def make_sampling_params(
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    n: int,
    seed: int | None,
):
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p if temperature > 0 else 1.0,
        "max_tokens": max_new_tokens,
        "n": n,
    }
    if seed is not None:
        kwargs["seed"] = seed
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("seed", None)
        return SamplingParams(**kwargs)


def load_vllm_model(model_path: str, dtype: str, gpu_memory_utilization: float):
    from vllm import LLM

    return LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
    )


def generate_batches(llm, prompts: list[str], sampling_params, batch_size: int) -> list[list[str]]:
    completions: list[list[str]] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[start : start + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        for output in outputs:
            completions.append([candidate.text for candidate in output.outputs])
    return completions


def release_generation_model(llm: Any) -> None:
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def majority_vote_answer(sampled_texts: list[str]) -> tuple[str, int]:
    answers = [extract_boxed_answer(text) for text in sampled_texts]
    answers = [answer for answer in answers if answer]
    if not answers:
        return "", 0

    counts = Counter(answers)
    best_count = max(counts.values())
    for answer in answers:
        if counts[answer] == best_count:
            return answer, best_count
    return "", 0


def score_majority_answer(answer: str, gold_answer: str) -> float:
    if not answer:
        return 0.0
    return safe_math_score(f"\\boxed{{{answer}}}", gold_answer)


def aggregate_step_scores(step_scores: list[float], mode: str) -> float:
    if not step_scores:
        return float("-inf")
    if mode == "mean":
        return float(sum(step_scores) / len(step_scores))
    if mode == "min":
        return float(min(step_scores))

    clamped = [min(max(score, 1e-6), 1.0) for score in step_scores]
    log_scores = [math.log(score) for score in clamped]
    if mode == "sum_log":
        return float(sum(log_scores))
    if mode == "mean_log":
        return float(sum(log_scores) / len(log_scores))
    raise ValueError(f"Unsupported PRM aggregation: {mode}")


def score_completions_with_prm(
    examples: list[EvalExample],
    sampled_completions: list[list[str]],
    args: argparse.Namespace,
) -> list[list[dict[str, Any]]]:
    device_map = None if args.verifier_device == "cpu" else "auto"
    model, tokenizer, label_tokens = load_model_bundle(
        args.verifier_model_path,
        device_map=device_map,
    )

    all_scores: list[list[dict[str, Any]]] = []
    iterator = zip(examples, sampled_completions)
    for example, completion_group in tqdm(iterator, total=len(examples), desc="Scoring with PRM"):
        group_scores: list[dict[str, Any]] = []
        for text in completion_group:
            steps = split_into_steps(text)
            step_scores = score_steps(
                example.problem,
                steps,
                model,
                tokenizer,
                label_tokens,
                device=args.verifier_device,
                max_length=args.verifier_max_length,
                batch_size=args.verifier_batch_size,
            )
            group_scores.append(
                {
                    "score": aggregate_step_scores(step_scores, args.prm_aggregation),
                    "step_scores": step_scores,
                    "n_steps": len(steps),
                }
            )
        all_scores.append(group_scores)
    return all_scores


def best_index(scores: list[dict[str, Any]]) -> int:
    if not scores:
        return -1
    return max(range(len(scores)), key=lambda idx: scores[idx]["score"])


def main() -> None:
    args = parse_args()
    examples = load_examples(args)
    if not examples:
        raise ValueError("No evaluation examples loaded.")

    prompts = [example.prompt for example in examples]
    output_dir = os.path.dirname(args.output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("PRM Best-of-N MATH Eval")
    print("=" * 80)
    print(f"base_model       : {args.model_path}")
    print(f"verifier_model   : {args.verifier_model_path}")
    print(f"data             : {os.path.join(args.data_dir, f'{args.split}.parquet')}")
    print(f"num_examples     : {len(examples)}")
    print(f"num_generations  : {args.num_generations}")
    print(f"sample_temperature: {args.sample_temperature}")
    print(f"prm_aggregation  : {args.prm_aggregation}")

    llm = load_vllm_model(args.model_path, args.dtype, args.vllm_gpu_memory_utilization)
    greedy_params = make_sampling_params(
        temperature=args.greedy_temperature,
        top_p=1.0,
        max_new_tokens=args.max_new_tokens,
        n=1,
        seed=args.seed,
    )
    sample_params = make_sampling_params(
        temperature=args.sample_temperature,
        top_p=args.sample_top_p,
        max_new_tokens=args.max_new_tokens,
        n=args.num_generations,
        seed=args.seed,
    )

    print("\nGenerating greedy completions...")
    greedy_completions = generate_batches(llm, prompts, greedy_params, args.generation_batch_size)
    greedy_texts = [group[0] if group else "" for group in greedy_completions]

    print("\nGenerating sampled completions...")
    sampled_completions = generate_batches(llm, prompts, sample_params, args.generation_batch_size)
    release_generation_model(llm)

    print("\nScoring sampled completions with PRM...")
    prm_scores = score_completions_with_prm(examples, sampled_completions, args)

    greedy_correct = []
    majority_correct = []
    prm_best_correct = []
    oracle_correct = []
    boxed_counts = []

    with open(args.output_jsonl, "w", encoding="utf-8") as writer:
        for row_idx, example in enumerate(examples):
            sampled_texts = sampled_completions[row_idx]
            sample_scores = [safe_math_score(text, example.gold_answer) for text in sampled_texts]
            sample_answers = [extract_boxed_answer(text) for text in sampled_texts]
            majority_answer, majority_count = majority_vote_answer(sampled_texts)
            best_prm_idx = best_index(prm_scores[row_idx])
            best_prm_text = sampled_texts[best_prm_idx] if best_prm_idx >= 0 else ""

            greedy_score = safe_math_score(greedy_texts[row_idx], example.gold_answer)
            majority_score = score_majority_answer(majority_answer, example.gold_answer)
            prm_score = safe_math_score(best_prm_text, example.gold_answer)
            oracle_score = max(sample_scores) if sample_scores else 0.0
            boxed_count = sum(1 for answer in sample_answers if answer)

            greedy_correct.append(greedy_score)
            majority_correct.append(majority_score)
            prm_best_correct.append(prm_score)
            oracle_correct.append(oracle_score)
            boxed_counts.append(boxed_count)

            sampled_records = []
            for sample_idx, text in enumerate(sampled_texts):
                score_record = prm_scores[row_idx][sample_idx]
                sampled_records.append(
                    {
                        "sample_idx": sample_idx,
                        "text": text,
                        "boxed_answer": sample_answers[sample_idx],
                        "math_correct": sample_scores[sample_idx],
                        "prm_score": score_record["score"],
                        "step_scores": score_record["step_scores"],
                        "n_steps": score_record["n_steps"],
                    }
                )

            record = {
                "row_idx": row_idx,
                "dataset_idx": example.idx,
                "prompt": example.prompt,
                "problem": example.problem,
                "gold_answer": example.gold_answer,
                "gold_solution": example.solution,
                "greedy_text": greedy_texts[row_idx],
                "greedy_boxed_answer": extract_boxed_answer(greedy_texts[row_idx]),
                "greedy_correct": greedy_score,
                "majority_answer": majority_answer,
                "majority_count": majority_count,
                "majority_correct": majority_score,
                "prm_best_index": best_prm_idx,
                "prm_best_score": prm_scores[row_idx][best_prm_idx]["score"] if best_prm_idx >= 0 else None,
                "prm_best_boxed_answer": extract_boxed_answer(best_prm_text),
                "prm_best_correct": prm_score,
                "sample_oracle_correct": oracle_score,
                "sample_boxed_count": boxed_count,
                "sampled": sampled_records,
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    n = len(examples)
    greedy_acc = sum(greedy_correct) / n
    majority_acc = sum(majority_correct) / n
    prm_best_acc = sum(prm_best_correct) / n
    oracle_acc = sum(oracle_correct) / n
    boxed_rate = sum(boxed_counts) / (n * args.num_generations)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"num_examples              : {n}")
    print(f"greedy_accuracy           : {greedy_acc:.4f}")
    print(f"majority_vote_accuracy    : {majority_acc:.4f}")
    print(f"prm_best_of_{args.num_generations}_accuracy : {prm_best_acc:.4f}")
    print(f"sample_oracle_accuracy    : {oracle_acc:.4f}")
    print(f"sample_boxed_rate         : {boxed_rate:.4f}")
    print(f"output_jsonl              : {args.output_jsonl}")


if __name__ == "__main__":
    main()
