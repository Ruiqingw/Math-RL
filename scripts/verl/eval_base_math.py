#!/usr/bin/env python3
"""
Evaluate a base causal LM on MATH-lighteval using plain-text prompts.

This is intentionally separate from the verl GRPO pipeline:
  - no chat template
  - no RL loop
  - direct generation from a base model
  - score with verl's math_reward
"""

import argparse
import json
import os
from typing import Any

import datasets
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.reward_score.math_reward import compute_score, last_boxed_only_string, remove_boxed


DEFAULT_MODEL_PATH = "/root/autodl-tmp/prm_grpo/models/Qwen2.5-Math-1.5B"
DEFAULT_DATASET = "DigitalLearningGmbH/MATH-lighteval"
DEFAULT_OUTPUT_JSONL = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a base model on MATH-lighteval.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Base model path.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="HF dataset name or local dataset path loadable by datasets.load_dataset().",
    )
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split to evaluate.")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "hf"], help="Generation backend.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype.")
    parser.add_argument("--max-samples", type=int, default=None, help="How many samples to evaluate. Default: full split.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation max_new_tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. 0 means greedy.")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p for sampling when temperature > 0.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for vLLM generation.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5, help="vLLM GPU memory utilization.")
    parser.add_argument(
        "--instruction",
        default="Solve the following math problem step by step. Put the final answer inside \\boxed{}.",
        help="Instruction suffix appended to each problem.",
    )
    parser.add_argument("--output-jsonl", default=DEFAULT_OUTPUT_JSONL, help="Optional path to save per-sample outputs.")
    return parser.parse_args()


def get_torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def extract_solution(solution_str: str) -> str:
    return remove_boxed(last_boxed_only_string(solution_str))


def build_prompt(problem: str, instruction: str) -> str:
    return f"Problem:\n{problem}\n\n{instruction}\n\nSolution:\n"


def load_model_and_tokenizer(model_path: str, device: str, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        hf_device_map: Any = {"": 0}
    elif device.startswith("cuda:"):
        hf_device_map = {"": int(device.split(":", 1)[1])}
    else:
        hf_device_map = {"": device}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=hf_device_map,
    )
    model.eval()
    return model, tokenizer


def load_vllm_model(model_path: str, dtype_name: str, gpu_memory_utilization: float):
    from vllm import LLM

    return LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=dtype_name,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
    )


def build_sampling_params(args: argparse.Namespace):
    from vllm import SamplingParams

    if args.temperature > 0:
        return SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )


def main() -> None:
    args = parse_args()
    dtype = get_torch_dtype(args.dtype)

    dataset = datasets.load_dataset(args.dataset)[args.split]
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True) if args.output_jsonl else None
    writer = open(args.output_jsonl, "w", encoding="utf-8") if args.output_jsonl else None

    total = 0
    total_correct = 0.0
    boxed_count = 0

    if args.backend == "hf":
        model, tokenizer = load_model_and_tokenizer(args.model_path, args.device, dtype)

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        iterable = enumerate(tqdm(dataset, desc=f"Evaluating {args.split}"))
        for idx, example in iterable:
            problem = example["problem"]
            answer_raw = example["solution"]
            gold_answer = extract_solution(answer_raw)
            prompt = build_prompt(problem, args.instruction)

            encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
            with torch.no_grad():
                outputs = model.generate(**encoded, **gen_kwargs)

            generated_ids = outputs[0][encoded["input_ids"].shape[1] :]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            score = float(compute_score(generated_text, ground_truth=gold_answer))
            total += 1
            total_correct += score
            if "\\boxed" in generated_text:
                boxed_count += 1

            record = {
                "idx": idx,
                "prompt": prompt,
                "generated_text": generated_text,
                "gold_solution_raw": answer_raw,
                "gold_answer": gold_answer,
                "reward": score,
            }
            if writer is not None:
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        llm = load_vllm_model(args.model_path, args.dtype, args.gpu_memory_utilization)
        sampling_params = build_sampling_params(args)

        prompts = []
        answers_raw = []
        gold_answers = []
        for example in dataset:
            prompts.append(build_prompt(example["problem"], args.instruction))
            answers_raw.append(example["solution"])
            gold_answers.append(extract_solution(example["solution"]))

        for start in tqdm(range(0, len(prompts), args.batch_size), desc=f"Evaluating {args.split}"):
            end = min(start + args.batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            outputs = llm.generate(batch_prompts, sampling_params)

            for offset, output in enumerate(outputs):
                idx = start + offset
                generated_text = output.outputs[0].text if output.outputs else ""
                score = float(compute_score(generated_text, ground_truth=gold_answers[idx]))
                total += 1
                total_correct += score
                if "\\boxed" in generated_text:
                    boxed_count += 1

                record = {
                    "idx": idx,
                    "prompt": batch_prompts[offset],
                    "generated_text": generated_text,
                    "gold_solution_raw": answers_raw[idx],
                    "gold_answer": gold_answers[idx],
                    "reward": score,
                }
                if writer is not None:
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    if writer is not None:
        writer.close()

    accuracy = total_correct / total if total else 0.0
    boxed_rate = boxed_count / total if total else 0.0

    print("=" * 60)
    print("Base Model MATH Eval")
    print("=" * 60)
    print(f"model_path      : {args.model_path}")
    print(f"dataset         : {args.dataset}")
    print(f"split           : {args.split}")
    print(f"backend         : {args.backend}")
    print(f"num_samples     : {total}")
    print(f"accuracy        : {accuracy:.4f}")
    print(f"boxed_rate      : {boxed_rate:.4f}")
    if args.output_jsonl:
        print(f"output_jsonl    : {args.output_jsonl}")


if __name__ == "__main__":
    main()
