#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/prm_grpo/models/Qwen2.5-Math-1.5B}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/prm_grpo/data/trl_math}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/prm_grpo/outputs/trl_grpo_math_mc}"
RUN_NAME="${RUN_NAME:-trl-grpo-math-mc}"

TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-7500}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-200}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
NUM_GENERATIONS_EVAL="${NUM_GENERATIONS_EVAL:-1}"

LEARNING_RATE="${LEARNING_RATE:-3e-5}"
BETA="${BETA:-0.0}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-50}"

USE_VLLM="${USE_VLLM:-1}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.3}"

MC_BLAME_MODEL_PATH="${MC_BLAME_MODEL_PATH:-$MODEL_PATH}"
MC_BLAME_DEVICE="${MC_BLAME_DEVICE:-cuda}"
MC_BLAME_BETA="${MC_BLAME_BETA:-0.5}"
MC_BLAME_MAX_NEW_TOKENS="${MC_BLAME_MAX_NEW_TOKENS:-512}"

cd "$PROJECT_ROOT"

ARGS=(--model-path "$MODEL_PATH" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" --run-name "$RUN_NAME" --train-max-samples "$TRAIN_MAX_SAMPLES" --eval-max-samples "$EVAL_MAX_SAMPLES" --max-prompt-length "$MAX_PROMPT_LENGTH" --max-completion-length "$MAX_COMPLETION_LENGTH" --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE" --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE" --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" --num-generations "$NUM_GENERATIONS" --num-generations-eval "$NUM_GENERATIONS_EVAL" --learning-rate "$LEARNING_RATE" --beta "$BETA" --num-train-epochs "$NUM_TRAIN_EPOCHS" --logging-steps "$LOGGING_STEPS" --eval-steps "$EVAL_STEPS" --save-steps "$SAVE_STEPS" --bf16 --mc-blame-model-path "$MC_BLAME_MODEL_PATH" --mc-blame-device "$MC_BLAME_DEVICE" --mc-blame-beta "$MC_BLAME_BETA" --mc-blame-max-new-tokens "$MC_BLAME_MAX_NEW_TOKENS")

if [[ "$USE_VLLM" == "1" ]]; then
  ARGS+=(--use-vllm --vllm-mode "$VLLM_MODE" --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
fi

accelerate launch scripts/trl/train_grpo_math_mc.py "${ARGS[@]}" "$@"
