#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/prm_grpo/models/Qwen2.5-Math-1.5B}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/prm_grpo/data/trl_math}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/prm_grpo/outputs/trl_grpo_math_verifier}"
RUN_NAME="${RUN_NAME:-trl-grpo-math-verifier}"

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
BETA="${BETA:-0.001}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-50}"

USE_VLLM="${USE_VLLM:-1}"
VLLM_MODE="${VLLM_MODE:-colocate}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.3}"

VERIFIER_MODEL_PATH="${VERIFIER_MODEL_PATH:-/root/autodl-tmp/prm_grpo/token_prm_runs/prm-token-fullft-phase2raw-nonneg-firsterr-eval12-negw10p0-focalg2p0-qwen25-math-1.5b/final}"
VERIFIER_DEVICE="${VERIFIER_DEVICE:-cuda}"
VERIFIER_MAX_LENGTH="${VERIFIER_MAX_LENGTH:-1024}"
VERIFIER_BATCH_SIZE="${VERIFIER_BATCH_SIZE:-4}"
VERIFIER_BETA="${VERIFIER_BETA:-0.3}"
VERIFIER_DELTA="${VERIFIER_DELTA:-0.05}"
VERIFIER_THRESHOLD="${VERIFIER_THRESHOLD:-0.4}"
VERIFIER_TIEBREAK_ONLY="${VERIFIER_TIEBREAK_ONLY:-1}"

cd "$PROJECT_ROOT"

ARGS=(
  --model-path "$MODEL_PATH"
  --data-dir "$DATA_DIR"
  --output-dir "$OUTPUT_DIR"
  --run-name "$RUN_NAME"
  --train-max-samples "$TRAIN_MAX_SAMPLES"
  --eval-max-samples "$EVAL_MAX_SAMPLES"
  --max-prompt-length "$MAX_PROMPT_LENGTH"
  --max-completion-length "$MAX_COMPLETION_LENGTH"
  --per-device-train-batch-size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per-device-eval-batch-size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  --num-generations "$NUM_GENERATIONS"
  --num-generations-eval "$NUM_GENERATIONS_EVAL"
  --learning-rate "$LEARNING_RATE"
  --beta "$BETA"
  --num-train-epochs "$NUM_TRAIN_EPOCHS"
  --logging-steps "$LOGGING_STEPS"
  --eval-steps "$EVAL_STEPS"
  --save-steps "$SAVE_STEPS"
  --bf16
  --verifier-model-path "$VERIFIER_MODEL_PATH"
  --verifier-device "$VERIFIER_DEVICE"
  --verifier-max-length "$VERIFIER_MAX_LENGTH"
  --verifier-batch-size "$VERIFIER_BATCH_SIZE"
  --verifier-beta "$VERIFIER_BETA"
  --verifier-delta "$VERIFIER_DELTA"
  --verifier-threshold "$VERIFIER_THRESHOLD"
)

if [[ "$VERIFIER_TIEBREAK_ONLY" == "1" ]]; then
  ARGS+=(--verifier-tiebreak-only)
else
  ARGS+=(--no-verifier-tiebreak-only)
fi

if [[ "$USE_VLLM" == "1" ]]; then
  ARGS+=(--use-vllm --vllm-mode "$VLLM_MODE" --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
fi

accelerate launch scripts/trl/train_grpo_math_verifier.py "${ARGS[@]}" "$@"
