#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-Math-1.5B-Instruct}"
DATA_DIR="${DATA_DIR:-data/trl_math}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/trl_grpo_math_instruct_verifier}"
RUN_NAME="${RUN_NAME:-trl-grpo-math-instruct-verifier}"

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

VERIFIER_MODEL_PATH="${VERIFIER_MODEL_PATH:-token_prm_runs/extra0-prm/final}"
VERIFIER_BACKEND="${VERIFIER_BACKEND:-auto}"
VERIFIER_DEVICE="${VERIFIER_DEVICE:-cuda}"
VERIFIER_MAX_LENGTH="${VERIFIER_MAX_LENGTH:-1024}"
VERIFIER_BATCH_SIZE="${VERIFIER_BATCH_SIZE:-4}"
VERIFIER_BETA="${VERIFIER_BETA:-0.05}"
VERIFIER_DELTA="${VERIFIER_DELTA:-0.05}"
VERIFIER_THRESHOLD="${VERIFIER_THRESHOLD:-0.4}"
VERIFIER_REWARD_MODE="${VERIFIER_REWARD_MODE:-wrong_only}"
VERIFIER_TIEBREAK_ONLY="${VERIFIER_TIEBREAK_ONLY:-0}"
VERIFIER_SERVER_URL="${VERIFIER_SERVER_URL:-http://127.0.0.1:8008}"
VERIFIER_SERVER_TIMEOUT="${VERIFIER_SERVER_TIMEOUT:-60}"

# Main beta sweep examples for wrong-only shaping:
#   VERIFIER_REWARD_MODE=wrong_only VERIFIER_BETA=0.05 bash scripts/trl/run_trl_grpo_math_verifier.sh
#   VERIFIER_REWARD_MODE=wrong_only VERIFIER_BETA=0.1  bash scripts/trl/run_trl_grpo_math_verifier.sh
#   VERIFIER_REWARD_MODE=wrong_only VERIFIER_BETA=0.2  bash scripts/trl/run_trl_grpo_math_verifier.sh
# Optional ablation:
#   VERIFIER_REWARD_MODE=all_wrong_tiebreak bash scripts/trl/run_trl_grpo_math_verifier.sh

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
  --verifier-backend "$VERIFIER_BACKEND"
  --verifier-device "$VERIFIER_DEVICE"
  --verifier-max-length "$VERIFIER_MAX_LENGTH"
  --verifier-batch-size "$VERIFIER_BATCH_SIZE"
  --verifier-beta "$VERIFIER_BETA"
  --verifier-delta "$VERIFIER_DELTA"
  --verifier-threshold "$VERIFIER_THRESHOLD"
  --verifier-reward-mode "$VERIFIER_REWARD_MODE"
  --verifier-server-url "$VERIFIER_SERVER_URL"
  --verifier-server-timeout "$VERIFIER_SERVER_TIMEOUT"
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
