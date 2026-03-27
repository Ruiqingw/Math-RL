#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERL_ROOT="${VERL_ROOT:-/root/autodl-tmp/verl-clean}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/prm_grpo/models/Qwen2.5-Math-1.5B}"
OUTPUT_JSONL="${OUTPUT_JSONL:-/root/autodl-tmp/prm_grpo/outputs/base_math_eval_full_test_vllm.jsonl}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

mkdir -p "$(dirname "$OUTPUT_JSONL")"

CMD=(
  python
  "$PROJECT_ROOT/scripts/verl/eval_base_math.py"
  --backend
  vllm
  --model-path
  "$MODEL_PATH"
  --split
  "$SPLIT"
  --batch-size
  "$BATCH_SIZE"
  --gpu-memory-utilization
  "$GPU_MEMORY_UTILIZATION"
  --output-jsonl
  "$OUTPUT_JSONL"
)

if [[ -n "$MAX_SAMPLES" ]]; then
  CMD+=(
    --max-samples
    "$MAX_SAMPLES"
  )
fi

echo "Running base MATH eval with vLLM..."
echo "VERL_ROOT=$VERL_ROOT"
echo "MODEL_PATH=$MODEL_PATH"
echo "SPLIT=$SPLIT"
echo "BATCH_SIZE=$BATCH_SIZE"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
echo "OUTPUT_JSONL=$OUTPUT_JSONL"
if [[ -n "$MAX_SAMPLES" ]]; then
  echo "MAX_SAMPLES=$MAX_SAMPLES"
else
  echo "MAX_SAMPLES=<full split>"
fi

PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}" "${CMD[@]}"
