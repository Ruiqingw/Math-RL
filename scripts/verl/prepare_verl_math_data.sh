#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERL_ROOT="${VERL_ROOT:-$PROJECT_ROOT/verl}"

# If you already mirrored the dataset locally, set RAW_DATASET_PATH to that path.
RAW_DATASET_PATH="${RAW_DATASET_PATH:-}"
LOCAL_SAVE_DIR="${LOCAL_SAVE_DIR:-/root/autodl-tmp/prm_grpo/data/verl_math}"

export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"

cd "$VERL_ROOT"

CMD=(
    python3 "$PROJECT_ROOT/scripts/verl/prepare_verl_math_data.py"
    --local_save_dir "$LOCAL_SAVE_DIR"
)

if [[ -n "$RAW_DATASET_PATH" ]]; then
    CMD+=(--local_dataset_path "$RAW_DATASET_PATH")
fi

echo "Preparing verl MATH parquet dataset..."
echo "VERL_ROOT=$VERL_ROOT"
echo "LOCAL_SAVE_DIR=$LOCAL_SAVE_DIR"

"${CMD[@]}"

echo "Done. Files written under: $LOCAL_SAVE_DIR"
