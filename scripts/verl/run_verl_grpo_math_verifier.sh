#!/usr/bin/env bash
set -euo pipefail

# Fix Ray + protobuf C++ extension incompatibility (is_repeated attribute error)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"
export RAY_USE_MULTIPROCESSING_CPU_COUNT="${RAY_USE_MULTIPROCESSING_CPU_COUNT:-1}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VERL_ROOT="${VERL_ROOT:-$PROJECT_ROOT/verl}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/prm_grpo/models/Qwen2.5-Math-1.5B}"
VERIFIER_MODEL_PATH="${VERIFIER_MODEL_PATH:-/root/autodl-tmp/prm_grpo/verifier_cls/final}"
VERIFIER_DEVICE="${VERIFIER_DEVICE:-cuda}"

DATA_DIR="${DATA_DIR:-/root/autodl-tmp/prm_grpo/data/verl_math}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
VAL_FILE="${VAL_FILE:-$DATA_DIR/test.parquet}"

TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-2000}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-200}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1536}"

PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-4}"

ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.3}"

LR="${LR:-3e-5}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:-10}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
LOG_VAL_GENERATIONS="${LOG_VAL_GENERATIONS:-8}"

LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"

VERIFIER_BETA="${VERIFIER_BETA:-0.1}"
VERIFIER_DELTA="${VERIFIER_DELTA:-0.05}"
VERIFIER_THRESHOLD="${VERIFIER_THRESHOLD:-0.5}"
VERIFIER_BATCH_SIZE="${VERIFIER_BATCH_SIZE:-1}"
VERIFIER_MAX_LENGTH="${VERIFIER_MAX_LENGTH:-1536}"
REWARD_NUM_WORKERS="${REWARD_NUM_WORKERS:-1}"

NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-10}"

WANDB_PROJECT="${WANDB_PROJECT:-math_rl_verl}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-grpo-math-verifier-quick}"
RUN_LOG_DIR="${RUN_LOG_DIR:-$PROJECT_ROOT/logs}"
RUN_LOG_PATH="${RUN_LOG_PATH:-$RUN_LOG_DIR/${WANDB_RUN_NAME}.log}"
VERIFIER_DEBUG_LOG="${VERIFIER_DEBUG_LOG:-$RUN_LOG_DIR/${WANDB_RUN_NAME}.reward.log}"
VERIFIER_DEBUG="${VERIFIER_DEBUG:-1}"

ROLLOUT_MAX_MODEL_LEN="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
CUSTOM_REWARD_PATH="$PROJECT_ROOT/scripts/verl/verl_verifier_reward.py"

mkdir -p "$RUN_LOG_DIR"

export PYTHONPATH="$VERL_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export VERIFIER_MODEL_PATH
export VERIFIER_DEVICE
export VERIFIER_BETA
export VERIFIER_DELTA
export VERIFIER_THRESHOLD
export VERIFIER_BATCH_SIZE
export VERIFIER_MAX_LENGTH
export VERIFIER_DEBUG
export VERIFIER_DEBUG_LOG

# Ray reward loop workers do not receive GPU visibility by default.
# When verifier reward should run on GPU, explicitly expose GPU 0 to the
# reward worker unless the caller overrides it.
if [[ "$VERIFIER_DEVICE" == cuda* ]]; then
  export VERL_REWARD_LOOP_CUDA_VISIBLE_DEVICES="${VERL_REWARD_LOOP_CUDA_VISIBLE_DEVICES:-0}"
fi

cd "$VERL_ROOT"

echo "Launching verifier-shaped GRPO with small validation set..."
echo "VERL_ROOT=$VERL_ROOT"
echo "MODEL_PATH=$MODEL_PATH"
echo "VERIFIER_MODEL_PATH=$VERIFIER_MODEL_PATH"
echo "TRAIN_FILE=$TRAIN_FILE"
echo "VAL_FILE=$VAL_FILE"
echo "RUN_LOG_PATH=$RUN_LOG_PATH"
echo "VERIFIER_DEBUG_LOG=$VERIFIER_DEBUG_LOG"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_max_samples="$TRAIN_MAX_SAMPLES" \
    data.val_max_samples="$VAL_MAX_SAMPLES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.validation_shuffle=False \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank="$LORA_RANK" \
    actor_rollout_ref.model.lora_alpha="$LORA_ALPHA" \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.max_model_len="$ROLLOUT_MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_MODEL_LEN" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward.num_workers="$REWARD_NUM_WORKERS" \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="$CUSTOM_REWARD_PATH" \
    reward.custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef="$KL_LOSS_COEF" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$WANDB_RUN_NAME" \
    trainer.log_val_generations="$LOG_VAL_GENERATIONS" \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes="$NNODES" \
    ray_kwargs.ray_init.num_cpus="$RAY_NUM_CPUS" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    +actor_rollout_ref.rollout.enable_sleep_mode=False \
    "$@" 2>&1 | tee -a "$RUN_LOG_PATH"
