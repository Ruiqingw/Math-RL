# Default Workflow

This project uses a split workflow:

- Code is edited locally.
- Training and evaluation run on the server.
- Code is synced through GitHub.

## 1. Default Paths

Local machine:

- Repo: `/Users/ruiqing/Documents/MATH_RL`

Server:

- Repo: `/root/autodl-tmp/prm_grpo/Math-RL`
- Base model: `/root/autodl-tmp/prm_grpo/models/Qwen2.5-Math-1.5B`
- Verifier checkpoints: `/root/autodl-tmp/prm_grpo/verifier_cls/...`
- GRPO data: `/root/autodl-tmp/prm_grpo/data/verl_math`
- External verl repo: `/root/autodl-tmp/verl`

## 2. Default Development Flow

1. Modify code locally.
2. Test syntax locally when needed.
3. Commit and push from the local machine.
4. On the server, enter the repo and run:

```bash
cd /root/autodl-tmp/prm_grpo/Math-RL
git pull origin main
```

5. Run training or evaluation on the server.
6. If the server reports errors, copy the logs back and debug locally.

## 3. Environment Rules

- Do not rely on Codex inside Remote SSH. Use local Codex plus server execution.
- Preferred remote operation path is browser-first: use the browser to log into AutoDL and W&B, inspect jobs, start runs, and monitor results.
- If direct control of the user's logged-in browser is unavailable, use the OpenClaw isolated browser to log into AutoDL and W&B.
- Browser-based execution is preferred over SSH for normal training operations in this project unless a task explicitly requires terminal-only access.
- Environment setup, package installation, git sync, and code inspection should use no-GPU mode when possible.
- Only switch to GPU mode for:
  - verifier training
  - base model evaluation
  - GRPO / verl training
  - heavy inference

This keeps GPU time focused on actual experiments.

### Browser Execution Notes

- AutoDL training may be launched either from a browser-based terminal or from browser UI controls, depending on what is available in the current AutoDL workspace.
- W&B should be checked in the browser before and after important runs to confirm logging is healthy.
- If the isolated browser is used, assume its session state is less durable than the user's normal browser profile. Important commands, run names, and result links should be written down in project notes or chat summaries.
- If the machine sleeps, OpenClaw restarts, or the isolated browser is reset, browser login state may be lost and must be re-established.

### China Network / External Access Notes

- The AutoDL server is in China. Before accessing GitHub, W&B, Hugging Face, or other external services from the server terminal, run:
  - `source /etc/network_turbo`
- Treat this as a standard preparation step before `git pull`, package installs, model downloads, or any network-dependent command.
- If network commands fail, first check whether the shell is still using stale proxy environment variables (for example `HTTP_PROXY` / `HTTPS_PROXY` pointing at a dead localhost port). Clearing stale proxy env vars only affects the current shell session; it does not permanently change the machine.

## 4. Code Layout

Verifier-related code:

- `scripts/verifier/`

verl / GRPO-related code:

- `scripts/verl/`

Compatibility wrappers still exist under `scripts/`, but new commands should prefer the subdirectories above.

## 5. Server Sync

First time:

```bash
cd /root/autodl-tmp/prm_grpo
git clone https://github.com/Ruiqingw/Math-RL.git
cd Math-RL
```

Later sync:

```bash
cd /root/autodl-tmp/prm_grpo/Math-RL
git pull origin main
```

## 6. Data Preparation

Prepare verl parquet data on the server:

```bash
cd /root/autodl-tmp/prm_grpo/Math-RL
VERL_ROOT=/root/autodl-tmp/verl \
bash scripts/verl/prepare_verl_math_data.sh
```

This writes:

- `/root/autodl-tmp/prm_grpo/data/verl_math/train.parquet`
- `/root/autodl-tmp/prm_grpo/data/verl_math/test.parquet`

## 7. Baseline Before Verifier

Always run ordinary GRPO first before verifier-guided GRPO.

Recommended quick baseline:

```bash
cd /root/autodl-tmp/prm_grpo/Math-RL

VERL_ROOT=/root/autodl-tmp/verl \
MODEL_PATH=/root/autodl-tmp/prm_grpo/models/Qwen2.5-Math-1.5B \
DATA_DIR=/root/autodl-tmp/prm_grpo/data/verl_math \
TRAIN_MAX_SAMPLES=1000 \
VAL_MAX_SAMPLES=50 \
ROLLOUT_N=2 \
TEST_FREQ=2 \
SAVE_FREQ=5 \
TOTAL_EPOCHS=1 \
LOG_VAL_GENERATIONS=8 \
bash scripts/verl/run_verl_grpo_math_baseline.sh
```

Only after baseline runs normally should verifier GRPO be started.

## 8. Verifier GRPO

Recommended quick verifier run:

```bash
cd /root/autodl-tmp/prm_grpo/Math-RL

VERL_ROOT=/root/autodl-tmp/verl \
MODEL_PATH=/root/autodl-tmp/prm_grpo/models/Qwen2.5-Math-1.5B \
DATA_DIR=/root/autodl-tmp/prm_grpo/data/verl_math \
VERIFIER_MODEL_PATH=/root/autodl-tmp/prm_grpo/verifier_cls/checkpoint-6000 \
VERIFIER_DEVICE=cpu \
TRAIN_MAX_SAMPLES=1000 \
VAL_MAX_SAMPLES=50 \
ROLLOUT_N=2 \
TEST_FREQ=2 \
SAVE_FREQ=5 \
TOTAL_EPOCHS=1 \
LOG_VAL_GENERATIONS=8 \
bash scripts/verl/run_verl_grpo_math_verifier.sh
```

Notes:

- Start with `VERIFIER_DEVICE=cpu` if the reward worker cannot see CUDA.
- If reward worker GPU access is configured correctly, `VERIFIER_DEVICE=cuda` can be tried later.

## 9. W&B Defaults

Before a real run, it is fine to test W&B with:

```bash
python - <<'PY'
import wandb
run = wandb.init(project="math_rl_verl", name="wandb-connection-test", mode="online")
wandb.log({"test_metric": 1.0})
run.finish()
print("wandb ok")
PY
```

For simple connectivity checks:

```bash
curl -I https://api.wandb.ai
```

An HTTP 404 still means the network path is working.

## 10. Evaluation Defaults

Current quick validation is intentionally small:

- `VAL_MAX_SAMPLES=50`

This is good for fast checks, but metrics will be noisy.

For more trustworthy comparisons, increase validation size later, such as:

- `VAL_MAX_SAMPLES=200`
- or `VAL_MAX_SAMPLES=500`

## 11. What To Compare

When comparing baseline vs verifier GRPO, look at:

- `val-core/.../acc/mean@1`
- `val/generations`
- response length metrics
- whether outputs still end with boxed answers

Do not over-interpret quick-run validation with only 50 examples.

## 12. Base Model Evaluation

To measure raw base-model performance on MATH outside GRPO:

```bash
cd /root/autodl-tmp/prm_grpo/Math-RL

PYTHONPATH=/root/autodl-tmp/verl:$PYTHONPATH \
python scripts/verl/eval_base_math.py \
  --model-path /root/autodl-tmp/prm_grpo/models/Qwen2.5-Math-1.5B \
  --split test \
  --max-samples 50 \
  --max-new-tokens 512 \
  --device cuda \
  --output-jsonl /root/autodl-tmp/prm_grpo/outputs/base_math_eval_50.jsonl
```

## 13. Debugging Rule

If a run fails:

1. Keep the full traceback.
2. Do not patch directly on the server first.
3. Fix locally.
4. Commit and push.
5. Pull on the server and rerun.

This avoids local/server drift.
