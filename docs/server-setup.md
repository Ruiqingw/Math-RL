# Server Setup Runbook

This runbook records the server environment for the Math-RL extra0 PRM work.
Run network-dependent commands from the same shell that sources the proxy
environment.

## Repository

```bash
cd /Work21/2024/luyuheng/Math-RL
git checkout token-prm-openai-style
git pull --ff-only
```

## Proxy

```bash
cd /Work21/2024/luyuheng/Log-TIR/mihomo-server-proxy
./start_mihomo.sh
source ./proxy_env.sh
```

Use this proxy shell for package installs, ModelScope downloads, W&B checks, and
training startup.

## Conda Environment

Use the user Miniconda installation:

```bash
CONDA=/Work21/2024/luyuheng/miniconda3/bin/conda

$CONDA config --set show_channel_urls yes
$CONDA config --remove-key channels 2>/dev/null || true
$CONDA config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
$CONDA config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
$CONDA config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2

$CONDA create -y -n math-rl python=3.10 pip
conda activate math-rl
```

The created environment path is:

```text
/Work21/2024/luyuheng/miniconda3/envs/math-rl
```

Use the environment Python directly in scripts or smoke tests when possible:

```bash
PY=/Work21/2024/luyuheng/miniconda3/envs/math-rl/bin/python
```

## Pip Mirror

```bash
$PY -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
$PY -m pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
```

## PyTorch

The server has four RTX 4090 GPUs with driver `550.40.07`. Install CUDA 12.4
PyTorch wheels:

```bash
PYTHONNOUSERSITE=1 $PY -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision torchaudio
```

Verified installed versions:

```text
torch==2.6.0+cu124
torchvision==0.21.0+cu124
torchaudio==2.6.0+cu124
```

## Core Training Dependencies

Use pinned wheel versions to avoid slow or fragile source builds:

```bash
PYTHONNOUSERSITE=1 $PY -m pip install --only-binary=:all: \
  numpy==1.26.4 pyarrow==16.1.0 \
  transformers==4.51.3 datasets==3.6.0 accelerate==1.6.0 \
  peft==0.15.2 trl==0.17.0 \
  wandb==0.19.11 modelscope==1.36.3 scikit-learn==1.6.1 \
  tqdm==4.67.1 requests==2.32.3 \
  fastapi==0.115.12 uvicorn==0.34.2 pydantic==2.11.4 \
  sentencepiece==0.2.0 protobuf==5.29.5 safetensors==0.5.3 \
  einops==0.8.1
```

Optional math-answer verification helper:

```bash
PYTHONNOUSERSITE=1 $PY -m pip install -i https://pypi.org/simple \
  math-verify==0.8.0
```

Do not install `latex2sympy2==1.9.1` together with `math-verify==0.8.0`;
their `antlr4-python3-runtime` constraints conflict.

## GPU Visibility Check

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
PYTHONNOUSERSITE=1 $PY - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
print("cuda_device_names=", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
```

Verified result:

```text
cuda_available= True
cuda_device_count= 4
cuda_device_names= ['NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4090', 'NVIDIA GeForce RTX 4090']
```

## Import Smoke Test

```bash
PYTHONNOUSERSITE=1 $PY - <<'PY'
import torch, transformers, datasets, accelerate, peft, trl, wandb, modelscope, sklearn
from transformers import AutoModelForTokenClassification, AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "accelerate": accelerate.__version__,
    "peft": peft.__version__,
    "trl": trl.__version__,
    "wandb": wandb.__version__,
    "modelscope": modelscope.__version__,
    "sklearn": sklearn.__version__,
})
print(GRPOTrainer.__name__, AutoModelForTokenClassification.__name__, LoraConfig.__name__)
PY
```

Verified result:

```text
torch 2.6.0+cu124
transformers 4.51.3
datasets 3.6.0
accelerate 1.6.0
peft 0.15.2
trl 0.17.0
wandb 0.19.11
modelscope 1.36.3
sklearn 1.6.1
```

## W&B Connectivity Smoke Test

Run this from the same shell/env/proxy that will launch training:

```bash
cd /Work21/2024/luyuheng/Log-TIR/mihomo-server-proxy
source ./proxy_env.sh
cd /Work21/2024/luyuheng/Math-RL

PY=/Work21/2024/luyuheng/miniconda3/envs/math-rl/bin/python
PYTHONNOUSERSITE=1 $PY - <<'PY'
import os
import wandb

print("wandb_version=", wandb.__version__)
print("WANDB_API_KEY_present=", bool(os.environ.get("WANDB_API_KEY")))
api = wandb.Api(timeout=15)
viewer = api.viewer
print("wandb_viewer=", getattr(viewer, "username", None) or getattr(viewer, "entity", None) or viewer)
print("wandb_connectivity=ok")
PY
```

Verified result on 2026-04-30:

```text
wandb_version= 0.19.11
WANDB_API_KEY_present= False
wandb_viewer= rwang817
wandb_connectivity=ok
```

If this fails before a long run, do not start training. Fix W&B login or network
first.

## Extra0 PRM W&B Metrics

`scripts/verifier/train_qwen_extra0_prm.py` records the extra0 PRM training
contract under these key families:

- `extra0/train_loss` and `extra0/eval_*` for step-level loss, accuracy,
  balanced accuracy, negative AUROC, and negative average precision.
- `extra0/train_*`, `extra0/eval_*`, and `extra0/dropped_label_frac` for label
  counts, natural negative fractions, `<extra_0>` positions, truncation, and
  dropped supervision.
- `extra0/neg_loss_weight`, `extra0/focal_gamma`,
  `extra0/effective_neg_weight_share`, `extra0/rebalance_mode`,
  `extra0/sampler_target_neg_frac`, `extra0/base_model`,
  `extra0/tuning_mode`, `extra0/lora_r`, `extra0/lora_alpha`,
  `extra0/best_checkpoint`, `extra0/best_metric_name`, and
  `extra0/save_total_limit` for run metadata.
- `best_of_n/reference_*` and `best_of_n/*` for fixed-candidate reranking
  references, PRM best-of-16 accuracy, gaps vs greedy/majority, misranking
  fraction, and correct/wrong candidate score means.

When the fixed best-of-16 JSONL is present, checkpoint selection uses
`eval_best_of_n_prm_best_of_16_accuracy` instead of step-level balanced
accuracy. If the JSONL is unavailable, the script logs a warning and falls back
to `eval_balanced_accuracy`.

The mainline should leave `--rebalance-mode none`. `--rebalance-mode sampler`
enables a row-level `WeightedRandomSampler` only for explicit imbalance
ablations; keep `--neg-loss-weight` and `--focal-gamma` as the first-line
controls.

## Offline Reranking Eval

Use `scripts/verifier/eval_prm_best_of_n.py` for fixed-candidate PRM reranking.
The active extra0 backend is `--verifier-backend extra0_token_cls`; `auto`
detects it from token-classification config or PEFT `TOKEN_CLS` adapter
metadata. The older `classifier` and `token_prm` backends remain available for
ablations.

The summary reports PRM accuracy against greedy, majority vote, and oracle on
the same candidate pool, plus correct/wrong candidate score means, misranking
count/fraction, and whether selected wrong candidates are longer or have more
parsed steps than the best-scored correct candidate in the same group.

## TRL Verifier Backend

`scripts/trl/rewards.py` supports explicit verifier backend selection:
`classifier`, `token_prm`, and `extra0_token_cls`; `auto` detects the backend
from checkpoint files. For extra0 PRM GRPO runs, set
`VERIFIER_BACKEND=extra0_token_cls` or pass
`--verifier-backend extra0_token_cls`. Reward logs include a numeric
`verifier_shaping_reward/backend_id` and
`verifier_shaping_reward/backend_is_extra0_token_cls`.

## Model Directory

Base models should be downloaded into the repository-local `models/` directory.
Keep all model paths configurable in scripts; do not hard-code user-specific
absolute paths except in local launch commands.

Use these defaults for the current extra0 PRM work:

```bash
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-models/Qwen2.5-Math-1.5B-Instruct}"
PRM_BASE_MODEL_PATH="${PRM_BASE_MODEL_PATH:-models/Qwen2.5-Math-7B-Instruct}"
```

Scripts should accept these paths through CLI flags or environment variables so
the same code can run with different server-local cache locations.

## ModelScope Downloads

Run model downloads from the repository root in the same proxy shell used for
package installs:

```bash
cd /Work21/2024/luyuheng/Log-TIR/mihomo-server-proxy
./start_mihomo.sh
source ./proxy_env.sh

cd /Work21/2024/luyuheng/Math-RL
mkdir -p models
PY=/Work21/2024/luyuheng/miniconda3/envs/math-rl/bin/python
```

Download the GRPO policy model:

```bash
PYTHONNOUSERSITE=1 $PY - <<'PY'
from modelscope import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    local_dir="models/Qwen2.5-Math-1.5B-Instruct",
)
PY
```

Download the PRM base model:

```bash
PYTHONNOUSERSITE=1 $PY - <<'PY'
from modelscope import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-Math-7B-Instruct",
    local_dir="models/Qwen2.5-Math-7B-Instruct",
)
PY
```

The expected local paths are:

```text
models/Qwen2.5-Math-1.5B-Instruct
models/Qwen2.5-Math-7B-Instruct
```

If a directory contains a `._____temp/` subdirectory or a partially downloaded
`model.safetensors`, treat the model as incomplete and rerun the same
`snapshot_download(...)` command from the proxy shell. Do not start training
until both model directories contain the tokenizer files, config files, and the
complete safetensors shards.

Minimal local load checks:

```bash
PYTHONNOUSERSITE=1 $PY - <<'PY'
from transformers import AutoConfig, AutoTokenizer

for path in [
    "models/Qwen2.5-Math-1.5B-Instruct",
    "models/Qwen2.5-Math-7B-Instruct",
]:
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    print(path, "vocab_size=", len(tok), "model_type=", cfg.model_type)
PY
```

For the extra0 PRM path, verify the tokenizer exposes the Qwen step marker:

```bash
PYTHONNOUSERSITE=1 $PY - <<'PY'
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "models/Qwen2.5-Math-7B-Instruct",
    trust_remote_code=True,
)
extra0_id = tok.convert_tokens_to_ids("<extra_0>")
print("extra0_id=", extra0_id)
assert extra0_id != tok.unk_token_id
PY
```

Current server note from 2026-05-01: `models/Qwen2.5-Math-1.5B-Instruct`
exists but still has a `._____temp/` download directory, so rerun the policy
model download before using it. `models/Qwen2.5-Math-7B-Instruct` still needs to
be downloaded.
