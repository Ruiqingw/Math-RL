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

## Model Directory

Base models should be downloaded into the repository-local `models/` directory.
Keep all model paths configurable in scripts; do not hard-code user-specific
absolute paths except in local launch commands.

