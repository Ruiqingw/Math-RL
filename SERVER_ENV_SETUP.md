# Server Env Setup

每次在新终端里开始训练前，先执行下面这几条：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY

export PYTHONPATH=/root/autodl-tmp/prm_grpo/Math-RL:/root/autodl-tmp/verl:/root/autodl-tmp/trl:$PYTHONPATH
```

如果这轮训练想先避免 W&B 网络问题，也可以额外执行：

```bash
export WANDB_MODE=offline
```
