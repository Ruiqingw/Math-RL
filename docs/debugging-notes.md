# Debugging Notes

Runtime and setup issues are recorded here in chronological order.

## 2026-04-30 - `conda run -n math-rl` used the wrong Python

- command or script: dependency installation through
  `/Work21/2024/luyuheng/miniconda3/bin/conda run -n math-rl python -m pip ...`
- environment details: repository `/Work21/2024/luyuheng/Math-RL`, branch
  `token-prm-openai-style`; target env
  `/Work21/2024/luyuheng/miniconda3/envs/math-rl`
- error message or symptom: pip output showed `/opt/software/anaconda3` Python
  3.8 paths and "Defaulting to user installation because normal site-packages
  is not writeable".
- root cause: `conda run` resolved through a mixed server conda setup and did
  not reliably execute the target environment Python.
- attempted fixes: stopped the in-flight pip process and inspected the target
  env's absolute Python path.
- final fix: used
  `/Work21/2024/luyuheng/miniconda3/envs/math-rl/bin/python` directly with
  `PYTHONNOUSERSITE=1` for package installation and smoke tests.
- verification status: verified by `sys.executable`, package import smoke test,
  and GPU visibility from the target environment.

## 2026-04-30 - Latest W&B package attempted source build

- command or script: `pip install transformers datasets accelerate peft trl
  wandb modelscope ...`
- environment details: `math-rl` Python 3.10 with Tsinghua PyPI mirror.
- error message or symptom: `wandb 0.26.1` was downloaded as a source archive
  and failed with `Did not find the 'go' binary`.
- root cause: the selected mirror/version combination did not provide a usable
  prebuilt wheel for that W&B release.
- attempted fixes: pinned W&B to a stable manylinux wheel.
- final fix: installed `wandb==0.19.11`.
- verification status: `import wandb` and W&B API connectivity both passed.

## 2026-04-30 - Latest pyarrow package attempted source build

- command or script: core dependency install from Tsinghua mirror.
- environment details: `math-rl` Python 3.10.
- error message or symptom: pip selected `pyarrow 24.0.0` source package and
  started installing build dependencies.
- root cause: the mirror did not provide a suitable prebuilt wheel for that
  latest pyarrow release.
- attempted fixes: stopped the source build and pinned the training stack to
  wheel-backed versions.
- final fix: installed `datasets==3.6.0` with `pyarrow==16.1.0` and
  `--only-binary=:all:`.
- verification status: core import smoke test passed.

## 2026-04-30 - `latex2sympy2` conflicted with `math-verify`

- command or script: `pip install latex2sympy2==1.9.1 math-verify==0.8.0`.
- environment details: `math-rl` Python 3.10, official PyPI through proxy.
- error message or symptom: dependency resolution failed because
  `latex2sympy2==1.9.1` requires `antlr4-python3-runtime==4.7.2`, while
  `latex2sympy2_extended==1.10.2` from `math-verify==0.8.0` requires
  `antlr4-python3-runtime>=4.9.3,<=4.13.2`.
- root cause: incompatible antlr runtime constraints.
- attempted fixes: removed the direct `latex2sympy2` install and kept the newer
  `math-verify` path.
- final fix: installed `math-verify==0.8.0`, which pulled
  `latex2sympy2_extended==1.10.2`.
- verification status: package installation succeeded.

## 2026-04-30 - Remote push blocked by missing Git credentials

- command or script: `git push origin token-prm-openai-style`.
- environment details: local branch `token-prm-openai-style`, local commit for
  setup milestone.
- error message or symptom: `fatal: could not read Username for
  'https://github.com': No such device or address`.
- root cause: this checkout used an HTTPS remote without non-interactive GitHub
  credentials. The server SSH key was already authorized for GitHub, but the
  HTTPS remote did not use it.
- attempted fixes: checked the connected GitHub App repository permissions and
  attempted to create a matching remote commit through the GitHub App. The app
  returned `Resource not accessible by integration` for Git database commit
  creation. Then verified SSH auth with `ssh -T git@github.com`.
- final fix: changed `origin` to `git@github.com:Ruiqingw/Math-RL.git`.
- verification status: verified. SSH authentication succeeded and the setup
  milestone was pushed to `origin/token-prm-openai-style`.
