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

## 2026-05-01 - Offline reranking evaluator required missing `verl`

- command or script:
  `PYTHONNOUSERSITE=1 /Work21/2024/luyuheng/miniconda3/envs/math-rl/bin/python scripts/verifier/eval_prm_best_of_n.py --help`
- environment details: repository `/Work21/2024/luyuheng/Math-RL`, branch
  `token-prm-openai-style`, conda env `math-rl`.
- error message or symptom: `ModuleNotFoundError: No module named 'verl'` from
  the top-level import of `verl.utils.reward_score.math_reward`.
- root cause: the evaluator was on the active TRL/verifier path but imported a
  legacy verl reward helper at module import time. The server environment does
  not install `verl` for the active path.
- attempted fixes: searched for local math reward helpers and inspected the
  installed `math_verify` API.
- final fix: added fallback implementations in
  `scripts/verifier/eval_prm_best_of_n.py` and `scripts/trl/rewards.py` using
  `math_verify` for answer checking and a local boxed-answer extractor when
  `verl` is unavailable.
- verification status: verified. The evaluator `--help` command starts in the
  `math-rl` environment, TRL reward imports start without `verl`, and backend
  auto-detection smoke tests pass for classifier, extra0 token-classification,
  and token PRM checkpoints.

## 2026-05-01 - FastAPI `TestClient` smoke test needed optional `httpx`

- command or script: mocked PRM server smoke test using
  `from fastapi.testclient import TestClient`.
- environment details: `math-rl` Python 3.10 with FastAPI/Starlette installed.
- error message or symptom: `RuntimeError: The starlette.testclient module
  requires the httpx package to be installed`.
- root cause: `httpx` is an optional test-client dependency and is not required
  for running the uvicorn PRM server.
- attempted fixes: avoided adding a new package for this task and switched the
  smoke test to direct calls of the FastAPI endpoint functions with mocked
  model state.
- final fix: direct endpoint smoke test passed for `/health` and `/score`
  request/response logic.
- verification status: verified by `server_endpoint_smoke=ok`.

## 2026-05-01 - Extra0 rollout format debug examples

- command or script:
  `PYTHONNOUSERSITE=1 $PY scripts/verifier/debug_extra0_rollout_format.py`
- environment details: repository `/Work21/2024/luyuheng/Math-RL`, active TRL
  reward path.
- error message or symptom: rollout completions are free-form text, while PRM
  training uses human step lists with inserted `<extra_0>` markers.
- root cause if known: distribution shift can appear if rollout text is parsed
  into different step boundaries from the PRM training examples.
- attempted fixes: standardized policy prompts to request blank-line-separated
  reasoning steps, tightened `step_splitter.py` to prefer blank lines and
  numbered steps, and added `scripts/verifier/debug_extra0_rollout_format.py`
  to compare raw rollout text, parsed steps, and inserted `<extra_0>` scoring
  input.
- final fix: use the debug script on synthetic examples or fixed-candidate
  JSONL before long GRPO runs.
- verification status: verified. The synthetic smoke test prints raw rollout
  text, three parsed steps, and the matching `<extra_0>` scoring input.
