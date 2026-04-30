# Experiment Log

## 2026-05-01 - Phase 6 lightweight syntax/import checks

- run name: `phase6-lightweight-syntax-import`
- branch: `token-prm-openai-style`
- environment details: conda env `math-rl` at
  `/Work21/2024/luyuheng/miniconda3/envs/math-rl`
- commands or scripts:
  - `python -m py_compile` for active extra0 PRM, reranking, server, TRL
    reward, TRL training, and data-prep modules.
  - `python -c "import scripts.trl.rewards ..."`
  - `python scripts/trl/train_grpo_math_verifier.py --help`
  - `python scripts/verifier/serve_qwen_extra0_prm.py --help`
  - `python scripts/verifier/eval_prm_best_of_n.py --help`
  - `python scripts/verifier/debug_extra0_rollout_format.py --limit 1`
  - `bash -n` for TRL launch scripts and legacy verl launcher syntax.
- result: passed.
- verification status: active reward modules import without loading full models,
  entrypoint help commands start, the rollout-format debug script emits raw
  rollout text, three parsed steps, and inserted `<extra_0>` scoring input.

## 2026-05-01 - Phase 6 synthetic extra0 scoring smoke

- run name: `phase6-extra0-synthetic-smoke`
- branch: `token-prm-openai-style`
- environment details: conda env `math-rl`, tokenizer path
  `models/Qwen2.5-Math-1.5B-Instruct`.
- command or script:
  `python scripts/verifier/smoke_extra0_synthetic.py --tokenizer-path models/Qwen2.5-Math-1.5B-Instruct`
- result: passed after adding explicit `<extra_0>` token setup.
- verification status: tokenizer has one `<extra_0>` token after setup, label
  positions match `<extra_0>` positions, and a tiny CPU Qwen2
  token-classification forward returned three scores for three steps.

## 2026-05-01 - Phase 6 offline reranking prerequisite check

- run name: `phase6-offline-rerank-prereq-check`
- branch: `token-prm-openai-style`
- environment details: conda env `math-rl`, current server session.
- commands or scripts:
  - searched for best-of-N JSONL and PRM checkpoints in the repository.
  - checked `/root/autodl-tmp/prm_grpo`.
  - checked model directory completeness under `models/`.
  - checked GPU visibility with PyTorch and `nvidia-smi`.
- result: blocked.
- verification status: no usable fixed-candidate JSONL or PRM checkpoint is
  accessible, the policy model download is incomplete, the PRM base model is
  absent, `/root/autodl-tmp/prm_grpo` is not readable, and PyTorch reports zero
  CUDA devices.

## 2026-05-01 - Server-local artifact rebuild

- run name: `phase6-server-local-artifact-rebuild`
- branch: `token-prm-openai-style`
- environment details: conda env `math-rl`, proxy shell from
  `/Work21/2024/luyuheng/Log-TIR/mihomo-server-proxy`.
- commands or scripts:
  - re-downloaded `Qwen2.5-Math-1.5B-Instruct` and
    `Qwen2.5-Math-7B-Instruct` from ModelScope into `models/`.
  - replaced Git LFS pointer JSONLs in `data/prm800k_raw/prm800k` with the
    actual PRM800K JSONL files.
  - verified local model config/tokenizer loading with `AutoConfig` and
    `AutoTokenizer`.
  - built `data/datasets/prm800k_openai_phase1_phase2_stepwise_nonneg_allsteps`
    from raw PRM800K with `neutral_policy=nonnegative` and
    `stop_at_first_negative=False`.
  - regenerated `data/trl_math/train.parquet` and `data/trl_math/test.parquet`
    from `DigitalLearningGmbH/MATH-lighteval`.
  - ran W&B API connectivity smoke from the same proxy/env shell.
- result: setup prerequisites rebuilt except GPU visibility.
- verification status:
  - policy model path: `models/Qwen2.5-Math-1.5B-Instruct`.
  - PRM base model path: `models/Qwen2.5-Math-7B-Instruct`.
  - PRM800K raw JSONL total size: `477105425` bytes.
  - PRM cache sizes: `{'train': 487315, 'test': 13451}`.
  - TRL MATH sizes: `{'train': 7500, 'test': 5000}`.
  - W&B viewer resolved as `rwang817`.
  - GPU remains blocked: PyTorch reports `cuda_available=False` and
    `device_count=0`.

## 2026-05-01 - Repo-local active defaults smoke

- run name: `phase6-repo-local-defaults-smoke`
- branch: `token-prm-openai-style`
- environment details: conda env `math-rl`.
- commands or scripts:
  - `python -m py_compile` for active PRM, reranking, TRL data-prep, baseline,
    MC, and verifier GRPO scripts.
  - `bash -n` for active TRL launchers.
  - direct `--help` smoke for data-prep, PRM training, baseline GRPO, MC GRPO,
    and verifier GRPO entrypoints.
- result: passed after adding repo-root bootstrapping to direct TRL train
  entrypoints and switching active defaults away from `/root/autodl-tmp/prm_grpo`.
- verification status: active scripts now default to repository-local
  `models/`, `data/trl_math`, `outputs/`, and
  `token_prm_runs/extra0-prm/final` paths.
