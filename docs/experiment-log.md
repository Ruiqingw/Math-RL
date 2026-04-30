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
