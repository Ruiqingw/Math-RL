# AGENTS.md

Instructions for Codex or any AI coding agent working in this repository.

## Authority Order

1. Follow this `AGENTS.md` for workflow and guardrails.
2. Use `tasks.md` as the authoritative execution queue.
3. Use `README.md` only as human-facing project context. Do not treat README as
   the task list.
4. Use older notes such as `WORKFLOW.md`, `RECENT_PRM_EXPERIMENTS.md`, and
   `VERIFIER_FAILURE_ANALYSIS.md` as historical context only.

## Current Objective

The active project direction is the `<extra_0>` PRM refactor described in
`tasks.md`.

Mainline decisions:

- Train a self-owned Qwen-style `<extra_0>` PRM.
- Use `AutoModelForTokenClassification`, not the old `Answer:` -> `+/-` causal
  token protocol as the mainline.
- Use `Qwen2.5-Math-7B-Instruct` with LoRA first.
- Use `raw_phase1_phase2 + all-steps`.
- Preserve the current label policy: `rating >= 0 -> positive`.
- Keep old classifier-head and causal-token PRM paths as ablations.
- Treat `scripts/verl/` as legacy unless the user explicitly reactivates it.
- Main GRPO reward mode is wrong-only continuous shaping:

```text
reward = base_correct + beta * (1 - base_correct) * prm_score
```

- Thresholds are not part of the main reward path.

## Execution Rules

- Before doing work, read `tasks.md` and identify the next unchecked task.
- Execute tasks sequentially unless the user explicitly asks otherwise.
- After completing and verifying a task, update only that task checkbox in
  `tasks.md`.
- Do not mark a task complete without verification.
- Do not skip ahead to later phases without a reason recorded in `docs/`.
- Keep edits scoped to the current task.
- Do not delete old verifier code unless the task explicitly asks for deletion.
- Do not modify generated artifacts, checkpoints, large model files, or vendored
  dependency trees unless explicitly requested.

## Documentation Requirements

Record operational knowledge in Markdown under `docs/`.

Required docs:

- `docs/server-setup.md`
- `docs/experiment-log.md`
- `docs/bad-cases.md`
- `docs/debugging-notes.md`

Record bad cases with:

- run name
- data split or example id
- problem excerpt
- model completion
- parsed steps
- PRM step scores
- final-answer correctness
- why the case is bad
- suspected cause
- follow-up action

Record runtime/debugging issues with:

- command or script
- environment details
- error message or symptom
- root cause if known
- attempted fixes
- final fix
- verification status

## Server Environment Rules

Training runs on the server.

- Create and use conda environment `math-rl`.
- Use Alibaba or Tsinghua mirrors by default.
- Download models from ModelScope into a local `models/` directory.
- Default PRM base model path should be configurable, with
  `models/Qwen2.5-Math-7B-Instruct` as the intended local path.

Server proxy:

```bash
cd /Work21/2024/luyuheng/Log-TIR/mihomo-server-proxy
./start_mihomo.sh
source ./proxy_env.sh
```

Use the proxy shell for model downloads, W&B checks, and training startup.
Before any long training run, run a W&B connectivity smoke test from the same
environment to avoid failures caused by W&B timeout.

## GPU Target

Single-GPU smoke tests are allowed during development. Final validation must
support:

```text
GPU 0,1,2: TRL GRPO policy training
GPU 3:     PRM reward server
```

The PRM reward server should load the PRM once on the dedicated GPU. TRL reward
functions should call the server instead of loading a full PRM inside every
policy rank.

## Experiment Policy

- Always run or identify a matched pure-rule GRPO baseline before claiming PRM
  improvement.
- Compare under the same data slice, seed, training steps, and eval set.
- Beta search is sequential with early stop:

```text
baseline -> beta=0.05 -> beta=0.1 -> beta=0.2
```

- Stop at the first wrong-only PRM GRPO run that beats the matched baseline.
- Do not over-interpret small eval sets.

## Git Rules

- Check `git status` before editing.
- The repository may contain unrelated dirty files; do not touch them.
- Stage only files changed for the current task.
- Commit after meaningful task milestones when requested.
- Keep commit messages specific.
- Push the active branch when the user asks for server execution readiness.
