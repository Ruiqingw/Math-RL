# Math-RL

Math-RL is an experimental codebase for reinforcement learning on mathematical
reasoning. The project focuses on MATH-style final-answer optimization with
GRPO, plus verifier and process reward model (PRM) experiments for scoring
intermediate reasoning trajectories.

The current active direction is a refactor from older verifier designs toward a
Qwen-style `<extra_0>` token-classification PRM. The task queue for that work is
tracked in [tasks.md](tasks.md).

## Project Status

This repository is in active research mode. The codebase contains both working
baselines and older verifier experiments that are kept for ablations.

Current high-level conclusions from prior experiments:

- Rule-based final-answer GRPO is the clean baseline.
- Earlier classifier-head and causal-token PRM variants learned some step-level
  signal, but did not reliably improve online GRPO under naive reward fusion.
- Offline PRM reranking was highly aggregation-sensitive. On a fixed
  100-problem, best-of-16 MATH candidate pool, the older token PRM improved from
  poor `mean_log` reranking to better `min` aggregation, but still trailed
  majority voting.
- The next mainline is to train a self-owned PRM using the official-style
  `<extra_0>` step marker and `AutoModelForTokenClassification`.

## Active Plan

The active refactor plan is:

1. Train a Qwen-style PRM with `<extra_0>` markers after each reasoning step.
2. Compute PRM loss only at `<extra_0>` positions.
3. Use LoRA first on `Qwen2.5-Math-7B-Instruct`.
4. Use `raw_phase1_phase2 + all-steps` PRM800K data with the existing label
   policy: `rating >= 0 -> positive`.
5. Select PRM checkpoints by fixed-candidate best-of-N reranking quality, not
   only step-level balanced accuracy.
6. Use PRM scores in GRPO through wrong-only continuous shaping:

```text
reward = base_correct + beta * (1 - base_correct) * prm_score
```

This keeps correct final answers above incorrect final answers while still
allowing the PRM to rank wrong completions.

The final target runtime shape is:

```text
GPU 0,1,2: TRL GRPO policy training
GPU 3:     PRM reward server
```

Single-GPU smoke tests are acceptable during development, but final validation
should use the 3+1 GPU setup.

## Repository Layout

```text
scripts/trl/        TRL-based GRPO training and reward functions
scripts/verifier/   PRM/verifier training, scoring, reranking, and diagnostics
scripts/verl/       Legacy verl path, not the current mainline
tasks.md            Current execution queue for the extra0 PRM refactor
WORKFLOW.md         Older local/server workflow notes
RECENT_PRM_EXPERIMENTS.md
                    Prior PRM and reranking experiment notes
VERIFIER_FAILURE_ANALYSIS.md
                    Analysis of why earlier verifier shaping underperformed
SERVER_ENV_SETUP.md Older server setup notes
```

## Main Components

### GRPO Baseline

The baseline trains with final boxed-answer correctness only:

```text
scripts/trl/train_grpo_math_baseline.py
scripts/trl/run_trl_grpo_math_baseline.sh
```

The reward function is `math_boxed_reward` in
`scripts/trl/rewards.py`.

### Verifier-Guided GRPO

The current TRL verifier path is:

```text
scripts/trl/train_grpo_math_verifier.py
scripts/trl/run_trl_grpo_math_verifier.sh
scripts/trl/rewards.py
```

This path is the intended integration point for the upcoming extra0 PRM and PRM
server backend.

### PRM Training And Evaluation

Older verifier implementations live under:

```text
scripts/verifier/train_verifier.py
scripts/verifier/train_token_prm.py
scripts/verifier/token_prm.py
scripts/verifier/token_reward_fn.py
```

Offline reranking utilities:

```text
scripts/verifier/eval_prm_best_of_n.py
scripts/verifier/analyze_prm_rerank_jsonl.py
scripts/verifier/hybrid_rerank.py
```

The planned extra0 PRM implementation will add new files under
`scripts/verifier/` rather than deleting these older ablation paths.

## Server And Environment

Training is intended to run on a remote GPU server. The planned environment is:

```text
conda environment: math-rl
model source:      ModelScope
local model dir:   models/
default base PRM:  models/Qwen2.5-Math-7B-Instruct
```

Server proxy startup:

```bash
cd /Work21/2024/luyuheng/Log-TIR/mihomo-server-proxy
./start_mihomo.sh
source ./proxy_env.sh
```

Run the proxy setup in the shell used for model downloads, W&B checks, and
training. Before any long training run, test W&B connectivity from that same
environment so a W&B timeout does not stop the run after GPU time has already
started.

The detailed server setup runbook is part of the task plan and should be added
under `docs/` before full training starts.

## Data And Artifacts

Large models, generated datasets, checkpoints, W&B logs, and evaluation JSONL
artifacts are not expected to be committed to Git.

Common server-side artifact categories:

```text
models/                 downloaded base models
data/                   prepared MATH / PRM data
outputs/                GRPO and evaluation outputs
token_prm_runs/         verifier / PRM checkpoints
```

Paths should remain configurable because local development and server training
use different directories.

## Experiment Logging

Future runs should record:

- exact model path
- PRM checkpoint
- dataset view
- reward mode
- beta
- PRM aggregation
- matched baseline details
- W&B run name
- bad cases and debugging notes

Planned Markdown logs:

```text
docs/experiment-log.md
docs/bad-cases.md
docs/debugging-notes.md
docs/server-setup.md
```

## Current Success Criterion

The current goal is to find one wrong-only PRM GRPO run that beats the matched
pure-rule GRPO baseline under the same data slice, seed, training steps, and
evaluation set.

The beta search should stop early:

```text
baseline -> beta=0.05 -> beta=0.1 -> beta=0.2
```

Stop at the first PRM-shaped run that beats the matched baseline.

## Development Notes

- Use [tasks.md](tasks.md) as the authoritative execution queue.
- Mark each task complete only after implementation and verification.
- Keep `verl` as legacy unless explicitly reactivated.
- Preserve old verifier implementations for ablations.
- Do not interpret quick validation runs with small eval sets as final results.
