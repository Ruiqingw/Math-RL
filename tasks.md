# PRM Extra0 Refactor Tasks

This file is the execution queue for the PRM refactor. Do not implement code from
these tasks until explicitly requested. When a task is executed, mark only that
task as completed after verification.

## Goal

Replace the current self-trained PRM mainline with a Qwen-style `<extra_0>`
token-classification PRM protocol while keeping the older verifier paths as
ablations.

The intended mainline is:

1. Train a self-owned PRM with `AutoModelForTokenClassification`.
2. Insert `<extra_0>` after each reasoning step.
3. Compute loss only at `<extra_0>` positions.
4. Select checkpoints by fixed-candidate best-of-N reranking quality, not only
   step-level balanced accuracy.
5. Use the PRM in GRPO as wrong-only shaping so a wrong answer cannot outrank a
   correct answer.

## Decisions To Preserve

- The user still wants to train their own PRM, not only call the official Qwen
  PRM.
- Training will run on the server. Download required base models from
  ModelScope into a local repo/server `models/` directory instead of assuming
  the model already exists.
- The server does not yet have the project environment. Create a dedicated
  conda environment named `math-rl` and configure dependencies inside it.
- Use Alibaba or Tsinghua package mirrors by default when configuring the
  server environment.
- Server network proxy is available at
  `/Work21/2024/luyuheng/Log-TIR/mihomo-server-proxy`. Start it with
  `./start_mihomo.sh`, then source `./proxy_env.sh` in training shells before
  network-dependent operations.
- Before starting any long training run, test W&B connectivity from the same
  environment and proxy shell to avoid training failures caused by W&B timeout.
- Main GRPO policy model should be `Qwen2.5-Math-1.5B-Instruct`.
- Earlier non-instruct `Qwen2.5-Math-1.5B` results are historical references
  only. Re-run majority voting and pure-rule GRPO baselines with the instruct
  policy before claiming PRM-shaped GRPO improvement.
- The old classifier-head verifier and `Answer:` -> `+/-` token PRM should be
  preserved as ablation baselines, not deleted.
- Negative imbalance handling should remain available, but strong negative
  resampling should not be the default for the new PRM. Prefer class weights
  and focal loss first; make sampler-based rebalance an explicit option.
- All-wrong group tie-break is too sparse to be the main GRPO intervention if
  all-wrong groups are around 3%. Keep it as an ablation only.
- `verl` is legacy for now. Do not maintain or extend the verl reward path
  unless the user explicitly reactivates it.
- Main PRM data view should be `raw_phase1_phase2 + all-steps`, with the
  current label policy preserved: `rating >= 0 -> positive`.
- Main PRM tuning mode is LoRA first. Do not start with QLoRA or full
  fine-tuning unless LoRA cannot run.
- Main online GRPO target is 4 GPUs total: three GPUs for TRL GRPO policy and
  one GPU for a PRM reward server. Single-GPU smoke tests are allowed during
  development, but final validation must use the 3+1 GPU shape.
- In the final 3+1 setup, use `Qwen2.5-Math-1.5B-Instruct` for the policy ranks
  and the trained `Qwen2.5-Math-7B-Instruct`-based PRM on the reward-server GPU.
- Success criterion: stop at the first wrong-only PRM GRPO run that beats the
  matched pure-rule baseline under the same data slice, seed, training steps,
  and eval set.
- Training bad cases, runtime issues, debugging notes, and fixes must be
  recorded in Markdown files under `docs/`.

## Phase 1: Add Extra0 PRM Core

- [x] Add server environment setup runbook.
  - Create or update a Markdown runbook under `docs/`, such as
    `docs/server-setup.md`.
  - Create a conda environment named `math-rl`.
  - Choose a concrete Python version compatible with the training stack.
  - Configure conda/pip to use Alibaba or Tsinghua mirrors by default.
  - Install PyTorch with CUDA support appropriate for 4090 GPUs.
  - Install `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `wandb`,
    `modelscope`, and other project dependencies.
  - Document server proxy startup:
    - `cd /Work21/2024/luyuheng/Log-TIR/mihomo-server-proxy`
    - `./start_mihomo.sh`
    - `source ./proxy_env.sh`
  - Include commands to activate the environment and verify GPU visibility.
  - Include a minimal import smoke test for core packages.
  - Include a W&B connectivity smoke test that must pass before training starts.

- [x] Add shared Qwen-style extra0 utilities.
  - Create a module such as `scripts/verifier/qwen_extra0_prm.py`.
  - Resolve `<extra_0>` token id from the model tokenizer.
  - Format one full solution as `problem + step_1 <extra_0> step_2 <extra_0> ...`.
  - Build token-classification labels where ordinary positions are `-100` and
    `<extra_0>` positions are `0/1`.
  - Provide `score_steps(...)` that returns positive probabilities at all
    `<extra_0>` positions from a single forward pass.

- [x] Add an extra0 PRM training script.
  - Create `scripts/verifier/train_qwen_extra0_prm.py`.
  - Use `AutoModelForTokenClassification` with `num_labels=2`.
  - Default base model should be a local `models/Qwen2.5-Math-7B-Instruct`
    path populated from ModelScope on the server.
  - Support LoRA/PEFT as the first implementation path.
  - Keep `bf16`, gradient checkpointing, and small-checkpoint behavior.
  - Default dataset view should be `raw_phase1_phase2 + all-steps`.
  - Preserve the existing label policy: `rating >= 0 -> positive`.
  - Log label counts, number of `<extra_0>` positions, truncation rate, and
    dropped-supervision rate.

- [x] Add server model download/setup notes.
  - Create or update a runbook section with the ModelScope download command for
    Qwen2.5-Math-1.5B-Instruct.
  - Create or update a runbook section with the ModelScope download command for
    Qwen2.5-Math-7B-Instruct.
  - Store downloaded models under a local `models/` directory.
  - Keep paths configurable so server-local paths can differ from laptop paths.

- [x] Add a W&B metric contract for extra0 PRM training.
  - Log training/eval basics:
    - `extra0/train_loss`
    - `extra0/eval_loss`
    - `extra0/eval_accuracy`
    - `extra0/eval_pos_accuracy`
    - `extra0/eval_neg_accuracy`
    - `extra0/eval_balanced_accuracy`
    - `extra0/eval_pred_neg_fraction`
    - `extra0/eval_neg_auroc`
    - `extra0/eval_neg_average_precision`
  - Log data/format health:
    - `extra0/train_pos_count`
    - `extra0/train_neg_count`
    - `extra0/eval_pos_count`
    - `extra0/eval_neg_count`
    - `extra0/train_natural_neg_frac`
    - `extra0/eval_natural_neg_frac`
    - `extra0/train_extra0_positions_mean`
    - `extra0/eval_extra0_positions_mean`
    - `extra0/train_truncation_rate`
    - `extra0/eval_truncation_rate`
    - `extra0/dropped_label_frac`
  - Log imbalance/training configuration:
    - `extra0/neg_loss_weight`
    - `extra0/focal_gamma`
    - `extra0/rebalance_mode`
    - `extra0/effective_neg_weight_share`
    - `extra0/base_model`
    - `extra0/tuning_mode`
    - `extra0/lora_r`
    - `extra0/lora_alpha`
  - Log fixed-candidate reranking:
    - `best_of_n/reference_greedy_accuracy`
    - `best_of_n/reference_majority_vote_accuracy`
    - `best_of_n/reference_sample_oracle_accuracy`
    - `best_of_n/prm_best_of_16_accuracy`
    - `best_of_n/vs_greedy_gap`
    - `best_of_n/vs_majority_gap`
    - `best_of_n/misranking_frac`
    - `best_of_n/candidate_correct_score_mean`
    - `best_of_n/candidate_wrong_score_mean`
  - Log checkpoint metadata:
    - `extra0/best_checkpoint`
    - `extra0/best_metric_name`
    - `extra0/save_total_limit`

- [x] Keep negative imbalance controls conservative.
  - Default: no `WeightedRandomSampler`.
  - Keep `NEG_LOSS_WEIGHT` and optional focal loss.
  - Add a CLI/config flag for sampler rebalance only as an ablation.
  - Log natural negative fraction and effective weighted negative share.

- [x] Use reranking metric for checkpoint selection.
  - Evaluate on the fixed best-of-16 candidate JSONL already used by
    `eval_prm_best_of_n.py`.
  - Track `prm_best_of_16_accuracy`, `vs_greedy_gap`, and `vs_majority_gap`.
  - Prefer this reranking metric over `eval_best_balanced_accuracy` for best
    checkpoint selection.

## Phase 2: Integrate Extra0 PRM Into Offline Evaluation

- [x] Extend `scripts/verifier/eval_prm_best_of_n.py`.
  - Add backend support for `extra0_token_cls`.
  - Auto-detect extra0 PRM checkpoints when possible.
  - Preserve existing `classifier` and `token_prm` backends.
  - Reuse the same aggregation options: `min`, `mean`, `mean_log`, `sum_log`.

- [x] Add reranking diagnostics for the new backend.
  - Report candidate correct/wrong score means.
  - Report misranking count among examples where at least one correct sample
    exists.
  - Report selected-wrong-more-steps and selected-wrong-longer rates.
  - Compare against greedy, majority vote, and oracle on the same candidate pool.

## Phase 3: Integrate Extra0 PRM Into TRL GRPO Reward

- [x] Extend `scripts/trl/rewards.py` verifier loading.
  - Support three backends: classifier-head, old causal token PRM, and new
    extra0 token-classification PRM.
  - Keep old backends for ablations.
  - Make backend selection explicit in logs/metrics.

- [x] Use wrong-only shaping as the main reward mode.
  - Main formula:
    `reward = base_correct + beta * (1 - base_correct) * prm_score`
  - Correct final answers must not be penalized by PRM noise.
  - Wrong final answers must not be able to exceed correct final answers.
  - Keep all-wrong-only tie-break as an optional ablation, not the default.

- [x] Add PRM server reward backend for final 3+1 GPU training.
  - Implement a server such as `scripts/verifier/serve_qwen_extra0_prm.py`.
  - Load the extra0 PRM once on the dedicated PRM GPU.
  - Expose a local scoring API that accepts problem text and parsed steps.
  - Batch score requests where practical.
  - Add a TRL reward backend that calls the server instead of loading the PRM
    inside every policy rank.
  - Final intended launch shape:
    - `CUDA_VISIBLE_DEVICES=3` for the PRM server.
    - `CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 3` for
      TRL GRPO.

- [x] Add reward diagnostics.
  - Log `all_wrong_group_frac`, `mixed_group_frac`, `all_correct_group_frac`.
  - Log `wrong_sample_frac`.
  - Log PRM score mean/std for correct samples and wrong samples separately.
  - Log how often PRM shaping changes within-group ranking among wrong samples.

- [ ] Add a W&B metric contract for GRPO reward runs.
  - Log reward identity/config:
    - `reward/backend`
    - `reward/mode`
    - `reward/beta`
    - `reward/prm_model_path`
    - `reward/prm_aggregation`
  - Log answer reward:
    - `reward/base_accuracy`
    - `reward/base_reward_mean`
    - `reward/base_reward_std`
  - Log group composition:
    - `reward/all_wrong_group_frac`
    - `reward/mixed_group_frac`
    - `reward/all_correct_group_frac`
    - `reward/wrong_sample_frac`
  - Log PRM score behavior:
    - `reward/prm_score_mean`
    - `reward/prm_score_std`
    - `reward/prm_score_correct_mean`
    - `reward/prm_score_wrong_mean`
    - `reward/prm_score_correct_std`
    - `reward/prm_score_wrong_std`
  - Log shaping behavior:
    - `reward/shaping_mean`
    - `reward/shaping_std`
    - `reward/shaping_nonzero_frac`
    - `reward/wrong_shaping_mean`
    - `reward/wrong_shaping_std`
  - Log server metrics when using the final 3+1 GPU setup:
    - `reward_server/request_count`
    - `reward_server/batch_size_mean`
    - `reward_server/latency_ms_mean`
    - `reward_server/latency_ms_p95`
    - `reward_server/error_count`

- [ ] Update `scripts/trl/run_trl_grpo_math_verifier.sh`.
  - Add a reward mode variable such as `VERIFIER_REWARD_MODE=wrong_only`.
  - Keep `all_wrong_tiebreak` as an explicit mode.
  - Add beta sweep defaults or documented examples for `0.05`, `0.1`, `0.2`.

## Phase 4: Mark Legacy verl Reward Path

- [ ] Mark `scripts/verl/verl_verifier_reward.py` as legacy.
  - Do not port extra0 backend support into verl for now.
  - Add a short file-level note or runbook note that the active path is TRL.
  - Do not spend time debugging verl environment issues unless the user
    explicitly reactivates this path.

- [ ] Avoid using legacy verl reward formulas in new experiments.
  - Treat `beta * avg - delta * first_error` as a legacy ablation only.
  - Do not use it for the new extra0 PRM GRPO result.

## Phase 5: Reduce Step-Splitting Distribution Shift

- [ ] Standardize rollout step formatting.
  - Update policy prompt/instruction to ask for blank-line-separated reasoning
    steps before the final boxed answer.
  - Keep the final answer in `\boxed{}`.

- [ ] Tighten `scripts/verifier/step_splitter.py`.
  - Prefer splitting on blank lines and numbered steps.
  - Keep sentence splitting only as fallback.
  - Log or expose the number of parsed steps for reward diagnostics.

- [ ] Match PRM training and rollout scoring as much as possible.
  - PRM800K training uses human step lists.
  - GRPO scoring uses parsed rollout steps.
  - Add debug examples comparing raw rollout text, parsed steps, and inserted
    `<extra_0>` scoring input.

## Phase 6: Validation Plan

- [ ] Run lightweight syntax/import checks after code changes.
  - Check new modules compile.
  - Check modified reward scripts import without loading full models where
    possible.

- [ ] Run a tiny synthetic extra0 scoring smoke test.
  - Confirm the tokenizer contains `<extra_0>`.
  - Confirm label positions match `<extra_0>` positions.
  - Confirm one forward pass returns one score per step.

- [ ] Run fixed-candidate offline reranking before GRPO.
  - Use the same 100-problem, 16-candidate JSONL.
  - Generate or reuse candidates from the new main policy
    `Qwen2.5-Math-1.5B-Instruct` when making the main comparison.
  - Compare extra0 PRM against old causal token PRM and majority vote.
  - Do not start GRPO unless offline reranking is at least competitive enough
    to justify online shaping.

- [ ] Run GRPO experiments in early-stop order.
  - Majority voting baseline with `Qwen2.5-Math-1.5B-Instruct`.
  - Pure rule reward baseline with `Qwen2.5-Math-1.5B-Instruct`.
  - Wrong-only PRM shaping with `beta=0.05`.
  - Run `beta=0.1` only if `beta=0.05` does not beat the matched pure-rule
    baseline.
  - Run `beta=0.2` only if `beta=0.1` does not beat the matched pure-rule
    baseline.
  - Stop at the first wrong-only PRM GRPO run that beats the matched baseline.
  - Use the same data slice, seed, training steps, and eval set when comparing
    against baseline.
  - Optional all-wrong-only tie-break ablation only if needed for diagnosis,
    not as the main success path.

- [ ] Validate final 4-GPU execution shape.
  - First allow single-GPU smoke tests for reward correctness.
  - Final validation must use three GPUs for TRL GRPO policy and one GPU for
    the PRM reward server.
  - Confirm W&B logs show reward backend, mode, beta, server latency, and
    server batch size metrics.

## Phase 7: Documentation And Experiment Log

- [ ] Create documentation files under `docs/`.
  - Create `docs/experiment-log.md` for run-level results.
  - Create `docs/bad-cases.md` for PRM/GRPO bad cases.
  - Create `docs/debugging-notes.md` for runtime issues and fixes.
  - Keep all notes in Markdown.

- [ ] Update experiment notes after each run.
  - Record exact model path, PRM checkpoint, reward mode, beta, aggregation, and
    dataset slice.
  - Separate offline reranking results from online GRPO results.

- [ ] Record bad cases during PRM and GRPO work.
  - For each bad case, record:
    - date/time
    - run name
    - data split or example id
    - problem text or a short identifying excerpt
    - model completion
    - parsed steps
    - PRM step scores
    - final-answer correctness
    - why the case is bad
    - suspected cause
    - follow-up action
  - Keep examples concise enough to inspect but complete enough to reproduce.

- [ ] Record runtime/debugging issues and fixes.
  - For each issue, record:
    - date/time
    - command or script
    - environment details when relevant
    - error message or symptom
    - root cause if known
    - attempted fixes
    - final fix
    - whether the fix was verified
  - Include unresolved issues with an explicit `status: open`.

- [ ] Update failure analysis once the new protocol has results.
  - Keep the old conclusion that verifier quality does not automatically
    transfer to GRPO reward quality unless new evidence changes it.
  - Clearly separate old classifier-head/token-PRM results from new extra0 PRM
    results.
