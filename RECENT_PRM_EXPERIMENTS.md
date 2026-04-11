# Recent PRM Experiment Results

Last updated: 2026-04-10

This note records the recent PRM / verifier experiments that we actually ran and
measured on the server. The goal is to separate hard results from interpretation.

## 1. Current Checkpoints And Defaults

- Current token-PRM checkpoint used in GRPO experiments:
  - `/root/autodl-tmp/prm_grpo/token_prm_runs/prm-token-fullft-phase2raw-nonneg-firsterr-eval12-negw10p0-focalg2p0-qwen25-math-1.5b/final`
- This checkpoint was trained with:
  - data source: `raw_phase2`
  - label mapping: `rating >= 0 -> positive`
  - truncation: `first-error-only`
  - training mode: full fine-tuning
  - negative loss weight: `10.0`
  - focal gamma: `2.0`
  - no negative oversampling / no sampler rebalance

- Current code default in `scripts/verifier/train_token_prm.py` is no longer phase2-only:
  - `DATASET_SOURCE = "raw_phase1_phase2"`
- This means:
  - current code default = `phase1 + phase2`
  - current deployed PRM checkpoint = `phase2 only`

## 2. Dataset Statistics We Already Measured

These numbers come from `scripts/verifier/prepare_openai_prm800k.py` runs on the server.

### 2.1 Phase2, `nonnegative`, `first-error-only`

Train split:

- rows = `89,611`
- steps = `581,390`
- pos = `579,770`
- neg = `1,620`
- neg_step_frac = `0.0028`

Test split:

- rows = `2,525`
- steps = `16,493`
- pos = `16,450`
- neg = `43`
- neg_step_frac = `0.0026`

Takeaway:

- Under `first-error-only`, phase2 becomes extremely positive-dominated at the step level.

### 2.2 Phase2, `nonnegative`, `all-steps`

Train split:

- rows = `474,962`
- steps = `2,968,416`
- pos = `2,733,589`
- neg = `234,827`
- neg_step_frac = `0.0791`
- neg_row_frac = `0.4944`

Test split:

- rows = `12,300`
- steps = `71,923`
- pos = `65,831`
- neg = `6,092`
- neg_step_frac = `0.0847`
- neg_row_frac = `0.4953`

Takeaway:

- Phase2 has many negative trajectories, but still mostly positive steps.
- Trajectory-level balance and step-level balance are very different.

### 2.3 Phase1 + Phase2

Train split:

- rows = `487,310`
- steps = `3,041,565`
- pos = `2,802,055`
- neg = `239,510`
- neg_step_frac = `0.0787`
- neg_row_frac = `0.4915`

Test split:

- rows = `13,451`
- steps = `79,478`
- pos = `72,909`
- neg = `6,569`
- neg_step_frac = `0.0827`
- neg_row_frac = `0.4884`

Takeaway:

- The mixed `phase1 + phase2` view is very close to `phase2 all-steps` at the step-level class ratio.
- For token-PRM training, the relevant imbalance is still step-level, not row-level.

## 3. Offline Best-of-N Evaluation

Protocol:

- evaluation set: `100` MATH test problems
- base model: `Qwen2.5-Math-1.5B`
- sampled completions per problem: `16`
- sample temperature: `0.8`
- `max_new_tokens = 1024`
- `verifier_max_length = 1024`

Common reference numbers for this candidate pool:

- greedy accuracy = `0.6000`
- majority vote accuracy = `0.7800`
- sample oracle accuracy = `0.9300`
- sample boxed rate = `0.9563`

Interpretation of the reference numbers:

- the candidate pool is strong
- in 93/100 problems, at least one of the 16 sampled answers is correct
- a useful PRM should be able to exploit this

### 3.1 Token PRM, `mean_log` aggregation

Run summary:

- `prm_best_of_16_accuracy = 0.5300`

Artifact:

- `/root/autodl-tmp/prm_grpo/outputs/prm_best_of_n/math_test_100_best_of_16.jsonl`

### 3.2 Classifier verifier checkpoint-6000

Run summary:

- `prm_best_of_16_accuracy = 0.5400`

Artifact:

- `/root/autodl-tmp/prm_grpo/outputs/prm_best_of_n/math_test_100_best_of_16_cls_checkpoint6000.jsonl`

### 3.3 Token PRM, `min` aggregation

Run summary:

- `prm_best_of_16_accuracy = 0.7000`

Artifact:

- `/root/autodl-tmp/prm_grpo/outputs/prm_best_of_n/math_test_100_best_of_16_tokenprm_min.jsonl`

### 3.4 Best-of-N Summary Table

| Method | Accuracy |
| --- | ---: |
| Greedy | `0.6000` |
| PRM best-of-16, token PRM, `mean_log` | `0.5300` |
| PRM best-of-16, cls checkpoint-6000 | `0.5400` |
| PRM best-of-16, token PRM, `min` | `0.7000` |
| Majority vote | `0.7800` |
| Oracle over 16 samples | `0.9300` |

Key takeaway:

- `mean_log` aggregation was actively harmful.
- `min` aggregation recovered a large amount of useful signal.
- The current token PRM is not useless; it is highly aggregation-sensitive.

## 4. Reranking Diagnostics

These numbers come from `scripts/verifier/analyze_prm_rerank_jsonl.py`.

### 4.1 Token PRM with `mean_log`

- num_examples = `100`
- greedy_accuracy = `0.6000`
- majority_vote_accuracy = `0.7800`
- prm_best_accuracy = `0.5300`
- sample_oracle_accuracy = `0.9300`
- correct_available_count = `93`
- misranking_count = `40`
- misranking_frac_among_correct_available = `0.4301`
- candidate_correct_score_mean = `-0.6414`
- candidate_wrong_score_mean = `-0.6210`
- selected_correct_score_mean = `-0.4663`
- selected_wrong_score_mean = `-0.4787`
- best_correct_rank_mean = `2.6774`
- selected_wrong_more_steps_frac = `0.3750`
- selected_wrong_longer_frac = `0.3250`

Interpretation:

- In the `mean_log` setting, wrong candidates received higher scores on average than correct candidates.
- This is strong evidence of systematic misranking.

### 4.2 Token PRM with `min`

- num_examples = `100`
- greedy_accuracy = `0.6000`
- majority_vote_accuracy = `0.7800`
- prm_best_accuracy = `0.7000`
- sample_oracle_accuracy = `0.9300`
- correct_available_count = `93`
- misranking_count = `23`
- misranking_frac_among_correct_available = `0.2473`
- candidate_correct_score_mean = `0.2679`
- candidate_wrong_score_mean = `0.2494`
- selected_correct_score_mean = `0.3934`
- selected_wrong_score_mean = `0.3894`
- best_correct_rank_mean = `2.1075`
- selected_wrong_more_steps_frac = `0.3043`
- selected_wrong_longer_frac = `0.3478`

Interpretation:

- Under `min`, correct candidates now score higher on average than wrong candidates.
- Misranking does not disappear, but it drops substantially.

## 5. What These Results Suggest

### 5.1 What seems clearly true

- The current PRM is **not** well used by `mean_log` aggregation.
- The same token PRM becomes much more useful under `min` aggregation.
- Therefore, at least part of the earlier failure came from **aggregation mismatch**, not just from a bad checkpoint.

### 5.2 What remains unresolved

- Even with `min`, PRM best-of-16 (`0.7000`) still trails majority vote (`0.7800`).
- So the current PRM is useful, but still not strong enough to dominate simple answer-level selection.
- We still do not have evidence that the current online GRPO reward shaping matches the successful offline reranking behavior.

## 6. GRPO Status Note

Recent GRPO experiments suggest:

- naive verifier shaping often had near-zero effect
- increasing verifier influence directly could hurt performance
- offline reranking utility does **not** automatically transfer to online reward shaping

Current important caveat:

- Offline best-of-N uses PRM for **ranking complete candidates**
- Current GRPO shaping uses PRM as an **auxiliary reward term**
- These are related, but they are not the same intervention

## 7. Next Useful Experiments

1. Measure exact class ratios for `raw_phase1_phase2`.
2. Rerun token-PRM training with a small ablation on `NEG_LOSS_WEIGHT`:
   - `10`
   - `5`
   - `2`
3. Evaluate every checkpoint with the same offline protocol:
   - same 100 problems
   - same reused 16 candidates
   - `prm_aggregation = min`
4. Prefer offline reranking metrics over training balanced accuracy when selecting PRM checkpoints.
