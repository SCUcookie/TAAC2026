# TAAC2026 Evaluation Workflow

Use this workflow to avoid wasting the daily evaluation quota.

## 1. Train locally or on platform training

For each run, record:
- training command
- best validation AUC / logloss
- best checkpoint path
- runtime
- any OOM / instability

Do not evaluate every run. Promote only the best locally validated checkpoint.

## 2. Run local inference before platform evaluation

Use the self-contained checkpoint directory:

```bash
export EVAL_DATA_PATH=/path/to/eval_data
export MODEL_OUTPUT_PATH=/path/to/global_stepXXXX.best_model
export EVAL_RESULT_PATH=./eval_result
python submission/taac2026_baseline_v1/infer.py --write_aux_outputs
python submission/taac2026_baseline_v1/verify_infer.py --result_dir ./eval_result
```

Pass conditions:
- `predictions.json` exists
- top-level JSON shape is `{"predictions": {user_id: score}}`
- all keys are strings
- all values are numeric and non-negative
- prediction count is plausible for the target dataset

## 3. Daily evaluation slot policy

- Slot 1: current best local checkpoint only
- Slot 2: one materially different backup checkpoint
- Slot 3: reserve for a packaging fallback or a confirmed stronger rerun

Do not use evaluation slots for:
- schema discovery
- logging experiments
- unverified inference packaging
- weak local candidates

## 4. Recommended experiment order

1. Sequence truncation via `seq_max_lens`
2. Throughput and stability via `batch_size` and `num_workers`
3. Model capacity via `d_model`, `num_hyformer_blocks`, `num_queries`
4. Sparse control via `emb_skip_threshold`
5. NS tokenizer variant only after the earlier items stabilize

## 5. Minimum run log template

Record each train/eval in `feedbacks/platform_runs/` with:
- date and time
- submission directory
- exact command or platform settings
- best checkpoint path
- local validation metric
- whether local inference verification passed
- whether platform evaluation passed
- observed leaderboard metric
