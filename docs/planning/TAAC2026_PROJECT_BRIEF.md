# TAAC2026 Project Brief

## Project

TAAC2026 pCVR training and platform-safe evaluation.

## Current Status

- Evaluation dataset structure has been confirmed from the platform probe.
- Evaluation data is flat Parquet under `EVAL_DATA_PATH=/data_ams/academic_infer_data/`.
- Eval set size is `310000` rows across `1000` parquet files / row groups.
- Column count is about `120`.
- Schema matches the baseline-style layout:
  - user sparse features
  - item sparse features
  - user dense features
  - 4 sequence domains: `seq_a`, `seq_b`, `seq_c`, `seq_d`
- `seq_a` is long and likely important: sampled max sequence length is near `1999`.
- Platform inference contract is now confirmed:
  - upload a root-level `infer.py`
  - output `predictions.json`
  - JSON format must be:
    - `{"predictions": {"<user_id>": <float>}}`
- The baseline inference path has already been fixed to match that contract.
- A local validator now exists to check inference output before using an evaluation slot:
  - `submission/taac2026_baseline_v1/verify_infer.py`

## Files Updated So Far

- `submission/taac2026_baseline_v1/infer.py`
- `submission/taac2026_baseline_v1/verify_infer.py`
- `submission/taac2026_baseline_v1/README.md`
- `docs/taac2026_submission_v1.md`
- `docs/taac2026_eval_workflow.md`

## Main Constraints

- Training a model takes several hours.
- Evaluation opportunities are limited to 3 per day.
- Failed packaging or format submissions waste valuable evaluation budget.
- Model iteration must be driven by local validation first, then platform evaluation.

## Main Risks

- Long sequence truncation may be harming model quality.
- High-cardinality sparse features may be inefficient or overfitting.
- Training cost is high, so too many full experiments will waste time.
- Evaluation slots are scarce, so inference packaging must remain stable.

## Immediate Objective

Improve model quality while keeping evaluation submissions low-risk.

## Tomorrow Priority Order

1. Review current training defaults in `run.sh` and `train.py`.
2. Prioritize `seq_max_lens` experiments.
3. Run short smoke experiments before any full training run.
4. Use local inference plus `verify_infer.py` before any platform evaluation.
5. Spend evaluation slots only on the strongest locally validated checkpoints.

## Recommended Experiment Order

1. Sequence truncation via `seq_max_lens`
2. Throughput and stability via `batch_size` and `num_workers`
3. Model capacity via `d_model`, `num_hyformer_blocks`, and `num_queries`
4. Sparse-control settings such as `emb_skip_threshold`
5. NS tokenizer variant only after the earlier items stabilize

## Suggested First Experiments

Keep the architecture unchanged at first:

- `d_model=64`
- `num_hyformer_blocks=2`
- `num_heads=4`
- `num_queries=2`

Focus first on sequence truncation:

- current:
  - `seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- test candidate A:
  - `seq_a:512,seq_b:256,seq_c:512,seq_d:512`
- test candidate B:
  - `seq_a:768,seq_b:256,seq_c:512,seq_d:512`

## Promotion Rule For Evaluation

Only spend an evaluation slot when all of the following are true:

- training finished cleanly
- validation metric is better than the current local best
- local inference finished successfully
- `verify_infer.py` passed
- the upload bundle still uses the correct root-level `infer.py`

## Working Principle

Do not use evaluation slots for:

- schema discovery
- logging experiments
- unverified packaging changes
- weak local checkpoints

Use local validation and local inference verification as the gate before platform evaluation.
