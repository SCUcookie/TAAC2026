# Evaluation Repeat 60 Too Long

## Date

2026-05-10 16:55

## Run ID

`2026-05-10_1655_eval-repeat60-too-long`

## Observation

The `EVAL_TTA_REPEAT=60` run was too slow.

Latest copied progress from `temp.md`:

- `tta_repeat`: `60`
- `seq_max_lens`: `seq_a:256,seq_b:256,seq_c:512,seq_d:448`
- At `1976.811s`, only `111723 / 310000` predictions were complete.
- Progress throughput was about `56.52 rows/s`.

Projected full runtime:

`310000 / 56.52 = 5484s`

## Diagnosis

The earlier repeat calibration from completed short runs was not linear for
large repeat counts. Runtime scales much worse once each row-group forward
contains 60 recent-view passes.

The observed stable average forward time was about `5.30s` per row-group batch.
There are `1000` row-group batches, so this setting cannot finish near the
target time.

## Action

Reduce packaged default:

`EVAL_TTA_REPEAT=14`

Rationale:

The observed `repeat=60` forward time was about `5.30s` per row-group batch.
Approximating the per-pass cost as `5.30 / 61 = 0.0869s`, repeat `14` gives
about `15 * 0.0869 * 1000 = 1303s` plus startup and output overhead.

## Next Sweep

Submit with default `EVAL_TTA_REPEAT=14`.

If runtime is still above `1350s`, use `EVAL_TTA_REPEAT=12`.

If runtime is below `1150s`, use `EVAL_TTA_REPEAT=16`.
