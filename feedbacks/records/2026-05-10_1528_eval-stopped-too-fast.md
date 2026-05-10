# Evaluation Stopped Too Fast

## Date

2026-05-10 15:28

## Run ID

`2026-05-10_1528_eval-stopped-too-fast`

## Observation

The evaluation process was stopped manually because the observed inference time
was about `145s`, still far below the previous useful baseline runtime around
`1200s`.

The latest copied `temp.md` still contains the earlier completed
`seq_a:256,seq_b:256,seq_c:256,seq_d:256` run with:

- AUC: `0.811994`
- Platform inference time: `61.5s`

## Diagnosis

The current evaluator is spending too little of the inference-time budget. The
low AUC is likely caused by aggressive truncation of the longer sequence
domains, especially `seq_c` and `seq_d`.

## Action

Change the default evaluation window in `New_baseline/infer.py` from the
middle-point `seq_a:256,seq_b:256,seq_c:384,seq_d:384` to:

`seq_a:256,seq_b:256,seq_c:512,seq_d:448`

This jumps much closer to the full-window setting while avoiding the known slow
`seq_c:512,seq_d:512` configuration that completed internally in about
`1890s`.

## Next Sweep

Submit with the packaged default first. If it is still too fast, override with:

`EVAL_SEQ_MAX_LENS=seq_a:256,seq_b:256,seq_c:512,seq_d:512`

If it is close to timeout, step down to:

`EVAL_SEQ_MAX_LENS=seq_a:256,seq_b:256,seq_c:512,seq_d:416`
