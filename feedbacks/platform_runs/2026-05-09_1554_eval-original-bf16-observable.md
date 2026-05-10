# Original Baseline BF16 Observable Evaluation

## Basic Info
- Date: 2026-05-09 15:54
- Run type: eval
- Submission directory: `submission/platform_uploads/2026-05-09_eval_original-bf16-observable.zip`
- Platform round: pending
- Platform score: pending
- Previous best score: `0.813033`

## Training Setup
- Entry script: `infer.py`
- Extra args: original baseline checkpoint selected on platform
- seq_max_lens: from original checkpoint train config, expected `seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- train_ratio / valid_ratio: N/A
- batch_size / num_workers: `2048 / 0`
- model config: original baseline checkpoint, no architecture changes
- loss config: N/A
- Environment notes: BF16 AMP enabled by default, with automatic fp32 fallback if BF16 is unsupported or produces non-finite scores

## Platform Log Signals
- Finished successfully: no
- Import / packaging errors: no
- `PLATFORM_INFER_CONFIG`: present, `use_amp=true`, `amp_dtype=bf16`, `batch_size=2048`
- `PLATFORM_DATASET_CONFIG`: present, `310000` estimated rows, `1000` row groups
- `PLATFORM_AMP_CONFIG`: present, BF16 supported and enabled
- `PLATFORM_FIRST_BATCH`: present, no score sanitization
- `PLATFORM_INFER_PROGRESS`: reached `237135` predictions after `1962.399s`
- `PLATFORM_AMP_FALLBACK`: absent
- `PLATFORM_INFER_SUMMARY`: not reached
- Local inference verified: syntax check only
- Platform leaderboard metric: pending
- Runtime: timed out after at least `1962.399s`
- prediction count: `237135` before timeout
- unique user count: not emitted
- score min / max / mean / std: pending
- score p01 / p50 / p99: pending

## Logs / Errors
- Key log lines: `PLATFORM_INFER_PROGRESS {"amp_mode":"bf16","batches":110,"elapsed_sec":1962.399,"predictions":237135,...,"timing_avg_sec":{"forward":0.9442,"merge":9.1777,"move":0.1489,"post":0.0006}}`
- Error message: platform timeout during inference

## Decision
- Compared with baseline `0.813033`: no score returned
- Keep / revert config: keep BF16, remove default batch merging
- Next change: submit `submission/platform_uploads/2026-05-09_eval_original-bf16-no-merge.zip`
