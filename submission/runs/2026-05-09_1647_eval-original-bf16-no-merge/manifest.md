# 2026-05-09_1647_eval-original-bf16-no-merge

## Purpose

Retry original-baseline evaluation after the BF16 observable run timed out. The log showed the model forward was fast but CPU-side row-group merging dominated runtime.

## Evaluation Entry

- Upload archive: `submission/platform_uploads/2026-05-09_eval_original-bf16-no-merge.zip`
- Entrypoint: `infer.py`
- Files: `infer.py`, `model.py`, `dataset.py`, `ns_groups.json`
- Model: original baseline checkpoint selected on the platform

## Default Config

- `EVAL_BATCH_SIZE=2048`
- `EVAL_USE_AMP=1`
- `EVAL_AMP_DTYPE=bf16`
- `EVAL_MERGE_BATCHES=0`
- `EVAL_PROGRESS_EVERY=10`
- Original checkpoint sequence lengths are preserved.

## Reason

The previous run reached `237135 / 310000` predictions after `1962.399s`. Average per-forward timing was approximately `merge=9.1777s`, `forward=0.9442s`, `move=0.1489s`, `post=0.0006s`. This retry disables batch merging by default and forwards each row-group batch immediately.
