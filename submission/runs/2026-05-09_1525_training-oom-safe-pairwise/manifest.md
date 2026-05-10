# 2026-05-09_1525_training-oom-safe-pairwise

## Purpose

Retry the failed training job with an OOM-safe configuration while keeping the RankMixer and pairwise-loss improvement path.

## Training Entry

- Upload archive: `submission/platform_uploads/2026-05-09_training_oom-safe-pairwise.zip`
- Entrypoint: `run.sh`
- Files: `run.sh`, `train.py`, `trainer.py`, `model.py`, `dataset.py`, `utils.py`, `ns_groups.json`

## Default Config

- `LOSS_TYPE=bce_pairwise`, with runtime fallback to `focal` if parser support is missing.
- `BATCH_SIZE=64`
- `SEQ_ENCODER_TYPE=longer`
- `SEQ_TOP_K=50`
- `SEQ_MAX_LENS=seq_a:128,seq_b:128,seq_c:256,seq_d:256`
- `NUM_WORKERS=4`
- `BUFFER_BATCHES=8`
- `NUM_QUERIES=2`
- `NUM_HYFORMER_BLOCKS=2`
- `D_MODEL=64`
- `NUM_HEADS=4`

## Reason

The previous platform log failed during the first forward pass with CUDA OOM under `BATCH_SIZE=256`, `SEQ_ENCODER_TYPE=transformer`, and sequence windows up to 512. This retry reduces activation memory through smaller batches, shorter windows, and the top-k `longer` sequence encoder.
