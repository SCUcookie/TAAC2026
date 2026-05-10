# New Baseline Training OOM

## Date

2026-05-10 09:47

## Run ID

`2026-05-10_0947_new-baseline-training-oom`

## Experiment

First platform training attempt with the modified `New_baseline/` training files. `infer.py` was not part of this attempt.

## Configuration

- Entry script: `New_baseline/run.sh`
- Batch size: `256`
- Sequence lengths: `seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- Sequence encoder: `transformer`
- High-cardinality mode: `hash`
- EMA: enabled
- Model size: `778,116,085` total parameters
- Sparse params: `774,311,296`
- Dense params: `3,804,789`

## Result

Training failed on the first batch with CUDA OOM inside sequence self-attention:

`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB`

The run reached data loading and model construction successfully, so this is a memory-capacity failure rather than packaging, path, or schema failure.

## Decision

The active launch recipe is too heavy for the platform GPU slice. The main memory risks are long full-sequence transformer attention, hashed high-cardinality embeddings, batch size `256`, buffer size `20`, and EMA copying all model tensors.

## Next Action

Retry with the OOM-safe `New_baseline/run.sh` defaults:

- batch size `64`
- buffer batches `8`
- sequence lengths `seq_a:128,seq_b:128,seq_c:256,seq_d:256`
- sequence encoder `longer`
- high-cardinality mode `zero`
- EMA disabled by default
- keep RankMixer/new feature stack active for the smoke run
