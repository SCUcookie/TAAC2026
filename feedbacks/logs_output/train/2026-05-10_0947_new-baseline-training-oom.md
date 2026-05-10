# New Baseline Training OOM Raw Log Pointer

The raw platform log for this failed training attempt was copied into `temp.md` on 2026-05-10.

Key failure:

`torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB`

Important signals:

- Data loaded successfully from `/data_ams/academic_training_data`.
- Row group split: `900` train row groups, `100` valid row groups.
- Train rows: `907381`; valid rows: `102619`.
- Model constructed successfully with `778,116,085` parameters.
- Failure occurred during the first training step in `model.py` sequence self-attention.
