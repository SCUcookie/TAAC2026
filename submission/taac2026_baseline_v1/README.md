# TAAC2026 Baseline Submission v1

This directory is the first submission bundle for the Tencent TAAC2026 pCVR task.

Included files:
- `run.sh`: training entrypoint for the platform
- `train.py`: training script
- `infer.py`: inference / estimation script
- `dataset.py`: raw parquet dataset loader
- `model.py`: PCVRHyFormer baseline model
- `trainer.py`: trainer and checkpoint saver
- `utils.py`: logging / early stopping / loss helpers
- `ns_groups.json`: optional NS grouping config

Platform contract used by this bundle:
- training reads `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH`, `TRAIN_LOG_PATH`, `TRAIN_TF_EVENTS_PATH`
- inference reads `EVAL_DATA_PATH`, `MODEL_OUTPUT_PATH`, `EVAL_RESULT_PATH`

Checkpoint expectation:
- best checkpoint is stored under a subdirectory like `global_stepXXXX.best_model/model.pt`
- sidecar files `schema.json` and `train_config.json` are copied into the same checkpoint directory

Platform output expectation during inference:
- `predictions.json`

Optional local-only helper outputs:
- `submission.csv`
- `scores.npy`
- `inference_summary.json`

Current training config in `run.sh`:
- `ns_tokenizer_type=rankmixer`
- `user_ns_tokens=5`
- `item_ns_tokens=2`
- `num_queries=2`
- `emb_skip_threshold=1000000`
- `num_workers=8`

May 8 sequence-length sweep:

```bash
export TRAIN_DATA_PATH=/path/to/train_data
bash run_seq_len_sweep.sh
```

Default sweep candidates:
- `baseline_a256`: `seq_a:256,seq_b:256,seq_c:512,seq_d:512`
- `candidate_a512`: `seq_a:512,seq_b:256,seq_c:512,seq_d:512`
- `candidate_a768`: `seq_a:768,seq_b:256,seq_c:512,seq_d:512`

The sweep writes per-run logs, checkpoints, TensorBoard events, and `run_record.md`
under `experiments/seq_len_2026_05_08/`. Override `TRAIN_RATIO`,
`BATCH_SIZE`, `NUM_EPOCHS`, or `BASE_OUTPUT_DIR` for local constraints.

Recommended workflow:
1. Submit this directory as code.
2. Run training with `run.sh`.
3. Pick the best checkpoint directory under `TRAIN_CKPT_PATH`.
4. Run local inference with `infer.py`.
5. Validate the produced `predictions.json` with `verify_infer.py`.
6. Only then spend a platform evaluation slot.
7. Record platform feedback under `feedbacks/platform_runs/`.

Example local verification:

```bash
export EVAL_DATA_PATH=/path/to/eval_data
export MODEL_OUTPUT_PATH=/path/to/global_stepXXXX.best_model
export EVAL_RESULT_PATH=./eval_result
python infer.py --write_aux_outputs
python verify_infer.py --result_dir ./eval_result
```
