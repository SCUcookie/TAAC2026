2026-05-09 10:46:00,280 - INFO - Saved status.json to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75473/results/status.json
2026-05-09 18:46:00.281
2026-05-09 18:46:00.278
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
2026-05-09 18:46:00.278
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
2026-05-09 18:46:00.278
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
2026-05-09 18:46:00.278
2026-05-09 10:46:00,278 - ERROR - Saved error result to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75473/results/result.json: CUDA error: out of memory
2026-05-09 18:46:00.278
2026-05-09 18:46:00.276
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
2026-05-09 18:46:00.276
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
2026-05-09 18:46:00.276
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
2026-05-09 18:46:00.276
2026-05-09 10:46:00,275 - ERROR - Inference failed: CUDA error: out of memory
2026-05-09 18:46:00.276
2026-05-09 10:45:59,696 - INFO - emb_skip_threshold=1000000: seq_c skipped 3/11 features
2026-05-09 18:45:59.696
2026-05-09 10:45:59,696 - INFO - emb_skip_threshold=1000000: seq_b skipped 1/13 features
2026-05-09 18:45:59.696
2026-05-09 10:45:57,685 - INFO - RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0
2026-05-09 18:45:57.685
2026-05-09 10:45:57,666 - INFO - RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1
2026-05-09 18:45:57.667
PLATFORM_DATASET_CONFIG {"batch_size": 2048, "item_dense_dim": 0, "item_int_features": 14, "num_row_groups": 1000, "num_rows_estimate": 310000, "seq_domains": ["seq_a", "seq_b", "seq_c", "seq_d"], "seq_max_lens": {"seq_a": 256, "seq_b": 256, "seq_c": 512, "seq_d": 512}, "user_dense_dim": 918, "user_int_features": 46}
2026-05-09 18:45:57.643
2026-05-09 10:45:57,642 - INFO - PCVRParquetDataset: 310000 rows from 1000 file(s), batch_size=2048, buffer_batches=0, shuffle=False
2026-05-09 18:45:57.643
PLATFORM_INFER_CONFIG {"amp_dtype": "bf16", "checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "device": "cuda", "eval_data_dir": "/data_ams/academic_infer_data/", "merge_batches": false, "model_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt", "num_workers": 0, "progress_every": 10, "result_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75473/results", "schema_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "train_config": {"d_model": 64, "emb_skip_threshold": 1000000, "eval_batch_size": 2048, "loss_type": "bce", "num_heads": 4, "num_hyformer_blocks": 2, "num_queries": 2, "seq_encoder_type": "transformer", "seq_max_lens": "seq_a:256,seq_b:256,seq_c:512,seq_d:512", "train_batch_size": 256}, "tta_crop_lens": "seq_a:128,seq_b:128,seq_c:256,seq_d:256", "tta_mode": "recent_blend", "tta_weight": 0.12, "use_amp": true}
2026-05-09 18:45:57.150
2026-05-09 10:45:57,150 - INFO - Using schema path: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json
2026-05-09 18:45:57.150
2026-05-09 10:45:57,149 - INFO - Using checkpoint directory: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model
2026-05-09 18:45:57.150
PLATFORM_CHECKPOINT_STATUS {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "model_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt", "size_bytes": 959871995}, "schema_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "size_bytes": 5560}, "train_config_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/train_config.json", "size_bytes": 1638}}
2026-05-09 18:45:57.147
PLATFORM_INFER_START {"cuda_available": true, "cwd": "/workspace", "time": "2026-05-09 10:45:57", "torch_version": "2.7.1+cu126"}
2026-05-09 18:45:57.140
====== Inferring ======
2026-05-09 18:45:55.858
=================================
2026-05-09 18:45:55.491
Working Dir: /workspace
2026-05-09 18:45:55.491
Environment: competition
2026-05-09 18:45:55.491
GPU Count: 1
2026-05-09 18:45:55.491
GPU Available: True
2026-05-09 18:45:54.073
PyTorch: 2.7.1+cu126
2026-05-09 18:45:52.701
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-09 18:45:50.404
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-09 18:45:50.404
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-09 18:45:50.404
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-09 18:45:50.404
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-09 18:45:50.401
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-09 18:45:50.401
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-09 18:45:50.401
Python: Python 3.10.20
2026-05-09 18:45:50.072
CUDA: 12.6.77
2026-05-09 18:45:50.068
=== Competition Environment Ready ===
2026-05-09 18:45:50.060
Complete setting network policy rules.
2026-05-09 18:45:46.622
Complete setting taiji user.
2026-05-09 18:45:46.566
Truncated to last 1000 lines.