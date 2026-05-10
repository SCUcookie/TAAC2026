PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 110, "elapsed_sec": 1962.399, "predictions": 237135, "recent_rows_per_sec": 122.28, "row_group_batches": 775, "rows_per_sec": 120.84, "timing_avg_sec": {"forward": 0.9442, "merge": 9.1777, "move": 0.1489, "post": 0.0006}}
2026-05-09 16:40:17.799
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 100, "elapsed_sec": 1794.912, "predictions": 214352, "recent_rows_per_sec": 120.02, "row_group_batches": 705, "rows_per_sec": 119.42, "timing_avg_sec": {"forward": 0.9374, "merge": 9.2244, "move": 0.1509, "post": 0.0006}}
2026-05-09 16:37:30.312
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 90, "elapsed_sec": 1624.27, "predictions": 192160, "recent_rows_per_sec": 129.61, "row_group_batches": 635, "rows_per_sec": 118.31, "timing_avg_sec": {"forward": 0.9366, "merge": 9.2737, "move": 0.1495, "post": 0.0006}}
2026-05-09 16:34:39.670
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 80, "elapsed_sec": 1466.259, "predictions": 170369, "recent_rows_per_sec": 168.33, "row_group_batches": 565, "rows_per_sec": 116.19, "timing_avg_sec": {"forward": 0.9502, "merge": 9.4478, "move": 0.1502, "post": 0.0006}}
2026-05-09 16:32:01.659
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 70, "elapsed_sec": 1344.597, "predictions": 148941, "recent_rows_per_sec": 85.41, "row_group_batches": 495, "rows_per_sec": 110.77, "timing_avg_sec": {"forward": 0.9677, "merge": 9.9032, "move": 0.1546, "post": 0.0006}}
2026-05-09 16:29:59.997
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 60, "elapsed_sec": 1104.825, "predictions": 128019, "recent_rows_per_sec": 84.65, "row_group_batches": 425, "rows_per_sec": 115.87, "timing_avg_sec": {"forward": 0.9652, "merge": 9.4504, "move": 0.156, "post": 0.0006}}
2026-05-09 16:26:00.225
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 50, "elapsed_sec": 862.902, "predictions": 106463, "recent_rows_per_sec": 92.53, "row_group_batches": 352, "rows_per_sec": 123.38, "timing_avg_sec": {"forward": 0.9692, "merge": 8.7537, "move": 0.1544, "post": 0.0006}}
2026-05-09 16:21:58.301
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 40, "elapsed_sec": 641.576, "predictions": 84821, "recent_rows_per_sec": 116.02, "row_group_batches": 280, "rows_per_sec": 132.21, "timing_avg_sec": {"forward": 0.9487, "merge": 8.269, "move": 0.155, "post": 0.0006}}
2026-05-09 16:18:16.975
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 30, "elapsed_sec": 465.059, "predictions": 63679, "recent_rows_per_sec": 133.96, "row_group_batches": 210, "rows_per_sec": 136.93, "timing_avg_sec": {"forward": 0.9396, "merge": 7.9391, "move": 0.1548, "post": 0.0006}}
2026-05-09 16:15:20.458
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 20, "elapsed_sec": 312.178, "predictions": 42862, "recent_rows_per_sec": 171.57, "row_group_batches": 140, "rows_per_sec": 137.3, "timing_avg_sec": {"forward": 0.9398, "merge": 7.7348, "move": 0.1501, "post": 0.0006}}
2026-05-09 16:12:47.578
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 10, "elapsed_sec": 192.81, "predictions": 21610, "recent_rows_per_sec": 106.22, "row_group_batches": 70, "rows_per_sec": 112.08, "timing_avg_sec": {"forward": 0.95, "merge": 9.6495, "move": 0.1391, "post": 0.0006}}
2026-05-09 16:10:48.210
PLATFORM_FIRST_BATCH {"batch_predictions": 2192, "rows_in_forward": 2192, "sample_scores": [0.00370789, 0.07470703, 0.03955078, 0.17675781, 0.44921875], "sample_user_ids": ["683974", "1334888", "51348", "1184407", "8854"], "sanitize_counts": {"nan_replaced": 0, "neginf_replaced": 0, "posinf_replaced": 0}, "timing_sec": {"forward": 1.4176, "merge": 7.8009, "move": 0.1474, "post": 0.0006}}
2026-05-09 16:07:57.647
2026-05-09 08:07:35,761 - INFO - NumExpr defaulting to 16 threads.
2026-05-09 16:07:35.762
2026-05-09 08:07:35,761 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-09 16:07:35.761
2026-05-09 08:07:35,761 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-09 16:07:35.761
PLATFORM_AMP_CONFIG {"cuda_bf16_supported": true, "enabled": true, "mode": "bf16", "requested": "bf16"}
2026-05-09 16:07:35.400
PLATFORM_MODEL_READY {"device": "cuda", "num_ns": 8, "num_parameters": 239931649}
2026-05-09 16:07:35.400
2026-05-09 08:07:28,530 - INFO - emb_skip_threshold=1000000: seq_c skipped 3/11 features
2026-05-09 16:07:28.530
2026-05-09 08:07:28,530 - INFO - emb_skip_threshold=1000000: seq_b skipped 1/13 features
2026-05-09 16:07:28.530
2026-05-09 08:07:26,546 - INFO - RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0
2026-05-09 16:07:26.546
2026-05-09 08:07:26,527 - INFO - RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1
2026-05-09 16:07:26.527
PLATFORM_DATASET_CONFIG {"batch_size": 2048, "item_dense_dim": 0, "item_int_features": 14, "num_row_groups": 1000, "num_rows_estimate": 310000, "seq_domains": ["seq_a", "seq_b", "seq_c", "seq_d"], "seq_max_lens": {"seq_a": 256, "seq_b": 256, "seq_c": 512, "seq_d": 512}, "user_dense_dim": 918, "user_int_features": 46}
2026-05-09 16:07:26.502
2026-05-09 08:07:26,502 - INFO - PCVRParquetDataset: 310000 rows from 1000 file(s), batch_size=2048, buffer_batches=0, shuffle=False
2026-05-09 16:07:26.502
PLATFORM_INFER_CONFIG {"amp_dtype": "bf16", "checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "device": "cuda", "eval_data_dir": "/data_ams/academic_infer_data/", "model_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt", "num_workers": 0, "progress_every": 10, "result_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75092/results", "schema_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "train_config": {"d_model": 64, "emb_skip_threshold": 1000000, "eval_batch_size": 2048, "loss_type": "bce", "num_heads": 4, "num_hyformer_blocks": 2, "num_queries": 2, "seq_encoder_type": "transformer", "seq_max_lens": "seq_a:256,seq_b:256,seq_c:512,seq_d:512", "train_batch_size": 256}, "use_amp": true}
2026-05-09 16:07:26.021
2026-05-09 08:07:26,021 - INFO - Using schema path: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json
2026-05-09 16:07:26.021
2026-05-09 08:07:26,021 - INFO - Using checkpoint directory: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model
2026-05-09 16:07:26.021
PLATFORM_CHECKPOINT_STATUS {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "model_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt", "size_bytes": 959871995}, "schema_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "size_bytes": 5560}, "train_config_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/train_config.json", "size_bytes": 1638}}
2026-05-09 16:07:26.019
PLATFORM_INFER_START {"cuda_available": true, "cwd": "/workspace", "time": "2026-05-09 08:07:26", "torch_version": "2.7.1+cu126"}
2026-05-09 16:07:26.012
====== Inferring ======
2026-05-09 16:07:24.765
=================================
2026-05-09 16:07:24.380
Working Dir: /workspace
2026-05-09 16:07:24.380
Environment: competition
2026-05-09 16:07:24.380
GPU Count: 1
2026-05-09 16:07:24.380
GPU Available: True
2026-05-09 16:07:22.825
PyTorch: 2.7.1+cu126
2026-05-09 16:07:21.315
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-09 16:07:18.918
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-09 16:07:18.918
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-09 16:07:18.918
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-09 16:07:18.918
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-09 16:07:18.915
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-09 16:07:18.915
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-09 16:07:18.915
Python: Python 3.10.20
2026-05-09 16:07:18.567
CUDA: 12.6.77
2026-05-09 16:07:18.562
=== Competition Environment Ready ===
2026-05-09 16:07:18.554
Complete setting network policy rules.
2026-05-09 16:07:15.153
Complete setting taiji user.
2026-05-09 16:07:15.097
Truncated to last 1000 lines.