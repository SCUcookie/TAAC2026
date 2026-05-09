PLATFORM_INFER_PROGRESS {"batches": 150, "elapsed_sec": 1738.428, "predictions": 181192, "rows_per_sec": 104.23}
2026-05-08 19:27:38.326
PLATFORM_INFER_PROGRESS {"batches": 100, "elapsed_sec": 949.173, "predictions": 120555, "rows_per_sec": 127.01}
2026-05-08 19:14:29.071
PLATFORM_INFER_PROGRESS {"batches": 50, "elapsed_sec": 351.278, "predictions": 60734, "rows_per_sec": 172.89}
2026-05-08 19:04:31.177
PLATFORM_FIRST_BATCH {"batch_predictions": 1249, "sample_scores": [0.00376677, 0.0752761, 0.03964277, 0.17658883, 0.44969115], "sample_user_ids": ["683974", "1334888", "51348", "1184407", "8854"], "sanitize_counts": {"nan_replaced": 0, "neginf_replaced": 0, "posinf_replaced": 0}}
2026-05-08 18:58:51.457
2026-05-08 10:58:40,261 - INFO - NumExpr defaulting to 16 threads.
2026-05-08 18:58:40.261
2026-05-08 10:58:40,261 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-08 18:58:40.261
2026-05-08 10:58:40,261 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-08 18:58:40.261
PLATFORM_MODEL_READY {"device": "cuda", "num_ns": 8, "num_parameters": 239931649}
2026-05-08 18:58:39.898
2026-05-08 10:58:33,342 - INFO - emb_skip_threshold=1000000: seq_c skipped 3/11 features
2026-05-08 18:58:33.342
2026-05-08 10:58:33,341 - INFO - emb_skip_threshold=1000000: seq_b skipped 1/13 features
2026-05-08 18:58:33.342
2026-05-08 10:58:31,327 - INFO - RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0
2026-05-08 18:58:31.327
2026-05-08 10:58:31,310 - INFO - RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1
2026-05-08 18:58:31.310
PLATFORM_DATASET_CONFIG {"batch_size": 1024, "item_dense_dim": 0, "item_int_features": 14, "num_row_groups": 1000, "num_rows_estimate": 310000, "seq_domains": ["seq_a", "seq_b", "seq_c", "seq_d"], "seq_max_lens": {"seq_a": 256, "seq_b": 256, "seq_c": 512, "seq_d": 512}, "user_dense_dim": 918, "user_int_features": 46}
2026-05-08 18:58:31.287
2026-05-08 10:58:31,286 - INFO - PCVRParquetDataset: 310000 rows from 1000 file(s), batch_size=1024, buffer_batches=0, shuffle=False
2026-05-08 18:58:31.286
PLATFORM_INFER_CONFIG {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "device": "cuda", "eval_data_dir": "/data_ams/academic_infer_data/", "model_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt", "num_workers": 0, "result_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/71570/results", "schema_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "train_config": {"d_model": 64, "emb_skip_threshold": 1000000, "eval_batch_size": 1024, "loss_type": "bce", "num_heads": 4, "num_hyformer_blocks": 2, "num_queries": 2, "seq_encoder_type": "transformer", "seq_max_lens": "seq_a:256,seq_b:256,seq_c:512,seq_d:512", "train_batch_size": 256}, "use_amp": false}
2026-05-08 18:58:30.802
2026-05-08 10:58:30,802 - INFO - Using schema path: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json
2026-05-08 18:58:30.802
2026-05-08 10:58:30,802 - INFO - Using checkpoint directory: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model
2026-05-08 18:58:30.802
PLATFORM_CHECKPOINT_STATUS {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "model_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt", "size_bytes": 959871995}, "schema_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "size_bytes": 5560}, "train_config_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/train_config.json", "size_bytes": 1638}}
2026-05-08 18:58:30.800
PLATFORM_INFER_START {"cuda_available": true, "cwd": "/workspace", "time": "2026-05-08 10:58:30", "torch_version": "2.7.1+cu126"}
2026-05-08 18:58:30.793
====== Inferring ======
2026-05-08 18:58:29.616
=================================
2026-05-08 18:58:29.238
Working Dir: /workspace
2026-05-08 18:58:29.238
Environment: competition
2026-05-08 18:58:29.238
GPU Count: 1
2026-05-08 18:58:29.238
GPU Available: True
2026-05-08 18:58:27.904
PyTorch: 2.7.1+cu126
2026-05-08 18:58:26.301
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-08 18:58:23.881
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-08 18:58:23.881
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-08 18:58:23.881
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-08 18:58:23.881
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-08 18:58:23.878
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-08 18:58:23.878
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-08 18:58:23.878
Python: Python 3.10.20
2026-05-08 18:58:23.380
CUDA: 12.6.77
2026-05-08 18:58:23.373
=== Competition Environment Ready ===
2026-05-08 18:58:23.365
Complete setting network policy rules.
2026-05-08 18:58:19.840
Complete setting taiji user.
2026-05-08 18:58:19.500
Truncated to last 1000 lines.