PLATFORM_INFER_PROGRESS {"batches": 50, "elapsed_sec": 1481.171, "predictions": 106463, "rows_per_sec": 71.88}
2026-05-09 11:27:22.770
PLATFORM_FIRST_BATCH {"batch_predictions": 2192, "sample_scores": [0.00376677, 0.07527605, 0.03964277, 0.17658883, 0.44969112], "sample_user_ids": ["683974", "1334888", "51348", "1184407", "8854"], "sanitize_counts": {"nan_replaced": 0, "neginf_replaced": 0, "posinf_replaced": 0}}
2026-05-09 11:03:08.584
2026-05-09 03:02:41,951 - INFO - NumExpr defaulting to 16 threads.
2026-05-09 11:02:41.951
2026-05-09 03:02:41,950 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-09 11:02:41.951
2026-05-09 03:02:41,950 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-09 11:02:41.951
PLATFORM_MODEL_READY {"device": "cuda", "num_ns": 8, "num_parameters": 239931649}
2026-05-09 11:02:41.598
2026-05-09 03:02:39,269 - INFO - emb_skip_threshold=1000000: seq_c skipped 3/11 features
2026-05-09 11:02:39.269
2026-05-09 03:02:39,269 - INFO - emb_skip_threshold=1000000: seq_b skipped 1/13 features
2026-05-09 11:02:39.269
2026-05-09 03:02:37,206 - INFO - RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0
2026-05-09 11:02:37.207
2026-05-09 03:02:37,188 - INFO - RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1
2026-05-09 11:02:37.188
PLATFORM_DATASET_CONFIG {"batch_size": 2048, "item_dense_dim": 0, "item_int_features": 14, "num_row_groups": 1000, "num_rows_estimate": 310000, "seq_domains": ["seq_a", "seq_b", "seq_c", "seq_d"], "seq_max_lens": {"seq_a": 256, "seq_b": 256, "seq_c": 512, "seq_d": 512}, "user_dense_dim": 918, "user_int_features": 46}
2026-05-09 11:02:37.164
2026-05-09 03:02:37,163 - INFO - PCVRParquetDataset: 310000 rows from 1000 file(s), batch_size=2048, buffer_batches=0, shuffle=False
2026-05-09 11:02:37.163
PLATFORM_INFER_CONFIG {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "device": "cuda", "eval_data_dir": "/data_ams/academic_infer_data/", "model_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt", "num_workers": 0, "result_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/73801/results", "schema_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "train_config": {"d_model": 64, "emb_skip_threshold": 1000000, "eval_batch_size": 2048, "loss_type": "bce", "num_heads": 4, "num_hyformer_blocks": 2, "num_queries": 2, "seq_encoder_type": "transformer", "seq_max_lens": "seq_a:256,seq_b:256,seq_c:512,seq_d:512", "train_batch_size": 256}, "use_amp": false}
2026-05-09 11:02:36.667
2026-05-09 03:02:36,667 - INFO - Using schema path: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json
2026-05-09 11:02:36.667
2026-05-09 03:02:36,667 - INFO - Using checkpoint directory: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model
2026-05-09 11:02:36.667
PLATFORM_CHECKPOINT_STATUS {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "model_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt", "size_bytes": 959871995}, "schema_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "size_bytes": 5560}, "train_config_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/train_config.json", "size_bytes": 1638}}
2026-05-09 11:02:36.665
PLATFORM_INFER_START {"cuda_available": true, "cwd": "/workspace", "time": "2026-05-09 03:02:36", "torch_version": "2.7.1+cu126"}
2026-05-09 11:02:36.654
====== Inferring ======
2026-05-09 11:02:35.390
=================================
2026-05-09 11:02:35.013
Working Dir: /workspace
2026-05-09 11:02:35.013
Environment: competition
2026-05-09 11:02:35.013
GPU Count: 1
2026-05-09 11:02:35.013
GPU Available: True
2026-05-09 11:02:33.575
PyTorch: 2.7.1+cu126
2026-05-09 11:02:32.214
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-09 11:02:29.846
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-09 11:02:29.846
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-09 11:02:29.846
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-09 11:02:29.846
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-09 11:02:29.843
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-09 11:02:29.843
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-09 11:02:29.843
Python: Python 3.10.20
2026-05-09 11:02:29.502
CUDA: 12.6.77
2026-05-09 11:02:29.497
=== Competition Environment Ready ===
2026-05-09 11:02:29.488
Complete setting network policy rules.
2026-05-09 11:02:26.068
Complete setting taiji user.
2026-05-09 11:02:25.941
Truncated to last 1000 lines.