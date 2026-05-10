====== Score Finished ======
2026-05-09 17:48:46.873
====== Scoring ======
2026-05-09 17:48:44.148
=================================
2026-05-09 17:48:43.762
Working Dir: /workspace
2026-05-09 17:48:43.762
Environment: competition
2026-05-09 17:48:43.762
GPU Count: 1
2026-05-09 17:48:43.762
GPU Available: True
2026-05-09 17:48:42.340
PyTorch: 2.7.1+cu126
2026-05-09 17:48:40.988
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-09 17:48:38.806
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-09 17:48:38.806
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-09 17:48:38.806
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-09 17:48:38.806
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-09 17:48:38.803
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-09 17:48:38.803
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-09 17:48:38.803
Python: Python 3.10.20
2026-05-09 17:48:38.447
CUDA: 12.6.77
2026-05-09 17:48:38.442
=== Competition Environment Ready ===
2026-05-09 17:48:38.433
Complete setting network policy rules.
2026-05-09 17:48:34.987
Complete setting taiji user.
2026-05-09 17:48:34.918
====== Infer Finished ======
2026-05-09 17:44:35.641
2026-05-09 09:44:34,808 - INFO - Saved status.json to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75304/results/status.json
2026-05-09 17:44:34.808
PLATFORM_INFER_SUMMARY {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "elapsed_sec": 1114.191, "num_predictions": 310000, "num_unique_user_ids": 310000, "output_shape": "predictions_dict_user_id_to_float", "result_files": {"csv": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75304/results/submission.csv", "size_bytes": 5643992}, "json": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75304/results/predictions.json", "size_bytes": 7808678}, "scores": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75304/results/scores.npy", "size_bytes": 1240128}}, "result_json": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75304/results/predictions.json", "score_stats": {"count": 310000, "inf_count": 0, "max": 0.9765625, "mean": 0.11910518, "min": 0.00064468, "nan_count": 0, "p01": 0.0008049, "p50": 0.06005859, "p99": 0.81640625, "std": 0.15703748}, "scores_npy": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75304/results/scores.npy", "submission_csv": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75304/results/submission.csv"}
2026-05-09 17:44:34.383
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 1000, "elapsed_sec": 1113.442, "predictions": 310000, "recent_rows_per_sec": 477.15, "row_group_batches": 1000, "rows_per_sec": 278.42, "timing_avg_sec": {"forward": 0.1306, "merge": 0.0, "move": 0.0061, "post": 0.0002}}
2026-05-09 17:44:33.630
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 990, "elapsed_sec": 1106.846, "predictions": 306853, "recent_rows_per_sec": 333.69, "row_group_batches": 990, "rows_per_sec": 277.23, "timing_avg_sec": {"forward": 0.1306, "merge": 0.0, "move": 0.0062, "post": 0.0002}}
2026-05-09 17:44:27.035
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 980, "elapsed_sec": 1097.248, "predictions": 303650, "recent_rows_per_sec": 511.84, "row_group_batches": 980, "rows_per_sec": 276.74, "timing_avg_sec": {"forward": 0.1307, "merge": 0.0, "move": 0.0062, "post": 0.0002}}
2026-05-09 17:44:17.436
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 970, "elapsed_sec": 1090.949, "predictions": 300426, "recent_rows_per_sec": 396.32, "row_group_batches": 970, "rows_per_sec": 275.38, "timing_avg_sec": {"forward": 0.1311, "merge": 0.0, "move": 0.006, "post": 0.0002}}
2026-05-09 17:44:11.138
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 960, "elapsed_sec": 1082.751, "predictions": 297177, "recent_rows_per_sec": 366.68, "row_group_batches": 960, "rows_per_sec": 274.46, "timing_avg_sec": {"forward": 0.1312, "merge": 0.0, "move": 0.0061, "post": 0.0002}}
2026-05-09 17:44:02.940
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 950, "elapsed_sec": 1073.85, "predictions": 293913, "recent_rows_per_sec": 478.28, "row_group_batches": 950, "rows_per_sec": 273.7, "timing_avg_sec": {"forward": 0.1313, "merge": 0.0, "move": 0.0061, "post": 0.0002}}
2026-05-09 17:43:54.038
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 940, "elapsed_sec": 1067.144, "predictions": 290706, "recent_rows_per_sec": 435.45, "row_group_batches": 940, "rows_per_sec": 272.41, "timing_avg_sec": {"forward": 0.1313, "merge": 0.0, "move": 0.0062, "post": 0.0002}}
2026-05-09 17:43:47.333
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 930, "elapsed_sec": 1059.747, "predictions": 287485, "recent_rows_per_sec": 435.11, "row_group_batches": 930, "rows_per_sec": 271.28, "timing_avg_sec": {"forward": 0.1313, "merge": 0.0, "move": 0.0062, "post": 0.0002}}
2026-05-09 17:43:39.936
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 920, "elapsed_sec": 1052.248, "predictions": 284222, "recent_rows_per_sec": 502.03, "row_group_batches": 920, "rows_per_sec": 270.11, "timing_avg_sec": {"forward": 0.1313, "merge": 0.0, "move": 0.0062, "post": 0.0002}}
2026-05-09 17:43:32.437
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 910, "elapsed_sec": 1045.743, "predictions": 280956, "recent_rows_per_sec": 294.31, "row_group_batches": 910, "rows_per_sec": 268.67, "timing_avg_sec": {"forward": 0.1314, "merge": 0.0, "move": 0.0062, "post": 0.0002}}
2026-05-09 17:43:25.931
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 900, "elapsed_sec": 1034.741, "predictions": 277718, "recent_rows_per_sec": 587.7, "row_group_batches": 900, "rows_per_sec": 268.39, "timing_avg_sec": {"forward": 0.1313, "merge": 0.0, "move": 0.0062, "post": 0.0002}}
2026-05-09 17:43:14.929
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 890, "elapsed_sec": 1029.239, "predictions": 274485, "recent_rows_per_sec": 414.64, "row_group_batches": 890, "rows_per_sec": 266.69, "timing_avg_sec": {"forward": 0.1313, "merge": 0.0, "move": 0.0063, "post": 0.0002}}
2026-05-09 17:43:09.428
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 880, "elapsed_sec": 1021.442, "predictions": 271252, "recent_rows_per_sec": 388.24, "row_group_batches": 880, "rows_per_sec": 265.56, "timing_avg_sec": {"forward": 0.1311, "merge": 0.0, "move": 0.0064, "post": 0.0002}}
2026-05-09 17:43:01.631
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 870, "elapsed_sec": 1013.045, "predictions": 267992, "recent_rows_per_sec": 390.24, "row_group_batches": 870, "rows_per_sec": 264.54, "timing_avg_sec": {"forward": 0.1309, "merge": 0.0, "move": 0.0064, "post": 0.0002}}
2026-05-09 17:42:53.234
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 860, "elapsed_sec": 1004.545, "predictions": 264675, "recent_rows_per_sec": 305.63, "row_group_batches": 860, "rows_per_sec": 263.48, "timing_avg_sec": {"forward": 0.1308, "merge": 0.0, "move": 0.0064, "post": 0.0002}}
2026-05-09 17:42:44.734
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 850, "elapsed_sec": 993.748, "predictions": 261375, "recent_rows_per_sec": 457.45, "row_group_batches": 850, "rows_per_sec": 263.02, "timing_avg_sec": {"forward": 0.1306, "merge": 0.0, "move": 0.0065, "post": 0.0002}}
2026-05-09 17:42:33.937
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 840, "elapsed_sec": 986.639, "predictions": 258123, "recent_rows_per_sec": 346.82, "row_group_batches": 840, "rows_per_sec": 261.62, "timing_avg_sec": {"forward": 0.1306, "merge": 0.0, "move": 0.0064, "post": 0.0002}}
2026-05-09 17:42:26.828
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 830, "elapsed_sec": 977.349, "predictions": 254901, "recent_rows_per_sec": 503.94, "row_group_batches": 830, "rows_per_sec": 260.81, "timing_avg_sec": {"forward": 0.1305, "merge": 0.0, "move": 0.0065, "post": 0.0002}}
2026-05-09 17:42:17.537
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 820, "elapsed_sec": 970.945, "predictions": 251674, "recent_rows_per_sec": 334.2, "row_group_batches": 820, "rows_per_sec": 259.21, "timing_avg_sec": {"forward": 0.1306, "merge": 0.0, "move": 0.0064, "post": 0.0002}}
2026-05-09 17:42:11.134
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 810, "elapsed_sec": 961.25, "predictions": 248434, "recent_rows_per_sec": 521.09, "row_group_batches": 810, "rows_per_sec": 258.45, "timing_avg_sec": {"forward": 0.1307, "merge": 0.0, "move": 0.0062, "post": 0.0002}}
2026-05-09 17:42:01.439
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 800, "elapsed_sec": 955.046, "predictions": 245201, "recent_rows_per_sec": 371.91, "row_group_batches": 800, "rows_per_sec": 256.74, "timing_avg_sec": {"forward": 0.131, "merge": 0.0, "move": 0.0062, "post": 0.0002}}
2026-05-09 17:41:55.235
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 790, "elapsed_sec": 946.439, "predictions": 242000, "recent_rows_per_sec": 534.23, "row_group_batches": 790, "rows_per_sec": 255.7, "timing_avg_sec": {"forward": 0.1314, "merge": 0.0, "move": 0.006, "post": 0.0002}}
2026-05-09 17:41:46.628
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 780, "elapsed_sec": 940.348, "predictions": 238746, "recent_rows_per_sec": 577.15, "row_group_batches": 780, "rows_per_sec": 253.89, "timing_avg_sec": {"forward": 0.1313, "merge": 0.0, "move": 0.006, "post": 0.0002}}
2026-05-09 17:41:40.537
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 770, "elapsed_sec": 934.743, "predictions": 235511, "recent_rows_per_sec": 339.3, "row_group_batches": 770, "rows_per_sec": 251.95, "timing_avg_sec": {"forward": 0.1312, "merge": 0.0, "move": 0.0061, "post": 0.0002}}
2026-05-09 17:41:34.932
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 760, "elapsed_sec": 925.049, "predictions": 232222, "recent_rows_per_sec": 350.44, "row_group_batches": 760, "rows_per_sec": 251.04, "timing_avg_sec": {"forward": 0.1311, "merge": 0.0, "move": 0.0061, "post": 0.0002}}
2026-05-09 17:41:25.238
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 750, "elapsed_sec": 915.65, "predictions": 228928, "recent_rows_per_sec": 263.66, "row_group_batches": 750, "rows_per_sec": 250.02, "timing_avg_sec": {"forward": 0.1309, "merge": 0.0, "move": 0.0062, "post": 0.0002}}
2026-05-09 17:41:15.839
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 740, "elapsed_sec": 903.24, "predictions": 225656, "recent_rows_per_sec": 229.14, "row_group_batches": 740, "rows_per_sec": 249.83, "timing_avg_sec": {"forward": 0.1308, "merge": 0.0, "move": 0.0062, "post": 0.0002}}
2026-05-09 17:41:03.429
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 730, "elapsed_sec": 889.244, "predictions": 222449, "recent_rows_per_sec": 198.95, "row_group_batches": 730, "rows_per_sec": 250.16, "timing_avg_sec": {"forward": 0.1307, "merge": 0.0, "move": 0.0063, "post": 0.0002}}
2026-05-09 17:40:49.433
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 720, "elapsed_sec": 873.04, "predictions": 219225, "recent_rows_per_sec": 254.01, "row_group_batches": 720, "rows_per_sec": 251.11, "timing_avg_sec": {"forward": 0.1305, "merge": 0.0, "move": 0.0063, "post": 0.0002}}
2026-05-09 17:40:33.228
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 710, "elapsed_sec": 860.245, "predictions": 215975, "recent_rows_per_sec": 253.44, "row_group_batches": 710, "rows_per_sec": 251.06, "timing_avg_sec": {"forward": 0.1303, "merge": 0.0, "move": 0.0063, "post": 0.0002}}
2026-05-09 17:40:20.434
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 700, "elapsed_sec": 847.445, "predictions": 212731, "recent_rows_per_sec": 211.43, "row_group_batches": 700, "rows_per_sec": 251.03, "timing_avg_sec": {"forward": 0.1297, "merge": 0.0, "move": 0.0064, "post": 0.0002}}
2026-05-09 17:40:07.634
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 690, "elapsed_sec": 832.347, "predictions": 209539, "recent_rows_per_sec": 249.69, "row_group_batches": 690, "rows_per_sec": 251.74, "timing_avg_sec": {"forward": 0.1295, "merge": 0.0, "move": 0.0065, "post": 0.0002}}
2026-05-09 17:39:52.536
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 680, "elapsed_sec": 819.539, "predictions": 206341, "recent_rows_per_sec": 215.09, "row_group_batches": 680, "rows_per_sec": 251.78, "timing_avg_sec": {"forward": 0.1292, "merge": 0.0, "move": 0.0066, "post": 0.0002}}
2026-05-09 17:39:39.728
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 670, "elapsed_sec": 804.843, "predictions": 203180, "recent_rows_per_sec": 217.26, "row_group_batches": 670, "rows_per_sec": 252.45, "timing_avg_sec": {"forward": 0.1289, "merge": 0.0, "move": 0.0066, "post": 0.0002}}
2026-05-09 17:39:25.032
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 660, "elapsed_sec": 790.34, "predictions": 200029, "recent_rows_per_sec": 253.6, "row_group_batches": 660, "rows_per_sec": 253.09, "timing_avg_sec": {"forward": 0.1287, "merge": 0.0, "move": 0.0067, "post": 0.0002}}
2026-05-09 17:39:10.529
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 650, "elapsed_sec": 777.939, "predictions": 196884, "recent_rows_per_sec": 279.8, "row_group_batches": 650, "rows_per_sec": 253.08, "timing_avg_sec": {"forward": 0.1285, "merge": 0.0, "move": 0.0068, "post": 0.0002}}
2026-05-09 17:38:58.128
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 640, "elapsed_sec": 766.749, "predictions": 193753, "recent_rows_per_sec": 305.86, "row_group_batches": 640, "rows_per_sec": 252.69, "timing_avg_sec": {"forward": 0.1283, "merge": 0.0, "move": 0.0069, "post": 0.0002}}
2026-05-09 17:38:46.937
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 630, "elapsed_sec": 756.349, "predictions": 190572, "recent_rows_per_sec": 227.49, "row_group_batches": 630, "rows_per_sec": 251.96, "timing_avg_sec": {"forward": 0.1283, "merge": 0.0, "move": 0.0069, "post": 0.0002}}
2026-05-09 17:38:36.537
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 620, "elapsed_sec": 742.449, "predictions": 187410, "recent_rows_per_sec": 196.59, "row_group_batches": 620, "rows_per_sec": 252.42, "timing_avg_sec": {"forward": 0.1276, "merge": 0.0, "move": 0.007, "post": 0.0002}}
2026-05-09 17:38:22.638
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 610, "elapsed_sec": 726.548, "predictions": 184284, "recent_rows_per_sec": 220.78, "row_group_batches": 610, "rows_per_sec": 253.64, "timing_avg_sec": {"forward": 0.1276, "merge": 0.0, "move": 0.0071, "post": 0.0002}}
2026-05-09 17:38:06.737
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 600, "elapsed_sec": 712.543, "predictions": 181192, "recent_rows_per_sec": 254.37, "row_group_batches": 600, "rows_per_sec": 254.29, "timing_avg_sec": {"forward": 0.1271, "merge": 0.0, "move": 0.0072, "post": 0.0002}}
2026-05-09 17:37:52.732
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 590, "elapsed_sec": 700.34, "predictions": 178088, "recent_rows_per_sec": 228.29, "row_group_batches": 590, "rows_per_sec": 254.29, "timing_avg_sec": {"forward": 0.1272, "merge": 0.0, "move": 0.0072, "post": 0.0002}}
2026-05-09 17:37:40.529
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 580, "elapsed_sec": 686.643, "predictions": 174961, "recent_rows_per_sec": 186.53, "row_group_batches": 580, "rows_per_sec": 254.81, "timing_avg_sec": {"forward": 0.1271, "merge": 0.0, "move": 0.0071, "post": 0.0002}}
2026-05-09 17:37:26.832
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 570, "elapsed_sec": 670.141, "predictions": 171883, "recent_rows_per_sec": 230.95, "row_group_batches": 570, "rows_per_sec": 256.49, "timing_avg_sec": {"forward": 0.1273, "merge": 0.0, "move": 0.0072, "post": 0.0002}}
2026-05-09 17:37:10.330
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 560, "elapsed_sec": 657.044, "predictions": 168858, "recent_rows_per_sec": 197.49, "row_group_batches": 560, "rows_per_sec": 257.0, "timing_avg_sec": {"forward": 0.1269, "merge": 0.0, "move": 0.0074, "post": 0.0002}}
2026-05-09 17:36:57.232
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 550, "elapsed_sec": 641.838, "predictions": 165855, "recent_rows_per_sec": 206.61, "row_group_batches": 550, "rows_per_sec": 258.41, "timing_avg_sec": {"forward": 0.1265, "merge": 0.0, "move": 0.0075, "post": 0.0002}}
2026-05-09 17:36:42.027
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 540, "elapsed_sec": 627.245, "predictions": 162840, "recent_rows_per_sec": 202.03, "row_group_batches": 540, "rows_per_sec": 259.61, "timing_avg_sec": {"forward": 0.1265, "merge": 0.0, "move": 0.0074, "post": 0.0002}}
2026-05-09 17:36:27.433
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 530, "elapsed_sec": 612.049, "predictions": 159770, "recent_rows_per_sec": 199.24, "row_group_batches": 530, "rows_per_sec": 261.04, "timing_avg_sec": {"forward": 0.1264, "merge": 0.0, "move": 0.0074, "post": 0.0002}}
2026-05-09 17:36:12.237
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 520, "elapsed_sec": 596.349, "predictions": 156642, "recent_rows_per_sec": 210.26, "row_group_batches": 520, "rows_per_sec": 262.67, "timing_avg_sec": {"forward": 0.1267, "merge": 0.0, "move": 0.0073, "post": 0.0002}}
2026-05-09 17:35:56.538
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 510, "elapsed_sec": 581.743, "predictions": 153571, "recent_rows_per_sec": 249.06, "row_group_batches": 510, "rows_per_sec": 263.98, "timing_avg_sec": {"forward": 0.1271, "merge": 0.0, "move": 0.0071, "post": 0.0002}}
2026-05-09 17:35:41.932
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 500, "elapsed_sec": 569.348, "predictions": 150484, "recent_rows_per_sec": 226.52, "row_group_batches": 500, "rows_per_sec": 264.31, "timing_avg_sec": {"forward": 0.127, "merge": 0.0, "move": 0.007, "post": 0.0002}}
2026-05-09 17:35:29.537
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 490, "elapsed_sec": 555.822, "predictions": 147420, "recent_rows_per_sec": 204.54, "row_group_batches": 490, "rows_per_sec": 265.23, "timing_avg_sec": {"forward": 0.1265, "merge": 0.0, "move": 0.0071, "post": 0.0002}}
2026-05-09 17:35:16.011
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 480, "elapsed_sec": 540.847, "predictions": 144357, "recent_rows_per_sec": 266.38, "row_group_batches": 480, "rows_per_sec": 266.91, "timing_avg_sec": {"forward": 0.126, "merge": 0.0, "move": 0.0073, "post": 0.0002}}
2026-05-09 17:35:01.036
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 470, "elapsed_sec": 529.641, "predictions": 141372, "recent_rows_per_sec": 277.65, "row_group_batches": 470, "rows_per_sec": 266.92, "timing_avg_sec": {"forward": 0.1259, "merge": 0.0, "move": 0.0074, "post": 0.0002}}
2026-05-09 17:34:49.830
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 460, "elapsed_sec": 519.048, "predictions": 138431, "recent_rows_per_sec": 239.38, "row_group_batches": 460, "rows_per_sec": 266.7, "timing_avg_sec": {"forward": 0.1263, "merge": 0.0, "move": 0.0072, "post": 0.0002}}
2026-05-09 17:34:39.237
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 450, "elapsed_sec": 506.741, "predictions": 135485, "recent_rows_per_sec": 265.33, "row_group_batches": 450, "rows_per_sec": 267.37, "timing_avg_sec": {"forward": 0.1261, "merge": 0.0, "move": 0.0073, "post": 0.0002}}
2026-05-09 17:34:26.930
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 440, "elapsed_sec": 495.54, "predictions": 132513, "recent_rows_per_sec": 357.08, "row_group_batches": 440, "rows_per_sec": 267.41, "timing_avg_sec": {"forward": 0.1258, "merge": 0.0, "move": 0.0075, "post": 0.0002}}
2026-05-09 17:34:15.729
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 430, "elapsed_sec": 487.144, "predictions": 129515, "recent_rows_per_sec": 290.65, "row_group_batches": 430, "rows_per_sec": 265.87, "timing_avg_sec": {"forward": 0.1259, "merge": 0.0, "move": 0.0076, "post": 0.0002}}
2026-05-09 17:34:07.333
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 420, "elapsed_sec": 476.843, "predictions": 126521, "recent_rows_per_sec": 215.47, "row_group_batches": 420, "rows_per_sec": 265.33, "timing_avg_sec": {"forward": 0.1258, "merge": 0.0, "move": 0.0078, "post": 0.0002}}
2026-05-09 17:33:57.032
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 410, "elapsed_sec": 462.948, "predictions": 123527, "recent_rows_per_sec": 218.44, "row_group_batches": 410, "rows_per_sec": 266.83, "timing_avg_sec": {"forward": 0.1263, "merge": 0.0, "move": 0.0077, "post": 0.0002}}
2026-05-09 17:33:43.137
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 400, "elapsed_sec": 449.343, "predictions": 120555, "recent_rows_per_sec": 163.42, "row_group_batches": 400, "rows_per_sec": 268.29, "timing_avg_sec": {"forward": 0.1259, "merge": 0.0, "move": 0.0079, "post": 0.0002}}
2026-05-09 17:33:29.531
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 390, "elapsed_sec": 431.04, "predictions": 117564, "recent_rows_per_sec": 180.26, "row_group_batches": 390, "rows_per_sec": 272.74, "timing_avg_sec": {"forward": 0.1257, "merge": 0.0, "move": 0.0076, "post": 0.0002}}
2026-05-09 17:33:11.229
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 380, "elapsed_sec": 414.736, "predictions": 114625, "recent_rows_per_sec": 197.61, "row_group_batches": 380, "rows_per_sec": 276.38, "timing_avg_sec": {"forward": 0.125, "merge": 0.0, "move": 0.0078, "post": 0.0002}}
2026-05-09 17:32:54.925
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 370, "elapsed_sec": 400.051, "predictions": 111723, "recent_rows_per_sec": 200.01, "row_group_batches": 370, "rows_per_sec": 279.27, "timing_avg_sec": {"forward": 0.1243, "merge": 0.0, "move": 0.008, "post": 0.0002}}
2026-05-09 17:32:40.240
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 360, "elapsed_sec": 385.442, "predictions": 108801, "recent_rows_per_sec": 184.8, "row_group_batches": 360, "rows_per_sec": 282.28, "timing_avg_sec": {"forward": 0.124, "merge": 0.0, "move": 0.0082, "post": 0.0002}}
2026-05-09 17:32:25.630
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 350, "elapsed_sec": 369.646, "predictions": 105882, "recent_rows_per_sec": 183.81, "row_group_batches": 350, "rows_per_sec": 286.44, "timing_avg_sec": {"forward": 0.1237, "merge": 0.0, "move": 0.0085, "post": 0.0002}}
2026-05-09 17:32:09.835
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 340, "elapsed_sec": 353.841, "predictions": 102977, "recent_rows_per_sec": 198.81, "row_group_batches": 340, "rows_per_sec": 291.03, "timing_avg_sec": {"forward": 0.1227, "merge": 0.0, "move": 0.0087, "post": 0.0002}}
2026-05-09 17:31:54.030
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 330, "elapsed_sec": 339.044, "predictions": 100035, "recent_rows_per_sec": 192.43, "row_group_batches": 330, "rows_per_sec": 295.05, "timing_avg_sec": {"forward": 0.1218, "merge": 0.0, "move": 0.0089, "post": 0.0002}}
2026-05-09 17:31:39.232
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 320, "elapsed_sec": 323.614, "predictions": 97066, "recent_rows_per_sec": 201.04, "row_group_batches": 320, "rows_per_sec": 299.94, "timing_avg_sec": {"forward": 0.1214, "merge": 0.0, "move": 0.0092, "post": 0.0002}}
2026-05-09 17:31:23.803
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 310, "elapsed_sec": 308.642, "predictions": 94056, "recent_rows_per_sec": 194.76, "row_group_batches": 310, "rows_per_sec": 304.74, "timing_avg_sec": {"forward": 0.1208, "merge": 0.0, "move": 0.0092, "post": 0.0002}}
2026-05-09 17:31:08.831
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 300, "elapsed_sec": 292.946, "predictions": 90999, "recent_rows_per_sec": 222.25, "row_group_batches": 300, "rows_per_sec": 310.63, "timing_avg_sec": {"forward": 0.1207, "merge": 0.0, "move": 0.0095, "post": 0.0002}}
2026-05-09 17:30:53.135
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 290, "elapsed_sec": 279.047, "predictions": 87910, "recent_rows_per_sec": 205.84, "row_group_batches": 290, "rows_per_sec": 315.04, "timing_avg_sec": {"forward": 0.1204, "merge": 0.0, "move": 0.0095, "post": 0.0002}}
2026-05-09 17:30:39.236
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 280, "elapsed_sec": 264.04, "predictions": 84821, "recent_rows_per_sec": 178.73, "row_group_batches": 280, "rows_per_sec": 321.24, "timing_avg_sec": {"forward": 0.1198, "merge": 0.0, "move": 0.0091, "post": 0.0002}}
2026-05-09 17:30:24.229
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 270, "elapsed_sec": 246.942, "predictions": 81765, "recent_rows_per_sec": 245.24, "row_group_batches": 270, "rows_per_sec": 331.11, "timing_avg_sec": {"forward": 0.12, "merge": 0.0, "move": 0.0087, "post": 0.0002}}
2026-05-09 17:30:07.130
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 260, "elapsed_sec": 234.448, "predictions": 78701, "recent_rows_per_sec": 306.62, "row_group_batches": 260, "rows_per_sec": 335.69, "timing_avg_sec": {"forward": 0.1196, "merge": 0.0, "move": 0.0083, "post": 0.0002}}
2026-05-09 17:29:54.637
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 250, "elapsed_sec": 224.546, "predictions": 75665, "recent_rows_per_sec": 215.63, "row_group_batches": 250, "rows_per_sec": 336.97, "timing_avg_sec": {"forward": 0.1202, "merge": 0.0, "move": 0.0078, "post": 0.0002}}
2026-05-09 17:29:44.735
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 240, "elapsed_sec": 210.541, "predictions": 72645, "recent_rows_per_sec": 195.25, "row_group_batches": 240, "rows_per_sec": 345.04, "timing_avg_sec": {"forward": 0.1204, "merge": 0.0, "move": 0.0081, "post": 0.0002}}
2026-05-09 17:29:30.730
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 230, "elapsed_sec": 195.038, "predictions": 69618, "recent_rows_per_sec": 356.44, "row_group_batches": 230, "rows_per_sec": 356.95, "timing_avg_sec": {"forward": 0.1202, "merge": 0.0, "move": 0.008, "post": 0.0002}}
2026-05-09 17:29:15.227
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 220, "elapsed_sec": 186.647, "predictions": 66627, "recent_rows_per_sec": 270.34, "row_group_batches": 220, "rows_per_sec": 356.97, "timing_avg_sec": {"forward": 0.1199, "merge": 0.0, "move": 0.0079, "post": 0.0002}}
2026-05-09 17:29:06.835
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 210, "elapsed_sec": 175.742, "predictions": 63679, "recent_rows_per_sec": 256.11, "row_group_batches": 210, "rows_per_sec": 362.34, "timing_avg_sec": {"forward": 0.1207, "merge": 0.0, "move": 0.0083, "post": 0.0002}}
2026-05-09 17:28:55.931
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 200, "elapsed_sec": 164.243, "predictions": 60734, "recent_rows_per_sec": 399.97, "row_group_batches": 200, "rows_per_sec": 369.78, "timing_avg_sec": {"forward": 0.1203, "merge": 0.0, "move": 0.0077, "post": 0.0002}}
2026-05-09 17:28:44.432
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 190, "elapsed_sec": 156.847, "predictions": 57776, "recent_rows_per_sec": 284.36, "row_group_batches": 190, "rows_per_sec": 368.36, "timing_avg_sec": {"forward": 0.122, "merge": 0.0, "move": 0.0076, "post": 0.0002}}
2026-05-09 17:28:37.036
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 180, "elapsed_sec": 146.441, "predictions": 54817, "recent_rows_per_sec": 324.89, "row_group_batches": 180, "rows_per_sec": 374.33, "timing_avg_sec": {"forward": 0.1215, "merge": 0.0, "move": 0.008, "post": 0.0002}}
2026-05-09 17:28:26.630
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 170, "elapsed_sec": 137.346, "predictions": 51862, "recent_rows_per_sec": 405.41, "row_group_batches": 170, "rows_per_sec": 377.6, "timing_avg_sec": {"forward": 0.1208, "merge": 0.0, "move": 0.0084, "post": 0.0002}}
2026-05-09 17:28:17.535
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 160, "elapsed_sec": 130.042, "predictions": 48901, "recent_rows_per_sec": 454.66, "row_group_batches": 160, "rows_per_sec": 376.04, "timing_avg_sec": {"forward": 0.1211, "merge": 0.0, "move": 0.0089, "post": 0.0002}}
2026-05-09 17:28:10.231
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 150, "elapsed_sec": 123.442, "predictions": 45900, "recent_rows_per_sec": 353.48, "row_group_batches": 150, "rows_per_sec": 371.84, "timing_avg_sec": {"forward": 0.1209, "merge": 0.0, "move": 0.0089, "post": 0.0002}}
2026-05-09 17:28:03.631
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 140, "elapsed_sec": 114.847, "predictions": 42862, "recent_rows_per_sec": 341.46, "row_group_batches": 140, "rows_per_sec": 373.21, "timing_avg_sec": {"forward": 0.1201, "merge": 0.0, "move": 0.0095, "post": 0.0002}}
2026-05-09 17:27:55.036
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 130, "elapsed_sec": 106.047, "predictions": 39857, "recent_rows_per_sec": 509.69, "row_group_batches": 130, "rows_per_sec": 375.84, "timing_avg_sec": {"forward": 0.1203, "merge": 0.0, "move": 0.0087, "post": 0.0002}}
2026-05-09 17:27:46.236
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 120, "elapsed_sec": 100.147, "predictions": 36850, "recent_rows_per_sec": 353.01, "row_group_batches": 120, "rows_per_sec": 367.96, "timing_avg_sec": {"forward": 0.1226, "merge": 0.0, "move": 0.0085, "post": 0.0002}}
2026-05-09 17:27:40.336
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 110, "elapsed_sec": 91.547, "predictions": 33814, "recent_rows_per_sec": 396.88, "row_group_batches": 110, "rows_per_sec": 369.36, "timing_avg_sec": {"forward": 0.125, "merge": 0.0, "move": 0.0075, "post": 0.0002}}
2026-05-09 17:27:31.736
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 100, "elapsed_sec": 83.847, "predictions": 30758, "recent_rows_per_sec": 361.97, "row_group_batches": 100, "rows_per_sec": 366.83, "timing_avg_sec": {"forward": 0.1305, "merge": 0.0, "move": 0.0042, "post": 0.0002}}
2026-05-09 17:27:24.036
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 90, "elapsed_sec": 75.437, "predictions": 27714, "recent_rows_per_sec": 484.6, "row_group_batches": 90, "rows_per_sec": 367.38, "timing_avg_sec": {"forward": 0.1312, "merge": 0.0, "move": 0.0047, "post": 0.0002}}
2026-05-09 17:27:15.626
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 80, "elapsed_sec": 69.137, "predictions": 24661, "recent_rows_per_sec": 406.96, "row_group_batches": 80, "rows_per_sec": 356.7, "timing_avg_sec": {"forward": 0.13, "merge": 0.0, "move": 0.0052, "post": 0.0002}}
2026-05-09 17:27:09.326
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 70, "elapsed_sec": 61.64, "predictions": 21610, "recent_rows_per_sec": 401.39, "row_group_batches": 70, "rows_per_sec": 350.58, "timing_avg_sec": {"forward": 0.1309, "merge": 0.0, "move": 0.0045, "post": 0.0002}}
2026-05-09 17:27:01.829
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 60, "elapsed_sec": 54.039, "predictions": 18559, "recent_rows_per_sec": 578.03, "row_group_batches": 60, "rows_per_sec": 343.44, "timing_avg_sec": {"forward": 0.1307, "merge": 0.0, "move": 0.0052, "post": 0.0002}}
2026-05-09 17:26:54.228
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 50, "elapsed_sec": 48.742, "predictions": 15497, "recent_rows_per_sec": 355.16, "row_group_batches": 50, "rows_per_sec": 317.94, "timing_avg_sec": {"forward": 0.1321, "merge": 0.0, "move": 0.0062, "post": 0.0002}}
2026-05-09 17:26:48.931
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 40, "elapsed_sec": 40.045, "predictions": 12408, "recent_rows_per_sec": 300.68, "row_group_batches": 40, "rows_per_sec": 309.86, "timing_avg_sec": {"forward": 0.1298, "merge": 0.0, "move": 0.0077, "post": 0.0002}}
2026-05-09 17:26:40.233
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 30, "elapsed_sec": 29.748, "predictions": 9312, "recent_rows_per_sec": 327.01, "row_group_batches": 30, "rows_per_sec": 313.03, "timing_avg_sec": {"forward": 0.1345, "merge": 0.0, "move": 0.0036, "post": 0.0002}}
2026-05-09 17:26:29.937
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 20, "elapsed_sec": 20.345, "predictions": 6237, "recent_rows_per_sec": 455.64, "row_group_batches": 20, "rows_per_sec": 306.57, "timing_avg_sec": {"forward": 0.1378, "merge": 0.0, "move": 0.0004, "post": 0.0002}}
2026-05-09 17:26:20.533
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 10, "elapsed_sec": 13.543, "predictions": 3138, "recent_rows_per_sec": 231.7, "row_group_batches": 10, "rows_per_sec": 231.7, "timing_avg_sec": {"forward": 0.1457, "merge": 0.0, "move": 0.0005, "post": 0.0002}}
2026-05-09 17:26:13.732
PLATFORM_FIRST_BATCH {"batch_predictions": 314, "rows_in_forward": 314, "sample_scores": [0.00370789, 0.07470703, 0.03955078, 0.17675781, 0.44921875], "sample_user_ids": ["683974", "1334888", "51348", "1184407", "8854"], "sanitize_counts": {"nan_replaced": 0, "neginf_replaced": 0, "posinf_replaced": 0}, "timing_sec": {"forward": 0.4113, "merge": 0.0, "move": 0.0026, "post": 0.0002}}
2026-05-09 17:26:03.509
2026-05-09 09:26:00,585 - INFO - NumExpr defaulting to 16 threads.
2026-05-09 17:26:00.586
2026-05-09 09:26:00,585 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-09 17:26:00.586
2026-05-09 09:26:00,585 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-09 17:26:00.585
PLATFORM_AMP_CONFIG {"cuda_bf16_supported": true, "enabled": true, "mode": "bf16", "requested": "bf16"}
2026-05-09 17:26:00.189
PLATFORM_MODEL_READY {"device": "cuda", "num_ns": 8, "num_parameters": 239931649}
2026-05-09 17:26:00.189
2026-05-09 09:25:58,534 - INFO - emb_skip_threshold=1000000: seq_c skipped 3/11 features
2026-05-09 17:25:58.535
2026-05-09 09:25:58,534 - INFO - emb_skip_threshold=1000000: seq_b skipped 1/13 features
2026-05-09 17:25:58.534
2026-05-09 09:25:56,465 - INFO - RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0
2026-05-09 17:25:56.466
2026-05-09 09:25:56,447 - INFO - RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1
2026-05-09 17:25:56.448
PLATFORM_DATASET_CONFIG {"batch_size": 2048, "item_dense_dim": 0, "item_int_features": 14, "num_row_groups": 1000, "num_rows_estimate": 310000, "seq_domains": ["seq_a", "seq_b", "seq_c", "seq_d"], "seq_max_lens": {"seq_a": 256, "seq_b": 256, "seq_c": 512, "seq_d": 512}, "user_dense_dim": 918, "user_int_features": 46}
2026-05-09 17:25:56.422
2026-05-09 09:25:56,422 - INFO - PCVRParquetDataset: 310000 rows from 1000 file(s), batch_size=2048, buffer_batches=0, shuffle=False
2026-05-09 17:25:56.422
PLATFORM_INFER_CONFIG {"amp_dtype": "bf16", "checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "device": "cuda", "eval_data_dir": "/data_ams/academic_infer_data/", "merge_batches": false, "model_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt", "num_workers": 0, "progress_every": 10, "result_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75304/results", "schema_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "train_config": {"d_model": 64, "emb_skip_threshold": 1000000, "eval_batch_size": 2048, "loss_type": "bce", "num_heads": 4, "num_hyformer_blocks": 2, "num_queries": 2, "seq_encoder_type": "transformer", "seq_max_lens": "seq_a:256,seq_b:256,seq_c:512,seq_d:512", "train_batch_size": 256}, "use_amp": true}
2026-05-09 17:25:55.942
2026-05-09 09:25:55,942 - INFO - Using schema path: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json
2026-05-09 17:25:55.942
2026-05-09 09:25:55,942 - INFO - Using checkpoint directory: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model
2026-05-09 17:25:55.942
PLATFORM_CHECKPOINT_STATUS {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "model_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt", "size_bytes": 959871995}, "schema_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "size_bytes": 5560}, "train_config_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/train_config.json", "size_bytes": 1638}}
2026-05-09 17:25:55.940
PLATFORM_INFER_START {"cuda_available": true, "cwd": "/workspace", "time": "2026-05-09 09:25:55", "torch_version": "2.7.1+cu126"}
2026-05-09 17:25:55.932
====== Inferring ======
2026-05-09 17:25:54.670
=================================
2026-05-09 17:25:54.281
Working Dir: /workspace
2026-05-09 17:25:54.281
Environment: competition
2026-05-09 17:25:54.281
GPU Count: 1
2026-05-09 17:25:54.281
GPU Available: True
2026-05-09 17:25:52.746
PyTorch: 2.7.1+cu126
2026-05-09 17:25:51.306
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-09 17:25:48.823
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-09 17:25:48.823
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-09 17:25:48.823
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-09 17:25:48.823
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-09 17:25:48.820
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-09 17:25:48.820
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-09 17:25:48.820
Python: Python 3.10.20
2026-05-09 17:25:48.473
CUDA: 12.6.77
2026-05-09 17:25:48.468
=== Competition Environment Ready ===
2026-05-09 17:25:48.459
Complete setting network policy rules.
2026-05-09 17:25:44.774
Complete setting taiji user.
2026-05-09 17:25:44.675
Truncated to last 1000 lines.