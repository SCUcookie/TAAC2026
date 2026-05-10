====== Score Finished ======
2026-05-09 19:30:41.162
====== Scoring ======
2026-05-09 19:30:38.432
=================================
2026-05-09 19:30:38.042
Working Dir: /workspace
2026-05-09 19:30:38.042
Environment: competition
2026-05-09 19:30:38.042
GPU Count: 1
2026-05-09 19:30:38.041
GPU Available: True
2026-05-09 19:30:36.665
PyTorch: 2.7.1+cu126
2026-05-09 19:30:35.348
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-09 19:30:33.072
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-09 19:30:33.072
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-09 19:30:33.072
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-09 19:30:33.072
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-09 19:30:33.069
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-09 19:30:33.069
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-09 19:30:33.069
Python: Python 3.10.20
2026-05-09 19:30:32.696
CUDA: 12.6.77
2026-05-09 19:30:32.691
=== Competition Environment Ready ===
2026-05-09 19:30:32.676
Complete setting network policy rules.
2026-05-09 19:30:29.193
Complete setting taiji user.
2026-05-09 19:30:29.106
====== Infer Finished ======
2026-05-09 19:26:29.839
2026-05-09 11:26:28,965 - INFO - Saved status.json to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75522/results/status.json
2026-05-09 19:26:28.966
PLATFORM_INFER_SUMMARY {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "elapsed_sec": 1283.838, "num_predictions": 310000, "num_unique_user_ids": 310000, "output_shape": "predictions_dict_user_id_to_float", "result_files": {"csv": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75522/results/submission.csv", "size_bytes": 5643992}, "json": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75522/results/predictions.json", "size_bytes": 7808678}, "scores": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75522/results/scores.npy", "size_bytes": 1240128}}, "result_json": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75522/results/predictions.json", "score_stats": {"count": 310000, "inf_count": 0, "max": 0.9765625, "mean": 0.11910518, "min": 0.00064468, "nan_count": 0, "p01": 0.0008049, "p50": 0.06005859, "p99": 0.81640625, "std": 0.15703748}, "scores_npy": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75522/results/scores.npy", "submission_csv": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75522/results/submission.csv"}
2026-05-09 19:26:28.567
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 1000, "elapsed_sec": 1282.936, "predictions": 310000, "recent_rows_per_sec": 302.86, "row_group_batches": 1000, "rows_per_sec": 241.63, "timing_avg_sec": {"forward": 0.2488, "merge": 0.0001, "move": 0.0069, "post": 0.0002}}
2026-05-09 19:26:27.663
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 990, "elapsed_sec": 1272.545, "predictions": 306853, "recent_rows_per_sec": 310.8, "row_group_batches": 990, "rows_per_sec": 241.13, "timing_avg_sec": {"forward": 0.2487, "merge": 0.0001, "move": 0.0068, "post": 0.0002}}
2026-05-09 19:26:17.273
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 980, "elapsed_sec": 1262.239, "predictions": 303650, "recent_rows_per_sec": 282.91, "row_group_batches": 980, "rows_per_sec": 240.56, "timing_avg_sec": {"forward": 0.2488, "merge": 0.0001, "move": 0.0067, "post": 0.0002}}
2026-05-09 19:26:06.967
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 970, "elapsed_sec": 1250.843, "predictions": 300426, "recent_rows_per_sec": 369.41, "row_group_batches": 970, "rows_per_sec": 240.18, "timing_avg_sec": {"forward": 0.2486, "merge": 0.0001, "move": 0.0068, "post": 0.0002}}
2026-05-09 19:25:55.571
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 960, "elapsed_sec": 1242.048, "predictions": 297177, "recent_rows_per_sec": 510.0, "row_group_batches": 960, "rows_per_sec": 239.26, "timing_avg_sec": {"forward": 0.2487, "merge": 0.0001, "move": 0.0069, "post": 0.0002}}
2026-05-09 19:25:46.776
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 950, "elapsed_sec": 1235.648, "predictions": 293913, "recent_rows_per_sec": 298.65, "row_group_batches": 950, "rows_per_sec": 237.86, "timing_avg_sec": {"forward": 0.2487, "merge": 0.0001, "move": 0.0069, "post": 0.0002}}
2026-05-09 19:25:40.376
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 940, "elapsed_sec": 1224.91, "predictions": 290706, "recent_rows_per_sec": 301.91, "row_group_batches": 940, "rows_per_sec": 237.33, "timing_avg_sec": {"forward": 0.2483, "merge": 0.0001, "move": 0.0069, "post": 0.0002}}
2026-05-09 19:25:29.638
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 930, "elapsed_sec": 1214.241, "predictions": 287485, "recent_rows_per_sec": 347.29, "row_group_batches": 930, "rows_per_sec": 236.76, "timing_avg_sec": {"forward": 0.2485, "merge": 0.0001, "move": 0.0069, "post": 0.0002}}
2026-05-09 19:25:18.969
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 920, "elapsed_sec": 1204.846, "predictions": 284222, "recent_rows_per_sec": 287.54, "row_group_batches": 920, "rows_per_sec": 235.9, "timing_avg_sec": {"forward": 0.2486, "merge": 0.0001, "move": 0.0067, "post": 0.0002}}
2026-05-09 19:25:09.573
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 910, "elapsed_sec": 1193.487, "predictions": 280956, "recent_rows_per_sec": 263.79, "row_group_batches": 910, "rows_per_sec": 235.41, "timing_avg_sec": {"forward": 0.2486, "merge": 0.0001, "move": 0.0068, "post": 0.0002}}
2026-05-09 19:24:58.215
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 900, "elapsed_sec": 1181.212, "predictions": 277718, "recent_rows_per_sec": 241.47, "row_group_batches": 900, "rows_per_sec": 235.11, "timing_avg_sec": {"forward": 0.2488, "merge": 0.0001, "move": 0.0069, "post": 0.0002}}
2026-05-09 19:24:45.940
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 890, "elapsed_sec": 1167.824, "predictions": 274485, "recent_rows_per_sec": 294.02, "row_group_batches": 890, "rows_per_sec": 235.04, "timing_avg_sec": {"forward": 0.2488, "merge": 0.0001, "move": 0.0068, "post": 0.0002}}
2026-05-09 19:24:32.551
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 880, "elapsed_sec": 1156.828, "predictions": 271252, "recent_rows_per_sec": 275.92, "row_group_batches": 880, "rows_per_sec": 234.48, "timing_avg_sec": {"forward": 0.2491, "merge": 0.0001, "move": 0.0068, "post": 0.0002}}
2026-05-09 19:24:21.555
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 870, "elapsed_sec": 1145.013, "predictions": 267992, "recent_rows_per_sec": 276.01, "row_group_batches": 870, "rows_per_sec": 234.05, "timing_avg_sec": {"forward": 0.2495, "merge": 0.0001, "move": 0.0065, "post": 0.0002}}
2026-05-09 19:24:09.740
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 860, "elapsed_sec": 1132.995, "predictions": 264675, "recent_rows_per_sec": 315.13, "row_group_batches": 860, "rows_per_sec": 233.61, "timing_avg_sec": {"forward": 0.2494, "merge": 0.0001, "move": 0.0066, "post": 0.0002}}
2026-05-09 19:23:57.723
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 850, "elapsed_sec": 1122.523, "predictions": 261375, "recent_rows_per_sec": 369.8, "row_group_batches": 850, "rows_per_sec": 232.85, "timing_avg_sec": {"forward": 0.2496, "merge": 0.0001, "move": 0.0066, "post": 0.0002}}
2026-05-09 19:23:47.251
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 840, "elapsed_sec": 1113.729, "predictions": 258123, "recent_rows_per_sec": 231.6, "row_group_batches": 840, "rows_per_sec": 231.76, "timing_avg_sec": {"forward": 0.2495, "merge": 0.0001, "move": 0.0065, "post": 0.0002}}
2026-05-09 19:23:38.457
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 830, "elapsed_sec": 1099.817, "predictions": 254901, "recent_rows_per_sec": 319.15, "row_group_batches": 830, "rows_per_sec": 231.77, "timing_avg_sec": {"forward": 0.2496, "merge": 0.0001, "move": 0.0066, "post": 0.0002}}
2026-05-09 19:23:24.545
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 820, "elapsed_sec": 1089.706, "predictions": 251674, "recent_rows_per_sec": 251.73, "row_group_batches": 820, "rows_per_sec": 230.96, "timing_avg_sec": {"forward": 0.2496, "merge": 0.0001, "move": 0.0066, "post": 0.0002}}
2026-05-09 19:23:14.433
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 810, "elapsed_sec": 1076.835, "predictions": 248434, "recent_rows_per_sec": 414.87, "row_group_batches": 810, "rows_per_sec": 230.71, "timing_avg_sec": {"forward": 0.2495, "merge": 0.0001, "move": 0.0066, "post": 0.0002}}
2026-05-09 19:23:01.562
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 800, "elapsed_sec": 1069.042, "predictions": 245201, "recent_rows_per_sec": 295.61, "row_group_batches": 800, "rows_per_sec": 229.37, "timing_avg_sec": {"forward": 0.2495, "merge": 0.0001, "move": 0.0067, "post": 0.0002}}
2026-05-09 19:22:53.770
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 790, "elapsed_sec": 1058.214, "predictions": 242000, "recent_rows_per_sec": 262.83, "row_group_batches": 790, "rows_per_sec": 228.69, "timing_avg_sec": {"forward": 0.2496, "merge": 0.0001, "move": 0.0068, "post": 0.0002}}
2026-05-09 19:22:42.941
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 780, "elapsed_sec": 1045.833, "predictions": 238746, "recent_rows_per_sec": 257.08, "row_group_batches": 780, "rows_per_sec": 228.28, "timing_avg_sec": {"forward": 0.2493, "merge": 0.0001, "move": 0.0068, "post": 0.0002}}
2026-05-09 19:22:30.561
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 770, "elapsed_sec": 1033.249, "predictions": 235511, "recent_rows_per_sec": 330.93, "row_group_batches": 770, "rows_per_sec": 227.93, "timing_avg_sec": {"forward": 0.2493, "merge": 0.0001, "move": 0.0066, "post": 0.0002}}
2026-05-09 19:22:17.977
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 760, "elapsed_sec": 1023.311, "predictions": 232222, "recent_rows_per_sec": 300.66, "row_group_batches": 760, "rows_per_sec": 226.93, "timing_avg_sec": {"forward": 0.2492, "merge": 0.0001, "move": 0.0067, "post": 0.0002}}
2026-05-09 19:22:08.038
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 750, "elapsed_sec": 1012.355, "predictions": 228928, "recent_rows_per_sec": 274.9, "row_group_batches": 750, "rows_per_sec": 226.13, "timing_avg_sec": {"forward": 0.2493, "merge": 0.0001, "move": 0.0066, "post": 0.0002}}
2026-05-09 19:21:57.082
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 740, "elapsed_sec": 1000.452, "predictions": 225656, "recent_rows_per_sec": 232.04, "row_group_batches": 740, "rows_per_sec": 225.55, "timing_avg_sec": {"forward": 0.2494, "merge": 0.0001, "move": 0.0066, "post": 0.0002}}
2026-05-09 19:21:45.180
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 730, "elapsed_sec": 986.631, "predictions": 222449, "recent_rows_per_sec": 202.69, "row_group_batches": 730, "rows_per_sec": 225.46, "timing_avg_sec": {"forward": 0.2494, "merge": 0.0001, "move": 0.0067, "post": 0.0002}}
2026-05-09 19:21:31.359
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 720, "elapsed_sec": 970.725, "predictions": 219225, "recent_rows_per_sec": 255.68, "row_group_batches": 720, "rows_per_sec": 225.84, "timing_avg_sec": {"forward": 0.2492, "merge": 0.0001, "move": 0.0067, "post": 0.0002}}
2026-05-09 19:21:15.452
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 710, "elapsed_sec": 958.014, "predictions": 215975, "recent_rows_per_sec": 272.89, "row_group_batches": 710, "rows_per_sec": 225.44, "timing_avg_sec": {"forward": 0.2489, "merge": 0.0001, "move": 0.0067, "post": 0.0002}}
2026-05-09 19:21:02.741
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 700, "elapsed_sec": 946.126, "predictions": 212731, "recent_rows_per_sec": 272.4, "row_group_batches": 700, "rows_per_sec": 224.84, "timing_avg_sec": {"forward": 0.2489, "merge": 0.0001, "move": 0.0068, "post": 0.0002}}
2026-05-09 19:20:50.853
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 690, "elapsed_sec": 934.408, "predictions": 209539, "recent_rows_per_sec": 224.08, "row_group_batches": 690, "rows_per_sec": 224.25, "timing_avg_sec": {"forward": 0.2489, "merge": 0.0001, "move": 0.0069, "post": 0.0002}}
2026-05-09 19:20:39.135
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 680, "elapsed_sec": 920.136, "predictions": 206341, "recent_rows_per_sec": 227.4, "row_group_batches": 680, "rows_per_sec": 224.25, "timing_avg_sec": {"forward": 0.2486, "merge": 0.0001, "move": 0.007, "post": 0.0002}}
2026-05-09 19:20:24.864
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 670, "elapsed_sec": 906.235, "predictions": 203180, "recent_rows_per_sec": 281.19, "row_group_batches": 670, "rows_per_sec": 224.2, "timing_avg_sec": {"forward": 0.2489, "merge": 0.0001, "move": 0.0066, "post": 0.0002}}
2026-05-09 19:20:10.963
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 660, "elapsed_sec": 895.029, "predictions": 200029, "recent_rows_per_sec": 221.4, "row_group_batches": 660, "rows_per_sec": 223.49, "timing_avg_sec": {"forward": 0.249, "merge": 0.0002, "move": 0.0067, "post": 0.0002}}
2026-05-09 19:19:59.757
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 650, "elapsed_sec": 880.824, "predictions": 196884, "recent_rows_per_sec": 206.21, "row_group_batches": 650, "rows_per_sec": 223.52, "timing_avg_sec": {"forward": 0.2492, "merge": 0.0002, "move": 0.0068, "post": 0.0002}}
2026-05-09 19:19:45.552
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 640, "elapsed_sec": 865.641, "predictions": 193753, "recent_rows_per_sec": 215.92, "row_group_batches": 640, "rows_per_sec": 223.83, "timing_avg_sec": {"forward": 0.2494, "merge": 0.0002, "move": 0.0065, "post": 0.0002}}
2026-05-09 19:19:30.369
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 630, "elapsed_sec": 850.909, "predictions": 190572, "recent_rows_per_sec": 271.06, "row_group_batches": 630, "rows_per_sec": 223.96, "timing_avg_sec": {"forward": 0.2491, "merge": 0.0002, "move": 0.0064, "post": 0.0002}}
2026-05-09 19:19:15.636
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 620, "elapsed_sec": 839.243, "predictions": 187410, "recent_rows_per_sec": 221.24, "row_group_batches": 620, "rows_per_sec": 223.31, "timing_avg_sec": {"forward": 0.2492, "merge": 0.0002, "move": 0.0062, "post": 0.0002}}
2026-05-09 19:19:03.971
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 610, "elapsed_sec": 825.114, "predictions": 184284, "recent_rows_per_sec": 192.12, "row_group_batches": 610, "rows_per_sec": 223.34, "timing_avg_sec": {"forward": 0.2494, "merge": 0.0002, "move": 0.006, "post": 0.0002}}
2026-05-09 19:18:49.842
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 600, "elapsed_sec": 809.02, "predictions": 181192, "recent_rows_per_sec": 261.33, "row_group_batches": 600, "rows_per_sec": 223.96, "timing_avg_sec": {"forward": 0.2492, "merge": 0.0002, "move": 0.0061, "post": 0.0002}}
2026-05-09 19:18:33.748
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 590, "elapsed_sec": 797.142, "predictions": 178088, "recent_rows_per_sec": 208.14, "row_group_batches": 590, "rows_per_sec": 223.41, "timing_avg_sec": {"forward": 0.2493, "merge": 0.0002, "move": 0.0062, "post": 0.0002}}
2026-05-09 19:18:21.870
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 580, "elapsed_sec": 782.118, "predictions": 174961, "recent_rows_per_sec": 224.41, "row_group_batches": 580, "rows_per_sec": 223.7, "timing_avg_sec": {"forward": 0.2491, "merge": 0.0002, "move": 0.0061, "post": 0.0002}}
2026-05-09 19:18:06.846
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 570, "elapsed_sec": 768.402, "predictions": 171883, "recent_rows_per_sec": 250.71, "row_group_batches": 570, "rows_per_sec": 223.69, "timing_avg_sec": {"forward": 0.2491, "merge": 0.0002, "move": 0.0062, "post": 0.0002}}
2026-05-09 19:17:53.130
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 560, "elapsed_sec": 756.337, "predictions": 168858, "recent_rows_per_sec": 170.3, "row_group_batches": 560, "rows_per_sec": 223.26, "timing_avg_sec": {"forward": 0.2491, "merge": 0.0002, "move": 0.0063, "post": 0.0002}}
2026-05-09 19:17:41.064
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 550, "elapsed_sec": 738.704, "predictions": 165855, "recent_rows_per_sec": 180.53, "row_group_batches": 550, "rows_per_sec": 224.52, "timing_avg_sec": {"forward": 0.249, "merge": 0.0002, "move": 0.0062, "post": 0.0002}}
2026-05-09 19:17:23.431
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 540, "elapsed_sec": 722.003, "predictions": 162840, "recent_rows_per_sec": 195.8, "row_group_batches": 540, "rows_per_sec": 225.54, "timing_avg_sec": {"forward": 0.2489, "merge": 0.0002, "move": 0.0062, "post": 0.0002}}
2026-05-09 19:17:06.730
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 530, "elapsed_sec": 706.324, "predictions": 159770, "recent_rows_per_sec": 279.34, "row_group_batches": 530, "rows_per_sec": 226.2, "timing_avg_sec": {"forward": 0.2488, "merge": 0.0002, "move": 0.0063, "post": 0.0002}}
2026-05-09 19:16:51.051
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 520, "elapsed_sec": 695.126, "predictions": 156642, "recent_rows_per_sec": 230.61, "row_group_batches": 520, "rows_per_sec": 225.34, "timing_avg_sec": {"forward": 0.2484, "merge": 0.0002, "move": 0.0064, "post": 0.0002}}
2026-05-09 19:16:39.853
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 510, "elapsed_sec": 681.809, "predictions": 153571, "recent_rows_per_sec": 234.39, "row_group_batches": 510, "rows_per_sec": 225.24, "timing_avg_sec": {"forward": 0.2485, "merge": 0.0002, "move": 0.0063, "post": 0.0002}}
2026-05-09 19:16:26.537
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 500, "elapsed_sec": 668.639, "predictions": 150484, "recent_rows_per_sec": 205.67, "row_group_batches": 500, "rows_per_sec": 225.06, "timing_avg_sec": {"forward": 0.2488, "merge": 0.0002, "move": 0.0062, "post": 0.0002}}
2026-05-09 19:16:13.367
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 490, "elapsed_sec": 653.742, "predictions": 147420, "recent_rows_per_sec": 343.74, "row_group_batches": 490, "rows_per_sec": 225.5, "timing_avg_sec": {"forward": 0.2482, "merge": 0.0002, "move": 0.0064, "post": 0.0002}}
2026-05-09 19:15:58.469
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 480, "elapsed_sec": 644.831, "predictions": 144357, "recent_rows_per_sec": 227.4, "row_group_batches": 480, "rows_per_sec": 223.87, "timing_avg_sec": {"forward": 0.2485, "merge": 0.0002, "move": 0.0061, "post": 0.0002}}
2026-05-09 19:15:49.559
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 470, "elapsed_sec": 631.704, "predictions": 141372, "recent_rows_per_sec": 239.79, "row_group_batches": 470, "rows_per_sec": 223.79, "timing_avg_sec": {"forward": 0.2483, "merge": 0.0002, "move": 0.0062, "post": 0.0002}}
2026-05-09 19:15:36.432
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 460, "elapsed_sec": 619.44, "predictions": 138431, "recent_rows_per_sec": 207.18, "row_group_batches": 460, "rows_per_sec": 223.48, "timing_avg_sec": {"forward": 0.248, "merge": 0.0002, "move": 0.0061, "post": 0.0002}}
2026-05-09 19:15:24.167
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 450, "elapsed_sec": 605.22, "predictions": 135485, "recent_rows_per_sec": 204.97, "row_group_batches": 450, "rows_per_sec": 223.86, "timing_avg_sec": {"forward": 0.2486, "merge": 0.0002, "move": 0.0058, "post": 0.0002}}
2026-05-09 19:15:09.948
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 440, "elapsed_sec": 590.72, "predictions": 132513, "recent_rows_per_sec": 193.38, "row_group_batches": 440, "rows_per_sec": 224.32, "timing_avg_sec": {"forward": 0.2487, "merge": 0.0002, "move": 0.0059, "post": 0.0002}}
2026-05-09 19:14:55.448
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 430, "elapsed_sec": 575.218, "predictions": 129515, "recent_rows_per_sec": 181.47, "row_group_batches": 430, "rows_per_sec": 225.16, "timing_avg_sec": {"forward": 0.2483, "merge": 0.0002, "move": 0.0061, "post": 0.0002}}
2026-05-09 19:14:39.945
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 420, "elapsed_sec": 558.719, "predictions": 126521, "recent_rows_per_sec": 245.56, "row_group_batches": 420, "rows_per_sec": 226.45, "timing_avg_sec": {"forward": 0.248, "merge": 0.0002, "move": 0.0062, "post": 0.0002}}
2026-05-09 19:14:23.446
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 410, "elapsed_sec": 546.526, "predictions": 123527, "recent_rows_per_sec": 226.48, "row_group_batches": 410, "rows_per_sec": 226.02, "timing_avg_sec": {"forward": 0.2486, "merge": 0.0002, "move": 0.0063, "post": 0.0002}}
2026-05-09 19:14:11.254
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 400, "elapsed_sec": 533.404, "predictions": 120555, "recent_rows_per_sec": 220.45, "row_group_batches": 400, "rows_per_sec": 226.01, "timing_avg_sec": {"forward": 0.2486, "merge": 0.0002, "move": 0.0065, "post": 0.0002}}
2026-05-09 19:13:58.131
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 390, "elapsed_sec": 519.836, "predictions": 117564, "recent_rows_per_sec": 257.46, "row_group_batches": 390, "rows_per_sec": 226.16, "timing_avg_sec": {"forward": 0.2486, "merge": 0.0003, "move": 0.0067, "post": 0.0002}}
2026-05-09 19:13:44.564
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 380, "elapsed_sec": 508.421, "predictions": 114625, "recent_rows_per_sec": 258.85, "row_group_batches": 380, "rows_per_sec": 225.45, "timing_avg_sec": {"forward": 0.2491, "merge": 0.0003, "move": 0.0068, "post": 0.0002}}
2026-05-09 19:13:33.148
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 370, "elapsed_sec": 497.21, "predictions": 111723, "recent_rows_per_sec": 199.03, "row_group_batches": 370, "rows_per_sec": 224.7, "timing_avg_sec": {"forward": 0.2487, "merge": 0.0003, "move": 0.007, "post": 0.0002}}
2026-05-09 19:13:21.937
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 360, "elapsed_sec": 482.528, "predictions": 108801, "recent_rows_per_sec": 275.46, "row_group_batches": 360, "rows_per_sec": 225.48, "timing_avg_sec": {"forward": 0.2486, "merge": 0.0003, "move": 0.0072, "post": 0.0002}}
2026-05-09 19:13:07.256
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 350, "elapsed_sec": 471.931, "predictions": 105882, "recent_rows_per_sec": 250.45, "row_group_batches": 350, "rows_per_sec": 224.36, "timing_avg_sec": {"forward": 0.2482, "merge": 0.0003, "move": 0.0074, "post": 0.0002}}
2026-05-09 19:12:56.659
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 340, "elapsed_sec": 460.332, "predictions": 102977, "recent_rows_per_sec": 189.65, "row_group_batches": 340, "rows_per_sec": 223.7, "timing_avg_sec": {"forward": 0.2485, "merge": 0.0003, "move": 0.0076, "post": 0.0002}}
2026-05-09 19:12:45.060
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 330, "elapsed_sec": 444.819, "predictions": 100035, "recent_rows_per_sec": 215.32, "row_group_batches": 330, "rows_per_sec": 224.89, "timing_avg_sec": {"forward": 0.2487, "merge": 0.0, "move": 0.0078, "post": 0.0002}}
2026-05-09 19:12:29.547
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 320, "elapsed_sec": 431.031, "predictions": 97066, "recent_rows_per_sec": 255.07, "row_group_batches": 320, "rows_per_sec": 225.2, "timing_avg_sec": {"forward": 0.2485, "merge": 0.0, "move": 0.008, "post": 0.0002}}
2026-05-09 19:12:15.758
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 310, "elapsed_sec": 419.23, "predictions": 94056, "recent_rows_per_sec": 226.31, "row_group_batches": 310, "rows_per_sec": 224.35, "timing_avg_sec": {"forward": 0.249, "merge": 0.0, "move": 0.0077, "post": 0.0002}}
2026-05-09 19:12:03.957
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 300, "elapsed_sec": 405.722, "predictions": 90999, "recent_rows_per_sec": 220.59, "row_group_batches": 300, "rows_per_sec": 224.29, "timing_avg_sec": {"forward": 0.249, "merge": 0.0, "move": 0.0079, "post": 0.0002}}
2026-05-09 19:11:50.449
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 290, "elapsed_sec": 391.719, "predictions": 87910, "recent_rows_per_sec": 202.12, "row_group_batches": 290, "rows_per_sec": 224.42, "timing_avg_sec": {"forward": 0.2481, "merge": 0.0, "move": 0.0082, "post": 0.0002}}
2026-05-09 19:11:36.446
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 280, "elapsed_sec": 376.436, "predictions": 84821, "recent_rows_per_sec": 198.37, "row_group_batches": 280, "rows_per_sec": 225.33, "timing_avg_sec": {"forward": 0.249, "merge": 0.0, "move": 0.0085, "post": 0.0002}}
2026-05-09 19:11:21.163
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 270, "elapsed_sec": 361.03, "predictions": 81765, "recent_rows_per_sec": 218.81, "row_group_batches": 270, "rows_per_sec": 226.48, "timing_avg_sec": {"forward": 0.2489, "merge": 0.0, "move": 0.0077, "post": 0.0002}}
2026-05-09 19:11:05.758
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 260, "elapsed_sec": 347.027, "predictions": 78701, "recent_rows_per_sec": 205.14, "row_group_batches": 260, "rows_per_sec": 226.79, "timing_avg_sec": {"forward": 0.2488, "merge": 0.0, "move": 0.0079, "post": 0.0002}}
2026-05-09 19:10:51.755
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 250, "elapsed_sec": 332.227, "predictions": 75665, "recent_rows_per_sec": 197.43, "row_group_batches": 250, "rows_per_sec": 227.75, "timing_avg_sec": {"forward": 0.248, "merge": 0.0, "move": 0.0082, "post": 0.0002}}
2026-05-09 19:10:36.955
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 240, "elapsed_sec": 316.931, "predictions": 72645, "recent_rows_per_sec": 187.67, "row_group_batches": 240, "rows_per_sec": 229.21, "timing_avg_sec": {"forward": 0.2467, "merge": 0.0, "move": 0.0086, "post": 0.0002}}
2026-05-09 19:10:21.658
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 230, "elapsed_sec": 300.801, "predictions": 69618, "recent_rows_per_sec": 198.09, "row_group_batches": 230, "rows_per_sec": 231.44, "timing_avg_sec": {"forward": 0.2466, "merge": 0.0, "move": 0.0089, "post": 0.0002}}
2026-05-09 19:10:05.529
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 220, "elapsed_sec": 285.702, "predictions": 66627, "recent_rows_per_sec": 196.95, "row_group_batches": 220, "rows_per_sec": 233.2, "timing_avg_sec": {"forward": 0.2476, "merge": 0.0, "move": 0.0089, "post": 0.0002}}
2026-05-09 19:09:50.430
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 210, "elapsed_sec": 270.734, "predictions": 63679, "recent_rows_per_sec": 208.42, "row_group_batches": 210, "rows_per_sec": 235.21, "timing_avg_sec": {"forward": 0.2471, "merge": 0.0, "move": 0.0084, "post": 0.0002}}
2026-05-09 19:09:35.462
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 200, "elapsed_sec": 256.604, "predictions": 60734, "recent_rows_per_sec": 249.13, "row_group_batches": 200, "rows_per_sec": 236.68, "timing_avg_sec": {"forward": 0.2459, "merge": 0.0, "move": 0.0088, "post": 0.0002}}
2026-05-09 19:09:21.331
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 190, "elapsed_sec": 244.731, "predictions": 57776, "recent_rows_per_sec": 230.49, "row_group_batches": 190, "rows_per_sec": 236.08, "timing_avg_sec": {"forward": 0.2469, "merge": 0.0, "move": 0.0087, "post": 0.0002}}
2026-05-09 19:09:09.458
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 180, "elapsed_sec": 231.893, "predictions": 54817, "recent_rows_per_sec": 248.91, "row_group_batches": 180, "rows_per_sec": 236.39, "timing_avg_sec": {"forward": 0.2472, "merge": 0.0, "move": 0.0086, "post": 0.0002}}
2026-05-09 19:08:56.620
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 170, "elapsed_sec": 220.021, "predictions": 51862, "recent_rows_per_sec": 217.99, "row_group_batches": 170, "rows_per_sec": 235.71, "timing_avg_sec": {"forward": 0.2465, "merge": 0.0, "move": 0.0085, "post": 0.0002}}
2026-05-09 19:08:44.748
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 160, "elapsed_sec": 206.438, "predictions": 48901, "recent_rows_per_sec": 225.53, "row_group_batches": 160, "rows_per_sec": 236.88, "timing_avg_sec": {"forward": 0.2466, "merge": 0.0, "move": 0.0091, "post": 0.0002}}
2026-05-09 19:08:31.165
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 150, "elapsed_sec": 193.131, "predictions": 45900, "recent_rows_per_sec": 241.28, "row_group_batches": 150, "rows_per_sec": 237.66, "timing_avg_sec": {"forward": 0.2473, "merge": 0.0, "move": 0.0083, "post": 0.0002}}
2026-05-09 19:08:17.859
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 140, "elapsed_sec": 180.54, "predictions": 42862, "recent_rows_per_sec": 198.94, "row_group_batches": 140, "rows_per_sec": 237.41, "timing_avg_sec": {"forward": 0.2494, "merge": 0.0, "move": 0.0082, "post": 0.0002}}
2026-05-09 19:08:05.268
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 130, "elapsed_sec": 165.435, "predictions": 39857, "recent_rows_per_sec": 275.76, "row_group_batches": 130, "rows_per_sec": 240.92, "timing_avg_sec": {"forward": 0.2507, "merge": 0.0, "move": 0.0088, "post": 0.0002}}
2026-05-09 19:07:50.163
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 120, "elapsed_sec": 154.531, "predictions": 36850, "recent_rows_per_sec": 206.72, "row_group_batches": 120, "rows_per_sec": 238.46, "timing_avg_sec": {"forward": 0.2502, "merge": 0.0, "move": 0.0079, "post": 0.0002}}
2026-05-09 19:07:39.258
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 110, "elapsed_sec": 139.844, "predictions": 33814, "recent_rows_per_sec": 234.87, "row_group_batches": 110, "rows_per_sec": 241.8, "timing_avg_sec": {"forward": 0.2498, "merge": 0.0, "move": 0.0076, "post": 0.0002}}
2026-05-09 19:07:24.571
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 100, "elapsed_sec": 126.832, "predictions": 30758, "recent_rows_per_sec": 228.97, "row_group_batches": 100, "rows_per_sec": 242.51, "timing_avg_sec": {"forward": 0.2512, "merge": 0.0, "move": 0.0074, "post": 0.0002}}
2026-05-09 19:07:11.560
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 90, "elapsed_sec": 113.537, "predictions": 27714, "recent_rows_per_sec": 237.87, "row_group_batches": 90, "rows_per_sec": 244.1, "timing_avg_sec": {"forward": 0.2513, "merge": 0.0, "move": 0.0082, "post": 0.0002}}
2026-05-09 19:06:58.265
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 80, "elapsed_sec": 100.703, "predictions": 24661, "recent_rows_per_sec": 300.1, "row_group_batches": 80, "rows_per_sec": 244.89, "timing_avg_sec": {"forward": 0.2535, "merge": 0.0, "move": 0.0067, "post": 0.0002}}
2026-05-09 19:06:45.430
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 70, "elapsed_sec": 90.536, "predictions": 21610, "recent_rows_per_sec": 238.24, "row_group_batches": 70, "rows_per_sec": 238.69, "timing_avg_sec": {"forward": 0.2529, "merge": 0.0, "move": 0.0062, "post": 0.0002}}
2026-05-09 19:06:35.264
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 60, "elapsed_sec": 77.73, "predictions": 18559, "recent_rows_per_sec": 277.98, "row_group_batches": 60, "rows_per_sec": 238.76, "timing_avg_sec": {"forward": 0.253, "merge": 0.0, "move": 0.0072, "post": 0.0002}}
2026-05-09 19:06:22.457
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 50, "elapsed_sec": 66.715, "predictions": 15497, "recent_rows_per_sec": 223.66, "row_group_batches": 50, "rows_per_sec": 232.29, "timing_avg_sec": {"forward": 0.2558, "merge": 0.0, "move": 0.0066, "post": 0.0002}}
2026-05-09 19:06:11.442
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 40, "elapsed_sec": 52.904, "predictions": 12408, "recent_rows_per_sec": 248.34, "row_group_batches": 40, "rows_per_sec": 234.54, "timing_avg_sec": {"forward": 0.2545, "merge": 0.0, "move": 0.0082, "post": 0.0002}}
2026-05-09 19:05:57.631
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 30, "elapsed_sec": 40.437, "predictions": 9312, "recent_rows_per_sec": 249.81, "row_group_batches": 30, "rows_per_sec": 230.29, "timing_avg_sec": {"forward": 0.2604, "merge": 0.0, "move": 0.0075, "post": 0.0002}}
2026-05-09 19:05:45.164
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 20, "elapsed_sec": 28.127, "predictions": 6237, "recent_rows_per_sec": 265.1, "row_group_batches": 20, "rows_per_sec": 221.74, "timing_avg_sec": {"forward": 0.263, "merge": 0.0, "move": 0.0062, "post": 0.0002}}
2026-05-09 19:05:32.855
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 10, "elapsed_sec": 16.438, "predictions": 3138, "recent_rows_per_sec": 190.9, "row_group_batches": 10, "rows_per_sec": 190.9, "timing_avg_sec": {"forward": 0.3083, "merge": 0.0, "move": 0.0121, "post": 0.0003}}
2026-05-09 19:05:21.165
PLATFORM_FIRST_BATCH {"batch_predictions": 314, "rows_in_forward": 314, "sample_scores": [0.00370789, 0.07470703, 0.03955078, 0.17675781, 0.44921875], "sample_user_ids": ["683974", "1334888", "51348", "1184407", "8854"], "sanitize_counts": {"nan_replaced": 0, "neginf_replaced": 0, "posinf_replaced": 0}, "timing_sec": {"forward": 0.6148, "merge": 0.0, "move": 0.1181, "post": 0.0002}, "tta_mode": "none"}
2026-05-09 19:05:08.227
2026-05-09 11:05:05,058 - INFO - NumExpr defaulting to 16 threads.
2026-05-09 19:05:05.058
2026-05-09 11:05:05,058 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-09 19:05:05.058
2026-05-09 11:05:05,057 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-09 19:05:05.058
PLATFORM_AMP_CONFIG {"cuda_bf16_supported": true, "enabled": true, "mode": "bf16", "requested": "bf16"}
2026-05-09 19:05:04.727
PLATFORM_MODEL_READY {"device": "cuda", "num_ns": 8, "num_parameters": 239931649}
2026-05-09 19:05:04.727
2026-05-09 11:04:57,304 - INFO - emb_skip_threshold=1000000: seq_c skipped 3/11 features
2026-05-09 19:04:57.304
2026-05-09 11:04:57,303 - INFO - emb_skip_threshold=1000000: seq_b skipped 1/13 features
2026-05-09 19:04:57.304
2026-05-09 11:04:55,217 - INFO - RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0
2026-05-09 19:04:55.218
2026-05-09 11:04:55,198 - INFO - RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1
2026-05-09 19:04:55.199
PLATFORM_DATASET_CONFIG {"batch_size": 2048, "item_dense_dim": 0, "item_int_features": 14, "num_row_groups": 1000, "num_rows_estimate": 310000, "seq_domains": ["seq_a", "seq_b", "seq_c", "seq_d"], "seq_max_lens": {"seq_a": 256, "seq_b": 256, "seq_c": 512, "seq_d": 512}, "user_dense_dim": 918, "user_int_features": 46}
2026-05-09 19:04:55.174
2026-05-09 11:04:55,174 - INFO - PCVRParquetDataset: 310000 rows from 1000 file(s), batch_size=2048, buffer_batches=0, shuffle=False
2026-05-09 19:04:55.174
PLATFORM_INFER_CONFIG {"amp_dtype": "bf16", "checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "device": "cuda", "eval_data_dir": "/data_ams/academic_infer_data/", "merge_batches": false, "model_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt", "num_workers": 0, "progress_every": 10, "result_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/75522/results", "schema_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "train_config": {"d_model": 64, "emb_skip_threshold": 1000000, "eval_batch_size": 2048, "loss_type": "bce", "num_heads": 4, "num_hyformer_blocks": 2, "num_queries": 2, "seq_encoder_type": "transformer", "seq_max_lens": "seq_a:256,seq_b:256,seq_c:512,seq_d:512", "train_batch_size": 256}, "tta_crop_lens": "seq_a:128,seq_b:128,seq_c:256,seq_d:256", "tta_mode": "none", "tta_weight": 0.12, "use_amp": true}
2026-05-09 19:04:54.688
2026-05-09 11:04:54,688 - INFO - Using schema path: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json
2026-05-09 19:04:54.688
2026-05-09 11:04:54,688 - INFO - Using checkpoint directory: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model
2026-05-09 19:04:54.688
PLATFORM_CHECKPOINT_STATUS {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "model_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt", "size_bytes": 959871995}, "schema_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "size_bytes": 5560}, "train_config_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/train_config.json", "size_bytes": 1638}}
2026-05-09 19:04:54.686
PLATFORM_INFER_START {"cuda_available": true, "cwd": "/workspace", "time": "2026-05-09 11:04:54", "torch_version": "2.7.1+cu126"}
2026-05-09 19:04:54.679
====== Inferring ======
2026-05-09 19:04:53.435
=================================
2026-05-09 19:04:53.050
Working Dir: /workspace
2026-05-09 19:04:53.050
Environment: competition
2026-05-09 19:04:53.050
GPU Count: 1
2026-05-09 19:04:53.050
GPU Available: True
2026-05-09 19:04:51.586
PyTorch: 2.7.1+cu126
2026-05-09 19:04:49.972
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-09 19:04:47.578
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-09 19:04:47.578
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-09 19:04:47.578
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-09 19:04:47.578
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-09 19:04:47.575
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-09 19:04:47.575
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-09 19:04:47.575
Python: Python 3.10.20
2026-05-09 19:04:47.245
CUDA: 12.6.77
2026-05-09 19:04:47.240
=== Competition Environment Ready ===
2026-05-09 19:04:47.232
Complete setting network policy rules.
2026-05-09 19:04:43.888
Complete setting taiji user.
2026-05-09 19:04:43.834
Truncated to last 1000 lines.