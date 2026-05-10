
====== Infer Finished ======
2026-05-10 11:45:24.061
2026-05-10 03:45:23,336 - INFO - Saved status.json to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/77616/results/status.json
2026-05-10 11:45:23.336
2026-05-10 03:45:22,899 - INFO - Saved 310000 predictions to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/77616/results/predictions.json
2026-05-10 11:45:22.899
PLATFORM_INFER_SUMMARY {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "elapsed_sec": 1890.286, "num_predictions": 310000, "num_unique_user_ids": 310000, "output_shape": "predictions_dict_user_id_to_float", "result_files": {"csv": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/77616/results/submission.csv", "size_bytes": 5643992}, "json": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/77616/results/predictions.json", "size_bytes": 7808678}, "scores": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/77616/results/scores.npy", "size_bytes": 1240128}}, "result_json": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/77616/results/predictions.json", "score_stats": {"count": 310000, "inf_count": 0, "max": 0.9765625, "mean": 0.11910518, "min": 0.00064468, "nan_count": 0, "p01": 0.0008049, "p50": 0.06005859, "p99": 0.81640625, "std": 0.15703748}, "scores_npy": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/77616/results/scores.npy", "submission_csv": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/77616/results/submission.csv"}
2026-05-10 11:45:22.899
2026-05-10 03:45:22,350 - INFO - Inference complete: 310000 predictions
2026-05-10 11:45:22.350
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 1000, "elapsed_sec": 1889.677, "predictions": 310000, "recent_rows_per_sec": 208.28, "rows_per_sec": 164.05, "timing_avg_sec": {"forward": 0.2386, "move": 0.0101, "post": 0.0064}}
2026-05-10 11:45:22.264
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 990, "elapsed_sec": 1874.568, "predictions": 306853, "recent_rows_per_sec": 175.95, "rows_per_sec": 163.69, "timing_avg_sec": {"forward": 0.2384, "move": 0.0101, "post": 0.0064}}
2026-05-10 11:45:07.154
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 980, "elapsed_sec": 1856.364, "predictions": 303650, "recent_rows_per_sec": 225.31, "rows_per_sec": 163.57, "timing_avg_sec": {"forward": 0.2384, "move": 0.01, "post": 0.0064}}
2026-05-10 11:44:48.951
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 970, "elapsed_sec": 1842.055, "predictions": 300426, "recent_rows_per_sec": 193.57, "rows_per_sec": 163.09, "timing_avg_sec": {"forward": 0.2384, "move": 0.0101, "post": 0.0064}}
2026-05-10 11:44:34.644
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 960, "elapsed_sec": 1825.271, "predictions": 297177, "recent_rows_per_sec": 157.81, "rows_per_sec": 162.81, "timing_avg_sec": {"forward": 0.2383, "move": 0.0101, "post": 0.0064}}
2026-05-10 11:44:17.857
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 950, "elapsed_sec": 1804.587, "predictions": 293913, "recent_rows_per_sec": 161.85, "rows_per_sec": 162.87, "timing_avg_sec": {"forward": 0.2383, "move": 0.01, "post": 0.0064}}
2026-05-10 11:43:57.174
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 940, "elapsed_sec": 1784.773, "predictions": 290706, "recent_rows_per_sec": 194.59, "rows_per_sec": 162.88, "timing_avg_sec": {"forward": 0.2381, "move": 0.0101, "post": 0.0064}}
2026-05-10 11:43:37.359
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 930, "elapsed_sec": 1768.219, "predictions": 287485, "recent_rows_per_sec": 166.1, "rows_per_sec": 162.58, "timing_avg_sec": {"forward": 0.238, "move": 0.0102, "post": 0.0064}}
2026-05-10 11:43:20.806
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 920, "elapsed_sec": 1748.575, "predictions": 284222, "recent_rows_per_sec": 190.84, "rows_per_sec": 162.54, "timing_avg_sec": {"forward": 0.2379, "move": 0.0104, "post": 0.0064}}
2026-05-10 11:43:01.161
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 910, "elapsed_sec": 1731.461, "predictions": 280956, "recent_rows_per_sec": 205.33, "rows_per_sec": 162.27, "timing_avg_sec": {"forward": 0.2377, "move": 0.0103, "post": 0.0064}}
2026-05-10 11:42:44.047
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 900, "elapsed_sec": 1715.691, "predictions": 277718, "recent_rows_per_sec": 171.85, "rows_per_sec": 161.87, "timing_avg_sec": {"forward": 0.2376, "move": 0.0104, "post": 0.0064}}
2026-05-10 11:42:28.278
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 890, "elapsed_sec": 1696.878, "predictions": 274485, "recent_rows_per_sec": 188.8, "rows_per_sec": 161.76, "timing_avg_sec": {"forward": 0.2375, "move": 0.0106, "post": 0.0064}}
2026-05-10 11:42:09.464
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 880, "elapsed_sec": 1679.754, "predictions": 271252, "recent_rows_per_sec": 214.75, "rows_per_sec": 161.48, "timing_avg_sec": {"forward": 0.2377, "move": 0.0105, "post": 0.0064}}
2026-05-10 11:41:52.340
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 870, "elapsed_sec": 1664.573, "predictions": 267992, "recent_rows_per_sec": 187.2, "rows_per_sec": 161.0, "timing_avg_sec": {"forward": 0.2375, "move": 0.0106, "post": 0.0064}}
2026-05-10 11:41:37.160
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 860, "elapsed_sec": 1646.854, "predictions": 264675, "recent_rows_per_sec": 183.1, "rows_per_sec": 160.72, "timing_avg_sec": {"forward": 0.2373, "move": 0.0107, "post": 0.0064}}
2026-05-10 11:41:19.440
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 850, "elapsed_sec": 1628.831, "predictions": 261375, "recent_rows_per_sec": 155.73, "rows_per_sec": 160.47, "timing_avg_sec": {"forward": 0.2372, "move": 0.0107, "post": 0.0064}}
2026-05-10 11:41:01.417
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 840, "elapsed_sec": 1607.949, "predictions": 258123, "recent_rows_per_sec": 167.97, "rows_per_sec": 160.53, "timing_avg_sec": {"forward": 0.2368, "move": 0.0108, "post": 0.0064}}
2026-05-10 11:40:40.536
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 830, "elapsed_sec": 1588.767, "predictions": 254901, "recent_rows_per_sec": 159.86, "rows_per_sec": 160.44, "timing_avg_sec": {"forward": 0.2367, "move": 0.0109, "post": 0.0064}}
2026-05-10 11:40:21.353
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 820, "elapsed_sec": 1568.58, "predictions": 251674, "recent_rows_per_sec": 168.56, "rows_per_sec": 160.45, "timing_avg_sec": {"forward": 0.2366, "move": 0.0109, "post": 0.0064}}
2026-05-10 11:40:01.166
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 810, "elapsed_sec": 1549.359, "predictions": 248434, "recent_rows_per_sec": 183.02, "rows_per_sec": 160.35, "timing_avg_sec": {"forward": 0.2366, "move": 0.0106, "post": 0.0064}}
2026-05-10 11:39:41.945
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 800, "elapsed_sec": 1531.694, "predictions": 245201, "recent_rows_per_sec": 212.87, "rows_per_sec": 160.08, "timing_avg_sec": {"forward": 0.2367, "move": 0.0105, "post": 0.0064}}
2026-05-10 11:39:24.280
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 790, "elapsed_sec": 1516.657, "predictions": 242000, "recent_rows_per_sec": 188.07, "rows_per_sec": 159.56, "timing_avg_sec": {"forward": 0.2366, "move": 0.0106, "post": 0.0064}}
2026-05-10 11:39:09.243
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 780, "elapsed_sec": 1499.355, "predictions": 238746, "recent_rows_per_sec": 192.85, "rows_per_sec": 159.23, "timing_avg_sec": {"forward": 0.2365, "move": 0.0105, "post": 0.0064}}
2026-05-10 11:38:51.941
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 770, "elapsed_sec": 1482.581, "predictions": 235511, "recent_rows_per_sec": 174.82, "rows_per_sec": 158.85, "timing_avg_sec": {"forward": 0.2364, "move": 0.0106, "post": 0.0064}}
2026-05-10 11:38:35.167
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 760, "elapsed_sec": 1463.767, "predictions": 232222, "recent_rows_per_sec": 178.32, "rows_per_sec": 158.65, "timing_avg_sec": {"forward": 0.2363, "move": 0.0106, "post": 0.0063}}
2026-05-10 11:38:16.353
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 750, "elapsed_sec": 1445.295, "predictions": 228928, "recent_rows_per_sec": 175.75, "rows_per_sec": 158.4, "timing_avg_sec": {"forward": 0.2361, "move": 0.0106, "post": 0.0064}}
2026-05-10 11:37:57.881
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 740, "elapsed_sec": 1426.678, "predictions": 225656, "recent_rows_per_sec": 256.09, "rows_per_sec": 158.17, "timing_avg_sec": {"forward": 0.2362, "move": 0.0103, "post": 0.0064}}
2026-05-10 11:37:39.264
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 730, "elapsed_sec": 1414.155, "predictions": 222449, "recent_rows_per_sec": 190.46, "rows_per_sec": 157.3, "timing_avg_sec": {"forward": 0.2362, "move": 0.0103, "post": 0.0064}}
2026-05-10 11:37:26.741
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 720, "elapsed_sec": 1397.227, "predictions": 219225, "recent_rows_per_sec": 165.35, "rows_per_sec": 156.9, "timing_avg_sec": {"forward": 0.2361, "move": 0.0103, "post": 0.0063}}
2026-05-10 11:37:09.814
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 710, "elapsed_sec": 1377.572, "predictions": 215975, "recent_rows_per_sec": 169.92, "rows_per_sec": 156.78, "timing_avg_sec": {"forward": 0.2364, "move": 0.0104, "post": 0.0063}}
2026-05-10 11:36:50.158
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 700, "elapsed_sec": 1358.481, "predictions": 212731, "recent_rows_per_sec": 145.04, "rows_per_sec": 156.59, "timing_avg_sec": {"forward": 0.2362, "move": 0.0104, "post": 0.0063}}
2026-05-10 11:36:31.067
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 690, "elapsed_sec": 1336.473, "predictions": 209539, "recent_rows_per_sec": 147.34, "rows_per_sec": 156.79, "timing_avg_sec": {"forward": 0.2361, "move": 0.0103, "post": 0.0063}}
2026-05-10 11:36:09.060
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 680, "elapsed_sec": 1314.768, "predictions": 206341, "recent_rows_per_sec": 148.32, "rows_per_sec": 156.94, "timing_avg_sec": {"forward": 0.2355, "move": 0.0103, "post": 0.0063}}
2026-05-10 11:35:47.354
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 670, "elapsed_sec": 1293.456, "predictions": 203180, "recent_rows_per_sec": 145.83, "rows_per_sec": 157.08, "timing_avg_sec": {"forward": 0.2356, "move": 0.01, "post": 0.0063}}
2026-05-10 11:35:26.043
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 660, "elapsed_sec": 1271.848, "predictions": 200029, "recent_rows_per_sec": 177.97, "rows_per_sec": 157.27, "timing_avg_sec": {"forward": 0.2354, "move": 0.01, "post": 0.0063}}
2026-05-10 11:35:04.435
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 650, "elapsed_sec": 1254.177, "predictions": 196884, "recent_rows_per_sec": 146.84, "rows_per_sec": 156.98, "timing_avg_sec": {"forward": 0.2353, "move": 0.0101, "post": 0.0063}}
2026-05-10 11:34:46.763
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 640, "elapsed_sec": 1232.854, "predictions": 193753, "recent_rows_per_sec": 145.99, "rows_per_sec": 157.16, "timing_avg_sec": {"forward": 0.2358, "move": 0.0097, "post": 0.0063}}
2026-05-10 11:34:25.441
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 630, "elapsed_sec": 1211.065, "predictions": 190572, "recent_rows_per_sec": 161.27, "rows_per_sec": 157.36, "timing_avg_sec": {"forward": 0.2353, "move": 0.0098, "post": 0.0063}}
2026-05-10 11:34:03.651
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 620, "elapsed_sec": 1191.458, "predictions": 187410, "recent_rows_per_sec": 157.0, "rows_per_sec": 157.29, "timing_avg_sec": {"forward": 0.2351, "move": 0.0098, "post": 0.0063}}
2026-05-10 11:33:44.044
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 610, "elapsed_sec": 1171.548, "predictions": 184284, "recent_rows_per_sec": 146.82, "rows_per_sec": 157.3, "timing_avg_sec": {"forward": 0.235, "move": 0.01, "post": 0.0063}}
2026-05-10 11:33:24.134
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 600, "elapsed_sec": 1150.488, "predictions": 181192, "recent_rows_per_sec": 141.62, "rows_per_sec": 157.49, "timing_avg_sec": {"forward": 0.2347, "move": 0.0102, "post": 0.0063}}
2026-05-10 11:33:03.074
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 590, "elapsed_sec": 1128.569, "predictions": 178088, "recent_rows_per_sec": 176.75, "rows_per_sec": 157.8, "timing_avg_sec": {"forward": 0.2344, "move": 0.0102, "post": 0.0063}}
2026-05-10 11:32:41.156
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 580, "elapsed_sec": 1110.877, "predictions": 174961, "recent_rows_per_sec": 147.88, "rows_per_sec": 157.5, "timing_avg_sec": {"forward": 0.2342, "move": 0.0101, "post": 0.0063}}
2026-05-10 11:32:23.464
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 570, "elapsed_sec": 1090.063, "predictions": 171883, "recent_rows_per_sec": 160.93, "rows_per_sec": 157.68, "timing_avg_sec": {"forward": 0.2342, "move": 0.0102, "post": 0.0063}}
2026-05-10 11:32:02.649
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 560, "elapsed_sec": 1071.266, "predictions": 168858, "recent_rows_per_sec": 145.8, "rows_per_sec": 157.62, "timing_avg_sec": {"forward": 0.2342, "move": 0.0102, "post": 0.0063}}
2026-05-10 11:31:43.852
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 550, "elapsed_sec": 1050.67, "predictions": 165855, "recent_rows_per_sec": 144.87, "rows_per_sec": 157.86, "timing_avg_sec": {"forward": 0.2342, "move": 0.01, "post": 0.0063}}
2026-05-10 11:31:23.256
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 540, "elapsed_sec": 1029.858, "predictions": 162840, "recent_rows_per_sec": 153.68, "rows_per_sec": 158.12, "timing_avg_sec": {"forward": 0.2342, "move": 0.01, "post": 0.0063}}
2026-05-10 11:31:02.444
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 530, "elapsed_sec": 1009.882, "predictions": 159770, "recent_rows_per_sec": 182.48, "rows_per_sec": 158.21, "timing_avg_sec": {"forward": 0.2343, "move": 0.0101, "post": 0.0063}}
2026-05-10 11:30:42.468
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 520, "elapsed_sec": 992.74, "predictions": 156642, "recent_rows_per_sec": 154.53, "rows_per_sec": 157.79, "timing_avg_sec": {"forward": 0.2336, "move": 0.0101, "post": 0.0063}}
2026-05-10 11:30:25.326
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 510, "elapsed_sec": 972.867, "predictions": 153571, "recent_rows_per_sec": 160.91, "rows_per_sec": 157.85, "timing_avg_sec": {"forward": 0.2335, "move": 0.0103, "post": 0.0063}}
2026-05-10 11:30:05.453
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 500, "elapsed_sec": 953.683, "predictions": 150484, "recent_rows_per_sec": 157.69, "rows_per_sec": 157.79, "timing_avg_sec": {"forward": 0.2331, "move": 0.0103, "post": 0.0063}}
2026-05-10 11:29:46.269
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 490, "elapsed_sec": 934.252, "predictions": 147420, "recent_rows_per_sec": 171.2, "rows_per_sec": 157.79, "timing_avg_sec": {"forward": 0.233, "move": 0.0103, "post": 0.0062}}
2026-05-10 11:29:26.838
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 480, "elapsed_sec": 916.36, "predictions": 144357, "recent_rows_per_sec": 144.11, "rows_per_sec": 157.53, "timing_avg_sec": {"forward": 0.2327, "move": 0.0101, "post": 0.0062}}
2026-05-10 11:29:08.947
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 470, "elapsed_sec": 895.646, "predictions": 141372, "recent_rows_per_sec": 166.26, "rows_per_sec": 157.84, "timing_avg_sec": {"forward": 0.2328, "move": 0.0095, "post": 0.0062}}
2026-05-10 11:28:48.233
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 460, "elapsed_sec": 877.958, "predictions": 138431, "recent_rows_per_sec": 161.15, "rows_per_sec": 157.67, "timing_avg_sec": {"forward": 0.2328, "move": 0.0094, "post": 0.0062}}
2026-05-10 11:28:30.544
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 450, "elapsed_sec": 859.676, "predictions": 135485, "recent_rows_per_sec": 143.46, "rows_per_sec": 157.6, "timing_avg_sec": {"forward": 0.2329, "move": 0.0095, "post": 0.0062}}
2026-05-10 11:28:12.262
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 440, "elapsed_sec": 838.96, "predictions": 132513, "recent_rows_per_sec": 161.98, "rows_per_sec": 157.95, "timing_avg_sec": {"forward": 0.2325, "move": 0.0097, "post": 0.0062}}
2026-05-10 11:27:51.546
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 430, "elapsed_sec": 820.451, "predictions": 129515, "recent_rows_per_sec": 153.35, "rows_per_sec": 157.86, "timing_avg_sec": {"forward": 0.2324, "move": 0.0099, "post": 0.0062}}
2026-05-10 11:27:33.037
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 420, "elapsed_sec": 800.928, "predictions": 126521, "recent_rows_per_sec": 154.44, "rows_per_sec": 157.97, "timing_avg_sec": {"forward": 0.2324, "move": 0.0101, "post": 0.0062}}
2026-05-10 11:27:13.514
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 410, "elapsed_sec": 781.541, "predictions": 123527, "recent_rows_per_sec": 160.79, "rows_per_sec": 158.06, "timing_avg_sec": {"forward": 0.2324, "move": 0.0099, "post": 0.0062}}
2026-05-10 11:26:54.127
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 400, "elapsed_sec": 763.058, "predictions": 120555, "recent_rows_per_sec": 139.74, "rows_per_sec": 157.99, "timing_avg_sec": {"forward": 0.2323, "move": 0.0101, "post": 0.0062}}
2026-05-10 11:26:35.644
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 390, "elapsed_sec": 741.653, "predictions": 117564, "recent_rows_per_sec": 146.24, "rows_per_sec": 158.52, "timing_avg_sec": {"forward": 0.2317, "move": 0.0104, "post": 0.0061}}
2026-05-10 11:26:14.239
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 380, "elapsed_sec": 721.556, "predictions": 114625, "recent_rows_per_sec": 148.67, "rows_per_sec": 158.86, "timing_avg_sec": {"forward": 0.231, "move": 0.0106, "post": 0.0061}}
2026-05-10 11:25:54.142
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 370, "elapsed_sec": 702.036, "predictions": 111723, "recent_rows_per_sec": 147.74, "rows_per_sec": 159.14, "timing_avg_sec": {"forward": 0.2308, "move": 0.0109, "post": 0.0061}}
2026-05-10 11:25:34.622
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 360, "elapsed_sec": 682.259, "predictions": 108801, "recent_rows_per_sec": 149.45, "rows_per_sec": 159.47, "timing_avg_sec": {"forward": 0.2309, "move": 0.0109, "post": 0.0061}}
2026-05-10 11:25:14.845
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 350, "elapsed_sec": 662.727, "predictions": 105882, "recent_rows_per_sec": 137.99, "rows_per_sec": 159.77, "timing_avg_sec": {"forward": 0.2306, "move": 0.0112, "post": 0.0061}}
2026-05-10 11:24:55.313
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 340, "elapsed_sec": 641.674, "predictions": 102977, "recent_rows_per_sec": 150.67, "rows_per_sec": 160.48, "timing_avg_sec": {"forward": 0.2308, "move": 0.0112, "post": 0.0061}}
2026-05-10 11:24:34.261
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 330, "elapsed_sec": 622.148, "predictions": 100035, "recent_rows_per_sec": 155.36, "rows_per_sec": 160.79, "timing_avg_sec": {"forward": 0.2298, "move": 0.0116, "post": 0.006}}
2026-05-10 11:24:14.734
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 320, "elapsed_sec": 603.038, "predictions": 97066, "recent_rows_per_sec": 137.59, "rows_per_sec": 160.96, "timing_avg_sec": {"forward": 0.2298, "move": 0.0116, "post": 0.006}}
2026-05-10 11:23:55.624
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 310, "elapsed_sec": 581.161, "predictions": 94056, "recent_rows_per_sec": 141.46, "rows_per_sec": 161.84, "timing_avg_sec": {"forward": 0.2294, "move": 0.0117, "post": 0.006}}
2026-05-10 11:23:33.748
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 300, "elapsed_sec": 559.551, "predictions": 90999, "recent_rows_per_sec": 161.81, "rows_per_sec": 162.63, "timing_avg_sec": {"forward": 0.2285, "move": 0.012, "post": 0.006}}
2026-05-10 11:23:12.137
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 290, "elapsed_sec": 540.461, "predictions": 87910, "recent_rows_per_sec": 171.18, "rows_per_sec": 162.66, "timing_avg_sec": {"forward": 0.2285, "move": 0.0118, "post": 0.006}}
2026-05-10 11:22:53.047
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 280, "elapsed_sec": 522.416, "predictions": 84821, "recent_rows_per_sec": 155.16, "rows_per_sec": 162.36, "timing_avg_sec": {"forward": 0.2284, "move": 0.0118, "post": 0.006}}
2026-05-10 11:22:35.002
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 270, "elapsed_sec": 502.72, "predictions": 81765, "recent_rows_per_sec": 164.34, "rows_per_sec": 162.65, "timing_avg_sec": {"forward": 0.2281, "move": 0.0119, "post": 0.006}}
2026-05-10 11:22:15.306
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 260, "elapsed_sec": 484.076, "predictions": 78701, "recent_rows_per_sec": 158.76, "rows_per_sec": 162.58, "timing_avg_sec": {"forward": 0.2289, "move": 0.0109, "post": 0.006}}
2026-05-10 11:21:56.662
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 250, "elapsed_sec": 464.952, "predictions": 75665, "recent_rows_per_sec": 159.59, "rows_per_sec": 162.74, "timing_avg_sec": {"forward": 0.2292, "move": 0.0109, "post": 0.006}}
2026-05-10 11:21:37.538
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 240, "elapsed_sec": 446.029, "predictions": 72645, "recent_rows_per_sec": 129.78, "rows_per_sec": 162.87, "timing_avg_sec": {"forward": 0.2296, "move": 0.0105, "post": 0.006}}
2026-05-10 11:21:18.615
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 230, "elapsed_sec": 422.704, "predictions": 69618, "recent_rows_per_sec": 181.98, "rows_per_sec": 164.7, "timing_avg_sec": {"forward": 0.2299, "move": 0.0105, "post": 0.006}}
2026-05-10 11:20:55.290
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 220, "elapsed_sec": 406.268, "predictions": 66627, "recent_rows_per_sec": 143.01, "rows_per_sec": 164.0, "timing_avg_sec": {"forward": 0.2301, "move": 0.0102, "post": 0.006}}
2026-05-10 11:20:38.854
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 210, "elapsed_sec": 385.654, "predictions": 63679, "recent_rows_per_sec": 145.55, "rows_per_sec": 165.12, "timing_avg_sec": {"forward": 0.2298, "move": 0.0098, "post": 0.006}}
2026-05-10 11:20:18.241
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 200, "elapsed_sec": 365.421, "predictions": 60734, "recent_rows_per_sec": 131.53, "rows_per_sec": 166.2, "timing_avg_sec": {"forward": 0.2311, "move": 0.0088, "post": 0.006}}
2026-05-10 11:19:58.008
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 190, "elapsed_sec": 342.932, "predictions": 57776, "recent_rows_per_sec": 175.18, "rows_per_sec": 168.48, "timing_avg_sec": {"forward": 0.2315, "move": 0.0082, "post": 0.006}}
2026-05-10 11:19:35.519
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 180, "elapsed_sec": 326.042, "predictions": 54817, "recent_rows_per_sec": 132.6, "rows_per_sec": 168.13, "timing_avg_sec": {"forward": 0.2342, "move": 0.0071, "post": 0.006}}
2026-05-10 11:19:18.628
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 170, "elapsed_sec": 303.757, "predictions": 51862, "recent_rows_per_sec": 204.06, "rows_per_sec": 170.74, "timing_avg_sec": {"forward": 0.2349, "move": 0.0069, "post": 0.0061}}
2026-05-10 11:18:56.343
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 160, "elapsed_sec": 289.247, "predictions": 48901, "recent_rows_per_sec": 156.47, "rows_per_sec": 169.06, "timing_avg_sec": {"forward": 0.2356, "move": 0.0073, "post": 0.0061}}
2026-05-10 11:18:41.833
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 150, "elapsed_sec": 270.068, "predictions": 45900, "recent_rows_per_sec": 155.0, "rows_per_sec": 169.96, "timing_avg_sec": {"forward": 0.2355, "move": 0.0071, "post": 0.0061}}
2026-05-10 11:18:22.654
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 140, "elapsed_sec": 250.468, "predictions": 42862, "recent_rows_per_sec": 208.42, "rows_per_sec": 171.13, "timing_avg_sec": {"forward": 0.2359, "move": 0.0076, "post": 0.0061}}
2026-05-10 11:18:03.054
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 130, "elapsed_sec": 236.049, "predictions": 39857, "recent_rows_per_sec": 184.66, "rows_per_sec": 168.85, "timing_avg_sec": {"forward": 0.2366, "move": 0.0081, "post": 0.0062}}
2026-05-10 11:17:48.635
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 120, "elapsed_sec": 219.766, "predictions": 36850, "recent_rows_per_sec": 159.71, "rows_per_sec": 167.68, "timing_avg_sec": {"forward": 0.2386, "move": 0.008, "post": 0.0062}}
2026-05-10 11:17:32.352
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 110, "elapsed_sec": 200.756, "predictions": 33814, "recent_rows_per_sec": 198.4, "rows_per_sec": 168.43, "timing_avg_sec": {"forward": 0.2402, "move": 0.0069, "post": 0.0064}}
2026-05-10 11:17:13.342
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 100, "elapsed_sec": 185.353, "predictions": 30758, "recent_rows_per_sec": 155.21, "rows_per_sec": 165.94, "timing_avg_sec": {"forward": 0.2415, "move": 0.0056, "post": 0.0064}}
2026-05-10 11:16:57.939
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 90, "elapsed_sec": 165.74, "predictions": 27714, "recent_rows_per_sec": 171.39, "rows_per_sec": 167.21, "timing_avg_sec": {"forward": 0.2428, "move": 0.0061, "post": 0.0065}}
2026-05-10 11:16:38.327
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 80, "elapsed_sec": 147.927, "predictions": 24661, "recent_rows_per_sec": 164.13, "rows_per_sec": 166.71, "timing_avg_sec": {"forward": 0.241, "move": 0.0068, "post": 0.0066}}
2026-05-10 11:16:20.514
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 70, "elapsed_sec": 129.338, "predictions": 21610, "recent_rows_per_sec": 148.82, "rows_per_sec": 167.08, "timing_avg_sec": {"forward": 0.2399, "move": 0.0077, "post": 0.0066}}
2026-05-10 11:16:01.924
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 60, "elapsed_sec": 108.837, "predictions": 18559, "recent_rows_per_sec": 211.81, "rows_per_sec": 170.52, "timing_avg_sec": {"forward": 0.2414, "move": 0.0072, "post": 0.0066}}
2026-05-10 11:15:41.423
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 50, "elapsed_sec": 94.38, "predictions": 15497, "recent_rows_per_sec": 151.1, "rows_per_sec": 164.2, "timing_avg_sec": {"forward": 0.2425, "move": 0.0066, "post": 0.0065}}
2026-05-10 11:15:26.966
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 40, "elapsed_sec": 73.937, "predictions": 12408, "recent_rows_per_sec": 192.53, "rows_per_sec": 167.82, "timing_avg_sec": {"forward": 0.243, "move": 0.0081, "post": 0.0065}}
2026-05-10 11:15:06.523
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 30, "elapsed_sec": 57.856, "predictions": 9312, "recent_rows_per_sec": 212.02, "rows_per_sec": 160.95, "timing_avg_sec": {"forward": 0.247, "move": 0.0106, "post": 0.0065}}
2026-05-10 11:14:50.442
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 20, "elapsed_sec": 43.352, "predictions": 6237, "recent_rows_per_sec": 191.17, "rows_per_sec": 143.87, "timing_avg_sec": {"forward": 0.2567, "move": 0.006, "post": 0.0064}}
2026-05-10 11:14:35.939
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 10, "elapsed_sec": 27.142, "predictions": 3138, "recent_rows_per_sec": 115.62, "rows_per_sec": 115.62, "timing_avg_sec": {"forward": 0.2693, "move": 0.0113, "post": 0.0061}}
2026-05-10 11:14:19.728
PLATFORM_FIRST_BATCH {"batch_predictions": 314, "sample_scores": [0.00370789, 0.07470703, 0.03955078, 0.17675781, 0.44921875], "sample_user_ids": ["683974", "1334888", "51348", "1184407", "8854"], "sanitize_counts": {"nan_replaced": 0, "neginf_replaced": 0, "posinf_replaced": 0}, "timing_sec": {"forward": 0.4431, "move": 0.107, "post": 0.0062}}
2026-05-10 11:14:04.647
2026-05-10 03:14:01,287 - INFO - NumExpr defaulting to 16 threads.
2026-05-10 11:14:01.287
2026-05-10 03:14:01,287 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-10 11:14:01.287
2026-05-10 03:14:01,287 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-10 11:14:01.287
PLATFORM_AMP_CONFIG {"cuda_bf16_supported": true, "enabled": true, "mode": "bf16", "requested": "bf16"}
2026-05-10 11:14:00.907
2026-05-10 03:14:00,907 - INFO - Starting inference...
2026-05-10 11:14:00.907
PLATFORM_MODEL_READY {"device": "cuda", "num_ns": 8, "num_parameters": 239931649}
2026-05-10 11:14:00.907
2026-05-10 03:14:00,906 - INFO - Model loaded successfully
2026-05-10 11:14:00.907
2026-05-10 03:13:57,667 - INFO - Loading checkpoint from /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt
2026-05-10 11:13:57.667
2026-05-10 03:13:55,225 - INFO - emb_skip_threshold=1000000: seq_c skipped 3/11 features
2026-05-10 11:13:55.225
2026-05-10 03:13:55,225 - INFO - emb_skip_threshold=1000000: seq_b skipped 1/13 features
2026-05-10 11:13:55.225
2026-05-10 03:13:53,144 - INFO - RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0
2026-05-10 11:13:53.144
2026-05-10 03:13:53,126 - INFO - RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1
2026-05-10 11:13:53.126
2026-05-10 03:13:53,102 - INFO - Building PCVRHyFormer with cfg: {'d_model': 64, 'emb_dim': 64, 'num_queries': 2, 'num_hyformer_blocks': 2, 'num_heads': 4, 'seq_encoder_type': 'transformer', 'hidden_mult': 4, 'dropout_rate': 0.01, 'seq_top_k': 50, 'seq_causal': False, 'action_num': 1, 'num_time_buckets': 64, 'rank_mixer_mode': 'full', 'use_rope': False, 'rope_base': 10000.0, 'emb_skip_threshold': 1000000, 'seq_id_threshold': 10000, 'ns_tokenizer_type': 'rankmixer', 'user_ns_tokens': 5, 'item_ns_tokens': 2, 'use_wide_dense_tokens': False, 'wide_dense_threshold': 32, 'dense_scalar_transform': 'none', 'xdomain_dense_dim': 0, 'use_xdomain_features': False, 'use_inter_time_buckets': False, 'use_continuous_time': False, 'high_cardinality_mode': 'zero', 'hash_num_buckets': 1048576, 'hash_num_hashes': 2, 'use_target_aware_attn': False, 'use_cross_domain_attn': False, 'seq_topk_mode': 'recent', 'ffn_type': 'gelu', 'norm_type': 'layernorm', 'random_id_mask_prob': 0.0, 'legacy_user_dense_proj': True, 'legacy_mixer_ffn_names': True}
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - INFO - No NS groups JSON found, using default: each feature as one group
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'legacy_mixer_ffn_names', using fallback = False
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'legacy_user_dense_proj', using fallback = False
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'random_id_mask_prob', using fallback = 0.0
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'norm_type', using fallback = layernorm
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'ffn_type', using fallback = gelu
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'seq_topk_mode', using fallback = recent
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'use_cross_domain_attn', using fallback = False
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'use_target_aware_attn', using fallback = False
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'hash_num_hashes', using fallback = 2
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'hash_num_buckets', using fallback = 1048576
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'high_cardinality_mode', using fallback = zero
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'use_continuous_time', using fallback = False
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'use_inter_time_buckets', using fallback = False
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'use_xdomain_features', using fallback = False
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'xdomain_dense_dim', using fallback = 0
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'dense_scalar_transform', using fallback = none
2026-05-10 11:13:53.102
2026-05-10 03:13:53,102 - WARNING - train_config missing 'wide_dense_threshold', using fallback = 32
2026-05-10 11:13:53.102
2026-05-10 03:13:53,101 - WARNING - train_config missing 'use_wide_dense_tokens', using fallback = False
2026-05-10 11:13:53.102
PLATFORM_DATASET_CONFIG {"batch_size": 2048, "item_dense_dim": 0, "item_int_features": 14, "num_row_groups": 1000, "num_rows_estimate": 310000, "seq_domains": ["seq_a", "seq_b", "seq_c", "seq_d"], "seq_max_lens": {"seq_a": 256, "seq_b": 256, "seq_c": 512, "seq_d": 512}, "user_dense_dim": 918, "user_int_features": 46}
2026-05-10 11:13:53.102
2026-05-10 03:13:53,101 - INFO - Total test samples: 310000
2026-05-10 11:13:53.101
2026-05-10 03:13:53,101 - INFO - PCVRParquetDataset: 310000 rows from 1000 file(s), batch_size=2048, buffer_batches=0, shuffle=False
2026-05-10 11:13:53.101
PLATFORM_INFER_CONFIG {"amp_dtype": "bf16", "batch_size": 2048, "checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "device": "cuda", "eval_data_dir": "/data_ams/academic_infer_data/", "num_workers": 0, "progress_every": 10, "result_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/77616/results", "schema_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "seq_max_lens": {"seq_a": 256, "seq_b": 256, "seq_c": 512, "seq_d": 512}, "train_config": {"d_model": 64, "emb_skip_threshold": 1000000, "high_cardinality_mode": null, "num_heads": 4, "num_hyformer_blocks": 2, "num_queries": 2, "seq_encoder_type": "transformer", "train_batch_size": 256}, "use_amp": true}
2026-05-10 11:13:52.594
PLATFORM_CHECKPOINT_STATUS {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "model_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt", "size_bytes": 959871995}, "schema_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "size_bytes": 5560}, "train_config_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/train_config.json", "size_bytes": 1638}}
2026-05-10 11:13:52.594
2026-05-10 03:13:52,594 - INFO - seq_max_lens: {'seq_a': 256, 'seq_b': 256, 'seq_c': 512, 'seq_d': 512}
2026-05-10 11:13:52.594
2026-05-10 03:13:52,594 - INFO - Loaded train_config from /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/train_config.json
2026-05-10 11:13:52.594
2026-05-10 03:13:52,591 - INFO - Using schema: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json
2026-05-10 11:13:52.591
PLATFORM_INFER_START {"cuda_available": true, "cwd": "/workspace", "time": "2026-05-10 03:13:52", "torch_version": "2.7.1+cu126"}
2026-05-10 11:13:52.589
====== Inferring ======
2026-05-10 11:13:51.315
=================================
2026-05-10 11:13:50.905
Working Dir: /workspace
2026-05-10 11:13:50.905
Environment: competition
2026-05-10 11:13:50.905
GPU Count: 1
2026-05-10 11:13:50.905
GPU Available: True
2026-05-10 11:13:49.460
PyTorch: 2.7.1+cu126
2026-05-10 11:13:48.021
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-10 11:13:45.686
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-10 11:13:45.686
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-10 11:13:45.686
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-10 11:13:45.686
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-10 11:13:45.682
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-10 11:13:45.682
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-10 11:13:45.682
Python: Python 3.10.20
2026-05-10 11:13:45.321
CUDA: 12.6.77
2026-05-10 11:13:45.316
=== Competition Environment Ready ===
2026-05-10 11:13:45.306
Complete setting network policy rules.
2026-05-10 11:13:41.787
Complete setting taiji user.
2026-05-10 11:13:41.653
Truncated to last 1000 lines.