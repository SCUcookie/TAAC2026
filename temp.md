====== Score Finished ======
2026-05-11 14:45:48.540
====== Scoring ======
2026-05-11 14:45:45.694
=================================
2026-05-11 14:45:45.321
Working Dir: /workspace
2026-05-11 14:45:45.321
Environment: competition
2026-05-11 14:45:45.321
GPU Count: 1
2026-05-11 14:45:45.321
GPU Available: True
2026-05-11 14:45:43.892
PyTorch: 2.7.1+cu126
2026-05-11 14:45:42.442
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-11 14:45:40.135
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-11 14:45:40.135
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-11 14:45:40.135
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-11 14:45:40.135
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-11 14:45:40.133
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-11 14:45:40.133
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-11 14:45:40.133
Python: Python 3.10.20
2026-05-11 14:45:39.787
CUDA: 12.6.77
2026-05-11 14:45:39.782
=== Competition Environment Ready ===
2026-05-11 14:45:39.774
Complete setting network policy rules.
2026-05-11 14:45:36.214
Complete setting taiji user.
2026-05-11 14:45:35.801
====== Infer Finished ======
2026-05-11 14:40:56.347
2026-05-11 06:40:55,484 - INFO - Saved status.json to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/82255/results/status.json
2026-05-11 14:40:55.485
2026-05-11 06:40:54,808 - INFO - Saved 310000 predictions to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/82255/results/predictions.json
2026-05-11 14:40:54.808
PLATFORM_INFER_SUMMARY {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model", "elapsed_sec": 1006.692, "num_predictions": 310000, "num_unique_user_ids": 310000, "output_shape": "predictions_dict_user_id_to_float", "result_files": {"csv": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/82255/results/submission.csv", "size_bytes": 5643992}, "json": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/82255/results/predictions.json", "size_bytes": 7851842}, "scores": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/82255/results/scores.npy", "size_bytes": 1240128}}, "result_json": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/82255/results/predictions.json", "score_stats": {"count": 310000, "inf_count": 0, "max": 0.9765625, "mean": 0.10907357, "min": 0.00062561, "nan_count": 0, "p01": 0.00068665, "p50": 0.05664062, "p99": 0.828125, "std": 0.14828748}, "scores_npy": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/82255/results/scores.npy", "submission_csv": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/82255/results/submission.csv"}
2026-05-11 14:40:54.808
2026-05-11 06:40:54,258 - INFO - Inference complete: 310000 predictions
2026-05-11 14:40:54.258
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 1000, "elapsed_sec": 1006.166, "predictions": 310000, "recent_rows_per_sec": 312.36, "row_group_batches": 1000, "rows_per_sec": 308.1, "timing_avg_sec": {"forward": 0.9852, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:40:54.258
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 990, "elapsed_sec": 996.091, "predictions": 306853, "recent_rows_per_sec": 314.75, "row_group_batches": 990, "rows_per_sec": 308.06, "timing_avg_sec": {"forward": 0.9851, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:40:44.183
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 980, "elapsed_sec": 985.915, "predictions": 303650, "recent_rows_per_sec": 314.65, "row_group_batches": 980, "rows_per_sec": 307.99, "timing_avg_sec": {"forward": 0.9848, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:40:34.007
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 970, "elapsed_sec": 975.668, "predictions": 300426, "recent_rows_per_sec": 317.38, "row_group_batches": 970, "rows_per_sec": 307.92, "timing_avg_sec": {"forward": 0.9844, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:40:23.760
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 960, "elapsed_sec": 965.431, "predictions": 297177, "recent_rows_per_sec": 315.44, "row_group_batches": 960, "rows_per_sec": 307.82, "timing_avg_sec": {"forward": 0.9841, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:40:13.523
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 950, "elapsed_sec": 955.084, "predictions": 293913, "recent_rows_per_sec": 315.18, "row_group_batches": 950, "rows_per_sec": 307.74, "timing_avg_sec": {"forward": 0.9836, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:40:03.176
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 940, "elapsed_sec": 944.909, "predictions": 290706, "recent_rows_per_sec": 313.92, "row_group_batches": 940, "rows_per_sec": 307.65, "timing_avg_sec": {"forward": 0.9833, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:39:53.001
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 930, "elapsed_sec": 934.649, "predictions": 287485, "recent_rows_per_sec": 317.96, "row_group_batches": 930, "rows_per_sec": 307.59, "timing_avg_sec": {"forward": 0.9829, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:39:42.740
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 920, "elapsed_sec": 924.386, "predictions": 284222, "recent_rows_per_sec": 318.98, "row_group_batches": 920, "rows_per_sec": 307.47, "timing_avg_sec": {"forward": 0.9825, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:39:32.478
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 910, "elapsed_sec": 914.147, "predictions": 280956, "recent_rows_per_sec": 319.05, "row_group_batches": 910, "rows_per_sec": 307.34, "timing_avg_sec": {"forward": 0.9821, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:39:22.239
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 900, "elapsed_sec": 903.998, "predictions": 277718, "recent_rows_per_sec": 316.36, "row_group_batches": 900, "rows_per_sec": 307.21, "timing_avg_sec": {"forward": 0.9818, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:39:12.090
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 890, "elapsed_sec": 893.779, "predictions": 274485, "recent_rows_per_sec": 314.09, "row_group_batches": 890, "rows_per_sec": 307.11, "timing_avg_sec": {"forward": 0.9814, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:39:01.870
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 880, "elapsed_sec": 883.486, "predictions": 271252, "recent_rows_per_sec": 316.97, "row_group_batches": 880, "rows_per_sec": 307.02, "timing_avg_sec": {"forward": 0.9809, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:38:51.577
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 870, "elapsed_sec": 873.201, "predictions": 267992, "recent_rows_per_sec": 315.97, "row_group_batches": 870, "rows_per_sec": 306.91, "timing_avg_sec": {"forward": 0.9804, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:38:41.292
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 860, "elapsed_sec": 862.703, "predictions": 264675, "recent_rows_per_sec": 317.26, "row_group_batches": 860, "rows_per_sec": 306.8, "timing_avg_sec": {"forward": 0.9797, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:38:30.794
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 850, "elapsed_sec": 852.301, "predictions": 261375, "recent_rows_per_sec": 316.54, "row_group_batches": 850, "rows_per_sec": 306.67, "timing_avg_sec": {"forward": 0.979, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:38:20.393
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 840, "elapsed_sec": 842.028, "predictions": 258123, "recent_rows_per_sec": 314.34, "row_group_batches": 840, "rows_per_sec": 306.55, "timing_avg_sec": {"forward": 0.9785, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:38:10.119
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 830, "elapsed_sec": 831.777, "predictions": 254901, "recent_rows_per_sec": 315.67, "row_group_batches": 830, "rows_per_sec": 306.45, "timing_avg_sec": {"forward": 0.978, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:37:59.869
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 820, "elapsed_sec": 821.555, "predictions": 251674, "recent_rows_per_sec": 314.85, "row_group_batches": 820, "rows_per_sec": 306.34, "timing_avg_sec": {"forward": 0.9775, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:37:49.647
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 810, "elapsed_sec": 811.264, "predictions": 248434, "recent_rows_per_sec": 316.02, "row_group_batches": 810, "rows_per_sec": 306.23, "timing_avg_sec": {"forward": 0.977, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:37:39.356
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 800, "elapsed_sec": 801.034, "predictions": 245201, "recent_rows_per_sec": 313.58, "row_group_batches": 800, "rows_per_sec": 306.11, "timing_avg_sec": {"forward": 0.9765, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:37:29.126
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 790, "elapsed_sec": 790.826, "predictions": 242000, "recent_rows_per_sec": 319.14, "row_group_batches": 790, "rows_per_sec": 306.01, "timing_avg_sec": {"forward": 0.976, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:37:18.918
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 780, "elapsed_sec": 780.63, "predictions": 238746, "recent_rows_per_sec": 319.24, "row_group_batches": 780, "rows_per_sec": 305.84, "timing_avg_sec": {"forward": 0.9755, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:37:08.722
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 770, "elapsed_sec": 770.496, "predictions": 235511, "recent_rows_per_sec": 321.7, "row_group_batches": 770, "rows_per_sec": 305.66, "timing_avg_sec": {"forward": 0.9751, "merge": 0.0, "move": 0.0005, "post": 0.0042}}
2026-05-11 14:36:58.588
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 760, "elapsed_sec": 760.273, "predictions": 232222, "recent_rows_per_sec": 319.51, "row_group_batches": 760, "rows_per_sec": 305.45, "timing_avg_sec": {"forward": 0.9745, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:36:48.364
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 750, "elapsed_sec": 749.963, "predictions": 228928, "recent_rows_per_sec": 317.81, "row_group_batches": 750, "rows_per_sec": 305.25, "timing_avg_sec": {"forward": 0.9738, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:36:38.055
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 740, "elapsed_sec": 739.668, "predictions": 225656, "recent_rows_per_sec": 314.57, "row_group_batches": 740, "rows_per_sec": 305.08, "timing_avg_sec": {"forward": 0.9731, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:36:27.760
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 730, "elapsed_sec": 729.473, "predictions": 222449, "recent_rows_per_sec": 316.17, "row_group_batches": 730, "rows_per_sec": 304.94, "timing_avg_sec": {"forward": 0.9726, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:36:17.565
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 720, "elapsed_sec": 719.276, "predictions": 219225, "recent_rows_per_sec": 314.94, "row_group_batches": 720, "rows_per_sec": 304.79, "timing_avg_sec": {"forward": 0.972, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:36:07.368
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 710, "elapsed_sec": 708.956, "predictions": 215975, "recent_rows_per_sec": 316.67, "row_group_batches": 710, "rows_per_sec": 304.64, "timing_avg_sec": {"forward": 0.9712, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:35:57.048
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 700, "elapsed_sec": 698.712, "predictions": 212731, "recent_rows_per_sec": 313.36, "row_group_batches": 700, "rows_per_sec": 304.46, "timing_avg_sec": {"forward": 0.9705, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:35:46.804
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 690, "elapsed_sec": 688.526, "predictions": 209539, "recent_rows_per_sec": 314.45, "row_group_batches": 690, "rows_per_sec": 304.33, "timing_avg_sec": {"forward": 0.9699, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:35:36.617
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 680, "elapsed_sec": 678.355, "predictions": 206341, "recent_rows_per_sec": 312.12, "row_group_batches": 680, "rows_per_sec": 304.18, "timing_avg_sec": {"forward": 0.9693, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:35:26.447
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 670, "elapsed_sec": 668.228, "predictions": 203180, "recent_rows_per_sec": 311.19, "row_group_batches": 670, "rows_per_sec": 304.06, "timing_avg_sec": {"forward": 0.9687, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:35:16.320
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 660, "elapsed_sec": 658.102, "predictions": 200029, "recent_rows_per_sec": 312.98, "row_group_batches": 660, "rows_per_sec": 303.95, "timing_avg_sec": {"forward": 0.9681, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:35:06.194
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 650, "elapsed_sec": 648.054, "predictions": 196884, "recent_rows_per_sec": 313.67, "row_group_batches": 650, "rows_per_sec": 303.81, "timing_avg_sec": {"forward": 0.9677, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:34:56.146
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 640, "elapsed_sec": 638.072, "predictions": 193753, "recent_rows_per_sec": 314.68, "row_group_batches": 640, "rows_per_sec": 303.65, "timing_avg_sec": {"forward": 0.9673, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:34:46.164
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 630, "elapsed_sec": 627.964, "predictions": 190572, "recent_rows_per_sec": 317.84, "row_group_batches": 630, "rows_per_sec": 303.48, "timing_avg_sec": {"forward": 0.9667, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:34:36.055
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 620, "elapsed_sec": 618.015, "predictions": 187410, "recent_rows_per_sec": 313.1, "row_group_batches": 620, "rows_per_sec": 303.24, "timing_avg_sec": {"forward": 0.9663, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:34:26.107
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 610, "elapsed_sec": 608.031, "predictions": 184284, "recent_rows_per_sec": 313.36, "row_group_batches": 610, "rows_per_sec": 303.08, "timing_avg_sec": {"forward": 0.9658, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:34:16.123
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 600, "elapsed_sec": 598.164, "predictions": 181192, "recent_rows_per_sec": 314.05, "row_group_batches": 600, "rows_per_sec": 302.91, "timing_avg_sec": {"forward": 0.9656, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:34:06.256
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 590, "elapsed_sec": 588.28, "predictions": 178088, "recent_rows_per_sec": 312.73, "row_group_batches": 590, "rows_per_sec": 302.73, "timing_avg_sec": {"forward": 0.9653, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:33:56.372
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 580, "elapsed_sec": 578.281, "predictions": 174961, "recent_rows_per_sec": 312.5, "row_group_batches": 580, "rows_per_sec": 302.55, "timing_avg_sec": {"forward": 0.9648, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:33:46.373
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 570, "elapsed_sec": 568.432, "predictions": 171883, "recent_rows_per_sec": 310.93, "row_group_batches": 570, "rows_per_sec": 302.38, "timing_avg_sec": {"forward": 0.9645, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:33:36.523
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 560, "elapsed_sec": 558.703, "predictions": 168858, "recent_rows_per_sec": 314.81, "row_group_batches": 560, "rows_per_sec": 302.23, "timing_avg_sec": {"forward": 0.9645, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:33:26.794
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 550, "elapsed_sec": 549.164, "predictions": 165855, "recent_rows_per_sec": 310.68, "row_group_batches": 550, "rows_per_sec": 302.01, "timing_avg_sec": {"forward": 0.9648, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:33:17.255
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 540, "elapsed_sec": 539.459, "predictions": 162840, "recent_rows_per_sec": 311.83, "row_group_batches": 540, "rows_per_sec": 301.86, "timing_avg_sec": {"forward": 0.9648, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:33:07.551
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 530, "elapsed_sec": 529.614, "predictions": 159770, "recent_rows_per_sec": 311.96, "row_group_batches": 530, "rows_per_sec": 301.67, "timing_avg_sec": {"forward": 0.9645, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:32:57.706
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 520, "elapsed_sec": 519.587, "predictions": 156642, "recent_rows_per_sec": 312.63, "row_group_batches": 520, "rows_per_sec": 301.47, "timing_avg_sec": {"forward": 0.9639, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:32:47.679
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 510, "elapsed_sec": 509.764, "predictions": 153571, "recent_rows_per_sec": 316.06, "row_group_batches": 510, "rows_per_sec": 301.26, "timing_avg_sec": {"forward": 0.9636, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:32:37.856
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 500, "elapsed_sec": 499.997, "predictions": 150484, "recent_rows_per_sec": 312.27, "row_group_batches": 500, "rows_per_sec": 300.97, "timing_avg_sec": {"forward": 0.9635, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:32:28.088
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 490, "elapsed_sec": 490.185, "predictions": 147420, "recent_rows_per_sec": 314.59, "row_group_batches": 490, "rows_per_sec": 300.74, "timing_avg_sec": {"forward": 0.9632, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:32:18.276
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 480, "elapsed_sec": 480.448, "predictions": 144357, "recent_rows_per_sec": 316.57, "row_group_batches": 480, "rows_per_sec": 300.46, "timing_avg_sec": {"forward": 0.9631, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:32:08.540
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 470, "elapsed_sec": 471.019, "predictions": 141372, "recent_rows_per_sec": 308.89, "row_group_batches": 470, "rows_per_sec": 300.14, "timing_avg_sec": {"forward": 0.9637, "merge": 0.0, "move": 0.0005, "post": 0.0039}}
2026-05-11 14:31:59.111
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 460, "elapsed_sec": 461.498, "predictions": 138431, "recent_rows_per_sec": 307.83, "row_group_batches": 460, "rows_per_sec": 299.96, "timing_avg_sec": {"forward": 0.9641, "merge": 0.0, "move": 0.0005, "post": 0.0039}}
2026-05-11 14:31:49.589
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 450, "elapsed_sec": 451.927, "predictions": 135485, "recent_rows_per_sec": 307.43, "row_group_batches": 450, "rows_per_sec": 299.79, "timing_avg_sec": {"forward": 0.9644, "merge": 0.0, "move": 0.0005, "post": 0.0039}}
2026-05-11 14:31:40.019
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 440, "elapsed_sec": 442.26, "predictions": 132513, "recent_rows_per_sec": 308.12, "row_group_batches": 440, "rows_per_sec": 299.63, "timing_avg_sec": {"forward": 0.9644, "merge": 0.0, "move": 0.0005, "post": 0.0039}}
2026-05-11 14:31:30.352
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 430, "elapsed_sec": 432.53, "predictions": 129515, "recent_rows_per_sec": 308.27, "row_group_batches": 430, "rows_per_sec": 299.44, "timing_avg_sec": {"forward": 0.9643, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:31:20.622
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 420, "elapsed_sec": 422.818, "predictions": 126521, "recent_rows_per_sec": 308.17, "row_group_batches": 420, "rows_per_sec": 299.23, "timing_avg_sec": {"forward": 0.9642, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:31:10.910
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 410, "elapsed_sec": 413.102, "predictions": 123527, "recent_rows_per_sec": 307.27, "row_group_batches": 410, "rows_per_sec": 299.02, "timing_avg_sec": {"forward": 0.9641, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:31:01.194
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 400, "elapsed_sec": 403.43, "predictions": 120555, "recent_rows_per_sec": 308.79, "row_group_batches": 400, "rows_per_sec": 298.82, "timing_avg_sec": {"forward": 0.9642, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:30:51.522
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 390, "elapsed_sec": 393.744, "predictions": 117564, "recent_rows_per_sec": 308.39, "row_group_batches": 390, "rows_per_sec": 298.58, "timing_avg_sec": {"forward": 0.9642, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:30:41.836
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 380, "elapsed_sec": 384.214, "predictions": 114625, "recent_rows_per_sec": 308.97, "row_group_batches": 380, "rows_per_sec": 298.34, "timing_avg_sec": {"forward": 0.9646, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:30:32.306
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 370, "elapsed_sec": 374.821, "predictions": 111723, "recent_rows_per_sec": 305.69, "row_group_batches": 370, "rows_per_sec": 298.07, "timing_avg_sec": {"forward": 0.9655, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:30:22.913
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 360, "elapsed_sec": 365.263, "predictions": 108801, "recent_rows_per_sec": 308.78, "row_group_batches": 360, "rows_per_sec": 297.87, "timing_avg_sec": {"forward": 0.9658, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:30:13.354
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 350, "elapsed_sec": 355.809, "predictions": 105882, "recent_rows_per_sec": 306.3, "row_group_batches": 350, "rows_per_sec": 297.58, "timing_avg_sec": {"forward": 0.9666, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:30:03.901
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 340, "elapsed_sec": 346.325, "predictions": 102977, "recent_rows_per_sec": 310.83, "row_group_batches": 340, "rows_per_sec": 297.34, "timing_avg_sec": {"forward": 0.9672, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:29:54.417
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 330, "elapsed_sec": 336.86, "predictions": 100035, "recent_rows_per_sec": 315.99, "row_group_batches": 330, "rows_per_sec": 296.96, "timing_avg_sec": {"forward": 0.968, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:29:44.952
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 320, "elapsed_sec": 327.464, "predictions": 97066, "recent_rows_per_sec": 309.82, "row_group_batches": 320, "rows_per_sec": 296.42, "timing_avg_sec": {"forward": 0.9691, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:29:35.556
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 310, "elapsed_sec": 317.749, "predictions": 94056, "recent_rows_per_sec": 311.95, "row_group_batches": 310, "rows_per_sec": 296.01, "timing_avg_sec": {"forward": 0.9692, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:29:25.841
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 300, "elapsed_sec": 307.95, "predictions": 90999, "recent_rows_per_sec": 313.07, "row_group_batches": 300, "rows_per_sec": 295.5, "timing_avg_sec": {"forward": 0.969, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:29:16.041
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 290, "elapsed_sec": 298.083, "predictions": 87910, "recent_rows_per_sec": 312.97, "row_group_batches": 290, "rows_per_sec": 294.92, "timing_avg_sec": {"forward": 0.9685, "merge": 0.0, "move": 0.0005, "post": 0.0041}}
2026-05-11 14:29:06.175
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 280, "elapsed_sec": 288.213, "predictions": 84821, "recent_rows_per_sec": 312.06, "row_group_batches": 280, "rows_per_sec": 294.3, "timing_avg_sec": {"forward": 0.9681, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:28:56.305
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 270, "elapsed_sec": 278.42, "predictions": 81765, "recent_rows_per_sec": 312.45, "row_group_batches": 270, "rows_per_sec": 293.67, "timing_avg_sec": {"forward": 0.9679, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:28:46.512
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 260, "elapsed_sec": 268.614, "predictions": 78701, "recent_rows_per_sec": 311.6, "row_group_batches": 260, "rows_per_sec": 292.99, "timing_avg_sec": {"forward": 0.9677, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:28:36.705
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 250, "elapsed_sec": 258.871, "predictions": 75665, "recent_rows_per_sec": 310.3, "row_group_batches": 250, "rows_per_sec": 292.29, "timing_avg_sec": {"forward": 0.9676, "merge": 0.0, "move": 0.0005, "post": 0.0039}}
2026-05-11 14:28:26.962
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 240, "elapsed_sec": 249.138, "predictions": 72645, "recent_rows_per_sec": 310.89, "row_group_batches": 240, "rows_per_sec": 291.59, "timing_avg_sec": {"forward": 0.9676, "merge": 0.0, "move": 0.0005, "post": 0.0039}}
2026-05-11 14:28:17.230
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 230, "elapsed_sec": 239.402, "predictions": 69618, "recent_rows_per_sec": 308.96, "row_group_batches": 230, "rows_per_sec": 290.8, "timing_avg_sec": {"forward": 0.9676, "merge": 0.0, "move": 0.0005, "post": 0.0039}}
2026-05-11 14:28:07.493
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 220, "elapsed_sec": 229.721, "predictions": 66627, "recent_rows_per_sec": 306.06, "row_group_batches": 220, "rows_per_sec": 290.03, "timing_avg_sec": {"forward": 0.9679, "merge": 0.0, "move": 0.0005, "post": 0.0038}}
2026-05-11 14:27:57.813
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 210, "elapsed_sec": 220.089, "predictions": 63679, "recent_rows_per_sec": 306.49, "row_group_batches": 210, "rows_per_sec": 289.33, "timing_avg_sec": {"forward": 0.9683, "merge": 0.0, "move": 0.0005, "post": 0.0039}}
2026-05-11 14:27:48.180
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 200, "elapsed_sec": 210.48, "predictions": 60734, "recent_rows_per_sec": 309.81, "row_group_batches": 200, "rows_per_sec": 288.55, "timing_avg_sec": {"forward": 0.9689, "merge": 0.0, "move": 0.0005, "post": 0.0039}}
2026-05-11 14:27:38.572
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 190, "elapsed_sec": 200.932, "predictions": 57776, "recent_rows_per_sec": 315.89, "row_group_batches": 190, "rows_per_sec": 287.54, "timing_avg_sec": {"forward": 0.9698, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:27:29.024
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 180, "elapsed_sec": 191.565, "predictions": 54817, "recent_rows_per_sec": 312.88, "row_group_batches": 180, "rows_per_sec": 286.15, "timing_avg_sec": {"forward": 0.972, "merge": 0.0, "move": 0.0005, "post": 0.0039}}
2026-05-11 14:27:19.657
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 170, "elapsed_sec": 182.12, "predictions": 51862, "recent_rows_per_sec": 307.65, "row_group_batches": 170, "rows_per_sec": 284.77, "timing_avg_sec": {"forward": 0.9739, "merge": 0.0, "move": 0.0005, "post": 0.0039}}
2026-05-11 14:27:10.212
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 160, "elapsed_sec": 172.496, "predictions": 48901, "recent_rows_per_sec": 309.21, "row_group_batches": 160, "rows_per_sec": 283.49, "timing_avg_sec": {"forward": 0.9749, "merge": 0.0, "move": 0.0005, "post": 0.0039}}
2026-05-11 14:27:00.587
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 150, "elapsed_sec": 162.79, "predictions": 45900, "recent_rows_per_sec": 310.38, "row_group_batches": 150, "rows_per_sec": 281.96, "timing_avg_sec": {"forward": 0.9755, "merge": 0.0, "move": 0.0005, "post": 0.004}}
2026-05-11 14:26:50.882
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 140, "elapsed_sec": 153.002, "predictions": 42862, "recent_rows_per_sec": 311.5, "row_group_batches": 140, "rows_per_sec": 280.14, "timing_avg_sec": {"forward": 0.9755, "merge": 0.0, "move": 0.0006, "post": 0.0041}}
2026-05-11 14:26:41.094
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 130, "elapsed_sec": 143.355, "predictions": 39857, "recent_rows_per_sec": 308.63, "row_group_batches": 130, "rows_per_sec": 278.03, "timing_avg_sec": {"forward": 0.9766, "merge": 0.0, "move": 0.0006, "post": 0.0042}}
2026-05-11 14:26:31.447
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 120, "elapsed_sec": 133.612, "predictions": 36850, "recent_rows_per_sec": 314.14, "row_group_batches": 120, "rows_per_sec": 275.8, "timing_avg_sec": {"forward": 0.9772, "merge": 0.0, "move": 0.0006, "post": 0.0043}}
2026-05-11 14:26:21.704
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 110, "elapsed_sec": 123.948, "predictions": 33814, "recent_rows_per_sec": 312.98, "row_group_batches": 110, "rows_per_sec": 272.81, "timing_avg_sec": {"forward": 0.9785, "merge": 0.0, "move": 0.0006, "post": 0.0044}}
2026-05-11 14:26:12.039
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 100, "elapsed_sec": 114.184, "predictions": 30758, "recent_rows_per_sec": 312.06, "row_group_batches": 100, "rows_per_sec": 269.37, "timing_avg_sec": {"forward": 0.9793, "merge": 0.0, "move": 0.0006, "post": 0.0044}}
2026-05-11 14:26:02.275
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 90, "elapsed_sec": 104.429, "predictions": 27714, "recent_rows_per_sec": 312.59, "row_group_batches": 90, "rows_per_sec": 265.39, "timing_avg_sec": {"forward": 0.9804, "merge": 0.0, "move": 0.0006, "post": 0.0044}}
2026-05-11 14:25:52.521
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 80, "elapsed_sec": 94.662, "predictions": 24661, "recent_rows_per_sec": 312.1, "row_group_batches": 80, "rows_per_sec": 260.52, "timing_avg_sec": {"forward": 0.9816, "merge": 0.0, "move": 0.0006, "post": 0.0043}}
2026-05-11 14:25:42.754
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 70, "elapsed_sec": 84.887, "predictions": 21610, "recent_rows_per_sec": 311.43, "row_group_batches": 70, "rows_per_sec": 254.57, "timing_avg_sec": {"forward": 0.983, "merge": 0.0, "move": 0.0006, "post": 0.0042}}
2026-05-11 14:25:32.978
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 60, "elapsed_sec": 75.09, "predictions": 18559, "recent_rows_per_sec": 311.96, "row_group_batches": 60, "rows_per_sec": 247.16, "timing_avg_sec": {"forward": 0.9845, "merge": 0.0, "move": 0.0006, "post": 0.0041}}
2026-05-11 14:25:23.181
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 50, "elapsed_sec": 65.274, "predictions": 15497, "recent_rows_per_sec": 313.86, "row_group_batches": 50, "rows_per_sec": 237.41, "timing_avg_sec": {"forward": 0.9863, "merge": 0.0, "move": 0.0006, "post": 0.004}}
2026-05-11 14:25:13.366
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 40, "elapsed_sec": 55.433, "predictions": 12408, "recent_rows_per_sec": 319.39, "row_group_batches": 40, "rows_per_sec": 223.84, "timing_avg_sec": {"forward": 0.9882, "merge": 0.0, "move": 0.0007, "post": 0.0039}}
2026-05-11 14:25:03.524
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 30, "elapsed_sec": 45.739, "predictions": 9312, "recent_rows_per_sec": 315.99, "row_group_batches": 30, "rows_per_sec": 203.59, "timing_avg_sec": {"forward": 0.996, "merge": 0.0, "move": 0.0007, "post": 0.0041}}
2026-05-11 14:24:53.831
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 10, "elapsed_sec": 26.217, "predictions": 3138, "recent_rows_per_sec": 119.69, "row_group_batches": 10, "rows_per_sec": 119.69, "timing_avg_sec": {"forward": 1.0453, "merge": 0.0, "move": 0.0013, "post": 0.0049}}
2026-05-11 14:24:34.309
PLATFORM_FIRST_BATCH {"batch_predictions": 314, "rows_in_forward": 314, "sample_scores": [0.0018692, 0.08398438, 0.05493164, 0.11572266, 0.32421875], "sample_user_ids": ["683974", "1334888", "51348", "1184407", "8854"], "sanitize_counts": {"nan_replaced": 0, "neginf_replaced": 0, "posinf_replaced": 0}, "timing_sec": {"forward": 1.4448, "merge": 0.0, "move": 0.0086, "post": 0.0051}, "tta_enabled": true}
2026-05-11 14:24:25.249
2026-05-11 06:24:22,042 - INFO - NumExpr defaulting to 16 threads.
2026-05-11 14:24:22.042
2026-05-11 06:24:22,042 - INFO - NumExpr defaulting to 16 threads.
2026-05-11 14:24:22.042
2026-05-11 06:24:22,042 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-11 14:24:22.042
2026-05-11 06:24:22,042 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-11 14:24:22.042
2026-05-11 06:24:22,042 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-11 14:24:22.042
2026-05-11 06:24:22,042 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-11 14:24:22.042
2026-05-11 06:24:22,042 - INFO - NumExpr defaulting to 16 threads.
2026-05-11 14:24:22.042
2026-05-11 06:24:22,042 - INFO - NumExpr defaulting to 16 threads.
2026-05-11 14:24:22.042
2026-05-11 06:24:22,042 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-11 14:24:22.042
2026-05-11 06:24:22,042 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-11 14:24:22.042
2026-05-11 06:24:22,042 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-11 14:24:22.042
2026-05-11 06:24:22,042 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-11 14:24:22.042
PLATFORM_AMP_CONFIG {"cuda_bf16_supported": true, "enabled": true, "mode": "bf16", "requested": "bf16"}
2026-05-11 14:24:21.544
2026-05-11 06:24:21,543 - INFO - Starting inference...
2026-05-11 14:24:21.543
PLATFORM_MODEL_READY {"device": "cuda", "num_ns": 13, "num_parameters": 778116085}
2026-05-11 14:24:21.543
2026-05-11 06:24:21,542 - INFO - Model loaded successfully
2026-05-11 14:24:21.542
2026-05-11 06:24:16,667 - INFO - Loading checkpoint from /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model/model.pt
2026-05-11 14:24:16.667
2026-05-11 06:24:08,663 - INFO - RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0
2026-05-11 14:24:08.663
2026-05-11 06:24:08,646 - INFO - RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1
2026-05-11 14:24:08.646
2026-05-11 06:24:08,634 - INFO - Building PCVRHyFormer with cfg: {'d_model': 64, 'emb_dim': 64, 'num_queries': 2, 'num_hyformer_blocks': 2, 'num_heads': 4, 'seq_encoder_type': 'transformer', 'hidden_mult': 4, 'dropout_rate': 0.01, 'seq_top_k': 50, 'seq_causal': False, 'action_num': 1, 'num_time_buckets': 64, 'rank_mixer_mode': 'learned', 'use_rope': False, 'rope_base': 10000.0, 'emb_skip_threshold': 1000000, 'seq_id_threshold': 10000, 'ns_tokenizer_type': 'rankmixer', 'user_ns_tokens': 5, 'item_ns_tokens': 2, 'use_wide_dense_tokens': True, 'wide_dense_threshold': 32, 'dense_scalar_transform': 'signlog', 'xdomain_dense_dim': 18, 'use_xdomain_features': True, 'use_inter_time_buckets': True, 'use_continuous_time': True, 'high_cardinality_mode': 'hash', 'hash_num_buckets': 1048576, 'hash_num_hashes': 2, 'use_target_aware_attn': True, 'use_cross_domain_attn': True, 'seq_topk_mode': 'recent', 'ffn_type': 'swiglu', 'norm_type': 'rms', 'random_id_mask_prob': 0.05, 'legacy_user_dense_proj': False, 'legacy_mixer_ffn_names': False}
2026-05-11 14:24:08.634
PLATFORM_DATASET_CONFIG {"batch_size": 2048, "item_dense_dim": 0, "item_int_features": 14, "num_row_groups": 1000, "num_rows_estimate": 310000, "seq_domains": ["seq_a", "seq_b", "seq_c", "seq_d"], "seq_max_lens": {"seq_a": 256, "seq_b": 256, "seq_c": 512, "seq_d": 448}, "user_dense_dim": 918, "user_int_features": 46}
2026-05-11 14:24:08.634
2026-05-11 06:24:08,634 - INFO - No NS groups JSON found, using default: each feature as one group
2026-05-11 14:24:08.634
2026-05-11 06:24:08,634 - WARNING - train_config missing 'legacy_mixer_ffn_names', using fallback = False
2026-05-11 14:24:08.634
2026-05-11 06:24:08,634 - WARNING - train_config missing 'legacy_user_dense_proj', using fallback = False
2026-05-11 14:24:08.634
2026-05-11 06:24:08,634 - INFO - Total test samples: 310000
2026-05-11 14:24:08.634
2026-05-11 06:24:08,634 - INFO - PCVRParquetDataset: 310000 rows from 1000 file(s), batch_size=2048, buffer_batches=0, shuffle=False
2026-05-11 14:24:08.634
PLATFORM_INFER_CONFIG {"amp_dtype": "bf16", "batch_size": 2048, "checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model", "device": "cuda", "eval_data_dir": "/data_ams/academic_infer_data/", "merge_batches": false, "num_workers": 4, "progress_every": 10, "result_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/82255/results", "schema_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model/schema.json", "seq_max_lens": {"seq_a": 256, "seq_b": 256, "seq_c": 512, "seq_d": 448}, "train_config": {"d_model": 64, "emb_skip_threshold": 1000000, "high_cardinality_mode": "hash", "num_heads": 4, "num_hyformer_blocks": 2, "num_queries": 2, "seq_encoder_type": "transformer", "train_batch_size": 256}, "tta_crop_lens": {"seq_a": 128, "seq_b": 128, "seq_c": 256, "seq_d": 256}, "tta_mode": "recent_blend", "tta_repeat": 14, "tta_weight": 0.12, "use_amp": true}
2026-05-11 14:24:08.124
PLATFORM_CHECKPOINT_STATUS {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model", "model_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model/model.pt", "size_bytes": 3112686103}, "schema_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model/schema.json", "size_bytes": 5560}, "train_config_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model/train_config.json", "size_bytes": 2302}}
2026-05-11 14:24:08.124
2026-05-11 06:24:08,123 - INFO - seq_max_lens: {'seq_a': 256, 'seq_b': 256, 'seq_c': 512, 'seq_d': 448}
2026-05-11 14:24:08.123
2026-05-11 06:24:08,123 - INFO - Loaded train_config from /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model/train_config.json
2026-05-11 14:24:08.123
2026-05-11 06:24:08,119 - INFO - Using schema: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model/schema.json
2026-05-11 14:24:08.120
PLATFORM_INFER_START {"cuda_available": true, "cwd": "/workspace", "time": "2026-05-11 06:24:08", "torch_version": "2.7.1+cu126"}
2026-05-11 14:24:08.115
====== Inferring ======
2026-05-11 14:24:06.665
=================================
2026-05-11 14:24:06.272
Working Dir: /workspace
2026-05-11 14:24:06.272
Environment: competition
2026-05-11 14:24:06.272
GPU Count: 1
2026-05-11 14:24:06.272
GPU Available: True
2026-05-11 14:24:04.585
PyTorch: 2.7.1+cu126
2026-05-11 14:24:03.013
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-11 14:24:00.644
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-11 14:24:00.644
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-11 14:24:00.644
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-11 14:24:00.644
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-11 14:24:00.641
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-11 14:24:00.641
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-11 14:24:00.641
Python: Python 3.10.20
2026-05-11 14:24:00.270
CUDA: 12.6.77
2026-05-11 14:24:00.265
=== Competition Environment Ready ===
2026-05-11 14:24:00.256
Complete setting network policy rules.
2026-05-11 14:23:56.674
Complete setting taiji user.
2026-05-11 14:23:56.621
Truncated to last 1000 lines.















































====== Score Finished ======
2026-05-11 15:19:53.709
====== Scoring ======
2026-05-11 15:19:50.528
=================================
2026-05-11 15:19:50.147
Working Dir: /workspace
2026-05-11 15:19:50.147
Environment: competition
2026-05-11 15:19:50.147
GPU Count: 1
2026-05-11 15:19:50.147
GPU Available: True
2026-05-11 15:19:48.641
PyTorch: 2.7.1+cu126
2026-05-11 15:19:47.166
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-11 15:19:44.803
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-11 15:19:44.803
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-11 15:19:44.803
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-11 15:19:44.803
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-11 15:19:44.799
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-11 15:19:44.799
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-11 15:19:44.799
Python: Python 3.10.20
2026-05-11 15:19:44.430
CUDA: 12.6.77
2026-05-11 15:19:44.425
=== Competition Environment Ready ===
2026-05-11 15:19:44.416
Complete setting network policy rules.
2026-05-11 15:19:40.937
Complete setting taiji user.
2026-05-11 15:19:40.856
====== Infer Finished ======
2026-05-11 15:16:29.287
2026-05-11 07:16:28,562 - INFO - Saved status.json to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/82407/results/status.json
2026-05-11 15:16:28.562
2026-05-11 07:16:28,062 - INFO - Saved 310000 predictions to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/82407/results/predictions.json
2026-05-11 15:16:28.062
2026-05-11 07:16:27,663 - INFO - Inference complete: 310000 predictions
2026-05-11 15:16:27.663
2026-05-11 07:16:27,001 - INFO - Processed 512000 samples
2026-05-11 15:16:27.001
2026-05-11 07:16:08,286 - INFO - Processed 486400 samples
2026-05-11 15:16:08.286
2026-05-11 07:15:49,475 - INFO - Processed 460800 samples
2026-05-11 15:15:49.476
2026-05-11 07:15:29,753 - INFO - Processed 435200 samples
2026-05-11 15:15:29.754
2026-05-11 07:15:10,208 - INFO - Processed 409600 samples
2026-05-11 15:15:10.208
2026-05-11 07:14:51,337 - INFO - Processed 384000 samples
2026-05-11 15:14:51.338
2026-05-11 07:14:32,631 - INFO - Processed 358400 samples
2026-05-11 15:14:32.631
2026-05-11 07:14:13,463 - INFO - Processed 332800 samples
2026-05-11 15:14:13.463
2026-05-11 07:13:54,411 - INFO - Processed 307200 samples
2026-05-11 15:13:54.411
2026-05-11 07:13:36,432 - INFO - Processed 281600 samples
2026-05-11 15:13:36.433
2026-05-11 07:13:18,508 - INFO - Processed 256000 samples
2026-05-11 15:13:18.508
2026-05-11 07:13:00,243 - INFO - Processed 230400 samples
2026-05-11 15:13:00.243
2026-05-11 07:12:41,881 - INFO - Processed 204800 samples
2026-05-11 15:12:41.881
2026-05-11 07:12:24,611 - INFO - Processed 179200 samples
2026-05-11 15:12:24.611
2026-05-11 07:12:07,025 - INFO - Processed 153600 samples
2026-05-11 15:12:07.026
2026-05-11 07:11:48,249 - INFO - Processed 128000 samples
2026-05-11 15:11:48.250
2026-05-11 07:11:29,862 - INFO - Processed 102400 samples
2026-05-11 15:11:29.863
2026-05-11 07:11:12,352 - INFO - Processed 76800 samples
2026-05-11 15:11:12.352
2026-05-11 07:10:54,455 - INFO - Processed 51200 samples
2026-05-11 15:10:54.456
2026-05-11 07:10:35,765 - INFO - Processed 25600 samples
2026-05-11 15:10:35.765
2026-05-11 07:10:14,772 - INFO - NumExpr defaulting to 16 threads.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - NumExpr defaulting to 16 threads.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - NumExpr defaulting to 16 threads.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - NumExpr defaulting to 16 threads.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - NumExpr defaulting to 16 threads.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - NumExpr defaulting to 16 threads.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,772 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,771 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-11 15:10:14.772
2026-05-11 07:10:14,771 - INFO - NumExpr defaulting to 16 threads.
2026-05-11 15:10:14.771
2026-05-11 07:10:14,771 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-11 15:10:14.771
2026-05-11 07:10:14,771 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-11 15:10:14.771
2026-05-11 07:10:14,771 - INFO - NumExpr defaulting to 16 threads.
2026-05-11 15:10:14.771
2026-05-11 07:10:14,771 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-11 15:10:14.771
2026-05-11 07:10:14,771 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-11 15:10:14.771
2026-05-11 07:10:13,943 - INFO - Starting inference...
2026-05-11 15:10:13.944
2026-05-11 07:10:13,943 - INFO - Model loaded successfully
2026-05-11 15:10:13.943
2026-05-11 07:10:05,034 - INFO - Loading checkpoint from /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model/model.pt
2026-05-11 15:10:05.035
2026-05-11 07:09:53,070 - INFO - RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0
2026-05-11 15:09:53.070
2026-05-11 07:09:53,051 - INFO - RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1
2026-05-11 15:09:53.051
2026-05-11 07:09:53,024 - INFO - Building PCVRHyFormer with cfg: {'d_model': 64, 'emb_dim': 64, 'num_queries': 2, 'num_hyformer_blocks': 2, 'num_heads': 4, 'seq_encoder_type': 'transformer', 'hidden_mult': 4, 'dropout_rate': 0.01, 'seq_top_k': 50, 'seq_causal': False, 'action_num': 1, 'num_time_buckets': 64, 'rank_mixer_mode': 'learned', 'use_rope': False, 'rope_base': 10000.0, 'emb_skip_threshold': 1000000, 'seq_id_threshold': 10000, 'ns_tokenizer_type': 'rankmixer', 'user_ns_tokens': 5, 'item_ns_tokens': 2, 'use_wide_dense_tokens': True, 'wide_dense_threshold': 32, 'dense_scalar_transform': 'signlog', 'xdomain_dense_dim': 18, 'use_xdomain_features': True, 'use_inter_time_buckets': True, 'use_continuous_time': True, 'high_cardinality_mode': 'hash', 'hash_num_buckets': 1048576, 'hash_num_hashes': 2, 'use_target_aware_attn': True, 'use_cross_domain_attn': True, 'seq_topk_mode': 'recent', 'ffn_type': 'swiglu', 'norm_type': 'rms', 'random_id_mask_prob': 0.05}
2026-05-11 15:09:53.024
2026-05-11 07:09:53,023 - INFO - No NS groups JSON found, using default: each feature as one group
2026-05-11 15:09:53.024
2026-05-11 07:09:53,023 - INFO - Total test samples: 310000
2026-05-11 15:09:53.024
2026-05-11 07:09:53,023 - INFO - PCVRParquetDataset: 310000 rows from 1000 file(s), batch_size=256, buffer_batches=0, shuffle=False
2026-05-11 15:09:53.023
2026-05-11 07:09:52,512 - INFO - seq_max_lens: {'seq_a': 256, 'seq_b': 256, 'seq_c': 512, 'seq_d': 512}
2026-05-11 15:09:52.512
2026-05-11 07:09:52,512 - INFO - Loaded train_config from /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model/train_config.json
2026-05-11 15:09:52.512
2026-05-11 07:09:52,510 - INFO - Using schema: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260510183451_da6a8363/self/95d1b0469e0fba1e019e1174539004f6/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model/schema.json
2026-05-11 15:09:52.510
====== Inferring ======
2026-05-11 15:09:51.167
=================================
2026-05-11 15:09:50.785
Working Dir: /workspace
2026-05-11 15:09:50.785
Environment: competition
2026-05-11 15:09:50.785
GPU Count: 1
2026-05-11 15:09:50.784
GPU Available: True
2026-05-11 15:09:49.322
PyTorch: 2.7.1+cu126
2026-05-11 15:09:47.885
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-11 15:09:45.539
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-11 15:09:45.539
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-11 15:09:45.539
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-11 15:09:45.539
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-11 15:09:45.536
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-11 15:09:45.536
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-11 15:09:45.536
Python: Python 3.10.20
2026-05-11 15:09:45.156
CUDA: 12.6.77
2026-05-11 15:09:45.151
=== Competition Environment Ready ===
2026-05-11 15:09:45.143
Complete setting network policy rules.
2026-05-11 15:09:41.170
Complete setting taiji user.
2026-05-11 15:09:41.107
Truncated to last 1000 lines.