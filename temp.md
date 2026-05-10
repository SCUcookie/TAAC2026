====== Score Finished ======
2026-05-10 14:53:42.571
====== Scoring ======
2026-05-10 14:53:39.656
=================================
2026-05-10 14:53:39.262
Working Dir: /workspace
2026-05-10 14:53:39.262
Environment: competition
2026-05-10 14:53:39.262
GPU Count: 1
2026-05-10 14:53:39.262
GPU Available: True
2026-05-10 14:53:37.870
PyTorch: 2.7.1+cu126
2026-05-10 14:53:36.283
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-10 14:53:33.859
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-10 14:53:33.859
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-10 14:53:33.859
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-10 14:53:33.859
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-10 14:53:33.856
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-10 14:53:33.856
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-10 14:53:33.856
Python: Python 3.10.20
2026-05-10 14:53:33.481
CUDA: 12.6.77
2026-05-10 14:53:33.475
=== Competition Environment Ready ===
2026-05-10 14:53:33.465
Complete setting network policy rules.
2026-05-10 14:53:29.810
Complete setting taiji user.
2026-05-10 14:53:29.743
====== Infer Finished ======
2026-05-10 14:49:25.760
2026-05-10 06:49:24,914 - INFO - Saved status.json to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/78557/results/status.json
2026-05-10 14:49:24.915
2026-05-10 06:49:24,096 - INFO - Saved 310000 predictions to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/78557/results/predictions.json
2026-05-10 14:49:24.096
PLATFORM_INFER_SUMMARY {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "elapsed_sec": 61.081, "num_predictions": 310000, "num_unique_user_ids": 310000, "output_shape": "predictions_dict_user_id_to_float", "result_files": {"csv": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/78557/results/submission.csv", "size_bytes": 5643992}, "json": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/78557/results/predictions.json", "size_bytes": 7821053}, "scores": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/78557/results/scores.npy", "size_bytes": 1240128}}, "result_json": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/78557/results/predictions.json", "score_stats": {"count": 310000, "inf_count": 0, "max": 0.9765625, "mean": 0.11806643, "min": 0.00064468, "nan_count": 0, "p01": 0.0007782, "p50": 0.05834961, "p99": 0.828125, "std": 0.15838892}, "scores_npy": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/78557/results/scores.npy", "submission_csv": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/78557/results/submission.csv"}
2026-05-10 14:49:24.096
2026-05-10 06:49:23,567 - INFO - Inference complete: 310000 predictions
2026-05-10 14:49:23.567
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 1000, "elapsed_sec": 60.581, "predictions": 310000, "recent_rows_per_sec": 5954.22, "row_group_batches": 1000, "rows_per_sec": 5117.14, "timing_avg_sec": {"forward": 0.0486, "merge": 0.0, "move": 0.0005, "post": 0.0017}}
2026-05-10 14:49:23.567
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 990, "elapsed_sec": 60.052, "predictions": 306853, "recent_rows_per_sec": 5902.6, "row_group_batches": 990, "rows_per_sec": 5109.77, "timing_avg_sec": {"forward": 0.0486, "merge": 0.0, "move": 0.0005, "post": 0.0017}}
2026-05-10 14:49:23.038
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 980, "elapsed_sec": 59.51, "predictions": 303650, "recent_rows_per_sec": 5430.26, "row_group_batches": 980, "rows_per_sec": 5102.54, "timing_avg_sec": {"forward": 0.0486, "merge": 0.0, "move": 0.0005, "post": 0.0017}}
2026-05-10 14:49:22.496
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 970, "elapsed_sec": 58.916, "predictions": 300426, "recent_rows_per_sec": 5528.95, "row_group_batches": 970, "rows_per_sec": 5099.24, "timing_avg_sec": {"forward": 0.0485, "merge": 0.0, "move": 0.0005, "post": 0.0016}}
2026-05-10 14:49:21.902
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 960, "elapsed_sec": 58.328, "predictions": 297177, "recent_rows_per_sec": 5395.81, "row_group_batches": 960, "rows_per_sec": 5094.91, "timing_avg_sec": {"forward": 0.0485, "merge": 0.0, "move": 0.0005, "post": 0.0016}}
2026-05-10 14:49:21.314
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 950, "elapsed_sec": 57.723, "predictions": 293913, "recent_rows_per_sec": 5714.02, "row_group_batches": 950, "rows_per_sec": 5091.76, "timing_avg_sec": {"forward": 0.0484, "merge": 0.0, "move": 0.0005, "post": 0.0016}}
2026-05-10 14:49:20.710
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 940, "elapsed_sec": 57.162, "predictions": 290706, "recent_rows_per_sec": 5054.67, "row_group_batches": 940, "rows_per_sec": 5085.65, "timing_avg_sec": {"forward": 0.0483, "merge": 0.0, "move": 0.0005, "post": 0.0016}}
2026-05-10 14:49:20.148
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 930, "elapsed_sec": 56.525, "predictions": 287485, "recent_rows_per_sec": 5821.28, "row_group_batches": 930, "rows_per_sec": 5086.0, "timing_avg_sec": {"forward": 0.0482, "merge": 0.0, "move": 0.0005, "post": 0.0016}}
2026-05-10 14:49:19.511
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 920, "elapsed_sec": 55.964, "predictions": 284222, "recent_rows_per_sec": 5001.6, "row_group_batches": 920, "rows_per_sec": 5078.63, "timing_avg_sec": {"forward": 0.0482, "merge": 0.0, "move": 0.0005, "post": 0.0016}}
2026-05-10 14:49:18.951
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 910, "elapsed_sec": 55.311, "predictions": 280956, "recent_rows_per_sec": 5674.76, "row_group_batches": 910, "rows_per_sec": 5079.54, "timing_avg_sec": {"forward": 0.048, "merge": 0.0, "move": 0.0005, "post": 0.0016}}
2026-05-10 14:49:18.298
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 900, "elapsed_sec": 54.741, "predictions": 277718, "recent_rows_per_sec": 5042.68, "row_group_batches": 900, "rows_per_sec": 5073.34, "timing_avg_sec": {"forward": 0.0479, "merge": 0.0, "move": 0.0005, "post": 0.0016}}
2026-05-10 14:49:17.727
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 890, "elapsed_sec": 54.1, "predictions": 274485, "recent_rows_per_sec": 5769.47, "row_group_batches": 890, "rows_per_sec": 5073.7, "timing_avg_sec": {"forward": 0.0478, "merge": 0.0, "move": 0.0005, "post": 0.0016}}
2026-05-10 14:49:17.086
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 880, "elapsed_sec": 53.539, "predictions": 271252, "recent_rows_per_sec": 5339.91, "row_group_batches": 880, "rows_per_sec": 5066.42, "timing_avg_sec": {"forward": 0.0477, "merge": 0.0, "move": 0.0005, "post": 0.0016}}
2026-05-10 14:49:16.525
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 870, "elapsed_sec": 52.929, "predictions": 267992, "recent_rows_per_sec": 5855.01, "row_group_batches": 870, "rows_per_sec": 5063.26, "timing_avg_sec": {"forward": 0.0476, "merge": 0.0, "move": 0.0005, "post": 0.0015}}
2026-05-10 14:49:15.915
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 860, "elapsed_sec": 52.362, "predictions": 264675, "recent_rows_per_sec": 5529.71, "row_group_batches": 860, "rows_per_sec": 5054.7, "timing_avg_sec": {"forward": 0.0476, "merge": 0.0, "move": 0.0005, "post": 0.0015}}
2026-05-10 14:49:15.348
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 850, "elapsed_sec": 51.765, "predictions": 261375, "recent_rows_per_sec": 5879.81, "row_group_batches": 850, "rows_per_sec": 5049.22, "timing_avg_sec": {"forward": 0.0475, "merge": 0.0, "move": 0.0005, "post": 0.0015}}
2026-05-10 14:49:14.752
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 840, "elapsed_sec": 51.212, "predictions": 258123, "recent_rows_per_sec": 5400.29, "row_group_batches": 840, "rows_per_sec": 5040.25, "timing_avg_sec": {"forward": 0.0474, "merge": 0.0, "move": 0.0005, "post": 0.0015}}
2026-05-10 14:49:14.199
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 830, "elapsed_sec": 50.616, "predictions": 254901, "recent_rows_per_sec": 5745.29, "row_group_batches": 830, "rows_per_sec": 5036.01, "timing_avg_sec": {"forward": 0.0473, "merge": 0.0, "move": 0.0005, "post": 0.0015}}
2026-05-10 14:49:13.602
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 820, "elapsed_sec": 50.054, "predictions": 251674, "recent_rows_per_sec": 5513.28, "row_group_batches": 820, "rows_per_sec": 5028.05, "timing_avg_sec": {"forward": 0.0472, "merge": 0.0, "move": 0.0005, "post": 0.0015}}
2026-05-10 14:49:13.040
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 810, "elapsed_sec": 49.466, "predictions": 248434, "recent_rows_per_sec": 5844.7, "row_group_batches": 810, "rows_per_sec": 5022.28, "timing_avg_sec": {"forward": 0.0471, "merge": 0.0, "move": 0.0005, "post": 0.0015}}
2026-05-10 14:49:12.453
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 800, "elapsed_sec": 48.913, "predictions": 245201, "recent_rows_per_sec": 5088.66, "row_group_batches": 800, "rows_per_sec": 5012.98, "timing_avg_sec": {"forward": 0.0471, "merge": 0.0, "move": 0.0005, "post": 0.0015}}
2026-05-10 14:49:11.899
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 790, "elapsed_sec": 48.284, "predictions": 242000, "recent_rows_per_sec": 5698.74, "row_group_batches": 790, "rows_per_sec": 5012.0, "timing_avg_sec": {"forward": 0.0469, "merge": 0.0, "move": 0.0005, "post": 0.0015}}
2026-05-10 14:49:11.270
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 780, "elapsed_sec": 47.713, "predictions": 238746, "recent_rows_per_sec": 5345.33, "row_group_batches": 780, "rows_per_sec": 5003.78, "timing_avg_sec": {"forward": 0.0468, "merge": 0.0, "move": 0.0005, "post": 0.0014}}
2026-05-10 14:49:10.700
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 770, "elapsed_sec": 47.108, "predictions": 235511, "recent_rows_per_sec": 5707.22, "row_group_batches": 770, "rows_per_sec": 4999.39, "timing_avg_sec": {"forward": 0.0467, "merge": 0.0, "move": 0.0005, "post": 0.0014}}
2026-05-10 14:49:10.094
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 760, "elapsed_sec": 46.532, "predictions": 232222, "recent_rows_per_sec": 5332.66, "row_group_batches": 760, "rows_per_sec": 4990.62, "timing_avg_sec": {"forward": 0.0466, "merge": 0.0, "move": 0.0005, "post": 0.0014}}
2026-05-10 14:49:09.518
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 750, "elapsed_sec": 45.914, "predictions": 228928, "recent_rows_per_sec": 5604.07, "row_group_batches": 750, "rows_per_sec": 4986.02, "timing_avg_sec": {"forward": 0.0464, "merge": 0.0, "move": 0.0005, "post": 0.0014}}
2026-05-10 14:49:08.900
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 740, "elapsed_sec": 45.33, "predictions": 225656, "recent_rows_per_sec": 5235.31, "row_group_batches": 740, "rows_per_sec": 4978.06, "timing_avg_sec": {"forward": 0.0463, "merge": 0.0, "move": 0.0005, "post": 0.0014}}
2026-05-10 14:49:08.316
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 730, "elapsed_sec": 44.718, "predictions": 222449, "recent_rows_per_sec": 5787.49, "row_group_batches": 730, "rows_per_sec": 4974.54, "timing_avg_sec": {"forward": 0.0462, "merge": 0.0, "move": 0.0005, "post": 0.0014}}
2026-05-10 14:49:07.704
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 720, "elapsed_sec": 44.16, "predictions": 219225, "recent_rows_per_sec": 5170.41, "row_group_batches": 720, "rows_per_sec": 4964.28, "timing_avg_sec": {"forward": 0.0461, "merge": 0.0, "move": 0.0005, "post": 0.0014}}
2026-05-10 14:49:07.147
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 710, "elapsed_sec": 43.532, "predictions": 215975, "recent_rows_per_sec": 3798.61, "row_group_batches": 710, "rows_per_sec": 4961.31, "timing_avg_sec": {"forward": 0.0459, "merge": 0.0, "move": 0.0005, "post": 0.0013}}
2026-05-10 14:49:06.518
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 700, "elapsed_sec": 42.678, "predictions": 212731, "recent_rows_per_sec": 5385.75, "row_group_batches": 700, "rows_per_sec": 4984.57, "timing_avg_sec": {"forward": 0.0454, "merge": 0.0, "move": 0.0005, "post": 0.0013}}
2026-05-10 14:49:05.664
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 690, "elapsed_sec": 42.085, "predictions": 209539, "recent_rows_per_sec": 5497.4, "row_group_batches": 690, "rows_per_sec": 4978.92, "timing_avg_sec": {"forward": 0.0452, "merge": 0.0, "move": 0.0005, "post": 0.0013}}
2026-05-10 14:49:05.071
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 680, "elapsed_sec": 41.503, "predictions": 206341, "recent_rows_per_sec": 5640.74, "row_group_batches": 680, "rows_per_sec": 4971.66, "timing_avg_sec": {"forward": 0.0451, "merge": 0.0, "move": 0.0005, "post": 0.0013}}
2026-05-10 14:49:04.491
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 670, "elapsed_sec": 40.943, "predictions": 203180, "recent_rows_per_sec": 5468.82, "row_group_batches": 670, "rows_per_sec": 4962.5, "timing_avg_sec": {"forward": 0.045, "merge": 0.0, "move": 0.0005, "post": 0.0013}}
2026-05-10 14:49:03.929
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 660, "elapsed_sec": 40.367, "predictions": 200029, "recent_rows_per_sec": 5296.27, "row_group_batches": 660, "rows_per_sec": 4955.27, "timing_avg_sec": {"forward": 0.0448, "merge": 0.0, "move": 0.0005, "post": 0.0013}}
2026-05-10 14:49:03.353
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 650, "elapsed_sec": 39.773, "predictions": 196884, "recent_rows_per_sec": 5649.68, "row_group_batches": 650, "rows_per_sec": 4950.18, "timing_avg_sec": {"forward": 0.0447, "merge": 0.0, "move": 0.0005, "post": 0.0012}}
2026-05-10 14:49:02.759
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 640, "elapsed_sec": 39.219, "predictions": 193753, "recent_rows_per_sec": 5726.31, "row_group_batches": 640, "rows_per_sec": 4940.29, "timing_avg_sec": {"forward": 0.0446, "merge": 0.0, "move": 0.0005, "post": 0.0012}}
2026-05-10 14:49:02.205
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 630, "elapsed_sec": 38.663, "predictions": 190572, "recent_rows_per_sec": 5258.58, "row_group_batches": 630, "rows_per_sec": 4929.0, "timing_avg_sec": {"forward": 0.0445, "merge": 0.0, "move": 0.0005, "post": 0.0012}}
2026-05-10 14:49:01.650
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 620, "elapsed_sec": 38.062, "predictions": 187410, "recent_rows_per_sec": 5924.57, "row_group_batches": 620, "rows_per_sec": 4923.79, "timing_avg_sec": {"forward": 0.0443, "merge": 0.0, "move": 0.0005, "post": 0.0011}}
2026-05-10 14:49:01.048
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 610, "elapsed_sec": 37.534, "predictions": 184284, "recent_rows_per_sec": 5563.14, "row_group_batches": 610, "rows_per_sec": 4909.73, "timing_avg_sec": {"forward": 0.0442, "merge": 0.0, "move": 0.0005, "post": 0.0011}}
2026-05-10 14:49:00.521
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 600, "elapsed_sec": 36.979, "predictions": 181192, "recent_rows_per_sec": 5622.02, "row_group_batches": 600, "rows_per_sec": 4899.91, "timing_avg_sec": {"forward": 0.0441, "merge": 0.0, "move": 0.0005, "post": 0.0011}}
2026-05-10 14:48:59.965
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 590, "elapsed_sec": 36.427, "predictions": 178088, "recent_rows_per_sec": 5570.51, "row_group_batches": 590, "rows_per_sec": 4888.96, "timing_avg_sec": {"forward": 0.0439, "merge": 0.0, "move": 0.0005, "post": 0.0011}}
2026-05-10 14:48:59.413
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 580, "elapsed_sec": 35.865, "predictions": 174961, "recent_rows_per_sec": 5793.46, "row_group_batches": 580, "rows_per_sec": 4878.29, "timing_avg_sec": {"forward": 0.0438, "merge": 0.0, "move": 0.0005, "post": 0.001}}
2026-05-10 14:48:58.852
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 570, "elapsed_sec": 35.334, "predictions": 171883, "recent_rows_per_sec": 5611.56, "row_group_batches": 570, "rows_per_sec": 4864.53, "timing_avg_sec": {"forward": 0.0437, "merge": 0.0, "move": 0.0005, "post": 0.001}}
2026-05-10 14:48:58.320
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 560, "elapsed_sec": 34.795, "predictions": 168858, "recent_rows_per_sec": 5714.44, "row_group_batches": 560, "rows_per_sec": 4852.96, "timing_avg_sec": {"forward": 0.0436, "merge": 0.0, "move": 0.0005, "post": 0.001}}
2026-05-10 14:48:57.781
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 550, "elapsed_sec": 34.269, "predictions": 165855, "recent_rows_per_sec": 5339.44, "row_group_batches": 550, "rows_per_sec": 4839.75, "timing_avg_sec": {"forward": 0.0435, "merge": 0.0, "move": 0.0005, "post": 0.001}}
2026-05-10 14:48:57.256
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 540, "elapsed_sec": 33.705, "predictions": 162840, "recent_rows_per_sec": 5674.36, "row_group_batches": 540, "rows_per_sec": 4831.38, "timing_avg_sec": {"forward": 0.0433, "merge": 0.0, "move": 0.0005, "post": 0.0009}}
2026-05-10 14:48:56.691
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 530, "elapsed_sec": 33.164, "predictions": 159770, "recent_rows_per_sec": 5225.71, "row_group_batches": 530, "rows_per_sec": 4817.62, "timing_avg_sec": {"forward": 0.0432, "merge": 0.0, "move": 0.0005, "post": 0.0009}}
2026-05-10 14:48:56.151
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 520, "elapsed_sec": 32.565, "predictions": 156642, "recent_rows_per_sec": 5837.17, "row_group_batches": 520, "rows_per_sec": 4810.12, "timing_avg_sec": {"forward": 0.0429, "merge": 0.0, "move": 0.0005, "post": 0.0009}}
2026-05-10 14:48:55.551
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 510, "elapsed_sec": 32.039, "predictions": 153571, "recent_rows_per_sec": 5356.29, "row_group_batches": 510, "rows_per_sec": 4793.26, "timing_avg_sec": {"forward": 0.0428, "merge": 0.0, "move": 0.0005, "post": 0.0009}}
2026-05-10 14:48:55.025
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 500, "elapsed_sec": 31.463, "predictions": 150484, "recent_rows_per_sec": 5708.39, "row_group_batches": 500, "rows_per_sec": 4782.94, "timing_avg_sec": {"forward": 0.0425, "merge": 0.0, "move": 0.0005, "post": 0.0008}}
2026-05-10 14:48:54.449
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 490, "elapsed_sec": 30.926, "predictions": 147420, "recent_rows_per_sec": 5399.7, "row_group_batches": 490, "rows_per_sec": 4766.88, "timing_avg_sec": {"forward": 0.0424, "merge": 0.0, "move": 0.0005, "post": 0.0008}}
2026-05-10 14:48:53.912
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 480, "elapsed_sec": 30.359, "predictions": 144357, "recent_rows_per_sec": 5555.33, "row_group_batches": 480, "rows_per_sec": 4755.06, "timing_avg_sec": {"forward": 0.0422, "merge": 0.0, "move": 0.0005, "post": 0.0008}}
2026-05-10 14:48:53.345
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 470, "elapsed_sec": 29.821, "predictions": 141372, "recent_rows_per_sec": 5244.61, "row_group_batches": 470, "rows_per_sec": 4740.64, "timing_avg_sec": {"forward": 0.042, "merge": 0.0, "move": 0.0005, "post": 0.0007}}
2026-05-10 14:48:52.808
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 460, "elapsed_sec": 29.261, "predictions": 138431, "recent_rows_per_sec": 5655.06, "row_group_batches": 460, "rows_per_sec": 4730.98, "timing_avg_sec": {"forward": 0.0418, "merge": 0.0, "move": 0.0005, "post": 0.0007}}
2026-05-10 14:48:52.247
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 450, "elapsed_sec": 28.74, "predictions": 135485, "recent_rows_per_sec": 5567.71, "row_group_batches": 450, "rows_per_sec": 4714.23, "timing_avg_sec": {"forward": 0.0416, "merge": 0.0, "move": 0.0005, "post": 0.0006}}
2026-05-10 14:48:51.726
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 440, "elapsed_sec": 28.206, "predictions": 132513, "recent_rows_per_sec": 5195.61, "row_group_batches": 440, "rows_per_sec": 4698.08, "timing_avg_sec": {"forward": 0.0414, "merge": 0.0, "move": 0.0005, "post": 0.0006}}
2026-05-10 14:48:51.192
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 430, "elapsed_sec": 27.629, "predictions": 129515, "recent_rows_per_sec": 5790.17, "row_group_batches": 430, "rows_per_sec": 4687.69, "timing_avg_sec": {"forward": 0.0411, "merge": 0.0, "move": 0.0005, "post": 0.0006}}
2026-05-10 14:48:50.615
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 420, "elapsed_sec": 27.112, "predictions": 126521, "recent_rows_per_sec": 5289.92, "row_group_batches": 420, "rows_per_sec": 4666.66, "timing_avg_sec": {"forward": 0.0409, "merge": 0.0, "move": 0.0005, "post": 0.0005}}
2026-05-10 14:48:50.098
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 410, "elapsed_sec": 26.546, "predictions": 123527, "recent_rows_per_sec": 5963.44, "row_group_batches": 410, "rows_per_sec": 4653.37, "timing_avg_sec": {"forward": 0.0406, "merge": 0.0, "move": 0.0005, "post": 0.0005}}
2026-05-10 14:48:49.532
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 400, "elapsed_sec": 26.047, "predictions": 120555, "recent_rows_per_sec": 7484.44, "row_group_batches": 400, "rows_per_sec": 4628.31, "timing_avg_sec": {"forward": 0.0405, "merge": 0.0, "move": 0.0005, "post": 0.0005}}
2026-05-10 14:48:49.034
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 390, "elapsed_sec": 25.648, "predictions": 117564, "recent_rows_per_sec": 6985.72, "row_group_batches": 390, "rows_per_sec": 4583.8, "timing_avg_sec": {"forward": 0.0405, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:48.634
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 380, "elapsed_sec": 25.227, "predictions": 114625, "recent_rows_per_sec": 5781.24, "row_group_batches": 380, "rows_per_sec": 4543.75, "timing_avg_sec": {"forward": 0.0406, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:48.213
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 370, "elapsed_sec": 24.725, "predictions": 111723, "recent_rows_per_sec": 6990.74, "row_group_batches": 370, "rows_per_sec": 4518.62, "timing_avg_sec": {"forward": 0.0406, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:47.711
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 360, "elapsed_sec": 24.307, "predictions": 108801, "recent_rows_per_sec": 7228.4, "row_group_batches": 360, "rows_per_sec": 4476.11, "timing_avg_sec": {"forward": 0.0407, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:47.293
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 350, "elapsed_sec": 23.903, "predictions": 105882, "recent_rows_per_sec": 6139.06, "row_group_batches": 350, "rows_per_sec": 4429.61, "timing_avg_sec": {"forward": 0.0409, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:46.889
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 340, "elapsed_sec": 23.43, "predictions": 102977, "recent_rows_per_sec": 6243.7, "row_group_batches": 340, "rows_per_sec": 4395.09, "timing_avg_sec": {"forward": 0.041, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:46.416
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 330, "elapsed_sec": 22.959, "predictions": 100035, "recent_rows_per_sec": 6995.3, "row_group_batches": 330, "rows_per_sec": 4357.15, "timing_avg_sec": {"forward": 0.0412, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:45.945
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 320, "elapsed_sec": 22.534, "predictions": 97066, "recent_rows_per_sec": 5964.02, "row_group_batches": 320, "rows_per_sec": 4307.46, "timing_avg_sec": {"forward": 0.0412, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:45.521
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 310, "elapsed_sec": 22.03, "predictions": 94056, "recent_rows_per_sec": 7661.04, "row_group_batches": 310, "rows_per_sec": 4269.51, "timing_avg_sec": {"forward": 0.0414, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:45.016
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 300, "elapsed_sec": 21.631, "predictions": 90999, "recent_rows_per_sec": 5064.16, "row_group_batches": 300, "rows_per_sec": 4206.94, "timing_avg_sec": {"forward": 0.0415, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:44.617
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 290, "elapsed_sec": 21.021, "predictions": 87910, "recent_rows_per_sec": 6622.33, "row_group_batches": 290, "rows_per_sec": 4182.07, "timing_avg_sec": {"forward": 0.0415, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:44.007
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 280, "elapsed_sec": 20.554, "predictions": 84821, "recent_rows_per_sec": 5874.98, "row_group_batches": 280, "rows_per_sec": 4126.69, "timing_avg_sec": {"forward": 0.0414, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:43.540
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 270, "elapsed_sec": 20.034, "predictions": 81765, "recent_rows_per_sec": 6300.9, "row_group_batches": 270, "rows_per_sec": 4081.3, "timing_avg_sec": {"forward": 0.0415, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:43.020
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 260, "elapsed_sec": 19.548, "predictions": 78701, "recent_rows_per_sec": 5673.02, "row_group_batches": 260, "rows_per_sec": 4026.08, "timing_avg_sec": {"forward": 0.0415, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:42.534
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 250, "elapsed_sec": 19.013, "predictions": 75665, "recent_rows_per_sec": 5418.69, "row_group_batches": 250, "rows_per_sec": 3979.73, "timing_avg_sec": {"forward": 0.0415, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:41.999
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 240, "elapsed_sec": 18.455, "predictions": 72645, "recent_rows_per_sec": 6883.05, "row_group_batches": 240, "rows_per_sec": 3936.27, "timing_avg_sec": {"forward": 0.0414, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:41.442
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 230, "elapsed_sec": 18.016, "predictions": 69618, "recent_rows_per_sec": 5076.32, "row_group_batches": 230, "rows_per_sec": 3864.34, "timing_avg_sec": {"forward": 0.0414, "merge": 0.0, "move": 0.0006, "post": 0.0005}}
2026-05-10 14:48:41.002
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 220, "elapsed_sec": 17.426, "predictions": 66627, "recent_rows_per_sec": 5983.19, "row_group_batches": 220, "rows_per_sec": 3823.36, "timing_avg_sec": {"forward": 0.0414, "merge": 0.0, "move": 0.0007, "post": 0.0005}}
2026-05-10 14:48:40.413
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 210, "elapsed_sec": 16.934, "predictions": 63679, "recent_rows_per_sec": 7152.21, "row_group_batches": 210, "rows_per_sec": 3760.51, "timing_avg_sec": {"forward": 0.0415, "merge": 0.0, "move": 0.0004, "post": 0.0005}}
2026-05-10 14:48:39.920
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 200, "elapsed_sec": 16.522, "predictions": 60734, "recent_rows_per_sec": 5021.42, "row_group_batches": 200, "rows_per_sec": 3675.98, "timing_avg_sec": {"forward": 0.0418, "merge": 0.0, "move": 0.0004, "post": 0.0005}}
2026-05-10 14:48:39.508
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 190, "elapsed_sec": 15.933, "predictions": 57776, "recent_rows_per_sec": 4975.66, "row_group_batches": 190, "rows_per_sec": 3626.24, "timing_avg_sec": {"forward": 0.0414, "merge": 0.0, "move": 0.0005, "post": 0.0005}}
2026-05-10 14:48:38.919
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 180, "elapsed_sec": 15.338, "predictions": 54817, "recent_rows_per_sec": 5678.93, "row_group_batches": 180, "rows_per_sec": 3573.92, "timing_avg_sec": {"forward": 0.0413, "merge": 0.0, "move": 0.0005, "post": 0.0005}}
2026-05-10 14:48:38.324
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 170, "elapsed_sec": 14.818, "predictions": 51862, "recent_rows_per_sec": 5458.87, "row_group_batches": 170, "rows_per_sec": 3500.0, "timing_avg_sec": {"forward": 0.0408, "merge": 0.0, "move": 0.0005, "post": 0.0005}}
2026-05-10 14:48:37.804
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 160, "elapsed_sec": 14.275, "predictions": 48901, "recent_rows_per_sec": 6398.38, "row_group_batches": 160, "rows_per_sec": 3425.57, "timing_avg_sec": {"forward": 0.0413, "merge": 0.0, "move": 0.0005, "post": 0.0005}}
2026-05-10 14:48:37.262
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 150, "elapsed_sec": 13.806, "predictions": 45900, "recent_rows_per_sec": 5270.21, "row_group_batches": 150, "rows_per_sec": 3324.58, "timing_avg_sec": {"forward": 0.0413, "merge": 0.0, "move": 0.0005, "post": 0.0004}}
2026-05-10 14:48:36.793
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 140, "elapsed_sec": 13.23, "predictions": 42862, "recent_rows_per_sec": 6182.35, "row_group_batches": 140, "rows_per_sec": 3239.8, "timing_avg_sec": {"forward": 0.041, "merge": 0.0, "move": 0.0005, "post": 0.0004}}
2026-05-10 14:48:36.216
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 130, "elapsed_sec": 12.744, "predictions": 39857, "recent_rows_per_sec": 5065.01, "row_group_batches": 130, "rows_per_sec": 3127.57, "timing_avg_sec": {"forward": 0.0419, "merge": 0.0, "move": 0.0005, "post": 0.0004}}
2026-05-10 14:48:35.730
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 120, "elapsed_sec": 12.15, "predictions": 36850, "recent_rows_per_sec": 5901.85, "row_group_batches": 120, "rows_per_sec": 3032.9, "timing_avg_sec": {"forward": 0.0418, "merge": 0.0, "move": 0.0005, "post": 0.0004}}
2026-05-10 14:48:35.136
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 110, "elapsed_sec": 11.636, "predictions": 33814, "recent_rows_per_sec": 6079.9, "row_group_batches": 110, "rows_per_sec": 2906.06, "timing_avg_sec": {"forward": 0.0423, "merge": 0.0, "move": 0.0005, "post": 0.0004}}
2026-05-10 14:48:34.622
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 100, "elapsed_sec": 11.133, "predictions": 30758, "recent_rows_per_sec": 6149.1, "row_group_batches": 100, "rows_per_sec": 2762.77, "timing_avg_sec": {"forward": 0.0432, "merge": 0.0, "move": 0.0005, "post": 0.0004}}
2026-05-10 14:48:34.119
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 90, "elapsed_sec": 10.638, "predictions": 27714, "recent_rows_per_sec": 5480.97, "row_group_batches": 90, "rows_per_sec": 2605.19, "timing_avg_sec": {"forward": 0.0429, "merge": 0.0, "move": 0.0005, "post": 0.0004}}
2026-05-10 14:48:33.624
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 80, "elapsed_sec": 10.081, "predictions": 24661, "recent_rows_per_sec": 7383.45, "row_group_batches": 80, "rows_per_sec": 2446.29, "timing_avg_sec": {"forward": 0.0431, "merge": 0.0, "move": 0.0005, "post": 0.0004}}
2026-05-10 14:48:33.067
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 70, "elapsed_sec": 9.668, "predictions": 21610, "recent_rows_per_sec": 5859.14, "row_group_batches": 70, "rows_per_sec": 2235.27, "timing_avg_sec": {"forward": 0.0439, "merge": 0.0, "move": 0.0005, "post": 0.0005}}
2026-05-10 14:48:32.654
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 60, "elapsed_sec": 9.147, "predictions": 18559, "recent_rows_per_sec": 7190.95, "row_group_batches": 60, "rows_per_sec": 2028.96, "timing_avg_sec": {"forward": 0.0447, "merge": 0.0, "move": 0.0006, "post": 0.0004}}
2026-05-10 14:48:32.133
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 50, "elapsed_sec": 8.721, "predictions": 15497, "recent_rows_per_sec": 7393.1, "row_group_batches": 50, "rows_per_sec": 1776.93, "timing_avg_sec": {"forward": 0.0456, "merge": 0.0, "move": 0.0006, "post": 0.0004}}
2026-05-10 14:48:31.707
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 40, "elapsed_sec": 8.303, "predictions": 12408, "recent_rows_per_sec": 6401.12, "row_group_batches": 40, "rows_per_sec": 1494.33, "timing_avg_sec": {"forward": 0.0479, "merge": 0.0, "move": 0.0006, "post": 0.0004}}
2026-05-10 14:48:31.290
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 30, "elapsed_sec": 7.82, "predictions": 9312, "recent_rows_per_sec": 7992.18, "row_group_batches": 30, "rows_per_sec": 1190.83, "timing_avg_sec": {"forward": 0.0484, "merge": 0.0, "move": 0.0007, "post": 0.0004}}
2026-05-10 14:48:30.806
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 20, "elapsed_sec": 7.435, "predictions": 6237, "recent_rows_per_sec": 7485.19, "row_group_batches": 20, "rows_per_sec": 838.87, "timing_avg_sec": {"forward": 0.0544, "merge": 0.0, "move": 0.0009, "post": 0.0004}}
2026-05-10 14:48:30.421
PLATFORM_INFER_PROGRESS {"amp_mode": "bf16", "batches": 10, "elapsed_sec": 7.021, "predictions": 3138, "recent_rows_per_sec": 446.95, "row_group_batches": 10, "rows_per_sec": 446.95, "timing_avg_sec": {"forward": 0.0688, "merge": 0.0, "move": 0.0014, "post": 0.0004}}
2026-05-10 14:48:30.007
PLATFORM_FIRST_BATCH {"batch_predictions": 314, "rows_in_forward": 314, "sample_scores": [0.00421143, 0.08056641, 0.0378418, 0.18554688, 0.44140625], "sample_user_ids": ["683974", "1334888", "51348", "1184407", "8854"], "sanitize_counts": {"nan_replaced": 0, "neginf_replaced": 0, "posinf_replaced": 0}, "timing_sec": {"forward": 0.3269, "merge": 0.0, "move": 0.0108, "post": 0.0008}}
2026-05-10 14:48:29.636
2026-05-10 06:48:28,162 - INFO - NumExpr defaulting to 16 threads.
2026-05-10 14:48:28.162
2026-05-10 06:48:28,162 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-10 14:48:28.162
2026-05-10 06:48:28,162 - INFO - NumExpr defaulting to 16 threads.
2026-05-10 14:48:28.162
2026-05-10 06:48:28,161 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-10 14:48:28.162
2026-05-10 06:48:28,161 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-10 14:48:28.162
2026-05-10 06:48:28,161 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-10 14:48:28.161
2026-05-10 06:48:28,161 - INFO - NumExpr defaulting to 16 threads.
2026-05-10 14:48:28.161
2026-05-10 06:48:28,161 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-10 14:48:28.161
2026-05-10 06:48:28,161 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-10 14:48:28.161
2026-05-10 06:48:28,161 - INFO - NumExpr defaulting to 16 threads.
2026-05-10 14:48:28.161
2026-05-10 06:48:28,161 - INFO - Note: NumExpr detected 384 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
2026-05-10 14:48:28.161
2026-05-10 06:48:28,161 - INFO - Note: detected 384 virtual cores but NumExpr set to maximum of 64, check "NUMEXPR_MAX_THREADS" environment variable.
2026-05-10 14:48:28.161
PLATFORM_AMP_CONFIG {"cuda_bf16_supported": true, "enabled": true, "mode": "bf16", "requested": "bf16"}
2026-05-10 14:48:27.632
2026-05-10 06:48:27,632 - INFO - Starting inference...
2026-05-10 14:48:27.632
PLATFORM_MODEL_READY {"device": "cuda", "num_ns": 8, "num_parameters": 239931649}
2026-05-10 14:48:27.632
2026-05-10 06:48:27,631 - INFO - Model loaded successfully
2026-05-10 14:48:27.631
2026-05-10 06:48:25,890 - INFO - Loading checkpoint from /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt
2026-05-10 14:48:25.890
2026-05-10 06:48:25,610 - INFO - emb_skip_threshold=1000000: seq_c skipped 3/11 features
2026-05-10 14:48:25.610
2026-05-10 06:48:25,610 - INFO - emb_skip_threshold=1000000: seq_b skipped 1/13 features
2026-05-10 14:48:25.610
2026-05-10 06:48:23,534 - INFO - RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0
2026-05-10 14:48:23.534
2026-05-10 06:48:23,516 - INFO - RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1
2026-05-10 14:48:23.516
2026-05-10 06:48:23,493 - INFO - Building PCVRHyFormer with cfg: {'d_model': 64, 'emb_dim': 64, 'num_queries': 2, 'num_hyformer_blocks': 2, 'num_heads': 4, 'seq_encoder_type': 'transformer', 'hidden_mult': 4, 'dropout_rate': 0.01, 'seq_top_k': 50, 'seq_causal': False, 'action_num': 1, 'num_time_buckets': 64, 'rank_mixer_mode': 'full', 'use_rope': False, 'rope_base': 10000.0, 'emb_skip_threshold': 1000000, 'seq_id_threshold': 10000, 'ns_tokenizer_type': 'rankmixer', 'user_ns_tokens': 5, 'item_ns_tokens': 2, 'use_wide_dense_tokens': False, 'wide_dense_threshold': 32, 'dense_scalar_transform': 'none', 'xdomain_dense_dim': 0, 'use_xdomain_features': False, 'use_inter_time_buckets': False, 'use_continuous_time': False, 'high_cardinality_mode': 'zero', 'hash_num_buckets': 1048576, 'hash_num_hashes': 2, 'use_target_aware_attn': False, 'use_cross_domain_attn': False, 'seq_topk_mode': 'recent', 'ffn_type': 'gelu', 'norm_type': 'layernorm', 'random_id_mask_prob': 0.0, 'legacy_user_dense_proj': True, 'legacy_mixer_ffn_names': True}
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - INFO - No NS groups JSON found, using default: each feature as one group
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'legacy_mixer_ffn_names', using fallback = False
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'legacy_user_dense_proj', using fallback = False
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'random_id_mask_prob', using fallback = 0.0
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'norm_type', using fallback = layernorm
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'ffn_type', using fallback = gelu
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'seq_topk_mode', using fallback = recent
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'use_cross_domain_attn', using fallback = False
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'use_target_aware_attn', using fallback = False
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'hash_num_hashes', using fallback = 2
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'hash_num_buckets', using fallback = 1048576
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'high_cardinality_mode', using fallback = zero
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'use_continuous_time', using fallback = False
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'use_inter_time_buckets', using fallback = False
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'use_xdomain_features', using fallback = False
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'xdomain_dense_dim', using fallback = 0
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'dense_scalar_transform', using fallback = none
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'wide_dense_threshold', using fallback = 32
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - WARNING - train_config missing 'use_wide_dense_tokens', using fallback = False
2026-05-10 14:48:23.493
PLATFORM_DATASET_CONFIG {"batch_size": 2048, "item_dense_dim": 0, "item_int_features": 14, "num_row_groups": 1000, "num_rows_estimate": 310000, "seq_domains": ["seq_a", "seq_b", "seq_c", "seq_d"], "seq_max_lens": {"seq_a": 256, "seq_b": 256, "seq_c": 256, "seq_d": 256}, "user_dense_dim": 918, "user_int_features": 46}
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - INFO - Total test samples: 310000
2026-05-10 14:48:23.493
2026-05-10 06:48:23,493 - INFO - PCVRParquetDataset: 310000 rows from 1000 file(s), batch_size=2048, buffer_batches=0, shuffle=False
2026-05-10 14:48:23.493
PLATFORM_INFER_CONFIG {"amp_dtype": "bf16", "batch_size": 2048, "checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "device": "cuda", "eval_data_dir": "/data_ams/academic_infer_data/", "merge_batches": false, "num_workers": 4, "progress_every": 10, "result_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/78557/results", "schema_path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "seq_max_lens": {"seq_a": 256, "seq_b": 256, "seq_c": 256, "seq_d": 256}, "train_config": {"d_model": 64, "emb_skip_threshold": 1000000, "high_cardinality_mode": null, "num_heads": 4, "num_hyformer_blocks": 2, "num_queries": 2, "seq_encoder_type": "transformer", "train_batch_size": 256}, "use_amp": true}
2026-05-10 14:48:22.995
PLATFORM_CHECKPOINT_STATUS {"checkpoint_dir": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model", "model_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/model.pt", "size_bytes": 959871995}, "schema_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json", "size_bytes": 5560}, "train_config_file": {"exists": true, "path": "/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/train_config.json", "size_bytes": 1638}}
2026-05-10 14:48:22.995
2026-05-10 06:48:22,995 - INFO - seq_max_lens: {'seq_a': 256, 'seq_b': 256, 'seq_c': 256, 'seq_d': 256}
2026-05-10 14:48:22.995
2026-05-10 06:48:22,994 - INFO - Loaded train_config from /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/train_config.json
2026-05-10 14:48:22.995
2026-05-10 06:48:22,992 - INFO - Using schema: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260506155640_b95bb20f/self/95cdb55f9dfb731c019dfc4a37ba02dc/ckpt/global_step21744.layer=2.head=4.hidden=64.best_model/schema.json
2026-05-10 14:48:22.992
PLATFORM_INFER_START {"cuda_available": true, "cwd": "/workspace", "time": "2026-05-10 06:48:22", "torch_version": "2.7.1+cu126"}
2026-05-10 14:48:22.989
====== Inferring ======
2026-05-10 14:48:21.206
=================================
2026-05-10 14:48:20.778
Working Dir: /workspace
2026-05-10 14:48:20.778
Environment: competition
2026-05-10 14:48:20.778
GPU Count: 1
2026-05-10 14:48:20.777
GPU Available: True
2026-05-10 14:48:19.016
PyTorch: 2.7.1+cu126
2026-05-10 14:48:17.194
[DEBUG][libvgpu]hijack_call.c:175 [p:79 t:79]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1
2026-05-10 14:48:14.824
[DEBUG][libvgpu]hijack_call.c:174 [p:79 t:79]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1
2026-05-10 14:48:14.824
[DEBUG][libvgpu]hijack_call.c:173 [p:79 t:79]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-10 14:48:14.824
[DEBUG][libvgpu]hijack_call.c:172 [p:79 t:79]hooked env NCCL_SET_THREAD_NAME to : 1
2026-05-10 14:48:14.824
[DEBUG][libvgpu]hijack_call.c:125 [p:79 t:79]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
2026-05-10 14:48:14.821
[DEBUG][libvgpu]hijack_call.c:107 [p:79 t:79]Thread pid:79, tid:79
2026-05-10 14:48:14.821
[DEBUG][libvgpu]hijack_call.c:106 [p:79 t:79]init cuda hook lib
2026-05-10 14:48:14.821
Python: Python 3.10.20
2026-05-10 14:48:14.083
CUDA: 12.6.77
2026-05-10 14:48:14.078
=== Competition Environment Ready ===
2026-05-10 14:48:14.070
Complete setting network policy rules.
2026-05-10 14:48:10.283
Complete setting taiji user.
2026-05-10 14:48:10.174
Truncated to last 1000 lines.