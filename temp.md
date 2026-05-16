2026-05-15 19:10:52.116 ##########################################################################

2026-05-15 19:10:52.116 ############################ TAIJI JOB START #############################

2026-05-15 19:10:52.116 ##########################################################################

2026-05-15 19:10:52.135 cp: cannot stat './/*': No such file or directory

2026-05-15 19:10:52.146 passwd: password expiry information changed.

2026-05-15 19:10:57.736 Complete setting taiji user.

2026-05-15 19:11:03.485 update taiji environ ...

2026-05-15 19:11:03.485 show args: --appName=95a8ce1c9e20cbf6019e2b54e93b193f --projectName=external-ams-competition-2025 --module=pytorch --script=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/run.sh --script_archive=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/dataset.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/model.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/ns_groups.json,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/train.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/trainer.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/utils.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/.taiji_run.json --script_args=

2026-05-15 19:11:03.587 INFO: ===begin to run init cmd===

2026-05-15 19:11:03.587 INFO: Export env: RUNTIME_SCRIPT_DIR=/home/taiji/dl/runtime/script

2026-05-15 19:11:03.593 INFO: Export env: RUNTIME_SCRIPT_SHARE_DIR=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/pipeline/95a8ce1c9e20cbf6019e2b54e93b193f/share

2026-05-15 19:11:03.593 INFO: script_path :/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/run.sh

2026-05-15 19:11:03.597 INFO: Success copy to /home/taiji/dl/runtime/script/run.sh

2026-05-15 19:11:03.600 INFO: Success copy to /home/taiji/dl/runtime/script/dataset.py

2026-05-15 19:11:03.604 INFO: Success copy to /home/taiji/dl/runtime/script/model.py

2026-05-15 19:11:03.608 INFO: Success copy to /home/taiji/dl/runtime/script/ns_groups.json

2026-05-15 19:11:03.611 INFO: Success copy to /home/taiji/dl/runtime/script/train.py

2026-05-15 19:11:03.615 INFO: Success copy to /home/taiji/dl/runtime/script/trainer.py

2026-05-15 19:11:03.618 INFO: Success copy to /home/taiji/dl/runtime/script/utils.py

2026-05-15 19:11:03.621 INFO: Success copy to /home/taiji/dl/runtime/script/.taiji_run.json

2026-05-15 19:11:03.621 INFO: python path: /home/taiji/dl/runtime/script/dataset.py:/home/taiji/dl/runtime/script/model.py:/home/taiji/dl/runtime/script/ns_groups.json:/home/taiji/dl/runtime/script/train.py:/home/taiji/dl/runtime/script/trainer.py:/home/taiji/dl/runtime/script/utils.py:/home/taiji/dl/runtime/script/.taiji_run.json

2026-05-15 19:11:03.621 INFO: Export env: TRAIN_CKPT_PATH=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/ckpt

2026-05-15 19:11:03.621 INFO: Export env: TRAIN_TF_EVENTS_PATH=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/events

2026-05-15 19:11:03.621 INFO: Export env: TRAIN_DATA_PATH=/data_ams/academic_training_data

2026-05-15 19:11:03.621 INFO: Export env: TRAIN_USER=ams_2026_1029731852466347124

2026-05-15 19:11:03.621 INFO: Export env: TRAIN_LOG_PATH=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/log

2026-05-15 19:11:03.621 INFO: Export env: TRAIN_CACHE_PATH=/apdcephfs_fsgm2/share_305170765/angel/groups/44/ams_2026_1029731852466347124

2026-05-15 19:11:03.621 INFO: Export env: USER_CACHE_PATH=/apdcephfs_fsgm2/share_305170765/angel/groups/44/ams_2026_1029731852466347124

2026-05-15 19:11:03.621 INFO: Export env: NP_BD_FLOW_PROJ=8054023

2026-05-15 19:11:03.632 touch: cannot touch '/data/initCmdCompletedFile': No such file or directory

2026-05-15 19:11:03.632 initCmd failed

2026-05-15 19:11:03.633 [TAIJI] Starting sshd

2026-05-15 19:11:04.659 [TAIJI] sshd started

2026-05-15 19:11:04.659 [TAIJI] waiting for all peers to be ready...

2026-05-15 19:11:05.983 Welcome to Ubuntu 22.04.5 LTS (GNU/Linux 5.4.241-1-tlinux4-0017.7 x86_64)

2026-05-15 19:11:05.983

2026-05-15 19:11:05.983 * Documentation: https://help.ubuntu.com

2026-05-15 19:11:05.983 * Management: https://landscape.canonical.com

2026-05-15 19:11:05.983 * Support: https://ubuntu.com/pro

2026-05-15 19:11:05.983

2026-05-15 19:11:05.983 This system has been minimized by removing packages and content that are

2026-05-15 19:11:05.983 not required on a system that users do not log into.

2026-05-15 19:11:05.983

2026-05-15 19:11:05.983 To restore this content, you can run the 'unminimize' command.

2026-05-15 19:11:05.983

2026-05-15 19:11:05.983 The programs included with the Ubuntu system are free software;

2026-05-15 19:11:05.983 the exact distribution terms for each program are described in the

2026-05-15 19:11:05.983 individual files in /usr/share/doc/*/copyright.

2026-05-15 19:11:05.983

2026-05-15 19:11:05.983 Ubuntu comes with ABSOLUTELY NO WARRANTY, to the extent permitted by

2026-05-15 19:11:05.983 applicable law.

2026-05-15 19:11:05.983

2026-05-15 19:11:07.134 [TAIJI] error log above can be ignored

2026-05-15 19:11:07.138 [TAIJI] run command 1778843467 [/bin/bash -c "${start_script}"]

2026-05-15 19:11:10.830 Complete setting network policy rules.

2026-05-15 19:11:16.429 update taiji environ ...

2026-05-15 19:11:16.429 show args: --appName=95a8ce1c9e20cbf6019e2b54e93b193f --projectName=external-ams-competition-2025n --module=pytorch --script=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/run.sh --script_archive=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/dataset.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/model.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/ns_groups.json,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/train.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/trainer.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/utils.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/.taiji_run.json --script_args=

2026-05-15 19:11:16.474 INFO: ===begin to run starttorch cmd===

2026-05-15 19:11:16.475 INFO: runtime_script_share_dir is /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/pipeline/95a8ce1c9e20cbf6019e2b54e93b193f/share

2026-05-15 19:11:16.475 INFO: script_path :/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/scripts/95a8ce1c9e20cbf6019e2b54e93b193f/run.sh

2026-05-15 19:11:16.475 INFO: framework is pytorch

2026-05-15 19:11:16.475 INFO: Run command: bash /home/taiji/dl/runtime/script/run.sh

2026-05-15 19:11:16.475 INFO: ====== Begin cmd ======

2026-05-15 19:11:17.877 [DEBUG][libvgpu]hijack_call.c:106 [p:273 t:273]init cuda hook lib

2026-05-15 19:11:17.877 [DEBUG][libvgpu]hijack_call.c:107 [p:273 t:273]Thread pid:273, tid:273

2026-05-15 19:11:17.877 [DEBUG][libvgpu]hijack_call.c:125 [p:273 t:273]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

2026-05-15 19:11:17.880 [DEBUG][libvgpu]hijack_call.c:172 [p:273 t:273]hooked env NCCL_SET_THREAD_NAME to : 1

2026-05-15 19:11:17.880 [DEBUG][libvgpu]hijack_call.c:173 [p:273 t:273]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

2026-05-15 19:11:17.880 [DEBUG][libvgpu]hijack_call.c:174 [p:273 t:273]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1

2026-05-15 19:11:17.880 [DEBUG][libvgpu]hijack_call.c:175 [p:273 t:273]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1

2026-05-15 19:11:21.761 05/15/26 19:11:21 - 0:00:00 - Args: {'data_dir': '/data_ams/academic_training_data', 'schema_path': None, 'ckpt_dir': '/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/ckpt', 'log_dir': '/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/log', 'batch_size': 256, 'lr': 0.0001, 'num_epochs': 999, 'patience': 5, 'seed': 20260515, 'device': 'cuda', 'num_workers': 8, 'buffer_batches': 20, 'train_ratio': 1.0, 'valid_ratio': 0.1, 'eval_every_n_steps': 0, 'log_every_n_steps': 200, 'seq_max_lens': 'seq_a:256,seq_b:256,seq_c:512,seq_d:512', 'd_model': 64, 'emb_dim': 64, 'num_queries': 2, 'num_hyformer_blocks': 2, 'num_heads': 4, 'seq_encoder_type': 'transformer', 'hidden_mult': 4, 'dropout_rate': 0.01, 'seq_top_k': 50, 'seq_causal': False, 'action_num': 1, 'use_time_buckets': True, 'rank_mixer_mode': 'learned', 'use_rope': False, 'rope_base': 10000.0, 'use_wide_dense_tokens': True, 'wide_dense_threshold': 32, 'dense_scalar_transform': 'signlog', 'use_xdomain_features': True, 'primary_seq_fids': '', 'use_inter_time_buckets': True, 'use_continuous_time': True, 'high_cardinality_mode': 'hash', 'hash_num_buckets': 1048576, 'hash_num_hashes': 2, 'use_target_aware_attn': True, 'use_cross_domain_attn': True, 'seq_topk_mode': 'recent', 'ffn_type': 'swiglu', 'norm_type': 'rms', 'random_id_mask_prob': 0.05, 'loss_type': 'focal', 'focal_alpha': 0.25, 'focal_gamma': 1.5, 'focal_alpha_mode': 'fixed', 'focal_alpha_start': 0.5, 'warmup_steps': 1000, 'min_lr_ratio': 0.1, 'use_ema': True, 'ema_decay': 0.999, 'sparse_lr': 0.05, 'sparse_weight_decay': 0.0, 'reinit_sparse_after_epoch': 1, 'reinit_cardinality_threshold': 0, 'emb_skip_threshold': 1000000, 'seq_id_threshold': 10000, 'ns_groups_json': '', 'ns_tokenizer_type': 'rankmixer', 'user_ns_tokens': 5, 'item_ns_tokens': 2, 'tf_events_dir': '/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/events', 'xdomain_dense_dim': 18}

2026-05-15 19:11:22.200 2026-05-15 19:11:22.200767: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

2026-05-15 19:11:22.210 2026-05-15 19:11:22.210071: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered

2026-05-15 19:11:22.221 WARNING: All log messages before absl::InitializeLog() is called are written to STDERR

2026-05-15 19:11:22.221 E0000 00:00:1778843482.221428 269 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered

2026-05-15 19:11:22.224 E0000 00:00:1778843482.224716 269 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered

2026-05-15 19:11:22.234 W0000 00:00:1778843482.234285 269 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.

2026-05-15 19:11:22.234 W0000 00:00:1778843482.234299 269 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.

2026-05-15 19:11:22.234 W0000 00:00:1778843482.234300 269 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.

2026-05-15 19:11:22.234 W0000 00:00:1778843482.234302 269 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.

2026-05-15 19:11:22.237 2026-05-15 19:11:22.237197: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.

2026-05-15 19:11:22.237 To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

2026-05-15 19:11:26.213 05/15/26 19:11:26 - 0:00:04 - generated new fontManager

2026-05-15 19:11:26.628 05/15/26 19:11:26 - 0:00:05 - Seq max_lens override: {'seq_a': 256, 'seq_b': 256, 'seq_c': 512, 'seq_d': 512}

2026-05-15 19:11:26.628 05/15/26 19:11:26 - 0:00:05 - Using Parquet data format (IterableDataset)

2026-05-15 19:11:27.130 05/15/26 19:11:27 - 0:00:05 - Row Group split: 900 train (907381 rows), 100 valid (102619 rows)

2026-05-15 19:11:27.631 05/15/26 19:11:27 - 0:00:06 - PCVRParquetDataset: 907381 rows from 1000 file(s), batch_size=256, buffer_batches=20, shuffle=True

2026-05-15 19:11:28.129 05/15/26 19:11:28 - 0:00:06 - PCVRParquetDataset: 102619 rows from 1000 file(s), batch_size=256, buffer_batches=0, shuffle=False

2026-05-15 19:11:28.129 05/15/26 19:11:28 - 0:00:06 - Parquet train: 907381 rows, valid: 102619 rows, batch_size=256, buffer_batches=20

2026-05-15 19:11:28.130 05/15/26 19:11:28 - 0:00:06 - No NS groups JSON found, using default: each feature as one group

2026-05-15 19:11:28.141 05/15/26 19:11:28 - 0:00:06 - RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1

2026-05-15 19:11:28.159 05/15/26 19:11:28 - 0:00:06 - RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0

2026-05-15 19:11:37.966 05/15/26 19:11:37 - 0:00:16 - PCVRHyFormer model created: num_ns=13, T=21, d_model=64, rank_mixer_mode=learned

2026-05-15 19:11:37.966 05/15/26 19:11:37 - 0:00:16 - User NS groups: [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35], [36], [37], [38], [39], [40], [41], [42], [43], [44], [45]]

2026-05-15 19:11:37.966 05/15/26 19:11:37 - 0:00:16 - Item NS groups: [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13]]

2026-05-15 19:11:37.967 05/15/26 19:11:37 - 0:00:16 - Total parameters: 778,116,085

2026-05-15 19:11:37.972 05/15/26 19:11:37 - 0:00:16 - Sparse params: 107 tensors, 774,311,296 parameters (Adagrad lr=0.05)

2026-05-15 19:11:37.972 05/15/26 19:11:37 - 0:00:16 - Dense params: 604 tensors, 3,804,789 parameters (AdamW lr=0.0001)

2026-05-15 19:11:40.558 05/15/26 19:11:40 - 0:00:19 - EMA enabled: 711 tensors, decay=0.999

2026-05-15 19:11:40.558 05/15/26 19:11:40 - 0:00:19 - PCVRHyFormerRankingTrainer loss_type=focal, focal_alpha=0.25, focal_gamma=1.5, focal_alpha_mode=fixed, reinit_sparse_after_epoch=1

2026-05-15 19:11:40.558 05/15/26 19:11:40 - 0:00:19 - TRAIN_START model=PCVRHyFormer epochs=999 train_steps_per_epoch=3733 valid_steps=461 log_every_n_steps=200

2026-05-15 19:11:40.560 05/15/26 19:11:40 - 0:00:19 - EPOCH_START epoch=1

2026-05-15 19:14:40.691 05/15/26 19:14:40 - 0:03:19 - TRAIN_PROGRESS epoch=1 step=200/3733 total_step=200 avg_loss=0.036686 last_loss=0.037546 dense_lr=2e-05

2026-05-15 19:17:30.658 05/15/26 19:17:30 - 0:06:09 - TRAIN_PROGRESS epoch=1 step=400/3733 total_step=400 avg_loss=0.034447 last_loss=0.037555 dense_lr=4e-05

2026-05-15 19:20:23.492 05/15/26 19:20:23 - 0:09:02 - TRAIN_PROGRESS epoch=1 step=600/3733 total_step=600 avg_loss=0.033477 last_loss=0.028304 dense_lr=6e-05

2026-05-15 19:23:15.327 05/15/26 19:23:15 - 0:11:54 - TRAIN_PROGRESS epoch=1 step=800/3733 total_step=800 avg_loss=0.033008 last_loss=0.027382 dense_lr=8e-05

2026-05-15 19:26:07.736 05/15/26 19:26:07 - 0:14:46 - TRAIN_PROGRESS epoch=1 step=1000/3733 total_step=1000 avg_loss=0.032702 last_loss=0.029353 dense_lr=0.0001

2026-05-15 19:29:00.320 05/15/26 19:29:00 - 0:17:39 - TRAIN_PROGRESS epoch=1 step=1200/3733 total_step=1200 avg_loss=0.032328 last_loss=0.034365 dense_lr=9.9999999e-05

2026-05-15 19:32:36.887 05/15/26 19:32:36 - 0:21:15 - TRAIN_PROGRESS epoch=1 step=1400/3733 total_step=1400 avg_loss=0.032122 last_loss=0.032033 dense_lr=9.9999997e-05

2026-05-15 19:36:12.387 05/15/26 19:36:12 - 0:24:51 - TRAIN_PROGRESS epoch=1 step=1600/3733 total_step=1600 avg_loss=0.031868 last_loss=0.029477 dense_lr=9.9999994e-05

2026-05-15 19:39:18.839 05/15/26 19:39:18 - 0:27:57 - TRAIN_PROGRESS epoch=1 step=1800/3733 total_step=1800 avg_loss=0.031693 last_loss=0.029242 dense_lr=9.999999e-05

2026-05-15 19:42:17.833 05/15/26 19:42:17 - 0:30:56 - TRAIN_PROGRESS epoch=1 step=2000/3733 total_step=2000 avg_loss=0.031589 last_loss=0.028554 dense_lr=9.9999984e-05

2026-05-15 19:46:00.897 05/15/26 19:46:00 - 0:34:39 - TRAIN_PROGRESS epoch=1 step=2200/3733 total_step=2200 avg_loss=0.031464 last_loss=0.031393 dense_lr=9.9999977e-05

2026-05-15 19:49:48.068 05/15/26 19:49:48 - 0:38:26 - TRAIN_PROGRESS epoch=1 step=2400/3733 total_step=2400 avg_loss=0.031379 last_loss=0.025821 dense_lr=9.9999969e-05

2026-05-15 19:53:36.643 05/15/26 19:53:36 - 0:42:15 - TRAIN_PROGRESS epoch=1 step=2600/3733 total_step=2600 avg_loss=0.031294 last_loss=0.020450 dense_lr=9.9999959e-05

2026-05-15 19:57:22.410 05/15/26 19:57:22 - 0:46:01 - TRAIN_PROGRESS epoch=1 step=2800/3733 total_step=2800 avg_loss=0.031197 last_loss=0.032685 dense_lr=9.9999948e-05

2026-05-15 20:01:11.560 05/15/26 20:01:11 - 0:49:50 - TRAIN_PROGRESS epoch=1 step=3000/3733 total_step=3000 avg_loss=0.031124 last_loss=0.032334 dense_lr=9.9999936e-05

2026-05-15 20:05:00.226 05/15/26 20:05:00 - 0:53:38 - TRAIN_PROGRESS epoch=1 step=3200/3733 total_step=3200 avg_loss=0.031053 last_loss=0.026345 dense_lr=9.9999923e-05

2026-05-15 20:08:50.997 05/15/26 20:08:50 - 0:57:29 - TRAIN_PROGRESS epoch=1 step=3400/3733 total_step=3400 avg_loss=0.030997 last_loss=0.031269 dense_lr=9.9999908e-05

2026-05-15 20:12:37.025 05/15/26 20:12:37 - 1:01:15 - TRAIN_PROGRESS epoch=1 step=3600/3733 total_step=3600 avg_loss=0.030968 last_loss=0.023813 dense_lr=9.9999892e-05

2026-05-15 20:13:04.555 05/15/26 20:13:04 - 1:01:43 - EPOCH_TRAIN_RESULT epoch=1 total_step=3624 avg_loss=0.030049

2026-05-15 20:13:04.555 05/15/26 20:13:04 - 1:01:43 - VALIDATION_START epoch=1 total_step=3624 reason=epoch

2026-05-15 20:13:04.555 05/15/26 20:13:04 - 1:01:43 - VALIDATION_LOOP_START epoch=1 valid_steps=461

2026-05-15 20:19:07.393 05/15/26 20:19:07 - 1:07:46 - Validation extra metrics | GAUC: 0.0, GAUC_groups: 0

2026-05-15 20:19:07.404 05/15/26 20:19:07 - 1:07:46 - Validation hist_bucket_0 | rows=25723, AUC=0.8883172619411883

2026-05-15 20:19:07.493 05/15/26 20:19:07 - 1:07:46 - Validation hist_bucket_1 | rows=25662, AUC=0.8592810195488135

2026-05-15 20:19:07.504 05/15/26 20:19:07 - 1:07:46 - Validation hist_bucket_2 | rows=51234, AUC=0.8456097792679336

2026-05-15 20:19:07.540 05/15/26 20:19:07 - 1:07:46 - VALIDATION_RESULT epoch=1 total_step=3624 auc=0.860589215769 logloss=0.263686537743 reason=epoch

2026-05-15 20:19:12.300 05/15/26 20:19:12 - 1:07:51 - Saved checkpoint to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/ckpt/global_step3624.layer=2.head=4.hidden=64.best_model/model.pt

2026-05-15 20:19:12.334 05/15/26 20:19:12 - 1:07:51 - Re-initialized 101 high-cardinality Embeddings (vocab>0), kept 1

2026-05-15 20:19:12.337 05/15/26 20:19:12 - 1:07:51 - Rebuilt Adagrad optimizer after epoch 1, restored optimizer state for 2 low-cardinality params

2026-05-15 20:19:12.337 05/15/26 20:19:12 - 1:07:51 - EPOCH_START epoch=2

2026-05-15 20:23:06.705 05/15/26 20:23:06 - 1:11:45 - TRAIN_PROGRESS epoch=2 step=200/3733 total_step=3824 avg_loss=0.032782 last_loss=0.022875 dense_lr=9.9999873e-05

2026-05-15 20:26:40.845 05/15/26 20:26:40 - 1:15:19 - TRAIN_PROGRESS epoch=2 step=400/3733 total_step=4024 avg_loss=0.031649 last_loss=0.034667 dense_lr=9.9999854e-05

2026-05-15 20:30:12.189 05/15/26 20:30:12 - 1:18:50 - TRAIN_PROGRESS epoch=2 step=600/3733 total_step=4224 avg_loss=0.031204 last_loss=0.022916 dense_lr=9.9999834e-05

2026-05-15 20:34:01.355 05/15/26 20:34:01 - 1:22:40 - TRAIN_PROGRESS epoch=2 step=800/3733 total_step=4424 avg_loss=0.031085 last_loss=0.025904 dense_lr=9.9999813e-05

2026-05-15 20:37:51.119 05/15/26 20:37:51 - 1:26:29 - TRAIN_PROGRESS epoch=2 step=1000/3733 total_step=4624 avg_loss=0.030905 last_loss=0.029115 dense_lr=9.999979e-05

2026-05-15 20:41:42.286 05/15/26 20:41:42 - 1:30:21 - TRAIN_PROGRESS epoch=2 step=1200/3733 total_step=4824 avg_loss=0.030773 last_loss=0.025094 dense_lr=9.9999766e-05

2026-05-15 20:45:30.166 05/15/26 20:45:30 - 1:34:08 - TRAIN_PROGRESS epoch=2 step=1400/3733 total_step=5024 avg_loss=0.030720 last_loss=0.019261 dense_lr=9.9999741e-05

2026-05-15 20:49:17.478 05/15/26 20:49:17 - 1:37:56 - TRAIN_PROGRESS epoch=2 step=1600/3733 total_step=5224 avg_loss=0.030551 last_loss=0.025183 dense_lr=9.9999715e-05

2026-05-15 20:53:03.468 05/15/26 20:53:03 - 1:41:42 - TRAIN_PROGRESS epoch=2 step=1800/3733 total_step=5424 avg_loss=0.030486 last_loss=0.038214 dense_lr=9.9999687e-05

2026-05-15 20:56:45.659 05/15/26 20:56:45 - 1:45:24 - TRAIN_PROGRESS epoch=2 step=2000/3733 total_step=5624 avg_loss=0.030450 last_loss=0.035326 dense_lr=9.9999658e-05

2026-05-15 21:00:29.300 05/15/26 21:00:29 - 1:49:08 - TRAIN_PROGRESS epoch=2 step=2200/3733 total_step=5824 avg_loss=0.030387 last_loss=0.032361 dense_lr=9.9999628e-05

2026-05-15 21:04:15.234 05/15/26 21:04:15 - 1:52:53 - TRAIN_PROGRESS epoch=2 step=2400/3733 total_step=6024 avg_loss=0.030355 last_loss=0.024093 dense_lr=9.9999597e-05

2026-05-15 21:08:03.125 05/15/26 21:08:03 - 1:56:41 - TRAIN_PROGRESS epoch=2 step=2600/3733 total_step=6224 avg_loss=0.030314 last_loss=0.031318 dense_lr=9.9999564e-05

2026-05-15 21:11:49.087 05/15/26 21:11:49 - 2:00:27 - TRAIN_PROGRESS epoch=2 step=2800/3733 total_step=6424 avg_loss=0.030238 last_loss=0.036320 dense_lr=9.999953e-05

2026-05-15 21:15:28.920 05/15/26 21:15:28 - 2:04:07 - TRAIN_PROGRESS epoch=2 step=3000/3733 total_step=6624 avg_loss=0.030197 last_loss=0.022178 dense_lr=9.9999495e-05

2026-05-15 21:19:15.917 05/15/26 21:19:15 - 2:07:54 - TRAIN_PROGRESS epoch=2 step=3200/3733 total_step=6824 avg_loss=0.030169 last_loss=0.025955 dense_lr=9.9999458e-05

2026-05-15 21:22:55.004 05/15/26 21:22:55 - 2:11:33 - TRAIN_PROGRESS epoch=2 step=3400/3733 total_step=7024 avg_loss=0.030152 last_loss=0.028191 dense_lr=9.999942e-05

2026-05-15 21:26:40.837 05/15/26 21:26:40 - 2:15:19 - TRAIN_PROGRESS epoch=2 step=3600/3733 total_step=7224 avg_loss=0.030162 last_loss=0.027308 dense_lr=9.9999381e-05

2026-05-15 21:27:08.418 05/15/26 21:27:08 - 2:15:47 - EPOCH_TRAIN_RESULT epoch=2 total_step=7248 avg_loss=0.029276

2026-05-15 21:27:08.419 05/15/26 21:27:08 - 2:15:47 - VALIDATION_START epoch=2 total_step=7248 reason=epoch

2026-05-15 21:27:08.419 05/15/26 21:27:08 - 2:15:47 - VALIDATION_LOOP_START epoch=2 valid_steps=461

2026-05-15 21:33:26.699 05/15/26 21:33:26 - 2:22:05 - Validation extra metrics | GAUC: 0.0, GAUC_groups: 0

2026-05-15 21:33:26.797 05/15/26 21:33:26 - 2:22:05 - Validation hist_bucket_0 | rows=25723, AUC=0.8904901357722605

2026-05-15 21:33:26.803 05/15/26 21:33:26 - 2:22:05 - Validation hist_bucket_1 | rows=25662, AUC=0.8618971207497809

2026-05-15 21:33:26.813 05/15/26 21:33:26 - 2:22:05 - Validation hist_bucket_2 | rows=51234, AUC=0.8484282714473701

2026-05-15 21:33:26.849 05/15/26 21:33:26 - 2:22:05 - VALIDATION_RESULT epoch=2 total_step=7248 auc=0.863115346167 logloss=0.258354216814 reason=epoch

2026-05-15 21:33:27.160 05/15/26 21:33:27 - 2:22:05 - Removed old best_model dir: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/ckpt/global_step3624.layer=2.head=4.hidden=64.best_model

2026-05-15 21:33:27.160 05/15/26 21:33:27 - 2:22:05 - model earlyStopping counter reset!

2026-05-15 21:33:31.758 05/15/26 21:33:31 - 2:22:10 - Saved checkpoint to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/ckpt/global_step7248.layer=2.head=4.hidden=64.best_model/model.pt

2026-05-15 21:33:31.767 05/15/26 21:33:31 - 2:22:10 - Re-initialized 101 high-cardinality Embeddings (vocab>0), kept 1

2026-05-15 21:33:31.771 05/15/26 21:33:31 - 2:22:10 - Rebuilt Adagrad optimizer after epoch 2, restored optimizer state for 2 low-cardinality params

2026-05-15 21:33:31.771 05/15/26 21:33:31 - 2:22:10 - EPOCH_START epoch=3

2026-05-15 21:37:26.262 05/15/26 21:37:26 - 2:26:05 - TRAIN_PROGRESS epoch=3 step=200/3733 total_step=7448 avg_loss=0.031614 last_loss=0.028192 dense_lr=9.9999336e-05

2026-05-15 21:41:11.286 05/15/26 21:41:11 - 2:29:50 - TRAIN_PROGRESS epoch=3 step=400/3733 total_step=7648 avg_loss=0.030786 last_loss=0.026825 dense_lr=9.9999294e-05

2026-05-15 21:45:01.677 05/15/26 21:45:01 - 2:33:40 - TRAIN_PROGRESS epoch=3 step=600/3733 total_step=7848 avg_loss=0.030553 last_loss=0.035856 dense_lr=9.9999251e-05

2026-05-15 21:48:51.235 05/15/26 21:48:51 - 2:37:29 - TRAIN_PROGRESS epoch=3 step=800/3733 total_step=8048 avg_loss=0.030425 last_loss=0.025460 dense_lr=9.9999206e-05

2026-05-15 21:52:41.755 05/15/26 21:52:41 - 2:41:20 - TRAIN_PROGRESS epoch=3 step=1000/3733 total_step=8248 avg_loss=0.030431 last_loss=0.023844 dense_lr=9.9999161e-05

2026-05-15 21:56:32.667 05/15/26 21:56:32 - 2:45:11 - TRAIN_PROGRESS epoch=3 step=1200/3733 total_step=8448 avg_loss=0.030292 last_loss=0.028904 dense_lr=9.9999114e-05

2026-05-15 22:00:19.020 05/15/26 22:00:19 - 2:48:57 - TRAIN_PROGRESS epoch=3 step=1400/3733 total_step=8648 avg_loss=0.030116 last_loss=0.028423 dense_lr=9.9999066e-05

2026-05-15 22:03:55.427 05/15/26 22:03:55 - 2:52:34 - TRAIN_PROGRESS epoch=3 step=1600/3733 total_step=8848 avg_loss=0.030034 last_loss=0.039636 dense_lr=9.9999016e-05

2026-05-15 22:07:39.419 05/15/26 22:07:39 - 2:56:18 - TRAIN_PROGRESS epoch=3 step=1800/3733 total_step=9048 avg_loss=0.029910 last_loss=0.024735 dense_lr=9.9998965e-05

2026-05-15 22:11:20.628 05/15/26 22:11:20 - 2:59:59 - TRAIN_PROGRESS epoch=3 step=2000/3733 total_step=9248 avg_loss=0.029896 last_loss=0.027457 dense_lr=9.9998913e-05

2026-05-15 22:14:49.695 05/15/26 22:14:49 - 3:03:28 - TRAIN_PROGRESS epoch=3 step=2200/3733 total_step=9448 avg_loss=0.029866 last_loss=0.031953 dense_lr=9.999886e-05

2026-05-15 22:18:19.712 05/15/26 22:18:19 - 3:06:58 - TRAIN_PROGRESS epoch=3 step=2400/3733 total_step=9648 avg_loss=0.029840 last_loss=0.025179 dense_lr=9.9998805e-05

2026-05-15 22:22:05.810 05/15/26 22:22:05 - 3:10:44 - TRAIN_PROGRESS epoch=3 step=2600/3733 total_step=9848 avg_loss=0.029817 last_loss=0.030213 dense_lr=9.9998749e-05

2026-05-15 22:25:49.346 05/15/26 22:25:49 - 3:14:28 - TRAIN_PROGRESS epoch=3 step=2800/3733 total_step=10048 avg_loss=0.029787 last_loss=0.026688 dense_lr=9.9998692e-05

2026-05-15 22:29:36.374 05/15/26 22:29:36 - 3:18:15 - TRAIN_PROGRESS epoch=3 step=3000/3733 total_step=10248 avg_loss=0.029750 last_loss=0.035524 dense_lr=9.9998634e-05

2026-05-15 22:33:24.769 05/15/26 22:33:24 - 3:22:03 - TRAIN_PROGRESS epoch=3 step=3200/3733 total_step=10448 avg_loss=0.029743 last_loss=0.028879 dense_lr=9.9998574e-05

2026-05-15 22:37:12.982 05/15/26 22:37:12 - 3:25:51 - TRAIN_PROGRESS epoch=3 step=3400/3733 total_step=10648 avg_loss=0.029729 last_loss=0.032467 dense_lr=9.9998513e-05

2026-05-15 22:40:57.163 05/15/26 22:40:57 - 3:29:35 - TRAIN_PROGRESS epoch=3 step=3600/3733 total_step=10848 avg_loss=0.029715 last_loss=0.031076 dense_lr=9.9998451e-05

2026-05-15 22:41:24.507 05/15/26 22:41:24 - 3:30:03 - EPOCH_TRAIN_RESULT epoch=3 total_step=10872 avg_loss=0.028846

2026-05-15 22:41:24.507 05/15/26 22:41:24 - 3:30:03 - VALIDATION_START epoch=3 total_step=10872 reason=epoch

2026-05-15 22:41:24.507 05/15/26 22:41:24 - 3:30:03 - VALIDATION_LOOP_START epoch=3 valid_steps=461

2026-05-15 22:47:12.691 05/15/26 22:47:12 - 3:35:51 - Validation extra metrics | GAUC: 0.0, GAUC_groups: 0

2026-05-15 22:47:12.701 05/15/26 22:47:12 - 3:35:51 - Validation hist_bucket_0 | rows=25723, AUC=0.8926754148659811

2026-05-15 22:47:12.707 05/15/26 22:47:12 - 3:35:51 - Validation hist_bucket_1 | rows=25662, AUC=0.8628094994220764

2026-05-15 22:47:12.717 05/15/26 22:47:12 - 3:35:51 - Validation hist_bucket_2 | rows=51234, AUC=0.8491367980439305

2026-05-15 22:47:12.735 05/15/26 22:47:12 - 3:35:51 - VALIDATION_RESULT epoch=3 total_step=10872 auc=0.864213397356 logloss=0.255907177925 reason=epoch

2026-05-15 22:47:13.029 05/15/26 22:47:13 - 3:35:51 - Removed old best_model dir: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/ckpt/global_step7248.layer=2.head=4.hidden=64.best_model

2026-05-15 22:47:13.029 05/15/26 22:47:13 - 3:35:51 - model earlyStopping counter reset!

2026-05-15 22:47:17.281 05/15/26 22:47:17 - 3:35:56 - Saved checkpoint to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/ckpt/global_step10872.layer=2.head=4.hidden=64.best_model/model.pt

2026-05-15 22:47:17.292 05/15/26 22:47:17 - 3:35:56 - Re-initialized 101 high-cardinality Embeddings (vocab>0), kept 1

2026-05-15 22:47:17.295 05/15/26 22:47:17 - 3:35:56 - Rebuilt Adagrad optimizer after epoch 3, restored optimizer state for 2 low-cardinality params

2026-05-15 22:47:17.295 05/15/26 22:47:17 - 3:35:56 - EPOCH_START epoch=4

2026-05-15 22:51:05.202 05/15/26 22:51:05 - 3:39:43 - TRAIN_PROGRESS epoch=4 step=200/3733 total_step=11072 avg_loss=0.031671 last_loss=0.027171 dense_lr=9.9998379e-05

2026-05-15 22:54:47.874 05/15/26 22:54:47 - 3:43:26 - TRAIN_PROGRESS epoch=4 step=400/3733 total_step=11272 avg_loss=0.030811 last_loss=0.027920 dense_lr=9.9998314e-05

2026-05-15 22:58:36.055 05/15/26 22:58:36 - 3:47:14 - TRAIN_PROGRESS epoch=4 step=600/3733 total_step=11472 avg_loss=0.030425 last_loss=0.036001 dense_lr=9.9998248e-05

2026-05-15 23:02:23.547 05/15/26 23:02:23 - 3:51:02 - TRAIN_PROGRESS epoch=4 step=800/3733 total_step=11672 avg_loss=0.030293 last_loss=0.031310 dense_lr=9.999818e-05

2026-05-15 23:06:11.396 05/15/26 23:06:11 - 3:54:50 - TRAIN_PROGRESS epoch=4 step=1000/3733 total_step=11872 avg_loss=0.030289 last_loss=0.036405 dense_lr=9.9998112e-05

2026-05-15 23:09:59.506 05/15/26 23:09:59 - 3:58:38 - TRAIN_PROGRESS epoch=4 step=1200/3733 total_step=12072 avg_loss=0.030128 last_loss=0.029078 dense_lr=9.9998042e-05

2026-05-15 23:13:42.697 05/15/26 23:13:42 - 4:02:21 - TRAIN_PROGRESS epoch=4 step=1400/3733 total_step=12272 avg_loss=0.030060 last_loss=0.029592 dense_lr=9.999797e-05

2026-05-15 23:17:27.001 05/15/26 23:17:27 - 4:06:05 - TRAIN_PROGRESS epoch=4 step=1600/3733 total_step=12472 avg_loss=0.029890 last_loss=0.027311 dense_lr=9.9997897e-05

2026-05-15 23:21:10.400 05/15/26 23:21:10 - 4:09:49 - TRAIN_PROGRESS epoch=4 step=1800/3733 total_step=12672 avg_loss=0.029770 last_loss=0.030351 dense_lr=9.9997824e-05

2026-05-15 23:24:50.409 05/15/26 23:24:50 - 4:13:29 - TRAIN_PROGRESS epoch=4 step=2000/3733 total_step=12872 avg_loss=0.029770 last_loss=0.031698 dense_lr=9.9997748e-05

2026-05-15 23:28:31.326 05/15/26 23:28:31 - 4:17:10 - TRAIN_PROGRESS epoch=4 step=2200/3733 total_step=13072 avg_loss=0.029745 last_loss=0.030478 dense_lr=9.9997672e-05

2026-05-15 23:32:16.534 05/15/26 23:32:16 - 4:20:55 - TRAIN_PROGRESS epoch=4 step=2400/3733 total_step=13272 avg_loss=0.029711 last_loss=0.032277 dense_lr=9.9997594e-05

2026-05-15 23:35:50.600 05/15/26 23:35:50 - 4:24:29 - TRAIN_PROGRESS epoch=4 step=2600/3733 total_step=13472 avg_loss=0.029705 last_loss=0.036491 dense_lr=9.9997515e-05

2026-05-15 23:39:34.051 05/15/26 23:39:34 - 4:28:12 - TRAIN_PROGRESS epoch=4 step=2800/3733 total_step=13672 avg_loss=0.029652 last_loss=0.027412 dense_lr=9.9997435e-05

2026-05-15 23:43:20.704 05/15/26 23:43:20 - 4:31:59 - TRAIN_PROGRESS epoch=4 step=3000/3733 total_step=13872 avg_loss=0.029613 last_loss=0.024780 dense_lr=9.9997353e-05

2026-05-15 23:47:09.041 05/15/26 23:47:09 - 4:35:47 - TRAIN_PROGRESS epoch=4 step=3200/3733 total_step=14072 avg_loss=0.029584 last_loss=0.032364 dense_lr=9.999727e-05

2026-05-15 23:50:35.067 05/15/26 23:50:35 - 4:39:13 - TRAIN_PROGRESS epoch=4 step=3400/3733 total_step=14272 avg_loss=0.029575 last_loss=0.029543 dense_lr=9.9997186e-05

2026-05-15 23:54:13.832 05/15/26 23:54:13 - 4:42:52 - TRAIN_PROGRESS epoch=4 step=3600/3733 total_step=14472 avg_loss=0.029575 last_loss=0.032985 dense_lr=9.99971e-05

2026-05-15 23:54:40.844 05/15/26 23:54:40 - 4:43:19 - EPOCH_TRAIN_RESULT epoch=4 total_step=14496 avg_loss=0.028709

2026-05-15 23:54:40.845 05/15/26 23:54:40 - 4:43:19 - VALIDATION_START epoch=4 total_step=14496 reason=epoch

2026-05-15 23:54:40.845 05/15/26 23:54:40 - 4:43:19 - VALIDATION_LOOP_START epoch=4 valid_steps=461

2026-05-16 00:01:13.125 05/16/26 00:01:13 - 4:49:51 - Validation extra metrics | GAUC: 0.0, GAUC_groups: 0

2026-05-16 00:01:13.135 05/16/26 00:01:13 - 4:49:51 - Validation hist_bucket_0 | rows=25723, AUC=0.8928482487088003

2026-05-16 00:01:13.141 05/16/26 00:01:13 - 4:49:51 - Validation hist_bucket_1 | rows=25662, AUC=0.8632875620548042

2026-05-16 00:01:13.189 05/16/26 00:01:13 - 4:49:51 - Validation hist_bucket_2 | rows=51234, AUC=0.8502681642340442

2026-05-16 00:01:13.208 05/16/26 00:01:13 - 4:49:51 - VALIDATION_RESULT epoch=4 total_step=14496 auc=0.864952111420 logloss=0.254468947649 reason=epoch

2026-05-16 00:01:13.506 05/16/26 00:01:13 - 4:49:52 - Removed old best_model dir: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/ckpt/global_step10872.layer=2.head=4.hidden=64.best_model

2026-05-16 00:01:13.506 05/16/26 00:01:13 - 4:49:52 - model earlyStopping counter reset!

2026-05-16 00:01:18.088 05/16/26 00:01:18 - 4:49:56 - Saved checkpoint to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/ckpt/global_step14496.layer=2.head=4.hidden=64.best_model/model.pt

2026-05-16 00:01:18.099 05/16/26 00:01:18 - 4:49:56 - Re-initialized 101 high-cardinality Embeddings (vocab>0), kept 1

2026-05-16 00:01:18.102 05/16/26 00:01:18 - 4:49:56 - Rebuilt Adagrad optimizer after epoch 4, restored optimizer state for 2 low-cardinality params

2026-05-16 00:01:18.102 05/16/26 00:01:18 - 4:49:56 - EPOCH_START epoch=5

2026-05-16 00:05:04.887 05/16/26 00:05:04 - 4:53:43 - TRAIN_PROGRESS epoch=5 step=200/3733 total_step=14696 avg_loss=0.031675 last_loss=0.027470 dense_lr=9.9997003e-05

2026-05-16 00:08:42.072 05/16/26 00:08:42 - 4:57:20 - TRAIN_PROGRESS epoch=5 step=400/3733 total_step=14896 avg_loss=0.030630 last_loss=0.035052 dense_lr=9.9996915e-05

2026-05-16 00:12:22.178 05/16/26 00:12:22 - 5:01:00 - TRAIN_PROGRESS epoch=5 step=600/3733 total_step=15096 avg_loss=0.030365 last_loss=0.031280 dense_lr=9.9996826e-05

2026-05-16 00:16:03.616 05/16/26 00:16:03 - 5:04:42 - TRAIN_PROGRESS epoch=5 step=800/3733 total_step=15296 avg_loss=0.030216 last_loss=0.023709 dense_lr=9.9996735e-05

2026-05-16 00:19:45.470 05/16/26 00:19:45 - 5:08:24 - TRAIN_PROGRESS epoch=5 step=1000/3733 total_step=15496 avg_loss=0.030186 last_loss=0.026050 dense_lr=9.9996643e-05

2026-05-16 00:23:28.362 05/16/26 00:23:28 - 5:12:07 - TRAIN_PROGRESS epoch=5 step=1200/3733 total_step=15696 avg_loss=0.029970 last_loss=0.025436 dense_lr=9.999655e-05

2026-05-16 00:27:08.756 05/16/26 00:27:08 - 5:15:47 - TRAIN_PROGRESS epoch=5 step=1400/3733 total_step=15896 avg_loss=0.029832 last_loss=0.025159 dense_lr=9.9996455e-05

2026-05-16 00:30:48.093 05/16/26 00:30:48 - 5:19:26 - TRAIN_PROGRESS epoch=5 step=1600/3733 total_step=16096 avg_loss=0.029690 last_loss=0.028484 dense_lr=9.9996359e-05

2026-05-16 00:34:26.331 05/16/26 00:34:26 - 5:23:05 - TRAIN_PROGRESS epoch=5 step=1800/3733 total_step=16296 avg_loss=0.029638 last_loss=0.018780 dense_lr=9.9996262e-05

2026-05-16 00:38:00.805 05/16/26 00:38:00 - 5:26:39 - TRAIN_PROGRESS epoch=5 step=2000/3733 total_step=16496 avg_loss=0.029643 last_loss=0.046396 dense_lr=9.9996164e-05

2026-05-16 00:41:37.141 05/16/26 00:41:37 - 5:30:15 - TRAIN_PROGRESS epoch=5 step=2200/3733 total_step=16696 avg_loss=0.029591 last_loss=0.031311 dense_lr=9.9996064e-05

2026-05-16 00:45:06.601 05/16/26 00:45:06 - 5:33:45 - TRAIN_PROGRESS epoch=5 step=2400/3733 total_step=16896 avg_loss=0.029533 last_loss=0.029055 dense_lr=9.9995963e-05

2026-05-16 00:48:29.311 05/16/26 00:48:29 - 5:37:08 - TRAIN_PROGRESS epoch=5 step=2600/3733 total_step=17096 avg_loss=0.029546 last_loss=0.028504 dense_lr=9.9995861e-05

2026-05-16 00:51:59.003 05/16/26 00:51:59 - 5:40:37 - TRAIN_PROGRESS epoch=5 step=2800/3733 total_step=17296 avg_loss=0.029478 last_loss=0.030985 dense_lr=9.9995757e-05

2026-05-16 00:55:40.122 05/16/26 00:55:40 - 5:44:18 - TRAIN_PROGRESS epoch=5 step=3000/3733 total_step=17496 avg_loss=0.029450 last_loss=0.026743 dense_lr=9.9995653e-05

2026-05-16 00:59:23.475 05/16/26 00:59:23 - 5:48:02 - TRAIN_PROGRESS epoch=5 step=3200/3733 total_step=17696 avg_loss=0.029456 last_loss=0.039923 dense_lr=9.9995547e-05

2026-05-16 01:03:06.380 05/16/26 01:03:06 - 5:51:45 - TRAIN_PROGRESS epoch=5 step=3400/3733 total_step=17896 avg_loss=0.029416 last_loss=0.020452 dense_lr=9.9995439e-05

2026-05-16 01:06:27.085 05/16/26 01:06:27 - 5:55:05 - TRAIN_PROGRESS epoch=5 step=3600/3733 total_step=18096 avg_loss=0.029406 last_loss=0.040448 dense_lr=9.9995331e-05

2026-05-16 01:06:54.199 05/16/26 01:06:54 - 5:55:32 - EPOCH_TRAIN_RESULT epoch=5 total_step=18120 avg_loss=0.028541

2026-05-16 01:06:54.199 05/16/26 01:06:54 - 5:55:32 - VALIDATION_START epoch=5 total_step=18120 reason=epoch

2026-05-16 01:06:54.200 05/16/26 01:06:54 - 5:55:32 - VALIDATION_LOOP_START epoch=5 valid_steps=461

2026-05-16 01:12:48.514 05/16/26 01:12:48 - 6:01:27 - Validation extra metrics | GAUC: 0.0, GAUC_groups: 0

2026-05-16 01:12:48.525 05/16/26 01:12:48 - 6:01:27 - Validation hist_bucket_0 | rows=25723, AUC=0.8931715343055497

2026-05-16 01:12:48.530 05/16/26 01:12:48 - 6:01:27 - Validation hist_bucket_1 | rows=25662, AUC=0.8633229833202488

2026-05-16 01:12:48.540 05/16/26 01:12:48 - 6:01:27 - Validation hist_bucket_2 | rows=51234, AUC=0.8508009149151944

2026-05-16 01:12:48.558 05/16/26 01:12:48 - 6:01:27 - VALIDATION_RESULT epoch=5 total_step=18120 auc=0.865318890151 logloss=0.253355085850 reason=epoch

2026-05-16 01:12:48.830 05/16/26 01:12:48 - 6:01:27 - Removed old best_model dir: /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/ckpt/global_step14496.layer=2.head=4.hidden=64.best_model

2026-05-16 01:12:48.830 05/16/26 01:12:48 - 6:01:27 - model earlyStopping counter reset!

2026-05-16 01:12:53.328 05/16/26 01:12:53 - 6:01:32 - Saved checkpoint to /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260515191039_7892ef1e/95a8ce1c9e20cbf6019e2b54e93b193f/ckpt/global_step18120.layer=2.head=4.hidden=64.best_model/model.pt

2026-05-16 01:12:53.338 05/16/26 01:12:53 - 6:01:32 - Re-initialized 101 high-cardinality Embeddings (vocab>0), kept 1

2026-05-16 01:12:53.341 05/16/26 01:12:53 - 6:01:32 - Rebuilt Adagrad optimizer after epoch 5, restored optimizer state for 2 low-cardinality params

2026-05-16 01:12:53.341 05/16/26 01:12:53 - 6:01:32 - EPOCH_START epoch=6

2026-05-16 01:16:43.996 05/16/26 01:16:43 - 6:05:22 - TRAIN_PROGRESS epoch=6 step=200/3733 total_step=18320 avg_loss=0.031123 last_loss=0.029184 dense_lr=9.9995208e-05

2026-05-16 01:20:25.195 05/16/26 01:20:25 - 6:09:03 - TRAIN_PROGRESS epoch=6 step=400/3733 total_step=18520 avg_loss=0.030343 last_loss=0.031377 dense_lr=9.9995096e-05

2026-05-16 01:24:11.972 05/16/26 01:24:11 - 6:12:50 - TRAIN_PROGRESS epoch=6 step=600/3733 total_step=18720 avg_loss=0.029962 last_loss=0.020555 dense_lr=9.9994984e-05

2026-05-16 01:27:57.733 05/16/26 01:27:57 - 6:16:36 - TRAIN_PROGRESS epoch=6 step=800/3733 total_step=18920 avg_loss=0.029880 last_loss=0.027717 dense_lr=9.999487e-05

2026-05-16 01:31:43.781 05/16/26 01:31:43 - 6:20:22 - TRAIN_PROGRESS epoch=6 step=1000/3733 total_step=19120 avg_loss=0.029907 last_loss=0.031652 dense_lr=9.9994755e-05

2026-05-16 01:35:31.089 05/16/26 01:35:31 - 6:24:09 - TRAIN_PROGRESS epoch=6 step=1200/3733 total_step=19320 avg_loss=0.029706 last_loss=0.022245 dense_lr=9.9994638e-05

2026-05-16 01:39:12.726 05/16/26 01:39:12 - 6:27:51 - TRAIN_PROGRESS epoch=6 step=1400/3733 total_step=19520 avg_loss=0.029755 last_loss=0.029555 dense_lr=9.999452e-05

2026-05-16 01:42:55.786 05/16/26 01:42:55 - 6:31:34 - TRAIN_PROGRESS epoch=6 step=1600/3733 total_step=19720 avg_loss=0.029639 last_loss=0.022596 dense_lr=9.9994402e-05

2026-05-16 01:46:37.744 05/16/26 01:46:37 - 6:35:16 - TRAIN_PROGRESS epoch=6 step=1800/3733 total_step=19920 avg_loss=0.029529 last_loss=0.037462 dense_lr=9.9994281e-05

2026-05-16 01:50:16.240 05/16/26 01:50:16 - 6:38:54 - TRAIN_PROGRESS epoch=6 step=2000/3733 total_step=20120 avg_loss=0.029525 last_loss=0.029052 dense_lr=9.999416e-05

2026-05-16 01:53:56.409 05/16/26 01:53:56 - 6:42:35 - TRAIN_PROGRESS epoch=6 step=2200/3733 total_step=20320 avg_loss=0.029506 last_loss=0.033036 dense_lr=9.9994037e-05

2026-05-16 01:57:28.033 05/16/26 01:57:28 - 6:46:06 - TRAIN_PROGRESS epoch=6 step=2400/3733 total_step=20520 avg_loss=0.029430 last_loss=0.035026 dense_lr=9.9993913e-05

2026-05-16 02:01:13.108 05/16/26 02:01:13 - 6:49:51 - TRAIN_PROGRESS epoch=6 step=2600/3733 total_step=20720 avg_loss=0.029403 last_loss=0.024003 dense_lr=9.9993787e-05

2026-05-16 02:04:55.580 05/16/26 02:04:55 - 6:53:34 - TRAIN_PROGRESS epoch=6 step=2800/3733 total_step=20920 avg_loss=0.029340 last_loss=0.029191 dense_lr=9.9993661e-05

2026-05-16 02:08:40.912 05/16/26 02:08:40 - 6:57:19 - TRAIN_PROGRESS epoch=6 step=3000/3733 total_step=21120 avg_loss=0.029316 last_loss=0.031346 dense_lr=9.9993533e-05

2026-05-16 02:12:28.348 05/16/26 02:12:28 - 7:01:07 - TRAIN_PROGRESS epoch=6 step=3200/3733 total_step=21320 avg_loss=0.029294 last_loss=0.022622 dense_lr=9.9993404e-05

2026-05-16 02:16:15.556 05/16/26 02:16:15 - 7:04:54 - TRAIN_PROGRESS epoch=6 step=3400/3733 total_step=21520 avg_loss=0.029258 last_loss=0.030782 dense_lr=9.9993273e-05

2026-05-16 02:19:58.167 05/16/26 02:19:58 - 7:08:36 - TRAIN_PROGRESS epoch=6 step=3600/3733 total_step=21720 avg_loss=0.029256 last_loss=0.029601 dense_lr=9.9993141e-05

2026-05-16 02:20:25.683 05/16/26 02:20:25 - 7:09:04 - EPOCH_TRAIN_RESULT epoch=6 total_step=21744 avg_loss=0.028395

2026-05-16 02:20:25.683 05/16/26 02:20:25 - 7:09:04 - VALIDATION_START epoch=6 total_step=21744 reason=epoch

2026-05-16 02:20:25.683 05/16/26 02:20:25 - 7:09:04 - VALIDATION_LOOP_START epoch=6 valid_steps=461

2026-05-16 02:26:28.495 05/16/26 02:26:28 - 7:15:07 - Validation extra metrics | GAUC: 0.0, GAUC_groups: 0

2026-05-16 02:26:28.596 05/16/26 02:26:28 - 7:15:07 - Validation hist_bucket_0 | rows=25723, AUC=0.8933626016583954

2026-05-16 02:26:28.602 05/16/26 02:26:28 - 7:15:07 - Validation hist_bucket_1 | rows=25662, AUC=0.8633458822628721

2026-05-16 02:26:28.612 05/16/26 02:26:28 - 7:15:07 - Validation hist_bucket_2 | rows=51234, AUC=0.850678664656178

2026-05-16 02:26:28.630 05/16/26 02:26:28 - 7:15:07 - VALIDATION_RESULT epoch=6 total_step=21744 auc=0.865258154232 logloss=0.252384305000 reason=epoch

2026-05-16 02:26:28.667 05/16/26 02:26:28 - 7:15:07 - model earlyStopping counter: 1 / 5

2026-05-16 02:26:28.677 05/16/26 02:26:28 - 7:15:07 - Re-initialized 101 high-cardinality Embeddings (vocab>0), kept 1

2026-05-16 02:26:28.680 05/16/26 02:26:28 - 7:15:07 - Rebuilt Adagrad optimizer after epoch 6, restored optimizer state for 2 low-cardinality params

2026-05-16 02:26:28.681 05/16/26 02:26:28 - 7:15:07 - EPOCH_START epoch=7

2026-05-16 02:30:19.232 05/16/26 02:30:19 - 7:18:57 - TRAIN_PROGRESS epoch=7 step=200/3733 total_step=21944 avg_loss=0.031125 last_loss=0.033031 dense_lr=9.9992992e-05

2026-05-16 02:33:57.695 05/16/26 02:33:57 - 7:22:36 - TRAIN_PROGRESS epoch=7 step=400/3733 total_step=22144 avg_loss=0.030232 last_loss=0.027075 dense_lr=9.9992858e-05

2026-05-16 02:37:44.119 05/16/26 02:37:44 - 7:26:22 - TRAIN_PROGRESS epoch=7 step=600/3733 total_step=22344 avg_loss=0.029899 last_loss=0.024964 dense_lr=9.9992722e-05

2026-05-16 02:41:29.661 05/16/26 02:41:29 - 7:30:08 - TRAIN_PROGRESS epoch=7 step=800/3733 total_step=22544 avg_loss=0.029795 last_loss=0.028084 dense_lr=9.9992585e-05

2026-05-16 02:45:15.757 05/16/26 02:45:15 - 7:33:54 - TRAIN_PROGRESS epoch=7 step=1000/3733 total_step=22744 avg_loss=0.029704 last_loss=0.021939 dense_lr=9.9992447e-05

2026-05-16 02:48:52.739 05/16/26 02:48:52 - 7:37:31 - TRAIN_PROGRESS epoch=7 step=1200/3733 total_step=22944 avg_loss=0.029592 last_loss=0.028536 dense_lr=9.9992307e-05

2026-05-16 02:52:35.579 05/16/26 02:52:35 - 7:41:14 - TRAIN_PROGRESS epoch=7 step=1400/3733 total_step=23144 avg_loss=0.029493 last_loss=0.032143 dense_lr=9.9992166e-05

2026-05-16 02:56:18.916 05/16/26 02:56:18 - 7:44:57 - TRAIN_PROGRESS epoch=7 step=1600/3733 total_step=23344 avg_loss=0.029350 last_loss=0.025606 dense_lr=9.9992024e-05

2026-05-16 03:00:01.919 05/16/26 03:00:01 - 7:48:40 - TRAIN_PROGRESS epoch=7 step=1800/3733 total_step=23544 avg_loss=0.029275 last_loss=0.024569 dense_lr=9.9991881e-05

2026-05-16 03:03:40.306 05/16/26 03:03:40 - 7:52:19 - TRAIN_PROGRESS epoch=7 step=2000/3733 total_step=23744 avg_loss=0.029239 last_loss=0.032289 dense_lr=9.9991736e-05

2026-05-16 03:07:20.725 05/16/26 03:07:20 - 7:55:59 - TRAIN_PROGRESS epoch=7 step=2200/3733 total_step=23944 avg_loss=0.029189 last_loss=0.025089 dense_lr=9.999159e-05

2026-05-16 03:11:05.630 05/16/26 03:11:05 - 7:59:44 - TRAIN_PROGRESS epoch=7 step=2400/3733 total_step=24144 avg_loss=0.029174 last_loss=0.033766 dense_lr=9.9991443e-05

2026-05-16 03:14:50.397 05/16/26 03:14:50 - 8:03:29 - TRAIN_PROGRESS epoch=7 step=2600/3733 total_step=24344 avg_loss=0.029152 last_loss=0.028226 dense_lr=9.9991294e-05

2026-05-16 03:18:33.080 05/16/26 03:18:33 - 8:07:11 - TRAIN_PROGRESS epoch=7 step=2800/3733 total_step=24544 avg_loss=0.029092 last_loss=0.025653 dense_lr=9.9991144e-05

2026-05-16 03:22:01.861 05/16/26 03:22:01 - 8:10:40 - TRAIN_PROGRESS epoch=7 step=3000/3733 total_step=24744 avg_loss=0.029042 last_loss=0.025896 dense_lr=9.9990993e-05

2026-05-16 03:25:33.439 05/16/26 03:25:33 - 8:14:12 - TRAIN_PROGRESS epoch=7 step=3200/3733 total_step=24944 avg_loss=0.029040 last_loss=0.026290 dense_lr=9.9990841e-05

2026-05-16 03:29:17.129 05/16/26 03:29:17 - 8:17:55 - TRAIN_PROGRESS epoch=7 step=3400/3733 total_step=25144 avg_loss=0.028990 last_loss=0.026999 dense_lr=9.9990687e-05

2026-05-16 03:33:00.022 05/16/26 03:33:00 - 8:21:38 - TRAIN_PROGRESS epoch=7 step=3600/3733 total_step=25344 avg_loss=0.028993 last_loss=0.040488 dense_lr=9.9990532e-05

2026-05-16 03:33:27.079 05/16/26 03:33:27 - 8:22:05 - EPOCH_TRAIN_RESULT epoch=7 total_step=25368 avg_loss=0.028148

2026-05-16 03:33:27.080 05/16/26 03:33:27 - 8:22:05 - VALIDATION_START epoch=7 total_step=25368 reason=epoch

2026-05-16 03:33:27.080 05/16/26 03:33:27 - 8:22:05 - VALIDATION_LOOP_START epoch=7 valid_steps=461

2026-05-16 03:39:42.291 05/16/26 03:39:42 - 8:28:21 - Validation extra metrics | GAUC: 0.0, GAUC_groups: 0

2026-05-16 03:39:42.301 05/16/26 03:39:42 - 8:28:21 - Validation hist_bucket_0 | rows=25723, AUC=0.8924749170850862

2026-05-16 03:39:42.307 05/16/26 03:39:42 - 8:28:21 - Validation hist_bucket_1 | rows=25662, AUC=0.8634542574309771

2026-05-16 03:39:42.317 05/16/26 03:39:42 - 8:28:21 - Validation hist_bucket_2 | rows=51234, AUC=0.850898042932134

2026-05-16 03:39:42.339 05/16/26 03:39:42 - 8:28:21 - VALIDATION_RESULT epoch=7 total_step=25368 auc=0.865171335483 logloss=0.251405954361 reason=epoch

2026-05-16 03:39:42.375 05/16/26 03:39:42 - 8:28:21 - model earlyStopping counter: 2 / 5

2026-05-16 03:39:42.392 05/16/26 03:39:42 - 8:28:21 - Re-initialized 101 high-cardinality Embeddings (vocab>0), kept 1

2026-05-16 03:39:42.395 05/16/26 03:39:42 - 8:28:21 - Rebuilt Adagrad optimizer after epoch 7, restored optimizer state for 2 low-cardinality params

2026-05-16 03:39:42.395 05/16/26 03:39:42 - 8:28:21 - EPOCH_START epoch=8

2026-05-16 03:43:27.832 05/16/26 03:43:27 - 8:32:06 - TRAIN_PROGRESS epoch=8 step=200/3733 total_step=25568 avg_loss=0.031422 last_loss=0.028100 dense_lr=9.9990357e-05

2026-05-16 03:47:09.059 05/16/26 03:47:09 - 8:35:47 - TRAIN_PROGRESS epoch=8 step=400/3733 total_step=25768 avg_loss=0.030281 last_loss=0.023640 dense_lr=9.99902e-05

2026-05-16 03:50:55.889 05/16/26 03:50:55 - 8:39:34 - TRAIN_PROGRESS epoch=8 step=600/3733 total_step=25968 avg_loss=0.029780 last_loss=0.022661 dense_lr=9.9990041e-05

2026-05-16 03:54:41.573 05/16/26 03:54:41 - 8:43:20 - TRAIN_PROGRESS epoch=8 step=800/3733 total_step=26168 avg_loss=0.029665 last_loss=0.031263 dense_lr=9.9989881e-05

2026-05-16 03:58:27.681 05/16/26 03:58:27 - 8:47:06 - TRAIN_PROGRESS epoch=8 step=1000/3733 total_step=26368 avg_loss=0.029553 last_loss=0.025787 dense_lr=9.9989719e-05

2026-05-16 04:02:14.380 05/16/26 04:02:14 - 8:50:53 - TRAIN_PROGRESS epoch=8 step=1200/3733 total_step=26568 avg_loss=0.029420 last_loss=0.023407 dense_lr=9.9989557e-05

2026-05-16 04:05:58.719 05/16/26 04:05:58 - 8:54:37 - TRAIN_PROGRESS epoch=8 step=1400/3733 total_step=26768 avg_loss=0.029354 last_loss=0.028046 dense_lr=9.9989393e-05

2026-05-16 04:09:42.380 05/16/26 04:09:42 - 8:58:21 - TRAIN_PROGRESS epoch=8 step=1600/3733 total_step=26968 avg_loss=0.029179 last_loss=0.029441 dense_lr=9.9989227e-05

2026-05-16 04:13:24.751 05/16/26 04:13:24 - 9:02:03 - TRAIN_PROGRESS epoch=8 step=1800/3733 total_step=27168 avg_loss=0.029112 last_loss=0.028343 dense_lr=9.9989061e-05

2026-05-16 04:17:03.217 05/16/26 04:17:03 - 9:05:41 - TRAIN_PROGRESS epoch=8 step=2000/3733 total_step=27368 avg_loss=0.029071 last_loss=0.027584 dense_lr=9.9988893e-05

2026-05-16 04:20:43.045 05/16/26 04:20:43 - 9:09:21 - TRAIN_PROGRESS epoch=8 step=2200/3733 total_step=27568 avg_loss=0.029016 last_loss=0.030903 dense_lr=9.9988724e-05

2026-05-16 04:24:23.419 05/16/26 04:24:23 - 9:13:02 - TRAIN_PROGRESS epoch=8 step=2400/3733 total_step=27768 avg_loss=0.028986 last_loss=0.029170 dense_lr=9.9988553e-05

2026-05-16 04:28:08.462 05/16/26 04:28:08 - 9:16:47 - TRAIN_PROGRESS epoch=8 step=2600/3733 total_step=27968 avg_loss=0.028953 last_loss=0.027430 dense_lr=9.9988382e-05

2026-05-16 04:31:50.337 05/16/26 04:31:50 - 9:20:29 - TRAIN_PROGRESS epoch=8 step=2800/3733 total_step=28168 avg_loss=0.028882 last_loss=0.028561 dense_lr=9.9988209e-05

2026-05-16 04:35:23.553 05/16/26 04:35:23 - 9:24:02 - TRAIN_PROGRESS epoch=8 step=3000/3733 total_step=28368 avg_loss=0.028840 last_loss=0.024989 dense_lr=9.9988034e-05

2026-05-16 04:39:10.815 05/16/26 04:39:10 - 9:27:49 - TRAIN_PROGRESS epoch=8 step=3200/3733 total_step=28568 avg_loss=0.028795 last_loss=0.027326 dense_lr=9.9987859e-05

2026-05-16 04:42:58.009 05/16/26 04:42:58 - 9:31:36 - TRAIN_PROGRESS epoch=8 step=3400/3733 total_step=28768 avg_loss=0.028764 last_loss=0.023005 dense_lr=9.9987682e-05

2026-05-16 04:46:40.693 05/16/26 04:46:40 - 9:35:19 - TRAIN_PROGRESS epoch=8 step=3600/3733 total_step=28968 avg_loss=0.028779 last_loss=0.035392 dense_lr=9.9987504e-05

2026-05-16 04:47:07.882 05/16/26 04:47:07 - 9:35:46 - EPOCH_TRAIN_RESULT epoch=8 total_step=28992 avg_loss=0.027934

2026-05-16 04:47:07.882 05/16/26 04:47:07 - 9:35:46 - VALIDATION_START epoch=8 total_step=28992 reason=epoch

2026-05-16 04:47:07.883 05/16/26 04:47:07 - 9:35:46 - VALIDATION_LOOP_START epoch=8 valid_steps=461

2026-05-16 04:54:33.689 05/16/26 04:54:33 - 9:43:12 - Validation extra metrics | GAUC: 0.0, GAUC_groups: 0

2026-05-16 04:54:33.699 05/16/26 04:54:33 - 9:43:12 - Validation hist_bucket_0 | rows=25723, AUC=0.8922505902719428

2026-05-16 04:54:33.704 05/16/26 04:54:33 - 9:43:12 - Validation hist_bucket_1 | rows=25662, AUC=0.8625702192041621

2026-05-16 04:54:33.714 05/16/26 04:54:33 - 9:43:12 - Validation hist_bucket_2 | rows=51234, AUC=0.8500439398876154

2026-05-16 04:54:33.736 05/16/26 04:54:33 - 9:43:12 - VALIDATION_RESULT epoch=8 total_step=28992 auc=0.864486180241 logloss=0.250292748213 reason=epoch

2026-05-16 04:54:33.767 05/16/26 04:54:33 - 9:43:12 - model earlyStopping counter: 3 / 5

2026-05-16 04:54:33.777 05/16/26 04:54:33 - 9:43:12 - Re-initialized 101 high-cardinality Embeddings (vocab>0), kept 1

2026-05-16 04:54:33.780 05/16/26 04:54:33 - 9:43:12 - Rebuilt Adagrad optimizer after epoch 8, restored optimizer state for 2 low-cardinality params

2026-05-16 04:54:33.781 05/16/26 04:54:33 - 9:43:12 - EPOCH_START epoch=9

2026-05-16 04:58:23.980 05/16/26 04:58:23 - 9:47:02 - TRAIN_PROGRESS epoch=9 step=200/3733 total_step=29192 avg_loss=0.031110 last_loss=0.030958 dense_lr=9.9987303e-05

2026-05-16 05:02:04.985 05/16/26 05:02:04 - 9:50:43 - TRAIN_PROGRESS epoch=9 step=400/3733 total_step=29392 avg_loss=0.030003 last_loss=0.030427 dense_lr=9.9987122e-05

2026-05-16 05:05:51.781 05/16/26 05:05:51 - 9:54:30 - TRAIN_PROGRESS epoch=9 step=600/3733 total_step=29592 avg_loss=0.029652 last_loss=0.023827 dense_lr=9.998694e-05

2026-05-16 05:09:37.737 05/16/26 05:09:37 - 9:58:16 - TRAIN_PROGRESS epoch=9 step=800/3733 total_step=29792 avg_loss=0.029444 last_loss=0.031619 dense_lr=9.9986757e-05

2026-05-16 05:13:23.985 05/16/26 05:13:23 - 10:02:02 - TRAIN_PROGRESS epoch=9 step=1000/3733 total_step=29992 avg_loss=0.029333 last_loss=0.031480 dense_lr=9.9986572e-05

2026-05-16 05:17:11.431 05/16/26 05:17:11 - 10:05:50 - TRAIN_PROGRESS epoch=9 step=1200/3733 total_step=30192 avg_loss=0.029132 last_loss=0.028138 dense_lr=9.9986386e-05

2026-05-16 05:20:53.155 05/16/26 05:20:53 - 10:09:31 - TRAIN_PROGRESS epoch=9 step=1400/3733 total_step=30392 avg_loss=0.029049 last_loss=0.029400 dense_lr=9.9986199e-05

2026-05-16 05:24:34.046 05/16/26 05:24:34 - 10:13:12 - TRAIN_PROGRESS epoch=9 step=1600/3733 total_step=30592 avg_loss=0.028875 last_loss=0.026323 dense_lr=9.9986011e-05

2026-05-16 05:28:06.072 05/16/26 05:28:06 - 10:16:44 - TRAIN_PROGRESS epoch=9 step=1800/3733 total_step=30792 avg_loss=0.028783 last_loss=0.031176 dense_lr=9.9985821e-05

2026-05-16 05:31:44.325 05/16/26 05:31:44 - 10:20:23 - TRAIN_PROGRESS epoch=9 step=2000/3733 total_step=30992 avg_loss=0.028752 last_loss=0.027466 dense_lr=9.998563e-05

2026-05-16 05:35:24.149 05/16/26 05:35:24 - 10:24:02 - TRAIN_PROGRESS epoch=9 step=2200/3733 total_step=31192 avg_loss=0.028652 last_loss=0.025838 dense_lr=9.9985438e-05

2026-05-16 05:39:08.409 05/16/26 05:39:08 - 10:27:47 - TRAIN_PROGRESS epoch=9 step=2400/3733 total_step=31392 avg_loss=0.028602 last_loss=0.027671 dense_lr=9.9985244e-05

2026-05-16 05:42:53.067 05/16/26 05:42:53 - 10:31:31 - TRAIN_PROGRESS epoch=9 step=2600/3733 total_step=31592 avg_loss=0.028580 last_loss=0.026732 dense_lr=9.9985049e-05

2026-05-16 05:46:35.377 05/16/26 05:46:35 - 10:35:14 - TRAIN_PROGRESS epoch=9 step=2800/3733 total_step=31792 avg_loss=0.028502 last_loss=0.027468 dense_lr=9.9984853e-05

2026-05-16 05:50:20.882 05/16/26 05:50:20 - 10:38:59 - TRAIN_PROGRESS epoch=9 step=3000/3733 total_step=31992 avg_loss=0.028449 last_loss=0.029509 dense_lr=9.9984656e-05

2026-05-16 05:54:04.959 05/16/26 05:54:04 - 10:42:43 - TRAIN_PROGRESS epoch=9 step=3200/3733 total_step=32192 avg_loss=0.028399 last_loss=0.025042 dense_lr=9.9984457e-05

2026-05-16 05:57:33.135 05/16/26 05:57:33 - 10:46:11 - TRAIN_PROGRESS epoch=9 step=3400/3733 total_step=32392 avg_loss=0.028360 last_loss=0.026092 dense_lr=9.9984257e-05

2026-05-16 06:00:20.409 05/16/26 06:00:20 - 10:48:59 - TRAIN_PROGRESS epoch=9 step=3600/3733 total_step=32592 avg_loss=0.028357 last_loss=0.034302 dense_lr=9.9984056e-05

2026-05-16 06:00:40.846 05/16/26 06:00:40 - 10:49:19 - EPOCH_TRAIN_RESULT epoch=9 total_step=32616 avg_loss=0.027524

2026-05-16 06:00:40.846 05/16/26 06:00:40 - 10:49:19 - VALIDATION_START epoch=9 total_step=32616 reason=epoch

2026-05-16 06:00:40.846 05/16/26 06:00:40 - 10:49:19 - VALIDATION_LOOP_START epoch=9 valid_steps=461

2026-05-16 06:07:58.487 05/16/26 06:07:58 - 10:56:37 - Validation extra metrics | GAUC: 0.0, GAUC_groups: 0

2026-05-16 06:07:58.497 05/16/26 06:07:58 - 10:56:37 - Validation hist_bucket_0 | rows=25723, AUC=0.8899169741876639

2026-05-16 06:07:58.503 05/16/26 06:07:58 - 10:56:37 - Validation hist_bucket_1 | rows=25662, AUC=0.8603084422341987

2026-05-16 06:07:58.513 05/16/26 06:07:58 - 10:56:37 - Validation hist_bucket_2 | rows=51234, AUC=0.8477180551698428

2026-05-16 06:07:58.536 05/16/26 06:07:58 - 10:56:37 - VALIDATION_RESULT epoch=9 total_step=32616 auc=0.862214106231 logloss=0.248830020428 reason=epoch

2026-05-16 06:07:58.565 05/16/26 06:07:58 - 10:56:37 - model earlyStopping counter: 4 / 5

2026-05-16 06:07:58.574 05/16/26 06:07:58 - 10:56:37 - Re-initialized 101 high-cardinality Embeddings (vocab>0), kept 1

2026-05-16 06:07:58.577 05/16/26 06:07:58 - 10:56:37 - Rebuilt Adagrad optimizer after epoch 9, restored optimizer state for 2 low-cardinality params

2026-05-16 06:07:58.577 05/16/26 06:07:58 - 10:56:37 - EPOCH_START epoch=10

2026-05-16 06:10:38.727 05/16/26 06:10:38 - 10:59:17 - TRAIN_PROGRESS epoch=10 step=200/3733 total_step=32816 avg_loss=0.030992 last_loss=0.031040 dense_lr=9.9983829e-05

2026-05-16 06:13:22.896 05/16/26 06:13:22 - 11:02:01 - TRAIN_PROGRESS epoch=10 step=400/3733 total_step=33016 avg_loss=0.029895 last_loss=0.029235 dense_lr=9.9983625e-05

2026-05-16 06:16:33.023 05/16/26 06:16:33 - 11:05:11 - TRAIN_PROGRESS epoch=10 step=600/3733 total_step=33216 avg_loss=0.029484 last_loss=0.027720 dense_lr=9.998342e-05

2026-05-16 06:20:21.121 05/16/26 06:20:21 - 11:08:59 - TRAIN_PROGRESS epoch=10 step=800/3733 total_step=33416 avg_loss=0.029254 last_loss=0.035996 dense_lr=9.9983214e-05

2026-05-16 06:24:09.746 05/16/26 06:24:09 - 11:12:48 - TRAIN_PROGRESS epoch=10 step=1000/3733 total_step=33616 avg_loss=0.029094 last_loss=0.020830 dense_lr=9.9983006e-05

2026-05-16 06:27:58.784 05/16/26 06:27:58 - 11:16:37 - TRAIN_PROGRESS epoch=10 step=1200/3733 total_step=33816 avg_loss=0.028893 last_loss=0.024507 dense_lr=9.9982797e-05

2026-05-16 06:31:45.551 05/16/26 06:31:45 - 11:20:24 - TRAIN_PROGRESS epoch=10 step=1400/3733 total_step=34016 avg_loss=0.028763 last_loss=0.028789 dense_lr=9.9982586e-05

2026-05-16 06:35:31.722 05/16/26 06:35:31 - 11:24:10 - TRAIN_PROGRESS epoch=10 step=1600/3733 total_step=34216 avg_loss=0.028533 last_loss=0.019830 dense_lr=9.9982375e-05

2026-05-16 06:39:16.115 05/16/26 06:39:16 - 11:27:54 - TRAIN_PROGRESS epoch=10 step=1800/3733 total_step=34416 avg_loss=0.028407 last_loss=0.024064 dense_lr=9.9982162e-05

2026-05-16 06:42:56.247 05/16/26 06:42:56 - 11:31:34 - TRAIN_PROGRESS epoch=10 step=2000/3733 total_step=34616 avg_loss=0.028368 last_loss=0.031730 dense_lr=9.9981948e-05

2026-05-16 06:46:38.693 05/16/26 06:46:38 - 11:35:17 - TRAIN_PROGRESS epoch=10 step=2200/3733 total_step=34816 avg_loss=0.028324 last_loss=0.027973 dense_lr=9.9981732e-05

2026-05-16 06:50:24.995 05/16/26 06:50:24 - 11:39:03 - TRAIN_PROGRESS epoch=10 step=2400/3733 total_step=35016 avg_loss=0.028250 last_loss=0.037661 dense_lr=9.9981516e-05

2026-05-16 06:54:12.361 05/16/26 06:54:12 - 11:42:51 - TRAIN_PROGRESS epoch=10 step=2600/3733 total_step=35216 avg_loss=0.028206 last_loss=0.028345 dense_lr=9.9981298e-05

2026-05-16 06:57:56.699 05/16/26 06:57:56 - 11:46:35 - TRAIN_PROGRESS epoch=10 step=2800/3733 total_step=35416 avg_loss=0.028114 last_loss=0.027463 dense_lr=9.9981078e-05

2026-05-16 07:01:42.494 05/16/26 07:01:42 - 11:50:21 - TRAIN_PROGRESS epoch=10 step=3000/3733 total_step=35616 avg_loss=0.028050 last_loss=0.030431 dense_lr=9.9980858e-05

2026-05-16 07:05:19.067 05/16/26 07:05:19 - 11:53:57 - TRAIN_PROGRESS epoch=10 step=3200/3733 total_step=35816 avg_loss=0.027996 last_loss=0.030491 dense_lr=9.9980636e-05

2026-05-16 07:09:08.348 05/16/26 07:09:08 - 11:57:47 - TRAIN_PROGRESS epoch=10 step=3400/3733 total_step=36016 avg_loss=0.027915 last_loss=0.022709 dense_lr=9.9980413e-05

2026-05-16 07:12:53.278 05/16/26 07:12:53 - 12:01:32 - TRAIN_PROGRESS epoch=10 step=3600/3733 total_step=36216 avg_loss=0.027878 last_loss=0.030135 dense_lr=9.9980189e-05

2026-05-16 07:13:20.706 05/16/26 07:13:20 - 12:01:59 - EPOCH_TRAIN_RESULT epoch=10 total_step=36240 avg_loss=0.027059

2026-05-16 07:13:20.706 05/16/26 07:13:20 - 12:01:59 - VALIDATION_START epoch=10 total_step=36240 reason=epoch

2026-05-16 07:13:20.706 05/16/26 07:13:20 - 12:01:59 - VALIDATION_LOOP_START epoch=10 valid_steps=461

2026-05-16 07:20:58.561 05/16/26 07:20:58 - 12:09:37 - Validation extra metrics | GAUC: 0.0, GAUC_groups: 0

2026-05-16 07:20:58.572 05/16/26 07:20:58 - 12:09:37 - Validation hist_bucket_0 | rows=25723, AUC=0.8873115553584744

2026-05-16 07:20:58.593 05/16/26 07:20:58 - 12:09:37 - Validation hist_bucket_1 | rows=25662, AUC=0.857692263787923

2026-05-16 07:20:58.604 05/16/26 07:20:58 - 12:09:37 - Validation hist_bucket_2 | rows=51234, AUC=0.8445707645753188

2026-05-16 07:20:58.626 05/16/26 07:20:58 - 12:09:37 - VALIDATION_RESULT epoch=10 total_step=36240 auc=0.859350877714 logloss=0.247860550880 reason=epoch

2026-05-16 07:20:58.663 05/16/26 07:20:58 - 12:09:37 - model earlyStopping counter: 5 / 5

2026-05-16 07:20:58.670 05/16/26 07:20:58 - 12:09:37 - Early stopping at epoch 10

2026-05-16 07:20:58.671 05/16/26 07:20:58 - 12:09:37 - Training complete!

2026-05-16 07:21:01.841 INFO: ======= End cmd =======

2026-05-16 07:21:01.849 [TAIJI] user exit code: 0

Truncated to last 1000 lines.