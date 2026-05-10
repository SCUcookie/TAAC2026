2026-05-09 11:14:57.851 ##########################################################################

2026-05-09 11:14:57.851 ############################ TAIJI JOB START #############################

2026-05-09 11:14:57.851 ##########################################################################

2026-05-09 11:14:57.865 cp: cannot stat './/*': No such file or directory

2026-05-09 11:14:57.874 passwd: password expiry information changed.

2026-05-09 11:15:03.334 Complete setting taiji user.

2026-05-09 11:15:08.908 update taiji environ ...

2026-05-09 11:15:08.908 show args: --appName=95d1b0069dfa2b61019e0abb06792c6d --projectName=external-ams-competition-2025 --module=pytorch --script=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/run.sh --script_archive=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/dataset.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/model.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/ns_groups.json,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/train.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/trainer.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/utils.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/.taiji_run.json --script_args=

2026-05-09 11:15:08.996 INFO: ===begin to run init cmd===

2026-05-09 11:15:08.996 INFO: Export env: RUNTIME_SCRIPT_DIR=/home/taiji/dl/runtime/script

2026-05-09 11:15:09.009 INFO: Export env: RUNTIME_SCRIPT_SHARE_DIR=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/95d1b0069dfa2b61019e0abb06792c6d/pipeline/95d1b0069dfa2b61019e0abb06792c6d/share

2026-05-09 11:15:09.009 INFO: script_path :/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/run.sh

2026-05-09 11:15:09.017 INFO: Success copy to /home/taiji/dl/runtime/script/run.sh

2026-05-09 11:15:09.024 INFO: Success copy to /home/taiji/dl/runtime/script/dataset.py

2026-05-09 11:15:09.032 INFO: Success copy to /home/taiji/dl/runtime/script/model.py

2026-05-09 11:15:09.039 INFO: Success copy to /home/taiji/dl/runtime/script/ns_groups.json

2026-05-09 11:15:09.046 INFO: Success copy to /home/taiji/dl/runtime/script/train.py

2026-05-09 11:15:09.053 INFO: Success copy to /home/taiji/dl/runtime/script/trainer.py

2026-05-09 11:15:09.060 INFO: Success copy to /home/taiji/dl/runtime/script/utils.py

2026-05-09 11:15:09.067 INFO: Success copy to /home/taiji/dl/runtime/script/.taiji_run.json

2026-05-09 11:15:09.067 INFO: python path: /home/taiji/dl/runtime/script/dataset.py:/home/taiji/dl/runtime/script/model.py:/home/taiji/dl/runtime/script/ns_groups.json:/home/taiji/dl/runtime/script/train.py:/home/taiji/dl/runtime/script/trainer.py:/home/taiji/dl/runtime/script/utils.py:/home/taiji/dl/runtime/script/.taiji_run.json

2026-05-09 11:15:09.067 INFO: Export env: TRAIN_CKPT_PATH=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/95d1b0069dfa2b61019e0abb06792c6d/ckpt

2026-05-09 11:15:09.067 INFO: Export env: TRAIN_TF_EVENTS_PATH=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/95d1b0069dfa2b61019e0abb06792c6d/events

2026-05-09 11:15:09.067 INFO: Export env: TRAIN_DATA_PATH=/data_ams/academic_training_data

2026-05-09 11:15:09.067 INFO: Export env: TRAIN_USER=ams_2026_1029731852466347124

2026-05-09 11:15:09.067 INFO: Export env: TRAIN_LOG_PATH=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/95d1b0069dfa2b61019e0abb06792c6d/log

2026-05-09 11:15:09.067 INFO: Export env: TRAIN_CACHE_PATH=/apdcephfs_fsgm2/share_305170765/angel/groups/44/ams_2026_1029731852466347124

2026-05-09 11:15:09.067 INFO: Export env: USER_CACHE_PATH=/apdcephfs_fsgm2/share_305170765/angel/groups/44/ams_2026_1029731852466347124

2026-05-09 11:15:09.067 INFO: Export env: NP_BD_FLOW_PROJ=7971918

2026-05-09 11:15:09.076 touch: cannot touch '/data/initCmdCompletedFile': No such file or directory

2026-05-09 11:15:09.077 initCmd failed

2026-05-09 11:15:09.077 [TAIJI] Starting sshd

2026-05-09 11:15:10.097 [TAIJI] sshd started

2026-05-09 11:15:10.097 [TAIJI] waiting for all peers to be ready...

2026-05-09 11:15:11.348 Welcome to Ubuntu 22.04.5 LTS (GNU/Linux 5.4.241-1-tlinux4-0017.7 x86_64)

2026-05-09 11:15:11.348

2026-05-09 11:15:11.348 * Documentation: https://help.ubuntu.com

2026-05-09 11:15:11.348 * Management: https://landscape.canonical.com

2026-05-09 11:15:11.348 * Support: https://ubuntu.com/pro

2026-05-09 11:15:11.348

2026-05-09 11:15:11.348 This system has been minimized by removing packages and content that are

2026-05-09 11:15:11.348 not required on a system that users do not log into.

2026-05-09 11:15:11.348

2026-05-09 11:15:11.348 To restore this content, you can run the 'unminimize' command.

2026-05-09 11:15:11.348

2026-05-09 11:15:11.348 The programs included with the Ubuntu system are free software;

2026-05-09 11:15:11.348 the exact distribution terms for each program are described in the

2026-05-09 11:15:11.348 individual files in /usr/share/doc/*/copyright.

2026-05-09 11:15:11.348

2026-05-09 11:15:11.348 Ubuntu comes with ABSOLUTELY NO WARRANTY, to the extent permitted by

2026-05-09 11:15:11.348 applicable law.

2026-05-09 11:15:11.348

2026-05-09 11:15:12.462 [TAIJI] error log above can be ignored

2026-05-09 11:15:12.466 [TAIJI] run command 1778296512 [/bin/bash -c "${start_script}"]

2026-05-09 11:15:15.967 Complete setting network policy rules.

2026-05-09 11:15:21.497 update taiji environ ...

2026-05-09 11:15:21.497 show args: --appName=95d1b0069dfa2b61019e0abb06792c6d --projectName=external-ams-competition-2025n --module=pytorch --script=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/run.sh --script_archive=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/dataset.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/model.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/ns_groups.json,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/train.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/trainer.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/utils.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/.taiji_run.json --script_args=

2026-05-09 11:15:21.540 INFO: ===begin to run starttorch cmd===

2026-05-09 11:15:21.541 INFO: runtime_script_share_dir is /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/95d1b0069dfa2b61019e0abb06792c6d/pipeline/95d1b0069dfa2b61019e0abb06792c6d/share

2026-05-09 11:15:21.541 INFO: script_path :/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/scripts/95d1b0069dfa2b61019e0abb06792c6d/run.sh

2026-05-09 11:15:21.541 INFO: framework is pytorch

2026-05-09 11:15:21.541 INFO: Run command: bash /home/taiji/dl/runtime/script/run.sh

2026-05-09 11:15:21.541 INFO: ====== Begin cmd ======

2026-05-09 11:15:22.671 PLATFORM_TRAIN_CONFIG {"ns_tokenizer_type":"rankmixer","user_ns_tokens":5,"item_ns_tokens":2,"num_queries":2,"ns_groups_json":"/home/taiji/dl/runtime/script/ns_groups.json","emb_skip_threshold":1000000,"num_workers":8,"seq_max_lens":"seq_a:256,seq_b:256,seq_c:512,seq_d:512","d_model":64,"num_hyformer_blocks":2,"num_heads":4,"seq_encoder_type":"transformer","requested_loss_type":"bce_pairwise","loss_type":"bce_pairwise","pairwise_supported":1,"batch_size":256,"pairwise_auc_weight":0.05,"pairwise_max_pairs":8192,"focal_alpha":0.1,"focal_gamma":2.0}

2026-05-09 11:15:22.960 [DEBUG][libvgpu]hijack_call.c:106 [p:274 t:274]init cuda hook lib

2026-05-09 11:15:22.960 [DEBUG][libvgpu]hijack_call.c:107 [p:274 t:274]Thread pid:274, tid:274

2026-05-09 11:15:22.960 [DEBUG][libvgpu]hijack_call.c:125 [p:274 t:274]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

2026-05-09 11:15:22.963 [DEBUG][libvgpu]hijack_call.c:172 [p:274 t:274]hooked env NCCL_SET_THREAD_NAME to : 1

2026-05-09 11:15:22.963 [DEBUG][libvgpu]hijack_call.c:173 [p:274 t:274]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

2026-05-09 11:15:22.963 [DEBUG][libvgpu]hijack_call.c:174 [p:274 t:274]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1

2026-05-09 11:15:22.963 [DEBUG][libvgpu]hijack_call.c:175 [p:274 t:274]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1

2026-05-09 11:15:27.015 05/09/26 11:15:27 - 0:00:00 - Args: {'data_dir': '/data_ams/academic_training_data', 'schema_path': None, 'ckpt_dir': '/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/95d1b0069dfa2b61019e0abb06792c6d/ckpt', 'log_dir': '/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/95d1b0069dfa2b61019e0abb06792c6d/log', 'batch_size': 256, 'lr': 0.0001, 'num_epochs': 999, 'patience': 5, 'seed': 42, 'device': 'cuda', 'num_workers': 8, 'buffer_batches': 20, 'train_ratio': 1.0, 'valid_ratio': 0.1, 'eval_every_n_steps': 0, 'seq_max_lens': 'seq_a:256,seq_b:256,seq_c:512,seq_d:512', 'd_model': 64, 'emb_dim': 64, 'num_queries': 2, 'num_hyformer_blocks': 2, 'num_heads': 4, 'seq_encoder_type': 'transformer', 'hidden_mult': 4, 'dropout_rate': 0.01, 'seq_top_k': 50, 'seq_causal': False, 'action_num': 1, 'use_time_buckets': True, 'rank_mixer_mode': 'full', 'use_rope': False, 'rope_base': 10000.0, 'loss_type': 'bce_pairwise', 'focal_alpha': 0.1, 'focal_gamma': 2.0, 'pairwise_auc_weight': 0.05, 'pairwise_max_pairs': 8192, 'sparse_lr': 0.05, 'sparse_weight_decay': 0.0, 'reinit_sparse_after_epoch': 1, 'reinit_cardinality_threshold': 0, 'emb_skip_threshold': 1000000, 'seq_id_threshold': 10000, 'ns_groups_json': '/home/taiji/dl/runtime/script/ns_groups.json', 'ns_tokenizer_type': 'rankmixer', 'user_ns_tokens': 5, 'item_ns_tokens': 2, 'tf_events_dir': '/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509111349_273a0efe/95d1b0069dfa2b61019e0abb06792c6d/events'}

2026-05-09 11:15:27.402 2026-05-09 11:15:27.402496: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.

2026-05-09 11:15:27.410 2026-05-09 11:15:27.410805: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered

2026-05-09 11:15:27.422 WARNING: All log messages before absl::InitializeLog() is called are written to STDERR

2026-05-09 11:15:27.422 E0000 00:00:1778296527.422194 270 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered

2026-05-09 11:15:27.425 E0000 00:00:1778296527.425635 270 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered

2026-05-09 11:15:27.435 W0000 00:00:1778296527.435374 270 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.

2026-05-09 11:15:27.435 W0000 00:00:1778296527.435396 270 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.

2026-05-09 11:15:27.435 W0000 00:00:1778296527.435397 270 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.

2026-05-09 11:15:27.435 W0000 00:00:1778296527.435399 270 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.

2026-05-09 11:15:27.438 2026-05-09 11:15:27.438598: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.

2026-05-09 11:15:27.438 To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

2026-05-09 11:15:31.727 05/09/26 11:15:31 - 0:00:05 - generated new fontManager

2026-05-09 11:15:32.145 05/09/26 11:15:32 - 0:00:05 - Seq max_lens override: {'seq_a': 256, 'seq_b': 256, 'seq_c': 512, 'seq_d': 512}

2026-05-09 11:15:32.146 05/09/26 11:15:32 - 0:00:05 - Using Parquet data format (IterableDataset)

2026-05-09 11:15:32.656 05/09/26 11:15:32 - 0:00:06 - Row Group split: 900 train (907381 rows), 100 valid (102619 rows)

2026-05-09 11:15:33.166 05/09/26 11:15:33 - 0:00:06 - PCVRParquetDataset: 907381 rows from 1000 file(s), batch_size=256, buffer_batches=20, shuffle=True

2026-05-09 11:15:33.675 05/09/26 11:15:33 - 0:00:07 - PCVRParquetDataset: 102619 rows from 1000 file(s), batch_size=256, buffer_batches=0, shuffle=False

2026-05-09 11:15:33.676 05/09/26 11:15:33 - 0:00:07 - Parquet train: 907381 rows, valid: 102619 rows, batch_size=256, buffer_batches=20

2026-05-09 11:15:33.676 05/09/26 11:15:33 - 0:00:07 - Loading NS groups from /home/taiji/dl/runtime/script/ns_groups.json

2026-05-09 11:15:33.676 05/09/26 11:15:33 - 0:00:07 - User NS groups (7): ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7']

2026-05-09 11:15:33.676 05/09/26 11:15:33 - 0:00:07 - Item NS groups (4): ['I1', 'I2', 'I3', 'I4']

2026-05-09 11:15:33.700 05/09/26 11:15:33 - 0:00:07 - RankMixerNSTokenizer: 46 fids, total_emb_dim=2944, chunk_dim=589, num_ns_tokens=5, pad=1

2026-05-09 11:15:33.719 05/09/26 11:15:33 - 0:00:07 - RankMixerNSTokenizer: 14 fids, total_emb_dim=896, chunk_dim=448, num_ns_tokens=2, pad=0

2026-05-09 11:15:35.770 05/09/26 11:15:35 - 0:00:09 - emb_skip_threshold=1000000: seq_b skipped 1/13 features

2026-05-09 11:15:35.770 05/09/26 11:15:35 - 0:00:09 - emb_skip_threshold=1000000: seq_c skipped 3/11 features

2026-05-09 11:15:39.974 05/09/26 11:15:39 - 0:00:13 - PCVRHyFormer model created: num_ns=8, T=16, d_model=64, rank_mixer_mode=full

2026-05-09 11:15:39.974 05/09/26 11:15:39 - 0:00:13 - User NS groups: [[0, 3], [4, 5, 25, 26, 27], [22], [7, 8, 9, 10, 24], [23, 28, 29], [6, 16, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], [1, 2, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21]]

2026-05-09 11:15:39.974 05/09/26 11:15:39 - 0:00:13 - Item NS groups: [[6, 8], [0, 1, 2, 3, 7], [9, 10, 11, 12, 13], [4, 5]]

2026-05-09 11:15:39.975 05/09/26 11:15:39 - 0:00:13 - Total parameters: 239,931,649

2026-05-09 11:15:39.978 05/09/26 11:15:39 - 0:00:13 - Sparse params: 98 tensors, 237,435,776 parameters (Adagrad lr=0.05)

2026-05-09 11:15:39.978 05/09/26 11:15:39 - 0:00:13 - Dense params: 380 tensors, 2,495,873 parameters (AdamW lr=0.0001)

2026-05-09 11:15:42.620 05/09/26 11:15:42 - 0:00:16 - PCVRHyFormerRankingTrainer loss_type=bce_pairwise, focal_alpha=0.1, focal_gamma=2.0, pairwise_auc_weight=0.05, pairwise_max_pairs=8192, reinit_sparse_after_epoch=1

2026-05-09 11:15:42.620 Start training (PCVRHyFormer)

2026-05-09 11:15:49.031 0%| | 0/3733 [00:00<?, ?it/s] 0%| | 0/3733 [00:06<?, ?it/s]

2026-05-09 11:15:49.032 Traceback (most recent call last):

2026-05-09 11:15:49.032 File "/home/taiji/dl/runtime/script/train.py", line 396, in <module>

2026-05-09 11:15:49.032 main()

2026-05-09 11:15:49.032 File "/home/taiji/dl/runtime/script/train.py", line 389, in main

2026-05-09 11:15:49.032 trainer.train()

2026-05-09 11:15:49.032 File "/home/taiji/dl/runtime/script/trainer.py", line 329, in train

2026-05-09 11:15:49.032 loss = self._train_step(batch)

2026-05-09 11:15:49.032 File "/home/taiji/dl/runtime/script/trainer.py", line 437, in _train_step

2026-05-09 11:15:49.032 logits = self.model(model_input) # (B, 1)

2026-05-09 11:15:49.032 File "/opt/conda/envs/competition/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl

2026-05-09 11:15:49.032 return self._call_impl(*args, **kwargs)

2026-05-09 11:15:49.032 File "/opt/conda/envs/competition/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl

2026-05-09 11:15:49.032 return forward_call(*args, **kwargs)

2026-05-09 11:15:49.032 File "/home/taiji/dl/runtime/script/model.py", line 1668, in forward

2026-05-09 11:15:49.032 output = self._run_multi_seq_blocks(

2026-05-09 11:15:49.032 File "/home/taiji/dl/runtime/script/model.py", line 1617, in _run_multi_seq_blocks

2026-05-09 11:15:49.033 curr_qs, curr_ns, curr_seqs, curr_masks = block(

2026-05-09 11:15:49.033 File "/opt/conda/envs/competition/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl

2026-05-09 11:15:49.033 return self._call_impl(*args, **kwargs)

2026-05-09 11:15:49.033 File "/opt/conda/envs/competition/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl

2026-05-09 11:15:49.033 return forward_call(*args, **kwargs)

2026-05-09 11:15:49.033 File "/home/taiji/dl/runtime/script/model.py", line 960, in forward

2026-05-09 11:15:49.033 decoded_q_i = self.cross_attns[i](

2026-05-09 11:15:49.033 File "/opt/conda/envs/competition/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl

2026-05-09 11:15:49.033 return self._call_impl(*args, **kwargs)

2026-05-09 11:15:49.033 File "/opt/conda/envs/competition/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl

2026-05-09 11:15:49.033 return forward_call(*args, **kwargs)

2026-05-09 11:15:49.033 File "/home/taiji/dl/runtime/script/model.py", line 296, in forward

2026-05-09 11:15:49.033 key_value = self.norm_kv(key_value)

2026-05-09 11:15:49.033 File "/opt/conda/envs/competition/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl

2026-05-09 11:15:49.034 return self._call_impl(*args, **kwargs)

2026-05-09 11:15:49.034 File "/opt/conda/envs/competition/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl

2026-05-09 11:15:49.034 return forward_call(*args, **kwargs)

2026-05-09 11:15:49.034 File "/opt/conda/envs/competition/lib/python3.10/site-packages/torch/nn/modules/normalization.py", line 217, in forward

2026-05-09 11:15:49.034 return F.layer_norm(

2026-05-09 11:15:49.034 File "/opt/conda/envs/competition/lib/python3.10/site-packages/torch/nn/functional.py", line 2910, in layer_norm

2026-05-09 11:15:49.034 return torch.layer_norm(

2026-05-09 11:15:49.034 torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB. GPU 0 has a total capacity of 19.12 GiB of which 19.12 GiB is free. Process 3886389 has 9.56 GiB memory in use. Process 3902710 has 46.79 GiB memory in use. Process 438488 has 3.28 GiB memory in use. Process 442226 has 22.25 GiB memory in use. Process 513993 has 13.06 GiB memory in use. Of the allocated memory 12.61 GiB is allocated by PyTorch, and 22.90 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

2026-05-09 11:16:30.848 INFO: ======= End cmd =======

2026-05-09 11:16:30.853 [TAIJI] user exit code: 1

Truncated to last 1000 lines.