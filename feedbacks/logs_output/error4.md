2026-05-09 11:03:16.274 ##########################################################################

2026-05-09 11:03:16.274 ############################ TAIJI JOB START #############################

2026-05-09 11:03:16.274 ##########################################################################

2026-05-09 11:03:16.290 cp: cannot stat './/*': No such file or directory

2026-05-09 11:03:16.299 passwd: password expiry information changed.

2026-05-09 11:03:21.836 Complete setting taiji user.

2026-05-09 11:03:27.675 update taiji environ ...

2026-05-09 11:03:27.675 show args: --appName=95d1b0069dfa2b61019e0ab03ccb2c44 --projectName=external-ams-competition-2025 --module=pytorch --script=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/run.sh --script_archive=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/dataset.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/model.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/ns_groups.json,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/train.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/trainer.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/utils.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/.taiji_run.json --script_args=

2026-05-09 11:03:27.777 INFO: ===begin to run init cmd===

2026-05-09 11:03:27.778 INFO: Export env: RUNTIME_SCRIPT_DIR=/home/taiji/dl/runtime/script

2026-05-09 11:03:27.792 INFO: Export env: RUNTIME_SCRIPT_SHARE_DIR=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/95d1b0069dfa2b61019e0ab03ccb2c44/pipeline/95d1b0069dfa2b61019e0ab03ccb2c44/share

2026-05-09 11:03:27.792 INFO: script_path :/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/run.sh

2026-05-09 11:03:27.800 INFO: Success copy to /home/taiji/dl/runtime/script/run.sh

2026-05-09 11:03:27.809 INFO: Success copy to /home/taiji/dl/runtime/script/dataset.py

2026-05-09 11:03:27.816 INFO: Success copy to /home/taiji/dl/runtime/script/model.py

2026-05-09 11:03:27.824 INFO: Success copy to /home/taiji/dl/runtime/script/ns_groups.json

2026-05-09 11:03:27.831 INFO: Success copy to /home/taiji/dl/runtime/script/train.py

2026-05-09 11:03:27.839 INFO: Success copy to /home/taiji/dl/runtime/script/trainer.py

2026-05-09 11:03:27.846 INFO: Success copy to /home/taiji/dl/runtime/script/utils.py

2026-05-09 11:03:27.853 INFO: Success copy to /home/taiji/dl/runtime/script/.taiji_run.json

2026-05-09 11:03:27.853 INFO: python path: /home/taiji/dl/runtime/script/dataset.py:/home/taiji/dl/runtime/script/model.py:/home/taiji/dl/runtime/script/ns_groups.json:/home/taiji/dl/runtime/script/train.py:/home/taiji/dl/runtime/script/trainer.py:/home/taiji/dl/runtime/script/utils.py:/home/taiji/dl/runtime/script/.taiji_run.json

2026-05-09 11:03:27.853 INFO: Export env: TRAIN_CKPT_PATH=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/95d1b0069dfa2b61019e0ab03ccb2c44/ckpt

2026-05-09 11:03:27.853 INFO: Export env: TRAIN_TF_EVENTS_PATH=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/95d1b0069dfa2b61019e0ab03ccb2c44/events

2026-05-09 11:03:27.853 INFO: Export env: TRAIN_DATA_PATH=/data_ams/academic_training_data

2026-05-09 11:03:27.853 INFO: Export env: TRAIN_USER=ams_2026_1029731852466347124

2026-05-09 11:03:27.853 INFO: Export env: TRAIN_LOG_PATH=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/95d1b0069dfa2b61019e0ab03ccb2c44/log

2026-05-09 11:03:27.853 INFO: Export env: TRAIN_CACHE_PATH=/apdcephfs_fsgm2/share_305170765/angel/groups/44/ams_2026_1029731852466347124

2026-05-09 11:03:27.853 INFO: Export env: USER_CACHE_PATH=/apdcephfs_fsgm2/share_305170765/angel/groups/44/ams_2026_1029731852466347124

2026-05-09 11:03:27.853 INFO: Export env: NP_BD_FLOW_PROJ=7971753

2026-05-09 11:03:27.864 touch: cannot touch '/data/initCmdCompletedFile': No such file or directory

2026-05-09 11:03:27.864 initCmd failed

2026-05-09 11:03:27.865 [TAIJI] Starting sshd

2026-05-09 11:03:28.887 [TAIJI] sshd started

2026-05-09 11:03:28.887 [TAIJI] waiting for all peers to be ready...

2026-05-09 11:03:30.161 Welcome to Ubuntu 22.04.5 LTS (GNU/Linux 5.4.241-1-tlinux4-0017.7 x86_64)

2026-05-09 11:03:30.161

2026-05-09 11:03:30.161 * Documentation: https://help.ubuntu.com

2026-05-09 11:03:30.161 * Management: https://landscape.canonical.com

2026-05-09 11:03:30.161 * Support: https://ubuntu.com/pro

2026-05-09 11:03:30.161

2026-05-09 11:03:30.161 This system has been minimized by removing packages and content that are

2026-05-09 11:03:30.161 not required on a system that users do not log into.

2026-05-09 11:03:30.161

2026-05-09 11:03:30.161 To restore this content, you can run the 'unminimize' command.

2026-05-09 11:03:30.161

2026-05-09 11:03:30.161 The programs included with the Ubuntu system are free software;

2026-05-09 11:03:30.161 the exact distribution terms for each program are described in the

2026-05-09 11:03:30.161 individual files in /usr/share/doc/*/copyright.

2026-05-09 11:03:30.161

2026-05-09 11:03:30.161 Ubuntu comes with ABSOLUTELY NO WARRANTY, to the extent permitted by

2026-05-09 11:03:30.161 applicable law.

2026-05-09 11:03:30.161

2026-05-09 11:03:31.341 [TAIJI] error log above can be ignored

2026-05-09 11:03:31.346 [TAIJI] run command 1778295811 [/bin/bash -c "${start_script}"]

2026-05-09 11:03:35.241 Complete setting network policy rules.

2026-05-09 11:03:41.028 update taiji environ ...

2026-05-09 11:03:41.028 show args: --appName=95d1b0069dfa2b61019e0ab03ccb2c44 --projectName=external-ams-competition-2025n --module=pytorch --script=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/run.sh --script_archive=/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/dataset.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/model.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/ns_groups.json,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/train.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/trainer.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/utils.py,/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/.taiji_run.json --script_args=

2026-05-09 11:03:41.073 INFO: ===begin to run starttorch cmd===

2026-05-09 11:03:41.073 INFO: runtime_script_share_dir is /apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/95d1b0069dfa2b61019e0ab03ccb2c44/pipeline/95d1b0069dfa2b61019e0ab03ccb2c44/share

2026-05-09 11:03:41.073 INFO: script_path :/apdcephfs_fsgm2/share_305170765/angel/ams_2026_1029731852466347124/angel_training_ams_2026_1029731852466347124_20260509110259_9fe4b94a/scripts/95d1b0069dfa2b61019e0ab03ccb2c44/run.sh

2026-05-09 11:03:41.074 INFO: framework is pytorch

2026-05-09 11:03:41.074 INFO: Run command: bash /home/taiji/dl/runtime/script/run.sh

2026-05-09 11:03:41.074 INFO: ====== Begin cmd ======

2026-05-09 11:03:42.219 PLATFORM_TRAIN_CONFIG {"ns_tokenizer_type":"rankmixer","user_ns_tokens":5,"item_ns_tokens":2,"num_queries":2,"ns_groups_json":"/home/taiji/dl/runtime/script/ns_groups.json","emb_skip_threshold":1000000,"num_workers":8,"seq_max_lens":"seq_a:256,seq_b:256,seq_c:512,seq_d:512","d_model":64,"num_hyformer_blocks":2,"num_heads":4,"seq_encoder_type":"transformer","loss_type":"bce_pairwise","batch_size":256,"pairwise_auc_weight":0.05,"pairwise_max_pairs":8192}

2026-05-09 11:03:42.539 [DEBUG][libvgpu]hijack_call.c:106 [p:267 t:267]init cuda hook lib

2026-05-09 11:03:42.539 [DEBUG][libvgpu]hijack_call.c:107 [p:267 t:267]Thread pid:267, tid:267

2026-05-09 11:03:42.539 [DEBUG][libvgpu]hijack_call.c:125 [p:267 t:267]env_ld_library_path: /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

2026-05-09 11:03:42.543 [DEBUG][libvgpu]hijack_call.c:172 [p:267 t:267]hooked env NCCL_SET_THREAD_NAME to : 1

2026-05-09 11:03:42.543 [DEBUG][libvgpu]hijack_call.c:173 [p:267 t:267]hooked LD_LIBRARY_PATH to : /usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

2026-05-09 11:03:42.543 [DEBUG][libvgpu]hijack_call.c:174 [p:267 t:267]hooked libnvml_realpath to : /lib/x86_64-linux-gnu/libnvidia-ml.so.1

2026-05-09 11:03:42.543 [DEBUG][libvgpu]hijack_call.c:175 [p:267 t:267]hooked libcuda_realpath to : /lib/x86_64-linux-gnu/libcuda.so.1

2026-05-09 11:03:46.707 usage: train.py [-h] [--data_dir DATA_DIR] [--schema_path SCHEMA_PATH]

2026-05-09 11:03:46.707 [--ckpt_dir CKPT_DIR] [--log_dir LOG_DIR]

2026-05-09 11:03:46.707 [--batch_size BATCH_SIZE] [--lr LR] [--num_epochs NUM_EPOCHS]

2026-05-09 11:03:46.707 [--patience PATIENCE] [--seed SEED] [--device DEVICE]

2026-05-09 11:03:46.707 [--num_workers NUM_WORKERS] [--buffer_batches BUFFER_BATCHES]

2026-05-09 11:03:46.707 [--train_ratio TRAIN_RATIO] [--valid_ratio VALID_RATIO]

2026-05-09 11:03:46.707 [--eval_every_n_steps EVAL_EVERY_N_STEPS]

2026-05-09 11:03:46.707 [--seq_max_lens SEQ_MAX_LENS] [--d_model D_MODEL]

2026-05-09 11:03:46.707 [--emb_dim EMB_DIM] [--num_queries NUM_QUERIES]

2026-05-09 11:03:46.707 [--num_hyformer_blocks NUM_HYFORMER_BLOCKS]

2026-05-09 11:03:46.707 [--num_heads NUM_HEADS]

2026-05-09 11:03:46.707 [--seq_encoder_type {swiglu,transformer,longer}]

2026-05-09 11:03:46.707 [--hidden_mult HIDDEN_MULT] [--dropout_rate DROPOUT_RATE]

2026-05-09 11:03:46.707 [--seq_top_k SEQ_TOP_K] [--seq_causal]

2026-05-09 11:03:46.707 [--action_num ACTION_NUM] [--use_time_buckets]

2026-05-09 11:03:46.707 [--no_time_buckets] [--rank_mixer_mode {full,ffn_only,none}]

2026-05-09 11:03:46.707 [--use_rope] [--rope_base ROPE_BASE] [--loss_type {bce,focal}]

2026-05-09 11:03:46.707 [--focal_alpha FOCAL_ALPHA] [--focal_gamma FOCAL_GAMMA]

2026-05-09 11:03:46.707 [--sparse_lr SPARSE_LR]

2026-05-09 11:03:46.707 [--sparse_weight_decay SPARSE_WEIGHT_DECAY]

2026-05-09 11:03:46.707 [--reinit_sparse_after_epoch REINIT_SPARSE_AFTER_EPOCH]

2026-05-09 11:03:46.707 [--reinit_cardinality_threshold REINIT_CARDINALITY_THRESHOLD]

2026-05-09 11:03:46.707 [--emb_skip_threshold EMB_SKIP_THRESHOLD]

2026-05-09 11:03:46.707 [--seq_id_threshold SEQ_ID_THRESHOLD]

2026-05-09 11:03:46.707 [--ns_groups_json NS_GROUPS_JSON]

2026-05-09 11:03:46.707 [--ns_tokenizer_type {group,rankmixer}]

2026-05-09 11:03:46.707 [--user_ns_tokens USER_NS_TOKENS]

2026-05-09 11:03:46.707 [--item_ns_tokens ITEM_NS_TOKENS]

2026-05-09 11:03:46.707 train.py: error: argument --loss_type: invalid choice: 'bce_pairwise' (choose from 'bce', 'focal')

2026-05-09 11:03:47.183 INFO: ======= End cmd =======

2026-05-09 11:03:47.189 [TAIJI] user exit code: 2