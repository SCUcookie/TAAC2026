# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview



### 赛题介绍

**全模态生成式推荐（All-Modality Generative Recommendation）**

参赛者需基于**脱敏**的用户历史行为序列，综合**协同、文本、视觉等多模态**信息，预测用户**下一次可能交互的广告**。每条行为包含广告 ID 类特征与多模态信息（如图像、文本等）。大赛提供基线模型，并在方案评审环节引导选手**突破传统判别式推荐框架**，探索**生成式推荐**方向。


### 模型训练

1. 环境变量

平台提供了训练过程中，需要用到的目录（比如训练数据集、产出、日志等），通过环境变量的方式传递到容器中，你的脚本可以读取这些变量。

| 变量名               | 描述                                                         |
| -------------------- | ------------------------------------------------------------ |
| USER_CACHE_PATH      | 用户缓存路径，配额20GB。训练、评测都会提供该变量，你可以使用来在训练、评测中共享一些文件。 |
| TRAIN_DATA_PATH      | 训练数据集路径                                               |
| TRAIN_CKPT_PATH      | 模型产出路径                                                 |
| TRAIN_TF_EVENTS_PATH | Tensorboard指标产出路径                                      |

### 模型评测

1. 环境变量

平台提供了评测过程中，需要用到的目录（比如训练数据集、产出、日志等），通过环境变量的方式传递到容器中，你的脚本可以读取这些变量。

| 变量名            | 描述                                                         |
| ----------------- | ------------------------------------------------------------ |
| USER_CACHE_PATH   | 用户缓存路径，配额20GB。训练、评测都会提供该变量，你可以使用来在训练、评测中共享一些文件。 |
| MODEL_OUTPUT_PATH | 模型产出路径                                                 |
| EVAL_DATA_PATH    | 数据集、推理候选集、检索索引路径                             |
| EVAL_RESULT_PATH  | 产出物路径，存放中间结果和result.json                        |
| EVAL_INFER_PATH   | 选手上传推理脚本文件目录                                     |

2. 产出规范

可以参考平台提供的脚本infer.py并实现推理函数，最终参赛者须在评测中返回top10s，user_list。

| 返回值    | 数据格式   | 说明                                                         |
| --------- | ---------- | ------------------------------------------------------------ |
| top10s    | list[list] | 每一个参赛者推理出由10个item_id组成的list，所有参赛者的list组合成一个新的list<br/>示例:[[a1,a2...a10],[b1,b2...b10],...[n1,n2...n10] |
| user_list | list[str]  | user_id组成的list，其中user_id为str格式示例:['user a','user_b',...'user_xxx'] |

```
注意:top10s的第一个list应为user_list中第一个用户的序列推理出的结果，以此类推。若顺序不一致，会影响最终得分!!
```





### 训练数据集文件（在环境变量TRAIN_DATA_PATH目录下）：

* indexer.pkl

* item_feat_dict.json

* item_feat_num.json

* seq.jsonl

* seq_offsets.pkl

* user_feat_num.json



### 评测数据文件（在环境变量EVAL_DATA_PATH目录下）：

- indexer.pkl

- item_feat_dict.json

- item_feat_num.json

- predict_seq.jsonl

- predict_seq_offsets.pkl

- predict_set.jsonl

- user_feat_num.json

### Training
```bash
# Set up environment and run training
./run.sh

# Or run directly with Python
python -u main.py
```

### Inference
```bash
python infer.py
```

### Environment Variables (set by run.sh)
- `TRAIN_LOG_PATH`: "./logs" - Training logs directory
- `TRAIN_TF_EVENTS_PATH`: "./tf_events" - TensorBoard events
- `TRAIN_DATA_PATH`: "./dataset/TencentGR_1k" - Training data directory
- `TRAIN_CKPT_PATH`: "./ckpt" - Model checkpoint directory
- `MODEL_OUTPUT_PATH`: Used by infer.py to locate trained model

### Data Analysis
The project includes comprehensive dataset analysis scripts in `explore_dataset/1m_real_dataset/`:
```bash
# Run complete dataset analysis
python explore_dataset/1m_real_dataset/run_all_analysis.py
```

## Model Parameters

Key hyperparameters configurable via command line:
- `--batch_size`: Default 256 for training, 128 for inference
- `--lr`: Learning rate, default 0.002 for training, 0.0007 for inference  
- `--hidden_units`: Hidden dimension, default 128
- `--num_blocks`: Transformer blocks, default 8
- `--num_heads`: Attention heads, default 8
- `--maxlen`: Maximum sequence length, default 101
- `--mm_emb_id`: Multi-modal embedding IDs (81-86)
- `--loss_type`: Loss function type (bce, infonce, mix)

## Data Structure

Dataset located in `dataset/TencentGR_1k/` contains:
- `seq_offsets.pkl`: Sequence position offsets
- `item_feat_dict.json`: Item feature dictionary
- User sequence files and feature mappings