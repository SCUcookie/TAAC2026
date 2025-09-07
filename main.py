# 全模态生成式推荐系统 - 训练主程序
# 用于腾讯算法大赛的多模态推荐模型训练

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MyDataset  # 导入自定义数据集类
from model import BaselineModel  # 导入基线模型


def compute_embeddings(model, seq, seq_feat, token_type):
    """
    计算序列的embedding表示
    
    Args:
        model: 模型实例
        seq: 序列ID
        seq_feat: 序列特征
        token_type: token类型
    
    Returns:
        embedding: 序列的embedding表示
    """
    log_feats = model.log2feats(seq, token_type, seq_feat)
    return log_feats


def get_args():
    """
    解析命令行参数，配置模型训练的各种超参数
    
    Returns:
        args: 包含所有训练参数的命名空间对象
    """
    parser = argparse.ArgumentParser()

    # 训练基础参数
    parser.add_argument('--batch_size', default=128, type=int, help='训练批次大小')
    parser.add_argument('--lr', default=0.002, type=float, help='学习率')
    parser.add_argument('--maxlen', default=101, type=int, help='用户序列最大长度')

    # 基线模型结构参数
    parser.add_argument('--hidden_units', default=128, type=int, help='隐藏层维度')
    parser.add_argument('--num_blocks', default=8, type=int, help='Transformer块数量')
    parser.add_argument('--num_epochs', default=15, type=int, help='训练轮数')
    parser.add_argument('--num_heads', default=8, type=int, help='注意力头数')
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='Dropout比例')
    parser.add_argument('--l2_emb', default=0.0, type=float, help='Embedding的L2正则化系数')
    parser.add_argument('--device', default='cuda', type=str, help='计算设备（cuda/cpu）')
    parser.add_argument('--inference_only', action='store_true', help='仅推理模式，不训练')
    parser.add_argument('--state_dict_path', default=None, type=str, help='预训练模型权重路径')
    parser.add_argument('--norm_first', default=True, type=bool, help='是否在注意力前先归一化')

    # 多模态嵌入特征ID（81-86对应不同的多模态特征）
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, 
                       choices=[str(s) for s in range(81, 87)], help='激活的多模态特征ID列表')

    # InfoNCE对比学习相关参数
    parser.add_argument('--loss_type', choices=['bce', 'infonce', 'mix'], default='infonce', 
                       help='损失函数类型：bce(二元交叉熵)、infonce(对比学习)、mix(混合损失)')
    parser.add_argument('--temp', type=float, default=0.03, help='InfoNCE损失的温度系数，控制相似度分布的锐利程度')
    parser.add_argument('--contrastive_weight', type=float, default=1.0, help='对比学习损失的权重')
    parser.add_argument('--neg_topk', type=int, default=256, help='选择top-k难负样本的数量，0表示使用所有负样本')
    parser.add_argument('--norm_output', action='store_true', help='是否对输出embedding进行L2归一化')
    parser.add_argument('--tau_schedule', choices=['const', 'cosine', 'linear'], default='const', 
                       help='温度系数调度策略')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='标签平滑参数')
    parser.add_argument('--use_action_weight', action='store_true', help='是否使用动作权重（曝光vs点击）')

    # 优化器相关参数
    parser.add_argument('--warmup_steps', default=800, type=int, help='学习率预热步数')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='权重衰减系数')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    训练主程序入口
    1. 初始化日志和TensorBoard
    2. 加载数据集和创建DataLoader
    3. 创建模型并初始化参数
    4. 设置优化器和开始训练循环
    """
    
    # 创建训练日志和TensorBoard事件目录
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))  # TensorBoard日志写入器
    
    # 从环境变量获取训练数据路径
    data_path = os.environ.get('TRAIN_DATA_PATH')

    # 解析命令行参数
    args = get_args()
    
    # 创建训练数据集和数据加载器
    dataset = MyDataset(data_path, args)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum  # 获取用户和物品数量
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types  # 获取特征统计信息

    # 创建基线模型并移动到指定设备（GPU/CPU）
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    # 初始化模型参数（除了Embedding层）
    for name, param in model.named_parameters():
        if("_emb" in name):  # 跳过embedding参数的初始化
            continue
        try:
            print('初始化{}参数，参数量{}'.format(name, param.data.size()))
            torch.nn.init.xavier_normal_(param.data)  # 使用Xavier正态分布初始化
        except Exception:
            pass

    # 将padding索引（索引为0）的embedding设置为0向量
    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    # 将所有稀疏特征embedding的padding索引设置为0向量
    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1  # 训练开始的epoch索引

    # 如果指定了预训练模型路径，加载预训练权重
    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            # 从文件名中解析出epoch信息，用于继续训练
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    # 定义损失函数和优化器
    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')  # 二元交叉熵损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)

    # 初始化训练监控变量
    best_val_ndcg, best_val_hr = 0.0, 0.0  # 最佳验证集NDCG和Hit Rate
    best_test_ndcg, best_test_hr = 0.0, 0.0  # 最佳测试集NDCG和Hit Rate
    T = 0.0  # 训练总时间
    t0 = time.time()  # 训练开始时间
    global_step = 0  # 全局训练步数计数器
    
    print("Start training")
    
    # 主训练循环：遍历每个epoch
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()  # 设置模型为训练模式
        
        # 如果只是推理模式，跳出训练循环
        if args.inference_only:
            break
            
        # 遍历训练数据的每个batch
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            # 解包batch数据
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            # 将数据移动到指定设备
            seq = seq.to(args.device)  # 用户序列ID
            pos = pos.to(args.device)  # 正样本ID
            neg = neg.to(args.device)  # 负样本ID
            
            optimizer.zero_grad()  # 清零梯度
            
            # 将token类型和动作类型移动到设备
            next_token_type = next_token_type.to(args.device)
            token_type = token_type.to(args.device)
            
            # 前向传播：计算正负样本的logits和损失掩码
            pos_logits, neg_logits, loss_mask = model(seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat)
            
            # 创建有效样本掩码（只在有下一个item的位置计算损失）
            valid_mask = loss_mask > 0
            indices = torch.where(loss_mask)[0] if loss_mask.any() else torch.tensor([], dtype=torch.long, device=args.device)
            # 根据损失类型计算损失
            if args.loss_type == 'bce':
                # BCE（二元交叉熵）损失：传统的推荐系统训练方式
                # 正样本标签为1，负样本标签为0
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                    neg_logits.shape, device=args.device
                )
                if len(indices) > 0:
                    # 只在有效位置计算BCE损失
                    loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                    loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                else:
                    # 如果没有有效样本，设置损失为0
                    loss = torch.tensor(0.0, device=args.device, requires_grad=True)
                
            elif args.loss_type == 'infonce':
                # InfoNCE损失：对比学习方式，更适合生成式推荐
                # 通过对比正样本和负样本的相似度来学习用户偏好表征
                seq_embs = compute_embeddings(model, seq, seq_feat, token_type)  # 计算序列表征
                pos_embs = model.feat2emb(pos, pos_feat, include_user=False)  # 正样本表征
                neg_embs = model.feat2emb(neg, neg_feat, include_user=False)  # 负样本表征
                
                # 计算InfoNCE对比损失
                loss, pos_mean_logit, neg_mean_logit = model.compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask)
                
                # 记录训练指标到TensorBoard
                with torch.no_grad():
                    writer.add_scalar("Model/pos_mean_logit", pos_mean_logit, global_step)  # 正样本平均相似度
                    writer.add_scalar("Model/neg_mean_logits", neg_mean_logit, global_step)  # 负样本平均相似度
                
            elif args.loss_type == 'mix':
                # 混合损失：结合BCE损失和InfoNCE损失的优势
                # 1. 计算BCE损失部分
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                    neg_logits.shape, device=args.device
                )
                if len(indices) > 0:
                    bce_loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                    bce_loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                else:
                    bce_loss = torch.tensor(0.0, device=args.device, requires_grad=True)
                
                # 2. 计算InfoNCE损失部分
                seq_embs = compute_embeddings(model, seq, seq_feat, token_type)
                pos_embs = model.feat2emb(pos, pos_feat, include_user=False)
                neg_embs = model.feat2emb(neg, neg_feat, include_user=False)
                
                infonce_loss = model.compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask)
                
                # 3. 组合损失（这里只使用BCE，可以调整权重组合）
                loss = bce_loss
                
                # 记录各项损失到TensorBoard
                writer.add_scalar('Loss/train_bce', bce_loss.item(), global_step)
                writer.add_scalar('Loss/train_infonce', infonce_loss.item(), global_step)
                
                # 记录InfoNCE相关的相似度指标
                with torch.no_grad():
                    # L2归一化embedding
                    seq_embs_norm = torch.nn.functional.normalize(seq_embs, p=2, dim=-1)
                    pos_embs_norm = torch.nn.functional.normalize(pos_embs, p=2, dim=-1)
                    neg_embs_norm = torch.nn.functional.normalize(neg_embs, p=2, dim=-1)
                    
                    # 计算正样本和负样本相似度
                    pos_sim = torch.nn.functional.cosine_similarity(seq_embs_norm, pos_embs_norm, dim=-1)
                    neg_sim = torch.matmul(seq_embs_norm.reshape(-1, seq_embs_norm.size(-1)), 
                                         neg_embs_norm.reshape(-1, neg_embs_norm.size(-1)).transpose(-1, -2))
                    
                    # 记录相似度指标
                    writer.add_scalar("Model/nce_pos_logits", pos_sim[loss_mask.bool()].mean().item(), global_step)
                    writer.add_scalar("Model/nce_neg_logits", neg_sim.mean().item(), global_step)
                    writer.add_scalar("Model/temp", model.temp, global_step)  # 温度系数

            # 记录训练日志
            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)  # 记录总损失

            global_step += 1  # 增加全局步数计数

            # 添加L2正则化损失到item embedding
            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            
            # 反向传播计算梯度
            loss.backward()

            # 学习率预热调度：在预热阶段逐渐增加学习率
            if global_step < args.warmup_steps:
                lr_scalar = min(1.0, float(global_step + 1) / args.warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scalar * args.lr
                lr = lr_scalar * args.lr
            else:
                lr = args.lr
            
            # 梯度裁剪：防止梯度爆炸
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            writer.add_scalar("Trainer/grad_norm_before_clip", total_norm, global_step)
            
            # 记录裁剪后的梯度范数
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            writer.add_scalar("Trainer/grad_norm_after_clip", total_norm, global_step)
            
            writer.add_scalar("Trainer/lr", lr, global_step)  # 记录当前学习率
            
            # 更新模型参数
            optimizer.step()

        # 每个epoch结束后保存模型检查点
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    # 关闭日志文件和TensorBoard写入器
    writer.close()
    log_file.close()
