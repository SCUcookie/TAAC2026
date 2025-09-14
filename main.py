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

from dataset import MyDataset
from model import BaselineModel


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
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument('--num_heads', default=16, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', default=True, type=bool)

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # InfoNCE 相关参数
    parser.add_argument('--loss_type', choices=['bce', 'infonce', 'mix'], default='infonce', help='选择损失函数类型')
    parser.add_argument('--temp', type=float, default=0.02, help='InfoNCE损失的温度系数')
    parser.add_argument('--contrastive_weight', type=float, default=1.0, help='对比学习损失的权重')
    parser.add_argument('--neg_topk', type=int, default=64, help='选择top-k难负样本的数量，0表示使用所有负样本')
    parser.add_argument('--norm_output', action='store_true', help='是否对输出embedding进行L2归一化')
    parser.add_argument('--tau_schedule', choices=['const', 'cosine', 'linear'], default='const', help='温度系数调度策略')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='标签平滑参数')
    parser.add_argument('--use_action_weight', action='store_true', help='是否使用动作权重')

    #训练参数相关
    parser.add_argument('--warmup_steps', default=800, type=int)
    parser.add_argument('--weight_decay', default=0.01, type=float)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # torch.manual_seed(42)

    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    data_path = os.environ.get('TRAIN_DATA_PATH')

    args = get_args()
    dataset = MyDataset(data_path, args)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=dataset.collate_fn
    )
    usernum, itemnum = dataset.usernum, dataset.itemnum
    feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types

    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)

    for name, param in model.named_parameters():
        if("_emb" in name):
            continue
        try:
            print('初始化{}参数，参数量{}'.format(name, param.data.size()))
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass


    model.item_emb.weight.data[0, :] = 0
    model.user_emb.weight.data[0, :] = 0

    for k in model.sparse_emb:
        model.sparse_emb[k].weight.data[0, :] = 0

    epoch_start_idx = 1

    if args.state_dict_path is not None:
        try:
            model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
            tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6 :]
            epoch_start_idx = int(tail[: tail.find('.')]) + 1
        except:
            print('failed loading state_dicts, pls check file path: ', end="")
            print(args.state_dict_path)
            raise RuntimeError('failed loading state_dicts, pls check file path!')

    bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)

    best_val_ndcg, best_val_hr = 0.0, 0.0
    best_test_ndcg, best_test_hr = 0.0, 0.0
    T = 0.0
    t0 = time.time()
    global_step = 0
    print("Start training")
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        model.train()


        if args.inference_only:
            break
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
            seq = seq.to(args.device)
            pos = pos.to(args.device)
            neg = neg.to(args.device)

            optimizer.zero_grad()
            # 修复tensor创建和索引处理
            next_token_type = next_token_type.to(args.device)
            token_type = token_type.to(args.device)
            loss_mask = (next_token_type == 1)
            indices = torch.where(loss_mask)[0] if loss_mask.any() else torch.tensor([], dtype=torch.long, device=args.device)

            # 根据损失类型计算损失
            if args.loss_type == 'bce':
                # 原始BCE损失
                pos_logits, neg_logits = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                    neg_logits.shape, device=args.device
                )
                if len(indices) > 0:
                    loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                    loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                else:
                    loss = torch.tensor(0.0, device=args.device, requires_grad=True)

            elif args.loss_type == 'infonce':
                # InfoNCE损失
                seq_embs = compute_embeddings(model, seq, seq_feat, token_type)
                pos_embs = model.feat2emb(pos, pos_feat, include_user=False)
                neg_embs = model.feat2emb(neg, neg_feat, include_user=False)

                loss, pos_mean_logit, neg_mean_logit = model.compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask)

                # 记录InfoNCE相关指标
                with torch.no_grad():
                    writer.add_scalar("Model/pos_mean_logit", pos_mean_logit, global_step)
                    writer.add_scalar("Model/neg_mean_logits", neg_mean_logit, global_step)

            elif args.loss_type == 'mix':
                # 混合损失
                # BCE损失
                pos_logits, neg_logits = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat
                )
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(
                    neg_logits.shape, device=args.device
                )
                if len(indices) > 0:
                    bce_loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                    bce_loss += bce_criterion(neg_logits[indices], neg_labels[indices])
                else:
                    bce_loss = torch.tensor(0.0, device=args.device, requires_grad=True)

                # InfoNCE损失
                seq_embs = compute_embeddings(model, seq, seq_feat, token_type)
                pos_embs = model.feat2emb(pos, pos_feat, include_user=False)
                neg_embs = model.feat2emb(neg, neg_feat, include_user=False)

                infonce_loss = model.compute_infonce_loss(seq_embs, pos_embs, neg_embs, loss_mask)

                # 组合损失
                loss = bce_loss + args.contrastive_weight * infonce_loss

                # 记录各项损失
                writer.add_scalar('Loss/train_bce', bce_loss.item(), global_step)
                writer.add_scalar('Loss/train_infonce', infonce_loss.item(), global_step)

                # 记录InfoNCE相关指标
                with torch.no_grad():
                    seq_embs_norm = torch.nn.functional.normalize(seq_embs, p=2, dim=-1)
                    pos_embs_norm = torch.nn.functional.normalize(pos_embs, p=2, dim=-1)
                    neg_embs_norm = torch.nn.functional.normalize(neg_embs, p=2, dim=-1)

                    pos_sim = torch.nn.functional.cosine_similarity(seq_embs_norm, pos_embs_norm, dim=-1)
                    neg_sim = torch.matmul(seq_embs_norm.reshape(-1, seq_embs_norm.size(-1)),
                                           neg_embs_norm.reshape(-1, neg_embs_norm.size(-1)).transpose(-1, -2))

                    writer.add_scalar("Model/nce_pos_logits", pos_sim[loss_mask.bool()].mean().item(), global_step)
                    writer.add_scalar("Model/nce_neg_logits", neg_sim.mean().item(), global_step)
                    writer.add_scalar("Model/temp", model.temp, global_step)

            log_json = json.dumps(
                {'global_step': global_step, 'loss': loss.item(), 'epoch': epoch, 'time': time.time()}
            )
            log_file.write(log_json + '\n')
            log_file.flush()
            print(log_json)

            writer.add_scalar('Loss/train', loss.item(), global_step)

            global_step += 1

            for param in model.item_emb.parameters():
                loss += args.l2_emb * torch.norm(param)
            loss.backward()

            if global_step < args.warmup_steps:
                lr_scalar = min(1.0, float(global_step + 1) / args.warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scalar * args.lr
                lr = lr_scalar * args.lr
            else:
                lr = args.lr


            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            writer.add_scalar("Trainer/grad_norm_before_clip", total_norm, global_step)

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
            writer.add_scalar("Trainer/grad_norm_after_clip", total_norm, global_step)

            writer.add_scalar("Trainer/lr", lr, global_step)
            optimizer.step()


        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}")
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model.pt")

    print("Done")
    writer.close()
    log_file.close()
