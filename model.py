import math
from pathlib import Path
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset import save_emb





def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x

class FlashMultiHeadAttention(torch.nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(FlashMultiHeadAttention, self).__init__()

        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate
        self._eps = 1e-6
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"


        self.q_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.k_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.v_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.u_linear = torch.nn.Linear(hidden_units, hidden_units)
        self.out_linear = torch.nn.Linear(hidden_units, hidden_units)

    def forward(self, query, key, value, attn_mask=None):
        batch_size, seq_len, _ = query.size()

        # 计算Q, K, V
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        U = self.u_linear(value)



        # reshape为multi-head格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)




        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)  # 标准缩放：除以head_dim的平方根
        scores = F.silu(scores)  # 激活函数移到缩放后
        scores = scores * attn_mask.unsqueeze(1)

        attn_weights = F.dropout(scores, p=self.dropout_rate, training=self.training)
        attn_output = torch.matmul(attn_weights, V)

        # reshape回原来的格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)

        attn_output = F.layer_norm(
            attn_output, normalized_shape=[self.hidden_units], eps=self._eps
        )
        # 乘以u矩阵，开启门控网络

        attn_output = U * attn_output
        output = self.out_linear(attn_output)

        return output, None


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        return outputs


class BaselineModel(torch.nn.Module):
    """
    Args:
        user_num: 用户数量
        item_num: 物品数量
        feat_statistics: 特征统计信息，key为特征ID，value为特征数量
        feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        args: 全局参数

    Attributes:
        user_num: 用户数量
        item_num: 物品数量
        dev: 设备
        norm_first: 是否先归一化
        maxlen: 序列最大长度
        item_emb: Item Embedding Table
        user_emb: User Embedding Table
        sparse_emb: 稀疏特征Embedding Table
        emb_transform: 多模态特征的线性变换
        userdnn: 用户特征拼接后经过的全连接层
        itemdnn: 物品特征拼接后经过的全连接层
    """

    def __init__(self, user_num, item_num, feat_statistics, feat_types, args):  #
        super(BaselineModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.norm_first = args.norm_first
        self.maxlen = args.maxlen
        self.embedding_dim = args.hidden_units

        # InfoNCE 相关参数
        self.temp = max(getattr(args, 'temp', 0.07), 0.01)  # 确保温度系数大于0
        self.loss_type = getattr(args, 'loss_type', 'bce')
        self.neg_topk = max(getattr(args, 'neg_topk', 0), 0)  # 确保非负
        self.norm_output = getattr(args, 'norm_output', False)
        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch

        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = torch.nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)

        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)
        self.sparse_emb = torch.nn.ModuleDict()
        self.emb_transform = torch.nn.ModuleDict()

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self._init_feat_info(feat_statistics, feat_types)


        userdim = args.hidden_units * (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        itemdim = (
                args.hidden_units * (len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT) + len(self.ITEM_TEMP_TIME_FEAT) - 2)
                + len(self.ITEM_CONTINUAL_FEAT)
                + args.hidden_units * len(self.ITEM_EMB_FEAT)
        )


        user_feature_num = (len(self.USER_SPARSE_FEAT) + 1 + len(self.USER_ARRAY_FEAT)) + len(
            self.USER_CONTINUAL_FEAT
        )
        item_feature_num = len(self.ITEM_SPARSE_FEAT) + 1 + len(self.ITEM_ARRAY_FEAT) + len(self.ITEM_TEMP_TIME_FEAT) - 2 + len(self.ITEM_CONTINUAL_FEAT) + len(self.ITEM_EMB_FEAT)


        self.userdnn = torch.nn.Linear(userdim, args.hidden_units)
        self.itemdnn = torch.nn.Linear(itemdim, args.hidden_units)

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = FlashMultiHeadAttention(
                args.hidden_units, args.num_heads, args.dropout_rate
            )  # 优化：用FlashAttention替代标准Attention
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

        for k in self.USER_SPARSE_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_SPARSE_FEAT:
            if(k=='111'):
                continue
            if(k=='115'):
                continue
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_SPARSE_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.USER_ARRAY_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.USER_ARRAY_FEAT[k] + 1, args.hidden_units, padding_idx=0)
        for k in self.ITEM_EMB_FEAT:
            self.emb_transform[k] = torch.nn.Linear(self.ITEM_EMB_FEAT[k], args.hidden_units)
        
        # 新增：物品侧临时时间特征的embedding声明
        for k in self.ITEM_TEMP_TIME_FEAT:
            self.sparse_emb[k] = torch.nn.Embedding(self.ITEM_TEMP_TIME_FEAT[k] + 1, args.hidden_units, padding_idx=0)

        self.reset_embeddings()

    def reset_embeddings(self) -> None:
        for name, params in self.named_parameters():
            if "item_emb" in name:
                self.item_emb.weight.data.zero_()
                self.item_emb.weight.data[0] = 0
                print(
                    f"Initialize {name} as zero normal: {params.data.size()} params"
                )
                continue
            if "user_emb" in name:
                self.user_emb.weight.data.zero_()
                self.user_emb.weight.data[0] = 0
                print(
                    f"Initialize {name} as zero normal: {params.data.size()} params"
                )
                continue
            if "pos_emb" in name:
                truncated_normal(
                    self.pos_emb.weight.data,
                    mean=0.0,
                    std=math.sqrt(1.0 / self.embedding_dim),
                )
                print(
                    f"Initialize {name} as math sqrt normal: {params.data.size()} params"
                )
                continue
            if "_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def _init_feat_info(self, feat_statistics, feat_types):
        """
        将特征统计信息（特征数量）按特征类型分组产生不同的字典，方便声明稀疏特征的Embedding Table

        Args:
            feat_statistics: 特征统计信息，key为特征ID，value为特征数量
            feat_types: 各个特征的特征类型，key为特征类型名称，value为包含的特征ID列表，包括user和item的sparse, array, emb, continual类型
        """
        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        # EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 1024}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度
        
        # 新增：物品侧临时时间特征 - 处理从用户侧传递过来的时间上下文特征
        self.ITEM_TEMP_TIME_FEAT = {k: feat_statistics[k] for k in feat_types.get('item_temp_time', [])}

    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """
        batch_size = len(seq_feature)

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor
            max_array_len = 0
            max_seq_len = 0

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                max_seq_len = max(max_seq_len, len(seq_data))
                max_array_len = max(max_array_len, max(len(item_data) for item_data in seq_data))

            batch_data = np.zeros((batch_size, max_seq_len, max_array_len), dtype=np.int64)
            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                for j, item_data in enumerate(seq_data):
                    actual_len = min(len(item_data), max_array_len)
                    batch_data[i, j, :actual_len] = item_data[:actual_len]

            return torch.from_numpy(batch_data).to(self.dev)
        else:
            # 如果特征是Sparse类型，直接转换为tensor
            max_seq_len = max(len(seq_feature[i]) for i in range(batch_size))
            batch_data = np.zeros((batch_size, max_seq_len), dtype=np.int64)

            for i in range(batch_size):
                seq_data = [item[k] for item in seq_feature[i]]
                batch_data[i] = seq_data

            return torch.from_numpy(batch_data).to(self.dev)

    def feat2emb(self, seq, feature_array, mask=None, include_user=False):
        """
        Args:
            seq: 序列ID
            feature_array: 特征list，每个元素为当前时刻的特征字典
            mask: 掩码，1表示item，2表示user
            include_user: 是否处理用户特征，在两种情况下不打开：1) 训练时在转换正负样本的特征时（因为正负样本都是item）;2) 生成候选库item embedding时。

        Returns:
            seqs_emb: 序列特征的Embedding
        """
        seq = seq.to(self.dev)
        # pre-compute embedding
        if include_user:
            # 确保mask在正确的设备上并且维度匹配
            mask = mask.to(self.dev)
            # 检查mask的值是否在合理范围内
            mask = torch.clamp(mask, 0, 2)  # 确保mask值在0-2范围内

            user_mask = (mask == 2)
            item_mask = (mask == 1)

            # 安全的掩码操作
            user_seq = torch.where(user_mask, seq, torch.zeros_like(seq))
            item_seq = torch.where(item_mask, seq, torch.zeros_like(seq))

            user_embedding = self.user_emb(user_seq)
            item_embedding = self.item_emb(item_seq)
            item_feat_list = [item_embedding]
            user_feat_list = [user_embedding]
        else:
            item_embedding = self.item_emb(seq)
            item_feat_list = [item_embedding]

        # batch-process all feature types
        all_feat_types = [
            (self.ITEM_SPARSE_FEAT, 'item_sparse', item_feat_list),
            (self.ITEM_ARRAY_FEAT, 'item_array', item_feat_list),
            (self.ITEM_CONTINUAL_FEAT, 'item_continual', item_feat_list),
            # 新增：物品侧临时时间特征处理
            (self.ITEM_TEMP_TIME_FEAT, 'item_temp_time', item_feat_list),
        ]

        if include_user:
            all_feat_types.extend(
                [
                    (self.USER_SPARSE_FEAT, 'user_sparse', user_feat_list),
                    (self.USER_ARRAY_FEAT, 'user_array', user_feat_list),
                    (self.USER_CONTINUAL_FEAT, 'user_continual', user_feat_list),
                ]
            )

        # batch-process each feature type
        for feat_dict, feat_type, feat_list in all_feat_types:
            if not feat_dict:
                continue

            for k in feat_dict:
                if(k=='111'):
                    continue
                if(k=='115'):
                    continue
                tensor_feature = feature_array[k]
                tensor_feature = tensor_feature.to(self.dev)
                if feat_type.endswith('sparse'):
                    feat_list.append(self.sparse_emb[k](tensor_feature))
                elif feat_type.endswith('array'):
                    feat_list.append(self.sparse_emb[k](tensor_feature).sum(2))
                elif feat_type.endswith('continual'):
                    feat_list.append(tensor_feature.unsqueeze(2))
                elif feat_type == 'item_temp_time':
                    # 物品侧临时时间特征处理：与稀疏特征相同的处理方式
                    time_emb = self.sparse_emb[k](tensor_feature)
                    # 直接输出时间权重，不使用可学习参数
                    feat_list.append(time_emb)

        for k in self.ITEM_EMB_FEAT:
            batch_emb_data = feature_array[k]
            tensor_feature = batch_emb_data.to(self.dev)
            item_feat_list.append(self.emb_transform[k](tensor_feature))

        # merge features
        b, s = seq.size(0), seq.size(1)
        item_feats = torch.stack(item_feat_list, dim=2)
        all_item_emb = item_feats.view(b, s, -1)
        all_item_emb = self.itemdnn(all_item_emb)
        if include_user:
            user_feats = torch.stack(user_feat_list, dim=2)
            all_user_emb = user_feats.view(b, s, -1)
            all_user_emb = self.userdnn(all_user_emb)
            seqs_emb = all_item_emb + all_user_emb
        else:
            seqs_emb = all_item_emb

        return seqs_emb

    def log2feats(self, log_seqs, mask, seq_feature):
        """
        Args:
            log_seqs: 序列ID
            mask: token类型掩码，1表示item token，2表示user token
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典

        Returns:
            seqs_emb: 序列的Embedding，形状为 [batch_size, maxlen, hidden_units]
        """
        batch_size = log_seqs.shape[0]
        maxlen = log_seqs.shape[1]
        seqs = self.feat2emb(log_seqs, seq_feature, mask=mask, include_user=True)
        seqs *= self.item_emb.embedding_dim**0.5

        seqs = self.emb_dropout(seqs)

        maxlen = seqs.shape[1]
        ones_matrix = torch.ones((maxlen, maxlen), dtype=torch.bool, device=self.dev)
        attention_mask_tril = torch.tril(ones_matrix)
        attention_mask_pad = (mask != 0).to(self.dev)
        attention_mask = attention_mask_tril.unsqueeze(0) & attention_mask_pad.unsqueeze(1)

        for i in range(len(self.attention_layers)):
            if self.norm_first:
                x = self.attention_layernorms[i](seqs)
                mha_outputs, _ = self.attention_layers[i](x, x, x, attn_mask=attention_mask)
                seqs = seqs + mha_outputs
                seqs = seqs + self.forward_layers[i](self.forward_layernorms[i](seqs))
            else:
                mha_outputs, _ = self.attention_layers[i](seqs, seqs, seqs, attn_mask=attention_mask)
                seqs = self.attention_layernorms[i](seqs + mha_outputs)
                seqs = self.forward_layernorms[i](seqs + self.forward_layers[i](seqs))

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(
            self, user_item, pos_seqs, neg_seqs, mask, next_mask, next_action_type, seq_feature, pos_feature, neg_feature
    ):
        """
        训练时调用，计算正负样本的logits

        Args:
            user_item: 用户序列ID
            pos_seqs: 正样本序列ID
            neg_seqs: 负样本序列ID
            mask: token类型掩码，1表示item token，2表示user token
            next_mask: 下一个token类型掩码，1表示item token，2表示user token
            next_action_type: 下一个token动作类型，0表示曝光，1表示点击
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            pos_feature: 正样本特征list，每个元素为当前时刻的特征字典
            neg_feature: 负样本特征list，每个元素为当前时刻的特征字典

        Returns:
            pos_logits: 正样本logits，形状为 [batch_size, maxlen]
            neg_logits: 负样本logits，形状为 [batch_size, maxlen]
        """
        log_feats = self.log2feats(user_item, mask, seq_feature)
        loss_mask = (next_mask == 1).to(self.dev)

        pos_embs = self.feat2emb(pos_seqs, pos_feature, include_user=False)
        neg_embs = self.feat2emb(neg_seqs, neg_feature, include_user=False)

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        pos_logits = pos_logits * loss_mask
        neg_logits = neg_logits * loss_mask

        return pos_logits, neg_logits

    def predict(self, log_seqs, seq_feature, mask):
        """
        计算用户序列的表征
        Args:
            log_seqs: 用户序列ID
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典
            mask: token类型掩码，1表示item token，2表示user token
        Returns:
            final_feat: 用户序列的表征，形状为 [batch_size, hidden_units]
        """
        log_feats = self.log2feats(log_seqs, mask, seq_feature)

        final_feat = log_feats[:, -1, :]

        # 如果启用了输出归一化，对最终特征进行L2归一化
        if self.norm_output:
            final_feat = F.normalize(final_feat, p=2, dim=-1)

        return final_feat

    def save_item_emb(self, item_ids, retrieval_ids, feat_dict, save_path, batch_size=1024):
        """
        生成候选库item embedding，用于检索

        Args:
            item_ids: 候选item ID（re-id形式）
            retrieval_ids: 候选item ID（检索ID，从0开始编号，检索脚本使用）
            feat_dict: 训练集所有item特征字典，key为特征ID，value为特征值
            save_path: 保存路径
            batch_size: 批次大小
        """
        all_embs = []

        for start_idx in tqdm(range(0, len(item_ids), batch_size), desc="Saving item embeddings"):
            end_idx = min(start_idx + batch_size, len(item_ids))

            item_seq = torch.tensor(item_ids[start_idx:end_idx], device=self.dev).unsqueeze(0)
            batch_feat_list = []
            for i in range(start_idx, end_idx):
                batch_feat_list.append(feat_dict[i])

            # 关键修复：将 list of dicts 转换为 dict of tensors
            feature_array = {}
            
            # 获取所有可能的特征ID
            all_feat_ids = set()
            for feat in batch_feat_list:
                all_feat_ids.update(feat.keys())
            
            # 确保所有需要的特征ID都存在于feature_array中
            for feat_id in self.ITEM_TEMP_TIME_FEAT.keys():
                if feat_id not in all_feat_ids:
                    all_feat_ids.add(feat_id)
            
            # 为每个特征ID创建tensor
            for feat_id in all_feat_ids:
                if feat_id in self.ITEM_SPARSE_FEAT or feat_id in self.USER_SPARSE_FEAT:
                    # 稀疏特征处理
                    feat_values = []
                    for feat in batch_feat_list:
                        if feat_id in feat:
                            feat_values.append(feat[feat_id])
                        else:
                            feat_values.append(0)  # 默认值
                    feature_array[feat_id] = torch.tensor(feat_values, device=self.dev).unsqueeze(0)
                    
                elif feat_id in self.ITEM_ARRAY_FEAT or feat_id in self.USER_ARRAY_FEAT:
                    # 数组特征处理
                    feat_arrays = []
                    max_len = 0
                    for feat in batch_feat_list:
                        if feat_id in feat:
                            feat_arrays.append(feat[feat_id])
                            max_len = max(max_len, len(feat[feat_id]))
                        else:
                            feat_arrays.append([])
                    
                    # Padding到相同长度
                    padded_arrays = []
                    for arr in feat_arrays:
                        if len(arr) < max_len:
                            padded_arr = arr + [0] * (max_len - len(arr))
                        else:
                            padded_arr = arr[:max_len]
                        padded_arrays.append(padded_arr)
                    
                    feature_array[feat_id] = torch.tensor(padded_arrays, device=self.dev).unsqueeze(0)
                    
                elif feat_id in self.ITEM_CONTINUAL_FEAT or feat_id in self.USER_CONTINUAL_FEAT:
                    # 连续特征处理
                    feat_values = []
                    for feat in batch_feat_list:
                        if feat_id in feat:
                            feat_values.append(feat[feat_id])
                        else:
                            feat_values.append(0.0)  # 默认值
                    feature_array[feat_id] = torch.tensor(feat_values, device=self.dev, dtype=torch.float32).unsqueeze(0)
                    
                elif feat_id in self.ITEM_EMB_FEAT:
                    # 嵌入特征处理
                    feat_embeddings = []
                    for feat in batch_feat_list:
                        if feat_id in feat:
                            feat_embeddings.append(feat[feat_id])
                        else:
                            # 使用零填充，维度从 ITEM_EMB_FEAT 获取
                            emb_dim = self.ITEM_EMB_FEAT[feat_id]
                            feat_embeddings.append(np.zeros(emb_dim, dtype=np.float32))
                    
                    feature_array[feat_id] = torch.tensor(np.array(feat_embeddings), device=self.dev).unsqueeze(0)
                
                elif feat_id in self.ITEM_TEMP_TIME_FEAT:
                    # 物品侧临时时间特征处理（与稀疏特征逻辑一致）
                    feat_values = []
                    for feat in batch_feat_list:
                        if feat_id in feat:
                            feat_values.append(feat[feat_id])
                        else:
                            feat_values.append(0)  # 默认值
                    feature_array[feat_id] = torch.tensor(feat_values, device=self.dev).unsqueeze(0)

            batch_emb = self.feat2emb(item_seq, feature_array, include_user=False).squeeze(0)

            all_embs.append(batch_emb.detach().cpu().numpy().astype(np.float32))

        # 合并所有批次的结果并保存
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        final_embs = np.concatenate(all_embs, axis=0)

        # 如果启用了输出归一化，对embedding进行L2归一化
        if self.norm_output:
            final_embs = final_embs / (np.linalg.norm(final_embs, axis=1, keepdims=True) + 1e-8)

        save_emb(final_embs, Path(save_path, 'embedding.fbin'))
        save_emb(final_ids, Path(save_path, 'id.u64bin'))

    def compute_infonce_loss(self, seq_embs, pos_embs, neg_embs, loss_mask):
        """
        计算InfoNCE对比学习损失

        Args:
            seq_embs: 序列embedding [batch_size, maxlen, hidden_units]
            pos_embs: 正样本embedding [batch_size, maxlen, hidden_units]
            neg_embs: 负样本embedding [batch_size, maxlen, hidden_units]
            loss_mask: 损失掩码 [batch_size, maxlen]

        Returns:
            loss: InfoNCE损失值
        """
        # 对embedding进行L2归一化
        seq_embs = F.normalize(seq_embs, p=2, dim=-1)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)

        # 计算正样本的余弦相似度
        pos_logits = F.cosine_similarity(seq_embs, pos_embs, dim=-1).unsqueeze(-1)

        # 计算负样本的余弦相似度 - 使用批内负样本
        batch_size, maxlen, hidden_size = seq_embs.shape

        # 只使用有效位置的embedding来减少计算量
        valid_seq_embs = seq_embs[loss_mask]  # [num_valid, hidden_size]
        valid_pos_embs = pos_embs[loss_mask]  # [num_valid, hidden_size]
        valid_neg_embs = neg_embs[loss_mask]  # [num_valid, hidden_size]

        if valid_seq_embs.size(0) == 0:
            return torch.tensor(0.0, device=seq_embs.device, requires_grad=True)

        # 计算正样本logits（只保留有效位置）
        valid_pos_logits = F.cosine_similarity(valid_seq_embs, valid_pos_embs, dim=-1).unsqueeze(-1)  # [num_valid, 1]

        # 计算负样本相似度（只在有效位置之间）
        neg_logits = torch.matmul(valid_seq_embs, valid_neg_embs.transpose(-1, -2))  # [num_valid, num_valid]
        # 初始化负样本张量（避免未定义错误）
        hard_neg_logits = torch.tensor([], device=neg_logits.device)
        easy_neg_logits = torch.tensor([], device=neg_logits.device)

        # 1. 处理难负样本 (top-k)
        if self.neg_topk > 0 and neg_logits.size(-1) > self.neg_topk:
            hard_neg_logits, _ = torch.topk(neg_logits, self.neg_topk, dim=-1)
        # 当不满足条件时，使用空张量或调整策略
        elif self.neg_topk > 0:
            # 若负样本数量不足，全部保留
            hard_neg_logits = neg_logits

        # 2. 处理简单负样本 (2*topk随机)
        num_valid = valid_seq_embs.size(0)
        # 确保需要的负样本数量有效
        required_easy = 2 * self.neg_topk if self.neg_topk > 0 else 0
        if required_easy > 0 and neg_logits.size(-1) >= 1:
            # 生成足够的随机索引（处理样本数量不足的情况）
            if num_valid >= required_easy:
                rand_indices = torch.randperm(num_valid, device=neg_logits.device)[:required_easy]
            else:
                # 样本不足时重复采样
                rand_indices = torch.randint(0, num_valid, (required_easy,), device=neg_logits.device)

            # 扩展索引形状并提取负样本
            expanded_indices = rand_indices.unsqueeze(0).expand(neg_logits.size(0), -1)
            easy_neg_logits = torch.gather(neg_logits, 1, expanded_indices)

        # 3. 拼接负样本（处理空张量情况）
        combined_neg_logits = torch.tensor([], device=neg_logits.device)
        if hard_neg_logits.numel() > 0 and easy_neg_logits.numel() > 0:
            # 确保两个张量第一维一致
            if hard_neg_logits.size(0) == easy_neg_logits.size(0):
                combined_neg_logits = torch.cat([hard_neg_logits, easy_neg_logits], dim=-1)
            else:
                # 维度不匹配时取最小长度
                min_len = min(hard_neg_logits.size(0), easy_neg_logits.size(0))
                combined_neg_logits = torch.cat([
                    hard_neg_logits[:min_len],
                    easy_neg_logits[:min_len]
                ], dim=-1)
        elif hard_neg_logits.numel() > 0:
            combined_neg_logits = hard_neg_logits
        elif easy_neg_logits.numel() > 0:
            combined_neg_logits = easy_neg_logits

        # 4. 拼接正负样本logits（确保有负样本）
        if combined_neg_logits.numel() == 0:
            # 无负样本时，可返回0损失或调整策略
            logits = valid_pos_logits
        else:
            logits = torch.cat([valid_pos_logits, combined_neg_logits], dim=-1)
        # 应用温度系数
        logits = logits / self.temp

        # 创建标签，正样本位置为0
        labels = torch.zeros(logits.size(0), device=logits.device, dtype=torch.int64)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss, valid_pos_logits.mean().item(), neg_logits.mean().item()