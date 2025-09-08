import json
import mmap
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from time_features import get_time_feature_extractor


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """
    _global_seq_offsets = None
    _global_mm_emb_dict = None
    _global_item_feat_dict = None
    _global_indexer = None

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()
        self.data_dir = Path(data_dir)

        self._load_data_and_offsets()
        self.seq_offsets = MyDataset._global_seq_offsets

        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id



        self.max_array_len = 10

        if MyDataset._global_mm_emb_dict is None:
            MyDataset._global_mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        self.mm_emb_dict =  MyDataset._global_mm_emb_dict


        if MyDataset._global_item_feat_dict is None:
            with open(Path(data_dir, "item_feat_dict.json"), 'r') as f:
                MyDataset._global_item_feat_dict = json.load(f)
        self.item_feat_dict = MyDataset._global_item_feat_dict

        if MyDataset._global_indexer is None:
            with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
                MyDataset._global_indexer = pickle.load(ff)
        self.itemnum = len(MyDataset._global_indexer['i'])
        self.usernum = len(MyDataset._global_indexer['u'])

        self.indexer_i_rev = {v: k for k, v in MyDataset._global_indexer['i'].items()}
        self.indexer_u_rev = {v: k for k, v in MyDataset._global_indexer['u'].items()}
        self.indexer = MyDataset._global_indexer

        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def __getstate__(self):
        """支持pickle序列化：排除文件句柄"""
        state = self.__dict__.copy()
        # 移除不能被pickle的文件句柄
        if '_local_data_file' in state:
            del state['_local_data_file']
        if hasattr(self, 'data_file'):
            if 'data_file' in state:
                del state['data_file']
        return state

    def __setstate__(self, state):
        """支持pickle反序列化：恢复状态"""
        self.__dict__.update(state)
        # 文件句柄将在使用时通过 _init_file_handle() 重新创建

    def _init_file_handle(self):
        """每个worker初始化自己的文件句柄"""
        if not hasattr(self, '_local_data_file'):
            self._local_data_file = open(self.data_dir / "seq.jsonl", 'rb')

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            MyDataset._global_seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """

        self._init_file_handle()  # 确保文件句柄存在
        self._local_data_file.seek(self.seq_offsets[uid])
        line = self._local_data_file.readline()
        data = json.loads(line)

        return data

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)
        return t

    def _random_neqs(self, l, r, s, num):
        """
        生成一个不在序列s中的不重复的随机整数数组t，长度为num, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列
            num: 需要生成的随机数数量

        Returns:
            t: 不在序列s中的随机整数数组
        """
        t = set()
        while len(t) < num:
            candidate = np.random.randint(l, r)
            if candidate not in s and str(candidate) in self.item_feat_dict and candidate not in t:
                t.add(candidate)
        return np.array(list(t), dtype=np.int64)

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        # 添加时间特征处理
        if user_sequence and len(user_sequence) > 0:
            extractor = get_time_feature_extractor()
            try:
                ts_array, user_sequence = extractor.add_time_features(user_sequence)
            except Exception as e:
                # 如果时间特征处理失败，使用原序列
                pass

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, _ = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)

        seq_feat = np.empty([self.maxlen + 1], dtype=object)
        pos_feat = np.empty([self.maxlen + 1], dtype=object)
        neg_feat = np.empty([self.maxlen + 1], dtype=object)

        nxt = ext_user_sequence[-1]
        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):
            i, feat, type_, act_type = record_tuple
            next_i, next_feat, next_type, next_act_type = nxt
            
            # 如果当前是user特征，且下一个token是item，需要将时间特征从user_feat传输到item_feat
            if type_ == 2 and next_type == 1:  # 当前是user，下一个是item
                extractor = get_time_feature_extractor()
                next_feat = extractor.transfer_context_features(
                    feat, next_feat, ["200", "201", "203", "204", "205"]
                )
            
            feat = self.fill_missing_feat(feat, i)
            next_feat = self.fill_missing_feat(next_feat, next_i)
            seq[idx] = i
            token_type[idx] = type_
            next_token_type[idx] = next_type
            if next_act_type is not None:
                next_action_type[idx] = next_act_type
            seq_feat[idx] = feat
            if next_type == 1 and next_i != 0:
                pos[idx] = next_i
                pos_feat[idx] = next_feat
                neg_id = self._random_neq(1, self.itemnum + 1, ts)
                neg[idx] = neg_id
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        seq_feat = self.all_feat2tensor(seq_feat)
        pos_feat = self.all_feat2tensor(pos_feat)
        neg_feat = self.all_feat2tensor(neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def all_feat2tensor(self, seq_feature):
        all_feat_in_tensor = {}
        all_feat_types = [self.ITEM_SPARSE_FEAT, self.ITEM_ARRAY_FEAT, self.ITEM_CONTINUAL_FEAT, self.USER_SPARSE_FEAT, self.USER_ARRAY_FEAT, self.USER_CONTINUAL_FEAT]

        for feat_dict in all_feat_types:
            for k in feat_dict:
                tensor_feature = self.feat2tensor(seq_feature, k)
                all_feat_in_tensor[k] = tensor_feature

        for k in self.ITEM_EMB_FEAT:
            max_seq_len = self.maxlen+1
            emb_dim = self.ITEM_EMB_FEAT[k]
            seq_emb_data = np.zeros((max_seq_len, emb_dim), dtype=np.float32)
            for i, item in enumerate(seq_feature):
                if k in item:
                    seq_emb_data[i] = item[k]
            all_feat_in_tensor[k] = torch.from_numpy(seq_emb_data)

        return all_feat_in_tensor

    def feat2tensor(self, seq_feature, k):
        """
        Args:
            seq_feature: 序列特征list，每个元素为当前时刻的特征字典，形状为 [batch_size, maxlen]
            k: 特征ID

        Returns:
            batch_data: 特征值的tensor，形状为 [batch_size, maxlen, max_array_len(if array)]
        """

        if k in self.ITEM_ARRAY_FEAT or k in self.USER_ARRAY_FEAT:
            # 如果特征是Array类型，需要先对array进行padding，然后转换为tensor

            max_seq_len = 0

            seq_data = [item[k] for item in seq_feature]
            max_seq_len = self.maxlen+1
            max_array_len = self.max_array_len

            pad_seq_data = np.zeros((max_seq_len, max_array_len), dtype=np.int64)

            for i, item_data in enumerate(seq_data):
                actual_len = min(len(item_data), max_array_len)
                pad_seq_data[i, :actual_len] = item_data[:actual_len]
            pad_seq_data = np.array(pad_seq_data)
            return torch.from_numpy(pad_seq_data)
        else:
            seq_data = np.array([item[k] for item in seq_feature])

            return torch.from_numpy(seq_data)

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}
        feat_statistics = {}
        feat_types = {}
        feat_types['user_sparse'] = ['103', '104', '105', '109', '200', '201', '203', '204', '205']
        feat_types['item_sparse'] = [
            '100',
            '117',
            '111',
            '118',
            '101',
            '102',
            '119',
            '120',
            '114',
            '112',
            '121',
            '115',
            '122',
            '116',
        ]
        feat_types['item_array'] = []
        feat_types['user_array'] = ['106', '107', '108', '110']
        feat_types['item_emb'] = self.mm_emb_ids
        feat_types['user_continual'] = []
        feat_types['item_continual'] = []

        for feat_id in feat_types['user_sparse']:
            feat_default_value[feat_id] = 0
            # 时间特征不在索引器中，需要特殊处理
            if feat_id in ['200', '201', '203', '204', '205']:  # 时间相关稀疏特征
                if feat_id == '200':  # 小时 (0-23)
                    feat_statistics[feat_id] = 24
                elif feat_id == '201':  # 星期 (0-6)  
                    feat_statistics[feat_id] = 7
                elif feat_id == '203':  # 对数间隔离散化 (0-99)
                    feat_statistics[feat_id] = 100  
                elif feat_id == '204':  # 月份 (1-12)
                    feat_statistics[feat_id] = 12
                elif feat_id == '205':  # 时间衰减离散化 (0-99)
                    feat_statistics[feat_id] = 100
            else:
                feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_sparse']:
            feat_default_value[feat_id] = 0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['item_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_array']:
            feat_default_value[feat_id] = [0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])
        for feat_id in feat_types['user_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_continual']:
            feat_default_value[feat_id] = 0
        for feat_id in feat_types['item_emb']:
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        self.USER_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['user_sparse']}
        self.USER_CONTINUAL_FEAT = feat_types['user_continual']
        self.ITEM_SPARSE_FEAT = {k: feat_statistics[k] for k in feat_types['item_sparse']}
        self.ITEM_CONTINUAL_FEAT = feat_types['item_continual']
        self.USER_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['user_array']}
        self.ITEM_ARRAY_FEAT = {k: feat_statistics[k] for k in feat_types['item_array']}
        # EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 1024}
        self.ITEM_EMB_FEAT = {k: EMB_SHAPE_DICT[k] for k in feat_types['item_emb']}  # 记录的是不同多模态特征的维度

        return feat_default_value, feat_types, feat_statistics

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat == None:
            feat = {}
        filled_feat = {}
        for k in feat.keys():
            filled_feat[k] = feat[k]

        all_feat_ids = []
        for feat_type in self.feature_types.values():
            all_feat_ids.extend(feat_type)
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:
            filled_feat[feat_id] = self.feature_default_value[feat_id]
        for feat_id in self.feature_types['item_emb']:
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                if type(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]) == np.ndarray:
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat

    @staticmethod
    def collate_dict_feature(seq_feat):
        merged_seq = {}
        for feat in seq_feat:
            for key in feat:
                if key not in merged_seq:
                    merged_seq[key] = []
                merged_seq[key].append(feat[key])
        for key in merged_seq:
            merged_seq[key] = torch.stack(merged_seq[key])

        return merged_seq

    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        seq_feat = MyDataset.collate_dict_feature(seq_feat)
        pos_feat = MyDataset.collate_dict_feature(pos_feat)
        neg_feat = MyDataset.collate_dict_feature(neg_feat)

        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    @staticmethod
    def worker_init_fn(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # 当前可能是 Subset
        if hasattr(dataset, 'dataset'):  # 如果是 Subset，则访问原始 dataset
            dataset = dataset.dataset
        dataset._init_file_handle()  # 初始化 worker 自己的文件句柄



class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)

    def _init_file_handle(self):
        """每个worker初始化自己的文件句柄"""
        if not hasattr(self, '_local_data_file'):
            self._local_data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')

    def _load_data_and_offsets(self):
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            MyDataset._global_seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if type(feat_value) == list:
                value_list = []
                for v in feat_value:
                    if type(v) == str:
                        value_list.append(0)
                    else:
                        value_list.append(v)
                processed_feat[feat_id] = value_list
            elif type(feat_value) == str:
                processed_feat[feat_id] = 0
            else:
                processed_feat[feat_id] = feat_value
        return processed_feat

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        # 添加时间特征处理 (与训练保持一致)
        if user_sequence and len(user_sequence) > 0:
            extractor = get_time_feature_extractor()
            try:
                ts_array, user_sequence = extractor.add_time_features(user_sequence)
            except Exception as e:
                # 如果时间特征处理失败，使用原序列
                pass

        ext_user_sequence = []
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, _, _ = record_tuple
            if u:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ext_user_sequence.insert(0, (u, user_feat, 2))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1))

        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen

        ts = set()
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for idx_ext, record_tuple in enumerate(reversed(ext_user_sequence)):
            i, feat, type_ = record_tuple
            
            # 如果当前是item，检查前一个是否是user，需要传输时间特征
            if type_ == 1 and idx_ext > 0:
                prev_record = list(reversed(ext_user_sequence[:-1]))[idx_ext - 1]
                if len(prev_record) > 2 and prev_record[2] == 2:  # 前一个是user
                    extractor = get_time_feature_extractor()
                    feat = extractor.transfer_context_features(
                        prev_record[1], feat, ["200", "201", "203", "204", "205"]
                    )
            
            feat = self.fill_missing_feat(feat, i)
            seq[idx] = i
            token_type[idx] = type_
            seq_feat[idx] = feat
            idx -= 1
            if idx == -1:
                break

        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        seq_feat = self.all_feat2tensor(seq_feat)

        return seq, token_type, seq_feat, user_id

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)

    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        seq, token_type, seq_feat, user_id = zip(*batch)
        seq = torch.from_numpy(np.array(seq))
        token_type = torch.from_numpy(np.array(token_type))


        seq_feat = MyTestDataset.collate_dict_feature(seq_feat)

        return seq, token_type, seq_feat, user_id




def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')
    with open(Path(save_path), 'wb') as f:
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)


# def load_mm_emb(mm_path, feat_ids):
#     """
#     加载多模态特征Embedding
#
#     Args:
#         mm_path: 多模态特征Embedding路径
#         feat_ids: 要加载的多模态特征ID列表
#
#     Returns:
#         mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
#     """
#     SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
#     mm_emb_dict = {}
#     for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
#         shape = SHAPE_DICT[feat_id]
#         emb_dict = {}
#         if feat_id != '81':
#             try:
#                 base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
#                 for json_file in base_path.glob('*.json'):
#                     with open(json_file, 'r', encoding='utf-8') as file:
#                         for line in file:
#                             data_dict_origin = json.loads(line.strip())
#                             insert_emb = data_dict_origin['emb']
#                             if isinstance(insert_emb, list):
#                                 insert_emb = np.array(insert_emb, dtype=np.float32)
#                             data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
#                             emb_dict.update(data_dict)
#             except Exception as e:
#                 print(f"transfer error: {e}")
#         if feat_id == '81':
#             with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:
#                 emb_dict = pickle.load(f)
#         mm_emb_dict[feat_id] = emb_dict
#         print(f'Loaded #{feat_id} mm_emb')
#     return mm_emb_dict


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}

    # 文件夹命名有问题
    NAME_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 4096, "86": 3584}
    mm_emb_dict = {}
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):
        shape = SHAPE_DICT[feat_id]

        # 文件夹命名有问题
        name_shape = NAME_SHAPE_DICT[feat_id]

        emb_dict = {}

        base_path = Path(mm_path, f'emb_{feat_id}_{name_shape}')
        for json_file in base_path.glob('part*'):
            with open(json_file, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        data_dict_origin = json.loads(line.strip())
                        insert_emb = data_dict_origin['emb']
                        if isinstance(insert_emb, list):
                            insert_emb = np.array(insert_emb, dtype=np.float32)
                        data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                        emb_dict.update(data_dict)
                    except Exception as e:
                        print('transfer error:{}'.format(e))
                        continue

        mm_emb_dict[feat_id] = emb_dict
        print(f'Loaded #{feat_id} mm_emb')
    return mm_emb_dict