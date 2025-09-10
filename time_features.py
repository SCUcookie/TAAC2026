"""
时间特征处理模块
用于为腾讯算法大赛推荐模型添加时间差特征

特征设计：
- 200: 小时特征 (0-23) 
- 201: 星期特征 (0-6)
- 203: 对数时间间隔特征
- 204: 月份特征 (1-12)  
- 205: 时间衰减特征 (相对序列末尾的衰减)
- 202: 原始时间间隔 (可选)
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimeFeatureExtractor:
    """时间特征提取器"""
    
    def __init__(self, tau: float = 86400.0, use_cache: bool = True):
        """
        初始化时间特征提取器
        
        Args:
            tau: 时间衰减参数，默认86400秒(1天)
            use_cache: 是否使用缓存机制
        """
        self.tau = tau
        self.use_cache = use_cache
        self.cache_dir = None
        if use_cache:
            cache_path = os.environ.get('USER_CACHE_PATH')
            if cache_path:
                self.cache_dir = Path(cache_path) / 'time_features_cache'
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"时间特征缓存目录: {self.cache_dir}")
    
    def _get_cache_key(self, user_sequence: List[Tuple]) -> str:
        """生成用户序列的缓存键"""
        if not self.use_cache or not self.cache_dir:
            return None
            
        # 基于序列内容生成哈希键
        sequence_str = str([(u, i, ts) for u, i, _, _, _, ts in user_sequence])
        return hashlib.md5(sequence_str.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Tuple[np.ndarray, List[Tuple]]]:
        """从缓存加载时间特征"""
        if not cache_key or not self.cache_dir:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"缓存加载失败: {e}")
                return None
        return None
    
    def _save_to_cache(self, cache_key: str, data: Tuple[np.ndarray, List[Tuple]]) -> None:
        """保存时间特征到缓存"""
        if not cache_key or not self.cache_dir:
            return
            
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"缓存保存失败: {e}")
    
    def add_time_features(self, user_sequence: List[Tuple]) -> Tuple[np.ndarray, List[Tuple]]:
        """
        为用户序列添加时间特征
        
        Args:
            user_sequence: 用户序列数据 [(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
            
        Returns:
            ts_array: 时间戳数组
            new_sequence: 添加时间特征后的序列
        """
        if not user_sequence:
            return np.array([]), []
        
        # 尝试从缓存加载
        cache_key = self._get_cache_key(user_sequence)
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            logger.debug("从缓存加载时间特征")
            return cached_result
        
        # 提取时间戳
        ts_array = np.array([r[5] for r in user_sequence], dtype=np.int64)
        
        # 计算时间差特征
        time_features = self._compute_time_features(ts_array)
        
        # 构建新序列
        new_sequence = []
        for idx, record in enumerate(user_sequence):
            u, i, user_feat, item_feat, action_type, ts = record
            
            # 复制用户特征并添加时间特征
            if user_feat is None:
                user_feat = {}
            else:
                user_feat = user_feat.copy()
            
            # 添加时间特征到用户特征中
            user_feat.update(time_features[idx])
            
            new_sequence.append((u, i, user_feat, item_feat, action_type, ts))
        
        result = (ts_array, new_sequence)
        
        # 保存到缓存
        self._save_to_cache(cache_key, result)
        
        return result
    
    def _compute_time_features(self, ts_array: np.ndarray) -> List[Dict[str, Any]]:
        """
        计算时间特征
        
        Args:
            ts_array: 时间戳数组
            
        Returns:
            时间特征列表
        """
        # 1. 时间差和对数间隔特征  
        prev_ts_array = np.roll(ts_array, 1)
        prev_ts_array[0] = ts_array[0]  # 第一个元素时间差为0
        time_gap = ts_array - prev_ts_array
        time_gap[0] = 0
        log_gap = np.log1p(time_gap.astype(np.float64))
        
        # 2. 小时、星期、月份特征 (UTC+8)
        ts_utc8 = ts_array + 8 * 3600
        hours = (ts_utc8 % 86400) // 3600
        weekdays = ((ts_utc8 // 86400 + 3) % 7).astype(np.int32)  # 周一=0
        
        # 使用pandas处理月份转换，提高效率
        try:
            months = pd.to_datetime(ts_utc8, unit='s').month.to_numpy()
        except Exception:
            # 降级方案：手动计算月份
            months = np.array([1] * len(ts_utc8), dtype=np.int32)
            logger.warning("月份特征计算失败，使用默认值1")
        
        # 3. 时间衰减特征
        last_ts = ts_array[-1]
        delta_t = last_ts - ts_array
        delta_scaled = np.log1p(delta_t.astype(np.float64) / self.tau)
        
        # 4. 新增：当前交互与序列首元素的时间差特征 (206)
        first_ts = ts_array[0]  # 序列第一个交互的时间戳
        first_gap = ts_array - first_ts  # 与首元素的时间差
        log_first_gap = np.log1p(first_gap.astype(np.float64))  # 取对数平滑
        first_gap_buckets = self._discretize_continuous_feature(log_first_gap, num_buckets=100)  # 离散化
        
        # 5. 将连续值离散化为稀疏特征索引
        log_gap_buckets = self._discretize_continuous_feature(log_gap, num_buckets=100)
        delta_buckets = self._discretize_continuous_feature(delta_scaled, num_buckets=100)
        
        # 组装特征字典列表
        features_list = []
        for idx in range(len(ts_array)):
            features = {
                "200": int(hours[idx]),              # 小时 (0-23)
                "201": int(weekdays[idx]),           # 星期 (0-6) 
                "203": int(log_gap_buckets[idx]),    # 对数时间间隔离散化 (0-99)
                "204": int(months[idx]),             # 月份 (1-12)
                "205": int(delta_buckets[idx]),      # 时间衰减离散化 (0-99)
                "206": int(first_gap_buckets[idx]),  # 与序列首元素时间差离散化 (0-99)
                # "202": int(time_gap[idx])          # 原始时间间隔 (可选)
            }
            features_list.append(features)
        
        return features_list
    
    def _discretize_continuous_feature(self, values: np.ndarray, num_buckets: int = 100) -> np.ndarray:
        """
        将连续特征值离散化为bucket索引
        
        Args:
            values: 连续特征值数组
            num_buckets: bucket数量
            
        Returns:
            离散化后的索引数组 (0 到 num_buckets-1)
        """
        if len(values) == 0:
            return np.array([], dtype=np.int32)
        
        # 处理所有值相同的情况
        if np.all(values == values[0]):
            return np.zeros(len(values), dtype=np.int32)
        
        # 使用分位数进行离散化
        try:
            # 计算分位数边界
            percentiles = np.linspace(0, 100, num_buckets + 1)
            boundaries = np.percentile(values, percentiles)
            
            # 确保边界值唯一且递增
            boundaries = np.unique(boundaries)
            
            # 使用searchsorted进行bucket分配
            bucket_indices = np.searchsorted(boundaries, values, side='right') - 1
            
            # 确保索引在有效范围内
            bucket_indices = np.clip(bucket_indices, 0, len(boundaries) - 2)
            
            # 重新映射到0-(num_buckets-1)范围
            if len(boundaries) > 1:
                bucket_indices = (bucket_indices * (num_buckets - 1) // (len(boundaries) - 2)).astype(np.int32)
            
            return bucket_indices
            
        except Exception as e:
            logger.warning(f"特征离散化失败: {e}，使用默认值0")
            return np.zeros(len(values), dtype=np.int32)
    
    def transfer_context_features(self, user_feat: Dict[str, Any], item_feat: Dict[str, Any], 
                                cols_to_trans: List[str]) -> Dict[str, Any]:
        """
        将用户特征中的时间特征传输到物品特征中
        
        Args:
            user_feat: 用户特征字典
            item_feat: 物品特征字典  
            cols_to_trans: 需要传输的特征列表
            
        Returns:
            更新后的物品特征字典
        """
        if item_feat is None:
            item_feat = {}
        else:
            item_feat = item_feat.copy()
            
        for col in cols_to_trans:
            if col in user_feat:
                item_feat[col] = user_feat[col]
                
        return item_feat


def process_user_sequences_batch(sequences_batch: List[Tuple[int, List[Tuple]]], 
                               tau: float = 86400.0) -> Dict[int, Tuple[np.ndarray, List[Tuple]]]:
    """
    批量处理用户序列的时间特征提取
    
    Args:
        sequences_batch: [(user_id, user_sequence)] 列表
        tau: 时间衰减参数
        
    Returns:
        {user_id: (ts_array, new_sequence)} 字典
    """
    extractor = TimeFeatureExtractor(tau=tau, use_cache=False)  # 进程内不使用缓存
    results = {}
    
    for user_id, user_sequence in sequences_batch:
        try:
            ts_array, new_sequence = extractor.add_time_features(user_sequence)
            results[user_id] = (ts_array, new_sequence)
        except Exception as e:
            logger.error(f"用户 {user_id} 时间特征处理失败: {e}")
            # 返回原序列作为降级方案
            results[user_id] = (np.array([r[5] for r in user_sequence]), user_sequence)
    
    return results


class BatchTimeFeatureProcessor:
    """批量时间特征处理器"""
    
    def __init__(self, tau: float = 86400.0, max_workers: int = None, batch_size: int = 1000):
        """
        初始化批量处理器
        
        Args:
            tau: 时间衰减参数
            max_workers: 最大工作进程数
            batch_size: 批处理大小
        """
        self.tau = tau
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.batch_size = batch_size
        self.extractor = TimeFeatureExtractor(tau=tau, use_cache=True)
        
    def process_sequences(self, user_sequences: Dict[int, List[Tuple]]) -> Dict[int, Tuple[np.ndarray, List[Tuple]]]:
        """
        并行处理用户序列
        
        Args:
            user_sequences: {user_id: user_sequence} 字典
            
        Returns:
            {user_id: (ts_array, new_sequence)} 字典
        """
        total_users = len(user_sequences)
        logger.info(f"开始处理 {total_users} 个用户序列，使用 {self.max_workers} 个进程")
        
        # 将用户序列分批
        user_items = list(user_sequences.items())
        batches = [user_items[i:i+self.batch_size] for i in range(0, len(user_items), self.batch_size)]
        
        results = {}
        processed = 0
        
        # 并行处理
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(process_user_sequences_batch, batch, self.tau): batch_idx
                for batch_idx, batch in enumerate(batches)
            }
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    results.update(batch_results)
                    processed += len(batch_results)
                    
                    if processed % 10000 == 0 or processed == total_users:
                        logger.info(f"已处理 {processed}/{total_users} 用户序列 ({processed/total_users*100:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"批次 {batch_idx} 处理失败: {e}")
        
        logger.info("时间特征处理完成")
        return results


# 全局缓存实例
_global_extractor = None

def get_time_feature_extractor(tau: float = 86400.0) -> TimeFeatureExtractor:
    """获取全局时间特征提取器实例"""
    global _global_extractor
    if _global_extractor is None:
        _global_extractor = TimeFeatureExtractor(tau=tau)
    return _global_extractor