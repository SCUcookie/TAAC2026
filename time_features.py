"""
时间特征工程模块
基于腾讯算法大赛数据分析结果设计的时间特征计算

数据洞察：
- 平均点击间隔：4.7分钟，中位数：2.7分钟
- 65.4%的行为集中在1-10分钟间隔
- 数据时间跨度：440.8小时
- 明显的时间周期性：工作日vs周末，一天中不同时段
"""

import numpy as np
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import json
import pickle


class TimeFeatureProcessor:
    """时间特征处理器"""
    
    def __init__(self, cache_dir=None, enable_cache=True):
        """
        初始化时间特征处理器
        
        Args:
            cache_dir: 缓存目录，默认使用USER_CACHE_PATH
            enable_cache: 是否启用缓存
        """
        if cache_dir is None:
            cache_dir = os.environ.get('USER_CACHE_PATH', './cache')
        
        self.cache_dir = Path(cache_dir)
        self.time_features_cache_dir = self.cache_dir / 'time_features'
        self.time_features_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_cache = enable_cache
        
        # 时间特征配置
        self.time_feature_config = {
            # 时间间隔分桶边界（分钟）
            'interval_buckets': [0, 1, 2, 5, 10, 30, 60, 180, 1440, float('inf')],
            
            # 时间衰减参数
            'decay_factor': 0.1,  # 基于65.4%集中在1-10分钟的发现
            
            # 时间周期特征
            'time_periods': {
                'morning_peak': (6, 12),      # 最活跃 34.5%
                'afternoon': (12, 18),        # 次活跃 28.2% 
                'evening': (18, 24),          # 晚高峰 28.0%
                'dawn': (0, 6)               # 低活跃 9.2%
            }
        }
    
    def compute_time_interval_features(self, timestamps):
        """
        计算时间间隔相关特征
        
        Args:
            timestamps: 时间戳序列 (Unix timestamp)
            
        Returns:
            dict: 时间间隔特征字典
        """
        if len(timestamps) < 2:
            return {
                'time_interval_minutes': 0.0,
                'time_interval_bucket': 0,
                'time_decay_weight': 1.0,
                'is_immediate_click': 0,
                'is_short_interval': 0,
                'is_medium_interval': 0,
                'is_long_interval': 0
            }
        
        # 计算时间间隔（分钟）
        time_diff_seconds = timestamps[-1] - timestamps[-2]
        time_interval_minutes = time_diff_seconds / 60.0
        
        # 时间间隔分桶
        interval_bucket = self._get_interval_bucket(time_interval_minutes)
        
        # 时间衰减权重 (基于指数衰减)
        time_decay_weight = np.exp(-self.time_feature_config['decay_factor'] * time_interval_minutes)
        
        # 间隔类型标识（基于数据分析发现）
        is_immediate_click = 1 if time_interval_minutes <= 1 else 0      # 23.5%
        is_short_interval = 1 if 1 < time_interval_minutes <= 10 else 0  # 65.4%
        is_medium_interval = 1 if 10 < time_interval_minutes <= 60 else 0  # 11.0%
        is_long_interval = 1 if time_interval_minutes > 60 else 0         # 0.1%
        
        return {
            'time_interval_minutes': float(time_interval_minutes),
            'time_interval_bucket': interval_bucket,
            'time_decay_weight': float(time_decay_weight),
            'is_immediate_click': is_immediate_click,
            'is_short_interval': is_short_interval,
            'is_medium_interval': is_medium_interval,
            'is_long_interval': is_long_interval
        }
    
    def compute_temporal_cycle_features(self, timestamp):
        """
        计算时间周期特征
        
        Args:
            timestamp: Unix时间戳
            
        Returns:
            dict: 时间周期特征字典
        """
        from datetime import datetime
        
        dt = datetime.fromtimestamp(timestamp)
        
        # 基础时间特征
        hour = dt.hour
        day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
        
        # 周期性编码（正弦余弦编码避免边界问题）
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        # 时间段特征（基于数据分析发现的活跃时段）
        time_period = self._get_time_period(hour)
        
        # 工作日/周末
        is_weekend = 1 if day_of_week >= 5 else 0  # 周末点击占28.9%
        is_workday = 1 - is_weekend
        
        # 活跃时段标识
        is_morning_peak = 1 if 6 <= hour < 12 else 0     # 34.5%
        is_afternoon = 1 if 12 <= hour < 18 else 0       # 28.2%
        is_evening = 1 if 18 <= hour < 24 else 0         # 28.0%
        is_dawn = 1 if 0 <= hour < 6 else 0              # 9.2%
        
        return {
            'hour': hour,
            'day_of_week': day_of_week,
            'hour_sin': float(hour_sin),
            'hour_cos': float(hour_cos),
            'day_sin': float(day_sin),
            'day_cos': float(day_cos),
            'time_period': time_period,
            'is_weekend': is_weekend,
            'is_workday': is_workday,
            'is_morning_peak': is_morning_peak,
            'is_afternoon': is_afternoon,
            'is_evening': is_evening,
            'is_dawn': is_dawn
        }
    
    def compute_sequence_temporal_features(self, timestamps, window_size=10):
        """
        计算序列级时间特征
        
        Args:
            timestamps: 时间戳序列
            window_size: 滑动窗口大小
            
        Returns:
            dict: 序列时间特征字典
        """
        if len(timestamps) < 2:
            return {
                'avg_interval_minutes': 0.0,
                'interval_variance': 0.0,
                'interval_trend': 0.0,
                'session_duration_minutes': 0.0,
                'click_frequency': 0.0,
                'temporal_consistency': 0.0
            }
        
        # 计算时间间隔序列（分钟）
        intervals = []
        for i in range(1, len(timestamps)):
            interval_minutes = (timestamps[i] - timestamps[i-1]) / 60.0
            intervals.append(interval_minutes)
        
        intervals = np.array(intervals)
        
        # 统计特征
        avg_interval = np.mean(intervals)
        interval_variance = np.var(intervals)
        
        # 时间间隔趋势（线性回归斜率）
        if len(intervals) > 2:
            x = np.arange(len(intervals))
            trend_coef = np.polyfit(x, intervals, 1)[0]
        else:
            trend_coef = 0.0
        
        # 会话持续时间（分钟）
        session_duration = (timestamps[-1] - timestamps[0]) / 60.0
        
        # 点击频率（次/分钟）
        click_frequency = len(timestamps) / max(session_duration, 1.0)
        
        # 时间一致性（间隔的变异系数的倒数）
        temporal_consistency = avg_interval / max(np.sqrt(interval_variance), 1e-6)
        
        return {
            'avg_interval_minutes': float(avg_interval),
            'interval_variance': float(interval_variance),
            'interval_trend': float(trend_coef),
            'session_duration_minutes': float(session_duration),
            'click_frequency': float(click_frequency),
            'temporal_consistency': float(temporal_consistency)
        }
    
    def _get_interval_bucket(self, interval_minutes):
        """获取时间间隔桶索引"""
        buckets = self.time_feature_config['interval_buckets']
        for i, boundary in enumerate(buckets[1:]):
            if interval_minutes <= boundary:
                return i
        return len(buckets) - 2
    
    def _get_time_period(self, hour):
        """获取时间段标识"""
        periods = self.time_feature_config['time_periods']
        for period_name, (start, end) in periods.items():
            if start <= hour < end:
                return hash(period_name) % 100  # 简单哈希映射到数值
        return 0
    
    def process_user_sequence_time_features(self, user_sequence, use_cache=True):
        """
        处理单个用户序列的时间特征
        
        Args:
            user_sequence: 用户行为序列 [(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
            use_cache: 是否使用缓存
            
        Returns:
            list: 增强后的用户序列，时间特征添加到user_feat和item_feat中
        """
        if not user_sequence:
            return user_sequence
        
        # 提取时间戳序列
        timestamps = [record[5] for record in user_sequence]
        
        # 计算序列级时间特征
        seq_temporal_features = self.compute_sequence_temporal_features(timestamps)
        
        enhanced_sequence = []
        
        for i, record in enumerate(user_sequence):
            user_id, item_id, user_feat, item_feat, action_type, timestamp = record
            
            # 复制原始特征
            enhanced_user_feat = user_feat.copy() if user_feat else {}
            enhanced_item_feat = item_feat.copy() if item_feat else {}
            
            # 计算当前位置的时间特征
            current_timestamps = timestamps[:i+1]
            
            # 时间间隔特征
            interval_features = self.compute_time_interval_features(current_timestamps)
            
            # 时间周期特征
            cycle_features = self.compute_temporal_cycle_features(timestamp)
            
            # 添加时间特征到用户特征中
            time_features_for_user = {
                'seq_avg_interval': seq_temporal_features['avg_interval_minutes'],
                'seq_interval_variance': seq_temporal_features['interval_variance'],
                'seq_click_frequency': seq_temporal_features['click_frequency'],
                'seq_temporal_consistency': seq_temporal_features['temporal_consistency']
            }
            
            # 添加时间特征到物品特征中（用于序列建模）
            time_features_for_item = {
                **interval_features,
                **cycle_features
            }
            
            # 合并特征
            enhanced_user_feat.update(time_features_for_user)
            enhanced_item_feat.update(time_features_for_item)
            
            enhanced_sequence.append((
                user_id, item_id, enhanced_user_feat, enhanced_item_feat, action_type, timestamp
            ))
        
        return enhanced_sequence
    
    def batch_process_time_features(self, user_sequences, num_workers=None):
        """
        批量处理用户序列的时间特征
        
        Args:
            user_sequences: 用户序列列表
            num_workers: 并行工作进程数，默认为CPU核数
            
        Returns:
            list: 处理后的用户序列列表
        """
        if num_workers is None:
            num_workers = min(mp.cpu_count(), len(user_sequences))
        
        print(f"正在使用 {num_workers} 个进程批量处理时间特征...")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            enhanced_sequences = list(executor.map(
                self.process_user_sequence_time_features, 
                user_sequences
            ))
        
        return enhanced_sequences
    
    def cache_time_features(self, user_id, time_features, cache_key="time_features"):
        """
        缓存用户时间特征
        
        Args:
            user_id: 用户ID
            time_features: 时间特征字典
            cache_key: 缓存键
        """
        if not self.enable_cache:
            return
        
        cache_file = self.time_features_cache_dir / f"{cache_key}_{user_id}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(time_features, f)
    
    def load_cached_time_features(self, user_id, cache_key="time_features"):
        """
        加载缓存的用户时间特征
        
        Args:
            user_id: 用户ID
            cache_key: 缓存键
            
        Returns:
            dict or None: 时间特征字典，如果不存在则返回None
        """
        if not self.enable_cache:
            return None
        
        cache_file = self.time_features_cache_dir / f"{cache_key}_{user_id}.pkl"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"加载缓存失败 {cache_file}: {e}")
            return None


def get_time_feature_names():
    """
    获取所有时间特征的名称列表
    
    Returns:
        dict: 特征名称字典，分为用户级和物品级特征
    """
    user_time_features = [
        'seq_avg_interval',
        'seq_interval_variance', 
        'seq_click_frequency',
        'seq_temporal_consistency'
    ]
    
    item_time_features = [
        'time_interval_minutes',
        'time_interval_bucket',
        'time_decay_weight',
        'is_immediate_click',
        'is_short_interval',
        'is_medium_interval',
        'is_long_interval',
        'hour',
        'day_of_week',
        'hour_sin',
        'hour_cos', 
        'day_sin',
        'day_cos',
        'time_period',
        'is_weekend',
        'is_workday',
        'is_morning_peak',
        'is_afternoon',
        'is_evening',
        'is_dawn'
    ]
    
    return {
        'user_time_features': user_time_features,
        'item_time_features': item_time_features
    }


# 单例时间特征处理器
_time_processor = None

def get_time_processor():
    """获取全局时间特征处理器单例"""
    global _time_processor
    if _time_processor is None:
        _time_processor = TimeFeatureProcessor()
    return _time_processor


if __name__ == "__main__":
    # 测试代码
    processor = TimeFeatureProcessor()
    
    # 模拟用户序列数据
    import time
    current_time = time.time()
    
    test_sequence = [
        (1, 100, {'103': 1}, {'111': 5}, 0, current_time - 3600),     # 1小时前
        (1, 101, {'103': 1}, {'111': 6}, 0, current_time - 1800),     # 30分钟前  
        (1, 102, {'103': 1}, {'111': 7}, 1, current_time - 900),      # 15分钟前（点击）
        (1, 103, {'103': 1}, {'111': 8}, 0, current_time - 300),      # 5分钟前
        (1, 104, {'103': 1}, {'111': 9}, 1, current_time - 60),       # 1分钟前（点击）
        (1, 105, {'103': 1}, {'111': 10}, 0, current_time)            # 现在
    ]
    
    enhanced_sequence = processor.process_user_sequence_time_features(test_sequence)
    
    # 打印时间特征示例
    print("时间特征示例：")
    for i, record in enumerate(enhanced_sequence[:3]):
        user_id, item_id, user_feat, item_feat, action_type, timestamp = record
        print(f"\n记录 {i}:")
        print(f"  用户时间特征: {[(k, v) for k, v in user_feat.items() if k.startswith('seq_')]}")
        print(f"  物品时间特征: {[(k, v) for k, v in item_feat.items() if any(x in k for x in ['time_', 'is_', 'hour', 'day'])]}")