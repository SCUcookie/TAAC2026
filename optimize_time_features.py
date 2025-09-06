"""
时间特征性能优化工具
利用缓存、并行处理和批处理优化时间特征计算性能

主要优化策略：
1. 利用20GB用户缓存空间预计算特征
2. 多进程并行处理用户序列
3. 向量化计算减少循环开销
4. 内存映射减少I/O开销
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache, partial
import mmap
from tqdm import tqdm
import torch
from time_features import TimeFeatureProcessor, get_time_feature_names
import psutil
import gc


class OptimizedTimeFeatureProcessor:
    """优化版时间特征处理器"""
    
    def __init__(self, data_path, cache_dir=None, num_workers=None):
        """
        初始化优化时间特征处理器
        
        Args:
            data_path: 数据文件路径
            cache_dir: 缓存目录，默认使用USER_CACHE_PATH
            num_workers: 并行工作进程数
        """
        self.data_path = Path(data_path)
        
        if cache_dir is None:
            cache_dir = os.environ.get('USER_CACHE_PATH', './cache')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_workers = num_workers or min(mp.cpu_count(), 8)
        self.time_processor = TimeFeatureProcessor(cache_dir=cache_dir)
        
        # 优化配置
        self.optimization_config = {
            'batch_size': 1000,  # 批处理大小
            'chunk_size': 10000,  # 数据块大小
            'cache_threshold': 0.8,  # 缓存使用阈值
            'memory_limit_gb': 16,  # 内存限制
        }
        
        print(f"✓ 优化时间特征处理器初始化完成")
        print(f"  - 并行工作进程数: {self.num_workers}")
        print(f"  - 缓存目录: {self.cache_dir}")
        print(f"  - 批处理大小: {self.optimization_config['batch_size']}")
    
    def precompute_user_time_features(self, force_rebuild=False):
        """
        预计算所有用户的时间特征并缓存
        
        Args:
            force_rebuild: 是否强制重建缓存
        """
        print("🚀 开始预计算用户时间特征...")
        
        # 检查缓存状态
        cache_file = self.cache_dir / "user_time_features_cache.pkl"
        if cache_file.exists() and not force_rebuild:
            print("✓ 发现现有缓存，跳过预计算")
            return
        
        # 加载用户序列偏移量
        with open(self.data_path / 'seq_offsets.pkl', 'rb') as f:
            seq_offsets = pickle.load(f)
        
        user_ids = list(seq_offsets.keys())
        print(f"  - 需要处理的用户数量: {len(user_ids):,}")
        
        # 分批处理用户
        batch_size = self.optimization_config['batch_size']
        user_batches = [user_ids[i:i+batch_size] for i in range(0, len(user_ids), batch_size)]
        
        print(f"  - 分为 {len(user_batches)} 个批次并行处理")
        
        # 并行处理各批次
        all_time_features = {}
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有批次任务
            future_to_batch = {
                executor.submit(self._process_user_batch, batch, seq_offsets): batch_idx 
                for batch_idx, batch in enumerate(user_batches)
            }
            
            # 收集结果
            with tqdm(total=len(user_batches), desc="处理用户批次") as pbar:
                for future in future_to_batch:
                    batch_idx = future_to_batch[future]
                    try:
                        batch_features = future.result()
                        all_time_features.update(batch_features)
                        pbar.update(1)
                    except Exception as e:
                        print(f"批次 {batch_idx} 处理失败: {e}")
        
        # 保存缓存
        print("💾 保存时间特征缓存...")
        with open(cache_file, 'wb') as f:
            pickle.dump(all_time_features, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"✓ 时间特征预计算完成，处理了 {len(all_time_features):,} 个用户")
        
        # 内存清理
        del all_time_features
        gc.collect()
    
    def _process_user_batch(self, user_batch, seq_offsets):
        """
        处理单个用户批次的时间特征
        
        Args:
            user_batch: 用户ID列表
            seq_offsets: 序列偏移量字典
            
        Returns:
            dict: 用户时间特征字典
        """
        batch_features = {}
        
        # 使用内存映射文件读取
        seq_file_path = self.data_path / "seq.jsonl"
        with open(seq_file_path, 'rb') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                for uid in user_batch:
                    try:
                        # 读取用户序列
                        mmapped_file.seek(seq_offsets[uid])
                        line = mmapped_file.readline()
                        user_sequence = json.loads(line.decode('utf-8'))
                        
                        # 处理时间特征
                        enhanced_sequence = self.time_processor.process_user_sequence_time_features(
                            user_sequence
                        )
                        
                        # 提取时间特征统计信息（用于缓存）
                        time_features = self._extract_time_feature_stats(enhanced_sequence)
                        batch_features[uid] = time_features
                        
                    except Exception as e:
                        print(f"处理用户 {uid} 失败: {e}")
                        continue
        
        return batch_features
    
    def _extract_time_feature_stats(self, enhanced_sequence):
        """
        提取时间特征统计信息用于缓存
        
        Args:
            enhanced_sequence: 增强后的用户序列
            
        Returns:
            dict: 时间特征统计信息
        """
        if not enhanced_sequence:
            return {}
        
        # 提取所有时间特征值
        time_feature_values = {}
        time_feature_names = get_time_feature_names()
        
        for feature_name in time_feature_names['item_time_features']:
            values = []
            for record in enhanced_sequence:
                item_feat = record[3]  # item_feat
                if feature_name in item_feat:
                    values.append(item_feat[feature_name])
            
            if values:
                time_feature_values[feature_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1]  # 最后一个值
                }
        
        for feature_name in time_feature_names['user_time_features']:
            # 用户级特征通常在最后一个记录中
            if enhanced_sequence:
                user_feat = enhanced_sequence[-1][2]  # user_feat
                if feature_name in user_feat:
                    time_feature_values[feature_name] = user_feat[feature_name]
        
        return time_feature_values
    
    def load_cached_time_features(self):
        """
        加载缓存的时间特征
        
        Returns:
            dict: 用户时间特征字典，如果不存在则返回None
        """
        cache_file = self.cache_dir / "user_time_features_cache.pkl"
        if not cache_file.exists():
            return None
        
        print("📂 加载缓存的时间特征...")
        try:
            with open(cache_file, 'rb') as f:
                features = pickle.load(f)
            print(f"✓ 成功加载 {len(features):,} 个用户的时间特征缓存")
            return features
        except Exception as e:
            print(f"⚠️ 加载缓存失败: {e}")
            return None
    
    def get_optimized_time_features(self, user_id):
        """
        获取优化的用户时间特征
        
        Args:
            user_id: 用户ID
            
        Returns:
            dict: 用户时间特征，如果不存在则返回空字典
        """
        # 尝试从缓存加载
        if not hasattr(self, '_cached_features'):
            self._cached_features = self.load_cached_time_features()
        
        if self._cached_features and user_id in self._cached_features:
            return self._cached_features[user_id]
        else:
            return {}
    
    def optimize_memory_usage(self):
        """优化内存使用"""
        print("🧹 优化内存使用...")
        
        # 检查内存使用情况
        memory_info = psutil.virtual_memory()
        memory_used_gb = memory_info.used / (1024**3)
        memory_percent = memory_info.percent
        
        print(f"  - 当前内存使用: {memory_used_gb:.2f}GB ({memory_percent:.1f}%)")
        
        # 如果内存使用超过阈值，进行清理
        if memory_percent > self.optimization_config['cache_threshold'] * 100:
            print("  - 内存使用过高，执行清理...")
            
            # 清理缓存的特征
            if hasattr(self, '_cached_features'):
                del self._cached_features
            
            # 强制垃圾回收
            gc.collect()
            
            # 重新检查内存
            memory_info = psutil.virtual_memory()
            print(f"  - 清理后内存使用: {memory_info.used / (1024**3):.2f}GB ({memory_info.percent:.1f}%)")
    
    def benchmark_performance(self, sample_size=1000):
        """
        性能基准测试
        
        Args:
            sample_size: 测试样本数量
        """
        print("📊 开始性能基准测试...")
        
        # 加载用户序列偏移量
        with open(self.data_path / 'seq_offsets.pkl', 'rb') as f:
            seq_offsets = pickle.load(f)
        
        # 随机选择测试用户
        user_ids = list(seq_offsets.keys())
        sample_users = np.random.choice(user_ids, min(sample_size, len(user_ids)), replace=False)
        
        # 测试原始方法
        import time
        print(f"  - 测试样本数量: {len(sample_users)}")
        
        # 原始方法性能测试
        start_time = time.time()
        original_results = []
        
        for uid in tqdm(sample_users[:100], desc="原始方法"):
            # 模拟原始处理
            try:
                with open(self.data_path / "seq.jsonl", 'rb') as f:
                    f.seek(seq_offsets[uid])
                    line = f.readline()
                    user_sequence = json.loads(line.decode('utf-8'))
                    
                enhanced = self.time_processor.process_user_sequence_time_features(user_sequence)
                original_results.append(len(enhanced))
            except:
                continue
        
        original_time = time.time() - start_time
        
        # 优化方法性能测试
        start_time = time.time()
        
        # 预计算特征（如果还没有）
        if not (self.cache_dir / "user_time_features_cache.pkl").exists():
            self.precompute_user_time_features()
        
        # 测试缓存访问速度
        cached_features = self.load_cached_time_features()
        optimized_results = []
        
        for uid in sample_users[:100]:
            features = cached_features.get(uid, {})
            optimized_results.append(len(features))
        
        optimized_time = time.time() - start_time
        
        # 输出结果
        print(f"\n📈 性能基准测试结果:")
        print(f"  - 原始方法耗时: {original_time:.2f}秒")
        print(f"  - 优化方法耗时: {optimized_time:.2f}秒")
        print(f"  - 性能提升: {original_time/max(optimized_time, 0.001):.1f}x")
        print(f"  - 处理用户数: {len(original_results)}")


def optimize_dataset_time_features(data_path, cache_dir=None, num_workers=None):
    """
    优化数据集的时间特征处理
    
    Args:
        data_path: 数据路径
        cache_dir: 缓存目录
        num_workers: 工作进程数
    """
    print("🎯 开始数据集时间特征优化...")
    
    optimizer = OptimizedTimeFeatureProcessor(
        data_path=data_path,
        cache_dir=cache_dir,
        num_workers=num_workers
    )
    
    # 预计算时间特征
    optimizer.precompute_user_time_features()
    
    # 性能测试
    optimizer.benchmark_performance()
    
    # 内存优化
    optimizer.optimize_memory_usage()
    
    print("✅ 数据集时间特征优化完成!")
    
    return optimizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="时间特征性能优化工具")
    parser.add_argument('--data_path', type=str, default='./dataset/TencentGR_1k',
                       help='训练数据路径')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='缓存目录，默认使用USER_CACHE_PATH')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='并行工作进程数')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能基准测试')
    parser.add_argument('--precompute_only', action='store_true',
                       help='仅预计算特征，不运行测试')
    
    args = parser.parse_args()
    
    # 设置环境变量
    if args.cache_dir is None:
        cache_dir = os.environ.get('USER_CACHE_PATH', './cache')
    else:
        cache_dir = args.cache_dir
    
    # 创建优化器
    optimizer = OptimizedTimeFeatureProcessor(
        data_path=args.data_path,
        cache_dir=cache_dir,
        num_workers=args.num_workers
    )
    
    if args.precompute_only:
        # 仅预计算
        optimizer.precompute_user_time_features()
    elif args.benchmark:
        # 运行基准测试
        optimizer.benchmark_performance()
    else:
        # 完整优化流程
        optimize_dataset_time_features(
            data_path=args.data_path,
            cache_dir=cache_dir,
            num_workers=args.num_workers
        )