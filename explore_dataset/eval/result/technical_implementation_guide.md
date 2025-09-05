# 评测数据集技术实现指南

## 🎯 概述

基于评测数据集分析结果，本指南提供具体的技术实现方案，帮助优化推理性能和处理数据质量问题。

---

## 🔧 数据预处理实现

### 1. 混合数据类型处理

基于分析发现，9个特征存在int/str混合类型问题。以下是增强版的处理函数：

```python
def enhanced_process_cold_start_feat(feat):
    """
    增强版冷启动特征处理函数
    解决混合数据类型和特征115缺失问题
    """
    if not isinstance(feat, dict):
        return {}
    
    processed_feat = {}
    
    # 混合类型特征列表
    mixed_type_features = ['102', '111', '115', '117', '118', '119', '120', '121', '122']
    
    for feat_id, feat_value in feat.items():
        if feat_id in mixed_type_features:
            # 处理混合类型
            if isinstance(feat_value, str):
                try:
                    # 尝试转换为整数
                    processed_feat[feat_id] = int(feat_value)
                except (ValueError, TypeError):
                    try:
                        # 尝试转换为浮点数再取整
                        processed_feat[feat_id] = int(float(feat_value))
                    except (ValueError, TypeError):
                        # 转换失败，使用默认值0
                        processed_feat[feat_id] = 0
                        print(f"Warning: Failed to convert feature {feat_id} value '{feat_value}' to int")
            elif isinstance(feat_value, (int, float)):
                processed_feat[feat_id] = int(feat_value)
            else:
                processed_feat[feat_id] = 0
        else:
            # 正常类型特征直接复制
            processed_feat[feat_id] = feat_value
    
    # 处理特征115缺失问题（70.37%缺失率）
    if '115' not in processed_feat:
        processed_feat['115'] = 493  # 使用分析得出的均值
    
    return processed_feat

def batch_process_candidates(candidates_data, batch_size=1000):
    """
    批量处理候选物品数据
    """
    processed_candidates = []
    
    for i in tqdm(range(0, len(candidates_data), batch_size), desc="Processing candidates"):
        batch = candidates_data[i:i+batch_size]
        
        for candidate in batch:
            # 处理features字段
            if 'features' in candidate:
                candidate['features'] = enhanced_process_cold_start_feat(candidate['features'])
            
            processed_candidates.append(candidate)
    
    return processed_candidates
```

### 2. 数据验证函数

```python
def validate_processed_data(processed_candidates):
    """
    验证处理后的数据质量
    """
    validation_stats = {
        'total_records': len(processed_candidates),
        'feature_completeness': {},
        'type_consistency': {},
        'value_ranges': {}
    }
    
    # 统计特征完整性
    for candidate in processed_candidates[:1000]:  # 抽样验证
        features = candidate.get('features', {})
        
        for feat_id in range(100, 123):
            feat_str = str(feat_id)
            if feat_str in features:
                if feat_str not in validation_stats['feature_completeness']:
                    validation_stats['feature_completeness'][feat_str] = 0
                validation_stats['feature_completeness'][feat_str] += 1
                
                # 检查类型一致性
                value = features[feat_str]
                if feat_str not in validation_stats['type_consistency']:
                    validation_stats['type_consistency'][feat_str] = set()
                validation_stats['type_consistency'][feat_str].add(type(value).__name__)
    
    return validation_stats
```

---

## 🚀 高效推理实现

### 1. 候选物品Embedding预计算

```python
import torch
import numpy as np
from tqdm import tqdm
import os

def precompute_candidate_embeddings(model, candidates, batch_size=128, save_path=None):
    """
    预计算所有候选物品的embedding
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_embeddings = []
    all_retrieval_ids = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(candidates), batch_size), desc="Computing embeddings"):
            batch = candidates[i:i+batch_size]
            
            # 构造批次数据
            batch_features = []
            batch_retrieval_ids = []
            
            for candidate in batch:
                features = candidate['features']
                retrieval_id = candidate['retrieval_id']
                
                batch_features.append(features)
                batch_retrieval_ids.append(retrieval_id)
            
            # 生成embedding
            batch_embs = model.generate_item_embeddings(batch_features)
            
            all_embeddings.append(batch_embs.cpu())
            all_retrieval_ids.extend(batch_retrieval_ids)
    
    # 合并所有embedding
    candidate_embeddings = torch.cat(all_embeddings, dim=0)
    
    # 保存到缓存
    if save_path:
        cache_data = {
            'embeddings': candidate_embeddings,
            'retrieval_ids': all_retrieval_ids,
            'embedding_dim': candidate_embeddings.shape[1]
        }
        torch.save(cache_data, save_path)
        print(f"Candidate embeddings saved to {save_path}")
    
    return candidate_embeddings, all_retrieval_ids

def load_cached_embeddings(cache_path):
    """
    加载缓存的embedding
    """
    if os.path.exists(cache_path):
        cache_data = torch.load(cache_path)
        return cache_data['embeddings'], cache_data['retrieval_ids']
    else:
        return None, None
```

### 2. 内存优化的用户序列处理

```python
def memory_efficient_user_processing(model, test_loader, device):
    """
    内存优化的用户序列处理
    """
    model.eval()
    
    all_user_embeddings = []
    all_user_ids = []
    
    # 使用生成器来减少内存占用
    def batch_generator():
        for batch in test_loader:
            seq, token_type, seq_feat, user_ids = batch
            seq = seq.to(device)
            
            with torch.no_grad():
                user_embs = model.predict(seq, seq_feat, token_type)
                
                # 立即移到CPU并转换为float16以节省内存
                user_embs = user_embs.cpu().half()
                
                yield user_embs, user_ids
    
    # 处理批次
    for user_embs, user_ids in tqdm(batch_generator(), desc="Processing users"):
        all_user_embeddings.append(user_embs)
        all_user_ids.extend(user_ids)
        
        # 定期清理GPU缓存
        if len(all_user_embeddings) % 10 == 0:
            torch.cuda.empty_cache()
    
    # 合并用户embedding
    user_embeddings = torch.cat(all_user_embeddings, dim=0).float()
    
    return user_embeddings, all_user_ids
```

### 3. 高效ANN检索实现

```python
import faiss

def setup_efficient_ann_search(candidate_embeddings, use_gpu=True):
    """
    设置高效的ANN检索
    """
    embedding_dim = candidate_embeddings.shape[1]
    
    # 根据数据规模选择索引类型
    if len(candidate_embeddings) > 100000:
        # 大规模数据使用HNSW
        index = faiss.IndexHNSWFlat(embedding_dim, 64)  # 64 connections
        index.hnsw.efConstruction = 1280
        index.hnsw.efSearch = 640
    else:
        # 中小规模数据使用IVF
        nlist = min(4096, len(candidate_embeddings) // 100)
        quantizer = faiss.IndexFlatIP(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
    
    # GPU加速（如果可用）
    if use_gpu and faiss.get_num_gpus() > 0:
        print("Using GPU for ANN search")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    # 训练索引（如果需要）
    if hasattr(index, 'train'):
        print("Training index...")
        index.train(candidate_embeddings.numpy().astype('float32'))
    
    # 添加向量
    print("Adding vectors to index...")
    index.add(candidate_embeddings.numpy().astype('float32'))
    
    return index

def batch_ann_search(index, user_embeddings, retrieval_ids, k=10, batch_size=1000):
    """
    批量ANN检索
    """
    all_top10s = []
    
    for i in tqdm(range(0, len(user_embeddings), batch_size), desc="ANN search"):
        batch_embs = user_embeddings[i:i+batch_size].numpy().astype('float32')
        
        # 搜索
        scores, indices = index.search(batch_embs, k)
        
        # 转换为creative_id
        batch_top10s = []
        for user_indices in indices:
            user_top10 = [retrieval_ids[idx] for idx in user_indices]
            batch_top10s.append(user_top10)
        
        all_top10s.extend(batch_top10s)
    
    return all_top10s
```

---

## 📊 性能监控和调试

### 1. 详细性能监控

```python
import time
import psutil
import torch

class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        
    def start_timer(self, name):
        self.timings[name + '_start'] = time.time()
        
    def end_timer(self, name):
        if name + '_start' in self.timings:
            duration = time.time() - self.timings[name + '_start']
            self.timings[name] = duration
            print(f"{name}: {duration:.2f} seconds")
    
    def log_memory(self, stage):
        memory_info = {
            'cpu_percent': psutil.virtual_memory().percent,
            'cpu_used_gb': psutil.virtual_memory().used / (1024**3)
        }
        
        if torch.cuda.is_available():
            memory_info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            memory_info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        
        self.memory_usage[stage] = memory_info
        print(f"Memory at {stage}: CPU {memory_info['cpu_percent']:.1f}% ({memory_info['cpu_used_gb']:.1f}GB)")
        
        if 'gpu_allocated_gb' in memory_info:
            print(f"GPU: {memory_info['gpu_allocated_gb']:.1f}GB allocated, {memory_info['gpu_reserved_gb']:.1f}GB reserved")
    
    def generate_report(self):
        print("\n=== Performance Report ===")
        total_time = sum(v for k, v in self.timings.items() if not k.endswith('_start'))
        print(f"Total execution time: {total_time:.2f} seconds")
        
        for stage, timing in self.timings.items():
            if not stage.endswith('_start'):
                percentage = (timing / total_time) * 100
                print(f"  {stage}: {timing:.2f}s ({percentage:.1f}%)")

# 使用示例
monitor = PerformanceMonitor()

# 在推理流程中使用
monitor.start_timer("data_loading")
# ... 数据加载代码 ...
monitor.end_timer("data_loading")
monitor.log_memory("after_data_loading")
```

### 2. 数据质量验证

```python
def validate_inference_results(top10s, user_list, candidates):
    """
    验证推理结果的质量
    """
    validation_report = {
        'result_count': len(top10s),
        'user_count': len(user_list),
        'consistency_check': True,
        'coverage_stats': {},
        'quality_issues': []
    }
    
    # 检查结果一致性
    if len(top10s) != len(user_list):
        validation_report['consistency_check'] = False
        validation_report['quality_issues'].append("Mismatch between top10s and user_list length")
    
    # 检查top10格式
    valid_results = 0
    for i, user_top10 in enumerate(top10s):
        if len(user_top10) == 10:
            valid_results += 1
        else:
            validation_report['quality_issues'].append(f"User {i} has {len(user_top10)} items instead of 10")
    
    validation_report['coverage_stats']['valid_results'] = valid_results
    validation_report['coverage_stats']['coverage_rate'] = valid_results / len(top10s) * 100
    
    # 检查推荐物品的有效性
    candidate_ids = set(c['creative_id'] for c in candidates)
    unique_recommended = set()
    
    for user_top10 in top10s[:100]:  # 抽样检查
        for item_id in user_top10:
            unique_recommended.add(item_id)
            if item_id not in candidate_ids:
                validation_report['quality_issues'].append(f"Invalid item_id {item_id} not in candidates")
    
    validation_report['coverage_stats']['unique_recommended'] = len(unique_recommended)
    
    return validation_report
```

---

## 🎯 完整推理流程

```python
def optimized_inference_pipeline():
    """
    完整的优化推理流程
    """
    monitor = PerformanceMonitor()
    
    # 1. 数据加载和预处理
    monitor.start_timer("data_loading")
    monitor.log_memory("start")
    
    # 加载评测数据
    eval_data_path = os.environ.get('EVAL_DATA_PATH')
    test_dataset = MyTestDataset(eval_data_path, args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 加载候选数据
    candidates = load_candidates(eval_data_path)
    candidates = batch_process_candidates(candidates)
    
    monitor.end_timer("data_loading")
    monitor.log_memory("after_data_loading")
    
    # 2. 模型加载
    monitor.start_timer("model_loading")
    
    model = load_trained_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    monitor.end_timer("model_loading")
    
    # 3. 候选物品embedding预计算
    monitor.start_timer("candidate_embedding")
    
    cache_path = os.path.join(os.environ.get('USER_CACHE_PATH', './cache'), 'candidate_embeddings.pt')
    candidate_embeddings, retrieval_ids = load_cached_embeddings(cache_path)
    
    if candidate_embeddings is None:
        candidate_embeddings, retrieval_ids = precompute_candidate_embeddings(
            model, candidates, save_path=cache_path
        )
    
    monitor.end_timer("candidate_embedding")
    monitor.log_memory("after_candidate_embedding")
    
    # 4. 用户序列编码
    monitor.start_timer("user_encoding")
    
    user_embeddings, user_list = memory_efficient_user_processing(model, test_loader, device)
    
    monitor.end_timer("user_encoding")
    monitor.log_memory("after_user_encoding")
    
    # 5. ANN检索
    monitor.start_timer("ann_search")
    
    index = setup_efficient_ann_search(candidate_embeddings, use_gpu=True)
    top10s = batch_ann_search(index, user_embeddings, retrieval_ids)
    
    monitor.end_timer("ann_search")
    monitor.log_memory("after_ann_search")
    
    # 6. 结果验证
    monitor.start_timer("validation")
    
    validation_report = validate_inference_results(top10s, user_list, candidates)
    print("Validation Report:", validation_report)
    
    monitor.end_timer("validation")
    
    # 7. 性能报告
    monitor.generate_report()
    
    return top10s, user_list

if __name__ == "__main__":
    top10s, user_list = optimized_inference_pipeline()
```

---

## 📋 部署检查清单

### ✅ 代码优化
- [ ] 实现enhanced_process_cold_start_feat函数
- [ ] 添加批量数据处理功能
- [ ] 实现候选embedding缓存机制
- [ ] 优化内存使用和GPU利用率

### ✅ 数据验证  
- [ ] 验证混合类型特征处理正确性
- [ ] 确认特征115默认值策略
- [ ] 检查数据预处理pipeline
- [ ] 验证推理结果格式

### ✅ 性能优化
- [ ] 配置合适的批次大小
- [ ] 实现ANN检索参数调优
- [ ] 添加性能监控和日志
- [ ] 测试不同硬件配置下的表现

### ✅ 错误处理
- [ ] 添加数据类型转换异常处理
- [ ] 实现推理过程错误恢复
- [ ] 配置详细的日志记录
- [ ] 准备回退方案

---

*技术指南版本: v1.0*  
*适用于腾讯算法大赛评测环境*  
*更新时间: 2025-09-05*