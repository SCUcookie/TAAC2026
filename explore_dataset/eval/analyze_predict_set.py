#!/usr/bin/env python3
"""
专门分析 predict_set.jsonl 文件的详细结构和内容
基于已有的分析结果，深入了解候选推荐集合的特征分布
"""

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_predict_set_detailed():
    """详细分析predict_set.jsonl文件"""
    
    eval_data_path = os.environ.get('EVAL_DATA_PATH', '../../dataset/TencentGR_1k')
    cache_path = os.environ.get('USER_CACHE_PATH', '../../analysis_cache')
    
    predict_set_file = Path(eval_data_path) / 'predict_set.jsonl'
    cache_dir = Path(cache_path)
    cache_dir.mkdir(exist_ok=True)
    
    if not predict_set_file.exists():
        print(f"错误: {predict_set_file} 不存在")
        return
    
    print(f"=== 分析 predict_set.jsonl ===")
    print(f"文件路径: {predict_set_file}")
    
    # 数据收集
    records = []
    creative_ids = set()
    retrieval_ids = set()
    feature_stats = defaultdict(list)
    feature_types = defaultdict(set)
    
    print("读取数据...")
    line_count = 0
    with open(predict_set_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="处理记录"):
            try:
                record = json.loads(line.strip())
                line_count += 1
                
                # 基本ID统计
                creative_ids.add(record.get('creative_id'))
                retrieval_ids.add(record.get('retrieval_id'))
                
                # 特征分析
                features = record.get('features', {})
                for feat_id, feat_value in features.items():
                    feature_types[feat_id].add(type(feat_value).__name__)
                    
                    # 收集特征值用于统计分析
                    if isinstance(feat_value, (int, float)):
                        feature_stats[feat_id].append(feat_value)
                    elif isinstance(feat_value, list):
                        feature_stats[f"{feat_id}_length"].append(len(feat_value))
                        if feat_value and isinstance(feat_value[0], (int, float)):
                            feature_stats[f"{feat_id}_values"].extend(feat_value)
                
                # 只保存前1000条记录的详细信息以节省内存
                if len(records) < 1000:
                    records.append(record)
                    
            except json.JSONDecodeError as e:
                continue
    
    print(f"\n=== 基础统计信息 ===")
    print(f"总记录数: {line_count:,}")
    print(f"唯一creative_id数量: {len(creative_ids):,}")
    print(f"唯一retrieval_id数量: {len(retrieval_ids):,}")
    print(f"creative_id与retrieval_id是否一一对应: {len(creative_ids) == len(retrieval_ids) == line_count}")
    
    print(f"\n=== 特征分析 ===")
    print(f"特征总数: {len(feature_types)}")
    
    # 特征类型分布
    print("\n特征类型分布:")
    for feat_id, types in sorted(feature_types.items()):
        print(f"  {feat_id}: {', '.join(types)}")
    
    # 数值特征统计
    print(f"\n=== 数值特征统计 ===")
    numeric_features = {}
    for feat_id, values in feature_stats.items():
        if values and len(values) > 0:
            values_array = np.array(values)
            if feat_id.endswith('_length') or feat_id.endswith('_values'):
                continue
            
            numeric_features[feat_id] = {
                'count': len(values),
                'mean': np.mean(values_array),
                'std': np.std(values_array),
                'min': np.min(values_array),
                'max': np.max(values_array),
                'unique_count': len(np.unique(values_array))
            }
    
    for feat_id, stats in sorted(numeric_features.items()):
        print(f"  {feat_id}:")
        print(f"    样本数: {stats['count']:,}")
        print(f"    均值: {stats['mean']:.4f}")
        print(f"    标准差: {stats['std']:.4f}")
        print(f"    范围: [{stats['min']}, {stats['max']}]")
        print(f"    唯一值数量: {stats['unique_count']:,}")
    
    # 数组特征分析
    print(f"\n=== 数组特征分析 ===")
    array_features = {}
    for feat_id, values in feature_stats.items():
        if feat_id.endswith('_length'):
            base_feat = feat_id.replace('_length', '')
            array_features[base_feat] = {
                'lengths': values,
                'avg_length': np.mean(values),
                'max_length': np.max(values),
                'min_length': np.min(values)
            }
    
    for feat_id, stats in sorted(array_features.items()):
        print(f"  {feat_id}:")
        print(f"    平均长度: {stats['avg_length']:.2f}")
        print(f"    长度范围: [{stats['min_length']}, {stats['max_length']}]")
    
    # 示例记录
    print(f"\n=== 示例记录 ===")
    if records:
        for i, record in enumerate(records[:3]):
            print(f"\n示例 {i+1}:")
            print(f"  creative_id: {record.get('creative_id')}")
            print(f"  retrieval_id: {record.get('retrieval_id')}")
            print(f"  特征数量: {len(record.get('features', {}))}")
            
            # 显示前几个特征
            features = record.get('features', {})
            for j, (feat_id, feat_value) in enumerate(list(features.items())[:5]):
                if isinstance(feat_value, list) and len(feat_value) > 5:
                    print(f"    {feat_id}: [{feat_value[0]}, {feat_value[1]}, ..., {feat_value[-1]}] (长度: {len(feat_value)})")
                else:
                    print(f"    {feat_id}: {feat_value}")
            
            if len(features) > 5:
                print(f"    ... 还有 {len(features) - 5} 个特征")
    
    # 保存分析结果
    analysis_result = {
        'basic_stats': {
            'total_records': line_count,
            'unique_creative_ids': len(creative_ids),
            'unique_retrieval_ids': len(retrieval_ids)
        },
        'feature_types': {k: list(v) for k, v in feature_types.items()},
        'numeric_features': numeric_features,
        'array_features': array_features,
        'sample_records': records[:10]  # 保存前10条记录作为样本
    }
    
    result_file = cache_dir / 'predict_set_detailed_analysis.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n详细分析结果已保存到: {result_file}")
    
    # 对比分析：与训练数据的item_feat_dict.json对比
    print(f"\n=== 与训练数据对比分析 ===")
    item_feat_file = Path(eval_data_path) / 'item_feat_dict.json'
    if item_feat_file.exists():
        print("加载训练数据的item特征字典...")
        with open(item_feat_file, 'r', encoding='utf-8') as f:
            item_feat_dict = json.load(f)
        
        print(f"训练数据中的item数量: {len(item_feat_dict):,}")
        
        # 检查predict_set中的creative_id是否在训练数据中
        training_creative_ids = set(item_feat_dict.keys())
        overlap = creative_ids.intersection(training_creative_ids)
        
        print(f"predict_set中的creative_id在训练数据中的覆盖率: {len(overlap)/len(creative_ids)*100:.2f}%")
        print(f"  重叠数量: {len(overlap):,}")
        print(f"  predict_set独有: {len(creative_ids - training_creative_ids):,}")
        print(f"  训练数据独有: {len(training_creative_ids - creative_ids):,}")
        
        # 特征字段对比
        if item_feat_dict:
            sample_training_features = set(list(item_feat_dict.values())[0].keys())
            predict_features = set(feature_types.keys())
            
            print(f"\n特征字段对比:")
            print(f"  训练数据特征数: {len(sample_training_features)}")
            print(f"  predict_set特征数: {len(predict_features)}")
            print(f"  共同特征: {len(sample_training_features.intersection(predict_features))}")
            print(f"  predict_set独有特征: {predict_features - sample_training_features}")
            print(f"  训练数据独有特征: {sample_training_features - predict_features}")
    
    return analysis_result

def main():
    try:
        result = analyze_predict_set_detailed()
        print("\n=== 分析完成 ===")
        return result
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()