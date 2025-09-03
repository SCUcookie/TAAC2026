#!/usr/bin/env python3
"""
腾讯广告算法大赛 - 真实数据集总体概述分析
比赛服务器数据集结构分析和基本统计信息

Author: Analysis Script
Date: 2025-01-20
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 导入报告工具
from report_utils import save_detailed_report, output_report_to_log, generate_statistical_summary, create_comprehensive_analysis_report

def load_json_safe(file_path, max_lines=None):
    """安全加载JSON文件，完整分析所有数据"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    print(f"警告: 文件 {file_path} 超过 {max_lines} 行，仅分析前 {max_lines} 行")
                    break
                try:
                    if line.strip():
                        data.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"警告: 第 {i+1} 行JSON解析失败")
                    continue
                    
                if (i + 1) % 10000 == 0:
                    print(f"已加载 {i + 1} 行数据...")
                    
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
    return data

def load_pickle_safe(file_path):
    """安全加载pickle文件"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"错误: 无法读取pickle文件 {file_path}: {e}")
        return None

def analyze_dataset_overview():
    """
    分析数据集总体概述
    """
    print("=" * 60)
    print("腾讯广告算法大赛 - 真实数据集总体概述分析")
    print("=" * 60)
    
    # 获取数据根目录
    data_root = os.environ.get('TRAIN_DATA_PATH')
    if not data_root:
        print("错误: 未设置环境变量 TRAIN_DATA_PATH")
        return
    
    data_root = Path(data_root)
    if not data_root.exists():
        print(f"错误: 数据目录不存在: {data_root}")
        return
    
    print(f"数据集根目录: {data_root}")
    print()
    
    # 分析文件结构和大小
    files_info = {}
    expected_files = [
        'indexer.pkl',
        'item_feat_dict.json', 
        'item_feat_num.json',
        'seq.jsonl',
        'seq_offsets.pkl',
        'user_feat_num.json'
    ]
    
    print("1. 数据文件概览:")
    print("-" * 40)
    for file_name in expected_files:
        file_path = data_root / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            files_info[file_name] = {
                'path': file_path,
                'size_mb': size_mb,
                'exists': True
            }
            print(f"✓ {file_name:20} - {size_mb:.2f} MB")
        else:
            files_info[file_name] = {'exists': False}
            print(f"✗ {file_name:20} - 文件不存在")
    
    print()
    
    # 分析 indexer.pkl
    if files_info.get('indexer.pkl', {}).get('exists'):
        print("2. 索引器 (indexer.pkl) 基本信息:")
        print("-" * 40)
        indexer = load_pickle_safe(files_info['indexer.pkl']['path'])
        if indexer:
            for key, value in indexer.items():
                if isinstance(value, dict):
                    print(f"  {key}: {len(value)} 项")
                elif isinstance(value, list):
                    print(f"  {key}: {len(value)} 项 (列表)")
                else:
                    print(f"  {key}: {type(value)}")
    
    print()
    
    # 分析序列数据基本信息
    if files_info.get('seq.jsonl', {}).get('exists'):
        print("3. 序列数据 (seq.jsonl) 基本信息:")
        print("-" * 40)
        seq_data = load_json_safe(files_info['seq.jsonl']['path'], max_lines=100)
        if seq_data:
            print(f"  前100行样本数: {len(seq_data)}")
            if seq_data:
                # 分析第一个样本结构
                sample = seq_data[0]
                print(f"  样本结构示例:")
                if isinstance(sample, list) and len(sample) > 0:
                    first_record = sample[0] if isinstance(sample[0], list) else sample
                    print(f"    记录字段数: {len(first_record) if isinstance(first_record, list) else 'Unknown'}")
                    print(f"    序列长度: {len(sample)}")
    
    print()
    
    # 分析特征数量信息
    for feat_file in ['item_feat_num.json', 'user_feat_num.json']:
        if files_info.get(feat_file, {}).get('exists'):
            print(f"4. 特征数量信息 ({feat_file}):")
            print("-" * 40)
            try:
                with open(files_info[feat_file]['path'], 'r', encoding='utf-8') as f:
                    feat_num = json.load(f)
                    if isinstance(feat_num, dict):
                        print(f"  特征类型数: {len(feat_num)}")
                        for feat_id, num in list(feat_num.items())[:10]:  # 显示前10个
                            print(f"    特征ID {feat_id}: {num} 个不同值")
                        if len(feat_num) > 10:
                            print(f"    ... 还有 {len(feat_num) - 10} 个特征")
            except Exception as e:
                print(f"  错误: {e}")
            print()
    
    # 生成详细统计分析
    print("5. 生成详细统计分析...")
    print("-" * 40)
    
    # 收集所有分析数据
    analysis_data = {
        'file_information': files_info,
        'indexer_summary': {},
        'feature_statistics': {},
        'completeness_analysis': {}
    }
    
    # 文件大小分析
    existing_files = {k: v for k, v in files_info.items() if v.get('exists')}
    if existing_files:
        file_sizes = [existing_files[name]['size_mb'] for name in existing_files.keys()]
        file_size_stats = generate_statistical_summary(file_sizes, "文件大小分布统计 (MB)")
        print(file_size_stats)
        analysis_data['file_size_statistics'] = file_size_stats
    
    # 特征数量详细分析
    feature_analysis = {}
    
    if files_info.get('item_feat_num.json', {}).get('exists'):
        try:
            with open(files_info['item_feat_num.json']['path'], 'r') as f:
                item_feat_num = json.load(f)
                if isinstance(item_feat_num, dict):
                    values = list(item_feat_num.values())
                    item_feat_stats = generate_statistical_summary(values, "物品特征数量分布统计")
                    print(item_feat_stats)
                    feature_analysis['item_features'] = {
                        'total_features': len(item_feat_num),
                        'feature_counts': values,
                        'top_features': sorted(item_feat_num.items(), key=lambda x: x[1], reverse=True)[:20]
                    }
        except Exception as e:
            print(f"分析物品特征数量时出错: {e}")
    
    if files_info.get('user_feat_num.json', {}).get('exists'):
        try:
            with open(files_info['user_feat_num.json']['path'], 'r') as f:
                user_feat_num = json.load(f)
                if isinstance(user_feat_num, dict):
                    values = list(user_feat_num.values())
                    user_feat_stats = generate_statistical_summary(values, "用户特征数量分布统计")
                    print(user_feat_stats)
                    feature_analysis['user_features'] = {
                        'total_features': len(user_feat_num),
                        'feature_counts': values,
                        'top_features': sorted(user_feat_num.items(), key=lambda x: x[1], reverse=True)[:20]
                    }
        except Exception as e:
            print(f"分析用户特征数量时出错: {e}")
    
    analysis_data['feature_statistics'] = feature_analysis
    
    # 数据完整性分析
    completeness = [1 if files_info[f].get('exists') else 0 for f in expected_files]
    completeness_stats = {
        'total_files': len(expected_files),
        'existing_files': sum(completeness),
        'missing_files': len(expected_files) - sum(completeness),
        'completeness_rate': sum(completeness) / len(expected_files) * 100,
        'missing_file_list': [f for f in expected_files if not files_info[f].get('exists')]
    }
    
    print(f"\n数据完整性分析:")
    print(f"  总文件数: {completeness_stats['total_files']}")
    print(f"  存在文件数: {completeness_stats['existing_files']}")
    print(f"  缺失文件数: {completeness_stats['missing_files']}")
    print(f"  完整性: {completeness_stats['completeness_rate']:.1f}%")
    if completeness_stats['missing_file_list']:
        print(f"  缺失文件: {completeness_stats['missing_file_list']}")
    
    analysis_data['completeness_analysis'] = completeness_stats
    
    # 保存详细报告到缓存
    report_path = save_detailed_report("dataset_overview", analysis_data, "数据集总体概述详细分析报告")
    
    print()
    print("6. 分析总结:")
    print("-" * 40)
    total_size = sum([info['size_mb'] for info in files_info.values() if info.get('exists')])
    existing_count = sum([1 for info in files_info.values() if info.get('exists')])
    
    print(f"  数据文件总数: {existing_count}/{len(expected_files)}")
    print(f"  数据集总大小: {total_size:.2f} MB")
    print(f"  数据完整性: {existing_count/len(expected_files)*100:.1f}%")
    
    if indexer:
        print(f"  用户数量: {len(indexer.get('u', {}))}")
        print(f"  物品数量: {len(indexer.get('i', {}))}")
        if 'f' in indexer:
            print(f"  特征字段数: {len(indexer.get('f', {}))}")
        
        # 添加到分析数据中
        analysis_data['indexer_summary'] = {
            'user_count': len(indexer.get('u', {})),
            'item_count': len(indexer.get('i', {})),
            'feature_count': len(indexer.get('f', {})) if 'f' in indexer else 0
        }
    
    # 输出详细报告到日志
    print(f"\n输出详细分析报告到日志...")
    output_report_to_log(report_path)
    
    print("\n数据集概览分析完成！")
    print("=" * 60)

if __name__ == "__main__":
    analyze_dataset_overview()
