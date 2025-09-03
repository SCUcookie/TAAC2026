#!/usr/bin/env python3
"""
腾讯广告算法大赛 - item_feat_num.json 文件详细分析
物品特征数量统计分析，包括特征基数、分布特征、特征重要性评估

Author: Analysis Script  
Date: 2025-01-20
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 导入报告工具
from report_utils import save_detailed_report, output_report_to_log, generate_statistical_summary

# 设置中文字体和样式

def load_item_feat_num(file_path):
    """加载物品特征数量数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            feat_num_data = json.load(f)
        
        print(f"✓ 成功加载物品特征数量数据，包含 {len(feat_num_data)} 个特征")
        return feat_num_data
        
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
        return {}

def analyze_feature_cardinality(feat_num_data):
    """分析特征基数分布"""
    print("\n1. 特征基数(Cardinality)分析:")
    print("-" * 40)
    
    if not feat_num_data:
        print("  无特征数量数据可分析")
        return {}
    
    feature_counts = list(feat_num_data.values())
    feature_ids = list(feat_num_data.keys())
    
    print(f"  特征总数: {len(feature_counts)}")
    print(f"  特征基数统计:")
    print(f"    最小基数: {min(feature_counts):,}")
    print(f"    最大基数: {max(feature_counts):,}")
    print(f"    平均基数: {np.mean(feature_counts):,.2f}")
    print(f"    中位数基数: {np.median(feature_counts):,.2f}")
    print(f"    基数标准差: {np.std(feature_counts):,.2f}")
    
    # 分析基数分布区间
    cardinality_ranges = {
        '极低基数 (1-10)': sum(1 for c in feature_counts if 1 <= c <= 10),
        '低基数 (11-100)': sum(1 for c in feature_counts if 11 <= c <= 100),
        '中等基数 (101-1000)': sum(1 for c in feature_counts if 101 <= c <= 1000),
        '高基数 (1001-10000)': sum(1 for c in feature_counts if 1001 <= c <= 10000),
        '极高基数 (>10000)': sum(1 for c in feature_counts if c > 10000)
    }
    
    print(f"\n  基数分布区间:")
    for range_name, count in cardinality_ranges.items():
        percentage = count / len(feature_counts) * 100
        print(f"    {range_name}: {count} 个特征 ({percentage:.1f}%)")
    
    # 分析分位数
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(feature_counts, percentiles)
    
    print(f"\n  基数分位数:")
    for p, value in zip(percentiles, percentile_values):
        print(f"    {p}%: {value:,.0f}")
    
    # 找出极值特征
    sorted_features = sorted(zip(feature_ids, feature_counts), key=lambda x: x[1])
    
    print(f"\n  极值特征:")
    print(f"    最低基数特征:")
    for feat_id, count in sorted_features[:5]:
        print(f"      特征ID {feat_id}: {count:,} 个不同值")
    
    print(f"    最高基数特征:")
    for feat_id, count in sorted_features[-5:]:
        print(f"      特征ID {feat_id}: {count:,} 个不同值")
    
    return {
        'total_features': len(feature_counts),
        'feature_counts': feature_counts,
        'feature_ids': feature_ids,
        'cardinality_stats': {
            'min': min(feature_counts),
            'max': max(feature_counts),
            'mean': np.mean(feature_counts),
            'median': np.median(feature_counts),
            'std': np.std(feature_counts)
        },
        'cardinality_ranges': cardinality_ranges,
        'percentiles': dict(zip([f'{p}%' for p in percentiles], percentile_values)),
        'lowest_cardinality': sorted_features[:10],
        'highest_cardinality': sorted_features[-10:]
    }

def analyze_feature_categories(cardinality_stats):
    """分析不同类别的特征特点"""
    print("\n2. 特征类别特点分析:")
    print("-" * 40)
    
    feature_counts = cardinality_stats.get('feature_counts', [])
    feature_ids = cardinality_stats.get('feature_ids', [])
    
    if not feature_counts or not feature_ids:
        print("  无数据可分析")
        return {}
    
    # 根据基数将特征分类
    categorical_features = []      # 低基数分类特征
    numerical_features = []        # 高基数数值特征
    identifier_features = []       # 极高基数标识符特征
    
    for feat_id, count in zip(feature_ids, feature_counts):
        if count <= 50:
            categorical_features.append((feat_id, count))
        elif count <= 10000:
            numerical_features.append((feat_id, count))
        else:
            identifier_features.append((feat_id, count))
    
    print(f"  特征分类结果:")
    print(f"    分类特征 (基数≤50): {len(categorical_features)} 个")
    print(f"    数值特征 (50<基数≤10000): {len(numerical_features)} 个")
    print(f"    标识符特征 (基数>10000): {len(identifier_features)} 个")
    
    # 分析各类特征的统计特点
    categories = {
        'categorical': [count for _, count in categorical_features],
        'numerical': [count for _, count in numerical_features],
        'identifier': [count for _, count in identifier_features]
    }
    
    category_stats = {}
    for cat_name, counts in categories.items():
        if counts:
            category_stats[cat_name] = {
                'count': len(counts),
                'mean_cardinality': np.mean(counts),
                'median_cardinality': np.median(counts),
                'max_cardinality': max(counts),
                'min_cardinality': min(counts)
            }
            
            print(f"\n    {cat_name.title()} 特征统计:")
            print(f"      特征数量: {len(counts)}")
            print(f"      平均基数: {np.mean(counts):,.2f}")
            print(f"      中位数基数: {np.median(counts):,.2f}")
            print(f"      基数范围: {min(counts):,} - {max(counts):,}")
    
    # 分析特征ID模式
    print(f"\n  特征ID模式分析:")
    
    # 尝试分析特征ID的数值模式
    numeric_feat_ids = []
    string_feat_ids = []
    
    for feat_id in feature_ids:
        try:
            int(feat_id)
            numeric_feat_ids.append(int(feat_id))
        except:
            string_feat_ids.append(feat_id)
    
    print(f"    数值型特征ID: {len(numeric_feat_ids)} 个")
    print(f"    字符串型特征ID: {len(string_feat_ids)} 个")
    
    if numeric_feat_ids:
        print(f"    数值特征ID范围: {min(numeric_feat_ids)} - {max(numeric_feat_ids)}")
        print(f"    特征ID分布是否连续: {'是' if max(numeric_feat_ids) - min(numeric_feat_ids) + 1 == len(set(numeric_feat_ids)) else '否'}")
    
    return {
        'feature_categories': {
            'categorical': categorical_features,
            'numerical': numerical_features,
            'identifier': identifier_features
        },
        'category_statistics': category_stats,
        'id_patterns': {
            'numeric_ids': numeric_feat_ids[:20],  # 保存前20个
            'string_ids': string_feat_ids[:20],
            'numeric_count': len(numeric_feat_ids),
            'string_count': len(string_feat_ids)
        }
    }

def analyze_feature_importance(cardinality_stats, category_analysis):
    """分析特征重要性和建模建议"""
    print("\n3. 特征重要性和建模建议:")
    print("-" * 40)
    
    feature_counts = cardinality_stats.get('feature_counts', [])
    feature_ids = cardinality_stats.get('feature_ids', [])
    
    if not feature_counts or not feature_ids:
        print("  无数据可分析")
        return {}
    
    # 基于基数的特征重要性评估
    importance_scores = {}
    modeling_suggestions = {}
    
    for feat_id, count in zip(feature_ids, feature_counts):
        # 计算重要性分数（基于信息熵的近似）
        if count <= 1:
            importance_score = 0  # 无变化的特征
            suggestion = "删除 - 无信息量"
        elif count <= 2:
            importance_score = 1  # 二分类特征
            suggestion = "保留 - 二元特征，可直接使用"
        elif count <= 10:
            importance_score = np.log2(count) * 2  # 低基数分类特征
            suggestion = "保留 - 分类特征，适合one-hot编码"
        elif count <= 100:
            importance_score = np.log2(count) * 1.5  # 中等基数特征
            suggestion = "保留 - 可考虑embedding或分组"
        elif count <= 1000:
            importance_score = np.log2(count)  # 高基数特征
            suggestion = "谨慎使用 - 高基数，考虑分组或embedding"
        elif count <= 10000:
            importance_score = np.log2(count) * 0.5  # 极高基数特征
            suggestion = "需要处理 - 极高基数，必须降维"
        else:
            importance_score = np.log2(count) * 0.1  # 标识符特征
            suggestion = "可能删除 - 疑似ID特征，信息量低"
        
        importance_scores[feat_id] = importance_score
        modeling_suggestions[feat_id] = suggestion
    
    # 按重要性排序
    sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  特征重要性评估 (基于信息熵近似):")
    print(f"    高重要性特征 (Top 10):")
    for feat_id, score in sorted_importance[:10]:
        cardinality = dict(zip(feature_ids, feature_counts))[feat_id]
        print(f"      特征ID {feat_id}: 重要性 {score:.2f}, 基数 {cardinality:,}")
    
    print(f"\n    低重要性特征 (Bottom 10):")
    for feat_id, score in sorted_importance[-10:]:
        cardinality = dict(zip(feature_ids, feature_counts))[feat_id]
        print(f"      特征ID {feat_id}: 重要性 {score:.2f}, 基数 {cardinality:,}")
    
    # 建模建议统计
    suggestion_counter = Counter(modeling_suggestions.values())
    
    print(f"\n  建模建议统计:")
    for suggestion, count in suggestion_counter.most_common():
        percentage = count / len(modeling_suggestions) * 100
        print(f"    {suggestion}: {count} 个特征 ({percentage:.1f}%)")
    
    # 特征工程建议
    print(f"\n  特征工程建议:")
    
    categorical_features = category_analysis.get('feature_categories', {}).get('categorical', [])
    numerical_features = category_analysis.get('feature_categories', {}).get('numerical', [])
    identifier_features = category_analysis.get('feature_categories', {}).get('identifier', [])
    
    print(f"    分类特征 ({len(categorical_features)} 个):")
    print(f"      - 推荐使用one-hot编码或target encoding")
    print(f"      - 考虑特征组合和交互")
    
    print(f"    数值特征 ({len(numerical_features)} 个):")
    print(f"      - 考虑embedding方法降维")
    print(f"      - 可尝试特征分箱(binning)")
    
    print(f"    标识符特征 ({len(identifier_features)} 个):")
    print(f"      - 建议删除或转换为统计特征")
    print(f"      - 可提取频次、排序等衍生特征")
    
    return {
        'importance_scores': dict(sorted_importance),
        'modeling_suggestions': modeling_suggestions,
        'suggestion_distribution': dict(suggestion_counter),
        'feature_engineering_recommendations': {
            'categorical_count': len(categorical_features),
            'numerical_count': len(numerical_features),
            'identifier_count': len(identifier_features)
        }
    }

def generate_detailed_analysis(cardinality_stats, category_analysis, importance_analysis):
    """生成详细分析报告（纯文本）"""
    print("\n4. 生成详细分析报告...")
    print("-" * 40)
    
    # 特征基数分布详细分析
    feature_counts = cardinality_stats.get('feature_counts', [])
    if feature_counts:
        cardinality_summary = generate_statistical_summary(feature_counts, "特征基数分布统计")
        print(cardinality_summary)
    
    # 基数分布区间详细分析
    cardinality_ranges = cardinality_stats.get('cardinality_ranges', {})
    if cardinality_ranges:
        print(f"\n特征基数区间分布:")
        print("-" * 40)
        total_features = sum(cardinality_ranges.values())
        
        for range_name, count in cardinality_ranges.items():
            percentage = count / total_features * 100
            print(f"  {range_name}: {count} 个特征 ({percentage:.2f}%)")
    
    # 特征类别分布详细分析
    category_stats = category_analysis.get('category_statistics', {})
    if category_stats:
        print(f"\n特征类别分布详细分析:")
        print("-" * 40)
        
        for cat_name, stats in category_stats.items():
            count = stats['count']
            mean_card = stats['mean_cardinality']
            median_card = stats['median_cardinality']
            max_card = stats['max_cardinality']
            min_card = stats['min_cardinality']
            
            print(f"  {cat_name.title()} 特征:")
            print(f"    数量: {count} 个")
            print(f"    平均基数: {mean_card:,.2f}")
            print(f"    中位数基数: {median_card:,.2f}")
            print(f"    基数范围: {min_card:,} - {max_card:,}")
            print()
    
    # 特征重要性分布分析
    importance_scores = list(importance_analysis.get('importance_scores', {}).values())
    if importance_scores:
        importance_summary = generate_statistical_summary(importance_scores, "特征重要性分布统计")
        print(importance_summary)
    
    # Top高基数特征详细分析
    highest_cardinality = cardinality_stats.get('highest_cardinality', [])
    if highest_cardinality:
        print(f"\nTop 10 高基数特征详细分析:")
        print("-" * 40)
        top_features = highest_cardinality[-10:]  # 最高的10个
        
        for i, (feat_id, count) in enumerate(reversed(top_features), 1):
            print(f"  {i:2d}. 特征ID {feat_id}: {count:,} 个不同值")
    
    # 建模建议分布详细分析
    suggestion_dist = importance_analysis.get('suggestion_distribution', {})
    if suggestion_dist:
        print(f"\n建模建议分布详细分析:")
        print("-" * 40)
        total_suggestions = sum(suggestion_dist.values())
        
        # 按建议类型分组
        suggestion_groups = {
            '保留特征': [],
            '需要处理': [],
            '删除特征': [],
            '其他': []
        }
        
        for suggestion, count in suggestion_dist.items():
            percentage = count / total_suggestions * 100
            if '保留' in suggestion:
                suggestion_groups['保留特征'].append((suggestion, count, percentage))
            elif '删除' in suggestion:
                suggestion_groups['删除特征'].append((suggestion, count, percentage))
            elif '谨慎' in suggestion or '需要处理' in suggestion:
                suggestion_groups['需要处理'].append((suggestion, count, percentage))
            else:
                suggestion_groups['其他'].append((suggestion, count, percentage))
        
        for group_name, suggestions in suggestion_groups.items():
            if suggestions:
                group_total = sum([count for _, count, _ in suggestions])
                group_percentage = group_total / total_suggestions * 100
                print(f"  {group_name} ({group_total} 个特征, {group_percentage:.1f}%):")
                
                for suggestion, count, percentage in suggestions:
                    print(f"    - {suggestion}: {count} 个 ({percentage:.1f}%)")
                print()
    
    # 特征工程建议总结
    feature_engineering_recommendations = importance_analysis.get('feature_engineering_recommendations', {})
    if feature_engineering_recommendations:
        print(f"\n特征工程建议总结:")
        print("-" * 40)
        
        categorical_count = feature_engineering_recommendations.get('categorical_count', 0)
        numerical_count = feature_engineering_recommendations.get('numerical_count', 0)
        identifier_count = feature_engineering_recommendations.get('identifier_count', 0)
        
        print(f"  分类特征 ({categorical_count} 个):")
        print(f"    - 推荐使用 One-hot 编码或 Target encoding")
        print(f"    - 考虑特征组合和交互项")
        print(f"    - 适合作为主要建模特征")
        print()
        
        print(f"  数值特征 ({numerical_count} 个):")
        print(f"    - 考虑使用 Embedding 方法降维") 
        print(f"    - 可尝试特征分箱 (binning)")
        print(f"    - 注意异常值处理")
        print()
        
        print(f"  标识符特征 ({identifier_count} 个):")
        print(f"    - 建议删除或转换为统计特征")
        print(f"    - 可提取频次、排序等衍生特征")
        print(f"    - 避免直接用于建模")
    
    # 返回所有分析数据
    return {
        'cardinality_statistics': cardinality_stats,
        'category_analysis': category_analysis,
        'importance_analysis': importance_analysis,
        'detailed_analysis': {
            'top_high_cardinality': top_features if 'top_features' in locals() else [],
            'suggestion_groups': suggestion_groups if 'suggestion_groups' in locals() else {},
            'feature_engineering_summary': feature_engineering_recommendations
        }
    }

def analyze_item_feat_num():
    """主分析函数"""
    print("=" * 60)
    print("item_feat_num.json 物品特征数量详细分析")
    print("=" * 60)
    
    # 获取数据文件路径
    data_root = os.environ.get('TRAIN_DATA_PATH')
    if not data_root:
        print("错误: 未设置环境变量 TRAIN_DATA_PATH")
        return
    
    feat_num_file = Path(data_root) / 'item_feat_num.json'
    if not feat_num_file.exists():
        print(f"错误: 文件不存在 {feat_num_file}")
        return
    
    print(f"分析文件: {feat_num_file}")
    print(f"文件大小: {feat_num_file.stat().st_size / 1024:.2f} KB")
    print()
    
    # 加载数据
    feat_num_data = load_item_feat_num(feat_num_file)
    
    if not feat_num_data:
        print("无法加载特征数量数据，分析终止")
        return
    
    # 执行各项分析
    cardinality_stats = analyze_feature_cardinality(feat_num_data)
    category_analysis = analyze_feature_categories(cardinality_stats)
    importance_analysis = analyze_feature_importance(cardinality_stats, category_analysis)
    
    # 生成详细分析
    analysis_results = generate_detailed_analysis(cardinality_stats, category_analysis, importance_analysis)
    
    # 保存详细报告
    print("\n5. 保存详细分析报告...")
    print("-" * 40)
    
    # 添加文件信息
    analysis_results['file_info'] = {
        'file_path': str(feat_num_file),
        'file_size_kb': feat_num_file.stat().st_size / 1024
    }
    
    # 保存到缓存并输出到日志
    report_path = save_detailed_report("item_feat_num_analysis", analysis_results, "物品特征数量详细分析报告")
    
    print(f"\n输出详细分析报告到日志...")
    output_report_to_log(report_path)
    
    print("\nitem_feat_num.json 物品特征数量分析完成！")
    print("=" * 60)

if __name__ == "__main__":
    analyze_item_feat_num()
