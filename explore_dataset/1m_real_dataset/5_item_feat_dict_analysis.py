#!/usr/bin/env python3
"""
腾讯广告算法大赛 - item_feat_dict.json 文件详细分析
物品特征字典数据分析，包括特征覆盖度、特征分布、特征质量评估

Author: Analysis Script  
Date: 2025-01-20
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# 导入报告工具
from report_utils import save_detailed_report, output_report_to_log, generate_statistical_summary

# 设置中文字体和样式

def load_item_features(file_path, max_items=None):
    """加载物品特征数据，完整分析所有数据"""
    try:
        print(f"正在加载物品特征数据，分析所有物品...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            item_features = json.load(f)
        
        if isinstance(item_features, dict):
            # 50GB内存足够处理大部分数据集，不再限制
            if max_items and len(item_features) > max_items:
                import random
                sampled_items = random.sample(list(item_features.keys()), max_items)
                item_features = {k: item_features[k] for k in sampled_items}
                print(f"  按要求限制，随机采样 {max_items} 个物品进行分析")
            
            print(f"✓ 成功加载 {len(item_features)} 个物品的特征数据")
            return item_features
        else:
            print(f"错误: 期望字典格式，实际得到 {type(item_features)}")
            return {}
            
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
        return {}

def analyze_item_coverage(item_features):
    """分析物品特征覆盖度"""
    print("\n1. 物品特征覆盖度分析:")
    print("-" * 40)
    
    if not item_features:
        print("  无物品特征数据可分析")
        return {}
    
    total_items = len(item_features)
    print(f"  总物品数: {total_items:,}")
    
    # 统计每个特征的覆盖度
    feature_coverage = defaultdict(int)
    feature_types = defaultdict(set)
    
    for item_id, features in item_features.items():
        if isinstance(features, dict):
            for feat_id, feat_value in features.items():
                if feat_value is not None:
                    feature_coverage[feat_id] += 1
                    feature_types[feat_id].add(type(feat_value).__name__)
    
    # 计算覆盖率
    coverage_stats = {}
    print(f"\n  特征覆盖度统计:")
    
    sorted_features = sorted(feature_coverage.items(), key=lambda x: x[1], reverse=True)
    
    for feat_id, count in sorted_features[:20]:  # 显示前20个特征
        coverage_rate = count / total_items * 100
        types_str = ', '.join(feature_types[feat_id])
        print(f"    特征ID {feat_id}: {count:,}/{total_items:,} = {coverage_rate:.1f}% (类型: {types_str})")
        
        coverage_stats[feat_id] = {
            'count': count,
            'coverage_rate': coverage_rate,
            'types': list(feature_types[feat_id])
        }
    
    if len(sorted_features) > 20:
        print(f"    ... 还有 {len(sorted_features) - 20} 个特征")
    
    # 分析覆盖度分布
    coverage_rates = [count / total_items * 100 for count in feature_coverage.values()]
    
    print(f"\n  覆盖度分布统计:")
    print(f"    平均覆盖度: {np.mean(coverage_rates):.1f}%")
    print(f"    中位数覆盖度: {np.median(coverage_rates):.1f}%")
    print(f"    最高覆盖度: {max(coverage_rates):.1f}%")
    print(f"    最低覆盖度: {min(coverage_rates):.1f}%")
    
    # 分析覆盖度区间
    high_coverage = sum(1 for rate in coverage_rates if rate >= 90)
    medium_coverage = sum(1 for rate in coverage_rates if 50 <= rate < 90)
    low_coverage = sum(1 for rate in coverage_rates if rate < 50)
    
    print(f"    高覆盖度特征 (≥90%): {high_coverage} 个")
    print(f"    中覆盖度特征 (50-90%): {medium_coverage} 个")
    print(f"    低覆盖度特征 (<50%): {low_coverage} 个")
    
    return {
        'total_items': total_items,
        'feature_coverage': dict(coverage_stats),
        'coverage_distribution': {
            'mean': np.mean(coverage_rates),
            'median': np.median(coverage_rates),
            'max': max(coverage_rates),
            'min': min(coverage_rates),
            'high_coverage_count': high_coverage,
            'medium_coverage_count': medium_coverage,
            'low_coverage_count': low_coverage
        },
        'all_coverage_rates': coverage_rates
    }

def analyze_feature_distribution(item_features):
    """分析特征值分布"""
    print("\n2. 特征值分布分析:")
    print("-" * 40)
    
    feature_value_dist = defaultdict(Counter)
    feature_stats = {}
    
    # 收集每个特征的值分布
    for item_id, features in list(item_features.items())[:5000]:  # 分析前5000个物品
        if isinstance(features, dict):
            for feat_id, feat_value in features.items():
                if feat_value is not None:
                    feature_value_dist[feat_id][feat_value] += 1
    
    print(f"  分析了 {min(5000, len(item_features))} 个物品的特征分布")
    print(f"\n  特征值分布统计:")
    
    # 分析每个特征的分布特征
    for feat_id, value_counter in list(feature_value_dist.items())[:15]:  # 分析前15个特征
        unique_values = len(value_counter)
        total_count = sum(value_counter.values())
        most_common = value_counter.most_common(3)
        
        print(f"\n    特征ID {feat_id}:")
        print(f"      不同值数量: {unique_values}")
        print(f"      总出现次数: {total_count}")
        
        if most_common:
            print(f"      最高频值:")
            for value, count in most_common:
                percentage = count / total_count * 100
                print(f"        值 '{value}': {count} 次 ({percentage:.1f}%)")
        
        # 计算基尼系数（衡量分布均匀性）
        if len(value_counter) > 1:
            values = list(value_counter.values())
            values.sort()
            n = len(values)
            cumsum = np.cumsum(values)
            gini = (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(values))) / (n * sum(values))
            print(f"      分布均匀性 (基尼系数): {gini:.3f} (越小越均匀)")
        
        feature_stats[feat_id] = {
            'unique_values': unique_values,
            'total_count': total_count,
            'most_common': dict(most_common),
            'gini_coefficient': gini if 'gini' in locals() else None
        }
    
    return feature_stats

def analyze_feature_quality(item_features, coverage_stats):
    """分析特征质量"""
    print("\n3. 特征质量分析:")
    print("-" * 40)
    
    quality_metrics = {}
    
    # 分析缺失值模式
    print(f"  特征缺失值模式分析:")
    
    missing_patterns = defaultdict(int)
    feature_missing_rates = {}
    
    for item_id, features in list(item_features.items())[:2000]:  # 分析前2000个物品
        if isinstance(features, dict):
            missing_features = []
            for feat_id in coverage_stats.get('feature_coverage', {}):
                if feat_id not in features or features[feat_id] is None:
                    missing_features.append(feat_id)
            
            missing_pattern = tuple(sorted(missing_features))
            missing_patterns[missing_pattern] += 1
    
    # 计算每个特征的缺失率
    total_analyzed = min(2000, len(item_features))
    for feat_id in coverage_stats.get('feature_coverage', {}):
        missing_count = 0
        for item_id, features in list(item_features.items())[:total_analyzed]:
            if isinstance(features, dict):
                if feat_id not in features or features[feat_id] is None:
                    missing_count += 1
        
        missing_rate = missing_count / total_analyzed * 100
        feature_missing_rates[feat_id] = missing_rate
    
    # 显示缺失率最高的特征
    sorted_missing = sorted(feature_missing_rates.items(), key=lambda x: x[1], reverse=True)
    print(f"    缺失率最高的特征:")
    for feat_id, missing_rate in sorted_missing[:10]:
        print(f"      特征ID {feat_id}: {missing_rate:.1f}% 缺失")
    
    # 分析数据类型一致性
    print(f"\n  数据类型一致性分析:")
    
    type_consistency = {}
    for feat_id in list(coverage_stats.get('feature_coverage', {}).keys())[:10]:
        types_found = defaultdict(int)
        
        for item_id, features in list(item_features.items())[:1000]:
            if isinstance(features, dict) and feat_id in features:
                if features[feat_id] is not None:
                    types_found[type(features[feat_id]).__name__] += 1
        
        total_non_null = sum(types_found.values())
        if total_non_null > 0:
            dominant_type = max(types_found.items(), key=lambda x: x[1])
            consistency_rate = dominant_type[1] / total_non_null * 100
            
            print(f"    特征ID {feat_id}:")
            print(f"      主要类型: {dominant_type[0]} ({consistency_rate:.1f}%)")
            if len(types_found) > 1:
                print(f"      其他类型: {dict(types_found)}")
            
            type_consistency[feat_id] = {
                'dominant_type': dominant_type[0],
                'consistency_rate': consistency_rate,
                'all_types': dict(types_found)
            }
    
    quality_metrics = {
        'feature_missing_rates': feature_missing_rates,
        'type_consistency': type_consistency,
        'total_analyzed_items': total_analyzed
    }
    
    return quality_metrics

def analyze_click_relevant_features(item_features, coverage_stats):
    """分析与点击行为相关的特征"""
    print("\n4. 点击行为相关特征分析:")
    print("-" * 40)
    
    click_relevant_analysis = {}
    
    # 分析可能影响点击的特征
    feature_coverage = coverage_stats.get('feature_coverage', {})
    
    # 根据特征覆盖度和类型推断点击相关性
    high_impact_features = []  # 高覆盖度，可能影响点击
    quality_features = []      # 内容质量相关特征
    category_features = []     # 分类相关特征
    
    print(f"  分析 {len(feature_coverage)} 个特征的点击相关性...")
    
    for feat_id, feat_info in feature_coverage.items():
        coverage_rate = feat_info['coverage_rate']
        feat_types = feat_info['types']
        
        # 基于特征ID和覆盖度推断特征类型
        if coverage_rate > 95:  # 高覆盖度特征
            if feat_id in ['111', '121']:  # 可能是ID特征
                continue  # 跳过ID特征
            elif feat_id in ['100', '114', '116', '117']:  # 可能是分类特征
                category_features.append((feat_id, coverage_rate, feat_info))
            else:
                high_impact_features.append((feat_id, coverage_rate, feat_info))
        elif 30 < coverage_rate <= 95:  # 中等覆盖度，可能是质量特征
            quality_features.append((feat_id, coverage_rate, feat_info))
    
    print(f"  高影响力特征 (覆盖度>95%): {len(high_impact_features)} 个")
    print(f"  内容质量特征 (覆盖度30-95%): {len(quality_features)} 个")
    print(f"  分类特征: {len(category_features)} 个")
    
    # 分析每类特征的点击预测价值
    for feat_category, feat_list, description in [
        ("高影响力特征", high_impact_features, "预计对点击率有显著影响"),
        ("内容质量特征", quality_features, "可能影响用户点击决策"),
        ("分类特征", category_features, "用于广告定向和分类")
    ]:
        if feat_list:
            print(f"\n  {feat_category} ({description}):")
            for feat_id, coverage_rate, feat_info in feat_list[:5]:
                feature_types = ', '.join(feat_info['types'])
                print(f"    特征ID {feat_id}: 覆盖率 {coverage_rate:.1f}%, 类型 {feature_types}")
    
    # 分析特征的点击预测潜力
    print(f"\n  点击预测特征建议:")
    
    prediction_features = {
        'primary_features': [],      # 主要预测特征
        'auxiliary_features': [],    # 辅助特征
        'contextual_features': []    # 上下文特征
    }
    
    # 基于覆盖度和特征ID分类
    for feat_id, feat_info in feature_coverage.items():
        coverage_rate = feat_info['coverage_rate']
        
        if feat_id in ['111', '121', '102', '122']:  # ID特征，跳过
            continue
        elif coverage_rate > 98 and feat_id in ['100', '114', '116']:  # 分类特征
            prediction_features['primary_features'].append((feat_id, coverage_rate))
        elif coverage_rate > 90:  # 高覆盖度特征
            prediction_features['auxiliary_features'].append((feat_id, coverage_rate))
        elif 30 < coverage_rate <= 90:  # 中等覆盖度特征
            prediction_features['contextual_features'].append((feat_id, coverage_rate))
    
    for category, features in prediction_features.items():
        if features:
            category_name = {
                'primary_features': '主要预测特征',
                'auxiliary_features': '辅助预测特征', 
                'contextual_features': '上下文特征'
            }[category]
            print(f"    {category_name}: {[f[0] for f in features[:5]]}")
    
    # 特征工程建议
    print(f"\n  特征工程建议:")
    engineering_suggestions = []
    
    if len(prediction_features['primary_features']) < 3:
        engineering_suggestions.append("增加主要预测特征 - 当前数量不足")
    
    if len(category_features) > 0:
        engineering_suggestions.append("分类特征适合one-hot编码")
    
    if len(quality_features) > 0:
        engineering_suggestions.append("质量特征可能需要缺失值处理")
    
    high_coverage_count = len([f for f in feature_coverage.values() if f['coverage_rate'] > 95])
    if high_coverage_count > 10:
        engineering_suggestions.append("高覆盖度特征丰富，适合深度学习模型")
    
    for suggestion in engineering_suggestions:
        print(f"    - {suggestion}")
    
    click_relevant_analysis = {
        'high_impact_features': high_impact_features,
        'quality_features': quality_features,
        'category_features': category_features,
        'prediction_features': prediction_features,
        'engineering_suggestions': engineering_suggestions,
        'feature_counts': {
            'high_impact': len(high_impact_features),
            'quality': len(quality_features), 
            'category': len(category_features)
        }
    }
    
    return click_relevant_analysis

def generate_detailed_analysis(coverage_stats, feature_dist_stats, quality_metrics):
    """生成详细分析报告（纯文本）"""
    print("\n4. 生成详细分析报告...")
    print("-" * 40)
    
    # 特征覆盖度详细分析
    coverage_rates = coverage_stats.get('all_coverage_rates', [])
    if coverage_rates:
        coverage_summary = generate_statistical_summary(coverage_rates, "特征覆盖度分布统计")
        print(coverage_summary)
    
    # Top特征覆盖度分析
    feature_coverage = coverage_stats.get('feature_coverage', {})
    if feature_coverage:
        print(f"\nTop 15 特征覆盖度分析:")
        print("-" * 40)
        top_features = sorted(feature_coverage.items(), key=lambda x: x[1]['coverage_rate'], reverse=True)[:15]
        
        for i, (feat_id, feat_info) in enumerate(top_features, 1):
            coverage_rate = feat_info['coverage_rate']
            count = feat_info['count']
            types = ', '.join(feat_info['types'])
            print(f"  {i:2d}. 特征ID {feat_id}: {coverage_rate:.2f}% ({count:,} 项) [类型: {types}]")
    
    # 覆盖度区间分布
    coverage_dist = coverage_stats.get('coverage_distribution', {})
    if coverage_dist:
        print(f"\n特征覆盖度区间分布:")
        print("-" * 40)
        high_count = coverage_dist.get('high_coverage_count', 0)
        medium_count = coverage_dist.get('medium_coverage_count', 0)
        low_count = coverage_dist.get('low_coverage_count', 0)
        total = high_count + medium_count + low_count
        
        print(f"  高覆盖度特征 (≥90%): {high_count} 个 ({high_count/total*100:.1f}%)")
        print(f"  中覆盖度特征 (50-90%): {medium_count} 个 ({medium_count/total*100:.1f}%)")
        print(f"  低覆盖度特征 (<50%): {low_count} 个 ({low_count/total*100:.1f}%)")
    
    # 特征缺失率详细分析
    missing_rates = list(quality_metrics.get('feature_missing_rates', {}).values())
    if missing_rates:
        missing_summary = generate_statistical_summary(missing_rates, "特征缺失率分布统计")
        print(missing_summary)
        
        # Top 10 高缺失率特征
        if quality_metrics.get('feature_missing_rates'):
            print(f"\nTop 10 高缺失率特征:")
            print("-" * 40)
            top_missing = sorted(quality_metrics['feature_missing_rates'].items(), 
                               key=lambda x: x[1], reverse=True)[:10]
            
            for i, (feat_id, missing_rate) in enumerate(top_missing, 1):
                print(f"  {i:2d}. 特征ID {feat_id}: {missing_rate:.2f}% 缺失")
    
    # 特征类型一致性分析
    type_consistency = quality_metrics.get('type_consistency', {})
    if type_consistency:
        consistency_rates = [info['consistency_rate'] for info in type_consistency.values()]
        consistency_summary = generate_statistical_summary(consistency_rates, "特征类型一致性分布统计")
        print(consistency_summary)
        
        print(f"\n特征类型一致性详细分析:")
        print("-" * 40)
        for feat_id, info in list(type_consistency.items())[:10]:
            dominant_type = info['dominant_type']
            consistency_rate = info['consistency_rate']
            all_types = info['all_types']
            print(f"  特征ID {feat_id}: {dominant_type} ({consistency_rate:.1f}%) - 全部类型: {all_types}")
    
    # 返回所有分析数据
    return {
        'coverage_statistics': coverage_stats,
        'feature_distribution_statistics': feature_dist_stats,
        'quality_metrics': quality_metrics,
        'detailed_analysis': {
            'top_coverage_features': top_features if 'top_features' in locals() else [],
            'top_missing_features': top_missing if 'top_missing' in locals() else [],
            'consistency_analysis': dict(list(type_consistency.items())[:20]) if type_consistency else {}
        }
    }

def analyze_item_feat_dict():
    """主分析函数"""
    print("=" * 60)
    print("item_feat_dict.json 物品特征字典详细分析")
    print("=" * 60)
    
    # 获取数据文件路径
    data_root = os.environ.get('TRAIN_DATA_PATH')
    if not data_root:
        print("错误: 未设置环境变量 TRAIN_DATA_PATH")
        return
    
    item_feat_file = Path(data_root) / 'item_feat_dict.json'
    if not item_feat_file.exists():
        print(f"错误: 文件不存在 {item_feat_file}")
        return
    
    print(f"分析文件: {item_feat_file}")
    print(f"文件大小: {item_feat_file.stat().st_size / (1024*1024):.2f} MB")
    print()
    
    # 加载数据
    item_features = load_item_features(item_feat_file)
    
    if not item_features:
        print("无法加载物品特征数据，分析终止")
        return
    
    # 执行各项分析
    coverage_stats = analyze_item_coverage(item_features)
    feature_dist_stats = analyze_feature_distribution(item_features)
    quality_metrics = analyze_feature_quality(item_features, coverage_stats)
    
    # 新增：点击行为相关特征分析
    click_relevant_stats = analyze_click_relevant_features(item_features, coverage_stats)
    
    # 生成详细分析
    analysis_results = generate_detailed_analysis(coverage_stats, feature_dist_stats, quality_metrics)
    
    # 保存详细报告
    print("\n5. 保存详细分析报告...")
    print("-" * 40)
    
    # 添加文件信息和点击相关分析
    analysis_results['file_info'] = {
        'file_path': str(item_feat_file),
        'file_size_mb': item_feat_file.stat().st_size / (1024*1024),
        'analyzed_items': len(item_features)
    }
    analysis_results['click_relevant_analysis'] = click_relevant_stats
    
    # 保存到缓存并输出到日志
    report_path = save_detailed_report("item_feat_dict_analysis", analysis_results, "物品特征字典详细分析报告")
    
    print(f"\n输出详细分析报告到日志...")
    output_report_to_log(report_path)
    
    print("\nitem_feat_dict.json 物品特征字典分析完成！")
    print("=" * 60)

if __name__ == "__main__":
    analyze_item_feat_dict()
