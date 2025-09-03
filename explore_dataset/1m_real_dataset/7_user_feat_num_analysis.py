#!/usr/bin/env python3
"""
腾讯广告算法大赛 - user_feat_num.json 文件详细分析
用户特征数量统计分析，包括用户特征基数、分布特征、隐私保护评估

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

def load_user_feat_num(file_path):
    """加载用户特征数量数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            feat_num_data = json.load(f)
        
        print(f"✓ 成功加载用户特征数量数据，包含 {len(feat_num_data)} 个特征")
        return feat_num_data
        
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
        return {}

def analyze_user_feature_cardinality(feat_num_data):
    """分析用户特征基数分布"""
    print("\n1. 用户特征基数(Cardinality)分析:")
    print("-" * 40)
    
    if not feat_num_data:
        print("  无用户特征数量数据可分析")
        return {}
    
    feature_counts = list(feat_num_data.values())
    feature_ids = list(feat_num_data.keys())
    
    print(f"  用户特征总数: {len(feature_counts)}")
    print(f"  特征基数统计:")
    print(f"    最小基数: {min(feature_counts):,}")
    print(f"    最大基数: {max(feature_counts):,}")
    print(f"    平均基数: {np.mean(feature_counts):,.2f}")
    print(f"    中位数基数: {np.median(feature_counts):,.2f}")
    print(f"    基数标准差: {np.std(feature_counts):,.2f}")
    
    # 分析用户特征的特殊性（相比物品特征）
    print(f"\n  用户特征特殊性分析:")
    
    # 基数分布区间（针对用户特征调整）
    cardinality_ranges = {
        '二元特征 (2)': sum(1 for c in feature_counts if c == 2),
        '小分类 (3-10)': sum(1 for c in feature_counts if 3 <= c <= 10),
        '中分类 (11-50)': sum(1 for c in feature_counts if 11 <= c <= 50),
        '大分类 (51-500)': sum(1 for c in feature_counts if 51 <= c <= 500),
        '连续化 (501-5000)': sum(1 for c in feature_counts if 501 <= c <= 5000),
        '高基数 (>5000)': sum(1 for c in feature_counts if c > 5000)
    }
    
    print(f"  基数分布区间:")
    for range_name, count in cardinality_ranges.items():
        percentage = count / len(feature_counts) * 100
        print(f"    {range_name}: {count} 个特征 ({percentage:.1f}%)")
    
    # 分析分位数
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(feature_counts, percentiles)
    
    print(f"\n  基数分位数:")
    for p, value in zip(percentiles, percentile_values):
        print(f"    {p}%: {value:,.0f}")
    
    # 找出典型用户特征
    sorted_features = sorted(zip(feature_ids, feature_counts), key=lambda x: x[1])
    
    print(f"\n  典型用户特征示例:")
    print(f"    最低基数特征 (可能的二元特征):")
    for feat_id, count in sorted_features[:5]:
        print(f"      特征ID {feat_id}: {count:,} 个不同值")
    
    print(f"    最高基数特征 (可能的连续/ID特征):")
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

def analyze_user_feature_types(cardinality_stats):
    """分析用户特征类型和隐私特征"""
    print("\n2. 用户特征类型和隐私特征分析:")
    print("-" * 40)
    
    feature_counts = cardinality_stats.get('feature_counts', [])
    feature_ids = cardinality_stats.get('feature_ids', [])
    
    if not feature_counts or not feature_ids:
        print("  无数据可分析")
        return {}
    
    # 根据基数推断用户特征类型
    demographic_features = []      # 人口统计特征
    behavioral_features = []       # 行为特征  
    preference_features = []       # 偏好特征
    identity_features = []         # 身份标识特征
    
    for feat_id, count in zip(feature_ids, feature_counts):
        if count == 2:
            demographic_features.append((feat_id, count))  # 二元特征，如性别
        elif count <= 10:
            demographic_features.append((feat_id, count))  # 小分类，如年龄段
        elif count <= 100:
            preference_features.append((feat_id, count))   # 偏好分类
        elif count <= 1000:
            behavioral_features.append((feat_id, count))   # 行为统计
        else:
            identity_features.append((feat_id, count))     # 可能的标识符
    
    print(f"  推断的用户特征类型:")
    print(f"    人口统计特征 (基数≤10): {len(demographic_features)} 个")
    print(f"      - 可能包括: 性别、年龄段、地区等")
    print(f"      - 隐私敏感度: 中等")
    
    print(f"    偏好特征 (10<基数≤100): {len(preference_features)} 个")
    print(f"      - 可能包括: 兴趣标签、应用偏好等")
    print(f"      - 隐私敏感度: 低")
    
    print(f"    行为特征 (100<基数≤1000): {len(behavioral_features)} 个")
    print(f"      - 可能包括: 点击频次、活跃度等")
    print(f"      - 隐私敏感度: 低")
    
    print(f"    身份标识特征 (基数>1000): {len(identity_features)} 个")
    print(f"      - 可能包括: 用户ID、设备ID等")
    print(f"      - 隐私敏感度: 高")
    
    # 隐私风险评估
    print(f"\n  隐私风险评估:")
    
    total_features = len(feature_counts)
    high_risk = len(identity_features)
    medium_risk = len(demographic_features)
    low_risk = len(preference_features) + len(behavioral_features)
    
    print(f"    高风险特征: {high_risk} 个 ({high_risk/total_features*100:.1f}%)")
    print(f"    中等风险特征: {medium_risk} 个 ({medium_risk/total_features*100:.1f}%)")
    print(f"    低风险特征: {low_risk} 个 ({low_risk/total_features*100:.1f}%)")
    
    # 分析特征ID模式
    print(f"\n  特征ID模式分析:")
    
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
        id_gaps = sorted(set(range(min(numeric_feat_ids), max(numeric_feat_ids) + 1)) - set(numeric_feat_ids))
        print(f"    缺失的特征ID: {len(id_gaps)} 个" + (f" (如: {id_gaps[:5]}...)" if id_gaps else ""))
    
    return {
        'feature_types': {
            'demographic': demographic_features,
            'preference': preference_features,
            'behavioral': behavioral_features,
            'identity': identity_features
        },
        'privacy_assessment': {
            'high_risk_count': high_risk,
            'medium_risk_count': medium_risk,
            'low_risk_count': low_risk,
            'risk_percentages': {
                'high': high_risk/total_features*100,
                'medium': medium_risk/total_features*100,
                'low': low_risk/total_features*100
            }
        },
        'id_patterns': {
            'numeric_ids': numeric_feat_ids[:20],
            'string_ids': string_feat_ids[:20],
            'numeric_count': len(numeric_feat_ids),
            'string_count': len(string_feat_ids),
            'missing_ids': id_gaps[:10] if 'id_gaps' in locals() else []
        }
    }

def analyze_modeling_recommendations(cardinality_stats, type_analysis):
    """分析建模建议和特征工程策略"""
    print("\n3. 建模建议和特征工程策略:")
    print("-" * 40)
    
    feature_counts = cardinality_stats.get('feature_counts', [])
    feature_ids = cardinality_stats.get('feature_ids', [])
    
    if not feature_counts or not feature_ids:
        print("  无数据可分析")
        return {}
    
    # 基于用户特征特点的建模建议
    modeling_strategies = {}
    privacy_concerns = {}
    
    for feat_id, count in zip(feature_ids, feature_counts):
        if count <= 2:
            strategy = "直接使用 - 二元特征，无需编码"
            privacy = "低风险"
        elif count <= 10:
            strategy = "One-hot编码 - 小分类特征"
            privacy = "中风险 - 可能涉及人口统计"
        elif count <= 50:
            strategy = "Label编码或Embedding - 中等分类"
            privacy = "低风险"
        elif count <= 200:
            strategy = "Target编码或Embedding - 大分类"
            privacy = "低风险"
        elif count <= 1000:
            strategy = "统计特征 - 转换为频次/分布"
            privacy = "低风险"
        else:
            strategy = "谨慎处理 - 可能为ID，考虑删除或哈希"
            privacy = "高风险 - 可能为身份标识"
        
        modeling_strategies[feat_id] = strategy
        privacy_concerns[feat_id] = privacy
    
    # 统计建议分布
    strategy_counter = Counter(modeling_strategies.values())
    privacy_counter = Counter(privacy_concerns.values())
    
    print(f"  建模策略分布:")
    for strategy, count in strategy_counter.most_common():
        percentage = count / len(modeling_strategies) * 100
        print(f"    {strategy}: {count} 个特征 ({percentage:.1f}%)")
    
    print(f"\n  隐私关注度分布:")
    for concern, count in privacy_counter.most_common():
        percentage = count / len(privacy_concerns) * 100
        print(f"    {concern}: {count} 个特征 ({percentage:.1f}%)")
    
    # 用户特征专门建议
    print(f"\n  用户特征专门建议:")
    
    demographic_count = len(type_analysis.get('feature_types', {}).get('demographic', []))
    preference_count = len(type_analysis.get('feature_types', {}).get('preference', []))
    behavioral_count = len(type_analysis.get('feature_types', {}).get('behavioral', []))
    identity_count = len(type_analysis.get('feature_types', {}).get('identity', []))
    
    print(f"    人口统计特征 ({demographic_count} 个):")
    print(f"      - 特征工程: 组合编码，如年龄×性别")
    print(f"      - 隐私保护: 考虑分桶处理，避免精确值")
    print(f"      - 建模价值: 高 - 强预测能力")
    
    print(f"    偏好特征 ({preference_count} 个):")
    print(f"      - 特征工程: Embedding学习用户兴趣向量")
    print(f"      - 隐私保护: 相对安全")
    print(f"      - 建模价值: 高 - 直接反映偏好")
    
    print(f"    行为特征 ({behavioral_count} 个):")
    print(f"      - 特征工程: 时间窗口统计，趋势分析")
    print(f"      - 隐私保护: 聚合处理，避免个体追踪")
    print(f"      - 建模价值: 中高 - 反映行为模式")
    
    print(f"    身份标识特征 ({identity_count} 个):")
    print(f"      - 特征工程: 转换为频次特征或删除")
    print(f"      - 隐私保护: 严格限制，考虑哈希处理")
    print(f"      - 建模价值: 低 - 避免过拟合")
    
    # 比较用户特征与物品特征的差异
    print(f"\n  用户vs物品特征对比:")
    print(f"    用户特征更注重:")
    print(f"      - 隐私保护和合规性")
    print(f"      - 跨时间的稳定性")
    print(f"      - 人口学和心理学解释性")
    print(f"    建议的差异化处理:")
    print(f"      - 更保守的特征选择")
    print(f"      - 更多的聚合和泛化")
    print(f"      - 更严格的A/B测试")
    
    return {
        'modeling_strategies': modeling_strategies,
        'privacy_concerns': privacy_concerns,
        'strategy_distribution': dict(strategy_counter),
        'privacy_distribution': dict(privacy_counter),
        'specialized_recommendations': {
            'demographic_count': demographic_count,
            'preference_count': preference_count,
            'behavioral_count': behavioral_count,
            'identity_count': identity_count
        }
    }

def analyze_user_feat_num():
    """主分析函数"""
    print("=" * 60)
    print("user_feat_num.json 用户特征数量详细分析")
    print("=" * 60)
    
    # 获取数据文件路径
    data_root = os.environ.get('TRAIN_DATA_PATH')
    if not data_root:
        print("错误: 未设置环境变量 TRAIN_DATA_PATH")
        return
    
    user_feat_file = Path(data_root) / 'user_feat_num.json'
    if not user_feat_file.exists():
        print(f"错误: 文件不存在 {user_feat_file}")
        return
    
    print(f"分析文件: {user_feat_file}")
    print(f"文件大小: {user_feat_file.stat().st_size / 1024:.2f} KB")
    print()
    
    # 加载数据
    feat_num_data = load_user_feat_num(user_feat_file)
    
    if not feat_num_data:
        print("无法加载用户特征数量数据，分析终止")
        return
    
    # 执行各项分析
    cardinality_stats = analyze_user_feature_cardinality(feat_num_data)
    type_analysis = analyze_user_feature_types(cardinality_stats)
    modeling_analysis = analyze_modeling_recommendations(cardinality_stats, type_analysis)
    
    # 生成可视化
    analysis_results = generate_detailed_analysis(cardinality_stats, type_analysis, modeling_analysis)
    
    # 保存分析报告
    print("\n5. 保存分析报告...")
    print("-" * 40)
    
    # 添加文件信息
    analysis_results['file_info'] = {
        'file_path': str(user_feat_file),
        'file_size_kb': user_feat_file.stat().st_size / 1024
    }
    
    # 保存到缓存并输出到日志
    report_path = save_detailed_report("user_feat_num_analysis", analysis_results, "用户特征数量详细分析报告")
    
    print(f"\n输出详细分析报告到日志...")
    output_report_to_log(report_path)
    
    print("\nuser_feat_num.json 用户特征数量分析完成！")
    print("=" * 60)

def generate_detailed_analysis(cardinality_stats, type_analysis, modeling_analysis):
    """生成详细分析报告（纯文本）"""
    print("\n4. 生成详细分析报告...")
    print("-" * 40)
    
    # 用户特征基数分布详细分析
    feature_counts = cardinality_stats.get('feature_counts', [])
    if feature_counts:
        cardinality_summary = generate_statistical_summary(feature_counts, "用户特征基数分布统计")
        print(cardinality_summary)
    
    # 基数分布区间详细分析
    cardinality_ranges = cardinality_stats.get('cardinality_ranges', {})
    if cardinality_ranges:
        print(f"\n用户特征基数区间分布:")
        print("-" * 40)
        total_features = sum(cardinality_ranges.values())
        
        for range_name, count in cardinality_ranges.items():
            percentage = count / total_features * 100
            print(f"  {range_name}: {count} 个特征 ({percentage:.2f}%)")
    
    # 用户特征类型分布详细分析
    feature_types = type_analysis.get('feature_types', {})
    if feature_types:
        print(f"\n用户特征类型分布详细分析:")
        print("-" * 40)
        
        demographic_count = len(feature_types.get('demographic', []))
        preference_count = len(feature_types.get('preference', []))
        behavioral_count = len(feature_types.get('behavioral', []))
        identity_count = len(feature_types.get('identity', []))
        
        total_analyzed = demographic_count + preference_count + behavioral_count + identity_count
        
        print(f"  人口统计特征: {demographic_count} 个 ({demographic_count/total_analyzed*100:.1f}%)")
        print(f"  偏好特征: {preference_count} 个 ({preference_count/total_analyzed*100:.1f}%)")
        print(f"  行为特征: {behavioral_count} 个 ({behavioral_count/total_analyzed*100:.1f}%)")
        print(f"  身份标识特征: {identity_count} 个 ({identity_count/total_analyzed*100:.1f}%)")
    
    # 隐私风险评估详细分析
    privacy_assessment = type_analysis.get('privacy_assessment', {})
    if privacy_assessment:
        print(f"\n隐私风险评估详细分析:")
        print("-" * 40)
        
        high_risk = privacy_assessment.get('high_risk_count', 0)
        medium_risk = privacy_assessment.get('medium_risk_count', 0)
        low_risk = privacy_assessment.get('low_risk_count', 0)
        total_risk_analyzed = high_risk + medium_risk + low_risk
        
        print(f"  高风险特征: {high_risk} 个 ({high_risk/total_risk_analyzed*100:.1f}%)")
        print(f"  中等风险特征: {medium_risk} 个 ({medium_risk/total_risk_analyzed*100:.1f}%)")
        print(f"  低风险特征: {low_risk} 个 ({low_risk/total_risk_analyzed*100:.1f}%)")
        
        # 风险评估建议
        if high_risk / total_risk_analyzed > 0.3:
            print(f"  风险评估: 需要高度关注 - 高风险特征占比超过30%")
        elif high_risk / total_risk_analyzed > 0.1:
            print(f"  风险评估: 需要适度关注 - 高风险特征占比10-30%")
        else:
            print(f"  风险评估: 风险可控 - 高风险特征占比低于10%")
    
    # 建模策略分布详细分析
    strategy_dist = modeling_analysis.get('strategy_distribution', {})
    if strategy_dist:
        print(f"\n建模策略分布详细分析:")
        print("-" * 40)
        total_strategies = sum(strategy_dist.values())
        
        # 按策略类型分组统计
        strategy_groups = {
            '可直接使用': 0,
            '需要编码': 0,
            '需要处理': 0,
            '建议删除': 0
        }
        
        for strategy, count in strategy_dist.items():
            percentage = count / total_strategies * 100
            print(f"  {strategy}: {count} 个特征 ({percentage:.1f}%)")
            
            # 分组统计
            if '直接使用' in strategy:
                strategy_groups['可直接使用'] += count
            elif 'One-hot' in strategy or 'Label' in strategy or 'Target' in strategy or 'Embedding' in strategy:
                strategy_groups['需要编码'] += count
            elif '统计特征' in strategy or '谨慎处理' in strategy:
                strategy_groups['需要处理'] += count
            elif '删除' in strategy:
                strategy_groups['建议删除'] += count
        
        print(f"\n策略分组统计:")
        for group_name, count in strategy_groups.items():
            if count > 0:
                percentage = count / total_strategies * 100
                print(f"  {group_name}: {count} 个特征 ({percentage:.1f}%)")
    
    # 特征工程总体建议
    print(f"\n特征工程总体建议:")
    print("-" * 40)
    specialized_recommendations = modeling_analysis.get('specialized_recommendations', {})
    
    if specialized_recommendations:
        demographic_count = specialized_recommendations.get('demographic_count', 0)
        preference_count = specialized_recommendations.get('preference_count', 0)
        behavioral_count = specialized_recommendations.get('behavioral_count', 0)
        identity_count = specialized_recommendations.get('identity_count', 0)
        
        print(f"  1. 优先使用人口统计特征 ({demographic_count} 个) - 预测能力强")
        print(f"  2. 重点关注偏好特征 ({preference_count} 个) - 直接反映兴趣")
        print(f"  3. 适度使用行为特征 ({behavioral_count} 个) - 反映使用模式")
        print(f"  4. 谨慎处理身份标识特征 ({identity_count} 个) - 隐私风险高")
        
        # 总体建议
        total_useful = demographic_count + preference_count + behavioral_count
        if total_useful > identity_count * 3:
            print(f"  总体评估: 特征质量良好，可用特征远多于风险特征")
        elif total_useful > identity_count:
            print(f"  总体评估: 特征质量中等，需要仔细筛选")
        else:
            print(f"  总体评估: 需要谨慎处理，风险特征占比较高")
    
    # 返回所有分析数据
    return {
        'cardinality_statistics': cardinality_stats,
        'type_analysis': type_analysis,
        'modeling_analysis': modeling_analysis,
        'detailed_analysis': {
            'strategy_groups': strategy_groups if 'strategy_groups' in locals() else {},
            'risk_assessment': privacy_assessment,
            'feature_engineering_summary': specialized_recommendations
        }
    }


if __name__ == "__main__":
    analyze_user_feat_num()
