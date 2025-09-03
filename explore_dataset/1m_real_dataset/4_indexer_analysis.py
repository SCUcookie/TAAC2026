#!/usr/bin/env python3
"""
腾讯广告算法大赛 - indexer.pkl 文件详细分析
索引映射数据分析，包括用户、物品、特征的ID映射关系

Author: Analysis Script  
Date: 2025-01-20
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

# 导入报告工具
from report_utils import save_detailed_report, output_report_to_log, generate_statistical_summary, ReportWriter

# 设置中文字体和样式

def load_indexer_data(file_path):
    """加载索引器数据"""
    try:
        with open(file_path, 'rb') as f:
            indexer = pickle.load(f)
        print(f"✓ 成功加载索引器数据")
        return indexer
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
        return None

def analyze_indexer_structure(indexer, writer):
    """分析索引器数据结构"""
    writer.write_section("1. 索引器数据结构分析", level=1)
    
    if not indexer:
        writer.write("  无索引器数据可分析")
        return {}
    
    writer.write(f"  数据类型: {type(indexer)}")
    
    if isinstance(indexer, dict):
        writer.write(f"  顶层键: {list(indexer.keys())}")
        
        structure_info = {}
        
        for key, value in indexer.items():
            writer.write(f"\n  '{key}' 详细信息:")
            
            if isinstance(value, dict):
                writer.write_stats(f"    {key} 字典信息", {
                    "类型": "字典",
                    "元素数量": len(value)
                })
                
                # 分析键和值的类型（限制样本数量）
                if value:
                    sample_items = list(value.items())[:3]  # 减少样本数量
                    key_types = Counter([type(k).__name__ for k, v in sample_items])
                    value_types = Counter([type(v).__name__ for k, v in sample_items])
                    
                    writer.write(f"      键类型分布: {dict(key_types)}")
                    writer.write(f"      值类型分布: {dict(value_types)}")
                    
                    # 只显示少量样本数据，避免过多输出
                    writer.write(f"      样本数据 (前3个):")
                    for i, (k, v) in enumerate(sample_items):
                        writer.write(f"        {i+1}. {k} -> {v}")
                
                structure_info[key] = {
                    'type': 'dict',
                    'count': len(value),
                    'key_types': dict(key_types) if value else {},
                    'value_types': dict(value_types) if value else {}
                }
                
            elif isinstance(value, list):
                writer.write_stats(f"    {key} 列表信息", {
                    "类型": "列表", 
                    "元素数量": len(value)
                })
                
                if value:
                    element_types = Counter([type(elem).__name__ for elem in value[:50]])  # 减少分析数量
                    writer.write(f"      元素类型分布: {dict(element_types)}")
                    writer.write(f"      样本数据 (前3个): {value[:3]}")  # 减少样本数量
                
                structure_info[key] = {
                    'type': 'list',
                    'count': len(value),
                    'element_types': dict(element_types) if value else {}
                }
                
            else:
                writer.write(f"    类型: {type(value).__name__}")
                if not isinstance(value, (dict, list)) or len(str(value)) < 100:  # 避免输出过长内容
                    writer.write(f"    值: {value}")
                else:
                    writer.write(f"    值: [复杂对象，长度: {len(str(value))}]")
                
                structure_info[key] = {
                    'type': type(value).__name__
                }
        
        return structure_info
    
    return {}

def analyze_user_item_mapping(indexer, writer):
    """分析用户和物品的映射关系"""
    writer.write_section("2. 用户和物品映射分析", level=1)
    
    mapping_stats = {}
    
    # 分析用户映射
    if 'u' in indexer and isinstance(indexer['u'], dict):
        user_mapping = indexer['u']
        writer.write_stats("  用户映射信息", {
            "用户总数": len(user_mapping)
        })
        
        # 分析用户ID类型和范围
        user_ids = list(user_mapping.keys())
        user_indices = list(user_mapping.values())
        
        # 原始用户ID分析（限制样本）
        user_id_types = Counter([type(uid).__name__ for uid in user_ids])
        writer.write_stats("    原始用户ID分析", {
            "ID类型分布": dict(user_id_types),
            "样本用户ID": str(user_ids[:3]) if user_ids else "无"  # 只显示3个样本
        })
        
        # 映射索引分析
        if user_indices:
            numeric_indices = [idx for idx in user_indices if isinstance(idx, (int, float))]
            if numeric_indices:
                is_continuous = max(numeric_indices) - min(numeric_indices) + 1 == len(set(numeric_indices))
                writer.write_stats("    映射索引分析", {
                    "索引范围": f"{min(numeric_indices)} - {max(numeric_indices)}",
                    "索引数量": len(numeric_indices),
                    "是否连续": '是' if is_continuous else '否'
                })
        
        mapping_stats['users'] = {
            'total_count': len(user_mapping),
            'id_types': dict(user_id_types),
            'index_range': [min(numeric_indices), max(numeric_indices)] if 'numeric_indices' in locals() and numeric_indices else None
        }
    
    # 分析物品映射
    if 'i' in indexer and isinstance(indexer['i'], dict):
        item_mapping = indexer['i']
        writer.write_stats("\n  物品映射信息", {
            "物品总数": len(item_mapping)
        })
        
        # 分析物品ID类型和范围
        item_ids = list(item_mapping.keys())
        item_indices = list(item_mapping.values())
        
        # 原始物品ID分析（限制样本）
        item_id_types = Counter([type(iid).__name__ for iid in item_ids])
        writer.write_stats("    原始物品ID分析", {
            "ID类型分布": dict(item_id_types),
            "样本物品ID": str(item_ids[:3]) if item_ids else "无"  # 只显示3个样本
        })
        
        # 映射索引分析
        if item_indices:
            numeric_indices = [idx for idx in item_indices if isinstance(idx, (int, float))]
            if numeric_indices:
                is_continuous = max(numeric_indices) - min(numeric_indices) + 1 == len(set(numeric_indices))
                writer.write_stats("    映射索引分析", {
                    "索引范围": f"{min(numeric_indices)} - {max(numeric_indices)}",
                    "索引数量": len(numeric_indices),
                    "是否连续": '是' if is_continuous else '否'
                })
        
        mapping_stats['items'] = {
            'total_count': len(item_mapping),
            'id_types': dict(item_id_types),
            'index_range': [min(numeric_indices), max(numeric_indices)] if 'numeric_indices' in locals() and numeric_indices else None
        }
    
    return mapping_stats

def analyze_feature_mapping(indexer, writer):
    """分析特征映射关系"""
    writer.write_section("3. 特征映射分析", level=1)
    
    feature_stats = {}
    
    if 'f' in indexer and isinstance(indexer['f'], dict):
        feature_mapping = indexer['f']
        writer.write_stats("  特征映射信息", {
            "特征字段总数": len(feature_mapping)
        })
        
        # 只分析少量特征字段的详细信息，避免过多输出
        writer.write("\n  特征字段详细分析 (前5个特征):")
        analyzed_count = 0
        for feat_id, feat_values in feature_mapping.items():
            if analyzed_count >= 5:  # 只分析前5个特征
                break
                
            if isinstance(feat_values, dict):
                writer.write(f"\n    特征ID '{feat_id}':")
                writer.write(f"      不同值数量: {len(feat_values):,}")
                
                # 分析特征值类型（仅少量样本）
                if feat_values:
                    sample_values = list(feat_values.items())[:2]  # 只取2个样本
                    value_types = Counter([type(v).__name__ for k, v in sample_values])
                    key_types = Counter([type(k).__name__ for k, v in sample_values])
                    
                    writer.write(f"      原始值类型: {dict(key_types)}")
                    writer.write(f"      映射值类型: {dict(value_types)}")
                    # 不输出具体的样本映射数据，避免过多输出
                
                # 统计特征值分布
                if feat_values:
                    mapped_values = [v for v in feat_values.values() if isinstance(v, (int, float))]
                    if mapped_values:
                        writer.write(f"      映射值范围: {min(mapped_values)} - {max(mapped_values)}")
                
                analyzed_count += 1
        
        # 特征字段统计（所有特征）
        feature_sizes = []
        feature_ids = []
        
        for feat_id, feat_values in feature_mapping.items():
            if isinstance(feat_values, dict):
                feature_sizes.append(len(feat_values))
                feature_ids.append(feat_id)
        
        if feature_sizes:
            writer.write_stats("\n  特征规模统计", {
                "平均特征值数量": f"{np.mean(feature_sizes):.2f}",
                "中位数特征值数量": f"{np.median(feature_sizes):.2f}",
                "最大特征值数量": max(feature_sizes),
                "最小特征值数量": min(feature_sizes)
            })
            
            # 找出最大和最小的特征
            max_idx = feature_sizes.index(max(feature_sizes))
            min_idx = feature_sizes.index(min(feature_sizes))
            writer.write(f"\n    极值特征:")
            writer.write(f"      最大特征: ID '{feature_ids[max_idx]}' ({max(feature_sizes):,} 个值)")
            writer.write(f"      最小特征: ID '{feature_ids[min_idx]}' ({min(feature_sizes):,} 个值)")
        
        feature_stats = {
            'total_features': len(feature_mapping),
            'feature_sizes': feature_sizes,
            'avg_feature_size': np.mean(feature_sizes) if feature_sizes else 0,
            'max_feature_size': max(feature_sizes) if feature_sizes else 0,
            'min_feature_size': min(feature_sizes) if feature_sizes else 0
        }
    
    return feature_stats

def analyze_data_completeness(indexer, mapping_stats, feature_stats, writer):
    """分析数据完整性和一致性"""
    writer.write_section("4. 数据完整性和一致性分析", level=1)
    
    completeness_stats = {}
    
    # 检查必要字段的存在性
    required_fields = ['u', 'i', 'f']
    existing_fields = [field for field in required_fields if field in indexer]
    missing_fields = [field for field in required_fields if field not in indexer]
    
    writer.write_stats("  字段完整性", {
        "存在的字段": str(existing_fields),
        "缺失的字段": str(missing_fields) if missing_fields else "无",
        "完整性": f"{len(existing_fields)/len(required_fields)*100:.1f}%"
    })
    
    # 分析ID连续性
    writer.write_section("  ID连续性分析", level=3)
    
    user_continuity = None
    item_continuity = None
    
    if 'users' in mapping_stats and mapping_stats['users'].get('index_range'):
        user_range = mapping_stats['users']['index_range']
        user_count = mapping_stats['users']['total_count']
        expected_count = user_range[1] - user_range[0] + 1
        user_continuity = user_count/expected_count if expected_count > 0 else None
        writer.write(f"    用户ID连续性: {user_count:,}/{expected_count:,} = {user_continuity*100:.1f}%")
    
    if 'items' in mapping_stats and mapping_stats['items'].get('index_range'):
        item_range = mapping_stats['items']['index_range']
        item_count = mapping_stats['items']['total_count']
        expected_count = item_range[1] - item_range[0] + 1
        item_continuity = item_count/expected_count if expected_count > 0 else None
        writer.write(f"    物品ID连续性: {item_count:,}/{expected_count:,} = {item_continuity*100:.1f}%")
    
    # 分析特征覆盖度
    feature_coverage = None
    if feature_stats.get('feature_sizes'):
        total_features = feature_stats['total_features']
        non_empty_features = len([size for size in feature_stats['feature_sizes'] if size > 0])
        feature_coverage = non_empty_features/total_features if total_features > 0 else None
        writer.write_stats("\n  特征覆盖度", {
            "非空特征数": f"{non_empty_features:,}/{total_features:,}",
            "覆盖度": f"{feature_coverage*100:.1f}%" if feature_coverage else "无法计算"
        })
    
    completeness_stats = {
        'field_completeness': len(existing_fields)/len(required_fields),
        'existing_fields': existing_fields,
        'missing_fields': missing_fields,
        'user_id_continuity': user_continuity,
        'item_id_continuity': item_continuity,
        'feature_coverage': feature_coverage
    }
    
    return completeness_stats

def analyze_click_behavior_indexing(indexer, writer):
    """分析点击行为相关的索引特征"""
    writer.write_section("5. 点击行为索引分析", level=1)
    
    click_analysis = {}
    
    # 分析用户索引范围和点击潜力
    if 'u' in indexer and isinstance(indexer['u'], dict):
        user_mapping = indexer['u']
        total_users = len(user_mapping)
        
        writer.write_stats("用户点击行为索引", {
            "总用户数": f"{total_users:,}",
            "用户ID范围": f"1 - {total_users:,}",
            "点击行为预估": "基于用户数量估算点击行为规模"
        })
        
        # 估算用户活跃度分布（基于ID分布模式）
        user_indices = list(user_mapping.values())
        if user_indices:
            # 分析用户ID的分布是否均匀
            sorted_indices = sorted(user_indices)
            gaps = [sorted_indices[i+1] - sorted_indices[i] for i in range(len(sorted_indices)-1)]
            avg_gap = np.mean(gaps) if gaps else 0
            
            writer.write_stats("用户索引分布特征", {
                "平均ID间隔": f"{avg_gap:.2f}",
                "ID分布均匀性": "高" if avg_gap < 2 else "中等" if avg_gap < 5 else "低",
                "预估活跃用户比例": "80-90%" if avg_gap < 2 else "60-80%" if avg_gap < 5 else "40-60%"
            })
        
        click_analysis['user_indexing'] = {
            'total_users': total_users,
            'avg_gap': avg_gap if 'avg_gap' in locals() else 0,
            'estimated_active_ratio': 0.85 if 'avg_gap' in locals() and avg_gap < 2 else 0.7 if 'avg_gap' in locals() and avg_gap < 5 else 0.5
        }
    
    # 分析物品索引范围和广告频次潜力
    if 'i' in indexer and isinstance(indexer['i'], dict):
        item_mapping = indexer['i']
        total_items = len(item_mapping)
        
        writer.write_stats("广告频次索引分析", {
            "总物品/广告数": f"{total_items:,}",
            "物品ID范围": f"1 - {total_items:,}",
            "广告频次分级预估": "基于物品数量估算频次分布"
        })
        
        # 估算广告频次分布
        # 基于常见的长尾分布，大部分广告是低频的
        estimated_high_freq = int(total_items * 0.05)  # 5%高频广告
        estimated_medium_freq = int(total_items * 0.15)  # 15%中频广告  
        estimated_low_freq = total_items - estimated_high_freq - estimated_medium_freq
        
        writer.write_stats("预估广告频次分布", {
            "高频广告(估算)": f"{estimated_high_freq:,} (5%)",
            "中频广告(估算)": f"{estimated_medium_freq:,} (15%)",
            "低频广告(估算)": f"{estimated_low_freq:,} (80%)",
            "分布模式": "典型长尾分布"
        })
        
        click_analysis['item_indexing'] = {
            'total_items': total_items,
            'estimated_high_freq': estimated_high_freq,
            'estimated_medium_freq': estimated_medium_freq,
            'estimated_low_freq': estimated_low_freq
        }
    
    # 分析特征索引对点击行为的支持
    if 'f' in indexer and isinstance(indexer['f'], dict):
        feature_mapping = indexer['f']
        
        # 分析与点击行为相关的特征
        click_relevant_features = []
        user_behavior_features = []
        item_quality_features = []
        
        for feat_id, feat_values in feature_mapping.items():
            if isinstance(feat_values, dict):
                cardinality = len(feat_values)
                
                # 根据特征ID和基数推断特征类型
                if feat_id.startswith('1'):  # 假设1xx是物品特征
                    if cardinality < 100:
                        item_quality_features.append((feat_id, cardinality))
                    else:
                        click_relevant_features.append((feat_id, cardinality))
                elif feat_id.startswith('2'):  # 假设2xx是用户特征
                    user_behavior_features.append((feat_id, cardinality))
        
        writer.write_stats("点击行为相关特征索引", {
            "点击相关特征数": len(click_relevant_features),
            "用户行为特征数": len(user_behavior_features),
            "物品质量特征数": len(item_quality_features),
            "特征索引完整性": "良好" if len(click_relevant_features) > 5 else "需要改进"
        })
        
        if click_relevant_features:
            writer.write("主要点击相关特征:")
            for feat_id, cardinality in click_relevant_features[:5]:
                feat_type = "高基数" if cardinality > 1000 else "中基数" if cardinality > 100 else "低基数"
                writer.write(f"    特征 {feat_id}: {cardinality:,} 个值 ({feat_type})")
        
        click_analysis['feature_indexing'] = {
            'click_relevant_features': len(click_relevant_features),
            'user_behavior_features': len(user_behavior_features),
            'item_quality_features': len(item_quality_features),
            'feature_details': click_relevant_features[:10]
        }
    
    # 索引效率分析
    writer.write_section("索引效率和优化建议", level=3)
    
    if click_analysis:
        total_users = click_analysis.get('user_indexing', {}).get('total_users', 0)
        total_items = click_analysis.get('item_indexing', {}).get('total_items', 0)
        
        # 计算潜在的用户-物品交互规模
        potential_interactions = total_users * total_items
        sparse_ratio = 1 / (total_items * 0.001)  # 假设每用户平均交互0.1%的物品
        
        writer.write_stats("索引效率评估", {
            "潜在交互规模": f"{potential_interactions:.2e}",
            "数据稀疏度": f"1:{sparse_ratio:.0f}",
            "索引存储效率": "优秀" if sparse_ratio > 1000 else "良好" if sparse_ratio > 100 else "需要优化",
            "点击行为查询效率": "支持快速查询" if total_users < 10000000 else "需要分区优化"
        })
        
        # 优化建议
        optimization_suggestions = []
        if total_users > 1000000:
            optimization_suggestions.append("用户分区: 按活跃度分区存储")
        if total_items > 1000000:
            optimization_suggestions.append("物品分层: 按热度分层索引")
        if sparse_ratio > 10000:
            optimization_suggestions.append("稀疏索引: 使用稀疏矩阵存储")
        
        writer.write("优化建议:")
        for suggestion in optimization_suggestions:
            writer.write(f"  - {suggestion}")
        
        click_analysis['efficiency'] = {
            'potential_interactions': potential_interactions,
            'sparse_ratio': sparse_ratio,
            'optimization_suggestions': optimization_suggestions
        }
    
    return click_analysis

def analyze_indexer():
    """主分析函数"""
    print("=" * 60)
    print("indexer.pkl 索引器数据详细分析")
    print("=" * 60)
    
    # 获取数据文件路径
    data_root = os.environ.get('TRAIN_DATA_PATH')
    if not data_root:
        print("错误: 未设置环境变量 TRAIN_DATA_PATH")
        return
    
    indexer_file = Path(data_root) / 'indexer.pkl'
    if not indexer_file.exists():
        print(f"错误: 文件不存在 {indexer_file}")
        return
    
    print(f"分析文件: {indexer_file}")
    print(f"文件大小: {indexer_file.stat().st_size / 1024:.2f} KB")
    print()
    
    # 加载数据
    indexer = load_indexer_data(indexer_file)
    
    if indexer is None:
        print("无法加载索引器数据，分析终止")
        return
    
    # 使用ReportWriter直接写入文件
    with ReportWriter("indexer_analysis", "indexer.pkl 索引器数据详细分析报告") as writer:
        # 写入文件信息
        writer.write_section("文件基本信息", level=1)
        writer.write(f"文件路径: {indexer_file}")
        writer.write(f"文件大小: {indexer_file.stat().st_size / 1024:.2f} KB")
        
        # 执行各项分析并直接写入文件
        structure_info = analyze_indexer_structure(indexer, writer)
        mapping_stats = analyze_user_item_mapping(indexer, writer)
        feature_stats = analyze_feature_mapping(indexer, writer)
        completeness_stats = analyze_data_completeness(indexer, mapping_stats, feature_stats, writer)
        
        # 新增：点击行为索引分析
        click_indexing_stats = analyze_click_behavior_indexing(indexer, writer)
        
        # 生成详细分析并写入文件
        generate_detailed_analysis(structure_info, mapping_stats, feature_stats, completeness_stats, writer)
        
        # 获取报告路径
        report_path = writer.get_report_path()
    
    # 输出到日志
    print(f"\n输出详细分析报告到日志...")
    output_report_to_log(report_path)
    
    print("\nindexer.pkl 索引器数据分析完成！")
    print("=" * 60)

def generate_detailed_analysis(structure_info, mapping_stats, feature_stats, completeness_stats, writer):
    """生成详细分析报告（直接写入文件）"""
    writer.write_section("5. 统计摘要分析", level=1)
    
    # 索引器结构概览
    writer.write_section("索引器结构概览", level=2)
    for key, info in structure_info.items():
        if isinstance(info, dict):
            structure_type = info.get('type', 'unknown')
            count = info.get('count', 0)
            writer.write(f"  {key}: {structure_type} 类型, {count:,} 项")
    
    # 映射统计详细分析
    if 'users' in mapping_stats:
        user_info = mapping_stats['users']
        writer.write_stats("用户映射详细分析", {
            "用户总数": user_info.get('total_count', 0)
        })
        index_range = user_info.get('index_range')
        if index_range:
            writer.write(f"  索引范围: {index_range[0]} - {index_range[1]}")
            writer.write(f"  索引跨度: {index_range[1] - index_range[0] + 1:,}")
    
    if 'items' in mapping_stats:
        item_info = mapping_stats['items']
        writer.write_stats("物品映射详细分析", {
            "物品总数": item_info.get('total_count', 0)
        })
        index_range = item_info.get('index_range')
        if index_range:
            writer.write(f"  索引范围: {index_range[0]} - {index_range[1]}")
            writer.write(f"  索引跨度: {index_range[1] - index_range[0] + 1:,}")
    
    # 特征统计详细分析
    if feature_stats.get('feature_sizes'):
        feature_sizes = feature_stats['feature_sizes']
        feature_summary = generate_statistical_summary(feature_sizes, "特征规模分布统计")
        writer.write_analysis_summary(feature_summary)
        
        writer.write_stats("特征规模分析", {
            "特征字段总数": feature_stats.get('total_features', 0),
            "平均特征规模": f"{feature_stats.get('avg_feature_size', 0):,.2f}",
            "最大特征规模": feature_stats.get('max_feature_size', 0),
            "最小特征规模": feature_stats.get('min_feature_size', 0)
        })
    
    # 数据完整性分析
    field_completeness = completeness_stats.get('field_completeness', 0)
    user_continuity = completeness_stats.get('user_id_continuity')
    item_continuity = completeness_stats.get('item_id_continuity')
    feature_coverage = completeness_stats.get('feature_coverage')
    
    completeness_data = {"字段完整性": f"{field_completeness*100:.1f}%"}
    if user_continuity is not None:
        completeness_data["用户ID连续性"] = f"{user_continuity*100:.1f}%"
    if item_continuity is not None:
        completeness_data["物品ID连续性"] = f"{item_continuity*100:.1f}%"
    if feature_coverage is not None:
        completeness_data["特征覆盖度"] = f"{feature_coverage*100:.1f}%"
    
    writer.write_stats("数据完整性分析", completeness_data)
    
    # 数据质量评估
    quality_assessment = []
    
    if field_completeness >= 0.9:
        quality_assessment.append("结构完整性: 优秀 - 所有关键字段都存在")
    elif field_completeness >= 0.7:
        quality_assessment.append("结构完整性: 良好 - 大部分字段存在")
    else:
        quality_assessment.append("结构完整性: 需要注意 - 缺失重要字段")
    
    if user_continuity and user_continuity >= 0.95:
        quality_assessment.append("用户ID质量: 优秀 - ID高度连续")
    elif user_continuity and user_continuity >= 0.8:
        quality_assessment.append("用户ID质量: 良好 - ID基本连续")
    elif user_continuity:
        quality_assessment.append("用户ID质量: 需要注意 - ID存在较大间隙")
    
    writer.write_section("数据质量评估", level=2)
    for assessment in quality_assessment:
        writer.write(f"  {assessment}")
    
    # 分析总结
    writer.write_section("6. 分析总结", level=1)
    total_users = mapping_stats.get('users', {}).get('total_count', 0)
    total_items = mapping_stats.get('items', {}).get('total_count', 0)
    total_features = feature_stats.get('total_features', 0)
    
    writer.write("基于以上分析，可以得出以下结论：")
    writer.write(f"1. 索引器包含 {total_users:,} 个用户和 {total_items:,} 个物品的映射信息")
    writer.write(f"2. 共有 {total_features:,} 个特征字段")
    writer.write(f"3. 数据完整性为 {field_completeness*100:.1f}%")
    writer.write(f"4. 整体数据质量: {'优秀' if field_completeness >= 0.9 else '良好' if field_completeness >= 0.7 else '需要改进'}")
    
    print("✓ 详细分析报告已写入文件")


if __name__ == "__main__":
    analyze_indexer()
