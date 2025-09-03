#!/usr/bin/env python3
"""
腾讯广告算法大赛 - seq_offsets.pkl 文件详细分析
序列偏移量数据分析，用于快速随机访问用户序列数据

Author: Analysis Script  
Date: 2025-01-20
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 导入报告工具
from report_utils import save_detailed_report, output_report_to_log, generate_statistical_summary, ReportWriter

# 设置中文字体和样式

def load_offsets_data(file_path):
    """加载偏移量数据"""
    try:
        with open(file_path, 'rb') as f:
            offsets = pickle.load(f)
        print(f"✓ 成功加载偏移量数据，包含 {len(offsets)} 个用户")
        return offsets
    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
        return None

def analyze_offset_structure(offsets, writer):
    """分析偏移量数据结构"""
    writer.write_section("1. 偏移量数据结构分析", level=1)
    
    if not offsets:
        writer.write("  无偏移量数据可分析")
        return {}
    
    writer.write(f"  用户数量: {len(offsets):,}")
    writer.write(f"  数据类型: {type(offsets)}")
    
    if isinstance(offsets, dict):
        # 分析用户ID类型和分布
        user_ids = list(offsets.keys())
        user_id_types = Counter([type(uid).__name__ for uid in user_ids])
        
        writer.write("\n  用户ID类型分布:")
        for id_type, count in user_id_types.items():
            writer.write(f"    {id_type}: {count:,} 个")
        
        # 分析偏移量值
        offset_values = list(offsets.values())
        offset_types = Counter([type(offset).__name__ for offset in offset_values])
        
        writer.write("\n  偏移量类型分布:")
        for offset_type, count in offset_types.items():
            writer.write(f"    {offset_type}: {count:,} 个")
        
        # 数值统计
        numeric_offsets = [offset for offset in offset_values if isinstance(offset, (int, float))]
        if numeric_offsets:
            writer.write_stats("  偏移量数值统计", {
                "最小偏移量": min(numeric_offsets),
                "最大偏移量": max(numeric_offsets),
                "平均偏移量": np.mean(numeric_offsets),
                "中位数偏移量": np.median(numeric_offsets),
                "偏移量标准差": np.std(numeric_offsets)
            })
        
        return {
            'total_users': len(offsets),
            'user_ids': user_ids[:100],  # 保存前100个用户ID用于分析
            'offset_values': numeric_offsets,
            'user_id_types': dict(user_id_types),
            'offset_types': dict(offset_types)
        }
    
    elif isinstance(offsets, list):
        writer.write(f"  偏移量列表长度: {len(offsets):,}")
        
        # 分析列表中的数据类型
        if offsets:
            element_types = Counter([type(elem).__name__ for elem in offsets])
            writer.write("\n  元素类型分布:")
            for elem_type, count in element_types.items():
                writer.write(f"    {elem_type}: {count:,} 个")
            
            # 如果是数值列表，进行统计分析
            numeric_elements = [elem for elem in offsets if isinstance(elem, (int, float))]
            if numeric_elements:
                writer.write_stats("  数值统计", {
                    "最小值": min(numeric_elements),
                    "最大值": max(numeric_elements),
                    "平均值": np.mean(numeric_elements),
                    "中位数": np.median(numeric_elements)
                })
        
        return {
            'total_elements': len(offsets),
            'elements': offsets[:100],  # 保存前100个元素
            'element_types': dict(element_types) if offsets else {}
        }
    
    return {}

def analyze_offset_distribution(stats, writer):
    """分析偏移量分布特征"""
    writer.write_section("2. 偏移量分布特征分析", level=1)
    
    offset_values = stats.get('offset_values', [])
    
    if not offset_values:
        writer.write("  无数值偏移量数据可分析")
        return {}
    
    # 计算偏移量差值（相邻用户之间的距离）
    sorted_offsets = sorted(offset_values)
    offset_diffs = [sorted_offsets[i+1] - sorted_offsets[i] for i in range(len(sorted_offsets)-1)]
    
    writer.write_stats("  偏移量差值分析", {
        "平均差值": f"{np.mean(offset_diffs):,.2f} 字节",
        "中位数差值": f"{np.median(offset_diffs):,.2f} 字节",
        "最小差值": f"{min(offset_diffs):,} 字节",
        "最大差值": f"{max(offset_diffs):,} 字节",
        "差值标准差": f"{np.std(offset_diffs):,.2f} 字节"
    })
    
    # 分析数据大小分布
    avg_record_size = np.mean(offset_diffs)
    total_data_size = max(sorted_offsets) - min(sorted_offsets) if len(sorted_offsets) > 1 else 0
    
    writer.write_stats("  数据大小分布估计", {
        "估计平均每用户数据大小": f"{avg_record_size:,.0f} 字节",
        "总数据范围": f"{total_data_size:,.0f} 字节 ({total_data_size/(1024*1024):.2f} MB)"
    })
    
    # 分析分布模式
    offset_percentiles = np.percentile(offset_values, [10, 25, 50, 75, 90, 95, 99])
    percentile_labels = ['10%', '25%', '50%', '75%', '90%', '95%', '99%']
    
    writer.write("\n  偏移量分位数:")
    for label, value in zip(percentile_labels, offset_percentiles):
        writer.write(f"    {label}: {value:,.0f}")
    
    return {
        'offset_diffs': offset_diffs,
        'avg_record_size': avg_record_size,
        'total_data_size': total_data_size,
        'percentiles': dict(zip(percentile_labels, offset_percentiles))
    }

def analyze_access_patterns(stats, writer):
    """分析数据访问模式"""
    writer.write_section("3. 数据访问模式分析", level=1)
    
    offset_values = stats.get('offset_values', [])
    user_ids = stats.get('user_ids', [])
    
    if not offset_values or not user_ids:
        writer.write("  无足够数据进行访问模式分析")
        return {}
    
    # 分析用户ID与偏移量的关系
    if len(user_ids) == len(offset_values):
        # 创建用户ID到偏移量的映射
        user_offset_pairs = list(zip(user_ids[:len(offset_values)], offset_values))
        
        # 按偏移量排序，分析顺序性
        sorted_pairs = sorted(user_offset_pairs, key=lambda x: x[1])
        
        # 检查用户ID的顺序性
        sorted_user_ids = [pair[0] for pair in sorted_pairs]
        
        # 如果用户ID是数值类型，分析其顺序性
        numeric_user_ids = []
        try:
            numeric_user_ids = [int(uid) if isinstance(uid, (int, str)) else uid for uid in sorted_user_ids]
            numeric_user_ids = [uid for uid in numeric_user_ids if isinstance(uid, int)]
        except:
            pass
        
        if numeric_user_ids:
            # 分析用户ID的连续性
            id_diffs = [numeric_user_ids[i+1] - numeric_user_ids[i] for i in range(len(numeric_user_ids)-1)]
            
            writer.write_stats("  用户ID顺序性分析", {
                "数值用户ID数量": len(numeric_user_ids),
                "平均ID差值": f"{np.mean(id_diffs):.2f}",
                "ID是否连续": '是' if all(diff == 1 for diff in id_diffs[:100]) else '否',
                "用户ID范围": f"{min(numeric_user_ids)} - {max(numeric_user_ids)}",
                "ID跨度": max(numeric_user_ids) - min(numeric_user_ids)
            })
    
    # 分析数据的聚集性
    sorted_offsets = sorted(offset_values)
    data_density = len(sorted_offsets) / (max(sorted_offsets) - min(sorted_offsets)) if len(sorted_offsets) > 1 else 0
    
    writer.write_stats("  数据聚集性分析", {
        "数据密度": f"{data_density:.2e} 用户/字节"
    })
    
    return {
        'data_density': data_density,
        'numeric_user_ids': numeric_user_ids[:100] if 'numeric_user_ids' in locals() else []
    }

def analyze_seq_offsets():
    """主分析函数"""
    print("=" * 60)
    print("seq_offsets.pkl 偏移量数据详细分析")
    print("=" * 60)
    
    # 获取数据文件路径
    data_root = os.environ.get('TRAIN_DATA_PATH')
    if not data_root:
        print("错误: 未设置环境变量 TRAIN_DATA_PATH")
        return
    
    offsets_file = Path(data_root) / 'seq_offsets.pkl'
    if not offsets_file.exists():
        print(f"错误: 文件不存在 {offsets_file}")
        return
    
    print(f"分析文件: {offsets_file}")
    print(f"文件大小: {offsets_file.stat().st_size / 1024:.2f} KB")
    print()
    
    # 加载数据
    offsets = load_offsets_data(offsets_file)
    
    if offsets is None:
        print("无法加载偏移量数据，分析终止")
        return
    
    # 使用ReportWriter直接写入文件
    with ReportWriter("seq_offsets_analysis", "seq_offsets.pkl 偏移量数据详细分析报告") as writer:
        # 写入文件信息
        writer.write_section("文件基本信息", level=1)
        writer.write(f"文件路径: {offsets_file}")
        writer.write(f"文件大小: {offsets_file.stat().st_size / 1024:.2f} KB")
        
        # 执行各项分析并直接写入文件
        structure_stats = analyze_offset_structure(offsets, writer)
        distribution_stats = analyze_offset_distribution(structure_stats, writer)
        access_stats = analyze_access_patterns(structure_stats, writer)
        
        # 生成详细分析并写入文件
        generate_detailed_analysis(structure_stats, distribution_stats, access_stats, writer)
        
        # 获取报告路径
        report_path = writer.get_report_path()
    
    # 输出到日志
    print(f"\n输出详细分析报告到日志...")
    output_report_to_log(report_path)
    
    print("\nseq_offsets.pkl 偏移量数据分析完成！")
    print("=" * 60)

def generate_detailed_analysis(structure_stats, distribution_stats, access_stats, writer):
    """生成详细分析报告（直接写入文件）"""
    writer.write_section("4. 详细统计分析", level=1)
    
    # 偏移量分布详细分析
    offset_values = structure_stats.get('offset_values', [])
    if offset_values:
        offset_summary = generate_statistical_summary(offset_values, "偏移量分布统计")
        writer.write_analysis_summary(offset_summary)
    
    # 偏移量差值分析
    offset_diffs = distribution_stats.get('offset_diffs', [])
    if offset_diffs:
        diff_summary = generate_statistical_summary(offset_diffs, "偏移量差值分布统计")
        writer.write_analysis_summary(diff_summary)
    
    # 数据规模分析
    total_users = structure_stats.get('total_users', 0)
    avg_record_size = distribution_stats.get('avg_record_size', 0)
    total_data_size = distribution_stats.get('total_data_size', 0)
    
    writer.write_stats("数据规模分析", {
        "总用户数": total_users,
        "估计平均记录大小": f"{avg_record_size:,.0f} 字节",
        "总数据范围": f"{total_data_size:,.0f} 字节 ({total_data_size/(1024*1024):.2f} MB)"
    })
    
    # 数据质量评估
    data_density = access_stats.get('data_density', 0)
    quality_assessment = ""
    
    if data_density > 1e-6:
        quality_assessment = "良好 - 数据分布紧密"
    elif data_density > 1e-9:
        quality_assessment = "中等 - 数据分布较稀疏"
    else:
        quality_assessment = "需要注意 - 数据分布很稀疏"
    
    writer.write_stats("数据质量评估", {
        "数据密度": f"{data_density:.2e} 用户/字节",
        "数据质量": quality_assessment
    })
    
    # 访问优化建议
    optimization_suggestion = ""
    if avg_record_size > 0:
        if avg_record_size < 1000:
            optimization_suggestion = "小记录优化: 记录较小，适合批量读取"
        elif avg_record_size < 10000:
            optimization_suggestion = "中等记录优化: 记录适中，标准访问模式"
        else:
            optimization_suggestion = "大记录优化: 记录较大，建议按需加载"
    
    writer.write_section("访问优化建议", optimization_suggestion, level=2)
    
    # 分析总结
    writer.write_section("5. 分析总结", level=1)
    writer.write("基于以上分析，可以得出以下结论：")
    writer.write(f"1. 数据集包含 {total_users:,} 个用户的偏移量信息")
    writer.write(f"2. 平均每用户数据大小约为 {avg_record_size:,.0f} 字节")
    writer.write(f"3. 数据质量{quality_assessment}")
    writer.write(f"4. 建议采用{optimization_suggestion}")
    
    print("✓ 详细分析报告已写入文件")


if __name__ == "__main__":
    analyze_seq_offsets()
