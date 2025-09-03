#!/usr/bin/env python3
"""
报告工具模块 - 用于生成详细文本报告并分批输出到日志

Author: Analysis Script
Date: 2025-01-20
"""

import os
import json
from pathlib import Path

def get_cache_dir():
    """获取用户缓存目录"""
    cache_dir = os.environ.get('USER_CACHE_PATH')
    if not cache_dir:
        cache_dir = os.path.expanduser('~/.cache/data_analysis')
    
    cache_path = Path(cache_dir) / 'analysis_reports'
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path

def save_detailed_report(report_name, data, description=""):
    """保存详细报告到缓存目录"""
    cache_dir = get_cache_dir()
    report_path = cache_dir / f"{report_name}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        if description:
            f.write(f"# {description}\n")
            f.write("=" * 80 + "\n\n")
        
        if isinstance(data, dict):
            f.write(format_dict_report(data))
        elif isinstance(data, list):
            f.write(format_list_report(data))
        else:
            f.write(str(data))
    
    print(f"✓ 详细报告已保存到缓存: {report_path}")
    return report_path

class ReportWriter:
    """报告文件写入器，用于直接将分析内容写入文件而不是打印"""
    
    def __init__(self, report_name, description=""):
        self.cache_dir = get_cache_dir()
        self.report_path = self.cache_dir / f"{report_name}.txt"
        self.file_handle = None
        self.description = description
        
    def __enter__(self):
        self.file_handle = open(self.report_path, 'w', encoding='utf-8')
        if self.description:
            self.file_handle.write(f"# {self.description}\n")
            self.file_handle.write("=" * 80 + "\n\n")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
        print(f"✓ 详细报告已保存到缓存: {self.report_path}")
        
    def write(self, content):
        """写入内容到文件"""
        if self.file_handle:
            self.file_handle.write(str(content))
            self.file_handle.write("\n")
            
    def write_section(self, title, content="", level=1):
        """写入章节标题和内容"""
        if level == 1:
            self.write(f"\n{title}")
            self.write("=" * 60)
        elif level == 2:
            self.write(f"\n{title}:")
            self.write("-" * 40)
        else:
            self.write(f"\n{title}:")
            
        if content:
            self.write(content)
            
    def write_stats(self, title, stats_dict):
        """写入统计信息"""
        self.write(f"\n{title}:")
        self.write("-" * 40)
        for key, value in stats_dict.items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    self.write(f"  {key}: {value:,.2f}")
                else:
                    self.write(f"  {key}: {value:,}")
            else:
                self.write(f"  {key}: {value}")
                
    def write_analysis_summary(self, summary_text):
        """写入分析摘要"""
        self.write(f"\n{summary_text}")
        
    def get_report_path(self):
        """获取报告文件路径"""
        return self.report_path

def format_dict_report(data, indent=0):
    """格式化字典为可读报告"""
    result = []
    prefix = "  " * indent
    
    for key, value in data.items():
        if isinstance(value, dict):
            result.append(f"{prefix}{key}:")
            result.append(format_dict_report(value, indent + 1))
        elif isinstance(value, list):
            result.append(f"{prefix}{key}: [{len(value)} 项]")
            if len(value) <= 20:  # 显示小列表的全部内容
                for i, item in enumerate(value):
                    result.append(f"{prefix}  [{i}] {item}")
            else:  # 大列表只显示前后几项
                for i in range(5):
                    result.append(f"{prefix}  [{i}] {value[i]}")
                result.append(f"{prefix}  ... (省略 {len(value) - 10} 项) ...")
                for i in range(len(value) - 5, len(value)):
                    result.append(f"{prefix}  [{i}] {value[i]}")
        else:
            result.append(f"{prefix}{key}: {value}")
    
    return "\n".join(result)

def format_list_report(data):
    """格式化列表为可读报告"""
    result = []
    
    if len(data) <= 50:  # 小列表显示全部
        for i, item in enumerate(data):
            result.append(f"[{i:4d}] {item}")
    else:  # 大列表显示前后部分
        for i in range(25):
            result.append(f"[{i:4d}] {data[i]}")
        result.append(f"... (省略 {len(data) - 50} 项) ...")
        for i in range(len(data) - 25, len(data)):
            result.append(f"[{i:4d}] {data[i]}")
    
    return "\n".join(result)

def output_report_to_log(report_path, max_lines_per_batch=1000):
    """分批将报告内容输出到日志"""
    if not os.path.exists(report_path):
        print(f"错误: 报告文件不存在 {report_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"开始输出报告: {report_path}")
    print(f"{'='*80}")
    
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    batch_count = (total_lines + max_lines_per_batch - 1) // max_lines_per_batch
    
    for batch_idx in range(batch_count):
        start_idx = batch_idx * max_lines_per_batch
        end_idx = min(start_idx + max_lines_per_batch, total_lines)
        
        print(f"\n--- 报告第 {batch_idx + 1}/{batch_count} 部分 (行 {start_idx + 1}-{end_idx}) ---")
        
        for line in lines[start_idx:end_idx]:
            print(line.rstrip())
        
        if batch_idx < batch_count - 1:
            print(f"\n--- 第 {batch_idx + 1} 部分结束，继续下一部分... ---")
    
    print(f"\n{'='*80}")
    print(f"报告输出完成，总计 {total_lines} 行")
    print(f"{'='*80}")

def generate_statistical_summary(data, title="统计摘要"):
    """生成数据的统计摘要"""
    import numpy as np
    
    summary = [f"# {title}", "=" * 60, ""]
    
    if isinstance(data, list) and data:
        if all(isinstance(x, (int, float)) for x in data):
            # 数值列表的统计
            summary.extend([
                f"数据点数量: {len(data):,}",
                f"最小值: {min(data):,.2f}",
                f"最大值: {max(data):,.2f}",
                f"平均值: {np.mean(data):,.2f}",
                f"中位数: {np.median(data):,.2f}",
                f"标准差: {np.std(data):,.2f}",
                f"25%分位数: {np.percentile(data, 25):,.2f}",
                f"75%分位数: {np.percentile(data, 75):,.2f}",
                ""
            ])
            
            # 分布区间统计
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            summary.append("分位数分布:")
            for p in percentiles:
                value = np.percentile(data, p)
                summary.append(f"  {p:2d}%: {value:,.2f}")
            summary.append("")
        
        else:
            # 非数值列表的统计
            from collections import Counter
            counter = Counter(data)
            summary.extend([
                f"总数量: {len(data):,}",
                f"唯一值数量: {len(counter):,}",
                f"最高频次: {counter.most_common(1)[0][1]:,}" if counter else "0",
                ""
            ])
            
            summary.append("频次分布 (Top 20):")
            for item, count in counter.most_common(20):
                percentage = count / len(data) * 100
                summary.append(f"  {item}: {count:,} ({percentage:.2f}%)")
            summary.append("")
    
    elif isinstance(data, dict):
        summary.extend([
            f"字典键数量: {len(data):,}",
            f"字典键: {list(data.keys())[:10]}" + ("..." if len(data) > 10 else ""),
            ""
        ])
    
    return "\n".join(summary)

def create_comprehensive_analysis_report(analysis_results, report_name):
    """创建综合分析报告"""
    cache_dir = get_cache_dir()
    report_path = cache_dir / f"{report_name}_comprehensive.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("腾讯广告算法大赛 - 数据分析综合报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 写入各项分析结果
        for section_name, section_data in analysis_results.items():
            f.write(f"## {section_name}\n")
            f.write("-" * 60 + "\n\n")
            
            if isinstance(section_data, dict):
                f.write(format_dict_report(section_data))
            else:
                f.write(str(section_data))
            
            f.write("\n\n")
    
    print(f"✓ 综合分析报告已保存: {report_path}")
    return report_path
