#!/usr/bin/env python3
"""
腾讯广告算法大赛 - 真实数据集完整分析执行脚本（无内存限制版本）
完整分析所有数据，使用用户缓存文件夹存储详细报告

Author: Analysis Script
Date: 2025-01-20
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def check_environment():
    """检查运行环境"""
    print("检查运行环境...")
    
    # 检查数据路径环境变量
    data_path = os.environ.get('TRAIN_DATA_PATH')
    if not data_path:
        print("错误: 未设置环境变量 TRAIN_DATA_PATH")
        return False
    
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"错误: 数据目录不存在: {data_path}")
        return False
    
    print(f"✓ 数据目录: {data_path}")
    
    # 检查缓存目录环境变量
    cache_path = os.environ.get('USER_CACHE_PATH')
    if not cache_path:
        print("警告: 未设置环境变量 USER_CACHE_PATH，将使用默认缓存目录")
        cache_path = os.path.expanduser('~/.cache/data_analysis')
    
    cache_path = Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ 缓存目录: {cache_path}")
    
    # 检查必要的文件
    required_files = [
        'indexer.pkl',
        'item_feat_dict.json',
        'item_feat_num.json',
        'seq.jsonl',
        'seq_offsets.pkl',
        'user_feat_num.json'
    ]
    
    existing_files = []
    missing_files = []
    
    for file_name in required_files:
        if (data_path / file_name).exists():
            file_size = (data_path / file_name).stat().st_size / (1024*1024)
            existing_files.append((file_name, file_size))
        else:
            missing_files.append(file_name)
    
    print(f"\n数据文件检查:")
    print(f"  存在的文件 ({len(existing_files)}/{len(required_files)}):")
    for file_name, size_mb in existing_files:
        print(f"    {file_name}: {size_mb:.2f} MB")
    
    if missing_files:
        print(f"  缺失的文件 ({len(missing_files)}):")
        for file_name in missing_files:
            print(f"    {file_name}")
    
    # 检查Python包
    required_packages = ['numpy', 'pandas', 'pathlib', 'collections']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"错误: 缺少必要的Python包: {missing_packages}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    
    print(f"✓ 所有必要的Python包都已安装")
    
    return len(existing_files) > 0

def run_analysis_script(script_name, description):
    """运行单个分析脚本"""
    print(f"\n{'='*80}")
    print(f"正在执行: {description}")
    print(f"脚本: {script_name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        # 运行脚本
        result = subprocess.run([
            sys.executable, script_name
        ], cwd=Path(__file__).parent, capture_output=True, text=True)
        
        # 打印输出（所有输出，因为我们移除了1000行限制）
        if result.stdout:
            print("=== 脚本输出 ===")
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("=== 错误输出 ===")
            print(f"错误输出: {result.stderr}")
            return False
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ {description} 完成，耗时: {elapsed_time:.2f} 秒")
        return True
        
    except Exception as e:
        print(f"✗ 执行 {script_name} 时出错: {e}")
        return False

def main():
    """主函数"""
    print("腾讯广告算法大赛 - 真实数据集完整分析（无限制版本）")
    print("="*80)
    print("注意: 此版本将分析所有数据，无行数和采样限制")
    print("预计将产生大量详细日志输出")
    print("="*80)
    
    # 检查环境
    if not check_environment():
        print("环境检查失败，程序退出")
        return
    
    # 显示系统信息
    print(f"\n系统信息:")
    print(f"  Python版本: {sys.version}")
    print(f"  工作目录: {os.getcwd()}")
    print(f"  数据路径: {os.environ.get('TRAIN_DATA_PATH')}")
    print(f"  缓存路径: {os.environ.get('USER_CACHE_PATH', '默认缓存目录')}")
    
    # 定义分析脚本列表
    analysis_scripts = [
        ('1_dataset_overview.py', '数据集总体概述分析'),
        ('2_seq_jsonl_analysis_lite.py', '用户序列数据分析 (轻量版本，避免内存问题)'),
        ('3_seq_offsets_analysis.py', '序列偏移量数据分析'),
        ('4_indexer_analysis.py', '索引器数据分析 (含点击索引)'),
        ('5_item_feat_dict_analysis.py', '物品特征字典分析 (含点击相关特征)'),
        ('6_item_feat_num_analysis.py', '物品特征数量分析'),
        ('7_user_feat_num_analysis.py', '用户特征数量分析'),
        ('8_click_behavior_analysis_optimized.py', '专门点击行为深度分析 (内存优化版本)')
    ]
    
    # 执行所有分析
    successful_analyses = 0
    total_start_time = time.time()
    
    for script_name, description in analysis_scripts:
        script_path = Path(__file__).parent / script_name
        if not script_path.exists():
            print(f"警告: 脚本文件不存在: {script_name}")
            continue
            
        if run_analysis_script(script_name, description):
            successful_analyses += 1
        else:
            print(f"脚本 {script_name} 执行失败，继续执行下一个...")
    
    # 输出最终结果
    total_time = time.time() - total_start_time
    print(f"\n{'='*80}")
    print(f"完整分析执行完成！")
    print(f"{'='*80}")
    print(f"成功完成的分析: {successful_analyses}/{len(analysis_scripts)}")
    print(f"总耗时: {total_time:.2f} 秒")
    
    # 显示缓存目录信息
    cache_dir = os.environ.get('USER_CACHE_PATH', os.path.expanduser('~/.cache/data_analysis'))
    cache_path = Path(cache_dir) / 'analysis_reports'
    
    if cache_path.exists():
        report_files = list(cache_path.glob('*.txt'))
        print(f"\n生成的详细报告:")
        print(f"  缓存目录: {cache_path}")
        print(f"  报告文件数: {len(report_files)}")
        for report_file in report_files:
            size_kb = report_file.stat().st_size / 1024
            print(f"    {report_file.name}: {size_kb:.2f} KB")
    
    if successful_analyses == len(analysis_scripts):
        print(f"\n🎉 所有分析都已成功完成！")
        print(f"💡 详细报告已保存在缓存目录中，并通过日志输出显示")
    else:
        print(f"\n⚠️  部分分析未能完成，请检查上述错误信息")
    
    print(f"\n📋 使用建议:")
    print(f"  1. 查看上述日志输出获取详细统计信息")
    print(f"  2. 在缓存目录中查看完整报告文件")
    print(f"  3. 根据分析结果进行特征工程优化")

if __name__ == "__main__":
    main()
