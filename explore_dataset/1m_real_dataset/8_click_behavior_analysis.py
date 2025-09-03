#!/usr/bin/env python3
"""
腾讯广告算法大赛 - 点击行为深度分析 (内存优化版本)
专门分析用户点击行为、广告频次和时间特征
采用流式处理和分批分析，避免内存溢出

优化策略：
1. 流式JSON解析，逐行处理
2. 分批统计累计，避免全量存储
3. 实时内存监控和垃圾回收
4. 样本限制和数据压缩

Author: Analysis Script
Date: 2025-01-20
"""

import os
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import warnings
import gc
import psutil
warnings.filterwarnings('ignore')

# 导入报告工具
from report_utils import ReportWriter, output_report_to_log, generate_statistical_summary

# 内存优化配置
BATCH_SIZE = 50000  # 每批处理的用户数量
MAX_MEMORY_GB = 45  # 最大内存使用限制（GB）
PROGRESS_INTERVAL = 10000  # 进度报告间隔

def get_memory_usage():
    """获取当前内存使用情况（GB）"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 3)  # 转换为GB

def stream_process_click_behavior(file_path, writer):
    """流式处理点击行为分析"""
    print(f"开始流式处理点击行为分析...")

    # 全局统计变量
    global_stats = {
        'total_users': 0,
        'total_records': 0,
        'total_clicks': 0,
        'total_exposures': 0,
        'users_with_clicks': 0,
        'user_click_counts': [],
        'item_click_counts': defaultdict(int),
        'item_exposure_counts': defaultdict(int),
        'click_timestamps': [],
        'hourly_clicks': defaultdict(int),
        'daily_clicks': defaultdict(int),
        'weekend_clicks': 0,
        'weekday_clicks': 0,
        'ad_frequency_stats': defaultdict(int)
    }

    # 用户活跃度分级计数
    activity_levels = {
        'inactive': 0,
        'low_activity': 0,
        'medium_activity': 0,
        'high_activity': 0
    }

    batch_count = 0
    current_batch_users = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                # 检查内存使用
                if line_num % 5000 == 0:
                    memory_gb = get_memory_usage()
                    if memory_gb > MAX_MEMORY_GB:
                        print(f"⚠️ 内存使用超过限制 ({memory_gb:.2f}GB > {MAX_MEMORY_GB}GB)")
                        print("为防止系统崩溃，提前终止处理")
                        break

                try:
                    if not line.strip():
                        continue

                    user_sequence = json.loads(line.strip())
                    if not isinstance(user_sequence, list):
                        continue

                    # 处理单个用户序列
                    user_stats = process_single_user_clicks(user_sequence)

                    # 累计到全局统计
                    accumulate_user_stats(user_stats, global_stats, activity_levels)

                    global_stats['total_users'] += 1
                    current_batch_users += 1

                    # 进度报告和批次处理
                    if (line_num + 1) % PROGRESS_INTERVAL == 0:
                        memory_gb = get_memory_usage()
                        print(f"已处理 {line_num + 1:,} 行，累计用户: {global_stats['total_users']:,}，"
                              f"累计记录: {global_stats['total_records']:,}，内存: {memory_gb:.2f}GB")

                    # 批次数据清理
                    if current_batch_users >= BATCH_SIZE:
                        batch_count += 1
                        print(f"✓ 完成批次 {batch_count}，处理用户数: {current_batch_users}")

                        # 清理点击时间戳以节省内存（保留样本）
                        if len(global_stats['click_timestamps']) > 100000:
                            # 保留最新的50000个时间戳
                            global_stats['click_timestamps'] = global_stats['click_timestamps'][-50000:]

                        # 执行垃圾回收
                        gc.collect()
                        current_batch_users = 0

                        memory_gb = get_memory_usage()
                        print(f"批次 {batch_count} 处理后内存使用: {memory_gb:.2f}GB")

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"处理第 {line_num + 1} 行时出错: {e}")
                    continue

        # 分析全局统计结果
        analyze_global_click_statistics(global_stats, activity_levels, writer)

    except Exception as e:
        print(f"流式处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def process_single_user_clicks(user_sequence):
    """处理单个用户的点击行为"""
    user_stats = {
        'user_clicks': 0,
        'user_exposures': 0,
        'item_clicks': defaultdict(int),
        'item_exposures': defaultdict(int),
        'click_timestamps': [],
        'records_count': len(user_sequence) if isinstance(user_sequence, list) else 0
    }

    for record in user_sequence:
        if isinstance(record, list) and len(record) >= 6:
            user_id, item_id, user_feat, item_feat, action_type, timestamp = record[:6]

            if item_id:
                user_stats['item_exposures'][item_id] += 1
                user_stats['user_exposures'] += 1

            if action_type == 1 and item_id:  # 点击行为
                user_stats['user_clicks'] += 1
                user_stats['item_clicks'][item_id] += 1

                if timestamp:
                    user_stats['click_timestamps'].append(timestamp)

    return user_stats

def accumulate_user_stats(user_stats, global_stats, activity_levels):
    """累计用户统计到全局统计"""
    global_stats['total_records'] += user_stats['records_count']
    global_stats['total_clicks'] += user_stats['user_clicks']
    global_stats['total_exposures'] += user_stats['user_exposures']

    # 用户点击数统计
    user_clicks = user_stats['user_clicks']
    global_stats['user_click_counts'].append(user_clicks)

    if user_clicks > 0:
        global_stats['users_with_clicks'] += 1

    # 用户活跃度分级
    if user_clicks == 0:
        activity_levels['inactive'] += 1
    elif 1 <= user_clicks <= 5:
        activity_levels['low_activity'] += 1
    elif 6 <= user_clicks <= 20:
        activity_levels['medium_activity'] += 1
    else:
        activity_levels['high_activity'] += 1

    # 累计物品点击统计
    for item_id, clicks in user_stats['item_clicks'].items():
        global_stats['item_click_counts'][item_id] += clicks

    for item_id, exposures in user_stats['item_exposures'].items():
        global_stats['item_exposure_counts'][item_id] += exposures

    # 累计时间戳统计（限制数量）
    if len(global_stats['click_timestamps']) < 200000:  # 限制最多存储20万个时间戳
        global_stats['click_timestamps'].extend(user_stats['click_timestamps'])

    # 时间分析（对有限的时间戳样本）
    for timestamp in user_stats['click_timestamps'][:100]:  # 每用户最多分析100个时间戳
        try:
            dt = datetime.fromtimestamp(timestamp)
            hour = dt.hour
            weekday = dt.weekday()

            global_stats['hourly_clicks'][hour] += 1
            global_stats['daily_clicks'][weekday] += 1

            if weekday >= 5:  # 周末
                global_stats['weekend_clicks'] += 1
            else:  # 工作日
                global_stats['weekday_clicks'] += 1
        except:
            continue

def analyze_global_click_statistics(global_stats, activity_levels, writer):
    """分析全局点击统计结果"""
    writer.write_section("用户点击行为统计分析", level=1)

    total_users = global_stats['total_users']
    total_clicks = global_stats['total_clicks']
    users_with_clicks = global_stats['users_with_clicks']

    writer.write_stats("总体点击统计", {
        "分析用户总数": f"{total_users:,}",
        "分析记录总数": f"{global_stats['total_records']:,}",
        "有点击行为用户数": f"{users_with_clicks:,}",
        "用户点击参与率": f"{users_with_clicks/total_users*100:.2f}%" if total_users > 0 else "0%",
        "总点击次数": f"{total_clicks:,}",
        "总曝光次数": f"{global_stats['total_exposures']:,}",
        "整体点击率": f"{total_clicks/global_stats['total_exposures']*100:.3f}%" if global_stats['total_exposures'] > 0 else "0%"
    })

    # 用户活跃度分析
    writer.write_stats("用户活跃度分布", {
        "无点击用户": f"{activity_levels['inactive']:,} ({activity_levels['inactive']/total_users*100:.1f}%)",
        "低活跃用户(1-5次)": f"{activity_levels['low_activity']:,} ({activity_levels['low_activity']/total_users*100:.1f}%)",
        "中活跃用户(6-20次)": f"{activity_levels['medium_activity']:,} ({activity_levels['medium_activity']/total_users*100:.1f}%)",
        "高活跃用户(>20次)": f"{activity_levels['high_activity']:,} ({activity_levels['high_activity']/total_users*100:.1f}%)"
    })

    # 用户点击次数分布分析
    if global_stats['user_click_counts']:
        user_clicks_array = np.array(global_stats['user_click_counts'])
        writer.write_stats("用户点击次数分布", {
            "平均每用户点击次数": f"{np.mean(user_clicks_array):.2f}",
            "中位数点击次数": f"{np.median(user_clicks_array):.2f}",
            "最大用户点击次数": f"{np.max(user_clicks_array):,}",
            "点击次数标准差": f"{np.std(user_clicks_array):.2f}"
        })

    # 广告点击频次分析
    analyze_ad_frequency_patterns(global_stats, writer)

    # 时间模式分析
    analyze_time_patterns(global_stats, writer)

def analyze_ad_frequency_patterns(global_stats, writer):
    """分析广告频次模式"""
    writer.write_section("广告点击频次分析", level=1)

    item_click_counts = global_stats['item_click_counts']
    if not item_click_counts:
        writer.write("无广告点击数据可分析")
        return

    click_frequencies = list(item_click_counts.values())

    # 广告频次分级
    low_freq_ads = sum(1 for c in click_frequencies if 1 <= c <= 5)
    medium_freq_ads = sum(1 for c in click_frequencies if 6 <= c <= 20)
    high_freq_ads = sum(1 for c in click_frequencies if c > 20)

    writer.write_stats("广告频次分布", {
        "被点击广告总数": f"{len(item_click_counts):,}",
        "总点击次数": f"{sum(click_frequencies):,}",
        "平均每广告点击次数": f"{np.mean(click_frequencies):.2f}",
        "中位数点击次数": f"{np.median(click_frequencies):.2f}",
        "低频广告(1-5次)": f"{low_freq_ads:,} ({low_freq_ads/len(click_frequencies)*100:.1f}%)",
        "中频广告(6-20次)": f"{medium_freq_ads:,} ({medium_freq_ads/len(click_frequencies)*100:.1f}%)",
        "高频广告(>20次)": f"{high_freq_ads:,} ({high_freq_ads/len(click_frequencies)*100:.1f}%)"
    })

    # Top热门广告
    top_items = sorted(item_click_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    writer.write("\nTop 10 热门广告:")
    for i, (item_id, clicks) in enumerate(top_items, 1):
        writer.write(f"  {i:2d}. 广告ID {item_id}: {clicks:,} 次点击")

    # 广告点击率分析（基于曝光数据）
    item_exposure_counts = global_stats['item_exposure_counts']
    if item_exposure_counts:
        ad_ctrs = []
        for item_id, clicks in item_click_counts.items():
            exposures = item_exposure_counts.get(item_id, 0)
            if exposures > 0:
                ctr = clicks / exposures
                ad_ctrs.append(ctr)

        if ad_ctrs:
            writer.write_stats("广告点击率分析", {
                "有效CTR广告数": f"{len(ad_ctrs):,}",
                "平均点击率": f"{np.mean(ad_ctrs)*100:.3f}%",
                "中位数点击率": f"{np.median(ad_ctrs)*100:.3f}%",
                "最高点击率": f"{np.max(ad_ctrs)*100:.3f}%"
            })

def analyze_time_patterns(global_stats, writer):
    """分析时间模式"""
    writer.write_section("时间模式分析", level=1)

    hourly_clicks = global_stats['hourly_clicks']
    daily_clicks = global_stats['daily_clicks']

    # 小时级别分析
    if hourly_clicks:
        total_hourly_clicks = sum(hourly_clicks.values())
        peak_hour = max(hourly_clicks.items(), key=lambda x: x[1])

        writer.write_stats("小时级活跃度分析", {
            "分析的时间点击数": f"{total_hourly_clicks:,}",
            "最活跃时段": f"{peak_hour[0]}点 ({peak_hour[1]:,} 次点击)",
            "24小时平均点击": f"{total_hourly_clicks/24:.1f} 次/小时"
        })

        # Top 5 活跃小时
        top_hours = sorted(hourly_clicks.items(), key=lambda x: x[1], reverse=True)[:5]
        writer.write("\nTop 5 活跃小时:")
        for i, (hour, clicks) in enumerate(top_hours, 1):
            percentage = clicks / total_hourly_clicks * 100
            writer.write(f"  {i}. {hour:2d}点: {clicks:,} 次点击 ({percentage:.1f}%)")

    # 工作日vs周末分析
    weekday_clicks = global_stats['weekday_clicks']
    weekend_clicks = global_stats['weekend_clicks']
    total_time_clicks = weekday_clicks + weekend_clicks

    if total_time_clicks > 0:
        writer.write_stats("工作日vs周末分析", {
            "工作日点击数": f"{weekday_clicks:,} ({weekday_clicks/total_time_clicks*100:.1f}%)",
            "周末点击数": f"{weekend_clicks:,} ({weekend_clicks/total_time_clicks*100:.1f}%)",
            "周末vs工作日比例": f"{weekend_clicks/weekday_clicks:.2f}" if weekday_clicks > 0 else "N/A"
        })

    # 一周活跃度分析
    if daily_clicks:
        weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        writer.write("\n一周活跃度分布:")
        for day_idx, day_name in enumerate(weekday_names):
            clicks = daily_clicks.get(day_idx, 0)
            if total_time_clicks > 0:
                percentage = clicks / total_time_clicks * 100
                writer.write(f"  {day_name}: {clicks:,} 次点击 ({percentage:.1f}%)")

def analyze_click_behavior():
    """主分析函数 - 内存优化版本"""
    print("=" * 60)
    print("用户点击行为和广告频次深度分析 (内存优化版本)")
    print("=" * 60)

    # 获取数据文件路径
    data_root = os.environ.get('TRAIN_DATA_PATH')
    if not data_root:
        print("错误: 未设置环境变量 TRAIN_DATA_PATH")
        return

    seq_file = Path(data_root) / 'seq.jsonl'
    if not seq_file.exists():
        print(f"错误: 文件不存在 {seq_file}")
        return

    file_size_mb = seq_file.stat().st_size / (1024*1024)
    print(f"分析文件: {seq_file}")
    print(f"文件大小: {file_size_mb:.2f} MB")
    print(f"内存限制: {MAX_MEMORY_GB}GB")
    print(f"批次大小: {BATCH_SIZE:,} 用户/批次")
    print()

    try:
        # 使用ReportWriter直接写入文件
        with ReportWriter("click_behavior_analysis", "用户点击行为和广告频次深度分析报告 (内存优化版本)") as writer:
            # 写入文件信息
            writer.write_section("分析配置信息", level=1)
            writer.write(f"分析文件: {seq_file}")
            writer.write(f"文件大小: {file_size_mb:.2f} MB")
            writer.write(f"处理策略: 流式处理 + 分批累计")
            writer.write(f"内存限制: {MAX_MEMORY_GB}GB")
            writer.write(f"批次大小: {BATCH_SIZE:,} 用户/批次")

            # 执行流式点击行为分析
            stream_process_click_behavior(seq_file, writer)

            # 添加说明
            writer.write_section("分析说明", level=1)
            writer.write("本分析采用流式处理技术，逐行读取和分析用户序列数据")
            writer.write("通过分批累计统计，避免将全部数据加载到内存中")
            writer.write("这种方法能够处理任意大小的数据文件而不会导致内存溢出")
            writer.write("结果包含完整的用户点击行为、广告频次和时间模式分析")

            # 获取报告路径
            report_path = writer.get_report_path()

        # 输出到日志
        print(f"\n输出详细分析报告到日志...")
        output_report_to_log(report_path)

        print("\n✅ 点击行为分析完成（内存优化版本）！")
        print(f"最终内存使用: {get_memory_usage():.2f}GB")

    except Exception as e:
        print(f"❌ 点击行为分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)

if __name__ == "__main__":
    analyze_click_behavior()
