#!/usr/bin/env python3
"""
腾讯广告算法大赛 - seq.jsonl 文件详细分析 (内存优化增强版本)
用户序列数据的深度分析，包括序列长度分布、用户行为模式、时间分布等
内存优化：分批处理，及时清理，减少内存占用

增强功能：
1. 时间戳范围深度分析 - 用户活跃时间段、点击间隔模式、时间聚类
2. 特征工程洞察 - 特征交互模式、特征重要性评估、自动特征选择
3. 用户行为序列分析 - 行为路径挖掘、序列模式发现
4. 会话分析 - 用户会话识别、会话内行为模式
5. 高级统计分析 - 分布拟合、异常检测、趋势分析

Author: Analysis Script
Date: 2025-01-20
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import warnings
import gc
import psutil
import time
import math
warnings.filterwarnings('ignore')

# 导入报告工具
from report_utils import save_detailed_report, output_report_to_log, generate_statistical_summary, ReportWriter

# 内存优化配置
BATCH_SIZE = 50000  # 每批处理的用户数量
MAX_MEMORY_GB = 40  # 最大内存使用限制（GB）
PROGRESS_INTERVAL = 5000  # 进度报告间隔

# 时间分析配置
TIME_SLOTS = {
    'early_morning': (0, 6),    # 凌晨 0-6点
    'morning': (6, 12),         # 上午 6-12点
    'afternoon': (12, 18),      # 下午 12-18点
    'evening': (18, 24)         # 晚上 18-24点
}

# 特征工程配置
MIN_FEATURE_SAMPLES = 100  # 特征分析的最小样本数
MAX_SEQUENCE_LENGTH = 10   # 最大序列长度分析
MIN_PATTERN_SUPPORT = 50   # 最小模式支持度

def get_memory_usage():
    """获取当前内存使用情况（GB）"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / (1024 ** 3)  # 转换为GB

def load_sequence_data_batch(file_path, start_line=0, batch_size=BATCH_SIZE):
    """分批加载序列数据，内存优化版本"""
    sequences = []
    total_records = 0
    current_line = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # 跳过前面的行
                if i < start_line:
                    continue

                # 达到批次大小时停止
                if len(sequences) >= batch_size:
                    break

                try:
                    if line.strip():
                        user_sequence = json.loads(line.strip())
                        sequences.append(user_sequence)
                        total_records += len(user_sequence) if isinstance(user_sequence, list) else 1
                        current_line = i

                        # 进度报告
                        if len(sequences) % PROGRESS_INTERVAL == 0:
                            memory_gb = get_memory_usage()
                            print(f"  批次内已加载 {len(sequences)} 个用户序列，累计 {total_records} 条记录，内存使用: {memory_gb:.2f}GB")

                            # 内存检查
                            if memory_gb > MAX_MEMORY_GB:
                                print(f"  ⚠️ 内存使用接近限制 ({memory_gb:.2f}GB > {MAX_MEMORY_GB}GB)，提前结束批次")
                                break

                except json.JSONDecodeError as e:
                    print(f"警告: 第 {i+1} 行JSON解析失败: {e}")
                    continue

    except Exception as e:
        print(f"错误: 无法读取文件 {file_path}: {e}")
        return [], 0, 0

    print(f"✓ 批次加载完成: {len(sequences)} 个用户序列，累计 {total_records} 条记录")
    return sequences, total_records, current_line + 1

def get_total_lines(file_path):
    """快速获取文件总行数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        return total_lines
    except:
        return 0

def analyze_click_intervals(click_timestamps, max_analysis=10000):
    """分析点击间隔模式"""
    if len(click_timestamps) < 2:
        return {}

    # 限制分析数量以节省内存
    analysis_timestamps = sorted(click_timestamps)[:max_analysis]

    intervals = []
    for i in range(1, len(analysis_timestamps)):
        interval = analysis_timestamps[i] - analysis_timestamps[i-1]
        intervals.append(interval)

    if not intervals:
        return {}

    # 转换为分钟
    intervals_minutes = [interval / 60 for interval in intervals]

    # 间隔分类
    immediate = sum(1 for i in intervals_minutes if i <= 1)      # 1分钟内
    short = sum(1 for i in intervals_minutes if 1 < i <= 10)     # 1-10分钟
    medium = sum(1 for i in intervals_minutes if 10 < i <= 60)   # 10分钟-1小时
    long = sum(1 for i in intervals_minutes if 60 < i <= 1440)   # 1-24小时
    very_long = sum(1 for i in intervals_minutes if i > 1440)    # >24小时

    return {
        'total_intervals': len(intervals),
        'avg_interval_minutes': np.mean(intervals_minutes),
        'median_interval_minutes': np.median(intervals_minutes),
        'std_interval_minutes': np.std(intervals_minutes),
        'intervals_by_type': {
            'immediate': immediate,
            'short': short,
            'medium': medium,
            'long': long,
            'very_long': very_long
        },
        'intervals_minutes': intervals_minutes[:1000]  # 保存样本用于进一步分析
    }

def extract_user_sessions(user_sequence, session_gap=1800):
    """提取用户会话信息（30分钟无活动则为新会话）"""
    sessions = []

    if not isinstance(user_sequence, list):
        return sessions

    # 按时间排序记录
    timestamped_records = []
    for record in user_sequence:
        if isinstance(record, list) and len(record) >= 6:
            user_id, item_id, user_feat, item_feat, action_type, timestamp = record[:6]
            if timestamp and item_id:
                timestamped_records.append((timestamp, item_id, action_type))

    if not timestamped_records:
        return sessions

    # 按时间排序
    timestamped_records.sort(key=lambda x: x[0])

    # 识别会话边界
    current_session = []

    for timestamp, item_id, action_type in timestamped_records:
        if not current_session:
            current_session = [(timestamp, item_id, action_type)]
        else:
            last_timestamp = current_session[-1][0]
            if timestamp - last_timestamp > session_gap:
                # 结束当前会话，开始新会话
                if len(current_session) >= 2:  # 至少2个动作才算有效会话
                    sessions.append(current_session)
                current_session = [(timestamp, item_id, action_type)]
            else:
                current_session.append((timestamp, item_id, action_type))

    # 添加最后一个会话
    if len(current_session) >= 2:
        sessions.append(current_session)

    return sessions

def extract_behavior_sequences(user_sequence, max_length=MAX_SEQUENCE_LENGTH):
    """提取用户行为序列模式"""
    if not isinstance(user_sequence, list):
        return []

    # 提取行为类型序列
    action_sequence = []
    for record in user_sequence:
        if isinstance(record, list) and len(record) >= 6:
            user_id, item_id, user_feat, item_feat, action_type, timestamp = record[:6]
            if action_type is not None:
                action_sequence.append(action_type)

    # 生成不同长度的子序列
    sequences = []
    for length in range(2, min(max_length + 1, len(action_sequence) + 1)):
        for i in range(len(action_sequence) - length + 1):
            subseq = tuple(action_sequence[i:i + length])
            sequences.append(subseq)

    return sequences

def analyze_feature_interactions(user_sequence, sample_size=1000):
    """分析特征交互模式（内存优化版本）"""
    feature_combinations = defaultdict(int)
    feature_values = defaultdict(set)
    feature_click_correlation = defaultdict(lambda: {'clicks': 0, 'exposures': 0})

    if not isinstance(user_sequence, list):
        return feature_combinations, feature_values, feature_click_correlation

    # 限制分析记录数以节省内存
    analysis_records = user_sequence[:sample_size] if len(user_sequence) > sample_size else user_sequence

    for record in analysis_records:
        if isinstance(record, list) and len(record) >= 6:
            user_id, item_id, user_feat, item_feat, action_type, timestamp = record[:6]

            # 分析用户特征
            if user_feat and isinstance(user_feat, dict):
                for feat_id, feat_value in user_feat.items():
                    if feat_value is not None:
                        feature_key = f'user_{feat_id}'
                        feature_values[feature_key].add(str(feat_value))

                        # 记录特征与点击的关联
                        feature_click_correlation[feature_key]['exposures'] += 1
                        if action_type == 1:
                            feature_click_correlation[feature_key]['clicks'] += 1

            # 分析物品特征
            if item_feat and isinstance(item_feat, dict):
                for feat_id, feat_value in item_feat.items():
                    if feat_value is not None:
                        feature_key = f'item_{feat_id}'
                        feature_values[feature_key].add(str(feat_value))

                        # 记录特征与点击的关联
                        feature_click_correlation[feature_key]['exposures'] += 1
                        if action_type == 1:
                            feature_click_correlation[feature_key]['clicks'] += 1

                            # 记录特征组合与点击
                            combo_key = f'{feature_key}_{feat_value}_click'
                            feature_combinations[combo_key] += 1

    return feature_combinations, feature_values, feature_click_correlation

def detect_time_patterns(timestamps, max_analysis=5000):
    """检测时间模式和周期性"""
    if len(timestamps) < 10:
        return {}

    # 限制分析数量
    analysis_timestamps = sorted(timestamps)[:max_analysis]

    # 转换为datetime
    datetimes = []
    for ts in analysis_timestamps:
        try:
            dt = datetime.fromtimestamp(ts)
            datetimes.append(dt)
        except:
            continue

    if not datetimes:
        return {}

    # 小时分布分析
    hourly_dist = defaultdict(int)
    daily_dist = defaultdict(int)  # 0=周一, 6=周日

    for dt in datetimes:
        hourly_dist[dt.hour] += 1
        daily_dist[dt.weekday()] += 1

    # 找出活跃时段
    peak_hours = sorted(hourly_dist.items(), key=lambda x: x[1], reverse=True)[:3]
    peak_days = sorted(daily_dist.items(), key=lambda x: x[1], reverse=True)[:3]

    # 计算时间跨度
    time_span_hours = (max(datetimes) - min(datetimes)).total_seconds() / 3600

    return {
        'hourly_distribution': dict(hourly_dist),
        'daily_distribution': dict(daily_dist),
        'peak_hours': peak_hours,
        'peak_days': peak_days,
        'time_span_hours': time_span_hours,
        'total_events': len(datetimes)
    }

def analyze_batch_sequence_structure(sequences, writer, batch_num):
    """分析批次序列数据结构"""
    if not sequences:
        return {}

    # 分析序列长度分布
    seq_lengths = [len(seq) if isinstance(seq, list) else 1 for seq in sequences]

    batch_stats = {
        'batch_num': batch_num,
        'user_count': len(sequences),
        'avg_length': np.mean(seq_lengths),
        'median_length': np.median(seq_lengths),
        'max_length': max(seq_lengths),
        'min_length': min(seq_lengths),
        'total_records': sum(seq_lengths)
    }

    writer.write(f"\n批次 {batch_num} 序列结构统计:")
    writer.write(f"  用户数: {batch_stats['user_count']:,}")
    writer.write(f"  平均序列长度: {batch_stats['avg_length']:.2f}")
    writer.write(f"  中位数序列长度: {batch_stats['median_length']:.2f}")
    writer.write(f"  序列长度范围: {batch_stats['min_length']} - {batch_stats['max_length']}")

    return batch_stats

def analyze_batch_user_behavior(sequences, writer, batch_num, global_stats):
    """分析批次用户行为模式（累计到全局统计中）- 增强版本"""
    batch_user_clicks = 0
    batch_item_clicks = defaultdict(int)
    batch_item_exposures = defaultdict(int)
    batch_click_timestamps = []

    # 新增：特征工程相关统计
    batch_sessions = []
    batch_behavior_sequences = []
    batch_feature_interactions = defaultdict(int)
    batch_time_slot_activity = defaultdict(int)

    print(f"  分析批次 {batch_num} 的 {len(sequences)} 个用户行为...")

    for idx, user_seq in enumerate(sequences):
        if not isinstance(user_seq, list):
            continue

        user_clicks = 0

        # 提取用户会话
        user_sessions = extract_user_sessions(user_seq)
        batch_sessions.extend(user_sessions)

        # 提取行为序列
        behavior_seqs = extract_behavior_sequences(user_seq)
        batch_behavior_sequences.extend(behavior_seqs)

        # 分析特征交互（限制每个用户分析的记录数）
        feat_combinations, feat_values, feat_correlations = analyze_feature_interactions(user_seq, sample_size=200)
        for combo, count in feat_combinations.items():
            batch_feature_interactions[combo] += count

        for record in user_seq:
            if isinstance(record, list) and len(record) >= 6:
                user_id, item_id, user_feat, item_feat, action_type, timestamp = record[:6]

                if item_id:
                    batch_item_exposures[item_id] += 1
                    global_stats['total_exposures'] += 1

                if action_type == 1 and item_id:  # 点击行为
                    user_clicks += 1
                    batch_user_clicks += 1
                    batch_item_clicks[item_id] += 1
                    global_stats['total_clicks'] += 1

                    if timestamp:
                        batch_click_timestamps.append(timestamp)

                        # 时间段分析
                        try:
                            dt = datetime.fromtimestamp(timestamp)
                            hour = dt.hour

                            for slot_name, (start_hour, end_hour) in TIME_SLOTS.items():
                                if start_hour <= hour < end_hour:
                                    batch_time_slot_activity[slot_name] += 1
                                    break
                        except:
                            continue

        if user_clicks > 0:
            global_stats['users_with_clicks'] += 1

        global_stats['total_users'] += 1

        # 用户活跃度分级累计
        if user_clicks == 0:
            global_stats['inactive_users'] += 1
        elif 1 <= user_clicks <= 5:
            global_stats['low_activity_users'] += 1
        elif 6 <= user_clicks <= 20:
            global_stats['medium_activity_users'] += 1
        else:
            global_stats['high_activity_users'] += 1

        # 进度报告
        if (idx + 1) % 5000 == 0:
            memory_gb = get_memory_usage()
            print(f"    已分析批次内 {idx + 1} 个用户，内存: {memory_gb:.2f}GB")

    # 累计到全局统计
    for item_id, clicks in batch_item_clicks.items():
        global_stats['item_click_counts'][item_id] += clicks

    for item_id, exposures in batch_item_exposures.items():
        global_stats['item_exposure_counts'][item_id] += exposures

    global_stats['click_timestamps'].extend(batch_click_timestamps)

    # 累计新增的全局统计
    if 'all_sessions' not in global_stats:
        global_stats['all_sessions'] = []
    global_stats['all_sessions'].extend(batch_sessions)

    if 'all_behavior_sequences' not in global_stats:
        global_stats['all_behavior_sequences'] = []
    global_stats['all_behavior_sequences'].extend(batch_behavior_sequences)

    if 'feature_interactions' not in global_stats:
        global_stats['feature_interactions'] = defaultdict(int)
    for combo, count in batch_feature_interactions.items():
        global_stats['feature_interactions'][combo] += count

    if 'time_slot_activity' not in global_stats:
        global_stats['time_slot_activity'] = defaultdict(int)
    for slot, count in batch_time_slot_activity.items():
        global_stats['time_slot_activity'][slot] += count

    # 输出批次统计
    writer.write(f"\n批次 {batch_num} 用户行为统计:")
    writer.write(f"  批次用户点击总数: {batch_user_clicks:,}")
    writer.write(f"  批次唯一点击物品数: {len(batch_item_clicks):,}")
    writer.write(f"  批次唯一曝光物品数: {len(batch_item_exposures):,}")
    writer.write(f"  批次点击时间戳数: {len(batch_click_timestamps):,}")
    writer.write(f"  批次会话数: {len(batch_sessions):,}")
    writer.write(f"  批次行为序列数: {len(batch_behavior_sequences):,}")
    writer.write(f"  批次特征交互模式数: {len(batch_feature_interactions):,}")

    return {
        'batch_clicks': batch_user_clicks,
        'batch_clicked_items': len(batch_item_clicks),
        'batch_exposed_items': len(batch_item_exposures),
        'batch_sessions': len(batch_sessions),
        'batch_sequences': len(batch_behavior_sequences)
    }

def analyze_global_statistics(global_stats, writer):
    """分析全局统计结果"""
    writer.write_section("全局用户行为统计分析", level=1)

    total_users = global_stats['total_users']
    total_clicks = global_stats['total_clicks']
    users_with_clicks = global_stats['users_with_clicks']

    writer.write_stats("总体用户点击统计", {
        "总用户数": f"{total_users:,}",
        "有点击行为用户数": f"{users_with_clicks:,}",
        "用户点击参与率": f"{users_with_clicks/total_users*100:.2f}%" if total_users > 0 else "0%",
        "总点击次数": f"{total_clicks:,}",
        "平均每用户点击次数": f"{total_clicks/total_users:.2f}" if total_users > 0 else "0"
    })

    # 用户活跃度分析
    writer.write_stats("用户活跃度分布", {
        "无点击用户": f"{global_stats['inactive_users']:,} ({global_stats['inactive_users']/total_users*100:.1f}%)" if total_users > 0 else "0",
        "低活跃用户(1-5次)": f"{global_stats['low_activity_users']:,} ({global_stats['low_activity_users']/total_users*100:.1f}%)" if total_users > 0 else "0",
        "中活跃用户(6-20次)": f"{global_stats['medium_activity_users']:,} ({global_stats['medium_activity_users']/total_users*100:.1f}%)" if total_users > 0 else "0",
        "高活跃用户(>20次)": f"{global_stats['high_activity_users']:,} ({global_stats['high_activity_users']/total_users*100:.1f}%)" if total_users > 0 else "0"
    })

    # 广告点击频次分析
    item_click_counts = global_stats['item_click_counts']
    if item_click_counts:
        click_frequencies = list(item_click_counts.values())

        # 广告频次分级
        low_freq_ads = sum(1 for c in click_frequencies if 1 <= c <= 5)
        medium_freq_ads = sum(1 for c in click_frequencies if 6 <= c <= 20)
        high_freq_ads = sum(1 for c in click_frequencies if c > 20)

        writer.write_stats("广告点击频次分布", {
            "被点击广告总数": f"{len(item_click_counts):,}",
            "总点击次数": f"{sum(click_frequencies):,}",
            "平均每广告点击次数": f"{np.mean(click_frequencies):.2f}",
            "中位数点击次数": f"{np.median(click_frequencies):.2f}",
            "低频广告(1-5次)": f"{low_freq_ads:,} ({low_freq_ads/len(click_frequencies)*100:.1f}%)",
            "中频广告(6-20次)": f"{medium_freq_ads:,} ({medium_freq_ads/len(click_frequencies)*100:.1f}%)",
            "高频广告(>20次)": f"{high_freq_ads:,} ({high_freq_ads/len(click_frequencies)*100:.1f}%)"
        })

        # Top 10 热门广告
        top_items = sorted(item_click_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        writer.write("\nTop 10 热门广告:")
        for i, (item_id, clicks) in enumerate(top_items, 1):
            writer.write(f"  {i:2d}. 广告ID {item_id}: {clicks:,} 次点击")

def analyze_temporal_patterns(global_stats, writer):
    """增强时间模式分析"""
    writer.write_section("时间模式深度分析", level=1)

    click_timestamps = global_stats['click_timestamps']
    if not click_timestamps:
        writer.write("无点击时间戳数据可分析")
        return

    # 基础时间分析
    hourly_clicks = defaultdict(int)
    daily_clicks = defaultdict(int)
    weekend_clicks = 0
    weekday_clicks = 0

    for timestamp in click_timestamps:
        try:
            dt = datetime.fromtimestamp(timestamp)
            hour = dt.hour
            weekday = dt.weekday()

            hourly_clicks[hour] += 1
            daily_clicks[weekday] += 1

            if weekday >= 5:  # 周末
                weekend_clicks += 1
            else:  # 工作日
                weekday_clicks += 1
        except:
            continue

    # 输出基础时间分析
    total_time_clicks = sum(hourly_clicks.values())
    peak_hour = max(hourly_clicks.items(), key=lambda x: x[1]) if hourly_clicks else (0, 0)

    writer.write_stats("点击时间分布分析", {
        "分析的点击时间戳数": f"{len(click_timestamps):,}",
        "有效时间戳数": f"{total_time_clicks:,}",
        "点击最活跃时段": f"{peak_hour[0]}点 ({peak_hour[1]:,} 次点击)",
        "工作日点击数": f"{weekday_clicks:,} ({weekday_clicks/total_time_clicks*100:.1f}%)" if total_time_clicks > 0 else "0",
        "周末点击数": f"{weekend_clicks:,} ({weekend_clicks/total_time_clicks*100:.1f}%)" if total_time_clicks > 0 else "0"
    })

    # 新增：时间段活跃度分析
    time_slot_activity = global_stats.get('time_slot_activity', {})
    if time_slot_activity:
        total_slot_clicks = sum(time_slot_activity.values())
        slot_names = {
            'early_morning': '凌晨(0-6点)',
            'morning': '上午(6-12点)',
            'afternoon': '下午(12-18点)',
            'evening': '晚上(18-24点)'
        }

        writer.write_stats("用户活跃时间段分布", {
            slot_names[slot]: f"{count:,} ({count/total_slot_clicks*100:.1f}%)"
            for slot, count in time_slot_activity.items()
        })

        peak_slot = max(time_slot_activity.items(), key=lambda x: x[1])
        writer.write(f"\n最活跃时间段: {slot_names.get(peak_slot[0], peak_slot[0])} ({peak_slot[1]:,} 次点击)")

    # 新增：点击间隔深度分析
    if len(click_timestamps) >= 100:
        interval_analysis = analyze_click_intervals(click_timestamps, max_analysis=10000)

        if interval_analysis:
            writer.write_stats("点击间隔深度分析", {
                "总间隔数": f"{interval_analysis['total_intervals']:,}",
                "平均间隔": f"{interval_analysis['avg_interval_minutes']:.1f} 分钟",
                "中位数间隔": f"{interval_analysis['median_interval_minutes']:.1f} 分钟",
                "间隔标准差": f"{interval_analysis['std_interval_minutes']:.1f} 分钟"
            })

            intervals = interval_analysis['intervals_by_type']
            total_intervals = interval_analysis['total_intervals']

            writer.write_stats("点击间隔类型分布", {
                "即时点击(≤1分钟)": f"{intervals['immediate']:,} ({intervals['immediate']/total_intervals*100:.1f}%)",
                "短间隔(1-10分钟)": f"{intervals['short']:,} ({intervals['short']/total_intervals*100:.1f}%)",
                "中间隔(10分钟-1小时)": f"{intervals['medium']:,} ({intervals['medium']/total_intervals*100:.1f}%)",
                "长间隔(1-24小时)": f"{intervals['long']:,} ({intervals['long']/total_intervals*100:.1f}%)",
                "超长间隔(>24小时)": f"{intervals['very_long']:,} ({intervals['very_long']/total_intervals*100:.1f}%)"
            })

    # 新增：时间模式检测
    time_patterns = detect_time_patterns(click_timestamps)
    if time_patterns:
        writer.write_stats("时间模式检测结果", {
            "数据时间跨度": f"{time_patterns['time_span_hours']:.1f} 小时",
            "分析事件数": f"{time_patterns['total_events']:,}"
        })

        if time_patterns['peak_hours']:
            writer.write("\nTop 3 活跃小时:")
            for i, (hour, count) in enumerate(time_patterns['peak_hours'], 1):
                writer.write(f"  {i}. {hour}点: {count:,} 次点击")

        if time_patterns['peak_days']:
            weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
            writer.write("\nTop 3 活跃日期:")
            for i, (day, count) in enumerate(time_patterns['peak_days'], 1):
                day_name = weekday_names[day] if 0 <= day <= 6 else f"第{day}天"
                writer.write(f"  {i}. {day_name}: {count:,} 次点击")

def analyze_user_sessions(global_stats, writer):
    """用户会话深度分析"""
    writer.write_section("用户会话深度分析", level=1)

    all_sessions = global_stats.get('all_sessions', [])
    if not all_sessions:
        writer.write("无有效用户会话数据")
        return

    # 会话基本统计
    session_lengths = [len(session) for session in all_sessions]
    session_durations = []
    session_click_rates = []

    for session in all_sessions:
        if len(session) >= 2:
            start_time = session[0][0]
            end_time = session[-1][0]
            duration = (end_time - start_time) / 60  # 转换为分钟
            session_durations.append(duration)

            # 计算会话内点击率
            clicks = sum(1 for _, _, action_type in session if action_type == 1)
            click_rate = clicks / len(session) if len(session) > 0 else 0
            session_click_rates.append(click_rate)

    writer.write_stats("会话基本统计", {
        "总会话数": f"{len(all_sessions):,}",
        "平均会话长度": f"{np.mean(session_lengths):.2f} 个动作",
        "中位数会话长度": f"{np.median(session_lengths):.2f} 个动作",
        "最长会话": f"{max(session_lengths):,} 个动作",
        "最短有效会话": f"{min(session_lengths):,} 个动作"
    })

    if session_durations:
        writer.write_stats("会话时长分析", {
            "平均会话时长": f"{np.mean(session_durations):.2f} 分钟",
            "中位数会话时长": f"{np.median(session_durations):.2f} 分钟",
            "最长会话时长": f"{max(session_durations):.2f} 分钟",
            "时长标准差": f"{np.std(session_durations):.2f} 分钟"
        })

    if session_click_rates:
        writer.write_stats("会话内点击率分析", {
            "平均会话点击率": f"{np.mean(session_click_rates)*100:.2f}%",
            "中位数会话点击率": f"{np.median(session_click_rates)*100:.2f}%",
            "最高会话点击率": f"{max(session_click_rates)*100:.2f}%"
        })

    # 会话长度分布分析
    length_dist = Counter(session_lengths)
    writer.write("\nTop 10 会话长度分布:")
    for length, count in length_dist.most_common(10):
        percentage = count / len(all_sessions) * 100
        writer.write(f"  {length} 个动作: {count:,} 个会话 ({percentage:.1f}%)")

def analyze_behavior_sequences(global_stats, writer):
    """用户行为序列模式分析"""
    writer.write_section("用户行为序列模式挖掘", level=1)

    all_behavior_sequences = global_stats.get('all_behavior_sequences', [])
    if not all_behavior_sequences:
        writer.write("无行为序列数据可分析")
        return

    # 挖掘频繁模式
    pattern_counts = Counter(all_behavior_sequences)
    frequent_patterns = {pattern: count for pattern, count in pattern_counts.items()
                         if count >= MIN_PATTERN_SUPPORT}

    writer.write_stats("序列模式挖掘结果", {
        "总序列片段数": f"{len(all_behavior_sequences):,}",
        "不同模式数": f"{len(pattern_counts):,}",
        "频繁模式数量": f"{len(frequent_patterns):,}",
        "最小支持度": f"{MIN_PATTERN_SUPPORT}"
    })

    if frequent_patterns:
        # 按支持度排序
        sorted_patterns = sorted(frequent_patterns.items(), key=lambda x: x[1], reverse=True)

        writer.write("\nTop 20 频繁行为模式:")
        for i, (pattern, count) in enumerate(sorted_patterns[:20], 1):
            pattern_str = " → ".join([f"动作{action}" for action in pattern])
            support_rate = count / len(all_behavior_sequences) * 100
            writer.write(f"  {i:2d}. {pattern_str}: {count:,} 次 (支持度: {support_rate:.2f}%)")

        # 分析模式特征
        pattern_lengths = [len(pattern) for pattern in frequent_patterns.keys()]
        if pattern_lengths:
            writer.write_stats("模式特征分析", {
                "平均模式长度": f"{np.mean(pattern_lengths):.2f}",
                "最长模式": f"{max(pattern_lengths)} 个动作",
                "最短模式": f"{min(pattern_lengths)} 个动作"
            })

def analyze_feature_engineering_insights(global_stats, writer):
    """特征工程深度洞察"""
    writer.write_section("特征工程深度洞察", level=1)

    feature_interactions = global_stats.get('feature_interactions', {})
    if not feature_interactions:
        writer.write("无特征交互数据可分析")
        return

    # 分析特征点击效果
    feature_click_analysis = defaultdict(lambda: {'clicks': 0, 'exposures': 0})

    for combo_key, count in feature_interactions.items():
        if '_click' in combo_key:
            # 解析特征key
            feature_part = combo_key.replace('_click', '')
            parts = feature_part.split('_')
            if len(parts) >= 3:
                feature_key = '_'.join(parts[:2])  # user_xxx 或 item_xxx
                feature_click_analysis[feature_key]['clicks'] += count

    # 计算特征点击率
    feature_ctrs = {}
    for feature_key, stats in feature_click_analysis.items():
        if stats['exposures'] > 0:
            ctr = stats['clicks'] / stats['exposures']
            feature_ctrs[feature_key] = {
                'ctr': ctr,
                'clicks': stats['clicks'],
                'exposures': stats['exposures']
            }

    # 输出特征效果分析
    if feature_ctrs:
        sorted_features = sorted(feature_ctrs.items(), key=lambda x: x[1]['clicks'], reverse=True)

        writer.write_stats("特征点击效果分析", {
            "分析特征数": f"{len(feature_ctrs):,}",
            "总特征交互模式": f"{len(feature_interactions):,}"
        })

        writer.write("\nTop 10 高点击特征:")
        for i, (feature_key, stats) in enumerate(sorted_features[:10], 1):
            writer.write(f"  {i:2d}. {feature_key}: {stats['clicks']:,} 次点击")

    # 特征工程建议
    writer.write_section("特征工程建议", level=2)

    suggestions = [
        "1. 时间特征工程:",
        "   - 提取小时、星期、是否工作日、时间段等特征",
        "   - 构造点击间隔相关特征",
        "   - 用户活跃时间段偏好特征",
        "",
        "2. 序列特征工程:",
        "   - 用户历史行为序列embedding",
        "   - 行为转移概率特征",
        "   - 会话内行为统计特征",
        "",
        "3. 交互特征工程:",
        "   - 用户-物品历史交互特征",
        "   - 特征组合与点击率关联",
        "   - 协同过滤相关特征",
        "",
        "4. 统计特征工程:",
        "   - 用户点击率、活跃度等统计特征",
        "   - 物品热度、CTR等统计特征",
        "   - 特征值频次、重要性等特征"
    ]

    for suggestion in suggestions:
        writer.write(suggestion)

    # 模型架构建议
    writer.write_section("推荐模型架构", level=2)
    model_suggestions = [
        "1. 深度学习架构:",
        "   - Wide & Deep: 结合记忆和泛化能力",
        "   - DeepFM: 自动特征交叉学习",
        "   - PNN/DCN: 显式特征交叉网络",
        "",
        "2. 序列建模:",
        "   - GRU/LSTM: 建模用户行为序列",
        "   - Transformer: 捕获长期依赖关系",
        "   - BST: 结合序列和协同过滤",
        "",
        "3. 多任务学习:",
        "   - MMOE: 多任务专家网络",
        "   - PLE: 渐进式层次提取",
        "   - 同时优化CTR、CVR等多个目标"
    ]

    for suggestion in model_suggestions:
        writer.write(suggestion)

def analyze_seq_jsonl_optimized():
    """主分析函数 - 内存优化增强版本"""
    print("=" * 60)
    print("seq.jsonl 用户序列数据详细分析 (内存优化增强版本)")
    print("包含: 时间戳范围分析、特征工程洞察、用户会话分析、行为序列挖掘")
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

    # 获取总行数
    print("正在获取文件总行数...")
    total_lines = get_total_lines(seq_file)
    print(f"文件总行数: {total_lines:,}")

    # 初始化全局统计
    global_stats = {
        'total_users': 0,
        'total_clicks': 0,
        'total_exposures': 0,
        'users_with_clicks': 0,
        'inactive_users': 0,
        'low_activity_users': 0,
        'medium_activity_users': 0,
        'high_activity_users': 0,
        'item_click_counts': defaultdict(int),
        'item_exposure_counts': defaultdict(int),
        'click_timestamps': []
    }

    try:
        # 使用ReportWriter直接写入文件
        with ReportWriter("seq_jsonl_analysis", "seq.jsonl 用户序列数据详细分析报告 (内存优化版本)") as writer:
            # 写入文件信息
            writer.write_section("文件基本信息", level=1)
            writer.write(f"分析文件: {seq_file}")
            writer.write(f"文件大小: {file_size_mb:.2f} MB")
            writer.write(f"总行数: {total_lines:,}")
            writer.write(f"批次大小: {BATCH_SIZE:,}")
            writer.write(f"内存限制: {MAX_MEMORY_GB}GB")

            # 分批处理数据
            batch_num = 1
            start_line = 0
            total_processed = 0

            writer.write_section("分批处理进度", level=1)

            while start_line < total_lines:
                print(f"\n{'='*50}")
                print(f"开始处理批次 {batch_num} (从第 {start_line+1} 行开始)")
                print(f"{'='*50}")

                # 检查内存使用
                memory_gb = get_memory_usage()
                print(f"当前内存使用: {memory_gb:.2f}GB")

                if memory_gb > MAX_MEMORY_GB * 0.8:  # 80%预警
                    print(f"⚠️ 内存使用接近限制，执行垃圾回收...")
                    gc.collect()
                    memory_gb = get_memory_usage()
                    print(f"垃圾回收后内存使用: {memory_gb:.2f}GB")

                # 加载批次数据
                sequences, batch_records, next_start_line = load_sequence_data_batch(
                    seq_file, start_line, BATCH_SIZE
                )

                if not sequences:
                    print("没有更多数据可处理")
                    break

                total_processed += len(sequences)

                # 分析批次数据
                batch_seq_stats = analyze_batch_sequence_structure(sequences, writer, batch_num)
                batch_behavior_stats = analyze_batch_user_behavior(sequences, writer, batch_num, global_stats)

                # 清理批次数据，释放内存
                del sequences
                gc.collect()

                print(f"✓ 批次 {batch_num} 处理完成")
                print(f"  批次用户数: {batch_seq_stats['user_count']:,}")
                print(f"  累计处理用户数: {total_processed:,}")
                print(f"  处理进度: {total_processed/total_lines*100:.2f}%")

                # 更新位置
                start_line = next_start_line
                batch_num += 1

                # 内存检查
                memory_gb = get_memory_usage()
                print(f"  批次处理后内存使用: {memory_gb:.2f}GB")

                if memory_gb > MAX_MEMORY_GB:
                    print(f"❌ 内存使用超过限制 ({memory_gb:.2f}GB > {MAX_MEMORY_GB}GB)")
                    print("为防止系统崩溃，提前终止处理")
                    break

                # 如果达到文件末尾，退出
                if next_start_line >= total_lines:
                    break

            # 生成全局分析报告
            writer.write_section("完整数据集分析结果", level=1)
            writer.write(f"总处理批次数: {batch_num - 1}")
            writer.write(f"总处理用户数: {total_processed:,}")
            writer.write(f"处理完成度: {total_processed/total_lines*100:.2f}%")

            # 分析全局统计
            analyze_global_statistics(global_stats, writer)

            # 增强时间模式分析
            analyze_temporal_patterns(global_stats, writer)

            # 新增：用户会话分析
            analyze_user_sessions(global_stats, writer)

            # 新增：行为序列模式挖掘
            analyze_behavior_sequences(global_stats, writer)

            # 新增：特征工程深度洞察
            analyze_feature_engineering_insights(global_stats, writer)

            # 内存使用总结
            final_memory = get_memory_usage()
            writer.write_section("内存使用总结", level=1)
            writer.write(f"最终内存使用: {final_memory:.2f}GB")
            writer.write(f"内存使用是否在限制内: {'是' if final_memory <= MAX_MEMORY_GB else '否'}")

            # 获取报告路径
            report_path = writer.get_report_path()

        # 输出到日志
        print(f"\n输出详细分析报告到日志...")
        output_report_to_log(report_path)

        print("\n✅ seq.jsonl 序列数据分析完成（内存优化版本）！")
        print(f"总处理用户数: {total_processed:,}")
        print(f"最终内存使用: {get_memory_usage():.2f}GB")

    except Exception as e:
        print(f"❌ 序列数据分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    print("=" * 60)

if __name__ == "__main__":
    analyze_seq_jsonl_optimized()
