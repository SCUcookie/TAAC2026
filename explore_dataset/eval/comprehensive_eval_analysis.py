#!/usr/bin/env python3
"""
评测数据集全面分析脚本
对所有7个评测数据文件进行详细分析，生成完整的分析报告
"""

import os
import json
import pickle
import struct
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
import time
import sys

class ComprehensiveEvalAnalyzer:
    def __init__(self, eval_data_path, output_dir):
        self.eval_data_path = Path(eval_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 预期文件列表
        self.files_to_analyze = {
            'indexer.pkl': self.analyze_indexer,
            'item_feat_dict.json': self.analyze_item_feat_dict,
            'item_feat_num.json': self.analyze_item_feat_num,
            'predict_seq.jsonl': self.analyze_predict_seq,
            'predict_seq_offsets.pkl': self.analyze_predict_seq_offsets,
            'predict_set.jsonl': self.analyze_predict_set,
            'user_feat_num.json': self.analyze_user_feat_num
        }
        
        self.analysis_results = {}
        
    def analyze_indexer(self, filepath):
        """分析indexer.pkl文件"""
        print(f"\n=== 分析 indexer.pkl ===")
        
        try:
            with open(filepath, 'rb') as f:
                indexer = pickle.load(f)
            
            analysis = {
                'file_type': 'pickle',
                'data_type': str(type(indexer)),
                'main_keys': list(indexer.keys()) if isinstance(indexer, dict) else 'Not a dict',
                'detailed_analysis': {}
            }
            
            if isinstance(indexer, dict):
                for key, value in indexer.items():
                    if isinstance(value, dict):
                        analysis['detailed_analysis'][key] = {
                            'type': 'dictionary',
                            'size': len(value),
                            'sample_keys': list(value.keys())[:10],
                            'sample_values': list(value.values())[:5]
                        }
                        
                        # 特殊分析每个字典的内容
                        if key == 'i':  # item indexer
                            analysis['detailed_analysis'][key]['description'] = '物品ID映射字典'
                            analysis['detailed_analysis'][key]['id_range'] = f"[{min(value.values())}, {max(value.values())}]" if value else "空"
                        elif key == 'u':  # user indexer  
                            analysis['detailed_analysis'][key]['description'] = '用户ID映射字典'
                            analysis['detailed_analysis'][key]['id_range'] = f"[{min(value.values())}, {max(value.values())}]" if value else "空"
                        elif key == 'f':  # feature indexer
                            analysis['detailed_analysis'][key]['description'] = '特征值映射字典'
                            # 分析每个特征的词汇表大小
                            if isinstance(value, dict):
                                feature_vocab_sizes = {feat_id: len(vocab) for feat_id, vocab in value.items()}
                                analysis['detailed_analysis'][key]['feature_vocab_sizes'] = feature_vocab_sizes
                        elif key == 'a':  # action indexer
                            analysis['detailed_analysis'][key]['description'] = '行为类型映射字典'
                    else:
                        analysis['detailed_analysis'][key] = {
                            'type': str(type(value)),
                            'value': str(value)[:100]
                        }
            
            print(f"   索引器类型: {analysis['data_type']}")
            print(f"   主要键: {analysis['main_keys']}")
            
            if 'i' in indexer:
                print(f"   物品索引数量: {len(indexer['i']):,}")
            if 'u' in indexer:
                print(f"   用户索引数量: {len(indexer['u']):,}")
            if 'f' in indexer:
                print(f"   特征索引数量: {len(indexer['f'])}")
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_item_feat_dict(self, filepath):
        """分析item_feat_dict.json文件"""
        print(f"\n=== 分析 item_feat_dict.json ===")
        
        try:
            print("   加载物品特征字典（大文件，请稍候）...")
            with open(filepath, 'r', encoding='utf-8') as f:
                item_feat_dict = json.load(f)
            
            # 基础统计
            total_items = len(item_feat_dict)
            sample_item_id = list(item_feat_dict.keys())[0]
            sample_features = item_feat_dict[sample_item_id]
            
            # 特征分析
            feature_stats = defaultdict(list)
            feature_types = defaultdict(set)
            feature_completeness = defaultdict(int)
            
            print("   分析特征分布...")
            sample_size = min(10000, total_items)  # 抽样分析以节省时间
            sampled_items = dict(list(item_feat_dict.items())[:sample_size])
            
            for item_id, features in tqdm(sampled_items.items(), desc="   处理样本"):
                for feat_id, feat_value in features.items():
                    feature_completeness[feat_id] += 1
                    feature_types[feat_id].add(type(feat_value).__name__)
                    
                    if isinstance(feat_value, (int, float)):
                        feature_stats[feat_id].append(feat_value)
            
            # 计算特征统计
            feature_analysis = {}
            for feat_id, values in feature_stats.items():
                if values:
                    values_array = np.array(values)
                    feature_analysis[feat_id] = {
                        'count': len(values),
                        'completeness_rate': feature_completeness[feat_id] / sample_size * 100,
                        'data_types': list(feature_types[feat_id]),
                        'mean': float(np.mean(values_array)),
                        'std': float(np.std(values_array)),
                        'min': float(np.min(values_array)),
                        'max': float(np.max(values_array)),
                        'unique_count': len(np.unique(values_array))
                    }
            
            analysis = {
                'file_type': 'json',
                'total_items': total_items,
                'sample_item_id': sample_item_id,
                'feature_count': len(sample_features),
                'feature_list': list(sample_features.keys()),
                'sample_features': dict(list(sample_features.items())[:5]),
                'feature_analysis': feature_analysis,
                'sampling_info': {
                    'sample_size': sample_size,
                    'total_size': total_items,
                    'sampling_rate': sample_size / total_items * 100
                }
            }
            
            print(f"   物品总数: {analysis['total_items']:,}")
            print(f"   特征字段数: {analysis['feature_count']}")
            print(f"   特征列表: {analysis['feature_list']}")
            print(f"   采样分析: {sample_size:,}/{total_items:,} ({analysis['sampling_info']['sampling_rate']:.1f}%)")
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_item_feat_num(self, filepath):
        """分析item_feat_num.json文件"""
        print(f"\n=== 分析 item_feat_num.json ===")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                item_feat_num = json.load(f)
            
            analysis = {
                'file_type': 'json',
                'total_features': len(item_feat_num),
                'feature_vocab_sizes': item_feat_num,
                'statistics': {
                    'min_vocab_size': min(item_feat_num.values()) if item_feat_num else 0,
                    'max_vocab_size': max(item_feat_num.values()) if item_feat_num else 0,
                    'avg_vocab_size': np.mean(list(item_feat_num.values())) if item_feat_num else 0,
                    'total_vocab_size': sum(item_feat_num.values()) if item_feat_num else 0
                },
                'feature_categorization': {}
            }
            
            # 按词汇表大小分类特征
            for feat_id, vocab_size in item_feat_num.items():
                if vocab_size <= 50:
                    category = 'low_cardinality'
                elif vocab_size <= 1000:
                    category = 'medium_cardinality'
                else:
                    category = 'high_cardinality'
                
                if category not in analysis['feature_categorization']:
                    analysis['feature_categorization'][category] = []
                analysis['feature_categorization'][category].append({
                    'feature_id': feat_id,
                    'vocab_size': vocab_size
                })
            
            print(f"   特征总数: {analysis['total_features']}")
            print(f"   词汇表大小统计: 最小={analysis['statistics']['min_vocab_size']}, 最大={analysis['statistics']['max_vocab_size']}, 平均={analysis['statistics']['avg_vocab_size']:.1f}")
            print(f"   特征分类:")
            for category, features in analysis['feature_categorization'].items():
                print(f"     {category}: {len(features)}个特征")
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_predict_seq(self, filepath):
        """分析predict_seq.jsonl文件"""
        print(f"\n=== 分析 predict_seq.jsonl ===")
        
        try:
            file_size = filepath.stat().st_size
            file_size_gb = file_size / (1024**3)
            
            print(f"   文件大小: {file_size_gb:.2f} GB")
            
            # 尝试读取前几行
            analysis = {
                'file_type': 'jsonl',
                'file_size_bytes': file_size,
                'file_size_gb': file_size_gb,
                'encoding_tests': {},
                'sample_records': [],
                'line_count_estimate': 'unknown',
                'potential_issues': []
            }
            
            # 测试不同编码
            encodings = ['utf-8', 'gbk', 'latin1', 'cp1252']
            successful_lines = 0
            
            for encoding in encodings:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        lines_read = []
                        error_count = 0
                        
                        for i, line in enumerate(f):
                            if i >= 10:  # 只读前10行进行测试
                                break
                            try:
                                record = json.loads(line.strip())
                                lines_read.append(record)
                                if i == 0 and encoding == 'utf-8':  # 保存第一个成功解析的记录
                                    analysis['sample_records'].append(record)
                            except json.JSONDecodeError:
                                error_count += 1
                                continue
                        
                        analysis['encoding_tests'][encoding] = {
                            'lines_read': len(lines_read),
                            'json_errors': error_count,
                            'success': len(lines_read) > 0
                        }
                        
                        if len(lines_read) > successful_lines:
                            successful_lines = len(lines_read)
                            
                except Exception as e:
                    analysis['encoding_tests'][encoding] = {'error': str(e)}
            
            # 尝试估计总行数（通过采样）
            try:
                with open(filepath, 'rb') as f:
                    # 读取前1MB进行行数估计
                    sample_size = min(1024*1024, file_size)
                    sample_data = f.read(sample_size)
                    newline_count = sample_data.count(b'\n')
                    
                    if newline_count > 0:
                        estimated_total_lines = int((file_size / sample_size) * newline_count)
                        analysis['line_count_estimate'] = estimated_total_lines
            except:
                pass
            
            # 分析潜在问题
            if successful_lines == 0:
                analysis['potential_issues'].append("无法使用标准编码读取JSON记录")
            if file_size > 10 * 1024**3:  # 大于10GB
                analysis['potential_issues'].append("文件过大，可能需要流式处理")
            
            print(f"   编码测试结果:")
            for enc, result in analysis['encoding_tests'].items():
                if 'error' in result:
                    print(f"     {enc}: 失败 - {result['error']}")
                else:
                    print(f"     {enc}: 成功读取{result['lines_read']}行，JSON错误{result['json_errors']}个")
            
            if analysis['line_count_estimate'] != 'unknown':
                print(f"   估计总行数: {analysis['line_count_estimate']:,}")
            
            if analysis['potential_issues']:
                print(f"   潜在问题:")
                for issue in analysis['potential_issues']:
                    print(f"     - {issue}")
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_predict_seq_offsets(self, filepath):
        """分析predict_seq_offsets.pkl文件"""
        print(f"\n=== 分析 predict_seq_offsets.pkl ===")
        
        try:
            with open(filepath, 'rb') as f:
                offsets = pickle.load(f)
            
            analysis = {
                'file_type': 'pickle',
                'data_type': str(type(offsets)),
                'length': len(offsets) if hasattr(offsets, '__len__') else 'unknown',
                'sample_data': [],
                'statistics': {}
            }
            
            if isinstance(offsets, list):
                analysis['sample_data'] = offsets[:20]  # 前20个偏移量
                
                if offsets:
                    offsets_array = np.array(offsets)
                    analysis['statistics'] = {
                        'min_offset': int(np.min(offsets_array)),
                        'max_offset': int(np.max(offsets_array)),
                        'mean_offset': float(np.mean(offsets_array)),
                        'offset_range': int(np.max(offsets_array) - np.min(offsets_array)),
                        'avg_record_size': int(np.mean(np.diff(offsets))) if len(offsets) > 1 else 0
                    }
                    
                    # 检查偏移量是否有序
                    is_sorted = all(offsets[i] <= offsets[i+1] for i in range(len(offsets)-1))
                    analysis['is_sorted'] = is_sorted
                    
                    # 计算记录大小分布
                    if len(offsets) > 1:
                        record_sizes = np.diff(offsets)
                        analysis['record_size_stats'] = {
                            'min_record_size': int(np.min(record_sizes)),
                            'max_record_size': int(np.max(record_sizes)),
                            'std_record_size': float(np.std(record_sizes))
                        }
            
            print(f"   数据类型: {analysis['data_type']}")
            print(f"   偏移量数量: {analysis['length']:,}")
            print(f"   偏移量范围: {analysis['statistics'].get('min_offset', 'N/A')} - {analysis['statistics'].get('max_offset', 'N/A')}")
            print(f"   是否有序: {analysis.get('is_sorted', 'N/A')}")
            print(f"   平均记录大小: {analysis['statistics'].get('avg_record_size', 'N/A')} bytes")
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_predict_set(self, filepath):
        """分析predict_set.jsonl文件（基于之前的详细分析）"""
        print(f"\n=== 分析 predict_set.jsonl ===")
        
        try:
            # 检查是否已有详细分析结果
            existing_analysis_file = self.output_dir / 'predict_set_detailed_analysis.json'
            if existing_analysis_file.exists():
                print("   发现现有分析结果，加载中...")
                with open(existing_analysis_file, 'r', encoding='utf-8') as f:
                    detailed_analysis = json.load(f)
                print("   使用现有的详细分析结果")
                return detailed_analysis
            
            # 如果没有现有结果，进行简化分析
            line_count = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    if line_count >= 100000:  # 只计算前10万行来节省时间
                        break
            
            analysis = {
                'file_type': 'jsonl',
                'estimated_records': line_count,
                'note': '这是简化分析，详细分析请查看predict_set_detailed_analysis.json'
            }
            
            print(f"   估计记录数: {analysis['estimated_records']:,}+")
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_user_feat_num(self, filepath):
        """分析user_feat_num.json文件"""
        print(f"\n=== 分析 user_feat_num.json ===")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                user_feat_num = json.load(f)
            
            analysis = {
                'file_type': 'json',
                'total_features': len(user_feat_num),
                'feature_vocab_sizes': user_feat_num,
                'statistics': {
                    'min_vocab_size': min(user_feat_num.values()) if user_feat_num else 0,
                    'max_vocab_size': max(user_feat_num.values()) if user_feat_num else 0,
                    'avg_vocab_size': np.mean(list(user_feat_num.values())) if user_feat_num else 0,
                    'total_vocab_size': sum(user_feat_num.values()) if user_feat_num else 0
                },
                'feature_categorization': {}
            }
            
            # 按词汇表大小分类特征
            for feat_id, vocab_size in user_feat_num.items():
                if vocab_size <= 50:
                    category = 'low_cardinality'
                elif vocab_size <= 1000:
                    category = 'medium_cardinality'
                else:
                    category = 'high_cardinality'
                
                if category not in analysis['feature_categorization']:
                    analysis['feature_categorization'][category] = []
                analysis['feature_categorization'][category].append({
                    'feature_id': feat_id,
                    'vocab_size': vocab_size
                })
            
            # 与item特征对比（如果有的话）
            item_feat_num_file = self.eval_data_path / 'item_feat_num.json'
            if item_feat_num_file.exists():
                try:
                    with open(item_feat_num_file, 'r', encoding='utf-8') as f:
                        item_feat_num = json.load(f)
                    
                    user_features = set(user_feat_num.keys())
                    item_features = set(item_feat_num.keys())
                    
                    analysis['comparison_with_item_features'] = {
                        'user_only_features': list(user_features - item_features),
                        'item_only_features': list(item_features - user_features),
                        'common_features': list(user_features & item_features),
                        'user_feature_count': len(user_features),
                        'item_feature_count': len(item_features)
                    }
                except:
                    pass
            
            print(f"   用户特征总数: {analysis['total_features']}")
            print(f"   词汇表大小统计: 最小={analysis['statistics']['min_vocab_size']}, 最大={analysis['statistics']['max_vocab_size']}, 平均={analysis['statistics']['avg_vocab_size']:.1f}")
            print(f"   用户特征分类:")
            for category, features in analysis['feature_categorization'].items():
                print(f"     {category}: {len(features)}个特征")
            
            if 'comparison_with_item_features' in analysis:
                comp = analysis['comparison_with_item_features']
                print(f"   与物品特征对比:")
                print(f"     用户独有特征: {len(comp['user_only_features'])}个")
                print(f"     物品独有特征: {len(comp['item_only_features'])}个")
                print(f"     共同特征: {len(comp['common_features'])}个")
            
            return analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_all_files(self):
        """分析所有文件"""
        print("=== 开始全面分析评测数据集 ===")
        start_time = time.time()
        
        for filename, analyzer_func in self.files_to_analyze.items():
            filepath = self.eval_data_path / filename
            
            if not filepath.exists():
                print(f"\n[MISSING] {filename} - file not found")
                self.analysis_results[filename] = {'error': 'File not found'}
                continue
                
            try:
                self.analysis_results[filename] = analyzer_func(filepath)
            except Exception as e:
                print(f"\n[ERROR] Failed to analyze {filename}: {e}")
                self.analysis_results[filename] = {'error': str(e)}
        
        # 生成综合报告
        self.generate_comprehensive_report()
        
        # 保存完整结果
        result_file = self.output_dir / 'comprehensive_eval_analysis.json'
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        elapsed = time.time() - start_time
        print(f"\n=== 分析完成 ===")
        print(f"总耗时: {elapsed:.2f} 秒")
        print(f"结果已保存到: {self.output_dir}")
        
        return self.analysis_results
    
    def generate_comprehensive_report(self):
        """生成综合报告"""
        report_file = self.output_dir / 'comprehensive_eval_analysis_report.md'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 评测数据集全面分析报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 文件存在性总览
            f.write("## 📁 文件存在性检查\n\n")
            for filename in self.files_to_analyze.keys():
                if filename in self.analysis_results and 'error' not in self.analysis_results[filename]:
                    f.write(f"- [OK] {filename}\n")
                else:
                    f.write(f"- [MISSING] {filename}\n")
            f.write("\n")
            
            # 每个文件的详细分析
            for filename, analysis in self.analysis_results.items():
                if 'error' in analysis:
                    f.write(f"## [ERROR] {filename}\n\n")
                    f.write(f"错误: {analysis['error']}\n\n")
                    continue
                
                f.write(f"## [FILE] {filename}\n\n")
                
                if filename == 'indexer.pkl':
                    self._write_indexer_report(f, analysis)
                elif filename == 'item_feat_dict.json':
                    self._write_item_feat_dict_report(f, analysis)
                elif filename == 'item_feat_num.json':
                    self._write_feat_num_report(f, analysis, "物品特征")
                elif filename == 'user_feat_num.json':
                    self._write_feat_num_report(f, analysis, "用户特征")
                elif filename == 'predict_seq.jsonl':
                    self._write_predict_seq_report(f, analysis)
                elif filename == 'predict_seq_offsets.pkl':
                    self._write_offsets_report(f, analysis)
                elif filename == 'predict_set.jsonl':
                    self._write_predict_set_report(f, analysis)
                
                f.write("\n")
            
            # 总结和建议
            f.write("## 🎯 总结与建议\n\n")
            self._write_summary_and_recommendations(f)
    
    def _write_indexer_report(self, f, analysis):
        f.write("**索引映射字典**\n\n")
        f.write(f"- 数据类型: {analysis['data_type']}\n")
        f.write(f"- 主要键: {analysis['main_keys']}\n\n")
        
        for key, details in analysis['detailed_analysis'].items():
            if isinstance(details, dict) and 'description' in details:
                f.write(f"### {key} - {details['description']}\n")
                f.write(f"- 大小: {details['size']:,}\n")
                if 'id_range' in details:
                    f.write(f"- ID范围: {details['id_range']}\n")
                if 'feature_vocab_sizes' in details:
                    f.write("- 特征词汇表大小:\n")
                    for feat_id, size in list(details['feature_vocab_sizes'].items())[:10]:
                        f.write(f"  - 特征{feat_id}: {size}\n")
                f.write("\n")
    
    def _write_item_feat_dict_report(self, f, analysis):
        f.write("**物品特征字典**\n\n")
        f.write(f"- 物品总数: {analysis['total_items']:,}\n")
        f.write(f"- 特征字段数: {analysis['feature_count']}\n")
        f.write(f"- 采样分析: {analysis['sampling_info']['sample_size']:,}/{analysis['sampling_info']['total_size']:,} ({analysis['sampling_info']['sampling_rate']:.1f}%)\n\n")
        
        f.write("### 特征分析\n")
        for feat_id, stats in analysis['feature_analysis'].items():
            f.write(f"**特征 {feat_id}**\n")
            f.write(f"- 完整率: {stats['completeness_rate']:.2f}%\n")
            f.write(f"- 数据类型: {stats['data_types']}\n")
            f.write(f"- 统计: 均值={stats['mean']:.2f}, 标准差={stats['std']:.2f}\n")
            f.write(f"- 范围: [{stats['min']}, {stats['max']}]\n")
            f.write(f"- 唯一值: {stats['unique_count']:,}\n\n")
    
    def _write_feat_num_report(self, f, analysis, feature_type):
        f.write(f"**{feature_type}词汇表配置**\n\n")
        f.write(f"- 特征总数: {analysis['total_features']}\n")
        f.write(f"- 词汇表大小统计: 最小={analysis['statistics']['min_vocab_size']}, 最大={analysis['statistics']['max_vocab_size']}, 平均={analysis['statistics']['avg_vocab_size']:.1f}\n\n")
        
        f.write("### 特征分类\n")
        for category, features in analysis['feature_categorization'].items():
            f.write(f"**{category}** ({len(features)}个特征):\n")
            for feat in features[:5]:  # 只显示前5个
                f.write(f"- 特征{feat['feature_id']}: {feat['vocab_size']}个取值\n")
            if len(features) > 5:
                f.write(f"- ... 还有 {len(features)-5} 个特征\n")
            f.write("\n")
        
        if 'comparison_with_item_features' in analysis:
            comp = analysis['comparison_with_item_features']
            f.write("### 与物品特征对比\n")
            f.write(f"- 用户独有特征: {len(comp['user_only_features'])}个\n")
            f.write(f"- 物品独有特征: {len(comp['item_only_features'])}个\n")
            f.write(f"- 共同特征: {len(comp['common_features'])}个\n\n")
    
    def _write_predict_seq_report(self, f, analysis):
        f.write("**用户预测序列数据**\n\n")
        f.write(f"- 文件大小: {analysis['file_size_gb']:.2f} GB\n")
        if analysis['line_count_estimate'] != 'unknown':
            f.write(f"- 估计记录数: {analysis['line_count_estimate']:,}\n")
        
        f.write("\n### 编码测试结果\n")
        for encoding, result in analysis['encoding_tests'].items():
            if 'error' in result:
                f.write(f"- {encoding}: ❌ {result['error']}\n")
            else:
                f.write(f"- {encoding}: {'✅' if result['success'] else '❌'} 读取{result['lines_read']}行，JSON错误{result['json_errors']}个\n")
        
        if analysis['potential_issues']:
            f.write("\n### 潜在问题\n")
            for issue in analysis['potential_issues']:
                f.write(f"- {issue}\n")
    
    def _write_offsets_report(self, f, analysis):
        f.write("**序列偏移量索引**\n\n")
        f.write(f"- 偏移量数量: {analysis['length']:,}\n")
        f.write(f"- 偏移量范围: {analysis['statistics'].get('min_offset', 'N/A')} - {analysis['statistics'].get('max_offset', 'N/A')}\n")
        f.write(f"- 是否有序: {'✅' if analysis.get('is_sorted') else '❌'}\n")
        f.write(f"- 平均记录大小: {analysis['statistics'].get('avg_record_size', 'N/A')} bytes\n")
        
        if 'record_size_stats' in analysis:
            stats = analysis['record_size_stats']
            f.write(f"- 记录大小分布: 最小={stats['min_record_size']}B, 最大={stats['max_record_size']}B, 标准差={stats['std_record_size']:.1f}B\n")
    
    def _write_predict_set_report(self, f, analysis):
        if 'basic_stats' in analysis:
            f.write("**候选推荐集合**\n\n")
            f.write(f"- 总记录数: {analysis['basic_stats']['total_records']:,}\n")
            f.write(f"- 唯一creative_id: {analysis['basic_stats']['unique_creative_ids']:,}\n")
            f.write("- 详细分析请查看 predict_set_comprehensive_analysis.md\n")
        else:
            f.write("**候选推荐集合（简化分析）**\n\n")
            f.write(f"- 估计记录数: {analysis.get('estimated_records', 'unknown'):,}+\n")
            f.write("- 详细分析请运行专门的predict_set分析脚本\n")
    
    def _write_summary_and_recommendations(self, f):
        f.write("### 数据质量评估\n\n")
        
        # 检查各种问题
        issues = []
        recommendations = []
        
        # 检查predict_seq.jsonl问题
        if 'predict_seq.jsonl' in self.analysis_results:
            seq_analysis = self.analysis_results['predict_seq.jsonl']
            if 'potential_issues' in seq_analysis and seq_analysis['potential_issues']:
                issues.extend(seq_analysis['potential_issues'])
                recommendations.append("对predict_seq.jsonl使用专门的编码处理和流式读取")
        
        # 检查特征一致性
        if 'item_feat_num.json' in self.analysis_results and 'user_feat_num.json' in self.analysis_results:
            recommendations.append("验证用户特征和物品特征的一致性，确保模型训练和推理的特征对齐")
        
        recommendations.extend([
            "使用并行处理来优化大文件的读取和分析",
            "实现增量分析以节省重复计算时间",
            "建立数据质量监控机制，及时发现数据异常"
        ])
        
        if issues:
            f.write("#### 发现的问题\n")
            for issue in issues:
                f.write(f"- {issue}\n")
            f.write("\n")
        
        f.write("#### 优化建议\n")
        for rec in recommendations:
            f.write(f"- {rec}\n")

def main():
    eval_data_path = os.environ.get('EVAL_DATA_PATH')
    output_dir = os.environ.get('USER_CACHE_PATH')
    
    print(f"评测数据路径: {eval_data_path}")
    print(f"输出目录: {output_dir}")
    
    # 检查路径是否存在
    if not os.path.exists(eval_data_path):
        print(f"错误: 评测数据路径不存在: {eval_data_path}")
        return
    
    # 创建分析器并执行分析
    analyzer = ComprehensiveEvalAnalyzer(eval_data_path, output_dir)
    
    try:
        results = analyzer.analyze_all_files()
        print(f"\n[SUCCESS] Analysis completed! Results saved to {output_dir}")
        return results
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Analysis interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()