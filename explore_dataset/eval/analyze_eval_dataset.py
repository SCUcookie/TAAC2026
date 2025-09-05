#!/usr/bin/env python3
"""
评测数据集文件格式分析脚本
分析七个评测数据文件的存储格式、内容和统计信息
"""

import os
import json
import pickle
import struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
import numpy as np
from tqdm import tqdm
import sys
from collections import Counter, defaultdict
import time

def process_jsonl_chunk(args):
    """
    处理JSONL文件的一个分块 - 全局函数以支持多进程
    
    Args:
        args: (filepath, start_line, end_line) 元组
    
    Returns:
        dict: 分析结果
    """
    filepath, start_line, end_line = args
    
    local_analysis = {
        'field_counts': Counter(),
        'field_types': defaultdict(Counter),
        'sample_records': [],
        'record_count': 0
    }
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i < start_line:
                    continue
                if i >= end_line:
                    break
                    
                try:
                    record = json.loads(line.strip())
                    local_analysis['record_count'] += 1
                    
                    if local_analysis['record_count'] <= 3:
                        local_analysis['sample_records'].append(record)
                    
                    # 分析字段
                    for field, value in record.items():
                        local_analysis['field_counts'][field] += 1
                        local_analysis['field_types'][field][str(type(value))] += 1
                        
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return {'error': str(e)}
    
    # 转换为可序列化格式
    return {
        'record_count': local_analysis['record_count'],
        'field_counts': dict(local_analysis['field_counts']),
        'field_types': {
            field: dict(type_counter) 
            for field, type_counter in local_analysis['field_types'].items()
        },
        'sample_records': local_analysis['sample_records']
    }

class EvalDatasetAnalyzer:
    def __init__(self, eval_data_path, cache_path=None):
        self.eval_data_path = Path(eval_data_path)
        self.cache_path = Path(cache_path) if cache_path else Path("../../analysis_cache")
        self.cache_path.mkdir(exist_ok=True)
        
        # 预期文件列表
        self.expected_files = [
            'indexer.pkl',
            'item_feat_dict.json', 
            'item_feat_num.json',
            'predict_seq.jsonl',
            'predict_seq_offsets.pkl',
            'predict_set.jsonl',
            'user_feat_num.json'
        ]
        
        self.analysis_results = {}
    
    def check_file_existence(self):
        """检查所有预期文件是否存在"""
        print("=== 文件存在性检查 ===")
        missing_files = []
        existing_files = []
        
        for filename in self.expected_files:
            filepath = self.eval_data_path / filename
            if filepath.exists():
                existing_files.append(filename)
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"[OK] {filename} - {size_mb:.2f} MB")
            else:
                missing_files.append(filename)
                print(f"[MISSING] {filename} - file not found")
        
        return existing_files, missing_files
    
    def analyze_pickle_file(self, filename):
        """分析pickle文件"""
        print(f"\n=== 分析 {filename} ===")
        filepath = self.eval_data_path / filename
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            analysis = {
                'type': str(type(data)),
                'content_type': None,
                'size': len(data) if hasattr(data, '__len__') else 'N/A',
                'keys': None,
                'sample_data': None
            }
            
            if isinstance(data, dict):
                analysis['content_type'] = 'dictionary'
                analysis['keys'] = list(data.keys())[:10]  # 只显示前10个key
                analysis['key_count'] = len(data)
                
                # 分析每个key的内容
                key_analysis = {}
                for key in list(data.keys())[:5]:  # 只分析前5个key
                    if hasattr(data[key], '__len__'):
                        key_analysis[key] = {
                            'type': str(type(data[key])),
                            'length': len(data[key]) if hasattr(data[key], '__len__') else 'N/A',
                            'sample': str(data[key])[:100] if len(str(data[key])) > 100 else str(data[key])
                        }
                analysis['key_analysis'] = key_analysis
            
            elif isinstance(data, (list, tuple)):
                analysis['content_type'] = 'list/tuple'
                analysis['length'] = len(data)
                analysis['sample_data'] = str(data[:3]) if len(data) > 3 else str(data)
            
            # 保存到缓存
            cache_file = self.cache_path / f"{filename}_analysis.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            print(f"类型: {analysis['type']}")
            print(f"内容类型: {analysis['content_type']}")
            print(f"大小: {analysis['size']}")
            if analysis['keys']:
                print(f"键(前10个): {analysis['keys']}")
            
            return analysis
            
        except Exception as e:
            print(f"分析 {filename} 时出错: {e}")
            return {'error': str(e)}
    
    def analyze_json_file(self, filename):
        """分析JSON文件"""
        print(f"\n=== 分析 {filename} ===")
        filepath = self.eval_data_path / filename
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            analysis = {
                'type': str(type(data)),
                'size': len(data) if hasattr(data, '__len__') else 'N/A'
            }
            
            if isinstance(data, dict):
                analysis['key_count'] = len(data)
                analysis['sample_keys'] = list(data.keys())[:10]
                
                # 分析值的类型分布
                value_types = Counter()
                sample_values = {}
                
                for i, (key, value) in enumerate(data.items()):
                    value_types[str(type(value))] += 1
                    if i < 5:  # 只取前5个样本
                        if isinstance(value, dict):
                            sample_values[key] = {k: v for k, v in list(value.items())[:3]}
                        else:
                            sample_values[key] = value
                
                analysis['value_types'] = dict(value_types)
                analysis['sample_values'] = sample_values
            
            elif isinstance(data, list):
                analysis['length'] = len(data)
                if data:
                    analysis['first_item_type'] = str(type(data[0]))
                    analysis['sample_items'] = data[:3]
            
            # 保存到缓存
            cache_file = self.cache_path / f"{filename}_analysis.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            print(f"类型: {analysis['type']}")
            print(f"大小: {analysis.get('size', 'N/A')}")
            if 'key_count' in analysis:
                print(f"键数量: {analysis['key_count']}")
                print(f"值类型分布: {analysis.get('value_types', {})}")
            
            return analysis
            
        except Exception as e:
            print(f"分析 {filename} 时出错: {e}")
            return {'error': str(e)}
    
    def analyze_jsonl_file_parallel(self, filename):
        """并行分析JSONL文件"""
        print(f"\n=== 并行分析 {filename} ===")
        filepath = self.eval_data_path / filename
        
        try:
            # 先统计行数
            line_count = 0
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
            
            print(f"总行数: {line_count}")
            
            # 如果数据量不大，使用单进程处理
            if line_count < 10000:
                return self._analyze_jsonl_single_process(filepath)
            
            # 并行处理文件分块
            chunk_size = max(1000, line_count // min(cpu_count(), 32))  # 限制最大进程数
            
            # 创建分块任务
            chunks = []
            for start in range(0, line_count, chunk_size):
                end = min(start + chunk_size, line_count)
                chunks.append((str(filepath), start, end))
            
            print(f"使用 {min(cpu_count(), len(chunks))} 个进程处理 {len(chunks)} 个分块...")
            
            # 并行处理
            with ProcessPoolExecutor(max_workers=min(cpu_count(), len(chunks))) as executor:
                chunk_results = list(tqdm(
                    executor.map(process_jsonl_chunk, chunks),
                    total=len(chunks),
                    desc="处理分块"
                ))
            
            # 合并结果
            final_analysis = self._merge_chunk_results(chunk_results)
            
            # 保存到缓存
            cache_file = self.cache_path / f"{filename}_analysis.json"
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(final_analysis, f, indent=2, ensure_ascii=False)
            
            print(f"总记录数: {final_analysis['total_records']}")
            print(f"字段统计: {dict(list(final_analysis['field_counts'].items())[:10])}")
            print("字段类型:")
            for field, types in list(final_analysis['field_types'].items())[:5]:
                print(f"  {field}: {types}")
            
            return final_analysis
            
        except Exception as e:
            print(f"分析 {filename} 时出错: {e}")
            return {'error': str(e)}
    
    def _analyze_jsonl_single_process(self, filepath):
        """单进程分析JSONL文件"""
        analysis = {
            'field_counts': Counter(),
            'field_types': defaultdict(Counter),
            'sample_records': [],
            'record_count': 0
        }
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="分析记录"):
                try:
                    record = json.loads(line.strip())
                    analysis['record_count'] += 1
                    
                    if analysis['record_count'] <= 5:
                        analysis['sample_records'].append(record)
                    
                    # 分析字段
                    for field, value in record.items():
                        analysis['field_counts'][field] += 1
                        analysis['field_types'][field][str(type(value))] += 1
                        
                except json.JSONDecodeError:
                    continue
        
        return self._format_analysis_result(analysis)
    
    def _merge_chunk_results(self, chunk_results):
        """合并分块处理结果"""
        final_analysis = {
            'total_records': sum(r.get('record_count', 0) for r in chunk_results if r),
            'field_counts': Counter(),
            'field_types': defaultdict(Counter),
            'sample_records': []
        }
        
        for result in chunk_results:
            if not result:
                continue
                
            final_analysis['field_counts'].update(result.get('field_counts', {}))
            for field, type_counter in result.get('field_types', {}).items():
                final_analysis['field_types'][field].update(type_counter)
            
            if not final_analysis['sample_records'] and result.get('sample_records'):
                final_analysis['sample_records'] = result['sample_records']
        
        return self._format_analysis_result(final_analysis)
    
    def _format_analysis_result(self, analysis):
        """格式化分析结果为JSON可序列化格式"""
        return {
            'total_records': analysis.get('record_count', analysis.get('total_records', 0)),
            'field_counts': dict(analysis['field_counts']),
            'field_types': {
                field: dict(type_counter) 
                for field, type_counter in analysis['field_types'].items()
            },
            'sample_records': analysis.get('sample_records', [])
        }
    
    def analyze_all_files(self):
        """分析所有文件"""
        print("开始分析评测数据集文件...")
        start_time = time.time()
        
        existing_files, missing_files = self.check_file_existence()
        
        self.analysis_results['missing_files'] = missing_files
        self.analysis_results['file_analyses'] = {}
        
        for filename in existing_files:
            if filename.endswith('.pkl'):
                self.analysis_results['file_analyses'][filename] = self.analyze_pickle_file(filename)
            elif filename.endswith('.json'):
                self.analysis_results['file_analyses'][filename] = self.analyze_json_file(filename)
            elif filename.endswith('.jsonl'):
                self.analysis_results['file_analyses'][filename] = self.analyze_jsonl_file_parallel(filename)
        
        # 保存完整分析结果
        summary_file = self.cache_path / "complete_analysis.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        
        elapsed = time.time() - start_time
        print(f"\n=== 分析完成 ===")
        print(f"总耗时: {elapsed:.2f} 秒")
        print(f"结果已保存到: {self.cache_path}")
        
        return self.analysis_results
    
    def generate_summary_report(self):
        """生成分析摘要报告"""
        print("\n" + "="*50)
        print("评测数据集文件分析摘要报告")
        print("="*50)
        
        print("\n1. 文件对应关系:")
        correspondences = {
            'indexer.pkl': '索引字典 - 与训练数据相同',
            'item_feat_dict.json': '物品特征字典 - 与训练数据相同',
            'item_feat_num.json': '物品特征数量统计 - 与训练数据相同',
            'predict_seq.jsonl': '用户预测序列 - 对应训练数据的seq.jsonl',
            'predict_seq_offsets.pkl': '预测序列偏移量 - 对应训练数据的seq_offsets.pkl',
            'predict_set.jsonl': '候选推荐集合 - 评测数据独有',
            'user_feat_num.json': '用户特征数量统计 - 与训练数据相同'
        }
        
        for filename, description in correspondences.items():
            status = "[OK]" if filename in self.analysis_results.get('file_analyses', {}) else "[MISSING]"
            print(f"  {status} {filename}: {description}")
        
        print("\n2. 缺失文件分析:")
        if self.analysis_results.get('missing_files'):
            for filename in self.analysis_results['missing_files']:
                print(f"  [MISSING] {filename} - may affect inference process")
        else:
            print("  [OK] All expected files exist")
        
        print("\n3. 关键差异:")
        print("  - predict_set.jsonl: 这是评测数据独有的文件，包含候选推荐物品集合")
        print("  - predict_seq.jsonl vs seq.jsonl: 结构相似但用于推理而非训练")
        print("  - 缺少多模态embedding文件夹(creative_emb): 可能需要从训练阶段复用")


def main():
    # 设置数据路径
    eval_data_path = os.environ.get('EVAL_DATA_PATH', '../../dataset/TencentGR_1k')
    cache_path = os.environ.get('USER_CACHE_PATH', '../../analysis_cache')
    
    print(f"评测数据路径: {eval_data_path}")
    print(f"缓存路径: {cache_path}")
    
    # 检查路径是否存在
    if not os.path.exists(eval_data_path):
        print(f"错误: 评测数据路径不存在: {eval_data_path}")
        print("请设置正确的 EVAL_DATA_PATH 环境变量")
        return
    
    # 创建分析器并执行分析
    analyzer = EvalDatasetAnalyzer(eval_data_path, cache_path)
    
    try:
        # 执行完整分析
        results = analyzer.analyze_all_files()
        
        # 生成摘要报告
        analyzer.generate_summary_report()
        
        print(f"\n详细分析结果已保存到: {cache_path}")
        
    except KeyboardInterrupt:
        print("\n分析被用户中断")
    except Exception as e:
        print(f"\n分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()