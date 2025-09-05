#!/usr/bin/env python3
"""
精确分析评测数据集文件的内部结构
解答关键问题：各个字典类型的具体含义和predict_seq.jsonl记录数为0的原因
"""

import os
import json
import pickle
from pathlib import Path
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
import sys

def analyze_file_structures():
    """精确分析各个文件的内部结构"""
    
    eval_data_path = os.environ.get('EVAL_DATA_PATH', '../../dataset/TencentGR_1k')
    cache_path = os.environ.get('USER_CACHE_PATH', '../../analysis_cache')
    
    eval_dir = Path(eval_data_path)
    cache_dir = Path(cache_path)
    
    # 创建输出目录
    output_dir = Path('')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print("=== 精确分析评测数据文件结构 ===\n")
    
    # 1. 分析 item_feat_dict.json 中的 <class 'dict'> 
    print("1. 分析 item_feat_dict.json...")
    item_feat_file = eval_dir / 'item_feat_dict.json'
    if item_feat_file.exists():
        try:
            print("   加载item_feat_dict.json（大文件，请稍候）...")
            with open(item_feat_file, 'r', encoding='utf-8') as f:
                item_feat_dict = json.load(f)
            
            # 分析字典结构
            sample_creative_id = list(item_feat_dict.keys())[0]
            sample_features = item_feat_dict[sample_creative_id]
            
            analysis = {
                'total_items': len(item_feat_dict),
                'sample_creative_id': sample_creative_id,
                'feature_fields': list(sample_features.keys()),
                'feature_count': len(sample_features),
                'feature_types': {k: str(type(v)) for k, v in sample_features.items()},
                'sample_values': {k: v for k, v in list(sample_features.items())[:5]}
            }
            
            results['item_feat_dict'] = analysis
            
            print(f"   - 总物品数: {analysis['total_items']:,}")
            print(f"   - 样本creative_id: {analysis['sample_creative_id']}")
            print(f"   - 特征字段数: {analysis['feature_count']}")
            print(f"   - 特征字段: {analysis['feature_fields']}")
            print(f"   - <class 'dict'>含义: 每个creative_id对应一个特征字典，包含{analysis['feature_count']}个特征")
            
        except Exception as e:
            results['item_feat_dict'] = {'error': str(e)}
            print(f"   分析失败: {e}")
    
    # 2. 分析 item_feat_num.json 中的内容
    print("\n2. 分析 item_feat_num.json...")
    item_feat_num_file = eval_dir / 'item_feat_num.json'
    if item_feat_num_file.exists():
        try:
            with open(item_feat_num_file, 'r', encoding='utf-8') as f:
                item_feat_num = json.load(f)
            
            analysis = {
                'total_features': len(item_feat_num),
                'feature_configs': item_feat_num,
                'feature_details': {k: f"特征{k}有{v}个不同取值" for k, v in item_feat_num.items()}
            }
            
            results['item_feat_num'] = analysis
            
            print(f"   - 特征配置数: {analysis['total_features']}")
            print(f"   - 配置内容: {analysis['feature_configs']}")
            print(f"   - <class 'int'>含义: 每个特征ID对应该特征的不同取值数量（词汇表大小）")
            
        except Exception as e:
            results['item_feat_num'] = {'error': str(e)}
            print(f"   分析失败: {e}")
    
    # 3. 分析 predict_seq_offsets.pkl 
    print("\n3. 分析 predict_seq_offsets.pkl...")
    seq_offsets_file = eval_dir / 'predict_seq_offsets.pkl'
    if seq_offsets_file.exists():
        try:
            with open(seq_offsets_file, 'rb') as f:
                seq_offsets = pickle.load(f)
            
            analysis = {
                'data_type': str(type(seq_offsets)),
                'total_users': len(seq_offsets),
                'sample_offsets': seq_offsets[:10] if len(seq_offsets) > 10 else seq_offsets,
                'offset_range': f"[{min(seq_offsets)}, {max(seq_offsets)}]" if seq_offsets else "空"
            }
            
            results['predict_seq_offsets'] = analysis
            
            print(f"   - 数据类型: {analysis['data_type']}")
            print(f"   - 用户数量: {analysis['total_users']:,}")
            print(f"   - 样本偏移量: {analysis['sample_offsets']}")
            print(f"   - 偏移量范围: {analysis['offset_range']}")
            print(f"   - <class 'list'>含义: 每个用户在predict_seq.jsonl文件中的字节偏移位置")
            
        except Exception as e:
            results['predict_seq_offsets'] = {'error': str(e)}
            print(f"   分析失败: {e}")
    
    # 4. 分析 user_feat_num.json
    print("\n4. 分析 user_feat_num.json...")
    user_feat_num_file = eval_dir / 'user_feat_num.json'
    if user_feat_num_file.exists():
        try:
            with open(user_feat_num_file, 'r', encoding='utf-8') as f:
                user_feat_num = json.load(f)
            
            analysis = {
                'total_features': len(user_feat_num),
                'feature_configs': user_feat_num,
                'feature_details': {k: f"用户特征{k}有{v}个不同取值" for k, v in user_feat_num.items()}
            }
            
            results['user_feat_num'] = analysis
            
            print(f"   - 用户特征配置数: {analysis['total_features']}")
            print(f"   - 配置内容: {analysis['feature_configs']}")
            print(f"   - <class 'int'>含义: 每个用户特征ID对应该特征的不同取值数量（词汇表大小）")
            
        except Exception as e:
            results['user_feat_num'] = {'error': str(e)}
            print(f"   分析失败: {e}")
    
    # 5. 分析 predict_seq.jsonl 记录数为0的原因
    print("\n5. 分析 predict_seq.jsonl 记录数为0的原因...")
    predict_seq_file = eval_dir / 'predict_seq.jsonl'
    if predict_seq_file.exists():
        try:
            # 检查文件编码和内容
            file_size = predict_seq_file.stat().st_size
            
            # 尝试不同编码读取前几行
            encodings = ['utf-8', 'gbk', 'latin1', 'cp1252']
            successful_reads = {}
            
            for encoding in encodings:
                try:
                    with open(predict_seq_file, 'r', encoding=encoding) as f:
                        lines_read = []
                        for i, line in enumerate(f):
                            if i >= 3:  # 只读前3行
                                break
                            lines_read.append(line.strip())
                        successful_reads[encoding] = lines_read
                except Exception as e:
                    successful_reads[encoding] = f"读取失败: {e}"
            
            # 尝试二进制读取查看文件头
            with open(predict_seq_file, 'rb') as f:
                file_header = f.read(1000)  # 读取前1000字节
                
            analysis = {
                'file_size_mb': file_size / (1024 * 1024),
                'encoding_tests': successful_reads,
                'file_header_sample': file_header[:200].hex(),  # 十六进制显示
                'possible_causes': [
                    "文件可能使用了特殊编码格式",
                    "文件可能是二进制格式而非文本",
                    "文件内容可能被压缩或加密",
                    "多进程处理时编码处理有问题"
                ]
            }
            
            results['predict_seq_issue'] = analysis
            
            print(f"   - 文件大小: {analysis['file_size_mb']:.2f} MB")
            print(f"   - 编码测试结果:")
            for enc, result in successful_reads.items():
                if isinstance(result, list):
                    print(f"     {enc}: 成功读取 {len(result)} 行")
                    if result:
                        print(f"       首行样本: {result[0][:100]}...")
                else:
                    print(f"     {enc}: {result}")
            
            print(f"   - 可能原因:")
            for cause in analysis['possible_causes']:
                print(f"     - {cause}")
                
        except Exception as e:
            results['predict_seq_issue'] = {'error': str(e)}
            print(f"   分析失败: {e}")
    
    # 6. 详细分析 predict_set.jsonl 并保存到指定目录
    print("\n6. 详细分析 predict_set.jsonl...")
    predict_set_file = eval_dir / 'predict_set.jsonl'
    if predict_set_file.exists():
        predict_set_analysis = analyze_predict_set_detailed(predict_set_file, output_dir)
        results['predict_set'] = predict_set_analysis
    
    # 保存完整分析结果
    result_file = output_dir / 'precise_analysis_results.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n=== 分析完成 ===")
    print(f"完整结果已保存到: {result_file}")
    
    return results

def analyze_predict_set_detailed(predict_set_file, output_dir):
    """详细分析predict_set.jsonl并保存结果"""
    
    print("   开始详细分析predict_set.jsonl...")
    
    # 数据收集
    creative_ids = set()
    retrieval_ids = set()
    feature_stats = defaultdict(list)
    feature_types = defaultdict(set)
    feature_missing_count = defaultdict(int)
    records = []
    
    line_count = 0
    with open(predict_set_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="   处理记录"):
            try:
                record = json.loads(line.strip())
                line_count += 1
                
                # ID统计
                creative_ids.add(record.get('creative_id'))
                retrieval_ids.add(record.get('retrieval_id'))
                
                # 特征分析
                features = record.get('features', {})
                all_possible_features = set(range(100, 123))  # 基于观察到的特征范围
                
                for feat_id in all_possible_features:
                    feat_str = str(feat_id)
                    if feat_str in features:
                        feat_value = features[feat_str]
                        feature_types[feat_str].add(type(feat_value).__name__)
                        
                        if isinstance(feat_value, (int, float)):
                            feature_stats[feat_str].append(feat_value)
                        elif isinstance(feat_value, str) and feat_value != 'nan':
                            try:
                                numeric_val = float(feat_value)
                                feature_stats[feat_str].append(numeric_val)
                            except:
                                pass
                    else:
                        feature_missing_count[feat_str] += 1
                
                # 保存样本记录
                if len(records) < 100:
                    records.append(record)
                    
            except json.JSONDecodeError:
                continue
    
    # 统计分析
    feature_analysis = {}
    for feat_id, values in feature_stats.items():
        if values:
            values_array = np.array(values)
            feature_analysis[feat_id] = {
                'count': len(values),
                'missing_count': feature_missing_count[feat_id],
                'missing_rate': feature_missing_count[feat_id] / line_count * 100,
                'mean': float(np.mean(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'unique_count': len(np.unique(values_array)),
                'percentiles': {
                    '25%': float(np.percentile(values_array, 25)),
                    '50%': float(np.percentile(values_array, 50)),
                    '75%': float(np.percentile(values_array, 75))
                }
            }
    
    # 生成详细报告
    detailed_analysis = {
        'basic_info': {
            'total_records': line_count,
            'unique_creative_ids': len(creative_ids),
            'unique_retrieval_ids': len(retrieval_ids),
            'id_mapping_perfect': len(creative_ids) == len(retrieval_ids) == line_count
        },
        'feature_analysis': feature_analysis,
        'feature_types': {k: list(v) for k, v in feature_types.items()},
        'missing_pattern': dict(feature_missing_count),
        'sample_records': records[:10]
    }
    
    # 保存详细分析结果
    predict_set_result_file = output_dir / 'predict_set_detailed_analysis.json'
    with open(predict_set_result_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_analysis, f, indent=2, ensure_ascii=False, default=str)
    
    # 生成可读性报告
    report_file = output_dir / 'predict_set_analysis_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# predict_set.jsonl 详细分析报告\n\n")
        
        f.write("## 基础信息\n")
        f.write(f"- 总记录数: {detailed_analysis['basic_info']['total_records']:,}\n")
        f.write(f"- 唯一creative_id数: {detailed_analysis['basic_info']['unique_creative_ids']:,}\n")
        f.write(f"- 唯一retrieval_id数: {detailed_analysis['basic_info']['unique_retrieval_ids']:,}\n")
        f.write(f"- ID映射完美匹配: {detailed_analysis['basic_info']['id_mapping_perfect']}\n\n")
        
        f.write("## 特征详细分析\n")
        for feat_id in sorted(feature_analysis.keys(), key=int):
            stats = feature_analysis[feat_id]
            f.write(f"### 特征 {feat_id}\n")
            f.write(f"- 有效值数量: {stats['count']:,}\n")
            f.write(f"- 缺失数量: {stats['missing_count']:,}\n")
            f.write(f"- 缺失率: {stats['missing_rate']:.2f}%\n")
            f.write(f"- 统计信息: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}\n")
            f.write(f"- 数值范围: [{stats['min']}, {stats['max']}]\n")
            f.write(f"- 唯一值数量: {stats['unique_count']:,}\n")
            f.write(f"- 分位数: 25%={stats['percentiles']['25%']:.2f}, 50%={stats['percentiles']['50%']:.2f}, 75%={stats['percentiles']['75%']:.2f}\n\n")
    
    print(f"   predict_set.jsonl 详细分析完成")
    print(f"   - JSON结果: {predict_set_result_file}")
    print(f"   - 报告文件: {report_file}")
    
    return detailed_analysis

def main():
    try:
        results = analyze_file_structures()
        
        print("\n=== 关键问题解答 ===")
        print("1. item_feat_dict.json中的<class 'dict'>: 每个creative_id映射到包含13-14个特征的字典")
        print("2. item_feat_num.json中的<class 'int'>: 每个特征ID对应该特征的词汇表大小") 
        print("3. predict_seq_offsets.pkl中的<class 'list'>: 100万+用户在predict_seq.jsonl中的字节偏移位置")
        print("4. user_feat_num.json中的<class 'int'>: 每个用户特征的词汇表大小")
        print("5. predict_seq.jsonl记录数为0: 可能是编码问题或文件格式特殊")
        
        return results
        
    except Exception as e:
        print(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()