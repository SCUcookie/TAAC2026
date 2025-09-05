#!/usr/bin/env python3
"""
生成精简版评测数据集分析结果
专门为比赛服务器环境设计，生成可复制的关键分析信息
"""

import json
import os
from pathlib import Path

def create_summary_analysis():
    """生成精简版分析结果"""
    
    # 读取完整分析结果
    cache_path = os.environ.get('USER_CACHE_PATH', '../../analysis_cache')
    full_file = Path(cache_path) / 'comprehensive_eval_analysis.json'
    
    if not full_file.exists():
        print("错误: 完整分析文件不存在")
        print(f"查找路径: {full_file}")
        return
    
    print("正在读取完整分析结果...")
    with open(full_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"原始文件包含 {len(data)} 个文件的分析结果")
    
    # 生成精简版
    summary = {
        'analysis_metadata': {
            'original_file_size': full_file.stat().st_size,
            'files_analyzed': len(data),
            'compression_note': 'This is a compressed summary of the full analysis'
        },
        'files': {}
    }
    
    for filename, analysis in data.items():
        print(f"处理文件: {filename}")
        
        if 'error' in analysis:
            summary['files'][filename] = {'error': analysis['error']}
            continue
        
        # 根据文件类型进行精简处理
        if filename == 'indexer.pkl':
            summary['files'][filename] = extract_indexer_summary(analysis)
        elif filename == 'item_feat_dict.json':
            summary['files'][filename] = extract_item_feat_dict_summary(analysis)
        elif filename in ['item_feat_num.json', 'user_feat_num.json']:
            summary['files'][filename] = extract_feat_num_summary(analysis)
        elif filename == 'predict_seq.jsonl':
            summary['files'][filename] = extract_predict_seq_summary(analysis)
        elif filename == 'predict_seq_offsets.pkl':
            summary['files'][filename] = extract_offsets_summary(analysis)
        elif filename == 'predict_set.jsonl':
            summary['files'][filename] = extract_predict_set_summary(analysis)
        else:
            # 默认处理：保留小于100个元素的数据
            summary['files'][filename] = extract_default_summary(analysis)
    
    # 保存精简版
    summary_file = Path(cache_path) / 'analysis_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== 精简版分析完成 ===")
    print(f"原始文件大小: {full_file.stat().st_size:,} bytes")
    print(f"精简版文件大小: {summary_file.stat().st_size:,} bytes")
    print(f"压缩比: {summary_file.stat().st_size / full_file.stat().st_size * 100:.1f}%")
    print(f"保存位置: {summary_file}")
    
    # 生成关键洞察报告
    create_key_insights(data, cache_path)
    
    return summary

def extract_indexer_summary(analysis):
    """提取indexer.pkl的精简信息"""
    summary = {
        'file_type': analysis.get('file_type'),
        'data_type': analysis.get('data_type'),
        'main_keys': analysis.get('main_keys'),
        'sizes': {}
    }
    
    if 'detailed_analysis' in analysis:
        for key, details in analysis['detailed_analysis'].items():
            if isinstance(details, dict) and 'size' in details:
                summary['sizes'][key] = {
                    'size': details['size'],
                    'description': details.get('description', ''),
                    'id_range': details.get('id_range', '')
                }
                
                # 对于特征词汇表，只保留统计信息
                if 'feature_vocab_sizes' in details:
                    vocab_sizes = details['feature_vocab_sizes']
                    summary['sizes'][key]['feature_vocab_stats'] = {
                        'total_features': len(vocab_sizes),
                        'min_vocab_size': min(vocab_sizes.values()) if vocab_sizes else 0,
                        'max_vocab_size': max(vocab_sizes.values()) if vocab_sizes else 0,
                        'sample_features': dict(list(vocab_sizes.items())[:10])
                    }
    
    return summary

def extract_item_feat_dict_summary(analysis):
    """提取item_feat_dict.json的精简信息"""
    summary = {
        'file_type': analysis.get('file_type'),
        'total_items': analysis.get('total_items'),
        'feature_count': analysis.get('feature_count'),
        'feature_list': analysis.get('feature_list'),
        'sampling_info': analysis.get('sampling_info')
    }
    
    # 特征分析的精简版
    if 'feature_analysis' in analysis:
        summary['feature_analysis'] = {}
        for feat_id, stats in analysis['feature_analysis'].items():
            summary['feature_analysis'][feat_id] = {
                'completeness_rate': stats.get('completeness_rate'),
                'data_types': stats.get('data_types'),
                'mean': round(stats.get('mean', 0), 2),
                'std': round(stats.get('std', 0), 2),
                'min': stats.get('min'),
                'max': stats.get('max'),
                'unique_count': stats.get('unique_count')
            }
    
    return summary

def extract_feat_num_summary(analysis):
    """提取特征数量配置文件的精简信息"""
    return {
        'file_type': analysis.get('file_type'),
        'total_features': analysis.get('total_features'),
        'feature_vocab_sizes': analysis.get('feature_vocab_sizes'),
        'statistics': analysis.get('statistics'),
        'feature_categorization': analysis.get('feature_categorization'),
        'comparison_with_item_features': analysis.get('comparison_with_item_features')
    }

def extract_predict_seq_summary(analysis):
    """提取predict_seq.jsonl的精简信息"""
    return {
        'file_type': analysis.get('file_type'),
        'file_size_gb': analysis.get('file_size_gb'),
        'line_count_estimate': analysis.get('line_count_estimate'),
        'encoding_tests': analysis.get('encoding_tests'),
        'potential_issues': analysis.get('potential_issues'),
        'sample_records_count': len(analysis.get('sample_records', []))
    }

def extract_offsets_summary(analysis):
    """提取offsets文件的精简信息"""
    summary = {
        'file_type': analysis.get('file_type'),
        'data_type': analysis.get('data_type'),
        'length': analysis.get('length'),
        'statistics': analysis.get('statistics'),
        'is_sorted': analysis.get('is_sorted')
    }
    
    # 只保留前20个样本
    if 'sample_data' in analysis:
        summary['sample_data'] = analysis['sample_data'][:20]
    
    return summary

def extract_predict_set_summary(analysis):
    """提取predict_set.jsonl的精简信息"""
    summary = {
        'file_type': analysis.get('file_type'),
        'basic_stats': analysis.get('basic_stats'),
        'feature_types': analysis.get('feature_types')
    }
    
    # 数值特征的精简版
    if 'numeric_features' in analysis:
        summary['numeric_features_summary'] = {}
        for feat_id, stats in analysis['numeric_features'].items():
            summary['numeric_features_summary'][feat_id] = {
                'count': stats.get('count'),
                'mean': round(stats.get('mean', 0), 2),
                'std': round(stats.get('std', 0), 2),
                'min': stats.get('min'),
                'max': stats.get('max'),
                'unique_count': stats.get('unique_count')
            }
    
    # 只保留前5个样本记录
    if 'sample_records' in analysis:
        summary['sample_records'] = analysis['sample_records'][:5]
    
    return summary

def extract_default_summary(analysis):
    """默认提取方法"""
    summary = {}
    for key, value in analysis.items():
        if isinstance(value, list) and len(value) > 100:
            summary[key] = f"[List with {len(value)} items - truncated]"
        elif isinstance(value, dict) and len(value) > 100:
            summary[key] = f"[Dict with {len(value)} keys - truncated]"
        else:
            summary[key] = value
    return summary

def create_key_insights(data, cache_path):
    """生成关键洞察报告"""
    
    insights = {
        'data_scale': {},
        'file_status': {},
        'feature_analysis': {},
        'data_quality_issues': [],
        'key_findings': []
    }
    
    for filename, analysis in data.items():
        if 'error' in analysis:
            insights['file_status'][filename] = 'missing'
            continue
        else:
            insights['file_status'][filename] = 'present'
        
        # 提取数据规模信息
        if filename == 'indexer.pkl' and 'detailed_analysis' in analysis:
            da = analysis['detailed_analysis']
            if 'i' in da and isinstance(da['i'], dict):
                insights['data_scale']['total_items_in_index'] = da['i'].get('size')
            if 'u' in da and isinstance(da['u'], dict):
                insights['data_scale']['total_users_in_index'] = da['u'].get('size')
        
        elif filename == 'item_feat_dict.json':
            insights['data_scale']['items_with_features'] = analysis.get('total_items')
            insights['data_scale']['item_feature_count'] = analysis.get('feature_count')
        
        elif filename == 'predict_set.jsonl' and 'basic_stats' in analysis:
            insights['data_scale']['candidate_items'] = analysis['basic_stats'].get('total_records')
            insights['feature_analysis']['predict_set_features'] = len(analysis.get('feature_types', {}))
            
            # 检查数据质量问题
            if 'feature_types' in analysis:
                for feat_id, types in analysis['feature_types'].items():
                    if len(types) > 1:  # 混合类型
                        insights['data_quality_issues'].append(f"Feature {feat_id} has mixed types: {types}")
        
        elif filename == 'predict_seq.jsonl':
            insights['data_scale']['user_sequences_estimate'] = analysis.get('line_count_estimate', 'unknown')
            if analysis.get('potential_issues'):
                insights['data_quality_issues'].extend(analysis['potential_issues'])
    
    # 生成关键发现
    if insights['data_scale'].get('candidate_items', 0) > 0:
        candidate_count = insights['data_scale']['candidate_items']
        training_items = insights['data_scale'].get('items_with_features', 0)
        if candidate_count != training_items:
            insights['key_findings'].append(f"Cold start scenario: {candidate_count:,} candidates vs {training_items:,} training items")
    
    if len(insights['data_quality_issues']) > 0:
        insights['key_findings'].append(f"Found {len(insights['data_quality_issues'])} data quality issues")
    
    # 保存关键洞察
    insights_file = Path(cache_path) / 'key_insights.json'
    with open(insights_file, 'w', encoding='utf-8') as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== 关键洞察报告 ===")
    print(f"数据规模:")
    for key, value in insights['data_scale'].items():
        if isinstance(value, int):
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n文件状态:")
    for filename, status in insights['file_status'].items():
        print(f"  {filename}: {status}")
    
    if insights['data_quality_issues']:
        print(f"\n数据质量问题 ({len(insights['data_quality_issues'])}个):")
        for issue in insights['data_quality_issues'][:5]:  # 只显示前5个
            print(f"  - {issue}")
        if len(insights['data_quality_issues']) > 5:
            print(f"  - ... 还有 {len(insights['data_quality_issues']) - 5} 个问题")
    
    print(f"\n洞察报告已保存: {insights_file}")

def main():
    try:
        summary = create_summary_analysis()
        
        print("\n=== 精简版分析完成 ===")
        print("现在您可以复制以下文件到本地:")
        print("1. analysis_summary.json - 精简版完整分析")
        print("2. key_insights.json - 关键洞察报告")
        print("\n这两个文件的大小应该在可复制范围内!")
        
        return summary
        
    except Exception as e:
        print(f"生成精简版分析时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()