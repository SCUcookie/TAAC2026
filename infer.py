import argparse
import json
import os
import struct
import time
import psutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb
from model import BaselineModel


def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.0007, type=float)
    parser.add_argument('--maxlen', default=101, type=int)

    # Baseline Model construction
    parser.add_argument('--hidden_units', default=128, type=int)
    parser.add_argument('--num_blocks', default=8, type=int)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', action='store_true')
    parser.add_argument('--state_dict_path', default=None, type=str)
    parser.add_argument('--norm_first', default=True, type=bool)

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # InfoNCE 相关参数
    parser.add_argument('--loss_type', choices=['bce', 'infonce', 'mix'], default='infonce', help='选择损失函数类型')
    parser.add_argument('--temp', type=float, default=0.03, help='InfoNCE损失的温度系数')
    parser.add_argument('--contrastive_weight', type=float, default=0.5, help='对比学习损失的权重')
    parser.add_argument('--neg_topk', type=int, default=64, help='选择top-k难负样本的数量，0表示使用所有负样本')
    parser.add_argument('--norm_output', action='store_true', help='是否对输出embedding进行L2归一化', default=True)

    args = parser.parse_args()

    return args


def read_result_ids(file_path):
    with open(file_path, 'rb') as f:
        # Read the header (num_points_query and FLAGS_query_ann_top_k)
        num_points_query = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes
        query_ann_top_k = struct.unpack('I', f.read(4))[0]  # uint32_t -> 4 bytes

        print(f"num_points_query: {num_points_query}, query_ann_top_k: {query_ann_top_k}")

        # Calculate how many result_ids there are (num_points_query * query_ann_top_k)
        num_result_ids = num_points_query * query_ann_top_k

        # Read result_ids (uint64_t, 8 bytes per value)
        result_ids = np.fromfile(f, dtype=np.uint64, count=num_result_ids)

        return result_ids.reshape((num_points_query, query_ann_top_k))


def enhanced_process_cold_start_feat(feat):
    """
    增强版冷启动特征处理函数
    解决混合数据类型和特征115缺失问题
    """
    if not isinstance(feat, dict):
        return {}
    
    processed_feat = {}
    
    # 混合类型特征列表（基于评测数据集分析）
    mixed_type_features = ['102', '111', '115', '117', '118', '119', '120', '121', '122']
    
    for feat_id, feat_value in feat.items():
        if feat_id in mixed_type_features:
            # 处理混合类型
            if isinstance(feat_value, str):
                try:
                    # 尝试转换为整数
                    processed_feat[feat_id] = int(feat_value)
                except (ValueError, TypeError):
                    try:
                        # 尝试转换为浮点数再取整
                        processed_feat[feat_id] = int(float(feat_value))
                    except (ValueError, TypeError):
                        # 转换失败，使用默认值0
                        processed_feat[feat_id] = 0
                        print(f"Warning: Failed to convert feature {feat_id} value '{feat_value}' to int")
            elif isinstance(feat_value, (int, float)):
                processed_feat[feat_id] = int(feat_value)
            else:
                processed_feat[feat_id] = 0
        elif isinstance(feat_value, list):
            # 处理列表类型特征
            value_list = []
            for v in feat_value:
                if isinstance(v, str):
                    try:
                        value_list.append(int(v))
                    except (ValueError, TypeError):
                        value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif isinstance(feat_value, str):
            try:
                # 尝试转换字符串为数字
                processed_feat[feat_id] = int(feat_value)
            except (ValueError, TypeError):
                processed_feat[feat_id] = 0
        else:
            # 正常类型特征直接复制
            processed_feat[feat_id] = feat_value
    
    # 处理特征115缺失问题（70.37%缺失率）
    if '115' not in processed_feat:
        processed_feat['115'] = 493  # 使用分析得出的均值
    
    return processed_feat


def batch_process_candidates(candidates_data, batch_size=1000):
    """
    批量处理候选物品数据
    """
    processed_candidates = []
    
    for i in tqdm(range(0, len(candidates_data), batch_size), desc="Processing candidates"):
        batch = candidates_data[i:i+batch_size]
        
        for candidate in batch:
            # 处理features字段
            if 'features' in candidate:
                candidate['features'] = enhanced_process_cold_start_feat(candidate['features'])
            
            processed_candidates.append(candidate)
    
    return processed_candidates


def process_cold_start_feat(feat):
    """
    保留原始函数以兼容性
    """
    return enhanced_process_cold_start_feat(feat)


def precompute_candidate_embeddings(model, candidates, batch_size=128, save_path=None):
    """
    预计算所有候选物品的embedding
    """
    model.eval()
    device = next(model.parameters()).device
    
    all_embeddings = []
    all_retrieval_ids = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(candidates), batch_size), desc="Computing embeddings"):
            batch = candidates[i:i+batch_size]
            
            # 构造批次数据
            batch_features = []
            batch_retrieval_ids = []
            
            for candidate in batch:
                features = candidate['features']
                retrieval_id = candidate['retrieval_id']
                
                batch_features.append(features)
                batch_retrieval_ids.append(retrieval_id)
            
            # 生成embedding（这里需要根据实际模型接口调整）
            # 暂时使用占位符，实际需要基于模型的feat2emb方法
            batch_embs = model.generate_item_embeddings(batch_features) if hasattr(model, 'generate_item_embeddings') else torch.randn(len(batch_features), 128)
            
            all_embeddings.append(batch_embs.cpu())
            all_retrieval_ids.extend(batch_retrieval_ids)
    
    # 合并所有embedding
    candidate_embeddings = torch.cat(all_embeddings, dim=0)
    
    # 保存到缓存
    if save_path:
        cache_data = {
            'embeddings': candidate_embeddings,
            'retrieval_ids': all_retrieval_ids,
            'embedding_dim': candidate_embeddings.shape[1]
        }
        torch.save(cache_data, save_path)
        print(f"Candidate embeddings saved to {save_path}")
    
    return candidate_embeddings, all_retrieval_ids


def load_cached_embeddings(cache_path):
    """
    加载缓存的embedding
    """
    if os.path.exists(cache_path):
        cache_data = torch.load(cache_path)
        return cache_data['embeddings'], cache_data['retrieval_ids']
    else:
        return None, None


def memory_efficient_user_processing(model, test_loader, device):
    """
    内存优化的用户序列处理
    """
    model.eval()
    
    all_user_embeddings = []
    all_user_ids = []
    
    # 使用生成器来减少内存占用
    def batch_generator():
        for batch in test_loader:
            seq, token_type, seq_feat, user_ids = batch
            seq = seq.to(device)
            
            with torch.no_grad():
                user_embs = model.predict(seq, seq_feat, token_type)
                
                # 立即移到CPU并转换为float16以节省内存
                user_embs = user_embs.cpu().half()
                
                yield user_embs, user_ids
    
    # 处理批次
    for user_embs, user_ids in tqdm(batch_generator(), desc="Processing users"):
        all_user_embeddings.append(user_embs)
        all_user_ids.extend(user_ids)
        
        # 定期清理GPU缓存
        if len(all_user_embeddings) % 10 == 0:
            torch.cuda.empty_cache()
    
    # 合并用户embedding
    user_embeddings = torch.cat(all_user_embeddings, dim=0).float()
    
    return user_embeddings, all_user_ids


class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        
    def start_timer(self, name):
        self.timings[name + '_start'] = time.time()
        
    def end_timer(self, name):
        if name + '_start' in self.timings:
            duration = time.time() - self.timings[name + '_start']
            self.timings[name] = duration
            print(f"{name}: {duration:.2f} seconds")
    
    def log_memory(self, stage):
        memory_info = {
            'cpu_percent': psutil.virtual_memory().percent,
            'cpu_used_gb': psutil.virtual_memory().used / (1024**3)
        }
        
        if torch.cuda.is_available():
            memory_info['gpu_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            memory_info['gpu_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        
        self.memory_usage[stage] = memory_info
        print(f"Memory at {stage}: CPU {memory_info['cpu_percent']:.1f}% ({memory_info['cpu_used_gb']:.1f}GB)")
        
        if 'gpu_allocated_gb' in memory_info:
            print(f"GPU: {memory_info['gpu_allocated_gb']:.1f}GB allocated, {memory_info['gpu_reserved_gb']:.1f}GB reserved")
    
    def generate_report(self):
        print("\n=== Performance Report ===")
        total_time = sum(v for k, v in self.timings.items() if not k.endswith('_start'))
        print(f"Total execution time: {total_time:.2f} seconds")
        
        for stage, timing in self.timings.items():
            if not stage.endswith('_start'):
                percentage = (timing / total_time) * 100
                print(f"  {stage}: {timing:.2f}s ({percentage:.1f}%)")


def setup_efficient_ann_search(candidate_embeddings, use_gpu=True):
    """
    设置高效的ANN检索（使用简化版本，因为原版使用外部FAISS工具）
    """
    print(f"Setting up ANN search for {len(candidate_embeddings)} candidates")
    print(f"Embedding dimension: {candidate_embeddings.shape[1]}")
    
    # 这里简化处理，实际环境会使用FAISS
    # 返回embedding tensor以供后续批量检索使用
    return candidate_embeddings.numpy().astype('float32')


def batch_ann_search_simple(candidate_embeddings, user_embeddings, retrieval_ids, k=10, batch_size=1000):
    """
    简化版批量ANN检索（使用余弦相似度）
    实际环境中会使用FAISS的外部工具
    """
    all_top10s = []
    
    # 标准化embedding以计算余弦相似度
    candidate_norm = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
    
    for i in tqdm(range(0, len(user_embeddings), batch_size), desc="ANN search"):
        batch_embs = user_embeddings[i:i+batch_size].numpy().astype('float32')
        batch_norm = batch_embs / (np.linalg.norm(batch_embs, axis=1, keepdims=True) + 1e-8)
        
        # 计算相似度矩阵
        similarities = np.dot(batch_norm, candidate_norm.T)
        
        # 获取top-k
        top_k_indices = np.argpartition(-similarities, k, axis=1)[:, :k]
        top_k_indices = top_k_indices[np.arange(len(top_k_indices))[:, None], 
                                    np.argsort(-similarities[np.arange(len(similarities))[:, None], top_k_indices])]
        
        # 转换为retrieval_id
        batch_top10s = []
        for user_indices in top_k_indices:
            user_top10 = [retrieval_ids[idx] for idx in user_indices]
            batch_top10s.append(user_top10)
        
        all_top10s.extend(batch_top10s)
    
    return all_top10s


def validate_inference_results(top10s, user_list, candidates):
    """
    验证推理结果的质量
    """
    validation_report = {
        'result_count': len(top10s),
        'user_count': len(user_list),
        'consistency_check': True,
        'coverage_stats': {},
        'quality_issues': []
    }
    
    # 检查结果一致性
    if len(top10s) != len(user_list):
        validation_report['consistency_check'] = False
        validation_report['quality_issues'].append("Mismatch between top10s and user_list length")
    
    # 检查top10格式
    valid_results = 0
    for i, user_top10 in enumerate(top10s):
        if len(user_top10) == 10:
            valid_results += 1
        else:
            validation_report['quality_issues'].append(f"User {i} has {len(user_top10)} items instead of 10")
    
    validation_report['coverage_stats']['valid_results'] = valid_results
    validation_report['coverage_stats']['coverage_rate'] = valid_results / len(top10s) * 100 if len(top10s) > 0 else 0
    
    # 检查推荐物品的有效性
    candidate_ids = set(c['creative_id'] for c in candidates)
    unique_recommended = set()
    
    for user_top10 in top10s[:100]:  # 抽样检查
        for item_id in user_top10:
            unique_recommended.add(item_id)
            if item_id not in candidate_ids:
                validation_report['quality_issues'].append(f"Invalid item_id {item_id} not in candidates")
    
    validation_report['coverage_stats']['unique_recommended'] = len(unique_recommended)
    
    return validation_report


def get_candidate_emb(indexer, feat_types, feat_default_value, mm_emb_dict, model):
    """
    生产候选库item的id和embedding

    Args:
        indexer: 索引字典
        feat_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feature_default_value: 特征缺省值
        mm_emb_dict: 多模态特征字典
        model: 模型
    Returns:
        retrieve_id2creative_id: 索引id->creative_id的dict
    """
    EMB_SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    item_ids, creative_ids, retrieval_ids, features = [], [], [], []
    retrieve_id2creative_id = {}

    with open(candidate_path, 'r') as f:
        for line in f:
            line = json.loads(line)
            # 读取item特征，并补充缺失值
            feature = line['features']
            creative_id = line['creative_id']
            retrieval_id = line['retrieval_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            missing_fields = set(
                feat_types['item_sparse'] + feat_types['item_array'] + feat_types['item_continual']
            ) - set(feature.keys())
            feature = enhanced_process_cold_start_feat(feature)
            for feat_id in missing_fields:
                feature[feat_id] = feat_default_value[feat_id]
            for feat_id in feat_types['item_emb']:
                if creative_id in mm_emb_dict[feat_id]:
                    feature[feat_id] = mm_emb_dict[feat_id][creative_id]
                else:
                    feature[feat_id] = np.zeros(EMB_SHAPE_DICT[feat_id], dtype=np.float32)

            item_ids.append(item_id)
            creative_ids.append(creative_id)
            retrieval_ids.append(retrieval_id)
            features.append(feature)
            retrieve_id2creative_id[retrieval_id] = creative_id

    # 保存候选库的embedding和sid
    model.save_item_emb(item_ids, retrieval_ids, features, os.environ.get('EVAL_RESULT_PATH'))
    with open(Path(os.environ.get('EVAL_RESULT_PATH'), "retrive_id2creative_id.json"), "w") as f:
        json.dump(retrieve_id2creative_id, f)
    return retrieve_id2creative_id


def optimized_inference_pipeline():
    """
    完整的优化推理流程
    """
    monitor = PerformanceMonitor()
    
    # 1. 数据加载和预处理
    monitor.start_timer("data_loading")
    monitor.log_memory("start")
    
    args = get_args()
    eval_data_path = os.environ.get('EVAL_DATA_PATH')
    test_dataset = MyTestDataset(eval_data_path, args)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn
    )
    
    # 加载候选数据并进行批量预处理
    candidate_path = Path(eval_data_path, 'predict_set.jsonl')
    candidates = []
    with open(candidate_path, 'r') as f:
        for line in f:
            candidates.append(json.loads(line))
    
    candidates = batch_process_candidates(candidates)
    
    monitor.end_timer("data_loading")
    monitor.log_memory("after_data_loading")
    
    # 2. 模型加载
    monitor.start_timer("model_loading")
    
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()
    
    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    
    monitor.end_timer("model_loading")
    
    # 3. 候选物品embedding预计算和缓存
    monitor.start_timer("candidate_embedding")
    
    cache_dir = os.environ.get('USER_CACHE_PATH', './cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'candidate_embeddings.pt')
    
    candidate_embeddings, retrieval_ids = load_cached_embeddings(cache_path)
    
    if candidate_embeddings is None:
        print("No cached embeddings found, computing candidate embeddings...")
        # 使用现有的get_candidate_emb函数但优化embedding处理
        retrieve_id2creative_id = get_candidate_emb(
            test_dataset.indexer['i'],
            test_dataset.feature_types,
            test_dataset.feature_default_value,
            test_dataset.mm_emb_dict,
            model,
        )
        
        # 读取生成的embedding文件
        embedding_file = Path(os.environ.get('EVAL_RESULT_PATH'), 'embedding.fbin')
        id_file = Path(os.environ.get('EVAL_RESULT_PATH'), 'id.u64bin')
        
        if embedding_file.exists():
            candidate_embeddings = load_fbin_embeddings(embedding_file)
            retrieval_ids = load_u64bin_ids(id_file)
            
            # 缓存到USER_CACHE_PATH
            cache_data = {
                'embeddings': torch.tensor(candidate_embeddings),
                'retrieval_ids': retrieval_ids,
                'embedding_dim': candidate_embeddings.shape[1]
            }
            torch.save(cache_data, cache_path)
            print(f"Candidate embeddings cached to {cache_path}")
    else:
        print("Using cached candidate embeddings...")
        retrieve_id2creative_id = {}
        for rid in retrieval_ids:
            # 需要从candidates重建映射
            for candidate in candidates:
                if candidate['retrieval_id'] == rid:
                    retrieve_id2creative_id[rid] = candidate['creative_id']
                    break
    
    monitor.end_timer("candidate_embedding")
    monitor.log_memory("after_candidate_embedding")
    
    # 4. 用户序列编码
    monitor.start_timer("user_encoding")
    
    user_embeddings, user_list = memory_efficient_user_processing(model, test_loader, args.device)
    
    monitor.end_timer("user_encoding")
    monitor.log_memory("after_user_encoding")
    
    # 5. ANN检索
    monitor.start_timer("ann_search")
    
    # 如果有外部FAISS工具可用，使用原有流程
    # 否则使用简化版检索
    use_external_faiss = Path("/workspace/faiss-based-ann/faiss_demo").exists()
    
    if use_external_faiss:
        # 使用原有的FAISS外部工具
        all_embs = user_embeddings.numpy().astype(np.float32)
        save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))
        
        ann_cmd = (
            str(Path("/workspace", "faiss-based-ann", "faiss_demo"))
            + " --dataset_vector_file_path="
            + str(Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin"))
            + " --dataset_id_file_path="
            + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin"))
            + " --query_vector_file_path="
            + str(Path(os.environ.get("EVAL_RESULT_PATH"), "query.fbin"))
            + " --result_id_file_path="
            + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
            + " --query_ann_top_k=10 --faiss_M=64 --faiss_ef_construction=1280 --query_ef_search=640 --faiss_metric_type=0"
        )
        os.system(ann_cmd)
        
        # 取出top-k
        top10s_retrieved = read_result_ids(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
        top10s_untrimmed = []
        for top10 in tqdm(top10s_retrieved):
            for item in top10:
                top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))
        
        top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]
    else:
        # 使用简化版ANN检索
        print("Using simplified ANN search...")
        candidate_emb_array = setup_efficient_ann_search(candidate_embeddings)
        top10s = batch_ann_search_simple(candidate_emb_array, user_embeddings, retrieval_ids)
        
        # 转换为creative_id
        for i, user_top10 in enumerate(top10s):
            top10s[i] = [retrieve_id2creative_id.get(rid, 0) for rid in user_top10]
    
    monitor.end_timer("ann_search")
    monitor.log_memory("after_ann_search")
    
    # 6. 结果验证
    monitor.start_timer("validation")
    
    validation_report = validate_inference_results(top10s, user_list, candidates)
    print("Validation Report:", validation_report)
    
    monitor.end_timer("validation")
    
    # 7. 性能报告
    monitor.generate_report()
    
    return top10s, user_list


def load_fbin_embeddings(file_path):
    """
    加载.fbin格式的embedding文件
    """
    with open(file_path, 'rb') as f:
        # 读取header
        num_vectors = struct.unpack('I', f.read(4))[0]
        dim = struct.unpack('I', f.read(4))[0]
        
        # 读取embeddings
        embeddings = np.fromfile(f, dtype=np.float32, count=num_vectors * dim)
        embeddings = embeddings.reshape(num_vectors, dim)
    
    return embeddings


def load_u64bin_ids(file_path):
    """
    加载.u64bin格式的ID文件
    """
    with open(file_path, 'rb') as f:
        ids = np.fromfile(f, dtype=np.uint64)
    
    return ids.tolist()


def infer():
    """
    推理入口函数，可选择使用优化流程或原始流程
    """
    # 检查是否启用优化流程
    use_optimized = os.environ.get('USE_OPTIMIZED_PIPELINE', 'true').lower() == 'true'
    
    if use_optimized:
        print("Using optimized inference pipeline...")
        return optimized_inference_pipeline()
    else:
        print("Using original inference pipeline...")
        return original_infer()


def original_infer():
    """
    保留原始推理流程
    """
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH')
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn
    )
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()

    ckpt_path = get_ckpt_path()
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    all_embs = []
    user_list = []
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):

        seq, token_type, seq_feat, user_id = batch
        seq = seq.to(args.device)
        logits = model.predict(seq, seq_feat, token_type)
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += user_id

    # 生成候选库的embedding 以及 id文件
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],
        test_dataset.feature_types,
        test_dataset.feature_default_value,
        test_dataset.mm_emb_dict,
        model,
    )
    all_embs = np.concatenate(all_embs, axis=0)
    # 保存query文件
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))
    # ANN 检索
    ann_cmd = (
        str(Path("/workspace", "faiss-based-ann", "faiss_demo"))
        + " --dataset_vector_file_path="
        + str(Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin"))
        + " --dataset_id_file_path="
        + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin"))
        + " --query_vector_file_path="
        + str(Path(os.environ.get("EVAL_RESULT_PATH"), "query.fbin"))
        + " --result_id_file_path="
        + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
        + " --query_ann_top_k=10 --faiss_M=64 --faiss_ef_construction=1280 --query_ef_search=640 --faiss_metric_type=0"
    )
    os.system(ann_cmd)

    # 取出top-k
    top10s_retrieved = read_result_ids(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
    top10s_untrimmed = []
    for top10 in tqdm(top10s_retrieved):
        for item in top10:
            top10s_untrimmed.append(retrieve_id2creative_id.get(int(item), 0))

    top10s = [top10s_untrimmed[i : i + 10] for i in range(0, len(top10s_untrimmed), 10)]

    return top10s, user_list
