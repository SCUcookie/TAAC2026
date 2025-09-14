# 全模态生成式推荐系统 - 推理模块
# 用于模型推理和生成Top-K推荐结果

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyTestDataset, save_emb  # 导入测试数据集和embedding保存函数
from model import BaselineModel  # 导入基线模型

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available, falling back to command line tool")


def get_ckpt_path():
    """
    获取训练好的模型检查点文件路径
    从环境变量MODEL_OUTPUT_PATH指定的目录中查找.pt文件
    
    Returns:
        str: 模型检查点文件的完整路径
        
    Raises:
        ValueError: 如果MODEL_OUTPUT_PATH环境变量未设置
    """
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    
    # 遍历目录找到.pt模型文件
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)


def get_args():
    parser = argparse.ArgumentParser()

    # Train params
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.002, type=float)
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
    parser.add_argument('--use_action_weight', action='store_true', help='是否使用动作权重')

    # MMemb Feature ID
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], type=str, choices=[str(s) for s in range(81, 87)])

    # InfoNCE 相关参数
    parser.add_argument('--loss_type', choices=['bce', 'infonce', 'mix'], default='infonce', help='选择损失函数类型')
    parser.add_argument('--temp', type=float, default=0.03, help='InfoNCE损失的温度系数')
    parser.add_argument('--contrastive_weight', type=float, default=0.5, help='对比学习损失的权重')
    parser.add_argument('--neg_topk', type=int, default=64, help='选择top-k难负样本的数量，0表示使用所有负样本')
    parser.add_argument('--norm_output', action='store_true', help='是否对输出embedding进行L2归一化', default=True)
    
    # FAISS 模式选择
    parser.add_argument('--use_python_faiss', action='store_true', help='使用Python FAISS而非命令行工具', default=False)
    
    # 多阶段召回参数
    parser.add_argument('--use_multistage_recall', action='store_true', help='启用多阶段召回系统', default=True)
    parser.add_argument('--stage1_topk', type=int, default=100, help='第一阶段粗召回的候选数量')
    parser.add_argument('--diversity_penalty_rate', type=float, default=0.95, help='多样性惩罚衰减率')
    parser.add_argument('--time_decay_factor', type=float, default=1.0, help='时间衰减因子')

    args = parser.parse_args()

    return args

def read_fbin_file(filepath):
    """读取.fbin格式的向量文件"""
    with open(filepath, 'rb') as f:
        # 读取文件头：向量数量和维度
        num_vectors, dimensions = struct.unpack('II', f.read(8))
        # 读取向量数据
        vectors = np.fromfile(f, dtype=np.float32).reshape(num_vectors, dimensions)
    return vectors, num_vectors, dimensions


def read_u64bin_file(filepath):
    """读取.u64bin格式的ID文件"""
    with open(filepath, 'rb') as f:
        # 读取文件头：ID数量和每个ID的维度(通常为1)
        num_ids, id_dim = struct.unpack('II', f.read(8))
        # 读取ID数据
        ids = np.fromfile(f, dtype=np.uint64).reshape(num_ids, id_dim)
    return ids.flatten()  # 返回一维数组


def write_u64bin_file(ids, filepath):
    """写入.u64bin格式的ID文件"""
    ids = ids.astype(np.uint64)
    num_ids = len(ids)
    with open(filepath, 'wb') as f:
        # 写入文件头
        f.write(struct.pack('II', num_ids, 1))
        # 写入ID数据
        ids.tofile(f)


def perform_python_faiss_search():
    """使用Python FAISS执行ANN搜索，替代命令行工具"""

    if not FAISS_AVAILABLE:
        raise ImportError("FAISS not available, cannot use Python FAISS mode")

    print("Performing ANN search with Python FAISS...")

    # 文件路径
    eval_result_path = os.environ.get("EVAL_RESULT_PATH")
    dataset_vector_path = Path(eval_result_path, "embedding.fbin")
    dataset_id_path = Path(eval_result_path, "id.u64bin")
    query_vector_path = Path(eval_result_path, "query.fbin")
    result_id_path = Path(eval_result_path, "id100.u64bin")

    # 读取候选库向量和ID
    print("Loading dataset vectors...")
    dataset_vectors, num_dataset, vector_dim = read_fbin_file(dataset_vector_path)
    dataset_ids = read_u64bin_file(dataset_id_path)

    # 读取查询向量
    print("Loading query vectors...")
    query_vectors, num_queries, _ = read_fbin_file(query_vector_path)

    # 构建FAISS索引 (HNSW + 内积)
    print("Building FAISS HNSW index...")
    # 使用IndexHNSWFlat，内积相似度
    index = faiss.IndexHNSWFlat(vector_dim, 64)  # M=64
    index.hnsw.efConstruction = 1280  # ef_construction=1280
    index.metric_type = faiss.METRIC_INNER_PRODUCT  # 内积相似度

    # 添加向量到索引
    index.add(dataset_vectors)

    # 设置查询参数
    index.hnsw.efSearch = 640  # ef_search=640

    # 执行搜索
    print("Performing search...")
    k = 100  # top-k=10
    similarities, indices = index.search(query_vectors, k)

    # 将索引转换为原始ID
    print("Converting indices to original IDs...")
    result_ids = []
    for query_idx in range(num_queries):
        query_result_ids = []
        for rank in range(k):
            dataset_idx = indices[query_idx][rank]
            if dataset_idx != -1:  # 有效结果
                original_id = dataset_ids[dataset_idx]
                query_result_ids.append(original_id)
            else:
                query_result_ids.append(0)  # 无效结果用0填充
        result_ids.extend(query_result_ids)

    # 保存结果 - 修复格式问题
    print("Saving results...")
    result_ids = np.array(result_ids, dtype=np.uint64)
    
    # 正确的格式：写入(num_queries, k)作为头部，而不是(total_results, 1)
    with open(result_id_path, 'wb') as f:
        # 写入正确的文件头：num_points_query, query_ann_top_k
        f.write(struct.pack('II', num_queries, k))
        # 写入ID数据
        result_ids.tofile(f)

    print(f"Python FAISS search completed. Results saved to {result_id_path}")
    print(f"Total queries: {num_queries}, Top-K: {k}")
    print(f"Result file contains {len(result_ids)} IDs")


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


def process_cold_start_feat(feat):
    """
    处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
    """
    processed_feat = {}
    for feat_id, feat_value in feat.items():
        if type(feat_value) == list:
            value_list = []
            for v in feat_value:
                if type(v) == str:
                    value_list.append(0)
                else:
                    value_list.append(v)
            processed_feat[feat_id] = value_list
        elif type(feat_value) == str:
            processed_feat[feat_id] = 0
        else:
            processed_feat[feat_id] = feat_value
    return processed_feat


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
            feature = process_cold_start_feat(feature)
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


def infer():
    """
    主推理函数：生成用户推荐结果
    
    推理流程：
    1. 加载测试数据集和预训练模型
    2. 对每个用户生成序列表征embedding
    3. 构建候选物品库的embedding
    4. 使用FAISS进行近似最近邻搜索
    5. 返回每个用户的Top-10推荐列表
    
    Returns:
        top10s: 每个用户的Top-10推荐物品ID列表
        user_list: 用户ID列表
    """
    # 解析推理参数
    args = get_args()
    data_path = os.environ.get('EVAL_DATA_PATH')
    
    # 创建测试数据集和数据加载器
    test_dataset = MyTestDataset(data_path, args)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_dataset.collate_fn
    )
    
    # 获取数据集基本信息
    usernum, itemnum = test_dataset.usernum, test_dataset.itemnum
    feat_statistics, feat_types = test_dataset.feat_statistics, test_dataset.feature_types
    
    # 创建模型并加载预训练权重
    model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args).to(args.device)
    model.eval()  # 设置为评估模式

    ckpt_path = get_ckpt_path()  # 获取模型检查点路径
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(args.device)))
    
    # 生成用户序列表征embedding
    all_embs = []  # 存储所有用户的embedding
    user_list = []  # 存储用户ID列表
    
    print("Generating user embeddings...")
    for step, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        seq, token_type, seq_feat, user_id = batch
        seq = seq.to(args.device)
        
        # 模型推理：生成用户序列的表征向量
        logits = model.predict(seq, seq_feat, token_type)
        
        # 收集每个用户的embedding
        for i in range(logits.shape[0]):
            emb = logits[i].unsqueeze(0).detach().cpu().numpy().astype(np.float32)
            all_embs.append(emb)
        user_list += user_id

    # 生成候选物品库的embedding和ID映射文件
    print("Generating candidate item embeddings...")
    retrieve_id2creative_id = get_candidate_emb(
        test_dataset.indexer['i'],  # 物品索引映射
        test_dataset.feature_types,  # 特征类型信息
        test_dataset.feature_default_value,  # 特征默认值
        test_dataset.mm_emb_dict,  # 多模态embedding字典
        model,  # 训练好的模型
    )
    
    # 合并所有用户embedding并保存为query文件
    all_embs = np.concatenate(all_embs, axis=0)
    save_emb(all_embs, Path(os.environ.get('EVAL_RESULT_PATH'), 'query.fbin'))
    
    # 使用FAISS进行近似最近邻(ANN)检索
    if args.use_python_faiss:
        # 使用Python FAISS包进行检索
        perform_python_faiss_search()
    else:
        # 使用命令行FAISS工具进行检索
        print("Performing ANN search with command line FAISS...")
        ann_cmd = (
            str(Path("/workspace", "faiss-based-ann", "faiss_demo"))
            + " --dataset_vector_file_path="  + str(Path(os.environ.get("EVAL_RESULT_PATH"), "embedding.fbin"))  # 候选库embedding文件
            + " --dataset_id_file_path=" + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id.u64bin"))  # 候选库ID文件
            + " --query_vector_file_path=" + str(Path(os.environ.get("EVAL_RESULT_PATH"), "query.fbin"))  # 用户query文件
            + " --result_id_file_path=" + str(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))  # 结果文件
            + " --query_ann_top_k=100"  # 返回Top-10结果
            + " --faiss_M=64 --faiss_ef_construction=1280 --query_ef_search=640"  # FAISS参数配置
            + " --faiss_metric_type=0"  # 相似度度量类型（0为内积）
        )
        os.system(ann_cmd)  # 执行FAISS检索命令

    # 多阶段召回系统实现
    print("Implementing multi-stage recall system...")
    
    if args.use_multistage_recall:
        # 阶段1：粗召回（更大的候选集）
        print("Stage 1: Coarse recall (Top-100)...")
        top100s_retrieved = read_result_ids(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
        
        # 调试信息
        print(f"DEBUG: Retrieved results shape: {len(top100s_retrieved)} users x {len(top100s_retrieved[0]) if len(top100s_retrieved) > 0 else 0} items")
        print(f"DEBUG: retrieve_id2creative_id mapping size: {len(retrieve_id2creative_id)}")
        print(f"DEBUG: User list size: {len(user_list)}")
        
        # 采样检查映射质量
        if len(retrieve_id2creative_id) > 0:
            sample_keys = list(retrieve_id2creative_id.keys())[:10]
            sample_mappings = [(k, retrieve_id2creative_id[k]) for k in sample_keys]
            print(f"DEBUG: Sample mappings: {sample_mappings}")
        
        # 简化多阶段召回：只进行基本的重排，避免复杂逻辑导致结果全是0
        print("Stage 2: Simplified re-ranking...")
        top10s_reranked = []
        
        for user_idx, top100_items in enumerate(top100s_retrieved):
            # 简化逻辑：直接转换ID并选择前10个有效的
            valid_items = []
            
            # 调试前3个用户的处理过程
            if user_idx < 3:
                print(f"DEBUG: Processing user {user_idx+1}, got {len(top100_items)} candidates")
            
            for item_retrieval_id in top100_items[:20]:  # 只检查前20个，避免过多计算
                item_creative_id = retrieve_id2creative_id.get(int(item_retrieval_id), "0")
                
                # 调试前3个用户前5个候选的转换
                if user_idx < 3 and len(valid_items) < 5:
                    print(f"DEBUG: User {user_idx+1}: retrieval_id={item_retrieval_id} -> creative_id='{item_creative_id}'")
                
                # 检查是否为有效ID
                try:
                    item_id_int = int(item_creative_id) if item_creative_id != "0" else 0
                    if item_id_int > 0:
                        valid_items.append(item_creative_id)
                except (ValueError, TypeError):
                    continue
                
                # 收集到10个有效项目就停止
                if len(valid_items) >= 10:
                    break
            
            # 如果有效项目不足10个，补充到10个
            while len(valid_items) < 10:
                # 从剩余的候选中继续寻找
                found = False
                for item_retrieval_id in top100_items[len(valid_items):]:
                    item_creative_id = retrieve_id2creative_id.get(int(item_retrieval_id), "0")
                    if item_creative_id != "0" and item_creative_id not in valid_items:
                        try:
                            if int(item_creative_id) > 0:
                                valid_items.append(item_creative_id)
                                found = True
                                break
                        except (ValueError, TypeError):
                            continue
                
                if not found:
                    valid_items.append("0")  # 无法找到更多有效项目时用0填充
            
            top10s_reranked.append(valid_items[:10])
        
        top10s = top10s_reranked
        print(f"Multi-stage recall completed. Users: {len(top10s)}")
        
    else:
        # 修复的单阶段召回：解决ID转换问题
        print("Using fixed single-stage recall...")
        top10s_retrieved = read_result_ids(Path(os.environ.get("EVAL_RESULT_PATH"), "id100.u64bin"))
        
        # 调试信息
        print(f"DEBUG: Retrieved results shape: {len(top10s_retrieved)} users x {len(top10s_retrieved[0]) if len(top10s_retrieved) > 0 else 0} items")
        print(f"DEBUG: retrieve_id2creative_id mapping size: {len(retrieve_id2creative_id)}")
        
        # 采样检查映射质量
        if len(retrieve_id2creative_id) > 0:
            sample_keys = list(retrieve_id2creative_id.keys())[:10]
            sample_mappings = [(k, retrieve_id2creative_id[k]) for k in sample_keys]
            print(f"DEBUG: Sample mappings: {sample_mappings}")
        
        top10s_final = []
        
        # 为每个用户处理Top-10候选
        for user_idx, user_candidates in enumerate(tqdm(top10s_retrieved, desc="Converting IDs")):
            user_results = []
            
            # 处理每个候选item（前10个）
            for item_idx, item_retrieval_id in enumerate(user_candidates[:10]):
                creative_id = retrieve_id2creative_id.get(int(item_retrieval_id), "0")
                
                # 调试前几个用户的转换过程
                if user_idx < 3 and item_idx < 5:
                    print(f"DEBUG: User {user_idx+1}, Item {item_idx+1}: retrieval_id={item_retrieval_id} -> creative_id='{creative_id}'")
                
                user_results.append(str(creative_id))
            
            # 确保每个用户都有10个结果
            while len(user_results) < 10:
                user_results.append("0")
            
            top10s_final.append(user_results[:10])
        
        top10s = top10s_final
        
        # 统计结果质量
        valid_count = sum(1 for user_list in top10s for item_id in user_list if item_id != "0")
        total_count = len(top10s) * 10
        print(f"DEBUG: Result quality: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%) valid recommendations")


    return top10s, user_list

if __name__ == "__main__":
    print(infer())