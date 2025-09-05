# 优化配置文件
# Configuration for the optimized recommendation system

import os

class OptimizationConfig:
    """
    优化系统配置类
    包含所有性能优化和缓存相关的配置选项
    """
    
    # 缓存配置
    USE_OPTIMIZED_PIPELINE = os.environ.get('USE_OPTIMIZED_PIPELINE', 'true').lower() == 'true'
    CACHE_DIR = os.environ.get('USER_CACHE_PATH', './cache')
    ENABLE_CANDIDATE_CACHING = True
    CACHE_EMBEDDING_BATCH_SIZE = 128
    
    # 内存优化配置
    MEMORY_EFFICIENT_USER_PROCESSING = True
    USE_FLOAT16_FOR_CACHE = True
    GPU_CACHE_CLEANUP_INTERVAL = 100  # 每100步清理一次GPU缓存
    
    # 数据预处理配置
    ENHANCED_FEATURE_PROCESSING = True
    CANDIDATE_BATCH_SIZE = 1000
    FEATURE_115_DEFAULT_VALUE = 493  # 基于数据分析的默认值
    
    # 混合类型特征列表（基于评测数据集分析）
    MIXED_TYPE_FEATURES = ['102', '111', '115', '117', '118', '119', '120', '121', '122']
    
    # ANN检索配置
    ANN_SEARCH_BATCH_SIZE = 1000
    FAISS_TOOL_PATH = "/workspace/faiss-based-ann/faiss_demo"
    USE_SIMPLIFIED_ANN_FALLBACK = True
    
    # FAISS参数
    FAISS_M = 64
    FAISS_EF_CONSTRUCTION = 1280
    FAISS_EF_SEARCH = 640
    FAISS_METRIC_TYPE = 0
    
    # 性能监控配置
    ENABLE_PERFORMANCE_MONITORING = True
    MEMORY_LOG_INTERVAL = 500  # 每500步记录内存使用
    TRAINING_PROGRESS_LOG_INTERVAL = 100  # 每100步记录训练进度
    
    # 训练优化配置
    PARALLEL_DATA_LOADING = True
    NUM_WORKERS = 4
    CHECKPOINT_CACHING = True  # 是否将checkpoint额外保存到缓存目录
    
    # 验证配置
    ENABLE_RESULT_VALIDATION = True
    VALIDATION_SAMPLE_SIZE = 100  # 验证时抽样检查的样本数量
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("=== Optimization Configuration ===")
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and not callable(getattr(cls, attr_name)):
                value = getattr(cls, attr_name)
                print(f"{attr_name}: {value}")
        print("=" * 35)
    
    @classmethod
    def get_cache_paths(cls):
        """获取缓存路径配置"""
        cache_dir = cls.CACHE_DIR
        return {
            'base': cache_dir,
            'embeddings': os.path.join(cache_dir, 'embeddings'),
            'checkpoints': os.path.join(cache_dir, 'checkpoints'),
            'candidate_embeddings': os.path.join(cache_dir, 'candidate_embeddings.pt'),
            'training_stats': os.path.join(cache_dir, 'training_stats.json')
        }


# 环境变量配置说明
ENVIRONMENT_VARIABLES = {
    'USE_OPTIMIZED_PIPELINE': 'true/false - 是否使用优化的推理流程',
    'USER_CACHE_PATH': '缓存目录路径（20GB配额）',
    'TRAIN_DATA_PATH': '训练数据集路径',
    'TRAIN_CKPT_PATH': '模型检查点保存路径',
    'TRAIN_TF_EVENTS_PATH': 'TensorBoard事件保存路径',
    'TRAIN_LOG_PATH': '训练日志保存路径',
    'EVAL_DATA_PATH': '评测数据集路径',
    'EVAL_RESULT_PATH': '评测结果保存路径',
    'MODEL_OUTPUT_PATH': '模型输出路径（推理时使用）'
}

def print_environment_guide():
    """打印环境变量使用指南"""
    print("=== Environment Variables Guide ===")
    for var, description in ENVIRONMENT_VARIABLES.items():
        value = os.environ.get(var, 'Not set')
        print(f"{var}: {description}")
        print(f"  Current value: {value}")
    print("=" * 37)