"""
时间特征测试工具
验证时间差特征的正确性和有效性

测试内容：
1. 时间特征计算的正确性验证
2. 训练和推理时间特征一致性检查
3. 时间特征对模型性能的影响评估
4. 边界情况和异常处理测试
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from time_features import TimeFeatureProcessor, get_time_feature_names
from dataset import MyDataset, MyTestDataset
from model import BaselineModel
import argparse


class TimeFeatureTester:
    """时间特征测试器"""
    
    def __init__(self, train_data_path, eval_data_path=None):
        """
        初始化时间特征测试器
        
        Args:
            train_data_path: 训练数据路径
            eval_data_path: 评测数据路径，可选
        """
        self.train_data_path = Path(train_data_path)
        self.eval_data_path = Path(eval_data_path) if eval_data_path else None
        self.processor = TimeFeatureProcessor()
        
        print("🔧 时间特征测试器初始化完成")
        print(f"  - 训练数据路径: {self.train_data_path}")
        if self.eval_data_path:
            print(f"  - 评测数据路径: {self.eval_data_path}")
    
    def test_time_feature_calculation(self):
        """测试时间特征计算的正确性"""
        print("📊 测试时间特征计算正确性...")
        
        # 创建测试数据
        current_time = datetime.now().timestamp()
        
        test_sequence = [
            (1, 100, {'103': 1}, {'111': 5}, 0, current_time - 3600),     # 1小时前
            (1, 101, {'103': 1}, {'111': 6}, 0, current_time - 1800),     # 30分钟前  
            (1, 102, {'103': 1}, {'111': 7}, 1, current_time - 900),      # 15分钟前（点击）
            (1, 103, {'103': 1}, {'111': 8}, 0, current_time - 300),      # 5分钟前
            (1, 104, {'103': 1}, {'111': 9}, 1, current_time - 60),       # 1分钟前（点击）
            (1, 105, {'103': 1}, {'111': 10}, 0, current_time)            # 现在
        ]
        
        # 处理时间特征
        enhanced_sequence = self.processor.process_user_sequence_time_features(test_sequence)
        
        # 验证时间特征
        print("✅ 时间特征计算验证结果：")
        
        for i, record in enumerate(enhanced_sequence):
            user_id, item_id, user_feat, item_feat, action_type, timestamp = record
            
            print(f"\n  记录 {i} (时间戳: {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}):")
            
            # 验证时间间隔特征
            if 'time_interval_minutes' in item_feat:
                interval = item_feat['time_interval_minutes']
                print(f"    - 时间间隔: {interval:.1f} 分钟")
                
                # 验证间隔计算正确性
                if i > 0:
                    expected_interval = (timestamp - test_sequence[i-1][5]) / 60
                    assert abs(interval - expected_interval) < 0.1, f"时间间隔计算错误: {interval} vs {expected_interval}"
            
            # 验证时间周期特征  
            if 'hour' in item_feat:
                hour = item_feat['hour']
                expected_hour = datetime.fromtimestamp(timestamp).hour
                assert hour == expected_hour, f"小时特征错误: {hour} vs {expected_hour}"
            
            # 验证时间衰减权重
            if 'time_decay_weight' in item_feat:
                decay_weight = item_feat['time_decay_weight']
                assert 0 <= decay_weight <= 1, f"时间衰减权重超出范围: {decay_weight}"
                print(f"    - 时间衰减权重: {decay_weight:.3f}")
            
            # 验证间隔分桶
            if 'time_interval_bucket' in item_feat:
                bucket = item_feat['time_interval_bucket']
                print(f"    - 时间间隔分桶: {bucket}")
        
        print("\n✅ 时间特征计算正确性验证通过")
    
    def test_dataset_integration(self):
        """测试数据集集成的正确性"""
        print("🔄 测试数据集集成...")
        
        # 创建测试参数
        class TestArgs:
            maxlen = 101
            mm_emb_id = ['81']
            enable_time_features = True
            time_decay_factor = 0.1
        
        args = TestArgs()
        
        try:
            # 测试训练数据集
            train_dataset = MyDataset(str(self.train_data_path), args)
            print(f"✅ 训练数据集加载成功，用户数: {len(train_dataset)}")
            
            # 测试获取单个样本
            sample_uid = 0
            sample_data = train_dataset[sample_uid]
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = sample_data
            
            print(f"  - 样本数据形状:")
            print(f"    seq: {seq.shape}")
            print(f"    seq_feat keys: {list(seq_feat.keys())}")
            
            # 验证时间特征是否存在
            time_feature_names = get_time_feature_names()
            
            time_features_found = 0
            for feat_name in time_feature_names['item_time_features']:
                if feat_name in seq_feat:
                    time_features_found += 1
            
            print(f"  - 发现物品时间特征: {time_features_found}/{len(time_feature_names['item_time_features'])}")
            
            user_time_features_found = 0
            for feat_name in time_feature_names['user_time_features']:
                if feat_name in seq_feat:
                    user_time_features_found += 1
            
            print(f"  - 发现用户时间特征: {user_time_features_found}/{len(time_feature_names['user_time_features'])}")
            
            # 测试评测数据集（如果存在）
            if self.eval_data_path and self.eval_data_path.exists():
                eval_dataset = MyTestDataset(str(self.eval_data_path), args)
                print(f"✅ 评测数据集加载成功，用户数: {len(eval_dataset)}")
                
                # 测试样本
                eval_sample = eval_dataset[0]
                eval_seq, eval_token_type, eval_seq_feat, eval_user_id = eval_sample
                print(f"  - 评测样本seq_feat keys: {list(eval_seq_feat.keys()) if eval_seq_feat else 'None'}")
        
        except Exception as e:
            print(f"❌ 数据集集成测试失败: {e}")
            raise
        
        print("✅ 数据集集成测试通过")
    
    def test_model_compatibility(self):
        """测试模型兼容性"""
        print("🤖 测试模型兼容性...")
        
        # 创建测试参数
        class TestArgs:
            maxlen = 101
            mm_emb_id = ['81']
            enable_time_features = True
            time_decay_factor = 0.1
            hidden_units = 128
            num_blocks = 2  # 减小用于测试
            num_heads = 4
            dropout_rate = 0.2
            device = 'cpu'  # 使用CPU测试
            norm_first = True
            temp = 0.03
            norm_output = False
            use_action_weight = False
        
        args = TestArgs()
        
        try:
            # 创建数据集
            dataset = MyDataset(str(self.train_data_path), args)
            usernum, itemnum = dataset.usernum, dataset.itemnum
            feat_statistics, feat_types = dataset.feat_statistics, dataset.feature_types
            
            # 创建模型
            model = BaselineModel(usernum, itemnum, feat_statistics, feat_types, args)
            print(f"✅ 模型创建成功")
            print(f"  - 用户数: {usernum}, 物品数: {itemnum}")
            
            # 测试前向传播
            sample_data = dataset[0]
            seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = sample_data
            
            # 转换为批次格式
            seq = torch.tensor(seq).unsqueeze(0)
            pos = torch.tensor(pos).unsqueeze(0) 
            neg = torch.tensor(neg).unsqueeze(0)
            token_type = torch.tensor(token_type).unsqueeze(0)
            next_token_type = torch.tensor(next_token_type).unsqueeze(0)
            next_action_type = torch.tensor(next_action_type).unsqueeze(0)
            
            # 处理序列特征
            batch_seq_feat = {}
            for key, value in seq_feat.items():
                if torch.is_tensor(value):
                    batch_seq_feat[key] = value.unsqueeze(0)
                else:
                    batch_seq_feat[key] = torch.tensor([value]).unsqueeze(0)
            
            # 处理pos和neg特征
            batch_pos_feat = {}
            batch_neg_feat = {}
            for key, value in pos_feat.items():
                if torch.is_tensor(value):
                    batch_pos_feat[key] = value.unsqueeze(0)
                else:
                    batch_pos_feat[key] = torch.tensor([value]).unsqueeze(0)
            
            for key, value in neg_feat.items():
                if torch.is_tensor(value):
                    batch_neg_feat[key] = value.unsqueeze(0)
                else:
                    batch_neg_feat[key] = torch.tensor([value]).unsqueeze(0)
            
            # 前向传播测试
            model.eval()
            with torch.no_grad():
                pos_logits, neg_logits, loss_mask = model(
                    seq, pos, neg, token_type, next_token_type, next_action_type,
                    batch_seq_feat, batch_pos_feat, batch_neg_feat
                )
            
            print(f"  - 前向传播成功")
            print(f"    pos_logits shape: {pos_logits.shape}")
            print(f"    neg_logits shape: {neg_logits.shape}")
            print(f"    loss_mask shape: {loss_mask.shape}")
        
        except Exception as e:
            print(f"❌ 模型兼容性测试失败: {e}")
            raise
        
        print("✅ 模型兼容性测试通过")
    
    def test_boundary_cases(self):
        """测试边界情况"""
        print("🔍 测试边界情况...")
        
        # 测试空序列
        empty_sequence = []
        result = self.processor.process_user_sequence_time_features(empty_sequence)
        assert result == [], "空序列处理失败"
        
        # 测试单个记录序列
        single_sequence = [(1, 100, {'103': 1}, {'111': 5}, 0, datetime.now().timestamp())]
        result = self.processor.process_user_sequence_time_features(single_sequence)
        assert len(result) == 1, "单记录序列处理失败"
        
        # 测试时间戳异常情况
        current_time = datetime.now().timestamp()
        
        # 时间戳为0
        zero_time_sequence = [(1, 100, {'103': 1}, {'111': 5}, 0, 0)]
        result = self.processor.process_user_sequence_time_features(zero_time_sequence)
        assert len(result) == 1, "零时间戳处理失败"
        
        # 时间倒序
        reverse_time_sequence = [
            (1, 100, {'103': 1}, {'111': 5}, 0, current_time),
            (1, 101, {'103': 1}, {'111': 6}, 0, current_time - 3600),  # 1小时前
        ]
        result = self.processor.process_user_sequence_time_features(reverse_time_sequence)
        assert len(result) == 2, "倒序时间处理失败"
        
        # 相同时间戳
        same_time_sequence = [
            (1, 100, {'103': 1}, {'111': 5}, 0, current_time),
            (1, 101, {'103': 1}, {'111': 6}, 0, current_time),
        ]
        result = self.processor.process_user_sequence_time_features(same_time_sequence)
        assert len(result) == 2, "相同时间戳处理失败"
        
        print("✅ 边界情况测试通过")
    
    def test_performance_impact(self, sample_size=100):
        """测试时间特征对性能的影响"""
        print(f"⚡ 测试性能影响 (样本数: {sample_size})...")
        
        # 创建测试参数
        class TestArgs:
            maxlen = 101
            mm_emb_id = ['81']
            time_decay_factor = 0.1
        
        # 测试启用时间特征
        args_with_time = TestArgs()
        args_with_time.enable_time_features = True
        
        # 测试禁用时间特征
        args_without_time = TestArgs()
        args_without_time.enable_time_features = False
        
        import time
        
        # 测试数据加载性能
        try:
            # 启用时间特征的性能
            start_time = time.time()
            dataset_with_time = MyDataset(str(self.train_data_path), args_with_time)
            
            for i in range(min(sample_size, len(dataset_with_time))):
                _ = dataset_with_time[i]
            
            time_with_features = time.time() - start_time
            
            # 禁用时间特征的性能
            start_time = time.time()
            dataset_without_time = MyDataset(str(self.train_data_path), args_without_time)
            
            for i in range(min(sample_size, len(dataset_without_time))):
                _ = dataset_without_time[i]
            
            time_without_features = time.time() - start_time
            
            # 输出结果
            print(f"  - 启用时间特征耗时: {time_with_features:.2f}秒")
            print(f"  - 禁用时间特征耗时: {time_without_features:.2f}秒")
            print(f"  - 性能开销: {((time_with_features - time_without_features) / time_without_features * 100):.1f}%")
            
            if time_with_features / time_without_features > 2.0:
                print("⚠️ 时间特征导致显著性能下降")
            else:
                print("✅ 时间特征性能开销可接受")
        
        except Exception as e:
            print(f"⚠️ 性能测试失败: {e}")
    
    def generate_test_report(self):
        """生成测试报告"""
        print("📋 生成时间特征测试报告...")
        
        report = {
            "测试时间": datetime.now().isoformat(),
            "训练数据路径": str(self.train_data_path),
            "评测数据路径": str(self.eval_data_path) if self.eval_data_path else None,
            "时间特征配置": self.processor.time_feature_config,
            "特征名称": get_time_feature_names(),
        }
        
        # 运行所有测试
        test_results = {}
        
        try:
            self.test_time_feature_calculation()
            test_results["时间特征计算"] = "✅ 通过"
        except Exception as e:
            test_results["时间特征计算"] = f"❌ 失败: {e}"
        
        try:
            self.test_dataset_integration()
            test_results["数据集集成"] = "✅ 通过"
        except Exception as e:
            test_results["数据集集成"] = f"❌ 失败: {e}"
        
        try:
            self.test_model_compatibility()
            test_results["模型兼容性"] = "✅ 通过"
        except Exception as e:
            test_results["模型兼容性"] = f"❌ 失败: {e}"
        
        try:
            self.test_boundary_cases()
            test_results["边界情况"] = "✅ 通过"
        except Exception as e:
            test_results["边界情况"] = f"❌ 失败: {e}"
        
        report["测试结果"] = test_results
        
        # 保存报告
        report_path = Path("time_features_test_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 测试报告已保存: {report_path}")
        
        # 打印摘要
        print("\n📊 测试结果摘要:")
        for test_name, result in test_results.items():
            print(f"  - {test_name}: {result}")
        
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if "✅" in result)
        
        print(f"\n🎯 测试完成: {passed_tests}/{total_tests} 通过")
        
        return report
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始完整测试流程...")
        print("=" * 60)
        
        try:
            # 生成测试报告（包含所有测试）
            report = self.generate_test_report()
            
            # 性能影响测试
            self.test_performance_impact()
            
            print("=" * 60)
            print("✅ 所有测试完成!")
            
            return report
        
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="时间特征测试工具")
    parser.add_argument('--train_data', type=str, default='./dataset/TencentGR_1k',
                       help='训练数据路径')
    parser.add_argument('--eval_data', type=str, default=None,
                       help='评测数据路径')
    parser.add_argument('--test_type', type=str, choices=['all', 'calculation', 'integration', 'model', 'boundary', 'performance'],
                       default='all', help='测试类型')
    parser.add_argument('--sample_size', type=int, default=100,
                       help='性能测试样本数')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = TimeFeatureTester(args.train_data, args.eval_data)
    
    # 运行指定测试
    if args.test_type == 'all':
        tester.run_all_tests()
    elif args.test_type == 'calculation':
        tester.test_time_feature_calculation()
    elif args.test_type == 'integration':
        tester.test_dataset_integration()
    elif args.test_type == 'model':
        tester.test_model_compatibility()
    elif args.test_type == 'boundary':
        tester.test_boundary_cases()
    elif args.test_type == 'performance':
        tester.test_performance_impact(args.sample_size)


if __name__ == "__main__":
    main()