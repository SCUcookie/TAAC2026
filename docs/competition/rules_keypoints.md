# 官方规则要点（来自 rules.md）

## 任务定义
- 目标：预测目标广告/商品的 pCVR。
- 输入：
  - 非序列多域特征（用户、广告、上下文、交叉特征）
  - 用户行为序列（含时间、动作等异构侧信息）
- 倡导方向：
  - 在同一骨干中统一处理序列 token 与非序列 token
  - 探索推荐模型的 scaling law

## 数据结构（官方描述）
- 总列数：120 列，6 个类别
- 主要类别：
  - ID & Label：5 列
  - User Int Features：46 列
  - User Dense Features：10 列
  - Item Int Features：14 列
  - Domain Sequence Features：45 列（4 个 domain）
- 特征形式：
  - 稀疏特征为匿名整数 ID
  - 稠密特征为定长 float 向量
  - 不提供原始文本/图像/URL 或个人敏感信息

## 评测规则
- 排行榜主指标：AUC of ROC（越高越好）
- 强约束：提交需满足官方推理时延预算；超时即无效，不参与排名

## 创新奖（与榜单排名独立）
- Unified Block Innovation Award
- Scaling Law Innovation Award
- 评审依据：技术报告、代码、方法创新与洞察，不只看 AUC

---

## 与当前仓库现状的关键差异
当前仓库主线仍是另一套赛制：
- 排名目标是 Top10 推荐（非 pCVR 标量预测）
- 评估逻辑侧重 HitRate/NDCG（非 AUC）
- 推理产物是用户 Top10 列表（非逐样本概率输出）

因此若要切到 rules.md 对应赛题，需要至少完成：
1. 训练目标切换为二分类概率学习（pCVR）
2. 评估与验证改为 AUC
3. 推理输出改为概率分数并满足官方协议
4. 增加推理延迟基准与约束检查
