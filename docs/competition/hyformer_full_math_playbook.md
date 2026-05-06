# HyFormer 全流程原理与公式详解（面向本次比赛）

> 目的：将比赛任务、数据结构、统一建模思路、权重矩阵设计、损失函数、训练/推理流程、时延约束、实验方案，全部给出可执行的数学化说明。
>
> 使用方式：后续每次迭代先快速回顾本文件，再进行代码改动与实验。

---

## 1. 先把比赛任务写成数学问题

### 1.1 输入与输出

每条样本可表示为：

$$
x=(u, c, i, s),\quad y\in\{0,1\}
$$

其中：
- $u$：用户侧非序列特征（离散 + 稠密）
- $c$：上下文与交叉特征
- $i$：目标广告/商品特征
- $s$：用户历史行为序列（多域、含时间与行为类型）

模型输出：

$$
\hat y=f_\theta(x)=P(y=1\mid u,c,i,s)
$$

这是 pCVR 估计问题。

### 1.2 排名指标与工程约束

官方给出的核心约束：
- 主榜指标：AUC of ROC（越高越好）
- 提交必须满足推理时延预算；超时即无效

所以目标不是“只提精度”，而是：

$$
\max_\theta\ \text{AUC}(\theta)\quad \text{s.t.}\quad \text{Latency}(\theta)\le L_{\max}
$$

这是一类带约束优化问题。

---

## 2. 数据如何映射为统一 token

设统一隐空间维度为 $d$，总 token 数为 $N=M+T$：
- $M$：非序列 token 数
- $T$：序列 token 数

最终输入主干矩阵：

$$
X\in\mathbb R^{N\times d}
$$

### 2.1 离散特征 token 化

第 $j$ 个离散字段词表大小 $V_j$，embedding 表：

$$
E_j\in\mathbb R^{V_j\times d_j}
$$

投影到统一维度：

$$
t_j=\text{LN}(E_j[x_j]W_j+b_j+e_{\text{field}(j)}),\quad W_j\in\mathbb R^{d_j\times d}
$$

这里 $e_{\text{field}(j)}$ 是字段类型 embedding，用于告诉模型“这是哪一类字段”。

### 2.2 稠密特征 token 化

第 $k$ 个稠密向量 $z_k\in\mathbb R^{m_k}$：

$$
t_k=\text{LN}(z_kU_k+b_k+e_{\text{field}(k)}),\quad U_k\in\mathbb R^{m_k\times d}
$$

### 2.3 序列事件 token 化

第 $t$ 个行为事件（item/action/time/domain）融合为：

$$
s_t=\text{LN}\Big(\sum_r E_r[x_{t,r}]W_r+\phi_{\text{time}}(\Delta t_t)W_\tau+\phi_{\text{act}}(a_t)W_a+e_{\text{domain}(t)}\Big)
$$

然后拼接：

$$
X=[t_1,\dots,t_M,s_1,\dots,s_T]^\top
$$

---

## 3. HyFormer 式统一块：为什么是“统一”

统一块的目标：同一套参数同时处理非序列 token 和序列 token，并让两者显式交互。

### 3.1 基础注意力方程

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
L=\frac{QK^\top}{\sqrt{d_h}}+B_{\text{struct}}+B_{\text{time}}+M
$$

$$
A=\text{softmax}(L),\quad O=AV
$$

残差 + 前馈：

$$
H'=\text{LN}(X+O),\quad H=\text{LN}(H'+\text{FFN}(H'))
$$

### 3.2 分块矩阵视角（核心）

将 token 分成非序列（ns）与序列（s）：

$$
X=\begin{bmatrix}X_{\text{ns}}\\X_{\text{s}}\end{bmatrix}
$$

logits 分块：

$$
L=\begin{bmatrix}
L_{\text{nn}} & L_{\text{ns}}\\
L_{\text{sn}} & L_{\text{ss}}
\end{bmatrix}
$$

四块含义：
- $L_{\text{nn}}$：非序列字段交互（多域特征组合）
- $L_{\text{ss}}$：序列时序建模（含因果和时间约束）
- $L_{\text{ns}},L_{\text{sn}}$：序列与非序列桥接（比赛主题重点）

这就是“统一块”的数学本质。

---

## 4. 权重矩阵如何按数据结构设计

### 4.1 标准全共享参数

最简单设置：

$$
W_Q,W_K,W_V\in\mathbb R^{d\times d}
$$

优点：实现简单；缺点：不同 token 类型差异大时，表达可能不够精细。

### 4.2 类型感知投影（推荐）

给不同 token 类型独立投影：

$$
Q_i=X_iW_Q^{(\tau_i)},\quad K_i=X_iW_K^{(\tau_i)},\quad V_i=X_iW_V^{(\tau_i)}
$$

其中 $\tau_i\in\{\text{ns},\text{s}\}$ 或更细粒度字段类型。

这会增加参数量，但显著提升异构数据适配能力。

### 4.3 字段关系偏置矩阵（强烈建议）

定义字段组集合 $\mathcal G$，学习关系表：

$$
R\in\mathbb R^{|\mathcal G|\times|\mathcal G|}
$$

对 token 对 $(i,j)$：

$$
(B_{\text{struct}})_{ij}=R_{g(i),g(j)}
$$

这等价于在注意力 logits 中加入“业务先验路由”。

### 4.4 时间偏置

常见写法：

$$
(B_{\text{time}})_{ij}=-\alpha\log(1+\Delta t_{ij})
$$

或

$$
(B_{\text{time}})_{ij}=-\beta\Delta t_{ij}
$$

时间越远，权重越低，符合点击/转化的短期依赖规律。

### 4.5 门控融合（防止跨域噪声）

$$
G=\sigma(XW_g+b_g),\quad O=G\odot O_{\text{cross}}+(1-G)\odot O_{\text{intra}}
$$

门控可抑制无效跨域交互，常用于工业 CTR/CVR 模型。

---

## 5. 从损失函数角度对齐 AUC

### 5.1 点式 BCE

$$
p=\sigma(w_o^\top h+b),\quad
\mathcal L_{\text{bce}}=-y\log p-(1-y)\log(1-p)
$$

优点：稳定；缺点：与 AUC 不完全一致。

### 5.2 对偶排序损失

采样正负对：

$$
\Delta=s^+-s^-,\quad
\mathcal L_{\text{pair}}=\log(1+e^{-\Delta})
$$

更直接优化“正样本分数高于负样本”。

### 5.3 组合目标

$$
\mathcal L=\lambda\mathcal L_{\text{bce}}+(1-\lambda)\mathcal L_{\text{pair}}+\gamma\lVert\theta\rVert_2^2
$$

建议初始：$\lambda\in[0.6,0.8]$。

---

## 6. 时延约束下的复杂度与矩阵压缩

全注意力复杂度：

$$
\mathcal O(N^2d)=\mathcal O((M+T)^2d)
$$

当 $T$ 大时，主要由 $T^2$ 主导。

### 6.1 可行降耗手段

1) 局部/块稀疏注意力：降低 $L_{\text{ss}}$ 计算成本  
2) 非序列 token 聚合：降低 $M$  
3) 低秩分解：

$$
W_Q=U_QV_Q^\top,\quad U_Q\in\mathbb R^{d\times r},\ V_Q\in\mathbb R^{d_h\times r}
$$

参数从 $dd_h$ 降为 $r(d+d_h)$。

4) 多头裁剪：减少 $H$ 和 $d_h$（注意 AUC 损失）

---

## 7. 结合本赛题字段结构的一个具体例子

假设：
- 非序列 token：
  - user int 6 个
  - user dense 2 个
  - item int 4 个
  - context/cross 4 个
  - 合计 $M=16$
- 序列长度截断 $T=80$
- 统一维度 $d=256$，头数 $H=8$，每头 $d_h=32$

则单层注意力 logits 维度：

$$
L\in\mathbb R^{96\times 96}
$$

其中分块：
- $L_{\text{nn}}\in\mathbb R^{16\times16}$
- $L_{\text{ns}}\in\mathbb R^{16\times80}$
- $L_{\text{sn}}\in\mathbb R^{80\times16}$
- $L_{\text{ss}}\in\mathbb R^{80\times80}$

若启用局部窗口 $w=20$，则 $L_{\text{ss}}$ 从 $80^2$ 变为约 $80\times20$ 级别，时延显著下降。

---

## 8. 训练流程的严格分步（可直接落地）

### Step 1 数据预处理
- 对 list<int64> 字段做长度截断与 padding
- 对 list<float> 对齐字段保留一一对应关系
- 序列按时间排序，构造 $\Delta t$

### Step 2 tokenizer
- 每个字段映射到统一维度 $d$
- 添加字段类型 embedding 与域标识 embedding

### Step 3 unified blocks
- 先 2 层小模型验证通路
- 再扩展到 4/6 层并做早停

### Step 4 目标函数
- 先 BCE 预热
- 再加 pairwise loss 提升 AUC

### Step 5 验证
- 每轮记录 AUC、LogLoss、推理吞吐
- 固定验证集，避免泄漏与漂移

### Step 6 推理时延
- 统计 p50/p95 单样本时延
- 超预算即回退模型规模

---

## 9. Scaling Law 的可操作实验网格

### 9.1 模型尺度
- 层数 $L\in\{2,4,6,8\}$
- 宽度 $d\in\{128,192,256,320\}$
- 头数 $H\in\{4,8\}$

### 9.2 数据尺度
- 训练样本比例 $\rho\in\{0.1,0.25,0.5,1.0\}$

### 9.3 经验拟合
可拟合：

$$
\text{AUC}_{\text{err}}(N)\approx aN^{-b}+c
$$

其中 $N$ 可是参数量或 token-budget。

---

## 10. 常见误区与纠偏

误区 1：只追 AUC，不看时延。  
纠偏：每次实验必须双指标记录（AUC + p95 latency）。

误区 2：把所有字段硬拼到同一个 embedding，不区分类型。  
纠偏：字段类型偏置 + 关系偏置必须上。

误区 3：序列时间信息只做绝对时间戳，不做时间差。  
纠偏：使用 $\Delta t$ 与可学习衰减偏置。

误区 4：只用 BCE，不做 AUC 对齐。  
纠偏：组合损失，或至少后期切换 pairwise 微调。

---

## 11. 与当前仓库主线的对齐建议

当前仓库历史主线更接近 TopK 推荐赛制，而规则文件对应 pCVR + AUC 赛制。建议分支化进行：

- 新建赛道分支：`track_auc_unifiedblock`
- 单独维护：
  - 数据读取器（支持 120 列结构）
  - AUC 训练脚本
  - 低时延推理脚本

避免在原 TopK 流程上硬改导致两套逻辑互相污染。

---

## 12. 快速复盘模板（每次实验前必看）

1. 当前模型是统一块还是拼接后 MLP？  
2. 是否有字段关系偏置矩阵 $R$？  
3. 是否有时间偏置 $B_{\text{time}}$？  
4. 是否记录了 AUC 与 p95 latency？  
5. 当前瓶颈在精度还是时延？  
6. 下一步调参只改一类变量，是否满足？

---

## 13. 一句话总结

本赛题的最优实践不是“堆更大的注意力”，而是：
- 用统一块把序列与非序列特征放在同一计算图中；
- 用结构偏置和时间偏置把业务归纳偏置写进权重矩阵；
- 用 AUC 对齐损失和时延约束共同驱动模型选择。

当你把这三者同时满足，模型才会在官方榜单和工程约束下都稳定有效。
