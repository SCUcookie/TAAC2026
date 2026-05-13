# TAAC2026 Model Runtime Flow And Formulas

This file contains the detailed end-to-end runtime path and formula-level explanation that was previously embedded in the main progress report.

## 1. Input To Features

For one sample, the raw input can be written as

$$
x = \{x^{(u)}_{int}, x^{(u)}_{dense}, x^{(i)}_{int}, x^{(i)}_{dense}, S_1, S_2, S_3, S_4\}
$$

where $S_k$ is the historical sequence of domain $k \in \{a,b,c,d\}$.

The dataset stage converts raw parquet columns into model tensors:

$$
E_{fid} = \mathrm{Embedding}(id_{fid})
$$

$$
D_{fid} = W_{fid} v_{fid} + b_{fid}
$$

For a sequence domain, each token is assembled as

$$
s_{k,t} = \phi([e_{k,t,1}; e_{k,t,2}; \dots ; e_{k,t,m_k}])
\; + p_{k,t} + b_{k,t} + \psi(\Delta_{k,t})
$$

where:

- $\phi$ is the per-domain projection layer
- $p_{k,t}$ is the time-bucket embedding
- $b_{k,t}$ is the inter-time bucket embedding
- $\psi(\Delta_{k,t})$ is the continuous time-delta MLP branch

Padding positions are masked out, so they do not contribute to the later attention sums.

## 2. NS Token And Query Construction

The model first builds non-sequence tokens $N$ by concatenating user, item, dense, and optional cross-domain dense tokens:

$$
N = [N_u; N_{u,d}; N_i; N_{i,d}; N_x]
$$

The query generator uses mean pooling on each domain sequence:

$$
\bar S_k = \frac{\sum_{t=1}^{L_k} m_{k,t} s_{k,t}}{\sum_{t=1}^{L_k} m_{k,t} + \epsilon}
$$

Then each domain gets its own global summary:

$$
g_k = \mathrm{LN}([\mathrm{vec}(N); \bar S_k])
$$

and generates $N_q$ query tokens:

$$
Q_k = [f_{k,1}(g_k), f_{k,2}(g_k), \dots, f_{k,N_q}(g_k)]
$$

This is the first reason the model works well: each sequence domain gets a separate query family instead of being collapsed into one global vector too early.

## 3. Sequence Attention With RoPE

The self-attention / cross-attention step uses the standard scaled dot-product attention:

$$
\mathrm{Attn}(Q,K,V)
= \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V
$$

where $M$ is the mask matrix and $d_h$ is the head dimension.

For RoPE, the head vectors are rotated before the dot product:

$$
\mathrm{RoPE}(x_t)
= x_t \odot \cos\theta_t
\; + \mathrm{rotate\_half}(x_t) \odot \sin\theta_t
$$

If we write query and key after projection as $q_t$ and $k_t$, then the effective attention score becomes

$$
\alpha_{t,j}
= \frac{\langle \mathrm{RoPE}(q_t), \mathrm{RoPE}(k_j) \rangle}{\sqrt{d_h}}
$$

The practical effect is that relative position information is injected directly into the similarity score, which is better than naive absolute-position addition for long, sparse behavior sequences.

## 4. Multi-Sequence Block Flow

For each block and each domain, the computation is:

1. encode sequence tokens
2. decode query tokens from the encoded sequence
3. optionally fuse target-aware and cross-domain context
4. concatenate all domain queries with NS tokens
5. run RankMixer

The domain-wise update can be written as

$$
H_k^{(l)} = \mathrm{SeqEnc}_k^{(l)}(S_k^{(l-1)})
$$

$$
\tilde Q_k^{(l)} = \mathrm{CrossAttn}_k^{(l)}(Q_k^{(l-1)}, H_k^{(l)})
$$

If target-aware attention is enabled, the query is further refined by the target token context:

$$
Q_{k,tar}^{(l)} = \mathrm{CrossAttn}(\mathrm{Proj}(N_i), H_k^{(l)})
$$

$$
\hat Q_k^{(l)} = \tilde Q_k^{(l)} + \sigma(W_g[\tilde Q_k^{(l)}; Q_{k,tar}^{(l)}]) \odot Q_{k,tar}^{(l)}
$$

If cross-domain attention is enabled, the domain queries exchange information through

$$
Q_k^{(l)} = \hat Q_k^{(l)} + \sigma(W_c[\mathrm{Mean}(\hat Q_k^{(l)}); \mathrm{Mean}(Q_{all}^{(l)})]) \odot \Delta_k^{(l)}
$$

where $Q_{all}^{(l)}$ is the concatenation of all domain queries.

## 5. RankMixer Formula

After the queries from all domains are concatenated with NS tokens, the token mixer runs on

$$
T^{(l)} = [Q_1^{(l)}; Q_2^{(l)}; \dots; Q_K^{(l)}; N^{(l)}]
$$

The learned RankMixer mode can be summarized as

$$
\Delta_T = \mathrm{MLP}_{tok}(\mathrm{LN}(T^{(l)}))
$$

$$
T' = \mathrm{LN}(T^{(l)} + \Delta_T)
$$

Then a shared FFN is applied token-wise:

$$
U = \mathrm{FFN}(\mathrm{LN}(T'))
$$

$$
T^{(l+1)} = \mathrm{LN}(T^{(l)} + U)
$$

This block is useful because it re-mixes information after the domain-specific attention already extracted useful summaries.

## 6. Final Score Head

After $L$ blocks, all domain queries are concatenated:

$$
z = \mathrm{vec}([Q_1^{(L)}; \dots ; Q_K^{(L)}])
$$

Then an output projection maps it to the final latent vector:

$$
h = W_o z + b_o
$$

The classifier outputs the final logit:

$$
\ell = W_c h + b_c
$$

The predicted probability is

$$
p = \sigma(\ell) = \frac{1}{1 + e^{-\ell}}
$$

## 7. Training Objective And Metrics

For binary training, the default loss is BCE:

$$
L_{BCE} = -y\log p - (1-y)\log(1-p)
$$

Focal loss is:

$$
L_{focal} = -\alpha_t (1-p_t)^\gamma \log(p_t)
$$

where

$$
p_t = y p + (1-y)(1-p)
$$

and $\alpha_t$ is the class weight chosen by the focal schedule.

Validation uses AUC:

$$
\mathrm{AUC} = P(s^+ > s^-)
$$

which measures ranking quality rather than calibration. That is why the project always tracks AUC together with logloss.

## 8. Why This Model Works Better

The model performs well because it combines several orthogonal advantages:

1. Sparse features are learned as embeddings, so high-cardinality categorical IDs are not hand-engineered away.
2. Sequence domains are modeled separately, which preserves domain-specific recency patterns.
3. Query generation gives the model multiple learnable views of each sequence instead of one pooled vector.
4. RoPE and time buckets make ordering and recency directly visible to attention.
5. RankMixer gives a second-stage token interaction layer after attention, which helps capture cross-feature composition.
6. Dual-optimizer training and EMA stabilize sparse + dense joint learning.
7. The archived single-pass evaluator avoids inference-side distortion and keeps the best score path consistent.
