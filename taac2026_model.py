from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SwiGLUEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim * 2)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        value, gate = self.gate(x).chunk(2, dim=-1)
        x = value * F.silu(gate)
        return residual + self.dropout(self.out(x))


class RankMixer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim * 4)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, user_vec: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [user_vec, item_vec, user_vec * item_vec, torch.abs(user_vec - item_vec)],
            dim=-1,
        )
        return self.mlp(self.norm(features))


class LongerEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        padding_mask = ~valid_mask.bool()
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)
        encoded = self.norm(encoded)
        weights = valid_mask.float().unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (encoded * weights).sum(dim=1) / denom


class MultiSeqQueryAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, seq_vecs: torch.Tensor) -> torch.Tensor:
        batch_size = seq_vecs.size(0)
        query = self.query.expand(batch_size, -1, -1)
        out, _ = self.attn(query, seq_vecs, seq_vecs, need_weights=False)
        return self.norm(out.squeeze(1))


class HyFormerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = SwiGLUEncoder(hidden_dim, dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normed = self.attn_norm(tokens)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attn_out
        return self.ffn(tokens)


class PCVRHyFormer(nn.Module):
    """Official-baseline-inspired PCVRHyFormer for TAAC2026 pCVR scoring."""

    def __init__(
        self,
        num_ns_groups: int,
        num_seq_domains: int,
        hash_size: int = 200_000,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_ns_groups = num_ns_groups
        self.num_seq_domains = num_seq_domains
        self.hash_size = hash_size
        self.hidden_dim = hidden_dim

        self.id_embedding = nn.Embedding(hash_size + 1, hidden_dim, padding_idx=0)
        self.ns_group_embedding = nn.Embedding(max(num_ns_groups, 1), hidden_dim)
        self.seq_domain_embedding = nn.Embedding(max(num_seq_domains, 1), hidden_dim)
        self.ns_value_proj = nn.Linear(1, hidden_dim)
        self.time_proj = nn.Sequential(nn.Linear(6, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        self.ns_encoders = nn.ModuleList([SwiGLUEncoder(hidden_dim, dropout) for _ in range(max(num_ns_groups, 1))])
        self.seq_encoders = nn.ModuleList(
            [LongerEncoder(hidden_dim, num_heads, max(1, num_layers // 2), dropout) for _ in range(max(num_seq_domains, 1))]
        )
        self.seq_query = MultiSeqQueryAttention(hidden_dim, num_heads, dropout)
        self.hyformer = nn.ModuleList([HyFormerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.rank_mixer = RankMixer(hidden_dim, dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.id_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.id_embedding.weight[0].fill_(0)

    def forward(
        self,
        ns_tokens: torch.Tensor,
        ns_dense: torch.Tensor,
        seq_tokens: torch.Tensor,
        seq_mask: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = ns_tokens.size(0)

        ns_emb = self.id_embedding(ns_tokens.clamp_min(0).clamp_max(self.hash_size))
        ns_emb = ns_emb + self.ns_value_proj(ns_dense.float().unsqueeze(-1))
        if self.num_ns_groups:
            group_ids = torch.arange(self.num_ns_groups, device=ns_tokens.device).unsqueeze(0)
            ns_emb = ns_emb + self.ns_group_embedding(group_ids)
            ns_parts = [self.ns_encoders[i](ns_emb[:, i, :]) for i in range(self.num_ns_groups)]
            ns_vecs = torch.stack(ns_parts, dim=1)
        else:
            ns_vecs = ns_emb

        seq_vec_parts = []
        for domain_idx in range(self.num_seq_domains):
            domain_tokens = seq_tokens[:, domain_idx, :]
            domain_mask = seq_mask[:, domain_idx, :]
            domain_emb = self.id_embedding(domain_tokens.clamp_min(0).clamp_max(self.hash_size))
            domain_id = torch.full((1, 1), domain_idx, device=seq_tokens.device, dtype=torch.long)
            domain_emb = domain_emb + self.seq_domain_embedding(domain_id)
            seq_vec_parts.append(self.seq_encoders[domain_idx](domain_emb, domain_mask))

        if seq_vec_parts:
            seq_vecs = torch.stack(seq_vec_parts, dim=1)
            seq_summary = self.seq_query(seq_vecs)
        else:
            seq_vecs = ns_vecs.new_zeros((batch_size, 1, self.hidden_dim))
            seq_summary = seq_vecs.squeeze(1)

        time_token = self.time_proj(time_features.float()).unsqueeze(1)
        tokens = torch.cat([ns_vecs, seq_vecs, time_token], dim=1)
        for block in self.hyformer:
            tokens = block(tokens)

        pooled = tokens.mean(dim=1)
        user_vec = tokens[:, : max(self.num_ns_groups, 1), :].mean(dim=1) + seq_summary
        item_vec = ns_vecs[:, 1, :] if self.num_ns_groups > 1 else ns_vecs.mean(dim=1)
        rank_vec = self.rank_mixer(user_vec, item_vec)
        logits = self.head(torch.cat([pooled, seq_summary, item_vec, rank_vec], dim=-1)).squeeze(-1)
        return logits


TAAC2026Scorer = PCVRHyFormer
