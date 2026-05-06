from __future__ import annotations

import torch
from torch import nn


class TAAC2026Scorer(nn.Module):
    """Compact unified-token pCVR scorer for TAAC2026 flat features."""

    def __init__(
        self,
        num_cat_fields: int,
        num_dense_fields: int,
        num_token_fields: int,
        hash_size: int = 200_000,
        dense_hash_size: int = 4_096,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_cat_fields = num_cat_fields
        self.num_dense_fields = num_dense_fields
        self.num_token_fields = num_token_fields
        self.hash_size = hash_size
        self.dense_hash_size = dense_hash_size
        self.hidden_dim = hidden_dim

        self.cat_embedding = nn.Embedding(hash_size + 1, hidden_dim, padding_idx=0)
        self.dense_embedding = nn.Embedding(dense_hash_size + 1, hidden_dim, padding_idx=0)
        self.dense_value_proj = nn.Linear(1, hidden_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.field_embedding = nn.Embedding(num_token_fields, hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cat_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dense_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.cat_embedding.weight[0].fill_(0)
            self.dense_embedding.weight[0].fill_(0)

    def forward(
        self,
        cat_tokens: torch.Tensor,
        dense_values: torch.Tensor,
        dense_mask: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = cat_tokens.size(0)
        pieces: list[torch.Tensor] = []

        if self.num_cat_fields:
            cat_emb = self.cat_embedding(cat_tokens.clamp_min(0).clamp_max(self.hash_size))
            pieces.append(cat_emb)

        if self.num_dense_fields:
            dense_values = torch.nan_to_num(dense_values.float(), nan=0.0, posinf=0.0, neginf=0.0)
            dense_ids = torch.arange(
                1, self.num_dense_fields + 1, device=dense_values.device, dtype=torch.long
            ).unsqueeze(0)
            dense_ids = dense_ids.expand(batch_size, -1)
            dense_emb = self.dense_embedding(dense_ids)
            dense_emb = dense_emb + self.dense_value_proj(dense_values.unsqueeze(-1))
            dense_emb = dense_emb * dense_mask.unsqueeze(-1).float()
            pieces.append(dense_emb)

        time_emb = self.time_proj(time_features.float()).unsqueeze(1)
        pieces.append(time_emb)

        tokens = torch.cat(pieces, dim=1)
        field_ids = torch.arange(tokens.size(1), device=tokens.device, dtype=torch.long).unsqueeze(0)
        tokens = self.token_norm(tokens + self.field_embedding(field_ids))

        encoded = self.encoder(tokens)
        pooled_mean = encoded.mean(dim=1)
        pooled_max = encoded.amax(dim=1)
        cls_like = encoded[:, 0, :]
        logits = self.head(torch.cat([cls_like, pooled_mean, pooled_max], dim=-1)).squeeze(-1)
        return logits
