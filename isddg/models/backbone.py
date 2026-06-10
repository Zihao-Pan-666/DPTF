from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FeatureBERT4Rec(nn.Module):
    """BERT4Rec-style feature sequence encoder.

    It uses item feature tables instead of trainable item-id embeddings, so the
    source-trained encoder can be switched to target-domain item features.
    """
    def __init__(
        self,
        item_features: torch.Tensor,
        hidden_dim: int = 128,
        max_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
        role_features: torch.Tensor | None = None,
        role_alpha: float = 0.0,
    ):
        super().__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.role_alpha = role_alpha

        self.register_buffer("item_features", item_features.float(), persistent=False)
        in_dim = item_features.size(-1)
        self.item_proj = nn.Linear(in_dim, hidden_dim)

        self.role_proj = None
        if role_features is not None:
            self.register_buffer("role_features", role_features.float(), persistent=False)
            self.role_proj = nn.Linear(role_features.size(-1), hidden_dim)
        else:
            self.role_features = None

        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def set_item_features(self, item_features: torch.Tensor, role_features: torch.Tensor | None = None) -> None:
        self.item_features = item_features.float().to(next(self.parameters()).device)
        if role_features is not None:
            self.role_features = role_features.float().to(next(self.parameters()).device)

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        feats = self.item_features[item_ids]
        x = self.item_proj(feats)
        if self.role_proj is not None and self.role_features is not None and self.role_alpha != 0.0:
            r = self.role_features[item_ids].detach()
            x = x + self.role_alpha * self.role_proj(r)
        return x

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: [B, L]
        B, L = history.shape
        x = self.encode_items(history)
        pos = torch.arange(L, device=history.device).unsqueeze(0).expand(B, L)
        x = self.dropout(x + self.pos_emb(pos))
        pad_mask = history.eq(0)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        h = self.layer_norm(h)
        # representation at the last non-pad position.
        lengths = (~pad_mask).sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(B, 1, 1).expand(B, 1, self.hidden_dim)
        return h.gather(dim=1, index=idx).squeeze(1)

    def score(self, history: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        user_h = self.forward(history)
        cand_h = self.encode_items(candidates)
        return torch.einsum("bd,bnd->bn", user_h, cand_h)

    def score_all(self, history: torch.Tensor) -> torch.Tensor:
        user_h = self.forward(history)
        item_ids = torch.arange(self.item_features.size(0), device=history.device)
        item_h = self.encode_items(item_ids)
        return user_h @ item_h.t()
