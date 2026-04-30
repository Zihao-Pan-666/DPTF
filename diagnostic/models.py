from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureSeqRec(nn.Module):
    """A shared zero-shot sequential recommender over fixed item features.

    It learns only feature projection + sequence encoder on the source domain, then
    switches `item_features` at test time for target-domain zero-shot ranking.
    """

    def __init__(self, item_features: torch.Tensor, hidden_units: int = 128, max_len: int = 50,
                 num_heads: int = 2, num_layers: int = 2, dropout: float = 0.2,
                 use_relative_time: bool = True):
        super().__init__()
        self.hidden_units = hidden_units
        self.max_len = max_len
        self.input_dim = int(item_features.shape[1])
        self.use_relative_time = use_relative_time
        self.register_buffer("item_features", item_features.float())
        self.item_projector = nn.Linear(self.input_dim, hidden_units)
        self.pos_embedding = nn.Embedding(max_len, hidden_units)
        self.time_embedding = nn.Embedding(10, hidden_units) if use_relative_time else None
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_units, nhead=num_heads, dim_feedforward=hidden_units * 4,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout)

    def set_item_features(self, item_features: torch.Tensor):
        self.item_features = item_features.float().to(next(self.parameters()).device)

    def encode_items(self) -> torch.Tensor:
        return self.item_projector(self.item_features)

    def encode_sequence(self, seq: torch.Tensor, rel_time: torch.Tensor | None = None) -> torch.Tensor:
        item_table = self.encode_items()
        x = item_table[seq]
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand_as(seq)
        x = x + self.pos_embedding(positions)
        if self.use_relative_time and rel_time is not None:
            x = x + self.time_embedding(rel_time.clamp(0, 9))
        x = self.dropout(x)
        padding_mask = seq.eq(0)
        h = self.encoder(x, src_key_padding_mask=padding_mask)
        h = self.layer_norm(h)
        lengths = (~padding_mask).sum(dim=1).clamp(min=1) - 1
        return h[torch.arange(seq.size(0), device=seq.device), lengths]

    def forward(self, seq: torch.Tensor, rel_time: torch.Tensor | None = None) -> torch.Tensor:
        user = self.encode_sequence(seq, rel_time)
        item_table = self.encode_items()
        return user @ item_table.t()


class GatedFusionSeqRec(nn.Module):
    def __init__(self, semantic_features: torch.Tensor, dynamics_features: torch.Tensor,
                 hidden_units: int = 128, max_len: int = 50, num_heads: int = 2,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.semantic_branch = FeatureSeqRec(semantic_features, hidden_units, max_len, num_heads, num_layers, dropout, True)
        self.dynamics_branch = FeatureSeqRec(dynamics_features, hidden_units, max_len, num_heads, num_layers, dropout, True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_units * 2, hidden_units), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_units, 1)
        )

    def set_item_features(self, semantic_features: torch.Tensor, dynamics_features: torch.Tensor):
        self.semantic_branch.set_item_features(semantic_features)
        self.dynamics_branch.set_item_features(dynamics_features)

    def forward(self, seq: torch.Tensor, rel_time: torch.Tensor | None = None):
        us = self.semantic_branch.encode_sequence(seq, rel_time)
        ud = self.dynamics_branch.encode_sequence(seq, rel_time)
        gate = torch.sigmoid(self.gate(torch.cat([us, ud], dim=-1)))
        sem_items = self.semantic_branch.encode_items()
        dyn_items = self.dynamics_branch.encode_items()
        sem_scores = us @ sem_items.t()
        dyn_scores = ud @ dyn_items.t()
        return gate * sem_scores + (1.0 - gate) * dyn_scores


class NaiveFusionSeqRec(nn.Module):
    def __init__(self, semantic_features: torch.Tensor, dynamics_features: torch.Tensor,
                 hidden_units: int = 128, max_len: int = 50, num_heads: int = 2,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.semantic_branch = FeatureSeqRec(semantic_features, hidden_units, max_len, num_heads, num_layers, dropout, True)
        self.dynamics_branch = FeatureSeqRec(dynamics_features, hidden_units, max_len, num_heads, num_layers, dropout, True)

    def set_item_features(self, semantic_features: torch.Tensor, dynamics_features: torch.Tensor):
        self.semantic_branch.set_item_features(semantic_features)
        self.dynamics_branch.set_item_features(dynamics_features)

    def forward(self, seq: torch.Tensor, rel_time: torch.Tensor | None = None):
        return 0.5 * self.semantic_branch(seq, rel_time) + 0.5 * self.dynamics_branch(seq, rel_time)
