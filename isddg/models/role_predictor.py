from __future__ import annotations

import torch
from torch import nn


class RolePredictor(nn.Module):
    def __init__(self, input_dim: int, role_dim: int = 4, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, role_dim),
        )

    def forward(self, item_emb: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(item_emb), dim=-1)
