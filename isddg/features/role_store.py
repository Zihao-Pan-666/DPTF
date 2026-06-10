from __future__ import annotations

from pathlib import Path
import torch


def load_role_table(path: str | Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def align_feature_table(table: torch.Tensor, num_items: int) -> torch.Tensor:
    if table.size(0) >= num_items + 1:
        return table[: num_items + 1]
    pad = torch.zeros(num_items + 1 - table.size(0), table.size(1), dtype=table.dtype)
    return torch.cat([table, pad], dim=0)
