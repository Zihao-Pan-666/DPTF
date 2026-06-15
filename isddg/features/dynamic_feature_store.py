from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict, Any
import json
import numpy as np
import pandas as pd
import torch

DEFAULT_CONTINUOUS_COLS = ["level", "trend", "abs_trend", "trend_mid", "accel", "volatility", "support_log"]

def align_feature_table(table: torch.Tensor, num_items: int) -> torch.Tensor:
    table = table.float()
    target_n = int(num_items) + 1
    if table.size(0) > target_n:
        out = table[:target_n]
    elif table.size(0) < target_n:
        pad = torch.zeros(target_n - table.size(0), table.size(1), dtype=table.dtype)
        out = torch.cat([table, pad], dim=0)
    else:
        out = table
    out[0] = 0.0
    return out

def load_pt_feature_table(path: str | Path, num_items: int | None = None) -> torch.Tensor:
    path = Path(path)
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("dynamic_table", "role_table", "table"):
            if key in obj:
                obj = obj[key]
                break
    table = obj.float()
    if num_items is not None:
        table = align_feature_table(table, num_items)
    return table

def build_continuous_table_from_observations(
    observations_path: str | Path,
    num_items: int,
    feature_cols: Iterable[str] | None = None,
    standardize: bool = True,
    stats_path: str | Path | None = None,
    out_path: str | Path | None = None,
) -> torch.Tensor:
    observations_path = Path(observations_path)
    if not observations_path.exists():
        raise FileNotFoundError(f"Cannot find source dynamic observations: {observations_path}")
    df = pd.read_parquet(observations_path)
    if "ItemId" not in df.columns:
        raise ValueError(f"{observations_path} must contain ItemId column.")
    feature_cols = list(feature_cols or DEFAULT_CONTINUOUS_COLS)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{observations_path} misses columns: {missing}. available={list(df.columns)}")

    grouped = df.groupby("ItemId")[feature_cols].mean()
    global_mean = df[feature_cols].mean().to_numpy(dtype=np.float32)
    table = np.zeros((int(num_items) + 1, len(feature_cols)), dtype=np.float32)
    table[1:] = global_mean[None, :]
    for item_id, row in grouped.iterrows():
        item_id = int(item_id)
        if 1 <= item_id <= num_items:
            table[item_id] = row.to_numpy(dtype=np.float32)

    mean = table[1:].mean(axis=0).astype(np.float32)
    std = np.maximum(table[1:].std(axis=0).astype(np.float32), 1e-6)
    if standardize:
        table[1:] = (table[1:] - mean[None, :]) / std[None, :]
    table[0] = 0.0

    stats = {
        "feature_cols": feature_cols,
        "standardize": bool(standardize),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "observations_path": str(observations_path),
        "num_items": int(num_items),
        "num_observed_items": int(len(grouped)),
    }
    if stats_path is not None:
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    tensor = torch.from_numpy(table)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"dynamic_table": tensor, "stats": stats}, out_path)
    return tensor

def load_continuous_stats(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
