from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .data_utils import build_history_only_interaction_dataframe, load_remapped_dataframe


def _rank_percentile(values: pd.Series) -> pd.Series:
    if values.empty:
        return values
    return values.rank(method="average", pct=True).fillna(0.0)


def _features_from_interactions(df_stats: pd.DataFrame, n_items: int) -> torch.Tensor:
    """Build lightweight PrepRec-core features from a chosen interaction subset.

    df_stats determines which interactions are allowed to contribute statistics.
    n_items should come from the full domain item vocabulary so the resulting
    feature table remains index-compatible with SequenceDataset.
    """
    if n_items <= 0:
        raise ValueError("n_items must be positive.")

    if df_stats.empty:
        return torch.zeros((n_items + 1, 8), dtype=torch.float32)

    df = df_stats.copy()
    t = df["Timestamp"].astype(float)
    if float(t.max()) == float(t.min()):
        df["time_norm"] = 0.0
    else:
        df["time_norm"] = (t - t.min()) / (t.max() - t.min())

    early = df[df["time_norm"] <= 0.5]
    recent = df[df["time_norm"] > 0.5]

    item_index = range(1, n_items + 1)
    total_counts = df.groupby("ItemId").size().reindex(item_index, fill_value=0)
    early_counts = early.groupby("ItemId").size().reindex(item_index, fill_value=0)
    recent_counts = recent.groupby("ItemId").size().reindex(item_index, fill_value=0)

    total_pct = _rank_percentile(total_counts)
    early_pct = _rank_percentile(early_counts)
    recent_pct = _rank_percentile(recent_counts)
    trend = recent_pct - early_pct
    log_count = np.log1p(total_counts) / max(float(np.log1p(total_counts.max())), 1.0)

    stats = df.groupby("ItemId")["time_norm"].agg(["mean", "std", "min", "max"]).reindex(item_index)
    mean_t = stats["mean"].fillna(0.0)
    std_t = stats["std"].fillna(0.0)
    span_t = (stats["max"] - stats["min"]).fillna(0.0)

    feat = np.stack([
        total_pct.values,
        early_pct.values,
        recent_pct.values,
        trend.values,
        np.asarray(log_count, dtype=np.float32),
        mean_t.values,
        std_t.values,
        span_t.values,
    ], axis=1).astype(np.float32)

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    out = torch.zeros((n_items + 1, feat.shape[1]), dtype=torch.float32)
    out[1:] = torch.tensor(feat, dtype=torch.float32)
    return out


def build_popularity_features(
    data_root: str,
    domain: str,
    force: bool = False,
    mode: str = "full",
) -> torch.Tensor:
    """Build a lightweight PrepRec-core item feature table.

    mode:
      - full:        uses the full domain interaction CSV. This matches the
                     previous diagnostic code and is NOT strict for target use.
      - history_only: uses only each user's observed history items[:-2]. This is
                      the recommended strict diagnostic setting for target use.
      - source_only: does not use the target domain interactions at all. Since
                     target item ids are not shared with the source domain, this
                     function returns all-zero target popularity features. For a
                     source domain, use mode='full' during training.
    """
    mode = mode.lower().strip()
    if mode not in {"full", "history_only", "source_only"}:
        raise ValueError(f"Unknown popularity mode: {mode}")

    out_dir = Path(data_root) / "popularity_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{domain}_popularity_{mode}.pt"
    if out_path.exists() and not force:
        return torch.load(out_path, map_location="cpu")

    full_df, _ = load_remapped_dataframe(data_root, domain)
    n_items = int(full_df["ItemId"].max())

    if mode == "full":
        stat_df = full_df
    elif mode == "history_only":
        stat_df = build_history_only_interaction_dataframe(full_df)
    else:
        stat_df = full_df.iloc[0:0].copy()

    out = _features_from_interactions(stat_df, n_items)
    torch.save(out, out_path)
    return out


def load_or_build_popularity_features(
    data_root: str,
    domain: str,
    force: bool = False,
    mode: str = "full",
) -> torch.Tensor:
    return build_popularity_features(data_root, domain, force=force, mode=mode)


def build_source_only_target_features(
    n_target_items: int,
    source_features: Optional[torch.Tensor] = None,
    strategy: str = "zeros",
) -> torch.Tensor:
    """Create a no-target-interaction feature table for source-only testing.

    This is a lower-bound sanity check rather than a competitive PrepRec setting.
    With no shared item ids and no target interaction statistics, target item
    popularity features are not identifiable. The default is zeros.

    strategy:
      - zeros: every target item has zero dynamics features.
      - source_mean: every target item gets the non-padding mean source feature.
    """
    strategy = strategy.lower().strip()
    if strategy not in {"zeros", "source_mean"}:
        raise ValueError(f"Unknown source_only strategy: {strategy}")

    feat_dim = 8
    if source_features is not None and source_features.ndim == 2:
        feat_dim = int(source_features.size(1))
    out = torch.zeros((n_target_items + 1, feat_dim), dtype=torch.float32)

    if strategy == "source_mean" and source_features is not None and source_features.size(0) > 1:
        mean_vec = source_features[1:].mean(dim=0)
        out[1:] = mean_vec.unsqueeze(0).expand(n_target_items, -1).clone()
    return out
