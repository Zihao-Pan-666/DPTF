from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def find_processed_csv(data_root: str, domain: str) -> Path:
    root = Path(data_root)
    candidates = [
        root / "processed" / f"{domain}.csv",
        root / domain / "processed_data.csv",
        root / domain / f"{domain}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find processed csv for {domain}. Tried: {candidates}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"userid", "user_id", "user", "reviewerid"}:
            rename[c] = "UserId"
        elif lc in {"itemid", "item_id", "item", "asin"}:
            rename[c] = "ItemId"
        elif lc in {"timestamp", "time", "unixreviewtime"}:
            rename[c] = "Timestamp"
    df = df.rename(columns=rename)
    missing = {"UserId", "ItemId", "Timestamp"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}. Columns={list(df.columns)}")
    return df[["UserId", "ItemId", "Timestamp"]].copy()


def _build_steam_parquet_id_map(data: pd.DataFrame) -> Dict[Any, int]:
    unique_item_ids = data["ItemId"].unique()
    return {old_id: new_id for new_id, old_id in enumerate(unique_item_ids, start=0)}


def _build_amazon_parquet_id_map(data: pd.DataFrame) -> Dict[Any, int]:
    unique_item_ids = data["ItemId"].unique()
    item_id_map_stage1 = {old_id: new_id for new_id, old_id in enumerate(unique_item_ids, start=0)}
    temp_ids = data["ItemId"].map(item_id_map_stage1)
    unique_item_ids_v2 = temp_ids.unique()
    item_id_map_stage2 = {old_id: new_id for new_id, old_id in enumerate(unique_item_ids_v2, start=1)}
    return {old: item_id_map_stage2[item_id_map_stage1[old]] for old in item_id_map_stage1}


def build_raw_to_model_item_id_map(data: pd.DataFrame, domain: str) -> Dict[Any, int]:
    if "steam" in domain.lower():
        parquet_id_map = _build_steam_parquet_id_map(data)
        return {raw_id: parquet_id + 1 for raw_id, parquet_id in parquet_id_map.items()}
    return _build_amazon_parquet_id_map(data)


def load_remapped_dataframe(data_root: str, domain: str) -> Tuple[pd.DataFrame, Dict[Any, int]]:
    path = find_processed_csv(data_root, domain)
    df = normalize_columns(pd.read_csv(path))
    df["UserId"] = df["UserId"].astype(str)
    raw_to_model = build_raw_to_model_item_id_map(df, domain)
    df["RawItemId"] = df["ItemId"]
    df["ItemId"] = df["ItemId"].map(raw_to_model)
    if df["ItemId"].isna().any():
        bad = int(df["ItemId"].isna().sum())
        raise ValueError(f"{domain}: {bad} ItemId values cannot be mapped.")
    df["ItemId"] = df["ItemId"].astype(int)
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce").fillna(0).astype(float)
    return df.sort_values(["UserId", "Timestamp"]).reset_index(drop=True), raw_to_model


def build_history_only_interaction_dataframe(df: pd.DataFrame, min_sequence_len: int = 3) -> pd.DataFrame:
    """Return only the observed history part of each user sequence.

    The diagnostic SequenceDataset uses:
        history = items[:-2], val = items[-2], test = items[-1]

    Therefore, this function removes each user's last two interactions before
    computing target-domain popularity features. It is meant for strict
    history-only target popularity construction.
    """
    parts = []
    for _, g in df.sort_values(["UserId", "Timestamp"]).groupby("UserId", sort=False):
        if len(g) >= min_sequence_len:
            hist = g.iloc[:-2]
            if len(hist) > 0:
                parts.append(hist)
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, axis=0).reset_index(drop=True)


def _parse_embedding_cell(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def find_embedding_parquet(data_root: str, domain: str) -> Path:
    root = Path(data_root)
    candidates = [
        root / "semantic_embeddings" / f"{domain}_embedding_llama.parquet",
        root / "semantic_embeddings" / f"{domain}_embedding_llama3.parquet",
        root / domain / f"{domain}_embedding_llama.parquet",
        root / domain / f"{domain}_embedding_llama3.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find embedding parquet for {domain}. Tried: {candidates}")


def load_semantic_embeddings(data_root: str, domain: str) -> torch.Tensor:
    p = find_embedding_parquet(data_root, domain)
    df = pd.read_parquet(p).copy().sort_values("ItemId").reset_index(drop=True)
    if "item_text_embedding" not in df.columns:
        raise ValueError(f"{p} must contain item_text_embedding")
    item_ids = df["ItemId"].astype(int)
    min_id, max_id = int(item_ids.min()), int(item_ids.max())
    is_zero_based = min_id == 0
    emb_dim = len(_parse_embedding_cell(df["item_text_embedding"].iloc[0]))
    n_items = max_id + 1 if is_zero_based else max_id
    out = torch.zeros((n_items + 1, emb_dim), dtype=torch.float32)
    for _, row in df.iterrows():
        pid = int(row["ItemId"])
        mid = pid + 1 if is_zero_based else pid
        out[mid] = torch.tensor(_parse_embedding_cell(row["item_text_embedding"]), dtype=torch.float32)
    return out


class SequenceDataset(Dataset):
    def __init__(self, data_root: str, domain: str, max_len: int = 50):
        self.domain = domain
        self.max_len = max_len
        self.df, self.raw_to_model = load_remapped_dataframe(data_root, domain)
        self.num_items = int(self.df["ItemId"].max())
        self.user_sequences: List[List[int]] = []
        self.user_times: List[List[float]] = []
        for _, g in self.df.groupby("UserId", sort=False):
            items = g["ItemId"].astype(int).tolist()
            times = g["Timestamp"].astype(float).tolist()
            if len(items) >= 3:
                self.user_sequences.append(items)
                self.user_times.append(times)

    def __len__(self) -> int:
        return len(self.user_sequences)

    def __getitem__(self, idx: int):
        items = self.user_sequences[idx]
        times = self.user_times[idx]
        hist, hist_t = items[:-2], times[:-2]
        val, test = items[-2], items[-1]
        seq = np.zeros(self.max_len, dtype=np.int64)
        rel_time = np.zeros(self.max_len, dtype=np.int64)
        keep = min(len(hist), self.max_len)
        if keep > 0:
            sub_items = hist[-keep:]
            sub_times = np.asarray(hist_t[-keep:], dtype=np.float64)
            seq[-keep:] = np.asarray(sub_items, dtype=np.int64)
            if keep > 1:
                gaps = np.diff(sub_times, prepend=sub_times[0])
                ranks = pd.Series(gaps).rank(method="first", pct=True).values
                rel_time[-keep:] = np.clip((ranks * 9).astype(np.int64), 0, 9)
        return torch.tensor(seq), torch.tensor(rel_time), torch.tensor(val), torch.tensor(test)

    def get_num_items(self) -> int:
        return self.num_items
