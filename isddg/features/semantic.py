# isddg/features/semantic.py strict replacement

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import ast
import numpy as np
import pandas as pd
import torch


def _parse_embedding_cell(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def find_embedding_parquet(data_root: str | Path, domain: str, embedding_dir: str = "semantic_embeddings") -> Path:
    root = Path(data_root)
    candidates = [
        root / embedding_dir / f"{domain}_embedding_llama_fixed.parquet",
        root / embedding_dir / f"{domain}_embedding_llama.parquet",
        root / embedding_dir / f"{domain}_embedding_llama3.parquet",
        root / "semantic_embeddings_fixed" / f"{domain}_embedding_llama_fixed.parquet",
        root / "semantic_embeddings" / f"{domain}_embedding_llama_fixed.parquet",
        root / "semantic_embeddings" / f"{domain}_embedding_llama.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find embedding parquet for {domain}. Tried={candidates}")


def load_semantic_embeddings(
    data_root: str | Path,
    domain: str,
    item_map: Dict[str, int],
    vector_col: str = "item_text_embedding",
    embedding_dir: str = "semantic_embeddings",
    strict: bool = True,
) -> torch.Tensor:
    if item_map is None:
        raise ValueError("Formal ISDDG experiments require item_map; row-order loading is forbidden.")

    path = find_embedding_parquet(data_root, domain, embedding_dir=embedding_dir)
    df = pd.read_parquet(path)

    id_candidates = [
        c for c in df.columns
        if c.lower() in {"rawitemid", "raw_item_id", "itemid", "item_id", "asin", "product_id"}
    ]
    if not id_candidates:
        raise ValueError(f"{path} has no item id column. Row-order alignment is forbidden.")
    id_col = "RawItemId" if "RawItemId" in df.columns else id_candidates[0]

    if vector_col not in df.columns:
        cands = [c for c in df.columns if "embedding" in c.lower()]
        if not cands:
            raise ValueError(f"No embedding column in {path}; columns={list(df.columns)}")
        vector_col = cands[0]

    raw_ids = df[id_col].astype(str).tolist()
    vectors = [_parse_embedding_cell(x) for x in df[vector_col].tolist()]
    dims = [int(v.shape[0]) for v in vectors]
    if len(set(dims)) != 1:
        raise ValueError(f"Inconsistent embedding dims in {path}: {sorted(set(dims))}")
    dim = dims[0]

    emb_by_raw = {}
    dups = []
    for raw, vec in zip(raw_ids, vectors):
        if raw in emb_by_raw:
            dups.append(raw)
        emb_by_raw[raw] = np.asarray(vec, dtype=np.float32)
    if dups:
        raise ValueError(f"{path} has duplicated item ids: {dups[:20]}")

    table = np.zeros((max(item_map.values()) + 1, dim), dtype=np.float32)
    missing, nonfinite, zero = [], [], []

    for raw, idx in item_map.items():
        raw = str(raw)
        if raw not in emb_by_raw:
            missing.append(raw)
            continue
        vec = emb_by_raw[raw]
        if not np.isfinite(vec).all():
            nonfinite.append(raw)
            if strict:
                continue
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.linalg.norm(vec)) == 0.0:
            zero.append(raw)
        table[idx] = vec

    if missing:
        raise ValueError(f"{domain}: {len(missing)} current items miss embeddings. samples={missing[:20]}")
    if strict and nonfinite:
        raise ValueError(f"{domain}: {len(nonfinite)} embeddings contain NaN/Inf. samples={nonfinite[:20]}")
    if strict and zero:
        raise ValueError(f"{domain}: {len(zero)} embeddings are zero vectors. samples={zero[:20]}")

    return torch.from_numpy(table)
