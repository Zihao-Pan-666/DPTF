from __future__ import annotations

import numpy as np


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    return float(np.mean(ranks < k))


def ndcg_at_k(ranks: np.ndarray, k: int) -> float:
    ok = ranks < k
    vals = np.zeros_like(ranks, dtype=np.float32)
    vals[ok] = 1.0 / np.log2(ranks[ok] + 2)
    return float(vals.mean())


def mrr_at_k(ranks: np.ndarray, k: int) -> float:
    ok = ranks < k
    vals = np.zeros_like(ranks, dtype=np.float32)
    vals[ok] = 1.0 / (ranks[ok] + 1)
    return float(vals.mean())
