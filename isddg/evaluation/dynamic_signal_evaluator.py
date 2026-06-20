from __future__ import annotations

from typing import Any, Dict, List, Sequence
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from isddg.evaluation.metrics import recall_at_k, ndcg_at_k, mrr_at_k


def parse_beta_grid(x) -> List[float]:
    if x is None:
        return [0.0]
    if isinstance(x, str):
        return [float(v.strip()) for v in x.split(",") if v.strip()]
    return [float(v) for v in x]


def _sample_unique_negatives(num_items: int, forbidden: set[int], n: int, rng: random.Random) -> List[int]:
    out: List[int] = []
    used = set(forbidden)
    target_n = min(int(n), max(int(num_items) - len(used), 0))
    tries = 0
    max_tries = max(1000, int(n) * 100)
    while len(out) < target_n and tries < max_tries:
        x = rng.randint(1, int(num_items))
        if x not in used:
            out.append(x)
            used.add(x)
        tries += 1
    if len(out) < target_n:
        for x in range(1, int(num_items) + 1):
            if x not in used:
                out.append(x)
                used.add(x)
                if len(out) >= target_n:
                    break
    return out


def _rank_from_scores(scores: np.ndarray, gt_pos: int, atol: float = 1e-8, tie_policy: str = "worst"):
    gt_score = float(scores[int(gt_pos)])
    greater = int(np.sum(scores > gt_score + atol))
    tied = int(np.sum(np.isclose(scores, gt_score, atol=atol, rtol=0.0)))
    all_equal = bool(np.allclose(scores, scores[0], atol=atol, rtol=0.0))
    if tie_policy == "worst":
        rank = greater + max(tied - 1, 0)
    elif tie_policy in {"strict", "best"}:
        rank = greater
    elif tie_policy == "average":
        rank = greater + (max(tied - 1, 0) / 2.0)
    else:
        raise ValueError(f"Unknown tie_policy={tie_policy}. Use worst, strict, best, or average.")
    return rank, tied, all_equal


def _metrics_from_ranks(ranks: Sequence[float], ks: Sequence[int]) -> Dict[str, float]:
    ranks_np = np.asarray(ranks, dtype=np.float64)
    out: Dict[str, float] = {}
    for k in ks:
        out[f"Recall@{k}"] = recall_at_k(ranks_np, k)
    for k in ks:
        out[f"NDCG@{k}"] = ndcg_at_k(ranks_np, k)
    for k in ks:
        out[f"MRR@{k}"] = mrr_at_k(ranks_np, k)
    return out


def _prefix_stats(prefix: str, x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float()
    if x.numel() == 0:
        return {}
    out: Dict[str, float] = {
        f"{prefix}_mean": float(x.mean().item()),
        f"{prefix}_std": float(x.std(unbiased=False).item()),
        f"{prefix}_min": float(x.min().item()),
        f"{prefix}_max": float(x.max().item()),
    }
    if x.ndim >= 2:
        row_std = x.std(dim=1, unbiased=False)
        out[f"{prefix}_row_std_mean"] = float(row_std.mean().item())
        out[f"{prefix}_row_std_min"] = float(row_std.min().item())
        out[f"{prefix}_row_std_max"] = float(row_std.max().item())
    return out


def history_state(
    history: torch.Tensor,
    feature_table: torch.Tensor,
    pooling: str = "decay",
    recent_k: int | None = 5,
    decay: float = 0.8,
) -> torch.Tensor:
    feature_table = feature_table.to(history.device)
    x = feature_table[history]
    valid = history.ne(0)
    batch_size, seq_len = history.shape
    lengths = valid.sum(dim=1).clamp(min=1)

    if pooling == "last":
        idx = (lengths - 1).view(batch_size, 1, 1).expand(batch_size, 1, x.size(-1))
        return x.gather(dim=1, index=idx).squeeze(1)

    if pooling == "mean":
        weights = valid.float()
    elif pooling == "recent":
        k = seq_len if recent_k is None or recent_k <= 0 else int(recent_k)
        rank = valid.long().cumsum(dim=1)
        keep = valid & (rank > (lengths - k).unsqueeze(1))
        weights = keep.float()
    elif pooling == "decay":
        rank = valid.long().cumsum(dim=1)
        dist = (lengths.unsqueeze(1) - rank).clamp(min=0)
        weights = (float(decay) ** dist.float()) * valid.float()
        if recent_k is not None and recent_k > 0:
            keep = valid & (rank > (lengths - int(recent_k)).unsqueeze(1))
            weights = weights * keep.float()
    else:
        raise ValueError(f"Unknown pooling={pooling}. Use mean, recent, decay, or last.")

    denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return (x * weights.unsqueeze(-1)).sum(dim=1) / denom


def normalize_scores(scores: torch.Tensor, mode: str | None = "zscore") -> torch.Tensor:
    """Normalize scores inside each candidate set.

    This is intentionally row-wise: each user/query has its own candidate set.
    Row-wise affine normalization does not change semantic-only ranks, but it
    prevents a z-scored dynamic score from dominating an unnormalized semantic
    score when beta is small.
    """
    if mode is None or mode == "none":
        return scores
    if mode == "center":
        return scores - scores.mean(dim=1, keepdim=True)
    if mode == "zscore":
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        return (scores - mean) / std
    raise ValueError(f"Unknown score norm mode={mode}. Use none, center, or zscore.")


@torch.no_grad()
def dynamic_scores(
    feature_table: torch.Tensor,
    history: torch.Tensor,
    candidates: torch.Tensor,
    pooling: str = "decay",
    recent_k: int | None = 5,
    decay: float = 0.8,
    score_norm: str | None = "zscore",
    return_raw: bool = False,
):
    feature_table = feature_table.to(history.device)
    user_state = history_state(history, feature_table, pooling=pooling, recent_k=recent_k, decay=decay)
    cand = feature_table[candidates]
    raw = torch.einsum("bd,bnd->bn", user_state, cand)
    normed = normalize_scores(raw, mode=score_norm)
    if return_raw:
        return normed, raw
    return normed


@torch.no_grad()
def combined_scores(
    model,
    feature_table: torch.Tensor,
    history: torch.Tensor,
    candidates: torch.Tensor,
    beta: float = 0.0,
    pooling: str = "decay",
    recent_k: int | None = 5,
    decay: float = 0.8,
    score_norm: str | None = "zscore",
    semantic_score_norm: str | None = None,
    dynamic_score_norm: str | None = None,
    semantic_weight: float = 1.0,
    return_components: bool = False,
):
    """Return fused scores.

    Backward-compatible behavior:
    - ``score_norm`` remains the default normalization mode.
    - If ``semantic_score_norm`` / ``dynamic_score_norm`` are omitted, both use
      ``score_norm``.

    Important fix compared with the old version:
    semantic and dynamic candidate scores are normalized with the same row-wise
    policy before fusion. The old implementation z-scored only the dynamic
    scores and left semantic scores raw, so even beta=0.01 could dominate.
    """
    if semantic_score_norm is None:
        semantic_score_norm = score_norm
    if dynamic_score_norm is None:
        dynamic_score_norm = score_norm

    comps: Dict[str, torch.Tensor | None] = {
        "semantic_raw": None,
        "semantic": None,
        "dynamic_raw": None,
        "dynamic": None,
    }

    if model is None or float(semantic_weight) == 0.0:
        dyn, dyn_raw = dynamic_scores(
            feature_table, history, candidates,
            pooling=pooling, recent_k=recent_k, decay=decay,
            score_norm=dynamic_score_norm, return_raw=True,
        )
        comps["dynamic_raw"] = dyn_raw
        comps["dynamic"] = dyn
        if return_components:
            return dyn, comps
        return dyn

    sem_raw = float(semantic_weight) * model.score(history, candidates)
    sem = normalize_scores(sem_raw, mode=semantic_score_norm)
    comps["semantic_raw"] = sem_raw
    comps["semantic"] = sem

    if float(beta) == 0.0 and not return_components:
        return sem

    dyn, dyn_raw = dynamic_scores(
        feature_table, history, candidates,
        pooling=pooling, recent_k=recent_k, decay=decay,
        score_norm=dynamic_score_norm, return_raw=True,
    )
    comps["dynamic_raw"] = dyn_raw
    comps["dynamic"] = dyn

    fused = sem + float(beta) * dyn
    if return_components:
        return fused, comps
    return fused


@torch.no_grad()
def evaluate_dynamic_signal(
    model,
    feature_table: torch.Tensor,
    loader: DataLoader,
    num_items: int,
    device: torch.device,
    beta: float = 0.0,
    semantic_weight: float = 1.0,
    ranking_mode: str = "sampled",
    num_negatives: int = 100,
    seed: int = 2026,
    ks: Sequence[int] = (10, 20),
    pooling: str = "decay",
    recent_k: int | None = 5,
    decay: float = 0.8,
    score_norm: str | None = "zscore",
    semantic_score_norm: str | None = None,
    dynamic_score_norm: str | None = None,
    desc: str | None = None,
    tie_policy: str = "worst",
    show_progress: bool = False,
    collect_score_stats: bool = False,
) -> Dict[str, Any]:
    if ranking_mode not in {"sampled", "full"}:
        raise ValueError(f"ranking_mode must be sampled or full, got {ranking_mode}")
    if model is not None:
        model.eval()
    feature_table = feature_table.float().to(device)
    rng = random.Random(seed)
    start = time.time()
    ranks: List[float] = []
    total_tie = tie_cases = all_equal_cases = total = 0
    first_score_stats: Dict[str, float] = {}

    eval_iter = loader
    if show_progress:
        eval_iter = tqdm(
            loader,
            desc=desc,
            leave=False,
            ascii=True,
            dynamic_ncols=True,
            mininterval=5.0,
            maxinterval=30.0,
        )

    for batch in eval_iter:
        hist = batch["history"].to(device)
        pos = batch["target"].to(device)
        batch_size = hist.size(0)

        if ranking_mode == "sampled":
            cand_rows: List[List[int]] = []
            gt_positions: List[int] = []
            hist_cpu = hist.detach().cpu().tolist()
            pos_cpu = pos.detach().cpu().tolist()
            for i, p in enumerate(pos_cpu):
                forbidden = {int(x) for x in hist_cpu[i] if int(x) != 0}
                forbidden.add(int(p))
                negs = _sample_unique_negatives(num_items, forbidden, num_negatives, rng)
                cands = [int(p)] + negs
                rng.shuffle(cands)
                gt_positions.append(cands.index(int(p)))
                cand_rows.append(cands)

            candidates = torch.tensor(cand_rows, dtype=torch.long, device=device)

            if collect_score_stats and not first_score_stats:
                score_t, comps = combined_scores(
                    model, feature_table, hist, candidates,
                    beta=beta, pooling=pooling, recent_k=recent_k, decay=decay,
                    score_norm=score_norm,
                    semantic_score_norm=semantic_score_norm,
                    dynamic_score_norm=dynamic_score_norm,
                    semantic_weight=semantic_weight,
                    return_components=True,
                )
                for key, value in comps.items():
                    if value is not None:
                        first_score_stats.update(_prefix_stats(key, value))
                first_score_stats.update(_prefix_stats("fused", score_t))
                scores = score_t.detach().cpu().numpy()
            else:
                scores = combined_scores(
                    model, feature_table, hist, candidates,
                    beta=beta, pooling=pooling, recent_k=recent_k, decay=decay,
                    score_norm=score_norm,
                    semantic_score_norm=semantic_score_norm,
                    dynamic_score_norm=dynamic_score_norm,
                    semantic_weight=semantic_weight,
                ).detach().cpu().numpy()

            for row, gt_pos in zip(scores, gt_positions):
                rank, tied, all_eq = _rank_from_scores(row, gt_pos, tie_policy=tie_policy)
                ranks.append(rank)
                total += 1
                total_tie += tied
                tie_cases += int(tied > 1)
                all_equal_cases += int(all_eq)
        else:
            candidates = torch.arange(0, num_items + 1, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)

            if collect_score_stats and not first_score_stats:
                score_t, comps = combined_scores(
                    model, feature_table, hist, candidates,
                    beta=beta, pooling=pooling, recent_k=recent_k, decay=decay,
                    score_norm=score_norm,
                    semantic_score_norm=semantic_score_norm,
                    dynamic_score_norm=dynamic_score_norm,
                    semantic_weight=semantic_weight,
                    return_components=True,
                )
                for key, value in comps.items():
                    if value is not None:
                        first_score_stats.update(_prefix_stats(key, value))
                first_score_stats.update(_prefix_stats("fused", score_t))
                scores = score_t
            else:
                scores = combined_scores(
                    model, feature_table, hist, candidates,
                    beta=beta, pooling=pooling, recent_k=recent_k, decay=decay,
                    score_norm=score_norm,
                    semantic_score_norm=semantic_score_norm,
                    dynamic_score_norm=dynamic_score_norm,
                    semantic_weight=semantic_weight,
                )

            scores[:, 0] = -torch.inf
            for i in range(batch_size):
                target_i = int(pos[i].item())
                for item in hist[i].detach().cpu().tolist():
                    item = int(item)
                    if item != 0 and item != target_i:
                        scores[i, item] = -torch.inf
            scores_np = scores.detach().cpu().numpy()
            for row, target_i in zip(scores_np, pos.detach().cpu().tolist()):
                rank, tied, all_eq = _rank_from_scores(row, int(target_i), tie_policy=tie_policy)
                ranks.append(rank)
                total += 1
                total_tie += tied
                tie_cases += int(tied > 1)
                all_equal_cases += int(all_eq)

    elapsed = time.time() - start
    out: Dict[str, Any] = _metrics_from_ranks(ranks, ks)
    out.update({
        "tie_case_ratio": float(tie_cases / max(total, 1)),
        "all_equal_ratio": float(all_equal_cases / max(total, 1)),
        "avg_tie_items": float(total_tie / max(total, 1)),
        "num_eval_users": int(total),
        "tie_policy": tie_policy,
        "beta": float(beta),
        "semantic_weight": float(semantic_weight),
        "pooling": pooling,
        "recent_k": int(recent_k) if recent_k is not None else "",
        "decay": float(decay),
        "score_norm": score_norm if score_norm is not None else "none",
        "semantic_score_norm": semantic_score_norm if semantic_score_norm is not None else (score_norm or "none"),
        "dynamic_score_norm": dynamic_score_norm if dynamic_score_norm is not None else (score_norm or "none"),
        "eval_elapsed_sec": float(elapsed),
        "eval_users_per_sec": float(total / max(elapsed, 1e-9)),
    })
    if collect_score_stats:
        out.update(first_score_stats)
    return out
