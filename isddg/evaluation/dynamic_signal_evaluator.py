from __future__ import annotations
from typing import Dict, List, Sequence
import random, time
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
    out, used = [], set(forbidden)
    target_n = min(n, max(num_items - len(used), 0))
    tries = 0
    while len(out) < target_n and tries < max(1000, n * 100):
        x = rng.randint(1, num_items)
        if x not in used:
            out.append(x); used.add(x)
        tries += 1
    if len(out) < target_n:
        for x in range(1, num_items + 1):
            if x not in used:
                out.append(x); used.add(x)
                if len(out) >= target_n:
                    break
    return out

def _rank_from_scores(scores: np.ndarray, gt_pos: int, atol: float = 1e-8):
    gt_score = float(scores[gt_pos])
    greater = int(np.sum(scores > gt_score + atol))
    tied = int(np.sum(np.isclose(scores, gt_score, atol=atol, rtol=0.0)))
    all_equal = bool(np.allclose(scores, scores[0], atol=atol, rtol=0.0))
    return int(greater + max(tied - 1, 0)), tied, all_equal

def _metrics_from_ranks(ranks: Sequence[int], ks: Sequence[int]) -> Dict[str, float]:
    ranks_np = np.asarray(ranks, dtype=np.int64)
    out = {}
    for k in ks:
        out[f"Recall@{k}"] = recall_at_k(ranks_np, k)
    for k in ks:
        out[f"NDCG@{k}"] = ndcg_at_k(ranks_np, k)
    for k in ks:
        out[f"MRR@{k}"] = mrr_at_k(ranks_np, k)
    return out

def history_state(history: torch.Tensor, feature_table: torch.Tensor, pooling: str = "decay", recent_k: int | None = 5, decay: float = 0.8) -> torch.Tensor:
    x = feature_table[history]
    valid = history.ne(0)
    B, L = history.shape
    lengths = valid.sum(dim=1).clamp(min=1)

    if pooling == "last":
        idx = (lengths - 1).view(B, 1, 1).expand(B, 1, x.size(-1))
        return x.gather(dim=1, index=idx).squeeze(1)

    if pooling == "mean":
        weights = valid.float()
    elif pooling == "recent":
        k = L if recent_k is None or recent_k <= 0 else int(recent_k)
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
        raise ValueError(f"Unknown pooling={pooling}")

    denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return (x * weights.unsqueeze(-1)).sum(dim=1) / denom

def normalize_scores(scores: torch.Tensor, mode: str = "zscore") -> torch.Tensor:
    if mode is None or mode == "none":
        return scores
    if mode == "center":
        return scores - scores.mean(dim=1, keepdim=True)
    if mode == "zscore":
        mean = scores.mean(dim=1, keepdim=True)
        std = scores.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        return (scores - mean) / std
    raise ValueError(f"Unknown score norm mode={mode}")

@torch.no_grad()
def dynamic_scores(feature_table: torch.Tensor, history: torch.Tensor, candidates: torch.Tensor, pooling: str = "decay", recent_k: int | None = 5, decay: float = 0.8, score_norm: str = "zscore") -> torch.Tensor:
    feature_table = feature_table.to(history.device)
    user_state = history_state(history, feature_table, pooling=pooling, recent_k=recent_k, decay=decay)
    cand = feature_table[candidates]
    scores = torch.einsum("bd,bnd->bn", user_state, cand)
    return normalize_scores(scores, mode=score_norm)

@torch.no_grad()
def combined_scores(model, feature_table: torch.Tensor, history: torch.Tensor, candidates: torch.Tensor, beta: float = 0.0, pooling: str = "decay", recent_k: int | None = 5, decay: float = 0.8, score_norm: str = "zscore", semantic_weight: float = 1.0) -> torch.Tensor:
    dyn = dynamic_scores(feature_table, history, candidates, pooling=pooling, recent_k=recent_k, decay=decay, score_norm=score_norm)
    if model is None or float(semantic_weight) == 0.0:
        return dyn
    sem = model.score(history, candidates)
    if float(beta) == 0.0:
        return sem
    return float(semantic_weight) * sem + float(beta) * dyn

@torch.no_grad()
def evaluate_dynamic_signal(model, feature_table: torch.Tensor, loader: DataLoader, num_items: int, device: torch.device, beta: float = 0.0, semantic_weight: float = 1.0, ranking_mode: str = "sampled", num_negatives: int = 100, seed: int = 2026, ks: Sequence[int] = (10, 20), pooling: str = "decay", recent_k: int | None = 5, decay: float = 0.8, score_norm: str = "zscore", desc: str = "dynamic-signal-eval") -> Dict[str, float]:
    if ranking_mode not in {"sampled", "full"}:
        raise ValueError(f"ranking_mode must be sampled or full, got {ranking_mode}")
    if model is not None:
        model.eval()
    feature_table = feature_table.float().to(device)
    rng = random.Random(seed)
    start = time.time()
    ranks, total_tie, tie_cases, all_equal_cases, total = [], 0, 0, 0, 0

    for batch in tqdm(loader, desc=desc, leave=False):
        hist = batch["history"].to(device)
        pos = batch["target"].to(device)
        B = hist.size(0)

        if ranking_mode == "sampled":
            cand_rows, gt_positions = [], []
            hcpu, pcpu = hist.detach().cpu().tolist(), pos.detach().cpu().tolist()
            for i, p in enumerate(pcpu):
                forbidden = {int(x) for x in hcpu[i] if int(x) != 0}
                forbidden.add(int(p))
                negs = _sample_unique_negatives(num_items, forbidden, num_negatives, rng)
                cands = [int(p)] + negs
                rng.shuffle(cands)
                gt_positions.append(cands.index(int(p)))
                cand_rows.append(cands)
            candidates = torch.tensor(cand_rows, dtype=torch.long, device=device)
            scores = combined_scores(model, feature_table, hist, candidates, beta=beta, pooling=pooling, recent_k=recent_k, decay=decay, score_norm=score_norm, semantic_weight=semantic_weight).detach().cpu().numpy()
            for row, gt in zip(scores, gt_positions):
                r, tied, all_eq = _rank_from_scores(row, gt)
                ranks.append(r); total += 1; total_tie += tied; tie_cases += int(tied > 1); all_equal_cases += int(all_eq)
        else:
            candidates = torch.arange(0, num_items + 1, dtype=torch.long, device=device).view(1, -1).expand(B, -1)
            scores = combined_scores(model, feature_table, hist, candidates, beta=beta, pooling=pooling, recent_k=recent_k, decay=decay, score_norm=score_norm, semantic_weight=semantic_weight)
            scores[:, 0] = -torch.inf
            for i in range(B):
                target_i = int(pos[i].item())
                for item in hist[i].detach().cpu().tolist():
                    item = int(item)
                    if item != 0 and item != target_i:
                        scores[i, item] = -torch.inf
            scores_np = scores.detach().cpu().numpy()
            for row, target_i in zip(scores_np, pos.detach().cpu().tolist()):
                r, tied, all_eq = _rank_from_scores(row, int(target_i))
                ranks.append(r); total += 1; total_tie += tied; tie_cases += int(tied > 1); all_equal_cases += int(all_eq)

    elapsed = time.time() - start
    out = _metrics_from_ranks(ranks, ks)
    out.update({
        "tie_case_ratio": float(tie_cases / max(total, 1)),
        "all_equal_ratio": float(all_equal_cases / max(total, 1)),
        "avg_tie_items": float(total_tie / max(total, 1)),
        "num_eval_users": int(total),
        "tie_policy": "worst",
        "beta": float(beta),
        "semantic_weight": float(semantic_weight),
        "pooling": pooling,
        "recent_k": int(recent_k) if recent_k is not None else "",
        "decay": float(decay),
        "score_norm": score_norm,
        "eval_elapsed_sec": float(elapsed),
        "eval_users_per_sec": float(total / max(elapsed, 1e-9)),
    })
    return out
