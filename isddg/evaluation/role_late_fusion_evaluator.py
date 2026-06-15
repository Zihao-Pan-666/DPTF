from __future__ import annotations

from typing import Dict, List, Sequence
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
    target_n = min(n, max(num_items - len(used), 0))
    tries = 0
    max_tries = max(1000, n * 100)
    while len(out) < target_n and tries < max_tries:
        x = rng.randint(1, num_items)
        if x not in used:
            out.append(x)
            used.add(x)
        tries += 1
    if len(out) < target_n:
        for x in range(1, num_items + 1):
            if x not in used:
                out.append(x)
                used.add(x)
                if len(out) >= target_n:
                    break
    return out


def _rank_from_scores(scores: np.ndarray, gt_pos: int, atol: float = 1e-8):
    gt_score = float(scores[gt_pos])
    greater = int(np.sum(scores > gt_score + atol))
    tied_mask = np.isclose(scores, gt_score, atol=atol, rtol=0.0)
    tie_count = int(np.sum(tied_mask))
    all_equal = bool(np.allclose(scores, scores[0], atol=atol, rtol=0.0))
    rank = greater + max(tie_count - 1, 0)
    return int(rank), tie_count, all_equal


def _metrics_from_ranks(ranks: Sequence[int], ks: Sequence[int]) -> Dict[str, float]:
    ranks_np = np.asarray(ranks, dtype=np.int64)
    out: Dict[str, float] = {}
    for k in ks:
        out[f"Recall@{k}"] = recall_at_k(ranks_np, k)
    for k in ks:
        out[f"NDCG@{k}"] = ndcg_at_k(ranks_np, k)
    for k in ks:
        out[f"MRR@{k}"] = mrr_at_k(ranks_np, k)
    return out


def _history_role_state(
    history: torch.Tensor,
    role_table: torch.Tensor,
    pooling: str = "decay",
    recent_k: int | None = 10,
    decay: float = 0.8,
) -> torch.Tensor:
    roles = role_table[history]
    valid = history.ne(0)
    B, L = history.shape

    if pooling == "last":
        lengths = valid.sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(B, 1, 1).expand(B, 1, roles.size(-1))
        return roles.gather(dim=1, index=idx).squeeze(1)

    lengths = valid.sum(dim=1).clamp(min=1)

    if pooling == "recent":
        if recent_k is None or recent_k <= 0:
            recent_k = L
        rank = valid.long().cumsum(dim=1)
        keep = valid & (rank > (lengths - int(recent_k)).unsqueeze(1))
        weights = keep.float()
    elif pooling == "decay":
        rank = valid.long().cumsum(dim=1)
        dist = (lengths.unsqueeze(1) - rank).clamp(min=0)
        weights = (float(decay) ** dist.float()) * valid.float()
        if recent_k is not None and recent_k > 0:
            keep = valid & (rank > (lengths - int(recent_k)).unsqueeze(1))
            weights = weights * keep.float()
    elif pooling == "mean":
        weights = valid.float()
    else:
        raise ValueError(f"Unknown pooling={pooling}. Use mean, recent, decay, or last.")

    denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return (roles * weights.unsqueeze(-1)).sum(dim=1) / denom


def _normalize_role_scores(role_scores: torch.Tensor, mode: str = "zscore") -> torch.Tensor:
    if mode == "none" or mode is None:
        return role_scores
    if mode == "center":
        return role_scores - role_scores.mean(dim=1, keepdim=True)
    if mode == "zscore":
        mean = role_scores.mean(dim=1, keepdim=True)
        std = role_scores.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        return (role_scores - mean) / std
    raise ValueError(f"Unknown role_score_norm={mode}. Use none, center, or zscore.")


@torch.no_grad()
def score_late_fusion(
    model,
    role_table: torch.Tensor,
    history: torch.Tensor,
    candidates: torch.Tensor,
    beta: float = 0.0,
    pooling: str = "decay",
    recent_k: int | None = 10,
    decay: float = 0.8,
    role_score_norm: str = "zscore",
):
    semantic_scores = model.score(history, candidates)
    if float(beta) == 0.0:
        return semantic_scores

    role_table = role_table.to(history.device)
    user_role = _history_role_state(
        history=history,
        role_table=role_table,
        pooling=pooling,
        recent_k=recent_k,
        decay=decay,
    )
    cand_role = role_table[candidates]
    role_scores = torch.einsum("bk,bnk->bn", user_role, cand_role)
    role_scores = _normalize_role_scores(role_scores, role_score_norm)
    return semantic_scores + float(beta) * role_scores


@torch.no_grad()
def evaluate_role_late_fusion(
    model,
    role_table: torch.Tensor,
    loader: DataLoader,
    num_items: int,
    device: torch.device,
    beta: float = 0.0,
    ranking_mode: str = "sampled",
    num_negatives: int = 100,
    seed: int = 2026,
    ks: Sequence[int] = (10, 20),
    pooling: str = "decay",
    recent_k: int | None = 10,
    decay: float = 0.8,
    role_score_norm: str = "zscore",
    desc: str = "role-late-fusion-eval",
) -> Dict[str, float]:
    if ranking_mode not in {"sampled", "full"}:
        raise ValueError(f"ranking_mode must be sampled or full, got {ranking_mode}")

    start = time.time()
    model.eval()
    role_table = role_table.to(device)
    rng = random.Random(seed)

    ranks: List[int] = []
    tie_cases = 0
    all_equal_cases = 0
    total_tie_items = 0
    total_cases = 0

    for batch in tqdm(loader, desc=desc, leave=False):
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
            scores = score_late_fusion(
                model=model,
                role_table=role_table,
                history=hist,
                candidates=candidates,
                beta=beta,
                pooling=pooling,
                recent_k=recent_k,
                decay=decay,
                role_score_norm=role_score_norm,
            )
            scores_np = scores.detach().cpu().numpy()

            for row, gt_pos in zip(scores_np, gt_positions):
                rank, tie_count, all_equal = _rank_from_scores(row, gt_pos)
                ranks.append(rank)
                total_cases += 1
                total_tie_items += tie_count
                tie_cases += int(tie_count > 1)
                all_equal_cases += int(all_equal)
        else:
            candidates = torch.arange(0, num_items + 1, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)
            scores = score_late_fusion(
                model=model,
                role_table=role_table,
                history=hist,
                candidates=candidates,
                beta=beta,
                pooling=pooling,
                recent_k=recent_k,
                decay=decay,
                role_score_norm=role_score_norm,
            )
            scores[:, 0] = -torch.inf
            for i in range(batch_size):
                target_i = int(pos[i].item())
                for item in hist[i].detach().cpu().tolist():
                    item = int(item)
                    if item != 0 and item != target_i:
                        scores[i, item] = -torch.inf

            scores_np = scores.detach().cpu().numpy()
            pos_cpu = pos.detach().cpu().tolist()
            for row, target_i in zip(scores_np, pos_cpu):
                rank, tie_count, all_equal = _rank_from_scores(row, int(target_i))
                ranks.append(rank)
                total_cases += 1
                total_tie_items += tie_count
                tie_cases += int(tie_count > 1)
                all_equal_cases += int(all_equal)

    elapsed = time.time() - start
    out = _metrics_from_ranks(ranks, ks=ks)
    out["tie_case_ratio"] = float(tie_cases / max(total_cases, 1))
    out["all_equal_ratio"] = float(all_equal_cases / max(total_cases, 1))
    out["avg_tie_items"] = float(total_tie_items / max(total_cases, 1))
    out["num_eval_users"] = int(total_cases)
    out["tie_policy"] = "worst"
    out["beta"] = float(beta)
    out["pooling"] = pooling
    out["recent_k"] = int(recent_k) if recent_k is not None else ""
    out["decay"] = float(decay)
    out["role_score_norm"] = role_score_norm
    out["eval_elapsed_sec"] = float(elapsed)
    out["eval_users_per_sec"] = float(total_cases / max(elapsed, 1e-9))
    return out
