from __future__ import annotations

from typing import Dict, List, Sequence
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from isddg.evaluation.metrics import recall_at_k, ndcg_at_k, mrr_at_k


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


@torch.no_grad()
def evaluate_semantic_ranking(
    model,
    loader: DataLoader,
    num_items: int,
    device: torch.device,
    ranking_mode: str = "sampled",
    num_negatives: int = 100,
    seed: int = 2026,
    ks: Sequence[int] = (10, 20),
    desc: str = "eval",
    tie_policy: str = "worst",
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> Dict[str, float]:
    if ranking_mode not in {"sampled", "full"}:
        raise ValueError(f"ranking_mode must be sampled or full, got {ranking_mode}")

    start = time.time()
    model.eval()
    rng = random.Random(seed)
    ranks: List[float] = []
    tie_cases = all_equal_cases = total_tie_items = total = 0

    for batch in tqdm(loader, desc=progress_desc or desc, leave=False, disable=not show_progress):
        hist = batch["history"].to(device)
        pos = batch["target"].to(device)
        batch_size = hist.size(0)

        if ranking_mode == "sampled":
            rows: List[List[int]] = []
            gt_positions: List[int] = []
            hist_cpu = hist.detach().cpu().tolist()
            pos_cpu = pos.detach().cpu().tolist()
            for i, p in enumerate(pos_cpu):
                forbidden = {int(x) for x in hist_cpu[i] if int(x) != 0}
                forbidden.add(int(p))
                cands = [int(p)] + _sample_unique_negatives(num_items, forbidden, num_negatives, rng)
                rng.shuffle(cands)
                gt_positions.append(cands.index(int(p)))
                rows.append(cands)
            candidates = torch.tensor(rows, dtype=torch.long, device=device)
            scores = model.score(hist, candidates).detach().cpu().numpy()
            for row, gt_pos in zip(scores, gt_positions):
                rank, tied, all_eq = _rank_from_scores(row, gt_pos, tie_policy=tie_policy)
                ranks.append(rank)
                total += 1
                total_tie_items += tied
                tie_cases += int(tied > 1)
                all_equal_cases += int(all_eq)
        else:
            if hasattr(model, "score_all"):
                scores = model.score_all(hist)
            else:
                candidates = torch.arange(0, num_items + 1, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)
                scores = model.score(hist, candidates)
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
                total_tie_items += tied
                tie_cases += int(tied > 1)
                all_equal_cases += int(all_eq)

    elapsed = time.time() - start
    out = _metrics_from_ranks(ranks, ks)
    out.update({
        "tie_case_ratio": float(tie_cases / max(total, 1)),
        "all_equal_ratio": float(all_equal_cases / max(total, 1)),
        "avg_tie_items": float(total_tie_items / max(total, 1)),
        "num_eval_users": int(total),
        "tie_policy": tie_policy,
        "eval_elapsed_sec": float(elapsed),
        "eval_users_per_sec": float(total / max(elapsed, 1e-9)),
    })
    return out
