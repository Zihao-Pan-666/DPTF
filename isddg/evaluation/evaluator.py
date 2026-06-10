from __future__ import annotations

from typing import Dict
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .metrics import recall_at_k, ndcg_at_k, mrr_at_k
from isddg.data.dataset import NegativeSampler


@torch.no_grad()
def evaluate_sampled(
    model,
    loader: DataLoader,
    num_items: int,
    device: torch.device,
    num_negatives: int = 100,
    seed: int = 2026,
    ks=(10, 20),
) -> Dict[str, float]:
    model.eval()
    sampler = NegativeSampler(num_items, seed=seed)
    rng = random.Random(seed)
    ranks = []
    tie_cases = 0
    total_cases = 0
    for batch in loader:
        hist = batch["history"].to(device)
        pos = batch["target"].tolist()
        cand_rows = []
        gt_positions = []
        for i, p in enumerate(pos):
            forbidden = set([x for x in hist[i].tolist() if x != 0])
            forbidden.add(int(p))
            negs = sampler.sample(forbidden, num_negatives)
            cands = [int(p)] + negs
            rng.shuffle(cands)
            gt_positions.append(cands.index(int(p)))
            cand_rows.append(cands)
        cands = torch.tensor(cand_rows, dtype=torch.long, device=device)
        scores = model.score(hist, cands) if hasattr(model, "score") else model(hist, cands)
        scores_np = scores.detach().cpu().numpy()
        for row, gt_pos in zip(scores_np, gt_positions):
            total_cases += 1
            if np.allclose(row, row[0]):
                tie_cases += 1
            # rank: number of candidates with strictly larger score.
            # This avoids giving the ground-truth item a free advantage in exact ties.
            rank = int(np.sum(row > row[gt_pos]))
            ranks.append(rank)
    ranks = np.asarray(ranks, dtype=np.int64)
    out = {f"Recall@{k}": recall_at_k(ranks, k) for k in ks}
    out.update({f"NDCG@{k}": ndcg_at_k(ranks, k) for k in ks})
    out["MRR@10"] = mrr_at_k(ranks, 10)
    out["tie_ratio"] = float(tie_cases / max(total_cases, 1))
    out["num_eval_users"] = int(total_cases)
    return out
