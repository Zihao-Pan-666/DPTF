from __future__ import annotations
from typing import Dict, List, Sequence
import random, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from isddg.evaluation.metrics import recall_at_k, ndcg_at_k, mrr_at_k

def _sample_unique_negatives(num_items, forbidden, n, rng):
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
                if len(out) >= target_n: break
    return out

def _rank(scores, gt_pos, atol=1e-8):
    gt = float(scores[gt_pos])
    greater = int(np.sum(scores > gt + atol))
    tied = int(np.sum(np.isclose(scores, gt, atol=atol, rtol=0.0)))
    all_equal = bool(np.allclose(scores, scores[0], atol=atol, rtol=0.0))
    return greater + max(tied - 1, 0), tied, all_equal

@torch.no_grad()
def evaluate_semantic_ranking(model, loader: DataLoader, num_items: int, device: torch.device, ranking_mode: str = "sampled", num_negatives: int = 100, seed: int = 2026, ks: Sequence[int] = (10, 20), desc: str = "eval") -> Dict[str, float]:
    start = time.time()
    model.eval(); rng = random.Random(seed)
    ranks, tie_cases, all_equal_cases, total_tie_items, total = [], 0, 0, 0, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        hist = batch["history"].to(device)
        pos = batch["target"].to(device)
        if ranking_mode == "sampled":
            rows, gt_pos = [], []
            hcpu, pcpu = hist.cpu().tolist(), pos.cpu().tolist()
            for i, p in enumerate(pcpu):
                forbidden = {int(x) for x in hcpu[i] if int(x) != 0}; forbidden.add(int(p))
                cands = [int(p)] + _sample_unique_negatives(num_items, forbidden, num_negatives, rng)
                rng.shuffle(cands); gt_pos.append(cands.index(int(p))); rows.append(cands)
            scores = model.score(hist, torch.tensor(rows, dtype=torch.long, device=device)).detach().cpu().numpy()
            for row, gp in zip(scores, gt_pos):
                r, tied, all_eq = _rank(row, gp)
                ranks.append(r); total += 1; total_tie_items += tied; tie_cases += int(tied > 1); all_equal_cases += int(all_eq)
        else:
            scores = model.score_all(hist); scores[:, 0] = -torch.inf
            for i in range(hist.size(0)):
                tgt = int(pos[i])
                for item in hist[i].cpu().tolist():
                    if item != 0 and int(item) != tgt: scores[i, int(item)] = -torch.inf
            scores_np = scores.detach().cpu().numpy()
            for row, tgt in zip(scores_np, pos.cpu().tolist()):
                r, tied, all_eq = _rank(row, int(tgt))
                ranks.append(r); total += 1; total_tie_items += tied; tie_cases += int(tied > 1); all_equal_cases += int(all_eq)
    ranks = np.asarray(ranks, dtype=np.int64)
    out = {}
    for k in ks: out[f"Recall@{k}"] = recall_at_k(ranks, k)
    for k in ks: out[f"NDCG@{k}"] = ndcg_at_k(ranks, k)
    for k in ks: out[f"MRR@{k}"] = mrr_at_k(ranks, k)
    elapsed = time.time() - start
    out.update({"tie_case_ratio": tie_cases / max(total, 1), "all_equal_ratio": all_equal_cases / max(total, 1), "avg_tie_items": total_tie_items / max(total, 1), "num_eval_users": int(total), "tie_policy": "worst", "eval_elapsed_sec": elapsed, "eval_users_per_sec": total / max(elapsed, 1e-9)})
    return out
