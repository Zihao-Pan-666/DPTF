from __future__ import annotations

from typing import Any, Dict, List, Sequence
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from isddg.evaluation.metrics import recall_at_k, ndcg_at_k, mrr_at_k
from isddg.evaluation.dynamic_signal_evaluator import normalize_scores
from isddg.prototypes.sequence_dynamic import SequenceDynamicPrototypeBank


def parse_float_grid(x) -> List[float]:
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
    out = {
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


@torch.no_grad()
def retrieve_prototype_values(
    bank: SequenceDynamicPrototypeBank,
    query: torch.Tensor,
    top_m: int = 16,
    temperature: float = 0.05,
) -> Dict[str, torch.Tensor]:
    keys = torch.nn.functional.normalize(bank.keys.to(query.device), dim=-1)
    q = torch.nn.functional.normalize(query, dim=-1)
    sim = q @ keys.t()
    k = min(int(top_m), int(sim.size(1)))
    vals, idx = torch.topk(sim, k=k, dim=1)
    attn = torch.softmax(vals / max(float(temperature), 1e-6), dim=1)

    sem_values = bank.semantic_values.to(query.device)
    dyn_values = bank.dynamic_values.to(query.device)
    role_values = bank.role_values.to(query.device)

    proto_sem = torch.einsum("bt,bth->bh", attn, sem_values[idx]) if sem_values.numel() else torch.zeros(query.size(0), 0, device=query.device)
    proto_dyn = torch.einsum("bt,btd->bd", attn, dyn_values[idx]) if dyn_values.numel() else torch.zeros(query.size(0), 0, device=query.device)
    proto_role = torch.einsum("bt,btk->bk", attn, role_values[idx]) if role_values.numel() else torch.zeros(query.size(0), 0, device=query.device)

    entropy = -(attn * attn.clamp(min=1e-8).log()).sum(dim=1)
    denom = torch.log(torch.tensor(float(k), device=query.device)).clamp(min=1e-8)
    confidence = (1.0 - entropy / denom).clamp(0.0, 1.0)
    return {
        "indices": idx,
        "attention": attn,
        "similarities": vals,
        "semantic": proto_sem,
        "dynamic": proto_dyn,
        "role": proto_role,
        "confidence": confidence,
    }


@torch.no_grad()
def sequence_prototype_scores(
    model,
    bank: SequenceDynamicPrototypeBank,
    history: torch.Tensor,
    candidates: torch.Tensor,
    beta_sem: float = 0.0,
    beta_dyn: float = 0.0,
    candidate_dynamic_table: torch.Tensor | None = None,
    top_m: int = 16,
    temperature: float = 0.05,
    semantic_score_norm: str | None = "zscore",
    prototype_score_norm: str | None = "zscore",
    dynamic_score_norm: str | None = "zscore",
    return_components: bool = False,
):
    user_h = model(history)
    sem_raw = model.score(history, candidates)
    sem = normalize_scores(sem_raw, mode=semantic_score_norm)

    retrieved = retrieve_prototype_values(bank, user_h, top_m=top_m, temperature=temperature)
    proto_sem_score = None
    proto_dyn_score = None

    if retrieved["semantic"].numel() and float(beta_sem) != 0.0:
        cand_h = model.encode_items(candidates)
        proto_sem_raw = torch.einsum("bd,bnd->bn", retrieved["semantic"], cand_h)
        proto_sem_score = normalize_scores(proto_sem_raw, mode=prototype_score_norm)
    elif retrieved["semantic"].numel() and return_components:
        cand_h = model.encode_items(candidates)
        proto_sem_raw = torch.einsum("bd,bnd->bn", retrieved["semantic"], cand_h)
        proto_sem_score = normalize_scores(proto_sem_raw, mode=prototype_score_norm)

    if candidate_dynamic_table is not None and retrieved["dynamic"].numel() and (float(beta_dyn) != 0.0 or return_components):
        dyn_table = candidate_dynamic_table.float().to(history.device)
        cand_dyn = dyn_table[candidates]
        proto_dyn_raw = torch.einsum("bd,bnd->bn", retrieved["dynamic"], cand_dyn)
        proto_dyn_score = normalize_scores(proto_dyn_raw, mode=dynamic_score_norm)

    fused = sem
    if proto_sem_score is not None and float(beta_sem) != 0.0:
        fused = fused + float(beta_sem) * proto_sem_score
    if proto_dyn_score is not None and float(beta_dyn) != 0.0:
        fused = fused + float(beta_dyn) * proto_dyn_score

    if not return_components:
        return fused
    return fused, {
        "semantic_raw": sem_raw,
        "semantic": sem,
        "prototype_semantic": proto_sem_score,
        "prototype_dynamic": proto_dyn_score,
        "prototype_confidence": retrieved["confidence"],
        "prototype_top_similarity": retrieved["similarities"][:, 0] if retrieved["similarities"].numel() else None,
    }


@torch.no_grad()
def evaluate_sequence_prototype(
    model,
    bank: SequenceDynamicPrototypeBank,
    loader: DataLoader,
    num_items: int,
    device: torch.device,
    beta_sem: float = 0.0,
    beta_dyn: float = 0.0,
    candidate_dynamic_table: torch.Tensor | None = None,
    ranking_mode: str = "sampled",
    num_negatives: int = 100,
    seed: int = 2026,
    ks: Sequence[int] = (10, 20),
    top_m: int = 16,
    temperature: float = 0.05,
    semantic_score_norm: str | None = "zscore",
    prototype_score_norm: str | None = "zscore",
    dynamic_score_norm: str | None = "zscore",
    tie_policy: str = "worst",
    desc: str | None = None,
    show_progress: bool = False,
    collect_score_stats: bool = False,
) -> Dict[str, Any]:
    if ranking_mode not in {"sampled", "full"}:
        raise ValueError(f"ranking_mode must be sampled or full, got {ranking_mode}")

    model.eval()
    bank = bank.to(device)
    if candidate_dynamic_table is not None:
        candidate_dynamic_table = candidate_dynamic_table.float().to(device)

    rng = random.Random(seed)
    start = time.time()
    ranks: List[float] = []
    total_tie = tie_cases = all_equal_cases = total = 0
    first_score_stats: Dict[str, float] = {}

    eval_iter = loader
    if show_progress:
        eval_iter = tqdm(loader, desc=desc, leave=False, ascii=True, dynamic_ncols=True, mininterval=5.0, maxinterval=30.0)

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
        else:
            candidates = torch.arange(0, num_items + 1, dtype=torch.long, device=device).view(1, -1).expand(batch_size, -1)
            gt_positions = [int(x) for x in pos.detach().cpu().tolist()]

        if collect_score_stats and not first_score_stats:
            scores_t, comps = sequence_prototype_scores(
                model=model,
                bank=bank,
                history=hist,
                candidates=candidates,
                beta_sem=beta_sem,
                beta_dyn=beta_dyn,
                candidate_dynamic_table=candidate_dynamic_table,
                top_m=top_m,
                temperature=temperature,
                semantic_score_norm=semantic_score_norm,
                prototype_score_norm=prototype_score_norm,
                dynamic_score_norm=dynamic_score_norm,
                return_components=True,
            )
            for key, value in comps.items():
                if value is not None:
                    first_score_stats.update(_prefix_stats(key, value))
            first_score_stats.update(_prefix_stats("fused", scores_t))
            scores = scores_t
        else:
            scores = sequence_prototype_scores(
                model=model,
                bank=bank,
                history=hist,
                candidates=candidates,
                beta_sem=beta_sem,
                beta_dyn=beta_dyn,
                candidate_dynamic_table=candidate_dynamic_table,
                top_m=top_m,
                temperature=temperature,
                semantic_score_norm=semantic_score_norm,
                prototype_score_norm=prototype_score_norm,
                dynamic_score_norm=dynamic_score_norm,
            )

        if ranking_mode == "full":
            scores[:, 0] = -torch.inf
            for i in range(batch_size):
                target_i = int(pos[i].item())
                for item in hist[i].detach().cpu().tolist():
                    item = int(item)
                    if item != 0 and item != target_i:
                        scores[i, item] = -torch.inf
            scores_np = scores.detach().cpu().numpy()
            for row, target_i in zip(scores_np, gt_positions):
                rank, tied, all_eq = _rank_from_scores(row, int(target_i), tie_policy=tie_policy)
                ranks.append(rank)
                total += 1
                total_tie += tied
                tie_cases += int(tied > 1)
                all_equal_cases += int(all_eq)
        else:
            scores_np = scores.detach().cpu().numpy()
            for row, gt_pos in zip(scores_np, gt_positions):
                rank, tied, all_eq = _rank_from_scores(row, gt_pos, tie_policy=tie_policy)
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
        "beta_sem": float(beta_sem),
        "beta_dyn": float(beta_dyn),
        "top_m": int(top_m),
        "temperature": float(temperature),
        "semantic_score_norm": semantic_score_norm if semantic_score_norm is not None else "none",
        "prototype_score_norm": prototype_score_norm if prototype_score_norm is not None else "none",
        "dynamic_score_norm": dynamic_score_norm if dynamic_score_norm is not None else "none",
        "eval_elapsed_sec": float(elapsed),
        "eval_users_per_sec": float(total / max(elapsed, 1e-9)),
    })
    if collect_score_stats:
        out.update(first_score_stats)
    return out
