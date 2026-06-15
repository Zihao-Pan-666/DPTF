from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from isddg.evaluation.semantic_evaluator import evaluate_semantic_ranking
from isddg.training.semantic_v0_trainer import bpr_loss, _sample_training_negatives_fast


def _cov(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    return x.t().matmul(x) / max(x.size(0) - 1, 1)


def coral_loss(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    if src.size(0) < 2 or tgt.size(0) < 2:
        return torch.zeros((), device=src.device)
    src = F.normalize(src, dim=-1)
    tgt = F.normalize(tgt, dim=-1)
    return F.mse_loss(src.mean(0), tgt.mean(0)) + F.mse_loss(_cov(src), _cov(tgt))


def mmd_loss(src: torch.Tensor, tgt: torch.Tensor, sigmas=(0.5, 1.0, 2.0, 4.0)) -> torch.Tensor:
    if src.size(0) < 2 or tgt.size(0) < 2:
        return torch.zeros((), device=src.device)
    src = F.normalize(src, dim=-1)
    tgt = F.normalize(tgt, dim=-1)
    xx, yy, xy = torch.cdist(src, src).pow(2), torch.cdist(tgt, tgt).pow(2), torch.cdist(src, tgt).pow(2)
    out = torch.zeros((), device=src.device)
    for s in sigmas:
        g = 1.0 / (2.0 * s * s)
        out = out + torch.exp(-g * xx).mean() + torch.exp(-g * yy).mean() - 2 * torch.exp(-g * xy).mean()
    return out / len(sigmas)


def _batch_source_ids(hist: torch.Tensor, pos: torch.Tensor, max_items: int) -> torch.Tensor:
    ids = torch.unique(torch.cat([hist.reshape(-1), pos.reshape(-1)], dim=0))
    ids = ids[ids > 0]
    if ids.numel() > max_items:
        ids = ids[torch.randperm(ids.numel(), device=ids.device)[:max_items]]
    return ids


def _sample_target_features(target_features: torch.Tensor, sample_size: int, device: torch.device) -> torch.Tensor:
    n = target_features.size(0) - 1
    idx = torch.randint(1, n + 1, (min(sample_size, n),), device=device)
    return target_features.to(device)[idx]


def item_align_loss(model, hist, pos, target_features, sample_size: int, method: str):
    device = next(model.parameters()).device
    src_ids = _batch_source_ids(hist, pos, sample_size).to(device)
    if src_ids.numel() < 2:
        return torch.zeros((), device=device)
    src_z = model.item_proj(model.item_features[src_ids].to(device))
    tgt_z = model.item_proj(_sample_target_features(target_features, sample_size, device))
    if method == "coral":
        return coral_loss(src_z, tgt_z)
    if method == "mmd":
        return mmd_loss(src_z, tgt_z)
    raise ValueError(f"Unsupported alignment method: {method}")


def train_semantic_align_item(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_items: int,
    target_item_features: torch.Tensor,
    device: torch.device,
    checkpoint_path: str | Path,
    epochs: int = 20,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    train_negatives: int = 5,
    eval_negatives: int = 100,
    eval_ranking_mode: str = "sampled",
    early_stop_metric: str = "NDCG@10",
    early_stop_patience: int = 5,
    eval_every: int = 1,
    grad_clip: float = 5.0,
    seed: int = 2026,
    ks: Sequence[int] = (10, 20),
    align_alpha: float = 0.001,
    align_sample_size: int = 256,
    align_method: str = "coral",
    align_warmup_epochs: int = 0,
    checkpoint_extra: Dict | None = None,
) -> Dict:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_metric, best_epoch, patience = -float("inf"), -1, 0
    history = []
    total_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.time()
        total_loss = total_rec = total_align = total_n = 0.0
        use_align = epoch > align_warmup_epochs and align_alpha > 0

        pbar = tqdm(train_loader, desc=f"align-item epoch {epoch:03d}", leave=False)
        for batch in pbar:
            hist = batch["history"].to(device, non_blocking=True)
            pos = batch["target"].to(device, non_blocking=True)

            neg = _sample_training_negatives_fast(hist, pos, num_items, train_negatives, device)
            scores = model.score(hist, torch.cat([pos.view(-1, 1), neg], dim=1))
            rec = bpr_loss(scores[:, 0], scores[:, 1:])
            align = item_align_loss(model, hist, pos, target_item_features, align_sample_size, align_method) if use_align else torch.zeros((), device=device)
            loss = rec + align_alpha * align

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bs = hist.size(0)
            total_loss += float(loss.item()) * bs
            total_rec += float(rec.item()) * bs
            total_align += float(align.item()) * bs
            total_n += bs
            pbar.set_postfix(loss=f"{total_loss/max(total_n,1):.6f}", rec=f"{total_rec/max(total_n,1):.6f}", align=f"{total_align/max(total_n,1):.6f}")

        elapsed = time.time() - start
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_n, 1),
            "train_rec_loss": total_rec / max(total_n, 1),
            "train_align_loss_raw": total_align / max(total_n, 1),
            "train_align_loss_weighted": align_alpha * total_align / max(total_n, 1),
            "train_elapsed_sec": elapsed,
            "train_samples_per_sec": total_n / max(elapsed, 1e-9),
            "align_method": align_method,
            "align_alpha": align_alpha,
            "align_sample_size": align_sample_size,
        }

        if epoch % eval_every == 0 or epoch == epochs:
            val = evaluate_semantic_ranking(model, val_loader, num_items, device, eval_ranking_mode, eval_negatives, seed + epoch, ks, "worst")
            row.update({f"val_{k}": v for k, v in val.items()})
            cur = float(val.get(early_stop_metric, -float("inf")))
            if cur > best_metric:
                best_metric, best_epoch, patience = cur, epoch, 0
                payload = {
                    "model_state": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "early_stop_metric": early_stop_metric,
                    "num_items": num_items,
                    "history": history + [row],
                    "alignment": {
                        "method": align_method,
                        "alpha": align_alpha,
                        "sample_size": align_sample_size,
                        "warmup_epochs": align_warmup_epochs,
                    },
                    "total_elapsed_sec": time.time() - total_start,
                }
                if checkpoint_extra:
                    payload.update(checkpoint_extra)
                torch.save(payload, checkpoint_path)
                status = "saved"
            else:
                patience += 1
                status = f"patience={patience}/{early_stop_patience}"

            print(f"[SemanticAlignItem][epoch={epoch:03d}] loss={row['train_loss']:.6f} rec={row['train_rec_loss']:.6f} align={row['train_align_loss_raw']:.6f} val_{early_stop_metric}={cur:.6f} best={best_metric:.6f}@{best_epoch} time={elapsed:.1f}s {status}")
            if patience >= early_stop_patience:
                print(f"[SemanticAlignItem] Early stopping at epoch {epoch}.")
                history.append(row)
                break

        history.append(row)

    return {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "early_stop_metric": early_stop_metric,
        "checkpoint_path": str(checkpoint_path),
        "total_elapsed_sec": time.time() - total_start,
        "history": history,
    }
