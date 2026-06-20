from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from isddg.evaluation.semantic_evaluator import evaluate_semantic_ranking


def bpr_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(pos_logits.unsqueeze(1) - neg_logits).mean()


@torch.no_grad()
def _sample_training_negatives_fast(
    train_seq: torch.Tensor,
    pos_items: torch.Tensor,
    num_items: int,
    num_negatives: int,
    device: torch.device,
    max_rounds: int = 16,
) -> torch.Tensor:
    batch_size, _ = train_seq.size()
    neg_items = torch.randint(1, int(num_items) + 1, size=(batch_size, int(num_negatives)), device=device)
    blocked = torch.cat([train_seq, pos_items.unsqueeze(1)], dim=1)

    for _ in range(max_rounds):
        invalid = (neg_items.unsqueeze(-1) == blocked.unsqueeze(1)).any(dim=-1)
        if not invalid.any():
            break
        neg_items[invalid] = torch.randint(
            1, int(num_items) + 1, size=(int(invalid.sum().item()),), device=device
        )

    invalid = (neg_items.unsqueeze(-1) == blocked.unsqueeze(1)).any(dim=-1)
    if invalid.any():
        invalid_indices = invalid.nonzero(as_tuple=False)
        for idx in invalid_indices:
            b, n = int(idx[0].item()), int(idx[1].item())
            blocked_set = {int(x) for x in blocked[b].detach().cpu().tolist() if int(x) != 0}
            while True:
                candidate = int(torch.randint(1, int(num_items) + 1, (1,), device=device).item())
                if candidate not in blocked_set:
                    neg_items[b, n] = candidate
                    break
    return neg_items


def train_semantic_v0(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_items: int,
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
    checkpoint_extra: Dict | None = None,
    show_progress: bool = True,
) -> Dict:
    """Train the semantic-only BERT4Rec baseline.

    Public inputs/outputs are kept unchanged. The evaluator call is aligned with
    semantic_evaluator's tie-policy-aware signature.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_metric = -float("inf")
    best_epoch = -1
    patience = 0
    history = []
    train_started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        total_loss = 0.0
        total_examples = 0

        pbar = tqdm(
            train_loader,
            desc=f"train epoch {epoch:03d}/{epochs:03d}",
            total=len(train_loader) if hasattr(train_loader, "__len__") else None,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        )

        for batch in pbar:
            hist = batch["history"].to(device, non_blocking=True)
            pos = batch["target"].to(device, non_blocking=True)
            neg = _sample_training_negatives_fast(hist, pos, num_items, train_negatives, device)

            candidates = torch.cat([pos.view(-1, 1), neg], dim=1)
            scores = model.score(hist, candidates)
            loss = bpr_loss(scores[:, 0], scores[:, 1:])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            batch_size = hist.size(0)
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            elapsed = max(time.perf_counter() - start, 1e-9)

            if show_progress:
                pbar.set_postfix(
                    loss=f"{total_loss / max(total_examples, 1):.6f}",
                    samples_s=f"{total_examples / elapsed:.1f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        avg_loss = total_loss / max(total_examples, 1)
        elapsed = max(time.perf_counter() - start, 1e-9)
        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_elapsed_sec": elapsed,
            "train_samples_per_sec": total_examples / elapsed,
        }

        if epoch % eval_every == 0 or epoch == epochs:
            val_metrics = evaluate_semantic_ranking(
                model=model,
                loader=val_loader,
                num_items=num_items,
                device=device,
                ranking_mode=eval_ranking_mode,
                num_negatives=eval_negatives,
                seed=seed + epoch,
                ks=ks,
                tie_policy="worst",
                show_progress=show_progress,
                progress_desc=f"source val epoch {epoch:03d}",
            )
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            current = float(val_metrics.get(early_stop_metric, -float("inf")))

            if current > best_metric:
                best_metric = current
                best_epoch = epoch
                patience = 0
                payload = {
                    "model_state": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "early_stop_metric": early_stop_metric,
                    "num_items": num_items,
                    "history": history + [row],
                }
                if checkpoint_extra:
                    payload.update(checkpoint_extra)
                torch.save(payload, checkpoint_path)
                status = "saved"
            else:
                patience += 1
                status = f"patience={patience}/{early_stop_patience}"

            print(
                f"[SemanticV0][epoch={epoch:03d}] "
                f"loss={avg_loss:.6f} "
                f"train_speed={total_examples / elapsed:.1f} samples/s "
                f"val_{early_stop_metric}={current:.6f} "
                f"best={best_metric:.6f}@{best_epoch} "
                f"val_speed={val_metrics.get('eval_users_per_sec', 0.0):.1f} users/s "
                f"time={elapsed:.1f}s {status}"
            )

            if patience >= early_stop_patience:
                print(f"[SemanticV0] Early stopping at epoch {epoch}.")
                history.append(row)
                break
        else:
            print(
                f"[SemanticV0][epoch={epoch:03d}] "
                f"loss={avg_loss:.6f} "
                f"train_speed={total_examples / elapsed:.1f} samples/s "
                f"time={elapsed:.1f}s"
            )

        history.append(row)

    total_elapsed = max(time.perf_counter() - train_started, 1e-9)
    return {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "early_stop_metric": early_stop_metric,
        "checkpoint_path": str(checkpoint_path),
        "total_elapsed_sec": total_elapsed,
        "history": history,
    }
