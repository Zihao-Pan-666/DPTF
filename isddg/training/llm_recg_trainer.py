from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from isddg.evaluation.llm_recg_evaluator import evaluate_llm_recg_ranking
from isddg.training.llm_recg_losses import bpr_loss, alignment_loss_with_sampled_entropy


def _sample_training_negatives_official(
    batch_size: int,
    num_items: int,
    num_negatives: int,
    device: torch.device,
) -> torch.Tensor:
    # Official-style simple negative sampling. No target-domain information is used.
    return torch.randint(1, int(num_items) + 1, (int(batch_size), int(num_negatives)), device=device)


def _resample_aux_embeddings(
    sampled_embeddings: torch.Tensor,
    sampled_domains: torch.Tensor,
    batch_size: int,
    num_aux_domains: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official-style: resample `batch_size` items per auxiliary domain each step."""
    sampled_embeddings = sampled_embeddings.to(device)
    sampled_domains = sampled_domains.to(device).long()

    raws = []
    doms = []
    for domain_id in range(int(num_aux_domains)):
        mask = sampled_domains.eq(domain_id)
        domain_embeddings = sampled_embeddings[mask]
        if domain_embeddings.size(0) == 0:
            continue
        if domain_embeddings.size(0) >= batch_size:
            idx = torch.randperm(domain_embeddings.size(0), device=device)[:batch_size]
        else:
            idx = torch.randint(0, domain_embeddings.size(0), (batch_size,), device=device)
        raws.append(domain_embeddings[idx])
        doms.append(torch.full((batch_size,), int(domain_id), dtype=torch.long, device=device))

    if not raws:
        return torch.empty((0, 0), device=device), torch.empty((0,), dtype=torch.long, device=device)
    return torch.cat(raws, dim=0), torch.cat(doms, dim=0)


def train_llm_recg(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_items: int,
    sampled_aux_embeddings: torch.Tensor,
    sampled_aux_domains: torch.Tensor,
    num_aux_domains: int,
    device: torch.device,
    checkpoint_path: str | Path,
    epochs: int = 50,
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
    alpha: float = 0.001,
    alignment_temperature: float = 1.0,
    use_source_val_selection: bool = True,
    checkpoint_extra: Dict | None = None,
    show_progress: bool = True,
) -> Dict:
    """Train BERT4Rec-RecG.

    This follows official LLM-RecG training mechanics:
      - BPR source recommendation loss;
      - source item raw embeddings sampled from the source catalog;
      - fixed pre-sampled auxiliary item embeddings;
      - domain-alignment projection before entropy loss;
      - L_gen = -alpha * H_intra + beta * H_inter.

    For this project, checkpoint selection is source-validation based by default
    to avoid target-domain tuning. Set `use_source_val_selection=False` to mimic
    the official train-loss checkpoint behavior more closely.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best_metric = -float("inf")
    best_loss = float("inf")
    best_epoch = -1
    patience = 0
    history = []
    train_started = time.perf_counter()

    sampled_aux_embeddings = sampled_aux_embeddings.float().to(device)
    sampled_aux_domains = sampled_aux_domains.long().to(device)
    num_domains_total = int(num_aux_domains) + 1
    current_domain_id = int(num_aux_domains)  # official convention: source/current domain is the last id.

    for epoch in range(1, int(epochs) + 1):
        model.train()
        start = time.perf_counter()

        total_loss = total_bpr = total_align = total_intra = total_inter = 0.0
        total_examples = 0

        pbar = tqdm(
            train_loader,
            desc=f"BERT4Rec-RecG train {epoch:03d}/{int(epochs):03d}",
            total=len(train_loader) if hasattr(train_loader, "__len__") else None,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        )

        for batch in pbar:
            hist = batch["history"].to(device, non_blocking=True)
            pos = batch["target"].to(device, non_blocking=True)
            bsz = int(hist.size(0))

            neg = _sample_training_negatives_official(
                batch_size=bsz,
                num_items=num_items,
                num_negatives=train_negatives,
                device=device,
            )
            candidates = torch.cat([pos.view(-1, 1), neg], dim=1)
            scores = model.score(hist, candidates, is_target_domain=False)
            rec_loss = bpr_loss(scores[:, 0], scores[:, 1:])

            current_raw = model.sample_internal_embeddings(sample_size=bsz, device=device)
            aux_raw, aux_domains = _resample_aux_embeddings(
                sampled_embeddings=sampled_aux_embeddings,
                sampled_domains=sampled_aux_domains,
                batch_size=bsz,
                num_aux_domains=num_aux_domains,
                device=device,
            )

            current_proj = model.irm_projection_embeddings(current_raw)
            current_domains = torch.full(
                (current_proj.size(0),),
                current_domain_id,
                dtype=torch.long,
                device=device,
            )

            if aux_raw.numel() > 0:
                aux_proj = model.irm_projection_embeddings(aux_raw)
                combined_proj = torch.cat([current_proj, aux_proj], dim=0)
                combined_domains = torch.cat([current_domains, aux_domains], dim=0)
            else:
                combined_proj = current_proj
                combined_domains = current_domains

            align_loss, align_log = alignment_loss_with_sampled_entropy(
                sampled_embeddings=combined_proj,
                sampled_domains=combined_domains,
                num_domains=num_domains_total,
                alpha_base=alpha,
                temperature=alignment_temperature,
            )

            loss = rec_loss + align_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_examples += bsz
            total_loss += float(loss.item()) * bsz
            total_bpr += float(rec_loss.item()) * bsz
            total_align += float(align_loss.item()) * bsz
            total_intra += float(align_log["intra_entropy"].item()) * bsz
            total_inter += float(align_log["inter_entropy"].item()) * bsz

            if show_progress:
                pbar.set_postfix(
                    loss=f"{total_loss / max(total_examples, 1):.5f}",
                    bpr=f"{total_bpr / max(total_examples, 1):.5f}",
                    align=f"{total_align / max(total_examples, 1):.6f}",
                )

        elapsed = max(time.perf_counter() - start, 1e-9)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_examples, 1),
            "train_bpr": total_bpr / max(total_examples, 1),
            "train_align": total_align / max(total_examples, 1),
            "train_intra_entropy": total_intra / max(total_examples, 1),
            "train_inter_entropy": total_inter / max(total_examples, 1),
            "train_elapsed_sec": elapsed,
            "train_samples_per_sec": total_examples / elapsed,
        }

        should_eval = (epoch % int(eval_every) == 0) or (epoch == int(epochs))
        if should_eval:
            val_metrics = evaluate_llm_recg_ranking(
                model=model,
                loader=val_loader,
                num_items=num_items,
                device=device,
                ranking_mode=eval_ranking_mode,
                num_negatives=eval_negatives,
                seed=seed + epoch,
                ks=ks,
                tie_policy="worst",
                is_target_domain=False,
                show_progress=show_progress,
                progress_desc=f"BERT4Rec-RecG source val epoch {epoch:03d}",
            )
            row.update({f"val_{k}": v for k, v in val_metrics.items()})

        current_metric = float(row.get(f"val_{early_stop_metric}", -float("inf")))
        current_loss = float(row["train_loss"])
        improved = False
        if use_source_val_selection and should_eval:
            improved = current_metric > best_metric
        elif not use_source_val_selection:
            improved = current_loss < best_loss

        if improved:
            best_metric = current_metric if should_eval else best_metric
            best_loss = current_loss
            best_epoch = epoch
            patience = 0
            payload = {
                "model_state": model.state_dict(),
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "best_train_loss": best_loss,
                "early_stop_metric": early_stop_metric,
                "selection_mode": "source_val" if use_source_val_selection else "train_loss",
                "embedding_dim": int(model.pretrained_dim),
                "model_hparams": {
                    "class": "LLMRecGBERT4Rec",
                    "hidden_dim": int(model.hidden_dim),
                    "max_len": int(model.max_len),
                    "num_layers": int(model.num_layers),
                    "num_heads": int(model.num_heads),
                    "dropout": float(model.dropout_rate),
                    "num_sequential_patterns": int(model.num_sequential_patterns),
                    "pattern_fusion": str(model.pattern_fusion),
                    "pattern_residual_weight": float(model.pattern_residual_weight),
                },
            }
            if checkpoint_extra:
                payload.update(checkpoint_extra)
            torch.save(payload, checkpoint_path)
            status = "saved"
        else:
            patience += 1
            status = f"patience={patience}/{int(early_stop_patience)}"

        history.append(row)
        if should_eval:
            print(
                f"[BERT4Rec-RecG][epoch={epoch:03d}] "
                f"loss={row['train_loss']:.6f} bpr={row['train_bpr']:.6f} align={row['train_align']:.6f} "
                f"val_NDCG@10={row.get('val_NDCG@10', float('nan')):.6f} "
                f"all_equal={row.get('val_all_equal_ratio', float('nan')):.6f} "
                f"best={best_metric:.6f}@{best_epoch} {status}"
            )
        else:
            print(
                f"[BERT4Rec-RecG][epoch={epoch:03d}] "
                f"loss={row['train_loss']:.6f} bpr={row['train_bpr']:.6f} align={row['train_align']:.6f} {status}"
            )

        if patience >= int(early_stop_patience):
            print(f"[BERT4Rec-RecG] Early stopping at epoch {epoch}.")
            break

    return {
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "best_train_loss": float(best_loss),
        "early_stop_metric": early_stop_metric,
        "selection_mode": "source_val" if use_source_val_selection else "train_loss",
        "checkpoint_path": str(checkpoint_path),
        "total_elapsed_sec": float(time.perf_counter() - train_started),
        "history": history,
    }
