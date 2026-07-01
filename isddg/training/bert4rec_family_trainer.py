from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - training still works without tqdm
    tqdm = None

from isddg.evaluation.bert4rec_family_evaluator import evaluate_bert4rec_family
from isddg.training.bert4rec_family_losses import (
    compute_alignment_loss,
    stable_bpr_loss,
)



_PROGRESS_TRUE = {"1", "true", "yes", "on", "show", "enabled"}
_PROGRESS_FALSE = {"0", "false", "no", "off", "hide", "disabled", "quiet", "batch"}


def _resolve_progress_enabled(explicit: bool | None = None) -> bool:
    """
    Direct launches show progress by default.

    A batch launcher can suppress all bars without changing YAML files by
    temporarily setting the environment variable ISDDG_PROGRESS=0.
    """
    raw = os.getenv("ISDDG_PROGRESS")
    if raw is not None:
        normalized = raw.strip().lower()
        if normalized in _PROGRESS_TRUE:
            return True
        if normalized in _PROGRESS_FALSE:
            return False
        raise ValueError(
            "ISDDG_PROGRESS must be one of "
            f"{sorted(_PROGRESS_TRUE | _PROGRESS_FALSE)}, got {raw!r}"
        )
    if explicit is not None:
        return bool(explicit)
    return True


def _safe_total(loader: Iterable, max_batches: int) -> int | None:
    try:
        total = len(loader)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return max_batches if max_batches > 0 else None
    if max_batches > 0:
        total = min(int(total), int(max_batches))
    return int(total)

def _assert_finite(name: str, value: torch.Tensor, context: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(
            f"{name} contains NaN/Inf at {context}. "
            "Training stopped before optimizer.step/checkpoint save."
        )


def _sample_training_negatives(
    histories: torch.Tensor,
    positives: torch.Tensor,
    num_items: int,
    num_negatives: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Sample negatives while excluding every non-PAD history item and the target.
    """
    history_cpu = histories.detach().cpu().numpy()
    positive_cpu = positives.detach().cpu().numpy()
    rows: list[list[int]] = []

    for history, positive in zip(history_cpu, positive_cpu):
        blocked = {int(x) for x in history.tolist() if int(x) > 0}
        blocked.add(int(positive))

        available = num_items - len([x for x in blocked if 1 <= x <= num_items])
        if available <= 0:
            raise ValueError("No valid training negatives remain")

        values: list[int] = []
        while len(values) < num_negatives:
            draw = torch.randint(
                low=1,
                high=num_items + 1,
                size=(max(32, num_negatives * 4),),
                generator=generator,
            ).tolist()
            for item in draw:
                item = int(item)
                if item not in blocked:
                    values.append(item)
                    if len(values) == num_negatives:
                        break
        rows.append(values)

    return torch.as_tensor(rows, dtype=torch.long, device=histories.device)


class AuxiliaryPoolSampler:
    def __init__(
        self,
        pools: dict[str, torch.Tensor],
        samples_per_domain: int,
        seed: int,
    ) -> None:
        self.pools = pools
        self.samples_per_domain = int(samples_per_domain)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))

    def sample(self, device: torch.device) -> list[tuple[str, torch.Tensor]]:
        sampled: list[tuple[str, torch.Tensor]] = []
        for domain, pool in self.pools.items():
            if pool.ndim != 2 or pool.shape[0] == 0:
                raise ValueError(f"Invalid auxiliary pool for {domain}")
            size = min(self.samples_per_domain, int(pool.shape[0]))
            indices = torch.randperm(
                int(pool.shape[0]), generator=self.generator
            )[:size]
            sampled.append((domain, pool[indices].to(device, non_blocking=True)))
        return sampled


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _build_alignment_batch(
    model: torch.nn.Module,
    positives: torch.Tensor,
    auxiliary_sampler: AuxiliaryPoolSampler,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_raw = model.raw_item_features(positives).float()
    raw_parts = [source_raw]
    labels = [
        torch.zeros(source_raw.shape[0], dtype=torch.long, device=device)
    ]

    for domain_index, (_, auxiliary_raw) in enumerate(
        auxiliary_sampler.sample(device), start=1
    ):
        if auxiliary_raw.shape[1] != source_raw.shape[1]:
            raise ValueError(
                "Source and auxiliary semantic embedding dimensions differ: "
                f"{source_raw.shape[1]} vs {auxiliary_raw.shape[1]}"
            )
        raw_parts.append(auxiliary_raw.float())
        labels.append(
            torch.full(
                (auxiliary_raw.shape[0],),
                domain_index,
                dtype=torch.long,
                device=device,
            )
        )

    raw = torch.cat(raw_parts, dim=0)
    domain_labels = torch.cat(labels, dim=0)
    projected = model.project_raw_for_alignment(raw)
    return projected, raw, domain_labels


def train_bert4rec_family(
    model: torch.nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    device: torch.device,
    num_items: int,
    mode: str,
    auxiliary_pools: dict[str, torch.Tensor],
    checkpoint_path: str | Path,
    training_cfg: dict,
    alignment_cfg: dict,
    evaluation_cfg: dict,
    seed: int,
    source_domain: str,
    run_name: str,
) -> dict[str, object]:
    mode = str(mode).lower()
    if mode not in {"sem", "arch0", "recg", "sage"}:
        raise ValueError(f"Unsupported mode: {mode}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(int(seed) + 17)

    requires_auxiliary = mode in {"recg", "sage"}
    if requires_auxiliary and not auxiliary_pools:
        raise ValueError(f"mode={mode} requires at least one auxiliary pool")
    auxiliary_sampler = AuxiliaryPoolSampler(
        pools=auxiliary_pools,
        samples_per_domain=int(
            alignment_cfg.get("samples_per_domain_per_step", 128)
        ),
        seed=int(seed) + 29,
    )

    epochs = int(training_cfg.get("epochs", 80))
    train_negatives = int(training_cfg.get("train_negatives", 5))
    patience_limit = int(training_cfg.get("early_stop_patience", 10))
    eval_every = int(training_cfg.get("eval_every", 1))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    max_train_batches = int(training_cfg.get("max_batches_per_epoch", 0))
    max_val_batches = int(evaluation_cfg.get("max_batches", 0))
    ranking = str(evaluation_cfg.get("ranking", "sampled"))
    eval_negatives = int(evaluation_cfg.get("eval_negatives", 100))
    tie_policy = str(evaluation_cfg.get("tie_policy", "worst"))
    ks = tuple(int(k) for k in evaluation_cfg.get("ks", [10, 20]))
    primary_metric = str(evaluation_cfg.get("primary_metric", "NDCG@10"))
    eval_seed = int(seed) + int(
        evaluation_cfg.get("fixed_negative_seed_offset", 10000)
    )
    show_progress = _resolve_progress_enabled(
        training_cfg.get("show_progress")
    )

    checkpoint_path = Path(checkpoint_path)
    best_metric = -math.inf
    best_epoch = 0
    patience = 0
    history: list[dict[str, float | int]] = []
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running = {
            "loss": 0.0,
            "bpr": 0.0,
            "align": 0.0,
            "intra": 0.0,
            "inter": 0.0,
            "sic": 0.0,
            "id": 0.0,
            "omega": 0.0,
            "delta": 0.0,
            "grad_norm": 0.0,
        }
        batches = 0

        progress_bar = None
        train_iterator = train_loader
        if show_progress and tqdm is not None:
            progress_bar = tqdm(
                train_loader,
                total=_safe_total(train_loader, max_train_batches),
                desc=f"{run_name} train {epoch:03d}/{epochs:03d}",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
                mininterval=0.5,
            )
            train_iterator = progress_bar

        try:
            for step, batch in enumerate(train_iterator, start=1):
                if max_train_batches > 0 and step > max_train_batches:
                    break

                histories = batch["history"].to(device, non_blocking=True)
                positives = batch["target"].to(device, non_blocking=True)
                negatives = _sample_training_negatives(
                    histories=histories,
                    positives=positives,
                    num_items=num_items,
                    num_negatives=train_negatives,
                    generator=train_generator,
                )
                candidates = torch.cat([positives[:, None], negatives], dim=1)

                optimizer.zero_grad(set_to_none=True)
                scores = model.score_candidates(histories, candidates)
                _assert_finite("scores", scores, f"epoch={epoch}, step={step}")
                bpr = stable_bpr_loss(scores)
                _assert_finite("bpr", bpr, f"epoch={epoch}, step={step}")

                if requires_auxiliary:
                    projected, raw, domain_labels = _build_alignment_batch(
                        model=model,
                        positives=positives,
                        auxiliary_sampler=auxiliary_sampler,
                        device=device,
                    )
                    _assert_finite(
                        "alignment projected embeddings",
                        projected,
                        f"epoch={epoch}, step={step}",
                    )
                    align_output = compute_alignment_loss(
                        mode=mode,
                        projected=projected.float(),
                        raw=raw.float(),
                        domain_labels=domain_labels,
                        config=alignment_cfg,
                    )
                else:
                    # Differentiable zero; Arch0 still trains its dual architecture
                    # through the recommendation loss.
                    projected = model.recommendation_projection.weight
                    align_output = compute_alignment_loss(
                        mode=mode,
                        projected=projected,
                        raw=projected,
                        domain_labels=torch.zeros(
                            projected.shape[0], dtype=torch.long, device=device
                        ),
                        config=alignment_cfg,
                    )

                _assert_finite(
                    "alignment loss",
                    align_output.total,
                    f"epoch={epoch}, step={step}",
                )
                loss = bpr + align_output.total
                _assert_finite("total loss", loss, f"epoch={epoch}, step={step}")
                loss.backward()

                grad_norm = clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip,
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError(
                        f"Non-finite gradient norm at epoch={epoch}, step={step}. "
                        "optimizer.step() was skipped."
                    )

                optimizer.step()

                # Catch a parameter corruption immediately, not one epoch later.
                for parameter_name, parameter in model.named_parameters():
                    if parameter.requires_grad and not torch.isfinite(parameter).all():
                        raise FloatingPointError(
                            f"Parameter {parameter_name} became non-finite at "
                            f"epoch={epoch}, step={step}"
                        )

                metrics = align_output.detached_metrics()
                running["loss"] += float(loss.detach().cpu())
                running["bpr"] += float(bpr.detach().cpu())
                for key in ("align", "intra", "inter", "sic", "id", "omega", "delta"):
                    running[key] += metrics[key]
                running["grad_norm"] += float(torch.as_tensor(grad_norm).cpu())
                batches += 1
                if progress_bar is not None:
                    progress_bar.set_postfix(
                        loss=f"{running['loss'] / batches:.4f}",
                        bpr=f"{running['bpr'] / batches:.4f}",
                        align=f"{running['align'] / batches:.4f}",
                        grad=f"{running['grad_norm'] / batches:.3f}",
                        refresh=False,
                    )

        finally:
            if progress_bar is not None:
                progress_bar.close()

        if batches == 0:
            raise RuntimeError("Training loader produced no batches")

        row: dict[str, float | int] = {
            "epoch": epoch,
            **{key: value / batches for key, value in running.items()},
        }

        if epoch % eval_every == 0:
            validation = evaluate_bert4rec_family(
                model=model,
                loader=val_loader,
                device=device,
                num_items=num_items,
                ranking=ranking,
                eval_negatives=eval_negatives,
                ks=ks,
                tie_policy=tie_policy,
                seed=eval_seed,
                max_batches=max_val_batches,
                show_progress=show_progress,
                progress_desc=f"{run_name} val {epoch:03d}",
            )
            row.update({f"val_{key}": value for key, value in validation.items()})
            current = float(validation[primary_metric])
            if not np.isfinite(current):
                raise FloatingPointError(
                    f"Primary validation metric {primary_metric} is non-finite"
                )

            improved = current > best_metric + 1e-12
            if improved:
                best_metric = current
                best_epoch = epoch
                patience = 0
                payload = {
                    "format_version": 1,
                    "run_name": run_name,
                    "mode": mode,
                    "source_domain": source_domain,
                    "seed": int(seed),
                    "best_epoch": int(best_epoch),
                    "best_metric_name": primary_metric,
                    "best_metric_value": float(best_metric),
                    "model_hparams": model.export_hparams(),
                    "model_state": model.state_dict(),
                    "training_cfg": dict(training_cfg),
                    "alignment_cfg": dict(alignment_cfg),
                    "evaluation_cfg": dict(evaluation_cfg),
                    "history": history + [row],
                }
                _atomic_torch_save(payload, checkpoint_path)
            else:
                patience += 1

            print(
                f"[{run_name}][epoch={epoch:03d}] "
                f"loss={row['loss']:.6f} bpr={row['bpr']:.6f} "
                f"align={row['align']:.6f} grad={row['grad_norm']:.4f} "
                f"val_{primary_metric}={current:.6f} "
                f"best={best_metric:.6f}@{best_epoch} "
                + ("saved" if improved else f"patience={patience}/{patience_limit}")
            )

            history.append(row)
            if patience >= patience_limit:
                print(f"[{run_name}] Early stopping at epoch {epoch}.")
                break
        else:
            history.append(row)

    if best_epoch == 0 or not checkpoint_path.exists():
        raise RuntimeError("No valid checkpoint was saved")

    return {
        "run_name": run_name,
        "mode": mode,
        "source_domain": source_domain,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_metric_name": primary_metric,
        "best_metric_value": float(best_metric),
        "checkpoint_path": str(checkpoint_path),
        "elapsed_seconds": float(time.time() - started),
        "history": history,
    }
