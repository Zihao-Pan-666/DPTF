from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from isddg.data.dataset import PrefixDataset, collate_prefix
from isddg.evaluation.dynamic_signal_evaluator import evaluate_dynamic_signal


class SemanticConditionedContinuousDynamicPrior(nn.Module):
    """Semantic-conditioned continuous dynamic prior learner.

    This is the formal mainline trainer, replacing the old v2 filename. It
    predicts a low-dimensional continuous dynamic vector from frozen item text
    embeddings. Optionally, it also predicts the soft dynamic-role distribution
    as an auxiliary structured target.

    The model does not use target-domain interactions. At target inference time,
    the same semantic -> dynamic mapper is applied to target item text embeddings
    only.
    """

    def __init__(
        self,
        input_dim: int,
        dynamic_dim: int,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        dropout: float = 0.1,
        role_dim: int = 0,
        use_layer_norm: bool = True,
        l2_normalize_input: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.dynamic_dim = int(dynamic_dim)
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)
        self.role_dim = int(role_dim)
        self.l2_normalize_input = bool(l2_normalize_input)

        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim, latent_dim), nn.GELU()])
        if use_layer_norm:
            layers.append(nn.LayerNorm(latent_dim))
        layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*layers)

        self.dynamic_head = nn.Linear(latent_dim, dynamic_dim)
        self.role_head = nn.Linear(latent_dim, role_dim) if role_dim > 0 else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.l2_normalize_input:
            x = F.normalize(x, p=2, dim=-1)
        return self.trunk(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(x)
        out = {"dynamic": self.dynamic_head(z)}
        if self.role_head is not None:
            out["role_logits"] = self.role_head(z)
            out["role_log_probs"] = F.log_softmax(out["role_logits"], dim=-1)
            out["role_probs"] = out["role_log_probs"].exp()
        return out


def _as_feature_weights(weights: Optional[Sequence[float]], dim: int, device: torch.device) -> torch.Tensor:
    if weights is None:
        return torch.ones(dim, dtype=torch.float32, device=device)
    if len(weights) != dim:
        raise ValueError(f"dynamic feature weights length mismatch: got {len(weights)}, expected {dim}")
    w = torch.tensor([float(x) for x in weights], dtype=torch.float32, device=device)
    return w / w.mean().clamp(min=1e-8)


def weighted_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    feature_weights: torch.Tensor,
    loss_type: str = "smooth_l1",
    smooth_l1_beta: float = 1.0,
) -> torch.Tensor:
    if loss_type == "mse":
        raw = (pred - target).pow(2)
    elif loss_type == "l1":
        raw = (pred - target).abs()
    elif loss_type == "smooth_l1":
        raw = F.smooth_l1_loss(pred, target, beta=float(smooth_l1_beta), reduction="none")
    else:
        raise ValueError(f"Unknown regression loss_type={loss_type}")
    return (raw * feature_weights.view(1, -1)).mean()


def role_kl_loss(role_log_probs: torch.Tensor, target_role: torch.Tensor) -> torch.Tensor:
    target = target_role.float().clamp(min=1e-8)
    target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return F.kl_div(role_log_probs, target, reduction="batchmean")


@torch.no_grad()
def _sample_training_negatives_fast(
    history: torch.Tensor,
    pos_items: torch.Tensor,
    num_items: int,
    num_negatives: int,
    device: torch.device,
    max_rounds: int = 16,
) -> torch.Tensor:
    batch_size, _ = history.size()
    neg_items = torch.randint(1, int(num_items) + 1, size=(batch_size, int(num_negatives)), device=device)
    blocked = torch.cat([history, pos_items.unsqueeze(1)], dim=1)

    for _ in range(max_rounds):
        invalid = (neg_items.unsqueeze(-1) == blocked.unsqueeze(1)).any(dim=-1)
        if not invalid.any():
            break
        neg_items[invalid] = torch.randint(1, int(num_items) + 1, size=(int(invalid.sum().item()),), device=device)

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


def _gather_predicted_vectors(
    model: SemanticConditionedContinuousDynamicPrior,
    item_features: torch.Tensor,
    item_ids: torch.Tensor,
) -> torch.Tensor:
    original_shape = item_ids.shape
    flat_ids = item_ids.reshape(-1)
    unique_ids, inverse = torch.unique(flat_ids, sorted=True, return_inverse=True)
    pred_unique = model(item_features[unique_ids])["dynamic"]
    if unique_ids.numel() > 0 and int(unique_ids[0].item()) == 0:
        pred_unique[0] = 0.0
    return pred_unique[inverse].reshape(*original_shape, pred_unique.size(-1))


def _history_state_from_predicted(
    history_dyn: torch.Tensor,
    history_ids: torch.Tensor,
    pooling: str = "decay",
    recent_k: Optional[int] = 5,
    decay: float = 0.8,
) -> torch.Tensor:
    valid = history_ids.ne(0)
    batch_size, seq_len = history_ids.shape
    lengths = valid.sum(dim=1).clamp(min=1)

    if pooling == "last":
        idx = (lengths - 1).view(batch_size, 1, 1).expand(batch_size, 1, history_dyn.size(-1))
        return history_dyn.gather(dim=1, index=idx).squeeze(1)

    if pooling == "mean":
        weights = valid.float()
    elif pooling == "recent":
        k = seq_len if recent_k is None or recent_k <= 0 else int(recent_k)
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
        raise ValueError(f"Unknown pooling={pooling}. Use mean, recent, decay, or last.")

    denom = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return (history_dyn * weights.unsqueeze(-1)).sum(dim=1) / denom


def dynamic_bpr_loss_from_batch(
    model: SemanticConditionedContinuousDynamicPrior,
    item_features: torch.Tensor,
    history: torch.Tensor,
    pos_items: torch.Tensor,
    neg_items: torch.Tensor,
    pooling: str = "decay",
    recent_k: Optional[int] = 5,
    decay: float = 0.8,
) -> torch.Tensor:
    history_dyn = _gather_predicted_vectors(model, item_features, history)
    pos_dyn = _gather_predicted_vectors(model, item_features, pos_items)
    neg_dyn = _gather_predicted_vectors(model, item_features, neg_items)
    state = _history_state_from_predicted(history_dyn, history, pooling=pooling, recent_k=recent_k, decay=decay)
    pos_scores = torch.einsum("bd,bd->b", state, pos_dyn)
    neg_scores = torch.einsum("bd,bnd->bn", state, neg_dyn)
    return -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()


def _make_source_val_loader(
    val_samples: Optional[List[dict]],
    num_items: int,
    max_len: int,
    batch_size: int,
    num_workers: int,
) -> Optional[DataLoader]:
    if not val_samples:
        return None
    return DataLoader(
        PrefixDataset(val_samples, num_items=int(num_items), max_len=max_len),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_prefix,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def predict_continuous_table(
    model: nn.Module,
    item_features: torch.Tensor,
    batch_size: int = 4096,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    outs: List[torch.Tensor] = []
    for start in range(0, item_features.size(0), batch_size):
        out = model(item_features[start : start + batch_size].to(device))
        if isinstance(out, dict):
            out = out["dynamic"]
        outs.append(out.detach().cpu())
    table = torch.cat(outs, dim=0)
    table[0] = 0.0
    return table


predict_continuous_table_v2 = predict_continuous_table


def _mse_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    err = y_pred - y_true
    return {"mse": float(err.pow(2).mean().item()), "mae": float(err.abs().mean().item())}


def _selection_value(
    row: Dict[str, Any],
    mode: str,
    recall_weight: float = 0.25,
    ndcg_key: str = "rank_NDCG@10",
    recall_key: str = "rank_Recall@10",
) -> float:
    if mode == "loss":
        return -float(row["val_total_loss"])
    if mode == "val_mse":
        return -float(row["val_reg_mse"])
    if mode == "dynamic_ndcg":
        return float(row.get(ndcg_key, -1e9))
    if mode == "dynamic_composite":
        return float(row.get(ndcg_key, 0.0)) + float(recall_weight) * float(row.get(recall_key, 0.0))
    raise ValueError(f"Unknown selection_mode={mode}")


def train_continuous_dynamic_prior(
    item_features: torch.Tensor,
    target_table: torch.Tensor,
    checkpoint_path: str | Path,
    role_table: Optional[torch.Tensor] = None,
    source_train_samples: Optional[List[dict]] = None,
    source_val_samples: Optional[List[dict]] = None,
    num_items: Optional[int] = None,
    max_len: int = 50,
    hidden_dim: int = 256,
    latent_dim: int = 128,
    dropout: float = 0.1,
    use_layer_norm: bool = True,
    l2_normalize_input: bool = False,
    epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    batch_size: int = 4096,
    sequence_batch_size: int = 512,
    val_ratio: float = 0.1,
    patience: int = 10,
    seed: int = 2026,
    device: Optional[torch.device] = None,
    regression_loss_type: str = "smooth_l1",
    smooth_l1_beta: float = 1.0,
    dynamic_feature_weights: Optional[Sequence[float]] = None,
    lambda_reg: float = 1.0,
    lambda_role: float = 0.2,
    lambda_bpr: float = 0.1,
    bpr_negatives: int = 20,
    bpr_pooling: str = "decay",
    bpr_recent_k: int = 5,
    bpr_decay: float = 0.8,
    source_val_ranking: bool = True,
    val_ranking_negatives: int = 100,
    val_ranking_batch_size: int = 256,
    val_ranking_mode: str = "sampled",
    val_score_pooling: str = "decay",
    val_score_recent_k: int = 5,
    val_score_decay: float = 0.8,
    val_score_norm: str = "zscore",
    selection_mode: str = "dynamic_composite",
    recall_weight: float = 0.25,
    grad_clip: float = 5.0,
    checkpoint_extra: Optional[Dict[str, Any]] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    x_cpu = item_features.float()
    y_cpu = target_table.float()
    if x_cpu.size(0) != y_cpu.size(0):
        raise ValueError(f"item_features and target_table row mismatch: {x_cpu.size(0)} vs {y_cpu.size(0)}")

    role_cpu = None
    role_dim = 0
    if role_table is not None:
        role_cpu = role_table.float()
        if role_cpu.size(0) != x_cpu.size(0):
            raise ValueError(f"role_table row mismatch: {role_cpu.size(0)} vs {x_cpu.size(0)}")
        role_dim = int(role_cpu.size(1))

    valid = torch.arange(1, x_cpu.size(0))
    generator = torch.Generator().manual_seed(seed)
    perm = valid[torch.randperm(valid.numel(), generator=generator)]
    n_val = max(1, int(len(perm) * float(val_ratio)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = SemanticConditionedContinuousDynamicPrior(
        input_dim=x_cpu.size(1),
        dynamic_dim=y_cpu.size(1),
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
        role_dim=role_dim if lambda_role > 0 and role_cpu is not None else 0,
        use_layer_norm=use_layer_norm,
        l2_normalize_input=l2_normalize_input,
    ).to(device)

    x = x_cpu.to(device)
    y = y_cpu.to(device)
    role = role_cpu.to(device) if role_cpu is not None else None
    feature_weights = _as_feature_weights(dynamic_feature_weights, y.size(1), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    seq_loader = None
    if lambda_bpr > 0:
        if not source_train_samples:
            raise ValueError("lambda_bpr > 0 requires source_train_samples.")
        if num_items is None:
            num_items = x_cpu.size(0) - 1
        seq_loader = DataLoader(
            PrefixDataset(source_train_samples, num_items=int(num_items), max_len=max_len),
            batch_size=sequence_batch_size,
            shuffle=True,
            collate_fn=collate_prefix,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = (
        _make_source_val_loader(
            source_val_samples,
            num_items=int(num_items or x_cpu.size(0) - 1),
            max_len=max_len,
            batch_size=val_ranking_batch_size,
            num_workers=0,
        )
        if source_val_ranking
        else None
    )

    best_value = -float("inf")
    best_epoch = -1
    bad = 0
    history: List[Dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
        seq_iter = iter(seq_loader) if seq_loader is not None else None
        total_reg = total_role = total_bpr = total_loss = 0.0
        total_n = 0

        train_iter = range(0, train_idx.numel(), batch_size)
        if show_progress:
            train_iter = tqdm(
                train_iter,
                desc=f"cont-dyn-prior epoch {epoch:03d}",
                leave=False,
                ascii=True,
                dynamic_ncols=True,
                mininterval=2.0,
                maxinterval=10.0,
            )

        for start in train_iter:
            idx = train_idx[start : start + batch_size].to(device)
            out = model(x[idx])
            reg_loss = weighted_regression_loss(out["dynamic"], y[idx], feature_weights, loss_type=regression_loss_type, smooth_l1_beta=smooth_l1_beta)

            role_loss = torch.tensor(0.0, device=device)
            if lambda_role > 0 and role is not None and "role_log_probs" in out:
                role_loss = role_kl_loss(out["role_log_probs"], role[idx])

            bpr_loss = torch.tensor(0.0, device=device)
            if lambda_bpr > 0 and seq_iter is not None:
                try:
                    seq_batch = next(seq_iter)
                except StopIteration:
                    seq_iter = iter(seq_loader)
                    seq_batch = next(seq_iter)
                hist = seq_batch["history"].to(device, non_blocking=True)
                pos = seq_batch["target"].to(device, non_blocking=True)
                neg = _sample_training_negatives_fast(hist, pos, int(num_items or x_cpu.size(0) - 1), int(bpr_negatives), device)
                bpr_loss = dynamic_bpr_loss_from_batch(model, x, hist, pos, neg, pooling=bpr_pooling, recent_k=bpr_recent_k, decay=bpr_decay)

            loss = float(lambda_reg) * reg_loss + float(lambda_role) * role_loss + float(lambda_bpr) * bpr_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            n = int(idx.numel())
            total_n += n
            total_reg += float(reg_loss.item()) * n
            total_role += float(role_loss.item()) * n
            total_bpr += float(bpr_loss.item()) * n
            total_loss += float(loss.item()) * n
            if show_progress and hasattr(train_iter, "set_postfix"):
                train_iter.set_postfix(loss=f"{total_loss / max(total_n, 1):.6f}")

        model.eval()
        val_preds: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, val_idx.numel(), batch_size):
                idx = val_idx[start : start + batch_size].to(device)
                val_preds.append(model(x[idx])["dynamic"].detach().cpu())
        pred_val = torch.cat(val_preds, dim=0)
        val_reg = _mse_mae(y_cpu[val_idx], pred_val)

        val_role_loss = None
        if role_cpu is not None and model.role_head is not None:
            with torch.no_grad():
                role_preds: List[torch.Tensor] = []
                for start in range(0, val_idx.numel(), batch_size):
                    idx = val_idx[start : start + batch_size].to(device)
                    role_preds.append(model(x[idx])["role_log_probs"].detach().cpu())
                role_log_probs = torch.cat(role_preds, dim=0)
                val_role_loss = float(role_kl_loss(role_log_probs, role_cpu[val_idx]).item())

        row: Dict[str, Any] = {
            "epoch": epoch,
            "train_total_loss": total_loss / max(total_n, 1),
            "train_reg_loss": total_reg / max(total_n, 1),
            "train_role_loss": total_role / max(total_n, 1),
            "train_bpr_loss": total_bpr / max(total_n, 1),
            "val_reg_mse": val_reg["mse"],
            "val_reg_mae": val_reg["mae"],
            "val_role_kl": val_role_loss if val_role_loss is not None else "",
        }
        row["val_total_loss"] = float(lambda_reg) * row["val_reg_mse"] + (float(lambda_role) * float(val_role_loss) if val_role_loss is not None else 0.0)

        if val_loader is not None:
            pred_table = predict_continuous_table(model, x_cpu, batch_size=4096, device=device)
            rank_metrics = evaluate_dynamic_signal(
                model=None,
                feature_table=pred_table,
                loader=val_loader,
                num_items=int(num_items or x_cpu.size(0) - 1),
                device=device,
                beta=1.0,
                semantic_weight=0.0,
                ranking_mode=val_ranking_mode,
                num_negatives=val_ranking_negatives,
                seed=seed + epoch,
                pooling=val_score_pooling,
                recent_k=val_score_recent_k,
                decay=val_score_decay,
                score_norm=val_score_norm,
                desc=f"pred-dyn-only source val epoch {epoch:03d}",
                show_progress=show_progress,
            )
            row.update({f"rank_{k}": v for k, v in rank_metrics.items()})

        current = _selection_value(row, selection_mode, recall_weight=recall_weight)
        row["selection_mode"] = selection_mode
        row["selection_value"] = current
        history.append(row)

        if current > best_value:
            best_value = current
            best_epoch = epoch
            bad = 0
            payload = {
                "model_state": model.state_dict(),
                "input_dim": int(x_cpu.size(1)),
                "output_dim": int(y_cpu.size(1)),
                "role_dim": int(role_dim),
                "hidden_dim": int(hidden_dim),
                "latent_dim": int(latent_dim),
                "dropout": float(dropout),
                "use_layer_norm": bool(use_layer_norm),
                "l2_normalize_input": bool(l2_normalize_input),
                "best_epoch": int(best_epoch),
                "best_selection_value": float(best_value),
                "selection_mode": selection_mode,
                "history": history,
                "loss_weights": {
                    "lambda_reg": float(lambda_reg),
                    "lambda_role": float(lambda_role),
                    "lambda_bpr": float(lambda_bpr),
                },
                "dynamic_feature_weights": list(dynamic_feature_weights) if dynamic_feature_weights is not None else None,
                "protocol": {
                    "name": "continuous_dynamic_prior",
                    "trainer": "semantic_conditioned_mainline",
                    "checkpoint_format": "enhanced",
                },
                "total_elapsed_sec": time.time() - t0,
            }
            if checkpoint_extra:
                payload.update(checkpoint_extra)
            torch.save(payload, checkpoint_path)
            status = "saved"
        else:
            bad += 1
            status = f"patience={bad}/{patience}"

        msg = (
            f"[ContDynPrior][epoch={epoch:03d}] "
            f"train_loss={row['train_total_loss']:.6f} "
            f"val_mse={row['val_reg_mse']:.6f} "
            f"select={current:.6f} best={best_value:.6f}@{best_epoch} "
        )
        if "rank_NDCG@10" in row:
            msg += f"rank_R@10={row['rank_Recall@10']:.6f} rank_N@10={row['rank_NDCG@10']:.6f} "
        print(msg + status, flush=True)

        if bad >= patience:
            break

    return {
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": int(best_epoch),
        "best_selection_value": float(best_value),
        "selection_mode": selection_mode,
        "history": history,
        "total_elapsed_sec": time.time() - t0,
    }


train_continuous_dynamic_prior_v2 = train_continuous_dynamic_prior


def load_continuous_predictor_from_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    model = SemanticConditionedContinuousDynamicPrior(
        input_dim=int(ckpt["input_dim"]),
        dynamic_dim=int(ckpt["output_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        latent_dim=int(ckpt.get("latent_dim", 128)),
        dropout=float(ckpt.get("dropout", 0.1)),
        role_dim=int(ckpt.get("role_dim", 0)),
        use_layer_norm=bool(ckpt.get("use_layer_norm", True)),
        l2_normalize_input=bool(ckpt.get("l2_normalize_input", False)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt

# Backward-compatible aliases for older scripts/checkpoints.
load_continuous_v2_from_checkpoint = load_continuous_predictor_from_checkpoint
load_continuous_from_checkpoint = load_continuous_predictor_from_checkpoint
