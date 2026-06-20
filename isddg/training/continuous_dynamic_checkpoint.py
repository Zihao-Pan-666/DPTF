from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn


def _looks_like_enhanced_checkpoint(ckpt: Dict[str, Any]) -> bool:
    """Detect the mainline semantic-conditioned continuous dynamic prior."""
    if "latent_dim" in ckpt:
        return True
    if "role_dim" in ckpt:
        return True
    protocol = ckpt.get("protocol", {})
    if isinstance(protocol, dict) and "continuous_dynamic_prior" in str(protocol.get("name", "")):
        return True
    if ckpt.get("selection_mode") in {"loss", "val_mse", "dynamic_ndcg", "dynamic_composite"}:
        return True
    if "loss_weights" in ckpt or "dynamic_feature_weights" in ckpt:
        return True
    return False


def load_continuous_dynamic_predictor_from_checkpoint(
    path: str | Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any], str]:
    """Load either the mainline enhanced prior or the legacy MSE-only prior.

    Canonical import paths:
      - enhanced/mainline: continuous_dynamic_prior_trainer.py
      - legacy MSE-only: continuous_dynamic_prior_mse_trainer.py

    Return format remains unchanged: (model, checkpoint_dict, kind).
    """
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu")

    if _looks_like_enhanced_checkpoint(ckpt):
        from isddg.training.continuous_dynamic_prior_trainer import (
            load_continuous_predictor_from_checkpoint,
        )
        model, loaded = load_continuous_predictor_from_checkpoint(path, device)
        return model, loaded, "enhanced"

    from isddg.training.continuous_dynamic_prior_mse_trainer import (
        load_continuous_predictor_from_checkpoint,
    )
    model, loaded = load_continuous_predictor_from_checkpoint(path, device)
    return model, loaded, "mse_only"


def _looks_like_enhanced_model(model: nn.Module) -> bool:
    if hasattr(model, "dynamic_head"):
        return True
    if hasattr(model, "role_head"):
        return True
    return model.__class__.__name__ == "SemanticConditionedContinuousDynamicPrior"


@torch.no_grad()
def predict_continuous_dynamic_table(
    model: nn.Module,
    item_features: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    """Predict a full item-level continuous dynamic table from item text embeddings."""
    if _looks_like_enhanced_model(model):
        from isddg.training.continuous_dynamic_prior_trainer import predict_continuous_table
        return predict_continuous_table(model, item_features, batch_size=batch_size, device=device)

    from isddg.training.continuous_dynamic_prior_mse_trainer import predict_continuous_table
    return predict_continuous_table(model, item_features, batch_size=batch_size, device=device)
