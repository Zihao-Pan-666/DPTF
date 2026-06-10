from __future__ import annotations

import torch
import torch.nn.functional as F


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    # pos: [B], neg: [B,N]
    return -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()


def kl_role_loss(target_dist: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
    target = target_dist.clamp(min=1e-8)
    pred = pred_dist.clamp(min=1e-8)
    return (target * (target.log() - pred.log())).sum(dim=-1).mean()
