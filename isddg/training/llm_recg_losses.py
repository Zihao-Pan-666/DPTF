from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def bpr_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    """Official-style BPR loss for one positive and K negatives."""
    if pos_logits.dim() == 1 and neg_logits.dim() == 2:
        pos_logits = pos_logits.unsqueeze(1).expand_as(neg_logits)
    return -torch.mean(torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-10))


def compute_cosine_similarity(embeddings: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    embeddings = F.normalize(embeddings, dim=1)
    centers = F.normalize(centers, dim=1)
    return embeddings @ centers.t()


def compute_entropy_from_similarity(similarity_matrix: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    # The official LLM-RecG code multiplies similarities by temperature.
    probabilities = F.softmax(similarity_matrix * float(temperature), dim=1)
    return -(probabilities * torch.log(probabilities + 1e-10)).sum(dim=1).mean()


def compute_diversity_entropy(embeddings: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if embeddings.size(0) <= 1:
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)
    sim = compute_cosine_similarity(embeddings, embeddings)
    return compute_entropy_from_similarity(sim, temperature=temperature)


def compute_inter_domain_entropy(
    sampled_embeddings: torch.Tensor,
    sampled_domains: torch.Tensor,
    domain_centers: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    if domain_centers.size(0) <= 1:
        return torch.zeros((), device=sampled_embeddings.device, dtype=sampled_embeddings.dtype)

    sim = compute_cosine_similarity(sampled_embeddings, domain_centers)
    mask = torch.zeros_like(sim, dtype=torch.bool)
    sampled_domains = sampled_domains.long()

    # Domain ids used here must be contiguous 0..num_domains-1.
    for i, d in enumerate(sampled_domains.tolist()):
        if 0 <= int(d) < sim.size(1):
            mask[i, int(d)] = True

    sim = sim.masked_fill(mask, -float("inf"))
    probs = F.softmax(sim * float(temperature), dim=1)
    probs = probs.masked_fill(mask, 0.0)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
    return entropy


def alignment_loss_with_sampled_entropy(
    sampled_embeddings: torch.Tensor,
    sampled_domains: torch.Tensor,
    num_domains: int,
    alpha_base: float = 0.001,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Official LLM-RecG item-level generalization loss.

    L_gen = - alpha * H_intra + beta * H_inter
    beta = alpha * (N / |D|^3)

    `sampled_embeddings` are already projected by the domain-alignment head.
    """
    device = sampled_embeddings.device
    dtype = sampled_embeddings.dtype
    sampled_domains = sampled_domains.to(device).long()
    num_domains = int(num_domains)

    if sampled_embeddings.numel() == 0 or sampled_domains.numel() == 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, {
            "align_loss": zero.detach(),
            "intra_entropy": zero.detach(),
            "inter_entropy": zero.detach(),
            "alpha": zero.detach(),
            "beta": zero.detach(),
            "num_align_items": zero.detach(),
        }

    alpha = torch.tensor(float(alpha_base), device=device, dtype=dtype)
    beta = alpha * (float(sampled_embeddings.size(0)) / max(float(num_domains ** 3), 1.0))

    intra_diversity = torch.zeros((), device=device, dtype=dtype)
    centers = []
    present_domains = 0

    for d in range(num_domains):
        mask = sampled_domains.eq(d)
        if mask.any():
            domain_emb = sampled_embeddings[mask]
            centers.append(domain_emb.mean(dim=0, keepdim=True))
            intra_diversity = intra_diversity + compute_diversity_entropy(domain_emb, temperature=temperature)
            present_domains += 1

    if not centers:
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, {
            "align_loss": zero.detach(),
            "intra_entropy": zero.detach(),
            "inter_entropy": zero.detach(),
            "alpha": alpha.detach(),
            "beta": beta.detach(),
            "num_align_items": torch.tensor(float(sampled_embeddings.size(0)), device=device),
        }

    # Official code divides by num_domains, assuming all domains are present.
    intra_diversity = intra_diversity / max(float(num_domains), 1.0)
    domain_centers = torch.cat(centers, dim=0)

    # If all domains are present, sampled_domains is already aligned to center index.
    # If any domain is absent, remap ids to center positions.
    if present_domains != num_domains:
        old_to_new = {}
        new_idx = 0
        remapped = torch.empty_like(sampled_domains)
        for d in range(num_domains):
            mask = sampled_domains.eq(d)
            if mask.any():
                old_to_new[d] = new_idx
                remapped[mask] = new_idx
                new_idx += 1
        sampled_domains_for_centers = remapped
    else:
        sampled_domains_for_centers = sampled_domains

    inter_entropy = compute_inter_domain_entropy(
        sampled_embeddings=sampled_embeddings,
        sampled_domains=sampled_domains_for_centers,
        domain_centers=domain_centers,
        temperature=temperature,
    )

    align_loss = -alpha * intra_diversity + beta * inter_entropy
    return align_loss, {
        "align_loss": align_loss.detach(),
        "intra_entropy": intra_diversity.detach(),
        "inter_entropy": inter_entropy.detach(),
        "alpha": alpha.detach(),
        "beta": beta.detach(),
        "num_align_items": torch.tensor(float(sampled_embeddings.size(0)), device=device),
    }


# Backward-compatible alias.
def recg_entropy_alignment_loss(
    embeddings: torch.Tensor,
    domain_ids: torch.Tensor,
    num_domains: int,
    alpha: float = 0.001,
    temperature: float = 1.0,
):
    return alignment_loss_with_sampled_entropy(
        sampled_embeddings=embeddings,
        sampled_domains=domain_ids,
        num_domains=num_domains,
        alpha_base=alpha,
        temperature=temperature,
    )
