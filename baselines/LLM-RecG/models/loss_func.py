import torch
import torch.nn.functional as F


def compute_cosine_similarity(embeddings, centers):
    embeddings = F.normalize(embeddings, dim=1)
    centers = F.normalize(centers, dim=1)
    return torch.matmul(embeddings, centers.T)


def compute_diversity_entropy(embeddings, temperature=1.0):
    similarity_matrix = compute_cosine_similarity(embeddings, embeddings)
    probabilities = F.softmax(similarity_matrix / max(temperature, 1e-8), dim=1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1).mean()
    return entropy


def compute_inter_domain_entropy(sampled_embeddings, sampled_domains, domain_centers, temperature=1.0):
    similarity_matrix = compute_cosine_similarity(sampled_embeddings, domain_centers)
    mask = torch.zeros_like(similarity_matrix, device=sampled_embeddings.device)
    for i, domain_id in enumerate(sampled_domains):
        mask[i, domain_id] = 1
    similarity_matrix = similarity_matrix.masked_fill(mask.bool(), -float("inf"))
    probabilities = F.softmax(similarity_matrix / max(temperature, 1e-8), dim=1)
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1).mean()
    return entropy


def alignment_loss_with_sampled_entropy(
        sampled_embeddings,
        sampled_domains,
        num_domains,
        alpha_base,
        temperature=1.0,
        total_item_count=None,
):
    num_samples = sampled_embeddings.size(0)
    alpha = alpha_base

    # [AUTHOR-CONFIRMED]
    # The author confirmed that |N| in Eq.(14) is implemented as the sampled batch size during training,
    # not the total item count across all domains.
    beta = alpha_base * (num_samples / (num_domains ** 3))

    intra_diversity = 0
    domain_centers = []
    for domain_id in range(num_domains):
        mask = (sampled_domains == domain_id)
        if mask.sum() > 0:
            domain_embeddings = sampled_embeddings[mask]
            domain_centers.append(domain_embeddings.mean(dim=0, keepdim=True))
            diversity_entropy = compute_diversity_entropy(domain_embeddings, temperature)
            intra_diversity += diversity_entropy

    if num_domains > 0:
        intra_diversity /= num_domains

    domain_centers = torch.cat(domain_centers, dim=0)
    inter_entropy = compute_inter_domain_entropy(
        sampled_embeddings, sampled_domains, domain_centers, temperature
    )

    # 彻底回归作者原版的组合方式，解除空间崩塌
    # return -alpha * intra_diversity + beta * inter_entropy
    loss = -alpha * intra_diversity + beta * inter_entropy
    return loss, intra_diversity.detach(), inter_entropy.detach(), torch.tensor(beta, device=sampled_embeddings.device)


def alignment_loss(*args, **kwargs):
    return alignment_loss_with_sampled_entropy(*args, **kwargs)