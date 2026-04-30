import torch
import torch.nn.functional as F
import math


def compute_cosine_similarity(x, y, eps=1e-8):
    """计算余弦相似度矩阵 [N, M]"""
    x_n = F.normalize(x, p=2, dim=1, eps=eps)
    y_n = F.normalize(y, p=2, dim=1, eps=eps)
    return torch.matmul(x_n, y_n.T)


def unified_alignment_loss(
        mode,  # ["sem", "recg", "sage", "sage_no_sa", "sage_no_id", "sage_no_adagen"]
        projected_embeddings,
        raw_embeddings,
        domain_labels,
        num_domains,
        source_domain_id,
        lambda_g=0.001,
        gamma_g=1.0,
        beta_id=0.01,
        tau=0.1
):
    device = projected_embeddings.device

    # === 1. SEM 基线：无任何对齐约束 ===
    if mode == "sem":
        zero_tensor = torch.tensor(0.0, device=device)
        return torch.tensor(0.0, device=device, requires_grad=True), zero_tensor, zero_tensor, lambda_g

    # === 2. RECG 基线：完全复刻 old 版本的交叉熵逻辑 ===
    if mode == "recg":
        num_samples = projected_embeddings.size(0)
        # old 版本中的 beta 权重计算
        beta = lambda_g * (num_samples / (max(num_domains, 1) ** 3))

        # 2.1 Intra-domain Diversity (域内多样性)
        intra_diversity = torch.tensor(0.0, device=device)
        domain_centers_proj = []
        for domain_id in range(num_domains):
            _mask = (domain_labels == domain_id)
            if _mask.sum() > 0:
                domain_proj = projected_embeddings[_mask]
                domain_centers_proj.append(domain_proj.mean(dim=0, keepdim=True))
                # 相似度熵
                sim_matrix = compute_cosine_similarity(domain_proj, domain_proj)
                probs = F.softmax(sim_matrix / tau, dim=1)
                intra_diversity += -torch.sum(probs * torch.log(probs + 1e-10), dim=1).mean()
            else:
                domain_centers_proj.append(torch.zeros((1, projected_embeddings.size(1)), device=device))

        if num_domains > 0:
            intra_diversity /= num_domains
            domain_centers_proj = torch.cat(domain_centers_proj, dim=0)

        # 2.2 Inter-domain Compactness (跨域紧凑性 standard_inter_entropy)
        sim_matrix_proj = compute_cosine_similarity(projected_embeddings, domain_centers_proj)
        inter_id_mask = torch.zeros_like(sim_matrix_proj, device=device)
        for i, domain_id in enumerate(domain_labels):
            inter_id_mask[i, int(domain_id.item())] = 1

        sim_matrix_proj = sim_matrix_proj.masked_fill(inter_id_mask.bool(), -float("inf"))
        probs_inter = F.softmax(sim_matrix_proj / tau, dim=1)
        base_inter_components = - (probs_inter * torch.log(probs_inter + 1e-10))
        standard_inter_entropy = torch.sum(base_inter_components, dim=1).mean()

        # old logic: loss = -alpha_base * intra_diversity + beta * standard_inter_entropy
        l_gen = -lambda_g * intra_diversity + beta * standard_inter_entropy
        return l_gen, standard_inter_entropy.detach(), intra_diversity.detach(), lambda_g

    # === 3. SAGE 及其消融变体（二进制 Source vs Target 对齐） ===
    source_mask = (domain_labels == source_domain_id)
    target_mask = ~source_mask
    z_s, z_t = projected_embeddings[source_mask], projected_embeddings[target_mask]
    e_raw_s, e_raw_t = raw_embeddings[source_mask], raw_embeddings[target_mask]

    # --- 3.1 Intra-domain Diversity (L_ID) ---
    l_id = torch.tensor(0.0, device=device)
    if mode != "sage_no_id":
        for domain_z in [z_s, z_t]:
            if domain_z.size(0) > 1:
                sim_matrix = compute_cosine_similarity(domain_z, domain_z) / tau
                p_ij = F.softmax(sim_matrix, dim=1)
                l_id += -torch.sum(p_ij * torch.log(p_ij + 1e-10), dim=1).mean()
        l_id = l_id / 2.0

    # --- 3.2 Inter-domain Compactness (L_SIC) ---
    l_sic = torch.tensor(0.0, device=device)
    if z_s.size(0) > 0 and z_t.size(0) > 0:
        z_s_norm = F.normalize(z_s, p=2, dim=1)
        z_t_norm = F.normalize(z_t, p=2, dim=1)
        dist_sq = 2.0 - 2.0 * torch.matmul(z_s_norm, z_t_norm.T)
        dist_sq = F.relu(dist_sq)

        if mode == "sage_no_sa":
            l_sic = dist_sq.mean()
        else:
            w_ij = F.relu(compute_cosine_similarity(e_raw_s, e_raw_t))
            l_sic = torch.sum(w_ij * dist_sq) / (torch.sum(w_ij) + 1e-8)

    # --- 3.3 Domain-Adaptive Weighting (omega) ---
    if mode == "sage_no_adagen":
        omega = lambda_g
    else:
        e_bar_s = e_raw_s.mean(dim=0, keepdim=True) if e_raw_s.size(0) > 0 else torch.zeros(1, raw_embeddings.size(1),
                                                                                            device=device)
        e_bar_t = e_raw_t.mean(dim=0, keepdim=True) if e_raw_t.size(0) > 0 else torch.zeros(1, raw_embeddings.size(1),
                                                                                            device=device)

        e_bar_s = F.normalize(e_bar_s, p=2, dim=1)
        e_bar_t = F.normalize(e_bar_t, p=2, dim=1)
        delta = torch.norm(e_bar_s - e_bar_t, p=2)
        omega = lambda_g * math.exp(-gamma_g * delta.item())

    # --- 3.4 Final Gen Loss ---
    l_gen = l_sic + beta_id * l_id
    return l_gen, l_sic.detach(), l_id.detach(), omega