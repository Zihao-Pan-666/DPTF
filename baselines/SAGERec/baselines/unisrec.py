import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

# ==========================================
# 从官方原代码库中补充的 PWLayer
# ==========================================
class PWLayer(nn.Module):
    """Single Parametric Whitening Layer"""
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)
# ==========================================


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor from UniSRec"""
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()
        self.n_exps = n_exps
        self.noisy_gating = noise
        self.experts = nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for _ in range(n_exps)])
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (..., n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(..., 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class UniSRecWithDomainAlignment(nn.Module):
    def __init__(self, hidden_units, max_seq_length, num_heads, num_layers, dropout_rate,
                 pretrained_item_embeddings, num_sequential_patterns=10, n_exps=8):
        super().__init__()
        self.hidden_units = hidden_units
        self.max_seq_length = max_seq_length
        self.num_sequential_patterns = num_sequential_patterns
        self.pretrained_dim = pretrained_item_embeddings.shape[1]

        self.pretrained_item_embedding = nn.Embedding.from_pretrained(
            pretrained_item_embeddings.float(), freeze=True, padding_idx=0
        )

        # 【UniSRec MoE 解耦】
        self.projection_layer = MoEAdaptorLayer(n_exps, [self.pretrained_dim, hidden_units], dropout_rate)
        self.domain_alignment_projection_layer = MoEAdaptorLayer(n_exps, [self.pretrained_dim, hidden_units],
                                                                 dropout_rate)

        self.merge_layer = nn.Linear(hidden_units * 2, hidden_units)
        self.pos_embedding = nn.Embedding(max_seq_length, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_units, nhead=num_heads,
                                       dim_feedforward=hidden_units * 4, dropout=dropout_rate,
                                       activation="gelu",
                                       batch_first=True) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-6)
        self.sequential_patterns = None
        self.pattern_fusion_layer = nn.Linear(hidden_units * 2, hidden_units)
        self.use_sequential_patterns = False

    def encode_sequence(self, item_seq: torch.Tensor, is_sem_baseline: bool = False) -> torch.Tensor:
        device = item_seq.device
        batch_size, seq_len = item_seq.size()
        raw_emb = self.pretrained_item_embedding(item_seq)

        # 【同步 BERT4Rec 修复】：掩码处理
        valid_mask = (item_seq != 0).float().unsqueeze(-1)

        emb_rec = self.projection_layer(raw_emb) * valid_mask

        if is_sem_baseline:
            seq_emb = emb_rec
        else:
            emb_irm = self.domain_alignment_projection_layer(raw_emb) * valid_mask
            seq_emb = self.merge_layer(torch.cat((emb_rec, emb_irm), dim=-1)) * valid_mask

        # UniSRec 建议使用 Causal Mask 以匹配 SASRec 风格
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        seq_emb = self.dropout(seq_emb + self.pos_embedding(positions))

        out = seq_emb
        for layer in self.attention_layers:
            out = layer(out, src_key_padding_mask=(item_seq == 0))

        out = self.layer_norm(out)
        out = out * valid_mask  # 【同步 BERT4Rec 修复】

        user_rep = out[:, -1, :]
        return user_rep

    def forward(self, item_seq: torch.LongTensor, is_target_domain: bool = False,
                is_sem_baseline: bool = False) -> torch.Tensor:
        user_rep = self.encode_sequence(item_seq, is_sem_baseline=is_sem_baseline)

        if is_target_domain and not is_sem_baseline and self.use_sequential_patterns and self.sequential_patterns is not None:
            attended_patterns = self._apply_sequential_pattern_attention(user_rep)
            user_rep = self.pattern_fusion_layer(torch.cat([user_rep, attended_patterns], dim=1))

        # 【核心逻辑】：非对称打分，通过 MoE 映射 Embedding 矩阵
        all_item_emb_rec = self.projection_layer(self.pretrained_item_embedding.weight)
        logits = torch.matmul(user_rep, all_item_emb_rec.T)
        logits[:, 0] = -1e9
        return logits

    def _apply_sequential_pattern_attention(self, user_rep):
        user_norm = F.normalize(user_rep, p=2, dim=1)
        pattern_norm = F.normalize(self.sequential_patterns, p=2, dim=1)
        attention = F.softmax(torch.matmul(user_norm, pattern_norm.T), dim=1)
        return torch.matmul(attention, self.sequential_patterns)

    def extract_sequential_patterns_from_source(self, source_sequences: torch.Tensor, batch_size=512):
        self.eval()
        device = next(self.parameters()).device
        embs = []
        with torch.no_grad():
            for i in range(0, source_sequences.size(0), batch_size):
                batch_seq = source_sequences[i:i + batch_size].to(device)
                # 复用 Merge 逻辑提取表征
                raw_emb = self.pretrained_item_embedding(batch_seq)
                merged = self.merge_layer(
                    torch.cat((self.projection_layer(raw_emb), self.domain_alignment_projection_layer(raw_emb)),
                              dim=-1))
                out = merged + self.pos_embedding(torch.arange(batch_seq.size(1), device=device))
                for layer in self.attention_layers:
                    out = layer(out, src_key_padding_mask=(batch_seq == 0))
                embs.append(self.layer_norm(out[:, -1, :]).cpu())
        kmeans = KMeans(n_clusters=self.num_sequential_patterns, random_state=42).fit(torch.cat(embs).numpy())
        self.sequential_patterns = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        self.use_sequential_patterns = True
        return self.sequential_patterns

    def predict(self, item_seq, candidate_items=None, is_target_domain=False, is_sem_baseline=False):
        logits = self.forward(item_seq, is_target_domain=is_target_domain, is_sem_baseline=is_sem_baseline)
        if candidate_items is not None:
            return torch.gather(logits, dim=1, index=candidate_items)
        return logits

    def project_items_for_alignment(self, item_ids: torch.Tensor) -> torch.Tensor:
        # 对齐损失专门使用此层提取特征
        return self.domain_alignment_projection_layer(self.pretrained_item_embedding(item_ids))

    def project_raw_for_alignment(self, raw_embeddings: torch.Tensor) -> torch.Tensor:
        return self.domain_alignment_projection_layer(raw_embeddings)

    def load_new_pretrain_embeddings(self, pretrained_item_embeddings: torch.Tensor):
        device = next(self.parameters()).device
        pretrained_item_embeddings = pretrained_item_embeddings.clone().float()
        pretrained_item_embeddings[0] = 0.0
        self.pretrained_item_embedding = nn.Embedding.from_pretrained(
            pretrained_item_embeddings.to(device), freeze=True, padding_idx=0,
        )

    def get_raw_item_embeddings(self, item_ids):
        return self.pretrained_item_embedding(item_ids)