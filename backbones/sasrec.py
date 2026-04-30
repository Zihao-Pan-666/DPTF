import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class SASRecWithDomainAlignment(nn.Module):
    def __init__(self, hidden_units: int, max_seq_length: int, num_heads: int,
                 num_layers: int, dropout_rate: float,
                 pretrained_item_embeddings: torch.Tensor = None,
                 num_sequential_patterns: int = 10):
        super().__init__()
        self.hidden_units = hidden_units
        self.max_seq_length = max_seq_length
        self.num_sequential_patterns = num_sequential_patterns

        self.pretrained_dim = pretrained_item_embeddings.shape[1]
        self.pretrained_item_embedding = nn.Embedding.from_pretrained(
            pretrained_item_embeddings.float(), freeze=True, padding_idx=0
        )

        self.projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
        self.domain_alignment_projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
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

        # 【同步 BERT4Rec 修复】：Padding 掩码
        valid_mask = (item_seq != 0).float().unsqueeze(-1)

        emb_rec = self.projection_layer(raw_emb) * valid_mask  # 屏蔽 Bias

        if is_sem_baseline:
            seq_emb = emb_rec
        else:
            emb_irm = self.domain_alignment_projection_layer(raw_emb) * valid_mask
            seq_emb = self.merge_layer(torch.cat((emb_rec, emb_irm), dim=-1)) * valid_mask

        # SASRec 核心：因果掩码
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        seq_emb = self.dropout(seq_emb + self.pos_embedding(positions))

        out = seq_emb
        for layer in self.attention_layers:
            # SASRec 需要同时传入 src_mask 和 src_key_padding_mask
            out = layer(out, src_key_padding_mask=(item_seq == 0))

        out = self.layer_norm(out)
        out = out * valid_mask  # 【同步 BERT4Rec 修复】：输出清理

        user_rep = out[:, -1, :]
        return user_rep

    def forward(self, item_seq: torch.LongTensor, is_target_domain: bool = False,
                is_sem_baseline: bool = False) -> torch.Tensor:
        user_rep = self.encode_sequence(item_seq, is_sem_baseline=is_sem_baseline)

        if is_target_domain and not is_sem_baseline and self.use_sequential_patterns and self.sequential_patterns is not None:
            attended_patterns = self._apply_sequential_pattern_attention(user_rep)
            user_rep = self.pattern_fusion_layer(torch.cat([user_rep, attended_patterns], dim=1))

        # 【核心逻辑】：非对称打分，仅使用推荐投影层
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
                # 复用修复后的编码逻辑进行聚类
                embs.append(self.encode_sequence(batch_seq, is_sem_baseline=False).cpu())
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
        return self.domain_alignment_projection_layer(self.pretrained_item_embedding(item_ids))

    def project_raw_for_alignment(self, raw_embeddings: torch.Tensor) -> torch.Tensor:
        return self.domain_alignment_projection_layer(raw_embeddings)

    def load_new_pretrain_embeddings(self, pretrained_item_embeddings: torch.Tensor):
        device = next(self.parameters()).device
        self.pretrained_item_embedding = nn.Embedding.from_pretrained(
            pretrained_item_embeddings.to(device).float(), freeze=True, padding_idx=0,
        )

    def get_raw_item_embeddings(self, item_ids):
        return self.pretrained_item_embedding(item_ids)