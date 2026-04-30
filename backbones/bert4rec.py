import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class BERT4RecWithDomainAlignment(nn.Module):
    def __init__(
            self, hidden_units: int, max_seq_length: int, num_heads: int,
            num_layers: int, dropout_rate: float,
            pretrained_item_embeddings: torch.Tensor = None,
            num_sequential_patterns: int = 10
    ):
        super(BERT4RecWithDomainAlignment, self).__init__()
        self.hidden_units = hidden_units
        self.max_seq_length = max_seq_length
        self.num_sequential_patterns = num_sequential_patterns

        if pretrained_item_embeddings is not None:
            self.pretrained_dim = pretrained_item_embeddings.shape[1]
            self.pretrained_item_embedding = nn.Embedding.from_pretrained(
                pretrained_item_embeddings, freeze=True, padding_idx=0
            )
            # 【核心架构 1】：推荐投影层 (主导域内打分)
            self.projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
            # 【核心架构 2】：泛化投影层 (主导跨域对齐)
            self.domain_alignment_projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
            # 【核心架构 3】：序列特征融合层
            self.merge_layer = nn.Linear(hidden_units * 2, hidden_units)
        else:
            self.pretrained_item_embedding = None

        self.pos_embedding = nn.Embedding(max_seq_length, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_units, nhead=num_heads,
                dim_feedforward=hidden_units * 4, dropout=dropout_rate, activation="gelu"
            ) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-6)

        # 目标域 Sequence Pattern 融合层
        self.sequential_patterns = None
        self.pattern_fusion_layer = nn.Linear(hidden_units * 2, hidden_units)
        self.use_sequential_patterns = False

    def encode_sequence(self, item_seq: torch.Tensor, is_sem_baseline: bool = False) -> torch.Tensor:
        """
        全量修复版：集成 Padding 屏蔽、输出归一化及过拟合控制
        """
        device = item_seq.device
        batch_size, seq_len = item_seq.size()
        raw_emb = self.pretrained_item_embedding(item_seq)

        # 【修复原因 1】：建立 Padding 掩码 (1 for real item, 0 for padding)
        # 必须在投影层输出后立即归零，否则 Linear 层的 bias 会污染特征空间 [cite: 785, 933]
        valid_mask = (item_seq != 0).float().unsqueeze(-1)

        # 1. 投影逻辑
        emb_rec = self.projection_layer(raw_emb)
        # 强制归零 Padding 位置的偏置项 (Bias) [cite: 785]
        emb_rec = emb_rec * valid_mask

        if is_sem_baseline:
            seq_emb = emb_rec
        else:
            emb_irm = self.domain_alignment_projection_layer(raw_emb)
            # 同样对泛化层执行归零
            emb_irm = emb_irm * valid_mask
            # Merge 融合
            merged = torch.cat((emb_rec, emb_irm), dim=-1)
            seq_emb = self.merge_layer(merged)
            # 融合后再次确保 Padding 位置纯净
            seq_emb = seq_emb * valid_mask

        # 【优化原因 3】：精简 Dropout
        # 移除投影层后的独立 Dropout，遵循 BERT 原版逻辑：仅在叠加位置编码后统一做一次 Dropout
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        seq_emb = seq_emb + self.pos_embedding(positions)
        seq_emb = self.dropout(seq_emb)  # 统一的 Dropout

        # 3. Transformer 编码
        padding_mask = (item_seq == 0)
        out = seq_emb
        for layer in self.attention_layers:
            out = layer(out.permute(1, 0, 2), src_key_padding_mask=padding_mask).permute(1, 0, 2)

        # 【修复原因 2】：输出端再次 Mask [cite: 785]
        # Transformer 的自注意力机制和 LayerNorm 可能会让 Padding 位置带上微弱数值
        # 在提取用户表征前执行最后一次清理，确保 user_rep 绝对纯净
        out = self.layer_norm(out)
        out = out * valid_mask  # 关键步骤：过滤残余计算量 [cite: 785]

        user_rep = out[:, -1, :]  # 取序列最后一个有效位置 [cite: 785, 933]
        return user_rep

    def forward(self, item_seq: torch.LongTensor, is_target_domain: bool = False,
                is_sem_baseline: bool = False) -> torch.Tensor:
        # 复用 encode_sequence 获取用户表征
        user_rep = self.encode_sequence(item_seq, is_sem_baseline=is_sem_baseline)

        # 【核心流程：零样本目标域模式聚合】
        if is_target_domain and not is_sem_baseline and self.use_sequential_patterns and self.sequential_patterns is not None:
            attended_patterns = self._apply_sequential_pattern_attention(user_rep)
            fused_representation = torch.cat([user_rep, attended_patterns], dim=1)
            user_rep = self.pattern_fusion_layer(fused_representation)

        # 【核心流程：非对称候选打分】 (只用 projection_layer)
        all_item_emb_rec = self.projection_layer(self.pretrained_item_embedding.weight)
        logits = torch.matmul(user_rep, all_item_emb_rec.T)
        logits[:, 0] = -1e9  # 屏蔽 padding
        return logits

    def _apply_sequential_pattern_attention(self, user_rep_gen: torch.Tensor) -> torch.Tensor:
        if self.sequential_patterns.numel() == 0:
            return torch.zeros_like(user_rep_gen)
        user_norm = F.normalize(user_rep_gen, p=2, dim=1)
        pattern_norm = F.normalize(self.sequential_patterns, p=2, dim=1)
        similarity = torch.matmul(user_norm, pattern_norm.T)
        attention = F.softmax(similarity, dim=1)
        attended = torch.matmul(attention, self.sequential_patterns)
        return attended

    def extract_sequential_patterns_from_source(self, source_sequences: torch.Tensor,
                                                batch_size: int = 512) -> torch.Tensor:
        device = next(self.parameters()).device
        self.eval()

        embeddings_list = []
        with torch.no_grad():
            for i in range(0, source_sequences.size(0), batch_size):
                batch_seq = source_sequences[i: i + batch_size].to(device)

                # 【修改点】：直接复用与 forward 完全一致的 Merge 编码逻辑
                raw_emb = self.pretrained_item_embedding(batch_seq)
                emb_rec = self.dropout(self.projection_layer(raw_emb))
                emb_irm = self.dropout(self.domain_alignment_projection_layer(raw_emb))

                merged = torch.cat((emb_rec, emb_irm), dim=-1)
                seq_emb = self.merge_layer(merged)

                positions = torch.arange(batch_seq.size(1), device=device).unsqueeze(0).expand(batch_seq.size(0), -1)
                seq_emb = self.dropout(seq_emb + self.pos_embedding(positions))
                padding_mask = (batch_seq == 0)

                out = seq_emb
                for layer in self.attention_layers:
                    out = layer(out.permute(1, 0, 2), src_key_padding_mask=padding_mask).permute(1, 0, 2)
                user_rep = self.layer_norm(out[:, -1, :])

                embeddings_list.append(user_rep.cpu())

        embeddings_np = torch.cat(embeddings_list, dim=0).numpy()
        kmeans = KMeans(n_clusters=self.num_sequential_patterns, random_state=42, n_init=10)
        kmeans.fit(embeddings_np)
        patterns = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        self.sequential_patterns = patterns
        self.use_sequential_patterns = True
        return patterns

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
