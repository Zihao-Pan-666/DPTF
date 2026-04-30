import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class GRU4RecWithDomainAlignment(nn.Module):
    def __init__(
            self, hidden_units: int, num_layers: int, dropout_rate: float,
            pretrained_item_embeddings: torch.Tensor = None,
            num_sequential_patterns: int = 10,
            max_seq_length: int = 50
    ):
        super(GRU4RecWithDomainAlignment, self).__init__()
        self.hidden_units = hidden_units
        self.num_sequential_patterns = num_sequential_patterns

        if pretrained_item_embeddings is not None:
            self.pretrained_dim = pretrained_item_embeddings.shape[1]
            self.pretrained_item_embedding = nn.Embedding.from_pretrained(
                pretrained_item_embeddings, freeze=True, padding_idx=0
            )
            self.projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
            self.domain_alignment_projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
            self.merge_layer = nn.Linear(hidden_units * 2, hidden_units)
        else:
            self.pretrained_item_embedding = None

        # 【优化 1】：增加输入端归一化层，解决 LLM 特征分布不均问题
        self.input_ln = nn.LayerNorm(hidden_units, eps=1e-6)

        self.gru = nn.GRU(
            input_size=hidden_units,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout_rate)
        # 【优化 2】：增加输出端归一化层
        self.output_ln = nn.LayerNorm(hidden_units, eps=1e-6)

        self.sequential_patterns = None
        self.pattern_fusion_layer = nn.Linear(hidden_units * 2, hidden_units)
        self.use_sequential_patterns = False

    def encode_sequence(self, item_seq: torch.Tensor, is_sem_baseline: bool = False) -> torch.Tensor:
        device = item_seq.device
        raw_emb = self.pretrained_item_embedding(item_seq)
        valid_mask = (item_seq != 0).float().unsqueeze(-1)

        # 1. 投影与融合逻辑
        emb_rec = self.projection_layer(raw_emb)
        if is_sem_baseline:
            seq_emb = emb_rec
        else:
            emb_irm = self.domain_alignment_projection_layer(raw_emb)
            merged = torch.cat((emb_rec, emb_irm), dim=-1)
            seq_emb = self.merge_layer(merged)

        # 【关键修复】：在进入 GRU 前执行归一化、激活和 Mask
        # 只有这样才能确保 Padding 位的 bias 不会影响 GRU 的隐藏状态更新
        seq_emb = self.input_ln(seq_emb)
        seq_emb = F.gelu(seq_emb)
        seq_emb = seq_emb * valid_mask
        seq_emb = self.dropout(seq_emb)

        # 2. GRU 编码
        output, _ = self.gru(seq_emb)

        # 3. 提取用户表征
        # 【性能优化】：移除 .cpu()，直接在 GPU 上完成索引计算
        seq_lengths = (item_seq != 0).sum(dim=1)
        batch_size = item_seq.size(0)
        idx = (seq_lengths - 1).clamp(min=0)

        # 提取最后一个有效物品对应的隐藏状态
        user_rep = output[torch.arange(batch_size), idx, :]

        # 【关键修复】：输出前执行最后一次 LayerNorm，增强打分稳定性
        user_rep = self.output_ln(user_rep)
        return user_rep

    def forward(self, item_seq: torch.LongTensor, is_target_domain: bool = False,
                is_sem_baseline: bool = False) -> torch.Tensor:
        user_rep = self.encode_sequence(item_seq, is_sem_baseline=is_sem_baseline)

        if is_target_domain and not is_sem_baseline and self.use_sequential_patterns and self.sequential_patterns is not None:
            attended_patterns = self._apply_sequential_pattern_attention(user_rep)
            fused_representation = torch.cat([user_rep, attended_patterns], dim=1)
            user_rep = self.pattern_fusion_layer(fused_representation)
            user_rep = self.output_ln(user_rep)  # 融合后再次归一化

        # 计算所有物品的推荐分
        all_item_emb_rec = self.projection_layer(self.pretrained_item_embedding.weight)
        logits = torch.matmul(user_rep, all_item_emb_rec.T)
        logits[:, 0] = -1e9  # Mask padding
        return logits

    # ... 其余方法 (extract_sequential_patterns_from_source 等) 保持与原代码一致 ...

    def _apply_sequential_pattern_attention(self, user_rep: torch.Tensor) -> torch.Tensor:
        if self.sequential_patterns is None or self.sequential_patterns.numel() == 0:
            return torch.zeros_like(user_rep)
        user_norm = F.normalize(user_rep, p=2, dim=1)
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
                user_rep = self.encode_sequence(batch_seq, is_sem_baseline=False)
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