import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class SASRecWithDomainAlignment(nn.Module):
    """
    Paper-aligned SASRec-RecG implementation.

    Mapping to the paper:
    - Eq. (2): semantic projection layer W_p
    - Paragraph after Eq. (15): add a second projection layer for generalization,
      then merge both outputs to form final item embeddings
    - Eq. (4)-(5): SeqRec encodes the sequence and scores candidate items with dot product
    - Eq. (17)-(21): sequence-level generalization for zero-shot target inference
    """

    def __init__(
        self,
        hidden_units: int,
        max_seq_length: int,
        num_heads: int,
        num_layers: int,
        dropout_rate: float,
        pretrained_item_embeddings: Optional[torch.Tensor] = None,
        num_sequential_patterns: int = 10,
    ):
        super().__init__()

        if pretrained_item_embeddings is None:
            raise ValueError("pretrained_item_embeddings must be provided.")

        self.hidden_units = hidden_units
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_sequential_patterns = num_sequential_patterns

        self.pretrained_dim = pretrained_item_embeddings.shape[1]
        pretrained_item_embeddings = pretrained_item_embeddings.clone().float()
        pretrained_item_embeddings[0] = 0.0  # 0 is reserved for padding

        self.pretrained_item_embedding = nn.Embedding.from_pretrained(
            pretrained_item_embeddings,
            freeze=True,
            padding_idx=0,
        )

        # Eq. (2): semantic projection layer W_p
        self.projection_layer = nn.Linear(self.pretrained_dim, hidden_units)

        # Paragraph after Eq. (15): an additional projection layer for generalization
        self.generalization_projection_layer = nn.Linear(self.pretrained_dim, hidden_units)

        # Merge the two projected representations into the final item embedding space
        self.merge_layer = nn.Linear(hidden_units * 2, hidden_units)

        self.pos_embedding = nn.Embedding(max_seq_length, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_units,
                    nhead=num_heads,
                    dim_feedforward=hidden_units * 4,
                    dropout=dropout_rate,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-6)

        # Eq. (20)-(21): fuse target user embedding with attended source pattern
        # self.pattern_fusion_layer = nn.Linear(hidden_units * 2, hidden_units)
        self.register_buffer("sequential_patterns", torch.empty(0), persistent=False)
        self.use_sequential_patterns = False

    def project_raw_embeddings(self, raw_embeddings: torch.Tensor) -> torch.Tensor:
        """
        [AUTHOR-CONFIRMED]
        For recommendation scoring, only the projection layer is used.
        The generalization projection is kept for alignment learning.
        """
        return self.projection_layer(raw_embeddings)

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        raw_embeddings = self.pretrained_item_embedding(item_ids)
        # [AUTHOR-CONFIRMED]
        # Candidate/item scoring uses only the projection layer.
        embs = self.project_raw_embeddings(raw_embeddings)
        # 【关键修复】强制把 padding (ID=0) 的表征归零，消除 Linear 层 bias 带来的污染
        mask = (item_ids != 0).float().unsqueeze(-1)
        return embs * mask

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def _select_last_hidden(self, hidden_states: torch.Tensor, item_seq: torch.Tensor) -> torch.Tensor:
        valid_lengths = (item_seq != 0).sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(item_seq.size(0), device=item_seq.device)
        return hidden_states[batch_indices, valid_lengths]

    def encode_sequence(self, item_seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = item_seq.size()
        device = item_seq.device

        seq_emb = self.get_item_embeddings(item_seq)
        seq_emb = seq_emb * math.sqrt(self.hidden_units)

        # 修复位置编码漂移：强制将绝对位置对齐到 max_seq_length 末端
        positions = torch.arange(self.max_seq_length - seq_len, self.max_seq_length, device=device).unsqueeze(0).expand(
            batch_size, -1)
        seq_emb = self.dropout(seq_emb + self.pos_embedding(positions))

        causal_mask = self._causal_mask(seq_len, device)

        # 移除 src_key_padding_mask，避免 PyTorch 的 Softmax 全 -inf 导致 NaN
        out = seq_emb
        for layer in self.attention_layers:
            out = layer(out, src_mask=causal_mask)

        out = self.layer_norm(out)

        valid_mask = (item_seq != 0).unsqueeze(-1).float()
        out = out * valid_mask

        user_rep = out[:, -1, :]
        return user_rep

    def _apply_sequential_pattern_attention(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Eq. (17)-(21):
        1) cosine similarity between target user embedding and source patterns
        2) softmax attention over patterns
        3) weighted sum of patterns
        4) concatenate and project back to hidden size
        """
        if self.sequential_patterns.numel() == 0:
            return user_embeddings

        user_norm = F.normalize(user_embeddings, p=2, dim=1)
        pattern_norm = F.normalize(self.sequential_patterns, p=2, dim=1)

        similarity = torch.matmul(user_norm, pattern_norm.T)  # Eq. (17)
        attention = F.softmax(similarity, dim=1)              # Eq. (18)
        attended = torch.matmul(attention, self.sequential_patterns)  # Eq. (19)

        return user_embeddings + 0.5 * attended

    def forward(self, item_seq: torch.Tensor, is_target_domain: bool = False) -> torch.Tensor:
        user_rep = self.encode_sequence(item_seq)
        if is_target_domain and self.use_sequential_patterns and self.sequential_patterns.numel() > 0:
            user_rep = self._apply_sequential_pattern_attention(user_rep)
        # IMPORTANT:
        # Candidate items must live in the SAME final embedding space as the sequence encoder input.
        # all_item_emb = self.project_raw_embeddings(self.pretrained_item_embedding.weight)
        # logits = torch.matmul(user_rep, all_item_emb.T)  # Eq. (5)

        # [AUTHOR-CONFIRMED]
        # For recommendation scoring, only the projection layer is used.
        all_raw_emb = self.pretrained_item_embedding.weight
        all_item_emb = self.projection_layer(all_raw_emb)

        logits = torch.matmul(user_rep, all_item_emb.T)

        # 【非常关键的安全补丁】：屏蔽 Padding (ID=0) 的推荐，防止 Padding 霸榜
        logits[:, 0] = -1e9

        return logits


    def predict(
        self,
        item_seq: torch.Tensor,
        candidate_items: Optional[torch.Tensor] = None,
        is_target_domain: bool = False,
    ) -> torch.Tensor:
        logits = self.forward(item_seq, is_target_domain=is_target_domain)
        if candidate_items is not None:
            return torch.gather(logits, dim=1, index=candidate_items)
        return logits

    def extract_sequential_patterns_from_source(self, source_sequences: torch.Tensor) -> torch.Tensor:
        """
        Eq. (16): cluster source user sequence embeddings into K sequential patterns.
        """
        device = next(self.parameters()).device
        self.eval()

        with torch.no_grad():
            source_sequences = source_sequences.to(device)
            source_user_embeddings = self.encode_sequence(source_sequences)

        embeddings_np = source_user_embeddings.detach().cpu().numpy()
        kmeans = KMeans(
            n_clusters=self.num_sequential_patterns,
            random_state=42,
            n_init=10,
        )
        kmeans.fit(embeddings_np)
        patterns = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        self.sequential_patterns = patterns
        self.use_sequential_patterns = True
        return patterns

    def load_new_pretrain_embeddings(self, pretrained_item_embeddings: torch.Tensor):
        """Used for zero-shot target-domain evaluation with the SAME learned projection layers."""
        device = next(self.parameters()).device
        pretrained_item_embeddings = pretrained_item_embeddings.clone().float()
        pretrained_item_embeddings[0] = 0.0

        self.pretrained_item_embedding = nn.Embedding.from_pretrained(
            pretrained_item_embeddings.to(device),
            freeze=True,
            padding_idx=0,
        )

    def project_items_for_alignment(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.get_item_embeddings(item_ids)

    def project_raw_for_alignment(self, raw_embeddings: torch.Tensor) -> torch.Tensor:
        return self.project_raw_embeddings(raw_embeddings)
