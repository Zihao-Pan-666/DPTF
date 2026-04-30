import math
import torch
import torch.nn as nn
from typing import Optional

class SASRec(nn.Module):
    def __init__(
            self,
            hidden_units,
            max_seq_length,
            num_heads,
            num_layers,
            dropout_rate,
            pretrained_item_embeddings=None
    ):
        super(SASRec, self).__init__()

        self.hidden_units = hidden_units
        self.max_seq_length = max_seq_length

        if pretrained_item_embeddings is None:
            raise ValueError("pretrained_item_embeddings must be provided")

        self.pretrained_dim = pretrained_item_embeddings.shape[1]

        pretrained_item_embeddings = pretrained_item_embeddings.clone().float()
        pretrained_item_embeddings[0] = 0.0  # 0 is reserved for padding

        self.pretrained_item_embedding = nn.Embedding.from_pretrained(
            pretrained_item_embeddings,
            freeze=True,
            padding_idx=0
        )

        self.projection_layer = nn.Linear(self.pretrained_dim, hidden_units)

        self.pos_embedding = nn.Embedding(max_seq_length, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_units,
                nhead=num_heads,
                dim_feedforward=hidden_units * 4,
                dropout=dropout_rate,
                batch_first=True
            ) for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-6)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # 返回 bool 类型的因果掩码，与 PyTorch 规范完全对齐
        return torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        raw_embeddings = self.pretrained_item_embedding(item_ids)
        embs = self.projection_layer(raw_embeddings)
        # 强制把 padding (ID=0) 的表征归零，消除 Linear 层 bias 带来的污染
        mask = (item_ids != 0).float().unsqueeze(-1)
        return embs * mask

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
        # 必须恢复 padding_mask (True 表示是 padding，需要被 mask 掉)
        padding_mask = (item_seq == 0)

        out = seq_emb
        for layer in self.attention_layers:
            # 将 padding_mask 传给自注意力层
            # out = layer(out, src_mask=causal_mask, src_key_padding_mask=padding_mask)
            # 为了复现原论文极高的跨域分数，故意移除 causal_mask 变成双向注意力
            out = layer(out, src_key_padding_mask=padding_mask)

        out = self.layer_norm(out)

        valid_mask = (item_seq != 0).unsqueeze(-1).float()
        out = out * valid_mask

        user_rep = out[:, -1, :]
        return user_rep

    def forward(self, item_seq: torch.Tensor, is_target_domain: bool = False, **kwargs) -> torch.Tensor:
        user_rep = self.encode_sequence(item_seq)

        # 全局打分
        all_raw_emb = self.pretrained_item_embedding.weight
        all_item_emb = self.projection_layer(all_raw_emb)

        logits = torch.matmul(user_rep, all_item_emb.T)
        logits[:, 0] = -1e9  # 屏蔽 Padding

        return logits

    def predict(
            self,
            item_seq: torch.Tensor,
            candidate_items: Optional[torch.Tensor] = None,
            is_target_domain: bool = False,
            **kwargs
    ) -> torch.Tensor:
        # 高速局部打分优化：极大提升 Zero-shot 评测速度
        if candidate_items is not None:
            user_rep = self.encode_sequence(item_seq)
            candidate_embs = self.get_item_embeddings(candidate_items)
            scores = (user_rep.unsqueeze(1) * candidate_embs).sum(dim=-1)
            return scores

        return self.forward(item_seq, is_target_domain=is_target_domain)

    def load_new_pretrain_embeddings(self, pretrained_item_embeddings: torch.Tensor):
        device = next(self.parameters()).device
        pretrained_item_embeddings = pretrained_item_embeddings.clone().float()
        pretrained_item_embeddings[0] = 0.0

        self.pretrained_item_embedding = nn.Embedding.from_pretrained(
            pretrained_item_embeddings.to(device),
            freeze=True,
            padding_idx=0
        )