from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
from torch import nn
import torch.nn.functional as F


class LLMRecGBERT4Rec(nn.Module):
    """BERT4Rec-RecG style model adapted to the ISDDG project.

    This implementation follows the official yunzhel2/LLM-RecG `bert4rec_recg.py`
    design at the method level:
      - frozen LLM item embedding table;
      - recommendation projection layer;
      - domain-alignment projection layer;
      - merge layer for BERT4Rec sequence input;
      - BERT-style TransformerEncoder blocks;
      - source sequential pattern extraction by KMeans;
      - target-side soft pattern attention.

    Project-specific safeguards:
      - PrefixDataset in this project uses left padding, so the latest item is
        always at index -1. We intentionally use out[:, -1, :] for user states.
      - Switching to a target item table preserves trained projection/encoder
        weights. Resetting projection layers would violate zero-shot transfer.
    """

    def __init__(
        self,
        item_features: torch.Tensor,
        hidden_dim: int = 256,
        max_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout: float = 0.5,
        num_sequential_patterns: int = 10,
        pattern_fusion: str = "residual",
        pattern_residual_weight: float = 0.5,
        init_pattern_fusion_as_residual: bool = True,
    ):
        super().__init__()
        if item_features.dim() != 2:
            raise ValueError(f"item_features must be [num_items+1, dim], got {tuple(item_features.shape)}")
        if pattern_fusion not in {"official_linear", "linear", "residual", "none"}:
            raise ValueError("pattern_fusion must be one of: official_linear, linear, residual, none.")

        self.hidden_units = int(hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_seq_length = int(max_len)
        self.max_len = int(max_len)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout_rate = float(dropout)
        self.num_sequential_patterns = int(num_sequential_patterns)
        self.pattern_fusion = str(pattern_fusion)
        self.pattern_residual_weight = float(pattern_residual_weight)

        item_features = item_features.clone().float()
        item_features[0] = 0.0
        self.register_buffer("item_features", item_features, persistent=False)
        self.pretrained_dim = int(item_features.size(1))

        # Official BERT4Rec-RecG naming.
        self.projection_layer = nn.Linear(self.pretrained_dim, self.hidden_units)
        self.domain_alignment_projection_layer = nn.Linear(self.pretrained_dim, self.hidden_units)
        self.merge_layer = nn.Linear(self.hidden_units * 2, self.hidden_units)

        self.pos_embedding = nn.Embedding(self.max_seq_length, self.hidden_units)
        self.dropout = nn.Dropout(self.dropout_rate)

        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_units,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_units * 4,
                dropout=self.dropout_rate,
                activation="gelu",
                batch_first=False,
            )
            for _ in range(self.num_layers)
        ])
        self.layer_norm = nn.LayerNorm(self.hidden_units, eps=1e-6)

        self.pattern_fusion_layer = nn.Linear(self.hidden_units * 2, self.hidden_units)
        if init_pattern_fusion_as_residual:
            self._init_pattern_fusion_as_residual()

        self.register_buffer("sequential_patterns", torch.empty(0, self.hidden_units), persistent=False)
        self.use_sequential_patterns = False

    def _init_pattern_fusion_as_residual(self) -> None:
        """Initialize official linear fusion as user_h + w * attended pattern.

        The official repository defines a linear layer for [user || attended]
        pattern fusion, but that layer is not trained before zero-shot inference
        if patterns are extracted post-training. This initialization keeps the
        official linear module available while avoiding random target corruption.
        """
        with torch.no_grad():
            self.pattern_fusion_layer.weight.zero_()
            self.pattern_fusion_layer.bias.zero_()
            eye = torch.eye(self.hidden_units)
            self.pattern_fusion_layer.weight[:, : self.hidden_units].copy_(eye)
            self.pattern_fusion_layer.weight[:, self.hidden_units :].copy_(eye * self.pattern_residual_weight)

    @property
    def num_items(self) -> int:
        return int(self.item_features.size(0) - 1)

    def set_item_features(self, item_features: torch.Tensor) -> None:
        """Switch source item table to a target item table for zero-shot inference.

        Projection, merge, position, Transformer, and pattern-fusion parameters
        are preserved. This is the zero-shot-safe behavior.
        """
        if int(item_features.size(1)) != self.pretrained_dim:
            raise ValueError(
                f"Target embedding dim mismatch: got {item_features.size(1)}, expected {self.pretrained_dim}."
            )
        item_features = item_features.clone().float()
        item_features[0] = 0.0
        self.item_features = item_features.to(next(self.parameters()).device)

    # Official-compatible alias.
    def load_new_pretrain_embeddings(self, pretrained_item_embeddings: torch.Tensor) -> None:
        self.set_item_features(pretrained_item_embeddings)

    def _mask_pad(self, item_ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * (item_ids != 0).float().unsqueeze(-1)

    def project_raw_rec(self, raw_features: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(raw_features.float())

    def project_raw_alignment(self, raw_features: torch.Tensor) -> torch.Tensor:
        return self.domain_alignment_projection_layer(raw_features.float())

    # Official-compatible names.
    def projection_embeddings(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        return self.project_raw_rec(item_embeddings)

    def irm_projection_embeddings(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        return self.project_raw_alignment(item_embeddings)

    def project_raw_gen(self, raw_features: torch.Tensor) -> torch.Tensor:
        return self.project_raw_alignment(raw_features)

    def encode_items_rec(self, item_ids: torch.Tensor) -> torch.Tensor:
        raw = self.item_features[item_ids]
        return self._mask_pad(item_ids, self.project_raw_rec(raw))

    def encode_items_align(self, item_ids: torch.Tensor) -> torch.Tensor:
        raw = self.item_features[item_ids]
        return self._mask_pad(item_ids, self.project_raw_alignment(raw))

    # Backward-compatible alias used by previous project code.
    def encode_items_gen(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_items_align(item_ids)

    def encode_items_sequence(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Official BERT4Rec-RecG sequence input: merge rec and alignment projections."""
        raw = self.item_features[item_ids]
        emb_rec = self.projection_layer(raw)
        emb_irm = self.domain_alignment_projection_layer(raw)
        emb_rec = self.dropout(emb_rec)
        emb_irm = self.dropout(emb_irm)
        seq_emb = self.merge_layer(torch.cat([emb_rec, emb_irm], dim=-1))
        return self._mask_pad(item_ids, seq_emb)

    def encode_sequence(self, history: torch.Tensor, apply_patterns: bool = False) -> torch.Tensor:
        B, L = history.shape
        if L > self.max_seq_length:
            history = history[:, -self.max_seq_length :]
            L = self.max_seq_length

        seq_emb = self.encode_items_sequence(history)
        pos = torch.arange(L, dtype=torch.long, device=history.device).unsqueeze(0).expand(B, L)
        seq_emb = self.dropout(seq_emb + self.pos_embedding(pos))

        padding_mask = history.eq(0)
        out = seq_emb
        for layer in self.attention_layers:
            out = layer(out.permute(1, 0, 2), src_key_padding_mask=padding_mask).permute(1, 0, 2)

        # The project uses left padding. The latest item is therefore at index -1.
        user_rep = self.layer_norm(out[:, -1, :])

        if apply_patterns and self.use_sequential_patterns and self.sequential_patterns.numel() > 0:
            user_rep = self._apply_sequential_pattern_attention(user_rep)
        return user_rep

    def forward(self, item_seq: torch.LongTensor, is_target_domain: bool = False) -> torch.Tensor:
        user_rep = self.encode_sequence(item_seq, apply_patterns=is_target_domain)
        all_item_emb = self.projection_layer(self.item_features)
        logits = user_rep @ all_item_emb.t()
        logits[:, 0] = -torch.inf
        return logits

    def _apply_sequential_pattern_attention(self, user_embeddings: torch.Tensor) -> torch.Tensor:
        if self.sequential_patterns.numel() == 0 or self.pattern_fusion == "none":
            return user_embeddings

        user_norm = F.normalize(user_embeddings, p=2, dim=1)
        pattern_norm = F.normalize(self.sequential_patterns, p=2, dim=1)
        similarity_scores = user_norm @ pattern_norm.t()
        attention_weights = F.softmax(similarity_scores, dim=1)
        attended_patterns = attention_weights @ self.sequential_patterns

        if self.pattern_fusion == "residual":
            return user_embeddings + self.pattern_residual_weight * attended_patterns

        # Official bert4rec_recg.py branch: concat + W_f projection.
        fused = torch.cat([user_embeddings, attended_patterns], dim=1)
        return self.pattern_fusion_layer(fused)

    def score(self, history: torch.Tensor, candidates: torch.Tensor, is_target_domain: bool = False) -> torch.Tensor:
        user_rep = self.encode_sequence(history, apply_patterns=is_target_domain)
        cand_emb = self.encode_items_rec(candidates)
        return torch.einsum("bd,bnd->bn", user_rep, cand_emb)

    def score_all(self, history: torch.Tensor, is_target_domain: bool = False) -> torch.Tensor:
        return self.forward(history, is_target_domain=is_target_domain)

    def predict(
        self,
        item_seq: torch.LongTensor,
        candidate_items: Optional[torch.LongTensor] = None,
        is_target_domain: bool = False,
    ) -> torch.Tensor:
        logits = self.forward(item_seq, is_target_domain=is_target_domain)
        if candidate_items is not None:
            return torch.gather(logits, dim=1, index=candidate_items)
        return logits

    def sample_internal_embeddings(self, sample_size: int, device: torch.device) -> torch.Tensor:
        if self.num_items <= 0:
            raise ValueError("No internal source embeddings available.")
        idx = torch.randint(1, self.item_features.size(0), (int(sample_size),), device=device)
        return self.item_features[idx]

    @torch.no_grad()
    def extract_sequential_patterns(
        self,
        history_loader,
        device: torch.device,
        num_patterns: Optional[int] = None,
        kmeans_batch_size: int = 4096,
        max_users: int = 0,
        show_progress: bool = True,
        use_minibatch: bool = False,
    ) -> torch.Tensor:
        """Extract source sequential patterns by clustering source user states."""
        from tqdm import tqdm

        self.eval()
        k = int(num_patterns or self.num_sequential_patterns)
        reps = []
        seen = 0
        for batch in tqdm(history_loader, desc="extract BERT4Rec-RecG source patterns", leave=False, disable=not show_progress):
            hist = batch["history"].to(device, non_blocking=True)
            rep = self.encode_sequence(hist, apply_patterns=False).detach().cpu().float()
            reps.append(rep)
            seen += int(rep.size(0))
            if max_users and seen >= int(max_users):
                break

        if not reps:
            raise RuntimeError("No source sequence representations were collected for pattern extraction.")

        x = torch.cat(reps, dim=0)
        if max_users and x.size(0) > int(max_users):
            x = x[: int(max_users)]

        x_np = x.numpy().astype(np.float32)
        k = min(k, x_np.shape[0])
        if use_minibatch:
            km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=max(int(kmeans_batch_size), k * 4), n_init="auto")
        else:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(x_np)

        patterns = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
        self.sequential_patterns = patterns
        self.use_sequential_patterns = True
        self.num_sequential_patterns = int(k)
        return patterns

    # Official-compatible name.
    def extract_sequential_patterns_from_source(self, source_sequences: torch.Tensor) -> torch.Tensor:
        self.eval()
        device = next(self.parameters()).device
        source_sequences = source_sequences.to(device)
        with torch.no_grad():
            reps = self.encode_sequence(source_sequences, apply_patterns=False).detach().cpu().numpy().astype(np.float32)
        k = min(self.num_sequential_patterns, reps.shape[0])
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(reps)
        patterns = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
        self.sequential_patterns = patterns
        self.use_sequential_patterns = True
        return patterns

    @torch.no_grad()
    def get_pattern_attention_weights(self, target_sequences: torch.Tensor) -> torch.Tensor:
        if not self.use_sequential_patterns or self.sequential_patterns.numel() == 0:
            raise ValueError("Sequential patterns not available. Call extract_sequential_patterns first.")
        user_embeddings = self.encode_sequence(target_sequences, apply_patterns=False)
        user_norm = F.normalize(user_embeddings, p=2, dim=1)
        pattern_norm = F.normalize(self.sequential_patterns, p=2, dim=1)
        return F.softmax(user_norm @ pattern_norm.t(), dim=1)


# Backward-compatible alias for old scripts.
LLMRecGFeatureBERT4Rec = LLMRecGBERT4Rec
