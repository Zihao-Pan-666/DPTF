import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np


class BERT4RecWithDomainAlignment(nn.Module):
    """
    BERT4Rec with dynamic pretrained embeddings, domain alignment projections, and sequential pattern generalization.
    """

    def __init__(
            self,
            hidden_units: int,
            max_seq_length: int,
            num_heads: int,
            num_layers: int,
            dropout_rate: float,
            pretrained_item_embeddings: torch.Tensor = None,
            num_sequential_patterns: int = 10
    ):
        super(BERT4RecWithDomainAlignment, self).__init__()

        self.hidden_units = hidden_units
        self.max_seq_length = max_seq_length
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_sequential_patterns = num_sequential_patterns

        # Pretrained item embeddings + projections
        if pretrained_item_embeddings is not None:
            self.pretrained_dim = pretrained_item_embeddings.shape[1]
            self.pretrained_item_embedding = nn.Embedding.from_pretrained(
                pretrained_item_embeddings, freeze=True, padding_idx=0
            )
            # projection for recommendation
            self.projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
            # projection for domain alignment
            self.domain_alignment_projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
            # merge back to hidden_units
            self.merge_layer = nn.Linear(hidden_units * 2, hidden_units)
        else:
            self.pretrained_item_embedding = None
            self.projection_layer = None
            self.domain_alignment_projection_layer = None
            self.merge_layer = None

        # Positional embeddings
        self.pos_embedding = nn.Embedding(max_seq_length, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)

        # BERT-style Transformer Encoder layers
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_units,
                nhead=num_heads,
                dim_feedforward=hidden_units * 4,
                dropout=dropout_rate,
                activation="gelu"
            ) for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_units, eps=1e-6)

        # Sequential pattern components
        self.sequential_patterns = None  # Will store k sequential patterns
        self.pattern_fusion_layer = nn.Linear(hidden_units * 2, hidden_units)  # W_f projection matrix
        self.use_sequential_patterns = False

    def forward(self, item_seq: torch.LongTensor, is_target_domain: bool = False) -> torch.Tensor:
        """
        Forward pass computing logits for all items.
        Args:
            item_seq (torch.LongTensor): Input item sequences, shape (batch_size, seq_len).
            is_target_domain (bool): Whether this is target domain inference.
        Returns:
            torch.Tensor: Logits for all items.
        """
        device = item_seq.device
        batch_size, seq_len = item_seq.size()

        if self.pretrained_item_embedding is None:
            raise ValueError("No pretrained embeddings loaded. Call load_new_pretrain_embeddings first.")

        # lookup and project embeddings
        pretrained_emb = self.pretrained_item_embedding(item_seq)  # (B, L, D)
        emb_rec = self.projection_layer(pretrained_emb)  # (B, L, H)
        emb_irm = self.domain_alignment_projection_layer(pretrained_emb)  # (B, L, H)

        # dropout
        emb_rec = self.dropout(emb_rec)
        emb_irm = self.dropout(emb_irm)

        # merge
        merged = torch.cat((emb_rec, emb_irm), dim=-1)  # (B, L, 2H)
        seq_emb = self.merge_layer(merged)  # (B, L, H)

        # add positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        seq_emb = seq_emb + self.pos_embedding(positions)
        seq_emb = self.dropout(seq_emb)

        # padding mask for transformer
        padding_mask = (item_seq == 0)

        # transformer layers
        out = seq_emb
        for layer in self.attention_layers:
            out = layer(out.permute(1, 0, 2), src_key_padding_mask=padding_mask).permute(1, 0, 2)

        # get user representation from [CLS] equivalent (last position)
        user_rep = self.layer_norm(out[:, -1, :])  # (B, H)

        # Apply sequential pattern attention if in target domain and patterns are available
        if is_target_domain and self.use_sequential_patterns and self.sequential_patterns is not None:
            user_rep = self._apply_sequential_pattern_attention(user_rep)

        # final logits via dot product
        all_item_emb = self.projection_layer(self.pretrained_item_embedding.weight)  # (N, H)
        logits = torch.matmul(user_rep, all_item_emb.T)  # (B, N)

        return logits

    def _apply_sequential_pattern_attention(self, user_embeddings):
        """
        Apply soft sequential pattern attention mechanism.
        Args:
            user_embeddings (Tensor): Target user sequence embeddings, shape (batch_size, hidden_units).
        Returns:
            fused_embeddings (Tensor): Fused embeddings with pattern information, shape (batch_size, hidden_units).
        """
        device = user_embeddings.device
        batch_size = user_embeddings.size(0)

        # Compute cosine similarity between user embeddings and sequential patterns
        # user_embeddings: (batch_size, hidden_units)
        # sequential_patterns: (num_patterns, hidden_units)
        user_norm = F.normalize(user_embeddings, p=2, dim=1)  # (batch_size, hidden_units)
        pattern_norm = F.normalize(self.sequential_patterns, p=2, dim=1)  # (num_patterns, hidden_units)

        # Compute similarity scores: (batch_size, num_patterns)
        similarity_scores = torch.matmul(user_norm, pattern_norm.T)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(similarity_scores, dim=1)  # (batch_size, num_patterns)

        # Compute attended pattern representation
        attended_patterns = torch.matmul(attention_weights, self.sequential_patterns)  # (batch_size, hidden_units)

        # Concatenate user embedding with attended pattern representation
        fused_representation = torch.cat([user_embeddings, attended_patterns], dim=1)  # (batch_size, 2*hidden_units)

        # Project back to original embedding space
        fused_embeddings = self.pattern_fusion_layer(fused_representation)  # (batch_size, hidden_units)

        return fused_embeddings

    def extract_sequential_patterns_from_source(self, source_sequences):
        """
        Extract sequential patterns from source domain user sequences using k-means clustering.
        Args:
            source_sequences (Tensor): Source domain item sequences, shape (num_users, seq_len).
        """
        device = next(self.parameters()).device

        # Encode source sequences to get user embeddings
        with torch.no_grad():
            batch_size, seq_len = source_sequences.size()

            if self.pretrained_item_embedding is None:
                raise ValueError("No pretrained embeddings loaded.")

            # lookup and project embeddings
            pretrained_emb = self.pretrained_item_embedding(source_sequences)
            emb_rec = self.projection_layer(pretrained_emb)
            emb_irm = self.domain_alignment_projection_layer(pretrained_emb)

            # dropout
            emb_rec = self.dropout(emb_rec)
            emb_irm = self.dropout(emb_irm)

            # merge
            merged = torch.cat((emb_rec, emb_irm), dim=-1)
            seq_emb = self.merge_layer(merged)

            # add positional embeddings
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            seq_emb = seq_emb + self.pos_embedding(positions)

            # padding mask for transformer
            padding_mask = (source_sequences == 0)

            # transformer layers
            out = seq_emb
            for layer in self.attention_layers:
                out = layer(out.permute(1, 0, 2), src_key_padding_mask=padding_mask).permute(1, 0, 2)

            # get user representation from [CLS] equivalent (last position)
            source_user_embeddings = self.layer_norm(out[:, -1, :])  # (num_users, hidden_units)

        # Convert to numpy for k-means clustering
        embeddings_np = source_user_embeddings.cpu().numpy()

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=self.num_sequential_patterns, random_state=42, n_init=10)
        kmeans.fit(embeddings_np)

        # Store sequential patterns (cluster centroids)
        patterns = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        self.sequential_patterns = nn.Parameter(patterns, requires_grad=False)
        self.use_sequential_patterns = True

        print(
            f"Successfully extracted {self.num_sequential_patterns} sequential patterns from {len(source_sequences)} source sequences.")

        return patterns

    def get_pattern_attention_weights(self, target_sequences):
        """
        Get attention weights for target sequences over sequential patterns.
        Args:
            target_sequences (Tensor): Target domain sequences, shape (batch_size, seq_len).
        Returns:
            attention_weights (Tensor): Attention weights, shape (batch_size, num_patterns).
        """
        if not self.use_sequential_patterns or self.sequential_patterns is None:
            raise ValueError("Sequential patterns not available. Call extract_sequential_patterns_from_source first.")

        device = target_sequences.device

        with torch.no_grad():
            batch_size, seq_len = target_sequences.size()

            # Encode target sequences (same as forward pass without pattern attention)
            pretrained_emb = self.pretrained_item_embedding(target_sequences)
            emb_rec = self.projection_layer(pretrained_emb)
            emb_irm = self.domain_alignment_projection_layer(pretrained_emb)

            emb_rec = self.dropout(emb_rec)
            emb_irm = self.dropout(emb_irm)

            merged = torch.cat((emb_rec, emb_irm), dim=-1)
            seq_emb = self.merge_layer(merged)

            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            seq_emb = seq_emb + self.pos_embedding(positions)

            padding_mask = (target_sequences == 0)

            out = seq_emb
            for layer in self.attention_layers:
                out = layer(out.permute(1, 0, 2), src_key_padding_mask=padding_mask).permute(1, 0, 2)

            user_embeddings = self.layer_norm(out[:, -1, :])

            # Compute cosine similarity and attention weights
            user_norm = F.normalize(user_embeddings, p=2, dim=1)
            pattern_norm = F.normalize(self.sequential_patterns, p=2, dim=1)
            similarity_scores = torch.matmul(user_norm, pattern_norm.T)
            attention_weights = F.softmax(similarity_scores, dim=1)

        return attention_weights

    def predict(
            self,
            item_seq: torch.LongTensor,
            candidate_items: torch.LongTensor = None,
            is_target_domain: bool = False
    ) -> torch.Tensor:
        """
        Predict scores for candidate items or all items.
        Args:
            item_seq (torch.LongTensor): Input item sequences.
            candidate_items (torch.LongTensor, optional): Candidate items for ranking.
            is_target_domain (bool): Whether this is target domain inference.
        Returns:
            torch.Tensor: Scores for candidate items or all items.
        """
        logits = self.forward(item_seq, is_target_domain=is_target_domain)
        if candidate_items is not None:
            return torch.gather(logits, dim=1, index=candidate_items)
        return logits

    def load_new_pretrain_embeddings(self, pretrained_item_embeddings: torch.Tensor):
        """
        Dynamically load new pretrained embeddings.
        """
        device = next(self.parameters()).device
        self.pretrained_item_embedding = nn.Embedding.from_pretrained(
            pretrained_item_embeddings.to(device), freeze=True, padding_idx=0
        )
        # update projection layers
        self.pretrained_dim = pretrained_item_embeddings.size(1)
        self.projection_layer = nn.Linear(self.pretrained_dim, self.hidden_units).to(device)
        self.domain_alignment_projection_layer = nn.Linear(self.pretrained_dim, self.hidden_units).to(device)
        self.merge_layer = nn.Linear(self.hidden_units * 2, self.hidden_units).to(device)
        print(f"New pretrained embeddings loaded successfully on device: {device}")

    def projection_embeddings(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        if self.projection_layer is None:
            raise ValueError("Projection layer not initialized.")
        return self.projection_layer(item_embeddings)

    def irm_projection_embeddings(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        if self.domain_alignment_projection_layer is None:
            raise ValueError("IRM projection layer not initialized.")
        return self.domain_alignment_projection_layer(item_embeddings)

    def sample_internal_embeddings(
            self, sample_size: int, device: torch.device
    ) -> torch.Tensor:
        if self.pretrained_item_embedding is None:
            raise ValueError("No internal embeddings available.")
        num_items = self.pretrained_item_embedding.num_embeddings
        idx = torch.randint(0, num_items, (sample_size,), device=device)
        return self.pretrained_item_embedding(idx)