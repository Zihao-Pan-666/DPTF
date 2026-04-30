import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np


class GRU4RecWithDomainAlignment(nn.Module):
    def __init__(self, hidden_units, num_layers, dropout_rate, pretrained_item_embeddings=None,
                 num_sequential_patterns=10):
        """
        GRU4Rec with dynamic pretrained embeddings and sequential pattern generalization.
        Args:
            hidden_units (int): Hidden size for the GRU and embeddings.
            num_layers (int): Number of GRU layers.
            dropout_rate (float): Dropout rate.
            pretrained_item_embeddings (Tensor, optional): Pretrained item embeddings.
            num_sequential_patterns (int): Number of sequential patterns to extract from source domain.
        """
        super(GRU4RecWithDomainAlignment, self).__init__()

        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_sequential_patterns = num_sequential_patterns

        if pretrained_item_embeddings is not None:
            self.pretrained_dim = pretrained_item_embeddings.shape[1]
            self.pretrained_item_embedding = nn.Embedding.from_pretrained(
                pretrained_item_embeddings, freeze=True, padding_idx=0
            )
            self.projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
            self.domain_alignment_projection_layer = nn.Linear(self.pretrained_dim, hidden_units)
        else:
            self.pretrained_item_embedding = None
            self.projection_layer = None

        # GRU for user sequence representation
        self.gru = nn.GRU(input_size=hidden_units, hidden_size=hidden_units, num_layers=num_layers,
                          batch_first=True, dropout=dropout_rate)
        self.merge_layer = nn.Linear(hidden_units * 2, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)

        # Sequential pattern components
        self.sequential_patterns = None  # Will store k sequential patterns
        self.pattern_fusion_layer = nn.Linear(hidden_units * 2, hidden_units)  # W_f projection matrix
        self.use_sequential_patterns = False

    def forward(self, item_seq, is_target_domain=False):
        """
        Forward pass to compute logits for all items.
        Args:
            item_seq (Tensor): Input item sequences, shape (batch_size, seq_len).
            is_target_domain (bool): Whether this is target domain inference.
        Returns:
            logits (Tensor): Logits for all items.
        """
        device = item_seq.device

        # Use pretrained embeddings for items
        if self.pretrained_item_embedding is not None:
            pretrained_emb = self.pretrained_item_embedding(item_seq).to(device)
            item_emb = self.projection_layer(pretrained_emb).to(device)
            item_irm_emb = self.domain_alignment_projection_layer(pretrained_emb).to(device)
        else:
            raise ValueError("No pretrained embeddings loaded. Use `load_new_pretrain_embeddings` first.")

        item_emb = self.dropout(item_emb)
        item_irm_emb = self.dropout(item_irm_emb)
        merged_item_emb = torch.cat((item_emb, item_irm_emb), dim=-1)
        merged_item_emb = self.merge_layer(merged_item_emb)

        # Get user sequence representation
        _, user_rep = self.gru(merged_item_emb)
        user_rep = user_rep[-1]  # Take the last GRU layer's output (batch_size, hidden_units)

        # Apply sequential pattern attention if in target domain and patterns are available
        if is_target_domain and self.use_sequential_patterns and self.sequential_patterns is not None:
            user_rep = self._apply_sequential_pattern_attention(user_rep)

        # Compute logits for items
        item_embeddings = self.projection_layer(self.pretrained_item_embedding.weight)
        logits = torch.matmul(user_rep, item_embeddings.T)

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
            if self.pretrained_item_embedding is not None:
                pretrained_emb = self.pretrained_item_embedding(source_sequences).to(device)
                item_emb = self.projection_layer(pretrained_emb).to(device)
                item_irm_emb = self.domain_alignment_projection_layer(pretrained_emb).to(device)
            else:
                raise ValueError("No pretrained embeddings loaded.")

            item_emb = self.dropout(item_emb)
            item_irm_emb = self.dropout(item_irm_emb)
            merged_item_emb = torch.cat((item_emb, item_irm_emb), dim=-1)
            merged_item_emb = self.merge_layer(merged_item_emb)

            # Get sequence embeddings
            _, source_user_embeddings = self.gru(merged_item_emb)
            source_user_embeddings = source_user_embeddings[-1]  # (num_users, hidden_units)

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
            # Encode target sequences
            pretrained_emb = self.pretrained_item_embedding(target_sequences).to(device)
            item_emb = self.projection_layer(pretrained_emb).to(device)
            item_irm_emb = self.domain_alignment_projection_layer(pretrained_emb).to(device)

            item_emb = self.dropout(item_emb)
            item_irm_emb = self.dropout(item_irm_emb)
            merged_item_emb = torch.cat((item_emb, item_irm_emb), dim=-1)
            merged_item_emb = self.merge_layer(merged_item_emb)

            _, user_embeddings = self.gru(merged_item_emb)
            user_embeddings = user_embeddings[-1]

            # Compute cosine similarity and attention weights
            user_norm = F.normalize(user_embeddings, p=2, dim=1)
            pattern_norm = F.normalize(self.sequential_patterns, p=2, dim=1)
            similarity_scores = torch.matmul(user_norm, pattern_norm.T)
            attention_weights = F.softmax(similarity_scores, dim=1)

        return attention_weights

    def load_new_pretrain_embeddings(self, pretrained_item_embeddings):
        """
        Load new pretrained item embeddings dynamically.
        Args:
            pretrained_item_embeddings (Tensor): New pretrained item embeddings.
        """
        device = next(self.parameters()).device
        self.pretrained_item_embedding = nn.Embedding.from_pretrained(
            pretrained_item_embeddings.to(device), freeze=True, padding_idx=0
        )
        print(f"New pretrained embeddings loaded successfully on device: {device}")

    def projection_embeddings(self, item_embeddings):
        if self.pretrained_item_embedding is None:
            raise ValueError("No projection layer included in current model.")
        return self.projection_layer(item_embeddings)

    def irm_projection_embeddings(self, item_embeddings):
        if self.pretrained_item_embedding is None:
            raise ValueError("No projection layer included in current model.")
        return self.domain_alignment_projection_layer(item_embeddings)

    def sample_internal_embeddings(self, sample_size, device):
        """
        Sample embeddings directly from the internal item embedding table.
        """
        if self.pretrained_item_embedding is None:
            raise ValueError("No internal item embedding included in the current model.")

        num_items = self.pretrained_item_embedding.num_embeddings
        sampled_indices = torch.randint(0, num_items, (sample_size,), device=device)
        sampled_embeddings = self.pretrained_item_embedding(sampled_indices)
        return sampled_embeddings

    def predict(self, item_seq, candidate_items=None, is_target_domain=False):
        """
        Predict scores for candidate items or all items using dot product similarity.
        Args:
            item_seq (Tensor): Input item sequences, shape (batch_size, seq_len).
            candidate_items (Tensor, optional): Candidate items for ranking, shape (batch_size, num_candidates).
            is_target_domain (bool): Whether this is target domain inference.
        Returns:
            scores (Tensor): Scores for candidate items or all items.
        """
        logits = self.forward(item_seq, is_target_domain=is_target_domain)
        if candidate_items is not None:
            scores = torch.gather(logits, dim=1, index=candidate_items)
            return scores
        return logits