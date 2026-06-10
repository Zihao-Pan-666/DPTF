from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from .backbone import FeatureBERT4Rec


class ISDDGModel(nn.Module):
    """Initial ISDDG model.

    The source-trained sequence encoder provides semantic score. The role/prototype
    branch provides candidate-aware dynamic score:

        s = s_sem + lambda_dyn * gamma_u * beta_c * gate(u,c) * s_dyn
    """
    def __init__(
        self,
        backbone: FeatureBERT4Rec,
        role_table: torch.Tensor,
        prototype_keys: torch.Tensor | None = None,
        prototype_values: torch.Tensor | None = None,
        top_m: int = 16,
        proto_temperature: float = 0.2,
        lambda_dyn: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("role_table", role_table.float(), persistent=False)
        self.prototype_keys = None
        self.prototype_values = None
        if prototype_keys is not None and prototype_values is not None:
            self.register_buffer("prototype_keys", prototype_keys.float(), persistent=False)
            self.register_buffer("prototype_values", prototype_values.float(), persistent=False)
        self.top_m = top_m
        self.proto_temperature = proto_temperature
        self.lambda_dyn = lambda_dyn
        role_dim = role_table.size(-1)
        hidden_dim = backbone.hidden_dim
        self.key_proj = nn.Linear(hidden_dim + role_dim + role_dim * role_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + role_dim + role_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def set_item_features(self, item_features: torch.Tensor, role_table: torch.Tensor | None = None):
        if role_table is not None:
            self.role_table = role_table.float().to(next(self.parameters()).device)
            self.backbone.set_item_features(item_features, role_table)
        else:
            self.backbone.set_item_features(item_features)

    def _local_context(self, history: torch.Tensor, eta: float = 0.2):
        roles = self.role_table[history].detach()  # [B,L,K]
        mask = history.ne(0).float()
        B, L, K = roles.shape
        pos = torch.arange(L, device=history.device).float()
        # More recent positions have larger weights.
        recency = torch.exp(-eta * (L - 1 - pos)).view(1, L) * mask
        weights = recency / recency.sum(dim=1, keepdim=True).clamp(min=1e-8)
        m = torch.einsum("bl,blk->bk", weights, roles)
        prev = roles[:, :-1, :]
        nxt = roles[:, 1:, :]
        w2 = weights[:, 1:]
        A = torch.einsum("bl,blk,blj->bkj", w2, prev, nxt)
        return m, A

    def build_query_key(self, history: torch.Tensor, user_h: torch.Tensor | None = None):
        if user_h is None:
            user_h = self.backbone(history)
        m, A = self._local_context(history)
        key_in = torch.cat([user_h, m, A.flatten(start_dim=1)], dim=-1)
        return self.key_proj(key_in), m, A

    def retrieve_next_role(self, query_key: torch.Tensor):
        if self.prototype_keys is None or self.prototype_values is None:
            B = query_key.size(0)
            K = self.role_table.size(-1)
            uniform = torch.full((B, K), 1.0 / K, device=query_key.device)
            conf = torch.zeros(B, device=query_key.device)
            return uniform, conf, None

        keys = F.normalize(self.prototype_keys, dim=-1)
        q = F.normalize(query_key, dim=-1)
        sim = q @ keys.t()
        top_m = min(self.top_m, sim.size(1))
        vals, idx = torch.topk(sim, k=top_m, dim=1)
        attn = torch.softmax(vals / max(self.proto_temperature, 1e-6), dim=1)
        proto_vals = self.prototype_values[idx]  # [B,top,K]
        rho_next = torch.einsum("bt,btk->bk", attn, proto_vals)
        entropy = -(attn * (attn.clamp(min=1e-8).log())).sum(dim=1)
        conf = 1.0 - entropy / torch.log(torch.tensor(float(top_m), device=query_key.device))
        return rho_next, conf.clamp(0, 1), idx

    def score(self, history: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        user_h = self.backbone(history)
        cand_h = self.backbone.encode_items(candidates)
        s_sem = torch.einsum("bd,bnd->bn", user_h, cand_h)

        q, _, _ = self.build_query_key(history, user_h)
        rho_next, gamma, _ = self.retrieve_next_role(q)
        cand_role = self.role_table[candidates].detach()
        s_dyn = torch.einsum("bk,bnk->bn", rho_next, cand_role)
        beta = cand_role.max(dim=-1).values

        gate_in = torch.cat([
            user_h.unsqueeze(1).expand(-1, candidates.size(1), -1),
            rho_next.unsqueeze(1).expand(-1, candidates.size(1), -1),
            cand_role,
        ], dim=-1)
        g = self.gate(gate_in).squeeze(-1)
        return s_sem + self.lambda_dyn * gamma.unsqueeze(1) * beta * g * s_dyn
