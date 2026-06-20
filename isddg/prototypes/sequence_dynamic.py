from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass
class SequenceDynamicPrototypeBank:
    """Sequence-level source prototype bank.

    keys:
        Prototype centroids in the source semantic sequence-state space, shape [M, H].
    semantic_values:
        Mean next-item projected semantic vectors for each prototype, shape [M, H].
    dynamic_values:
        Mean next-item continuous dynamic vectors for each prototype, shape [M, D].
        D may be 0 when the bank was built without continuous dynamic values.
    role_values:
        Mean next-item role distributions for each prototype, shape [M, K].
        K may be 0 when the bank was built without role values.
    counts:
        Number of source prefix samples assigned to each prototype.
    meta:
        Serializable metadata for protocol auditing.
    """

    keys: torch.Tensor
    semantic_values: torch.Tensor
    dynamic_values: torch.Tensor
    role_values: torch.Tensor
    counts: torch.Tensor
    meta: Dict[str, Any]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "keys": self.keys.detach().cpu().float(),
                "semantic_values": self.semantic_values.detach().cpu().float(),
                "dynamic_values": self.dynamic_values.detach().cpu().float(),
                "role_values": self.role_values.detach().cpu().float(),
                "counts": self.counts.detach().cpu().long(),
                "meta": dict(self.meta or {}),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SequenceDynamicPrototypeBank":
        obj = torch.load(Path(path), map_location="cpu")
        # Backward compatibility with the old role-only bank format.
        if "values" in obj and "semantic_values" not in obj:
            values = obj["values"].float()
            return cls(
                keys=obj["keys"].float(),
                semantic_values=torch.zeros(obj["keys"].size(0), 0),
                dynamic_values=torch.zeros(obj["keys"].size(0), 0),
                role_values=values,
                counts=obj.get("counts", torch.ones(obj["keys"].size(0))).long(),
                meta={"format": "legacy_role_only"},
            )
        return cls(
            keys=obj["keys"].float(),
            semantic_values=obj.get("semantic_values", torch.zeros(obj["keys"].size(0), 0)).float(),
            dynamic_values=obj.get("dynamic_values", torch.zeros(obj["keys"].size(0), 0)).float(),
            role_values=obj.get("role_values", torch.zeros(obj["keys"].size(0), 0)).float(),
            counts=obj.get("counts", torch.ones(obj["keys"].size(0))).long(),
            meta=dict(obj.get("meta", {})),
        )

    @property
    def num_prototypes(self) -> int:
        return int(self.keys.size(0))

    @property
    def hidden_dim(self) -> int:
        return int(self.keys.size(1)) if self.keys.ndim == 2 else 0

    @property
    def dynamic_dim(self) -> int:
        return int(self.dynamic_values.size(1)) if self.dynamic_values.ndim == 2 else 0

    @property
    def role_dim(self) -> int:
        return int(self.role_values.size(1)) if self.role_values.ndim == 2 else 0

    def to(self, device: torch.device | str) -> "SequenceDynamicPrototypeBank":
        return SequenceDynamicPrototypeBank(
            keys=self.keys.to(device),
            semantic_values=self.semantic_values.to(device),
            dynamic_values=self.dynamic_values.to(device),
            role_values=self.role_values.to(device),
            counts=self.counts.to(device),
            meta=self.meta,
        )
