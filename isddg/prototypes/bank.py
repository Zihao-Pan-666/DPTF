from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans


@dataclass
class PrototypeBank:
    keys: torch.Tensor
    values: torch.Tensor
    counts: torch.Tensor

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"keys": self.keys, "values": self.values, "counts": self.counts}, path)

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeBank":
        obj = torch.load(path, map_location="cpu")
        return cls(keys=obj["keys"], values=obj["values"], counts=obj["counts"])


def build_prototype_bank(
    keys: torch.Tensor,
    next_roles: torch.Tensor,
    M: int = 128,
    seed: int = 2026,
    batch_size: int = 4096,
) -> PrototypeBank:
    X = keys.detach().cpu().numpy().astype(np.float32)
    Y = next_roles.detach().cpu().numpy().astype(np.float32)
    M = min(M, len(X))
    km = MiniBatchKMeans(n_clusters=M, random_state=seed, batch_size=min(batch_size, max(M * 4, 256)))
    labels = km.fit_predict(X)
    centers = km.cluster_centers_.astype(np.float32)
    K = Y.shape[1]
    values = np.zeros((M, K), dtype=np.float32)
    counts = np.zeros(M, dtype=np.int64)
    for lab, y in zip(labels, Y):
        values[lab] += y
        counts[lab] += 1
    values = values / np.maximum(counts[:, None], 1)
    # Empty clusters, if any, receive global role prior.
    global_prior = Y.mean(axis=0)
    empty = counts == 0
    values[empty] = global_prior
    return PrototypeBank(
        keys=torch.from_numpy(centers),
        values=torch.from_numpy(values),
        counts=torch.from_numpy(counts),
    )
