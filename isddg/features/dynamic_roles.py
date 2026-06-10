from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
import joblib


@dataclass
class DynamicRoleArtifacts:
    gmm: GaussianMixture
    role_table: torch.Tensor  # [num_items+1, K], source item-level role prior supervision
    obs_df: pd.DataFrame      # item-time observation table with role probabilities
    centroids: pd.DataFrame


def _percentile_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    if len(x) <= 1:
        return np.zeros_like(x, dtype=np.float32)
    return (order / (len(x) - 1)).astype(np.float32)


def build_causal_dynamic_observations(
    df: pd.DataFrame,
    num_items: int,
    n_bins: int = 20,
    recent_quantile: float = 0.3,
    long_gamma: float = 0.9,
) -> pd.DataFrame:
    """Build source-only item-time dynamic statistics.

    For each time bin tau, only interactions before tau are used. This is intentionally
    conservative and avoids future information leakage.
    """
    if df.empty:
        raise ValueError("Empty interaction dataframe.")
    data = df.sort_values("Timestamp").copy()
    t_min, t_max = float(data["Timestamp"].min()), float(data["Timestamp"].max())
    if t_max <= t_min:
        data["_bin"] = 0
    else:
        data["_bin"] = np.floor((data["Timestamp"] - t_min) / (t_max - t_min + 1e-9) * n_bins).astype(int)
        data["_bin"] = data["_bin"].clip(0, n_bins - 1)

    recent_width = max(1, int(np.ceil(n_bins * recent_quantile)))
    rows = []
    for tau in range(1, n_bins):
        past = data[data["_bin"] < tau]
        if past.empty:
            continue
        long_counts = np.zeros(num_items + 1, dtype=np.float32)
        recent_counts = np.zeros(num_items + 1, dtype=np.float32)

        for b in range(tau):
            g = past[past["_bin"] == b]
            if g.empty:
                continue
            counts = g["ItemId"].value_counts()
            weight = long_gamma ** (tau - 1 - b)
            for item, cnt in counts.items():
                long_counts[int(item)] += float(cnt) * weight

        recent_start = max(0, tau - recent_width)
        recent = past[past["_bin"] >= recent_start]
        counts = recent["ItemId"].value_counts()
        for item, cnt in counts.items():
            recent_counts[int(item)] += float(cnt)

        item_ids = np.arange(1, num_items + 1)
        p_long = _percentile_rank(long_counts[1:])
        p_short = _percentile_rank(recent_counts[1:])
        delta = p_short - p_long
        active = (long_counts[1:] > 0) | (recent_counts[1:] > 0)
        for idx, item in enumerate(item_ids):
            if not active[idx]:
                continue
            rows.append({
                "ItemId": int(item),
                "tau": int(tau),
                "p_long": float(p_long[idx]),
                "p_short": float(p_short[idx]),
                "delta": float(delta[idx]),
            })
    if not rows:
        raise ValueError("No dynamic observations were produced; check timestamps and min sequence length.")
    return pd.DataFrame(rows)


def fit_gmm_roles(obs: pd.DataFrame, num_items: int, K: int = 4, seed: int = 2026) -> DynamicRoleArtifacts:
    X = obs[["p_long", "p_short", "delta"]].to_numpy(dtype=np.float32)
    gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=seed, reg_covar=1e-5)
    gmm.fit(X)
    probs = gmm.predict_proba(X).astype(np.float32)
    obs_role = obs.copy()
    for k in range(K):
        obs_role[f"role_{k}"] = probs[:, k]

    role_table = np.zeros((num_items + 1, K), dtype=np.float32)
    counts = np.zeros((num_items + 1, 1), dtype=np.float32)
    for item, p in zip(obs_role["ItemId"].astype(int).tolist(), probs):
        role_table[item] += p
        counts[item] += 1.0
    role_table = role_table / np.maximum(counts, 1.0)
    # Items unseen in dynamic observations get the global role prior.
    global_prior = probs.mean(axis=0)
    unseen = counts[:, 0] == 0
    role_table[unseen] = global_prior
    role_table[0] = 0.0

    labels = gmm.predict(X)
    centroids = []
    for k in range(K):
        mask = labels == k
        if mask.any():
            vals = X[mask].mean(axis=0)
            mass = float(mask.mean())
        else:
            vals = np.zeros(3, dtype=np.float32)
            mass = 0.0
        centroids.append({
            "role": k, "p_long": vals[0], "p_short": vals[1], "delta": vals[2],
            "mass": mass, "suggested_name": suggest_role_name(vals[0], vals[1], vals[2])
        })
    centroids = pd.DataFrame(centroids)
    return DynamicRoleArtifacts(gmm=gmm, role_table=torch.from_numpy(role_table), obs_df=obs_role, centroids=centroids)


def suggest_role_name(p_long: float, p_short: float, delta: float) -> str:
    if p_long > 0.65 and p_short > 0.65 and abs(delta) < 0.2:
        return "stable-head"
    if delta > 0.25:
        return "rising"
    if delta < -0.25:
        return "declining"
    if p_long < 0.35 and p_short < 0.35:
        return "tail"
    return "mixed"


def save_role_artifacts(art: DynamicRoleArtifacts, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(art.gmm, out / "gmm.pkl")
    torch.save(art.role_table, out / "source_role_table.pt")
    art.obs_df.to_parquet(out / "source_role_observations.parquet", index=False)
    art.centroids.to_csv(out / "role_centroids.csv", index=False)
