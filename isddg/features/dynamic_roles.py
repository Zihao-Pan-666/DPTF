from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class DynamicRoleArtifacts:
    gmm: GaussianMixture
    role_table: torch.Tensor
    obs_df: pd.DataFrame
    centroids: pd.DataFrame
    diagnostics: Dict
    scaler: StandardScaler | None = None
    feature_cols: List[str] | None = None


def _percentile_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    if len(x) <= 1:
        return np.zeros_like(x, dtype=np.float32)
    return (order / (len(x) - 1)).astype(np.float32)


def _weighted_counts_by_bin(
    data: pd.DataFrame,
    num_items: int,
    tau: int,
    start_bin: int,
    end_bin: int,
    long_gamma: float | None = None,
) -> np.ndarray:
    counts_arr = np.zeros(num_items + 1, dtype=np.float32)
    start_bin = max(0, int(start_bin))
    end_bin = max(start_bin, int(end_bin))

    for b in range(start_bin, end_bin):
        g = data[data["_bin"] == b]
        if g.empty:
            continue
        weight = 1.0
        if long_gamma is not None:
            weight = float(long_gamma ** (tau - 1 - b))
        counts = g["ItemId"].value_counts()
        for item, cnt in counts.items():
            counts_arr[int(item)] += float(cnt) * weight
    return counts_arr


def build_causal_dynamic_observations(
    df: pd.DataFrame,
    num_items: int,
    n_bins: int = 30,
    recent_quantile: float = 0.2,
    very_recent_quantile: float = 0.1,
    mid_quantile: float = 0.5,
    long_gamma: float = 0.9,
) -> pd.DataFrame:
    """
    Build source-only item-time dynamic statistics.

    For each time bin tau, only interactions before tau are used.
    This keeps the role signal causal and avoids future information leakage.

    Output columns include raw percentile popularity at multiple scales and
    derived dynamic features:
    - p_long: exponentially decayed long-term popularity percentile.
    - p_mid: medium-window popularity percentile.
    - p_short: recent-window popularity percentile.
    - p_very_short: very recent popularity percentile.
    - trend: p_short - p_long.
    - accel: p_very_short - p_short.
    - level: average of p_long and p_short.
    - abs_trend: absolute trend magnitude.
    - volatility: std of multi-scale popularity percentiles.
    - support_log: log-scaled historical support.
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
    very_recent_width = max(1, int(np.ceil(n_bins * very_recent_quantile)))
    mid_width = max(recent_width, int(np.ceil(n_bins * mid_quantile)))

    rows = []
    item_ids = np.arange(1, num_items + 1)

    for tau in range(1, n_bins):
        past = data[data["_bin"] < tau]
        if past.empty:
            continue

        long_counts = _weighted_counts_by_bin(data, num_items, tau, 0, tau, long_gamma=long_gamma)
        mid_counts = _weighted_counts_by_bin(data, num_items, tau, max(0, tau - mid_width), tau, long_gamma=None)
        short_counts = _weighted_counts_by_bin(data, num_items, tau, max(0, tau - recent_width), tau, long_gamma=None)
        very_short_counts = _weighted_counts_by_bin(data, num_items, tau, max(0, tau - very_recent_width), tau, long_gamma=None)

        p_long = _percentile_rank(long_counts[1:])
        p_mid = _percentile_rank(mid_counts[1:])
        p_short = _percentile_rank(short_counts[1:])
        p_very_short = _percentile_rank(very_short_counts[1:])

        trend = p_short - p_long
        trend_mid = p_mid - p_long
        accel = p_very_short - p_short
        level = 0.5 * (p_long + p_short)
        abs_trend = np.abs(trend)
        volatility = np.std(np.stack([p_long, p_mid, p_short, p_very_short], axis=1), axis=1)
        support = long_counts[1:] + short_counts[1:]
        support_log = np.log1p(support)

        active = (long_counts[1:] > 0) | (mid_counts[1:] > 0) | (short_counts[1:] > 0) | (very_short_counts[1:] > 0)
        for idx, item in enumerate(item_ids):
            if not active[idx]:
                continue
            rows.append({
                "ItemId": int(item),
                "tau": int(tau),
                "p_long": float(p_long[idx]),
                "p_mid": float(p_mid[idx]),
                "p_short": float(p_short[idx]),
                "p_very_short": float(p_very_short[idx]),
                "delta": float(trend[idx]),
                "trend": float(trend[idx]),
                "trend_mid": float(trend_mid[idx]),
                "accel": float(accel[idx]),
                "level": float(level[idx]),
                "abs_trend": float(abs_trend[idx]),
                "volatility": float(volatility[idx]),
                "support_log": float(support_log[idx]),
            })

    if not rows:
        raise ValueError("No dynamic observations were produced; check timestamps and min sequence length.")
    return pd.DataFrame(rows)


def _cluster_features(
    obs: pd.DataFrame,
    feature_mode: str,
    trend_weight: float,
    abs_trend_weight: float,
    accel_weight: float,
):
    if feature_mode == "raw":
        cols = ["p_long", "p_short", "delta"]
        X = obs[cols].to_numpy(dtype=np.float32)
        scaler = None
        return X, cols, scaler

    if feature_mode == "level_trend":
        cols = ["level", "trend", "abs_trend"]
        X = obs[cols].to_numpy(dtype=np.float32).copy()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X[:, 1] *= float(trend_weight)
        X[:, 2] *= float(abs_trend_weight)
        return X.astype(np.float32), cols, scaler

    if feature_mode == "multi_scale":
        cols = ["level", "trend", "trend_mid", "abs_trend", "accel", "volatility", "support_log"]
        X = obs[cols].to_numpy(dtype=np.float32).copy()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X[:, cols.index("trend")] *= float(trend_weight)
        X[:, cols.index("abs_trend")] *= float(abs_trend_weight)
        X[:, cols.index("accel")] *= float(accel_weight)
        return X.astype(np.float32), cols, scaler

    raise ValueError(f"Unknown feature_mode={feature_mode}")


def _adaptive_role_names(centroids: pd.DataFrame) -> List[str]:
    c = centroids.copy()
    c["role_name"] = "mixed"

    level = c["level"].to_numpy()
    trend = c["trend"].to_numpy()
    abs_trend = np.abs(trend)

    rising_idx = int(np.argmax(trend))
    declining_idx = int(np.argmin(trend))
    head_idx = int(np.argmax(level))
    tail_idx = int(np.argmin(level))

    names = ["mixed"] * len(c)

    # Assign dynamic extremes first if there is any separation at all.
    # These are adaptive labels; they should be interpreted as relative roles.
    names[rising_idx] = "rising-like"
    names[declining_idx] = "declining-like"

    # Stable head/tail should not overwrite strong trend extremes unless no alternative exists.
    for idx, name in [(head_idx, "stable-head"), (tail_idx, "stable-tail")]:
        if names[idx] == "mixed":
            names[idx] = name

    # If duplicates remain due to small K or overlapping extrema, use the second-best candidates.
    if len(set(names)) < len(names):
        order_head = list(np.argsort(-level))
        order_tail = list(np.argsort(level))
        for idx in order_head:
            if "stable-head" not in names and names[idx] == "mixed":
                names[int(idx)] = "stable-head"
                break
        for idx in order_tail:
            if "stable-tail" not in names and names[idx] == "mixed":
                names[int(idx)] = "stable-tail"
                break

    # Final cleanup.
    for i, name in enumerate(names):
        if name == "mixed":
            if trend[i] > 0:
                names[i] = "mild-rising"
            elif trend[i] < 0:
                names[i] = "mild-declining"
            else:
                names[i] = "stable-mixed"
    return names


def _diagnostics(obs_role: pd.DataFrame, probs: np.ndarray, centroids: pd.DataFrame, min_role_mass: float) -> Dict:
    role_mass = probs.mean(axis=0)
    entropy = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1)
    hard = probs.argmax(axis=1)
    hard_counts = {int(k): int((hard == k).sum()) for k in range(probs.shape[1])}

    trend_values = centroids["trend"].to_numpy(dtype=np.float32)
    level_values = centroids["level"].to_numpy(dtype=np.float32)

    return {
        "role_mass": {str(i): float(x) for i, x in enumerate(role_mass.tolist())},
        "hard_counts": hard_counts,
        "mean_assignment_entropy": float(entropy.mean()),
        "median_assignment_entropy": float(np.median(entropy)),
        "min_role_mass": float(role_mass.min()),
        "max_role_mass": float(role_mass.max()),
        "num_roles_below_min_mass": int(np.sum(role_mass < float(min_role_mass))),
        "trend_centroid_range": float(trend_values.max() - trend_values.min()),
        "level_centroid_range": float(level_values.max() - level_values.min()),
        "centroid_names": centroids[["role", "suggested_name"]].to_dict(orient="records"),
        "warnings": [],
    }


def fit_gmm_roles(
    obs: pd.DataFrame,
    num_items: int,
    K: int = 4,
    seed: int = 2026,
    feature_mode: str = "level_trend",
    trend_weight: float = 3.0,
    abs_trend_weight: float = 1.5,
    accel_weight: float = 1.0,
    min_role_mass: float = 0.03,
    covariance_type: str = "full",
) -> DynamicRoleArtifacts:
    """
    Fit soft dynamic roles with a trend-aware GMM.

    Backward-compatible default output:
    - source_role_table.pt remains [num_items+1, K].
    - source_role_observations.parquet contains role probabilities.
    - role_centroids.csv contains human-readable diagnostics.

    Compared with the old [p_long, p_short, delta] input, the default
    level_trend mode standardizes features and upweights trend, preventing
    GMM from degenerating into simple popularity-level clustering.
    """
    X_cluster, feature_cols, scaler = _cluster_features(
        obs=obs,
        feature_mode=feature_mode,
        trend_weight=trend_weight,
        abs_trend_weight=abs_trend_weight,
        accel_weight=accel_weight,
    )

    gmm = GaussianMixture(
        n_components=K,
        covariance_type=covariance_type,
        random_state=seed,
        reg_covar=1e-5,
        n_init=5,
    )
    gmm.fit(X_cluster)
    probs = gmm.predict_proba(X_cluster).astype(np.float32)
    labels = probs.argmax(axis=1)

    obs_role = obs.copy()
    for k in range(K):
        obs_role[f"role_{k}"] = probs[:, k]
    obs_role["hard_role"] = labels.astype(int)
    obs_role["assignment_entropy"] = (-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1)).astype(np.float32)

    role_table = np.zeros((num_items + 1, K), dtype=np.float32)
    counts = np.zeros((num_items + 1, 1), dtype=np.float32)
    for item, p in zip(obs_role["ItemId"].astype(int).tolist(), probs):
        role_table[item] += p
        counts[item] += 1.0
    role_table = role_table / np.maximum(counts, 1.0)

    global_prior = probs.mean(axis=0)
    unseen = counts[:, 0] == 0
    role_table[unseen] = global_prior
    role_table[0] = 0.0

    centroid_rows = []
    centroid_cols = [
        "p_long", "p_mid", "p_short", "p_very_short",
        "delta", "trend", "trend_mid", "accel",
        "level", "abs_trend", "volatility", "support_log",
    ]
    for k in range(K):
        mask = labels == k
        if mask.any():
            vals = obs.loc[mask, centroid_cols].mean(axis=0)
            mass = float(mask.mean())
            soft_mass = float(probs[:, k].mean())
            ent = float(obs_role.loc[mask, "assignment_entropy"].mean())
        else:
            vals = pd.Series({col: 0.0 for col in centroid_cols})
            mass = 0.0
            soft_mass = 0.0
            ent = 0.0

        row = {"role": k}
        for col in centroid_cols:
            row[col] = float(vals[col])
        row["mass"] = mass
        row["soft_mass"] = soft_mass
        row["mean_assignment_entropy"] = ent
        centroid_rows.append(row)

    centroids = pd.DataFrame(centroid_rows)
    centroids["suggested_name"] = _adaptive_role_names(centroids)

    diagnostics = _diagnostics(obs_role, probs, centroids, min_role_mass=min_role_mass)
    if diagnostics["num_roles_below_min_mass"] > 0:
        diagnostics["warnings"].append(
            f"{diagnostics['num_roles_below_min_mass']} roles have soft mass below {min_role_mass}; consider smaller K or diag covariance."
        )
    if diagnostics["trend_centroid_range"] < 0.05:
        diagnostics["warnings"].append(
            "Trend centroid range is small; roles may still be dominated by popularity level. Try larger trend_weight or smaller recent_quantile."
        )

    return DynamicRoleArtifacts(
        gmm=gmm,
        role_table=torch.from_numpy(role_table),
        obs_df=obs_role,
        centroids=centroids,
        diagnostics=diagnostics,
        scaler=scaler,
        feature_cols=feature_cols,
    )


def suggest_role_name(p_long: float, p_short: float, delta: float) -> str:
    # Backward-compatible helper. New code uses adaptive naming.
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
    if art.scaler is not None:
        joblib.dump(art.scaler, out / "role_feature_scaler.pkl")
    torch.save(art.role_table, out / "source_role_table.pt")
    art.obs_df.to_parquet(out / "source_role_observations.parquet", index=False)
    art.centroids.to_csv(out / "role_centroids.csv", index=False)
    pd.DataFrame([art.diagnostics]).to_json(out / "role_diagnostics.json", orient="records", indent=2, force_ascii=False)
    if art.feature_cols is not None:
        (out / "role_feature_cols.txt").write_text("\n".join(art.feature_cols), encoding="utf-8")
