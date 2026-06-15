from __future__ import annotations

import argparse
import time
from pathlib import Path

from isddg.config import load_config, ensure_dirs
from isddg.utils.io import save_json
from isddg.data.io import load_interactions
from isddg.features.dynamic_roles import (
    build_causal_dynamic_observations,
    fit_gmm_roles,
    save_role_artifacts,
)


def cfg_get(cfg, section, key, default):
    return cfg.get(section, {}).get(key, default)


def main():
    ap = argparse.ArgumentParser(description="Build enhanced source-domain dynamic role supervision.")
    ap.add_argument("--config", default="configs/dynamic_role_signal_v2.yaml")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--n_bins", type=int, default=None)
    ap.add_argument("--recent_quantile", type=float, default=None)
    ap.add_argument("--very_recent_quantile", type=float, default=None)
    ap.add_argument("--mid_quantile", type=float, default=None)
    ap.add_argument("--long_gamma", type=float, default=None)
    ap.add_argument("--feature_mode", default=None, choices=["raw", "level_trend", "multi_scale"])
    ap.add_argument("--trend_weight", type=float, default=None)
    ap.add_argument("--abs_trend_weight", type=float, default=None)
    ap.add_argument("--accel_weight", type=float, default=None)
    ap.add_argument("--min_role_mass", type=float, default=None)
    ap.add_argument("--covariance_type", default=None, choices=["full", "diag", "tied", "spherical"])
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    data_root = args.data_root or cfg.get("data_root", "./data")
    source = args.source or cfg.get("source", "amazon_movies_and_tv")
    out_dir = Path(args.out_dir or cfg.get("paths", {}).get("role_dir", "artifacts/roles_v2"))

    K = args.K if args.K is not None else cfg_get(cfg, "roles", "K", 4)
    n_bins = args.n_bins if args.n_bins is not None else cfg_get(cfg, "roles", "n_bins", 30)
    recent_quantile = args.recent_quantile if args.recent_quantile is not None else cfg_get(cfg, "roles", "recent_quantile", 0.2)
    very_recent_quantile = args.very_recent_quantile if args.very_recent_quantile is not None else cfg_get(cfg, "roles", "very_recent_quantile", 0.1)
    mid_quantile = args.mid_quantile if args.mid_quantile is not None else cfg_get(cfg, "roles", "mid_quantile", 0.5)
    long_gamma = args.long_gamma if args.long_gamma is not None else cfg_get(cfg, "roles", "long_gamma", 0.9)
    feature_mode = args.feature_mode or cfg_get(cfg, "roles", "feature_mode", "level_trend")
    trend_weight = args.trend_weight if args.trend_weight is not None else cfg_get(cfg, "roles", "trend_weight", 3.0)
    abs_trend_weight = args.abs_trend_weight if args.abs_trend_weight is not None else cfg_get(cfg, "roles", "abs_trend_weight", 1.5)
    accel_weight = args.accel_weight if args.accel_weight is not None else cfg_get(cfg, "roles", "accel_weight", 1.0)
    min_role_mass = args.min_role_mass if args.min_role_mass is not None else cfg_get(cfg, "roles", "min_role_mass", 0.03)
    covariance_type = args.covariance_type or cfg_get(cfg, "roles", "covariance_type", "full")
    seed = args.seed if args.seed is not None else cfg_get(cfg, "data", "seed", 2026)

    t0 = time.time()
    print("=" * 80, flush=True)
    print("[DynamicRolesV2] Build enhanced source dynamic roles", flush=True)
    print(f"source={source} data_root={data_root}", flush=True)
    print(
        f"K={K} n_bins={n_bins} feature_mode={feature_mode} "
        f"trend_weight={trend_weight} abs_trend_weight={abs_trend_weight}",
        flush=True,
    )
    print("=" * 80, flush=True)

    df, item_map = load_interactions(data_root, source)
    print(f"[Data] interactions={len(df):,} items={len(item_map):,}", flush=True)

    obs = build_causal_dynamic_observations(
        df=df,
        num_items=len(item_map),
        n_bins=n_bins,
        recent_quantile=recent_quantile,
        very_recent_quantile=very_recent_quantile,
        mid_quantile=mid_quantile,
        long_gamma=long_gamma,
    )
    print(f"[Obs] rows={len(obs):,} columns={list(obs.columns)}", flush=True)

    art = fit_gmm_roles(
        obs=obs,
        num_items=len(item_map),
        K=K,
        seed=seed,
        feature_mode=feature_mode,
        trend_weight=trend_weight,
        abs_trend_weight=abs_trend_weight,
        accel_weight=accel_weight,
        min_role_mass=min_role_mass,
        covariance_type=covariance_type,
    )
    save_role_artifacts(art, out_dir)

    summary = {
        "source": source,
        "num_interactions": int(len(df)),
        "num_items": int(len(item_map)),
        "num_observations": int(len(obs)),
        "K": int(K),
        "n_bins": int(n_bins),
        "recent_quantile": float(recent_quantile),
        "very_recent_quantile": float(very_recent_quantile),
        "mid_quantile": float(mid_quantile),
        "long_gamma": float(long_gamma),
        "feature_mode": feature_mode,
        "trend_weight": float(trend_weight),
        "abs_trend_weight": float(abs_trend_weight),
        "accel_weight": float(accel_weight),
        "min_role_mass": float(min_role_mass),
        "covariance_type": covariance_type,
        "seed": int(seed),
        "out_dir": str(out_dir),
        "elapsed_sec": time.time() - t0,
        "diagnostics": art.diagnostics,
        "centroids": art.centroids.to_dict(orient="records"),
    }
    save_json(summary, out_dir / "dynamic_roles_v2_summary.json")

    print("[Centroids]", flush=True)
    print(art.centroids, flush=True)
    print("[Diagnostics]", flush=True)
    print(art.diagnostics, flush=True)
    print(f"[Saved] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
