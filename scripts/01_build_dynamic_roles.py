from __future__ import annotations
import argparse
from pathlib import Path
from isddg.data.io import load_interactions
from isddg.features.dynamic_roles import (
    build_causal_dynamic_observations,
    fit_gmm_roles,
    save_role_artifacts,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--source", required=True)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--n_bins", type=int, default=20)
    ap.add_argument("--recent_quantile", type=float, default=0.3)
    ap.add_argument("--long_gamma", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out_dir", default="artifacts/roles")
    args = ap.parse_args()

    df, item_map = load_interactions(args.data_root, args.source)
    obs = build_causal_dynamic_observations(
        df, num_items=len(item_map), n_bins=args.n_bins,
        recent_quantile=args.recent_quantile, long_gamma=args.long_gamma,
    )
    art = fit_gmm_roles(obs, num_items=len(item_map), K=args.K, seed=args.seed)
    out = Path(args.out_dir)
    save_role_artifacts(art, out)
    print(f"Saved role artifacts to {out}")
    print(art.centroids)


if __name__ == "__main__":
    main()
