from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from isddg.config import ensure_dirs, load_config
from isddg.data.io import group_user_sequences, load_interactions
from isddg.data.semantic_splits import build_source_train_val_samples
from isddg.features.dynamic_feature_store import (
    build_continuous_table_from_observations,
    load_pt_feature_table,
)
from isddg.features.semantic import load_semantic_embeddings
from isddg.training.continuous_dynamic_prior_trainer import (
    load_continuous_predictor_from_checkpoint,
    predict_continuous_table,
    train_continuous_dynamic_prior,
)
from isddg.utils.device import get_device
from isddg.utils.io import append_csv, save_json
from isddg.utils.seed import set_seed

def resolve_show_progress(mode: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def cfg_get(cfg: Dict[str, Any], sec: str, key: str, default: Any) -> Any:
    return cfg.get(sec, {}).get(key, default)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Train semantic-conditioned continuous dynamic prior "
            "(weighted regression + optional role auxiliary + optional dynamic BPR)."
        )
    )
    ap.add_argument("--config", default="configs/continuous_dynamic.yaml")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--embedding_dir", default=None)
    ap.add_argument("--observations_path", default=None)
    ap.add_argument("--role_table_path", default=None)
    ap.add_argument("--source_table_path", default=None)
    ap.add_argument("--stats_path", default=None)
    ap.add_argument("--checkpoint_path", default=None)
    ap.add_argument("--results_path", default=None)
    ap.add_argument("--summary_path", default=None)
    ap.add_argument("--pred_source_table_path", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_source_val_ranking", action="store_true")
    ap.add_argument(
        "--progress",
        choices=["auto", "on", "off"],
        default="auto",
        help="Progress bar mode. auto shows tqdm only in an interactive terminal; off is best for overnight logs.",
    )
    args = ap.parse_args()
    show_progress = resolve_show_progress(args.progress)

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    data_root = args.data_root or cfg.get("data_root", "./data")
    source = args.source or cfg.get("source", "amazon_movies_and_tv")
    seed = args.seed if args.seed is not None else cfg_get(cfg, "data", "seed", 2026)
    embedding_dir = args.embedding_dir or cfg_get(cfg, "data", "embedding_dir", "semantic_embeddings")

    set_seed(seed)
    device = get_device(args.device or cfg_get(cfg, "training", "device", "auto"))

    dyn_cfg = cfg.get("continuous_dynamic", {})
    paths = cfg.get("paths", {})
    result_dir = Path(paths.get("result_dir", "results/mainline"))
    dynamic_dir = Path(paths.get("dynamic_dir", f"artifacts/dynamics/{source}_continuous"))
    role_dir = Path(paths.get("role_dir", f"artifacts/roles/{source}_k4_default"))

    obs_path = Path(
        args.observations_path
        or dyn_cfg.get("source_observations_path", role_dir / "source_role_observations.parquet")
    )
    role_table_path = Path(
        args.role_table_path
        or dyn_cfg.get("role_table_path", role_dir / "source_role_table.pt")
    )
    source_table_path = Path(
        args.source_table_path
        or dyn_cfg.get("source_table_path", dynamic_dir / "source_continuous_dynamic_table.pt")
    )
    stats_path = Path(
        args.stats_path
        or dyn_cfg.get("stats_path", dynamic_dir / "continuous_dynamic_stats.json")
    )
    ckpt_path = Path(
        args.checkpoint_path
        or dyn_cfg.get(
            "checkpoint",
            f"artifacts/checkpoints/continuous_dynamic/semantic_conditioned_prior_enhanced_{source}_seed{seed}.pt",
        )
    )
    results_path = Path(args.results_path or result_dir / "continuous_dynamic_prior_source_val.csv")
    summary_path = Path(
        args.summary_path or result_dir / f"continuous_dynamic_prior_{source}_seed{seed}_summary.json"
    )
    pred_path = Path(
        args.pred_source_table_path
        or dynamic_dir / f"pred_source_continuous_dynamic_table_{source}_seed{seed}.pt"
    )

    print("=" * 80)
    print("[ContinuousDynamicPrior] semantic -> continuous dynamic prior")
    print(f"source={source}")
    print(f"observations_path={obs_path}")
    print(f"role_table_path={role_table_path}")
    print(f"source_table_path={source_table_path}")
    print(f"checkpoint_path={ckpt_path}")
    print(f"device={device}")
    print("=" * 80)

    max_len = cfg_get(cfg, "data", "max_len", 50)
    min_len = cfg_get(cfg, "data", "min_len", 3)

    df, item_map = load_interactions(data_root, source)
    seqs = group_user_sequences(df, min_len=min_len)
    train_samples, val_samples = build_source_train_val_samples(
        seqs=seqs,
        max_len=max_len,
        min_prefix=1,
    )

    sem = load_semantic_embeddings(
        data_root=data_root,
        domain=source,
        item_map=item_map,
        embedding_dir=embedding_dir,
        strict=True,
    )[: len(item_map) + 1]
    sem[0] = 0.0

    feature_cols = dyn_cfg.get(
        "feature_cols",
        ["level", "trend", "abs_trend", "trend_mid", "accel", "volatility", "support_log"],
    )

    dyn_table = build_continuous_table_from_observations(
        observations_path=obs_path,
        num_items=len(item_map),
        feature_cols=feature_cols,
        standardize=bool(dyn_cfg.get("standardize", True)),
        stats_path=stats_path,
        out_path=source_table_path,
    )

    role_table = None
    if bool(dyn_cfg.get("use_role_auxiliary", True)):
        if not role_table_path.exists():
            raise FileNotFoundError(
                f"use_role_auxiliary=True but role table not found: {role_table_path}"
            )
        role_table = load_pt_feature_table(role_table_path, num_items=len(item_map))

    source_val_ranking = bool(dyn_cfg.get("source_val_ranking", True)) and (
        not args.no_source_val_ranking
    )

    summary = train_continuous_dynamic_prior(
        item_features=sem,
        target_table=dyn_table,
        checkpoint_path=ckpt_path,
        role_table=role_table,
        source_train_samples=train_samples,
        source_val_samples=val_samples,
        num_items=len(item_map),
        max_len=max_len,
        hidden_dim=int(dyn_cfg.get("hidden_dim", 256)),
        latent_dim=int(dyn_cfg.get("latent_dim", 128)),
        dropout=float(dyn_cfg.get("dropout", 0.1)),
        use_layer_norm=bool(dyn_cfg.get("use_layer_norm", True)),
        l2_normalize_input=bool(dyn_cfg.get("l2_normalize_input", False)),
        epochs=int(dyn_cfg.get("epochs", 80)),
        lr=float(dyn_cfg.get("lr", 3e-4)),
        weight_decay=float(dyn_cfg.get("weight_decay", 1e-5)),
        batch_size=int(dyn_cfg.get("batch_size", 4096)),
        sequence_batch_size=int(dyn_cfg.get("sequence_batch_size", 512)),
        val_ratio=float(dyn_cfg.get("val_ratio", 0.1)),
        patience=int(dyn_cfg.get("patience", 10)),
        seed=seed,
        device=device,
        regression_loss_type=str(dyn_cfg.get("regression_loss_type", "smooth_l1")),
        smooth_l1_beta=float(dyn_cfg.get("smooth_l1_beta", 1.0)),
        dynamic_feature_weights=dyn_cfg.get("feature_weights", None),
        lambda_reg=float(dyn_cfg.get("lambda_reg", 1.0)),
        lambda_role=float(dyn_cfg.get("lambda_role", 0.2)) if role_table is not None else 0.0,
        lambda_bpr=float(dyn_cfg.get("lambda_bpr", 0.1)),
        bpr_negatives=int(dyn_cfg.get("bpr_negatives", 20)),
        bpr_pooling=str(dyn_cfg.get("bpr_pooling", "decay")),
        bpr_recent_k=int(dyn_cfg.get("bpr_recent_k", 5)),
        bpr_decay=float(dyn_cfg.get("bpr_decay", 0.8)),
        source_val_ranking=source_val_ranking,
        val_ranking_negatives=int(cfg_get(cfg, "data", "eval_negatives", 100)),
        val_ranking_batch_size=int(cfg_get(cfg, "training", "batch_size", 128)),
        val_ranking_mode=str(cfg_get(cfg, "data", "ranking", "sampled")),
        val_score_pooling=str(cfg_get(cfg, "late_fusion", "pooling", "decay")),
        val_score_recent_k=int(cfg_get(cfg, "late_fusion", "recent_k", 5)),
        val_score_decay=float(cfg_get(cfg, "late_fusion", "decay", 0.8)),
        val_score_norm=str(cfg_get(cfg, "late_fusion", "score_norm", "zscore")),
        selection_mode=str(dyn_cfg.get("selection_mode", "dynamic_composite")),
        recall_weight=float(dyn_cfg.get("recall_weight", 0.25)),
        show_progress=show_progress,
        checkpoint_extra={
            "cfg": cfg,
            "source": source,
            "seed": seed,
            "feature_cols": feature_cols,
            "source_table_path": str(source_table_path),
            "stats_path": str(stats_path),
            "role_table_path": str(role_table_path) if role_table is not None else "",
            "embedding_dim": int(sem.size(1)),
            "dynamic_dim": int(dyn_table.size(1)),
            "protocol": {
                "name": "semantic_conditioned_continuous_dynamic_prior_enhanced",
                "target_interaction_usage": "none",
                "checkpoint_selection": str(dyn_cfg.get("selection_mode", "dynamic_composite")),
                "source_val_ranking": source_val_ranking,
            },
        },
    )

    model, _ = load_continuous_predictor_from_checkpoint(ckpt_path, device)
    pred_source_table = predict_continuous_table(model, sem, batch_size=4096, device=device)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dynamic_table": pred_source_table,
            "source": source,
            "seed": seed,
            "checkpoint": str(ckpt_path),
            "feature_cols": feature_cols,
            "version": "enhanced",
        },
        pred_path,
    )

    out = {
        **summary,
        "source": source,
        "seed": seed,
        "source_table_path": str(source_table_path),
        "pred_source_table_path": str(pred_path),
        "stats_path": str(stats_path),
        "role_table_path": str(role_table_path) if role_table is not None else "",
        "num_source_items": len(item_map),
        "num_train_samples": len(train_samples),
        "num_source_val_users": len(val_samples),
    }
    save_json(out, summary_path)

    best_epoch = int(summary["best_epoch"])
    best_row = {}
    for row in summary.get("history", []):
        if int(row.get("epoch", -1)) == best_epoch:
            best_row = row
            break

    append_csv(
        {
            "model": "semantic_conditioned_continuous_dynamic_prior",
            "stage": "source_prior",
            "source": source,
            "seed": seed,
            "checkpoint": str(ckpt_path),
            "best_epoch": best_epoch,
            "selection_mode": summary["selection_mode"],
            "best_selection_value": summary["best_selection_value"],
            "dynamic_dim": int(dyn_table.size(1)),
            "embedding_dim": int(sem.size(1)),
            "lambda_reg": float(dyn_cfg.get("lambda_reg", 1.0)),
            "lambda_role": float(dyn_cfg.get("lambda_role", 0.2)) if role_table is not None else 0.0,
            "lambda_bpr": float(dyn_cfg.get("lambda_bpr", 0.1)),
            "source_table_path": str(source_table_path),
            "pred_source_table_path": str(pred_path),
            **{f"best_{k}": v for k, v in best_row.items() if isinstance(v, (int, float, str))},
        },
        results_path,
    )

    print(f"[Saved] source_table={source_table_path}")
    print(f"[Saved] pred_source_table={pred_path}")
    print(f"[Saved] summary={summary_path}")
    print(f"[Saved] results_csv={results_path}")


if __name__ == "__main__":
    main()
