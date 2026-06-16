from __future__ import annotations
import argparse
from pathlib import Path
import torch
from isddg.config import load_config, ensure_dirs
from isddg.utils.seed import set_seed
from isddg.utils.device import get_device
from isddg.utils.io import save_json, append_csv
from isddg.data.io import load_interactions
from isddg.features.semantic import load_semantic_embeddings
from isddg.features.dynamic_feature_store import build_continuous_table_from_observations
from isddg.training.continuous_dynamic_prior_trainer import train_continuous_dynamic_prior, load_continuous_predictor_from_checkpoint, predict_continuous_table

def cfg_get(cfg, sec, key, default):
    return cfg.get(sec, {}).get(key, default)

def main():
    ap = argparse.ArgumentParser(description="Train semantic-to-continuous-dynamic prior predictor.")
    ap.add_argument("--config", default="configs/dynamic_signal_diagnostics_v1.yaml")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--embedding_dir", default="semantic_embeddings")
    ap.add_argument("--observations_path", default=None)
    ap.add_argument("--source_table_path", default=None)
    ap.add_argument("--stats_path", default=None)
    ap.add_argument("--checkpoint_path", default=None)
    ap.add_argument("--results_path", default=None)
    ap.add_argument("--summary_path", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config); ensure_dirs(cfg)
    data_root = args.data_root or cfg.get("data_root", "./data")
    source = args.source or cfg.get("source")
    seed = args.seed if args.seed is not None else cfg_get(cfg, "data", "seed", 2026)
    set_seed(seed)
    device = get_device(args.device or cfg_get(cfg, "training", "device", "auto"))

    obs_path = Path(args.observations_path or cfg_get(cfg, "continuous_dynamic", "source_observations_path", ""))
    source_table_path = Path(args.source_table_path or cfg_get(cfg, "continuous_dynamic", "source_table_path", "artifacts/dynamics/source_continuous_dynamic_table.pt"))
    stats_path = Path(args.stats_path or cfg_get(cfg, "continuous_dynamic", "stats_path", "artifacts/dynamics/continuous_dynamic_stats.json"))
    ckpt_path = Path(args.checkpoint_path or cfg_get(cfg, "continuous_dynamic", "checkpoint", f"artifacts/checkpoints/continuous_dynamic_prior_{source}_seed{seed}.pt"))
    result_dir = Path(cfg["paths"]["result_dir"])
    results_path = Path(args.results_path or result_dir / "continuous_dynamic_prior_source_val.csv")
    summary_path = Path(args.summary_path or result_dir / f"continuous_dynamic_prior_{source}_seed{seed}_summary.json")

    print("="*80)
    print("[ContinuousDynamicPrior] Train semantic -> continuous dynamic feature predictor")
    print(f"source={source}")
    print(f"observations_path={obs_path}")
    print(f"checkpoint_path={ckpt_path}")
    print("="*80)

    df, item_map = load_interactions(data_root, source)
    sem = load_semantic_embeddings(data_root=data_root, domain=source, item_map=item_map, embedding_dir=args.embedding_dir, strict=True)[:len(item_map)+1]
    sem[0] = 0.0

    feature_cols = cfg_get(cfg, "continuous_dynamic", "feature_cols", None)
    dyn_table = build_continuous_table_from_observations(
        observations_path=obs_path,
        num_items=len(item_map),
        feature_cols=feature_cols,
        standardize=True,
        stats_path=stats_path,
        out_path=source_table_path,
    )

    summary = train_continuous_dynamic_prior(
        item_features=sem,
        target_table=dyn_table,
        checkpoint_path=ckpt_path,
        hidden_dim=cfg_get(cfg, "continuous_dynamic", "hidden_dim", 256),
        dropout=cfg_get(cfg, "continuous_dynamic", "dropout", 0.1),
        epochs=cfg_get(cfg, "continuous_dynamic", "epochs", 100),
        lr=cfg_get(cfg, "continuous_dynamic", "lr", 3e-4),
        weight_decay=cfg_get(cfg, "continuous_dynamic", "weight_decay", 1e-5),
        patience=cfg_get(cfg, "continuous_dynamic", "patience", 10),
        val_ratio=cfg_get(cfg, "continuous_dynamic", "val_ratio", 0.1),
        seed=seed,
        device=device,
        checkpoint_extra={
            "cfg": cfg,
            "source": source,
            "seed": seed,
            "feature_cols": feature_cols,
            "source_table_path": str(source_table_path),
            "stats_path": str(stats_path),
            "embedding_dim": int(sem.size(1)),
            "dynamic_dim": int(dyn_table.size(1)),
        },
    )

    model, _ = load_continuous_predictor_from_checkpoint(ckpt_path, device)
    pred_source_table = predict_continuous_table(model, sem, device=device)
    pred_path = Path(cfg["paths"]["dynamic_dir"]) / f"pred_source_continuous_dynamic_table_{source}_seed{seed}.pt"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"dynamic_table": pred_source_table, "source": source, "seed": seed, "checkpoint": str(ckpt_path)}, pred_path)

    out = {**summary, "source_table_path": str(source_table_path), "pred_source_table_path": str(pred_path), "stats_path": str(stats_path)}
    save_json(out, summary_path)
    append_csv({"model":"continuous_dynamic_prior_v1","stage":"source_prior","source":source,"seed":seed,"checkpoint":str(ckpt_path),"best_epoch":summary["best_epoch"],"best_val_mse":summary["best_val_mse"],"dynamic_dim":int(dyn_table.size(1)),"source_table_path":str(source_table_path),"pred_source_table_path":str(pred_path)}, results_path)
    print(f"[Saved] source_table={source_table_path}")
    print(f"[Saved] pred_source_table={pred_path}")
    print(f"[Saved] summary={summary_path}")

if __name__ == "__main__":
    main()
