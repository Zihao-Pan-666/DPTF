from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from isddg.config import load_config, ensure_dirs
from isddg.utils.seed import set_seed
from isddg.utils.device import get_device
from isddg.utils.io import save_json, append_csv
from isddg.data.io import load_interactions, group_user_sequences
from isddg.data.dataset import PrefixDataset, collate_prefix
from isddg.data.semantic_splits import build_source_train_val_samples
from isddg.features.semantic import load_semantic_embeddings
from isddg.features.dynamic_feature_store import load_pt_feature_table
from isddg.models.backbone import FeatureBERT4Rec
from isddg.evaluation.dynamic_signal_evaluator import evaluate_dynamic_signal, parse_beta_grid

def cfg_get(cfg, sec, key, default):
    return cfg.get(sec, {}).get(key, default)

def main():
    ap = argparse.ArgumentParser(description="Evaluate pure dynamic-signal-only ranking on source validation.")
    ap.add_argument("--config", default="configs/continuous_dynamic.yaml")
    ap.add_argument("--feature_table_path", default=None)
    ap.add_argument("--feature_type", default="oracle_role", choices=["oracle_role", "continuous_oracle"])
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--results_path", default=None)
    ap.add_argument("--summary_path", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config); ensure_dirs(cfg)
    data_root = args.data_root or cfg.get("data_root", "./data")
    source = args.source or cfg.get("source")
    seed = args.seed if args.seed is not None else cfg_get(cfg, "data", "seed", 2026)
    set_seed(seed)
    device = get_device(args.device or cfg_get(cfg, "training", "device", "auto"))
    batch_size = args.batch_size or cfg_get(cfg, "training", "batch_size", 128)
    max_len = cfg_get(cfg, "data", "max_len", 50)
    min_len = cfg_get(cfg, "data", "min_len", 3)
    ranking = cfg_get(cfg, "data", "ranking", "sampled")
    eval_negatives = cfg_get(cfg, "data", "eval_negatives", 100)
    pooling = cfg_get(cfg, "late_fusion", "pooling", "decay")
    recent_k = cfg_get(cfg, "late_fusion", "recent_k", 5)
    decay = cfg_get(cfg, "late_fusion", "decay", 0.8)
    score_norm = cfg_get(cfg, "late_fusion", "score_norm", "zscore")

    if args.feature_table_path:
        feature_path = Path(args.feature_table_path)
    elif args.feature_type == "oracle_role":
        feature_path = Path(cfg_get(cfg, "role", "oracle_role_table_path", ""))
    else:
        feature_path = Path(cfg_get(cfg, "continuous_dynamic", "source_table_path", ""))

    result_dir = Path(cfg["paths"]["result_dir"])
    results_path = Path(args.results_path or result_dir / "dynamic_only_source_val.csv")
    summary_path = Path(args.summary_path or result_dir / f"dynamic_only_{args.feature_type}_{source}_seed{seed}.json")

    print("="*80)
    print("[DynamicOnly] source validation without semantic score")
    print(f"feature_type={args.feature_type} feature_path={feature_path}")
    print("="*80)

    df, item_map = load_interactions(data_root, source)
    seqs = group_user_sequences(df, min_len=min_len)
    _, val_samples = build_source_train_val_samples(seqs, max_len=max_len, min_prefix=1)
    feature_table = load_pt_feature_table(feature_path, num_items=len(item_map))
    loader = DataLoader(PrefixDataset(val_samples, len(item_map), max_len), batch_size=batch_size, shuffle=False, collate_fn=collate_prefix, num_workers=cfg_get(cfg, "data", "num_workers", 0), pin_memory=torch.cuda.is_available())

    metrics = evaluate_dynamic_signal(None, feature_table, loader, len(item_map), device, beta=1.0, semantic_weight=0.0, ranking_mode=ranking, num_negatives=eval_negatives, seed=seed, pooling=pooling, recent_k=recent_k, decay=decay, score_norm=score_norm, desc=f"dynamic-only {args.feature_type}")
    row = {"model":"dynamic_only_ranking_v1","stage":"source_val","source":source,"target":"","seed":seed,"feature_type":args.feature_type,"feature_table":str(feature_path),"ranking_mode":ranking,"eval_negatives":eval_negatives,**metrics}
    append_csv(row, results_path)
    save_json(row, summary_path)
    print(f"[DynamicOnly] R@10={metrics['Recall@10']:.6f} N@10={metrics['NDCG@10']:.6f} MRR@10={metrics['MRR@10']:.6f}")
    print(f"[Saved] {summary_path}")

if __name__ == "__main__":
    main()
