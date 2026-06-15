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
from isddg.training.continuous_dynamic_prior_trainer import load_continuous_predictor_from_checkpoint, predict_continuous_table
from isddg.evaluation.dynamic_signal_evaluator import evaluate_dynamic_signal, parse_beta_grid

def cfg_get(cfg, sec, key, default):
    return cfg.get(sec, {}).get(key, default)

def main():
    ap = argparse.ArgumentParser(description="Tune late fusion with continuous dynamic feature table.")
    ap.add_argument("--config", default="configs/dynamic_signal_diagnostics_v1.yaml")
    ap.add_argument("--dynamic_source", choices=["predicted", "oracle"], default="predicted")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--embedding_dir", default="semantic_embeddings")
    ap.add_argument("--semantic_checkpoint", default=None)
    ap.add_argument("--continuous_checkpoint", default=None)
    ap.add_argument("--continuous_table_path", default=None)
    ap.add_argument("--beta_grid", default=None)
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

    semantic_checkpoint = Path(args.semantic_checkpoint or cfg_get(cfg, "semantic", "checkpoint", ""))
    cont_ckpt = Path(args.continuous_checkpoint or cfg_get(cfg, "continuous_dynamic", "checkpoint", ""))
    continuous_table_path = Path(args.continuous_table_path or cfg_get(cfg, "continuous_dynamic", "source_table_path", ""))

    beta_grid = parse_beta_grid(args.beta_grid) if args.beta_grid else parse_beta_grid(cfg_get(cfg, "late_fusion", "beta_grid", [0.0]))
    select_metric = cfg_get(cfg, "late_fusion", "select_metric", "NDCG@10")
    result_dir = Path(cfg["paths"]["result_dir"])
    results_path = Path(args.results_path or result_dir / f"continuous_late_fusion_{args.dynamic_source}_source_val.csv")
    summary_path = Path(args.summary_path or result_dir / f"continuous_late_fusion_{args.dynamic_source}_{source}_seed{seed}_summary.json")

    max_len = cfg_get(cfg, "data", "max_len", 50)
    min_len = cfg_get(cfg, "data", "min_len", 3)
    ranking = cfg_get(cfg, "data", "ranking", "sampled")
    eval_negatives = cfg_get(cfg, "data", "eval_negatives", 100)
    pooling = cfg_get(cfg, "late_fusion", "pooling", "decay")
    recent_k = cfg_get(cfg, "late_fusion", "recent_k", 5)
    decay = cfg_get(cfg, "late_fusion", "decay", 0.8)
    score_norm = cfg_get(cfg, "late_fusion", "score_norm", "zscore")

    print("="*80)
    print("[ContinuousLateFusion] source validation")
    print(f"dynamic_source={args.dynamic_source}")
    print(f"semantic_checkpoint={semantic_checkpoint}")
    print(f"continuous_checkpoint={cont_ckpt}")
    print(f"continuous_table_path={continuous_table_path}")
    print("="*80)

    df, item_map = load_interactions(data_root, source)
    seqs = group_user_sequences(df, min_len=min_len)
    _, val_samples = build_source_train_val_samples(seqs, max_len=max_len, min_prefix=1)

    sem = load_semantic_embeddings(data_root=data_root, domain=source, item_map=item_map, embedding_dir=args.embedding_dir, strict=True)[:len(item_map)+1]
    sem[0] = 0.0

    ckpt = torch.load(semantic_checkpoint, map_location="cpu")
    hp = ckpt.get("model_hparams", {})
    model = FeatureBERT4Rec(item_features=sem, hidden_dim=int(hp.get("hidden_dim", cfg_get(cfg, "model", "hidden_dim", 128))), max_len=max_len, num_layers=int(hp.get("num_layers", cfg_get(cfg, "model", "num_layers", 2))), num_heads=int(hp.get("num_heads", cfg_get(cfg, "model", "num_heads", 2))), dropout=float(hp.get("dropout", cfg_get(cfg, "model", "dropout", 0.2))), role_features=None, role_alpha=0.0)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"[SemanticCheckpoint] missing={missing} unexpected={unexpected}")
    model.to(device)

    if args.dynamic_source == "oracle":
        dyn_table = load_pt_feature_table(continuous_table_path, num_items=len(item_map))
        feature_source = str(continuous_table_path)
    else:
        predictor, _ = load_continuous_predictor_from_checkpoint(cont_ckpt, device)
        dyn_table = predict_continuous_table(predictor, sem, device=device)
        feature_source = str(cont_ckpt)

    loader = DataLoader(PrefixDataset(val_samples, len(item_map), max_len), batch_size=batch_size, shuffle=False, collate_fn=collate_prefix, num_workers=cfg_get(cfg, "data", "num_workers", 0), pin_memory=torch.cuda.is_available())

    rows, best = [], None
    for beta in beta_grid:
        metrics = evaluate_dynamic_signal(model, dyn_table, loader, len(item_map), device, beta=beta, semantic_weight=1.0, ranking_mode=ranking, num_negatives=eval_negatives, seed=seed, pooling=pooling, recent_k=recent_k, decay=decay, score_norm=score_norm, desc=f"cont-dyn {args.dynamic_source} beta={beta}")
        row = {"model":"continuous_late_fusion_v1","stage":"source_val","source":source,"target":"","seed":seed,"semantic_checkpoint":str(semantic_checkpoint),"feature_source":feature_source,"feature_type":f"continuous_{args.dynamic_source}","ranking_mode":ranking,"eval_negatives":eval_negatives,**metrics}
        append_csv(row, results_path); rows.append(row)
        if best is None or float(row[select_metric]) > float(best[select_metric]):
            best = row
        print(f"[beta={beta}] R@10={metrics['Recall@10']:.6f} N@10={metrics['NDCG@10']:.6f} MRR@10={metrics['MRR@10']:.6f}", flush=True)

    summary = {"model":"continuous_late_fusion_v1","source":source,"seed":seed,"dynamic_source":args.dynamic_source,"select_metric":select_metric,"best_beta":float(best["beta"]),"best_metric":float(best[select_metric]),"feature_source":feature_source,"history":rows}
    save_json(summary, summary_path)
    print(f"[Best] beta={summary['best_beta']} {select_metric}={summary['best_metric']:.6f}")
    print(f"[Saved] {summary_path}")

if __name__ == "__main__":
    main()
