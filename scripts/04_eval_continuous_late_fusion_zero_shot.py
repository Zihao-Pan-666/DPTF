from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from isddg.config import load_config
from isddg.utils.seed import set_seed
from isddg.utils.device import get_device
from isddg.utils.io import save_json, append_csv
from isddg.data.io import load_interactions, group_user_sequences, build_target_eval_samples
from isddg.data.dataset import PrefixDataset, collate_prefix
from isddg.features.semantic import load_semantic_embeddings
from isddg.models.backbone import FeatureBERT4Rec
from isddg.training.continuous_dynamic_prior_trainer import load_continuous_predictor_from_checkpoint, predict_continuous_table
from isddg.evaluation.dynamic_signal_evaluator import evaluate_dynamic_signal

def cfg_get(cfg, sec, key, default):
    return cfg.get(sec, {}).get(key, default)

def parse_targets(args, cfg):
    if args.targets:
        return [x.strip() for x in args.targets.split(",") if x.strip()]
    return list(cfg.get("targets", []))

def main():
    ap = argparse.ArgumentParser(description="Zero-shot evaluation for continuous dynamic late fusion.")
    ap.add_argument("--config", default="configs/dynamic_signal_diagnostics_v1.yaml")
    ap.add_argument("--semantic_checkpoint", default=None)
    ap.add_argument("--continuous_checkpoint", default=None)
    ap.add_argument("--beta_summary", default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--embedding_dir", default="semantic_embeddings")
    ap.add_argument("--results_path", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = args.data_root or cfg.get("data_root", "./data")
    source = args.source or cfg.get("source")
    targets = parse_targets(args, cfg)
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

    semantic_checkpoint = Path(args.semantic_checkpoint or cfg_get(cfg, "semantic", "checkpoint", ""))
    cont_ckpt = Path(args.continuous_checkpoint or cfg_get(cfg, "continuous_dynamic", "checkpoint", ""))
    summary_path = Path(args.beta_summary or Path(cfg["paths"]["result_dir"]) / f"continuous_late_fusion_predicted_{source}_seed{seed}_summary.json")
    if args.beta is not None:
        beta = float(args.beta); beta_source = "cli"
    elif summary_path.exists():
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        beta = float(obj.get("best_beta", 0.0)); beta_source = str(summary_path)
    else:
        beta = 0.0; beta_source = "default_zero"

    result_dir = Path(cfg["paths"]["result_dir"])
    results_path = Path(args.results_path or result_dir / "continuous_late_fusion_zero_shot.csv")

    print("="*80)
    print("[ContinuousLateFusion] zero-shot")
    print(f"semantic_checkpoint={semantic_checkpoint}")
    print(f"continuous_checkpoint={cont_ckpt}")
    print(f"beta={beta} beta_source={beta_source}")
    print("="*80)

    ckpt = torch.load(semantic_checkpoint, map_location="cpu")
    hp = ckpt.get("model_hparams", {})
    predictor, _ = load_continuous_predictor_from_checkpoint(cont_ckpt, device)

    for target in targets:
        df, item_map = load_interactions(data_root, target)
        seqs = group_user_sequences(df, min_len=min_len)
        samples = build_target_eval_samples(seqs, max_len=max_len)
        sem = load_semantic_embeddings(data_root=data_root, domain=target, item_map=item_map, embedding_dir=args.embedding_dir, strict=True)[:len(item_map)+1]
        sem[0] = 0.0
        dyn_table = predict_continuous_table(predictor, sem, device=device)

        model = FeatureBERT4Rec(item_features=sem, hidden_dim=int(hp.get("hidden_dim", cfg_get(cfg, "model", "hidden_dim", 128))), max_len=max_len, num_layers=int(hp.get("num_layers", cfg_get(cfg, "model", "num_layers", 2))), num_heads=int(hp.get("num_heads", cfg_get(cfg, "model", "num_heads", 2))), dropout=float(hp.get("dropout", cfg_get(cfg, "model", "dropout", 0.2))), role_features=None, role_alpha=0.0)
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[target={target}] missing={missing} unexpected={unexpected}")
        model.to(device)

        loader = DataLoader(PrefixDataset(samples, len(item_map), max_len), batch_size=batch_size, shuffle=False, collate_fn=collate_prefix, num_workers=cfg_get(cfg, "data", "num_workers", 0), pin_memory=torch.cuda.is_available())
        metrics = evaluate_dynamic_signal(model, dyn_table, loader, len(item_map), device, beta=beta, semantic_weight=1.0, ranking_mode=ranking, num_negatives=eval_negatives, seed=seed, pooling=pooling, recent_k=recent_k, decay=decay, score_norm=score_norm, desc=f"cont-zero-shot {target}")
        row = {"model":"continuous_late_fusion_v1","stage":"zero_shot","source":source,"target":target,"seed":seed,"semantic_checkpoint":str(semantic_checkpoint),"continuous_checkpoint":str(cont_ckpt),"beta":beta,"beta_source":beta_source,"ranking_mode":ranking,"eval_negatives":eval_negatives,"num_target_users":len(samples),"num_target_items":len(item_map),"embedding_dim":int(sem.size(1)),"dynamic_dim":int(dyn_table.size(1)),**metrics}
        append_csv(row, results_path)
        out_json = result_dir / f"continuous_late_fusion_v1_{source}_to_{target}_{ranking}_seed{seed}.json"
        save_json(row, out_json)
        print(f"[target={target}] R@10={metrics['Recall@10']:.6f} N@10={metrics['NDCG@10']:.6f} MRR@10={metrics['MRR@10']:.6f}")
        print(f"[Saved] {out_json}")

if __name__ == "__main__":
    main()
