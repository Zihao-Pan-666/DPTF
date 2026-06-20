from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from isddg.config import load_config
from isddg.data.dataset import PrefixDataset, collate_prefix
from isddg.data.io import build_target_eval_samples, group_user_sequences, load_interactions
from isddg.evaluation.sequence_prototype_evaluator import evaluate_sequence_prototype
from isddg.features.dynamic_feature_store import load_pt_feature_table
from isddg.features.semantic import load_semantic_embeddings
from isddg.models.backbone import FeatureBERT4Rec
from isddg.prototypes.sequence_dynamic import SequenceDynamicPrototypeBank
from isddg.training.continuous_dynamic_checkpoint import (
    load_continuous_dynamic_predictor_from_checkpoint,
    predict_continuous_dynamic_table,
)
from isddg.utils.device import get_device
from isddg.utils.io import append_csv, load_json, save_json
from isddg.utils.seed import set_seed


def cfg_get(cfg: Dict[str, Any], section: str, key: str, default: Any) -> Any:
    return cfg.get(section, {}).get(key, default)


def parse_targets(args, cfg) -> List[str]:
    if args.targets:
        return [x.strip() for x in args.targets.split(",") if x.strip()]
    return list(cfg.get("targets", []))


def load_semantic_model(cfg, checkpoint_path: Path, sem: torch.Tensor, max_len: int, device: torch.device) -> FeatureBERT4Rec:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hp = ckpt.get("model_hparams", {})
    model = FeatureBERT4Rec(
        item_features=sem,
        hidden_dim=int(hp.get("hidden_dim", cfg_get(cfg, "model", "hidden_dim", 128))),
        max_len=max_len,
        num_layers=int(hp.get("num_layers", cfg_get(cfg, "model", "num_layers", 2))),
        num_heads=int(hp.get("num_heads", cfg_get(cfg, "model", "num_heads", 2))),
        dropout=float(hp.get("dropout", cfg_get(cfg, "model", "dropout", 0.2))),
        role_features=None,
        role_alpha=0.0,
    )
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"[SemanticCheckpoint] loaded={checkpoint_path} missing={missing} unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model


def build_target_dynamic_table(args, cfg, sem: torch.Tensor, num_items: int, device: torch.device) -> tuple[torch.Tensor | None, str]:
    proto_cfg = cfg.get("prototype", {})
    mode = args.candidate_dynamic_source or proto_cfg.get("candidate_dynamic_source", "predicted")
    if mode in {"none", "off"}:
        return None, "none"
    if mode == "table":
        path = Path(args.candidate_dynamic_table or proto_cfg.get("candidate_dynamic_table", ""))
        if not path.exists():
            raise FileNotFoundError(f"candidate_dynamic_source=table but table not found: {path}")
        return load_pt_feature_table(path, num_items=num_items), str(path)
    if mode == "predicted":
        ckpt = Path(args.continuous_checkpoint or cfg_get(cfg, "continuous_dynamic", "checkpoint", proto_cfg.get("continuous_checkpoint", "")))
        if not ckpt.exists():
            print(f"[TargetDynamic] predicted requested but checkpoint not found: {ckpt}. Use dynamic branch OFF.")
            return None, "predicted_missing"
        predictor, _, checkpoint_kind = load_continuous_dynamic_predictor_from_checkpoint(ckpt, device)
        table = predict_continuous_dynamic_table(predictor, sem, device=device)
        return table[: num_items + 1].cpu(), f"{ckpt}::{checkpoint_kind}"
    if mode == "oracle":
        raise ValueError("Target oracle dynamic table is forbidden in strict zero-shot evaluation. Use predicted/table/none.")
    raise ValueError(f"Unknown candidate_dynamic_source={mode}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict zero-shot evaluation for sequence dynamic prototype fusion.")
    ap.add_argument("--config", default="configs/sequence_prototype.yaml")
    ap.add_argument("--semantic_checkpoint", default=None)
    ap.add_argument("--prototype_bank", default=None)
    ap.add_argument("--selection_summary", default=None)
    ap.add_argument("--continuous_checkpoint", default=None)
    ap.add_argument("--candidate_dynamic_source", choices=["predicted", "table", "none", "oracle"], default=None)
    ap.add_argument("--candidate_dynamic_table", default=None)
    ap.add_argument("--beta_sem", type=float, default=None)
    ap.add_argument("--beta_dyn", type=float, default=None)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--embedding_dir", default=None)
    ap.add_argument("--results_path", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--no_progress", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = args.data_root or cfg.get("data_root", "./data")
    source = args.source or cfg.get("source", "amazon_movies_and_tv")
    targets = parse_targets(args, cfg)
    seed = args.seed if args.seed is not None else cfg_get(cfg, "data", "seed", 2026)
    max_len = int(cfg_get(cfg, "data", "max_len", 50))
    min_len = int(cfg_get(cfg, "data", "min_len", 3))
    num_workers = int(cfg_get(cfg, "data", "num_workers", 0))
    eval_negatives = int(cfg_get(cfg, "data", "eval_negatives", 100))
    ranking = str(cfg_get(cfg, "data", "ranking", "sampled"))
    embedding_dir = args.embedding_dir or cfg_get(cfg, "data", "embedding_dir", "semantic_embeddings")
    batch_size = int(args.batch_size or cfg_get(cfg, "training", "batch_size", 128))
    show_progress = not args.no_progress

    proto_cfg = cfg.get("prototype", {})
    semantic_checkpoint = Path(args.semantic_checkpoint or cfg_get(cfg, "semantic", "checkpoint", ""))
    prototype_bank = Path(args.prototype_bank or proto_cfg.get("bank_path", ""))
    selection_summary = Path(args.selection_summary or proto_cfg.get("selection_summary", "")) if (args.selection_summary or proto_cfg.get("selection_summary", "")) else None

    beta_sem = args.beta_sem
    beta_dyn = args.beta_dyn
    if selection_summary is not None and selection_summary.exists():
        sel = load_json(selection_summary)
        if beta_sem is None:
            beta_sem = float(sel.get("best_beta_sem", 0.0))
        if beta_dyn is None:
            beta_dyn = float(sel.get("best_beta_dyn", 0.0))
    beta_sem = float(beta_sem if beta_sem is not None else proto_cfg.get("default_beta_sem", 0.0))
    beta_dyn = float(beta_dyn if beta_dyn is not None else proto_cfg.get("default_beta_dyn", 0.0))

    top_m = int(proto_cfg.get("top_m", 16))
    temperature = float(proto_cfg.get("temperature", 0.05))
    semantic_score_norm = proto_cfg.get("semantic_score_norm", "zscore")
    prototype_score_norm = proto_cfg.get("prototype_score_norm", "zscore")
    dynamic_score_norm = proto_cfg.get("dynamic_score_norm", "zscore")

    result_dir = Path(cfg.get("paths", {}).get("result_dir", "results/mainline"))
    results_path = Path(args.results_path or result_dir / "sequence_prototype_zero_shot.csv")

    set_seed(seed)
    device = get_device(args.device or cfg_get(cfg, "training", "device", "auto"))
    bank = SequenceDynamicPrototypeBank.load(prototype_bank)

    print("=" * 80)
    print("[SequencePrototype] strict zero-shot evaluation")
    print(f"source={source}")
    print(f"targets={targets}")
    print(f"semantic_checkpoint={semantic_checkpoint}")
    print(f"prototype_bank={prototype_bank}")
    print(f"selection_summary={selection_summary}")
    print(f"beta_sem={beta_sem} beta_dyn={beta_dyn}")
    print("=" * 80)

    for target in targets:
        print(f"\n[Target] {target}")
        df, item_map = load_interactions(data_root, target)
        seqs = group_user_sequences(df, min_len=min_len)
        eval_samples = build_target_eval_samples(seqs, max_len=max_len)
        sem = load_semantic_embeddings(data_root, target, item_map=item_map, embedding_dir=embedding_dir, strict=True)[: len(item_map) + 1]
        sem[0] = 0.0
        model = load_semantic_model(cfg, semantic_checkpoint, sem, max_len=max_len, device=device)
        candidate_dyn, candidate_dyn_source = build_target_dynamic_table(args, cfg, sem, len(item_map), device)
        loader = DataLoader(
            PrefixDataset(eval_samples, len(item_map), max_len),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_prefix,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        metrics = evaluate_sequence_prototype(
            model=model,
            bank=bank,
            loader=loader,
            num_items=len(item_map),
            device=device,
            beta_sem=beta_sem,
            beta_dyn=beta_dyn,
            candidate_dynamic_table=candidate_dyn,
            ranking_mode=ranking,
            num_negatives=eval_negatives,
            seed=seed,
            top_m=top_m,
            temperature=temperature,
            semantic_score_norm=semantic_score_norm,
            prototype_score_norm=prototype_score_norm,
            dynamic_score_norm=dynamic_score_norm,
            tie_policy="worst",
            desc=f"proto {source}->{target}",
            show_progress=show_progress,
            collect_score_stats=True,
        )
        row = {
            "model": "sequence_dynamic_prototype",
            "stage": "zero_shot",
            "source": source,
            "target": target,
            "seed": seed,
            "semantic_checkpoint": str(semantic_checkpoint),
            "prototype_bank": str(prototype_bank),
            "selection_summary": str(selection_summary) if selection_summary is not None else "",
            "candidate_dynamic_source": candidate_dyn_source,
            "ranking_mode": ranking,
            "eval_negatives": eval_negatives,
            "num_target_users": len(eval_samples),
            "num_target_items": len(item_map),
            **metrics,
            "protocol_target_interactions": "forward_only_no_training_no_tuning",
        }
        append_csv(row, results_path)
        metrics_json = result_dir / f"sequence_prototype_{source}_to_{target}_{ranking}_seed{seed}.json"
        save_json(row, metrics_json)
        print(
            f"[target={target}] R@10={metrics['Recall@10']:.6f} N@10={metrics['NDCG@10']:.6f} "
            f"MRR@10={metrics['MRR@10']:.6f} tie={metrics['tie_case_ratio']:.6f}"
        )
        print(f"[Saved] {metrics_json}")
    print(f"[Saved] appended zero-shot results to {results_path}")


if __name__ == "__main__":
    main()
