from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

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
from isddg.models.backbone import FeatureBERT4Rec
from isddg.training.semantic_align_item_trainer import train_semantic_align_item


def cfg_get(cfg: Dict[str, Any], section: str, key: str, default: Any) -> Any:
    return cfg.get(section, {}).get(key, default)


def stage(msg: str) -> float:
    print(f"\n[Stage] {msg}", flush=True)
    return time.time()


def done(t0: float) -> None:
    print(f"[Done] elapsed={time.time() - t0:.2f}s", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Train LLM-RecG-Item-lite semantic alignment baseline.")
    ap.add_argument("--config", default="configs/semantic_align_item.yaml")
    ap.add_argument("--source", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--embedding_dir", default="semantic_embeddings")
    ap.add_argument("--checkpoint_path", default=None)
    ap.add_argument("--results_path", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--min_len", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--num_layers", type=int, default=None)
    ap.add_argument("--num_heads", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--train_negatives", type=int, default=None)
    ap.add_argument("--eval_negatives", type=int, default=None)
    ap.add_argument("--eval_ranking_mode", choices=["sampled", "full"], default=None)
    ap.add_argument("--early_stop_metric", default=None)
    ap.add_argument("--early_stop_patience", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--align_method", choices=["coral", "mmd"], default=None)
    ap.add_argument("--align_alpha", type=float, default=None)
    ap.add_argument("--align_sample_size", type=int, default=None)
    ap.add_argument("--align_warmup_epochs", type=int, default=None)
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def main():
    total_t0 = time.time()
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    source = args.source or cfg.get("source", "amazon_movies_and_tv")
    target = args.target or cfg_get(cfg, "alignment", "target_domain", None) or cfg.get("targets", ["amazon_cds_and_vinyl"])[0]
    data_root = args.data_root or cfg.get("data_root", "./data")
    seed = args.seed if args.seed is not None else cfg_get(cfg, "data", "seed", 2026)
    max_len = args.max_len if args.max_len is not None else cfg_get(cfg, "data", "max_len", 50)
    min_len = args.min_len if args.min_len is not None else cfg_get(cfg, "data", "min_len", 3)
    batch_size = args.batch_size if args.batch_size is not None else cfg_get(cfg, "training", "batch_size", 128)
    num_workers = args.num_workers if args.num_workers is not None else cfg_get(cfg, "data", "num_workers", 0)
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else cfg_get(cfg, "model", "hidden_dim", 128)
    num_layers = args.num_layers if args.num_layers is not None else cfg_get(cfg, "model", "num_layers", 2)
    num_heads = args.num_heads if args.num_heads is not None else cfg_get(cfg, "model", "num_heads", 2)
    dropout = args.dropout if args.dropout is not None else cfg_get(cfg, "model", "dropout", 0.2)
    epochs = args.epochs if args.epochs is not None else cfg_get(cfg, "training", "epochs", 20)
    lr = args.lr if args.lr is not None else cfg_get(cfg, "training", "lr", 1e-4)
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg_get(cfg, "training", "weight_decay", 0.0)
    train_negatives = args.train_negatives if args.train_negatives is not None else cfg_get(cfg, "data", "train_negatives", 5)
    eval_negatives = args.eval_negatives if args.eval_negatives is not None else cfg_get(cfg, "data", "eval_negatives", 100)
    eval_ranking_mode = args.eval_ranking_mode or cfg_get(cfg, "data", "ranking", "sampled")
    early_stop_metric = args.early_stop_metric or cfg_get(cfg, "training", "early_stop_metric", "NDCG@10")
    early_stop_patience = args.early_stop_patience if args.early_stop_patience is not None else cfg_get(cfg, "training", "early_stop_patience", 5)
    eval_every = args.eval_every if args.eval_every is not None else cfg_get(cfg, "training", "eval_every", 1)
    align_method = args.align_method or cfg_get(cfg, "alignment", "method", "coral")
    align_alpha = args.align_alpha if args.align_alpha is not None else cfg_get(cfg, "alignment", "alpha", 0.001)
    align_sample_size = args.align_sample_size if args.align_sample_size is not None else cfg_get(cfg, "alignment", "sample_size", 256)
    align_warmup_epochs = args.align_warmup_epochs if args.align_warmup_epochs is not None else cfg_get(cfg, "alignment", "warmup_epochs", 0)
    device = get_device(args.device or cfg_get(cfg, "training", "device", "auto"))
    set_seed(seed)

    ckpt_dir = Path(cfg.get("paths", {}).get("checkpoint_dir", "artifacts/checkpoints"))
    result_dir = Path(cfg.get("paths", {}).get("result_dir", "results"))
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else ckpt_dir / f"semantic_align_item_{source}_to_{target}_seed{seed}.pt"
    results_path = Path(args.results_path) if args.results_path else result_dir / "semantic_align_item_source_val.csv"

    print("=" * 80, flush=True)
    print("[SemanticAlignItem] Source BPR + target item-text alignment", flush=True)
    print(f"source={source} target_text_domain={target} data_root={data_root}", flush=True)
    print(f"embedding_dir={args.embedding_dir}", flush=True)
    print(f"align_method={align_method} alpha={align_alpha} sample_size={align_sample_size}", flush=True)
    print(f"checkpoint_path={checkpoint_path}", flush=True)
    print("=" * 80, flush=True)

    t = stage("1/7 Load source interactions and build source train/validation samples")
    src_df, src_map = load_interactions(data_root, source)
    src_seqs = group_user_sequences(src_df, min_len=min_len)
    train_samples, val_samples = build_source_train_val_samples(src_seqs, max_len=max_len, min_prefix=1)
    if len(train_samples) == 0 or len(val_samples) == 0:
        raise RuntimeError(
            f"Empty samples: train_samples={len(train_samples)}, val_samples={len(val_samples)}. "
            f"Please check data_root={data_root}, source={source}, min_len={min_len}."
        )
    print(
        f"[Data] source_interactions={len(src_df):,} source_users={len(src_seqs):,} "
        f"source_items={len(src_map):,} train_samples={len(train_samples):,} "
        f"source_val_users={len(val_samples):,}",
        flush=True,
    )
    done(t)

    t = stage("2/7 Load source semantic embeddings")
    src_feat = load_semantic_embeddings(
        data_root=data_root,
        domain=source,
        item_map=src_map,
        embedding_dir=args.embedding_dir,
        strict=True,
    )[: len(src_map) + 1]
    src_feat[0] = 0.0
    done(t)

    t = stage("3/7 Load target catalog and target item-text embeddings for unsupervised alignment")
    # Target interactions are used only to recover the target item universe / item_map.
    # The training objective does not use target next-item labels, target timestamps,
    # target popularity, target transition statistics, or target validation metrics.
    tgt_df, tgt_map = load_interactions(data_root, target)
    tgt_feat = load_semantic_embeddings(
        data_root=data_root,
        domain=target,
        item_map=tgt_map,
        embedding_dir=args.embedding_dir,
        strict=True,
    )[: len(tgt_map) + 1]
    tgt_feat[0] = 0.0
    print(f"[TargetText] target_items={len(tgt_map):,}", flush=True)
    print(f"[Embedding] source_shape={tuple(src_feat.shape)} target_shape={tuple(tgt_feat.shape)}", flush=True)
    done(t)

    t = stage("4/7 Build model")
    model = FeatureBERT4Rec(
        item_features=src_feat,
        hidden_dim=hidden_dim,
        max_len=max_len,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )
    done(t)

    t = stage("5/7 Build dataloaders")
    train_loader = DataLoader(
        PrefixDataset(train_samples, len(src_map), max_len),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_prefix,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        PrefixDataset(val_samples, len(src_map), max_len),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_prefix,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[Loader] train_batches={len(train_loader):,} val_batches={len(val_loader):,} batch_size={batch_size}", flush=True)
    done(t)

    t = stage("6/7 Start source training with item-level semantic alignment")
    summary = train_semantic_align_item(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_items=len(src_map),
        target_item_features=tgt_feat,
        device=device,
        checkpoint_path=checkpoint_path,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        train_negatives=train_negatives,
        eval_negatives=eval_negatives,
        eval_ranking_mode=eval_ranking_mode,
        early_stop_metric=early_stop_metric,
        early_stop_patience=early_stop_patience,
        eval_every=eval_every,
        seed=seed,
        align_alpha=align_alpha,
        align_sample_size=align_sample_size,
        align_method=align_method,
        align_warmup_epochs=align_warmup_epochs,
        checkpoint_extra={
            "cfg": cfg,
            "source": source,
            "target_text_domain": target,
            "seed": seed,
            "max_len": max_len,
            "embedding_dim": int(src_feat.size(1)),
            "model_hparams": {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "dropout": dropout,
            },
            "protocol": {
                "name": "semantic_align_item",
                "checkpoint_selection": "source_validation_only",
                "target_usage": "target item text embeddings only for unsupervised item-level alignment",
            },
        },
    )
    done(t)

    t = stage("7/7 Save summary and source-validation row")
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_json = result_dir / f"semantic_align_item_{source}_to_{target}_seed{seed}_train_summary.json"
    save_json(summary, summary_json)

    best_row = {}
    if summary.get("history"):
        for row in summary["history"]:
            if int(row.get("epoch", -1)) == int(summary.get("best_epoch", -2)):
                best_row = row
                break

    append_csv({
        "model": "semantic_align_item",
        "stage": "source_val",
        "source": source,
        "target_text_domain": target,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "best_epoch": summary["best_epoch"],
        "early_stop_metric": summary["early_stop_metric"],
        "best_metric": summary["best_metric"],
        "ranking_mode": eval_ranking_mode,
        "eval_negatives": eval_negatives,
        "source_items": len(src_map),
        "target_items": len(tgt_map),
        "embedding_dim": int(src_feat.size(1)),
        "align_method": align_method,
        "align_alpha": align_alpha,
        "align_sample_size": align_sample_size,
        "best_train_loss": best_row.get("train_loss", ""),
        "best_train_rec_loss": best_row.get("train_rec_loss", ""),
        "best_train_align_loss_raw": best_row.get("train_align_loss_raw", ""),
        "best_train_align_loss_weighted": best_row.get("train_align_loss_weighted", ""),
        "best_val_Recall@10": best_row.get("val_Recall@10", ""),
        "best_val_NDCG@10": best_row.get("val_NDCG@10", ""),
        "best_val_MRR@10": best_row.get("val_MRR@10", ""),
        "total_elapsed_sec": summary.get("total_elapsed_sec", time.time() - total_t0),
    }, results_path)
    done(t)

    print("=" * 80, flush=True)
    print("[SemanticAlignItem] Training finished", flush=True)
    print(f"best_epoch={summary['best_epoch']}", flush=True)
    print(f"best_{summary['early_stop_metric']}={summary['best_metric']:.6f}", flush=True)
    print(f"saved_checkpoint={checkpoint_path}", flush=True)
    print(f"saved_summary={summary_json}", flush=True)
    print(f"appended_results={results_path}", flush=True)
    print(f"total_elapsed_sec={time.time() - total_t0:.2f}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
