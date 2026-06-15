from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from isddg.config import load_config, ensure_dirs
from isddg.utils.seed import set_seed
from isddg.utils.device import get_device
from isddg.utils.io import save_json, append_csv
from isddg.data.io import load_interactions, group_user_sequences
from isddg.data.dataset import PrefixDataset, collate_prefix
from isddg.data.semantic_splits import build_source_train_val_samples
from isddg.features.semantic import load_semantic_embeddings
from isddg.models.backbone import FeatureBERT4Rec
from isddg.training.semantic_v0_trainer import train_semantic_v0


def cfg_get(cfg: Dict[str, Any], section: str, key: str, default: Any) -> Any:
    return cfg.get(section, {}).get(key, default)


def fmt_sec(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes)}m{sec:.1f}s"


@contextmanager
def timed(name: str):
    start = time.perf_counter()
    print(f"[Step] {name} ...", flush=True)
    yield
    elapsed = time.perf_counter() - start
    print(f"[Step] {name} done in {fmt_sec(elapsed)}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Train V0 Semantic-only BERT4Rec on source domain only.")
    ap.add_argument("--config", default="configs/semantic_v0.yaml")
    ap.add_argument("--source", default=None)
    ap.add_argument("--data_root", default="./data")
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
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    return ap.parse_args()


def main():
    total_start = time.perf_counter()
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    source = args.source or cfg.get("source", "amazon_movies_and_tv")
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
    device = get_device(args.device or cfg_get(cfg, "training", "device", "auto"))
    show_progress = not args.no_progress
    set_seed(seed)

    ckpt_dir = Path(cfg.get("paths", {}).get("checkpoint_dir", "artifacts/checkpoints"))
    result_dir = Path(cfg.get("paths", {}).get("result_dir", "results"))
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else ckpt_dir / f"semantic_v0_{source}_seed{seed}.pt"
    results_path = Path(args.results_path) if args.results_path else result_dir / "semantic_v0_source_val.csv"

    print("=" * 80)
    print("[SemanticV0] Source-only training")
    print(f"source={source}")
    print(f"data_root={data_root}")
    print(f"embedding_dir={args.embedding_dir}")
    print(f"seed={seed}")
    print(f"device={device}")
    print(f"checkpoint_path={checkpoint_path}")
    print("=" * 80)

    prep = tqdm(total=7, desc="prepare source pipeline", unit="step", dynamic_ncols=True, disable=not show_progress)

    with timed("load source interactions"):
        df, item_map = load_interactions(data_root, source)
    prep.update(1)

    with timed("group user sequences"):
        seqs = group_user_sequences(df, min_len=min_len)
    prep.update(1)

    with timed("build source train/validation samples"):
        train_samples, val_samples = build_source_train_val_samples(seqs=seqs, max_len=max_len, min_prefix=1)
    prep.update(1)

    print(
        f"[Data] interactions={len(df):,} users={len(seqs):,} items={len(item_map):,} "
        f"train_samples={len(train_samples):,} source_val_users={len(val_samples):,}"
    )

    with timed("load and align source semantic embeddings"):
        item_features = load_semantic_embeddings(
            data_root=data_root,
            domain=source,
            item_map=item_map,
            embedding_dir=args.embedding_dir,
            strict=True,
        )[: len(item_map) + 1]
        item_features[0] = 0.0
    prep.update(1)
    print(f"[Embedding] table_shape={tuple(item_features.shape)}")

    with timed("initialize FeatureBERT4Rec"):
        model = FeatureBERT4Rec(
            item_features=item_features,
            hidden_dim=hidden_dim,
            max_len=max_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
    prep.update(1)

    with timed("build PrefixDataset"):
        train_ds = PrefixDataset(train_samples, num_items=len(item_map), max_len=max_len)
        val_ds = PrefixDataset(val_samples, num_items=len(item_map), max_len=max_len)
    prep.update(1)

    with timed("build DataLoader"):
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_prefix,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_prefix,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    prep.update(1)
    prep.close()

    print(
        f"[Loader] train_batches={len(train_loader):,} val_batches={len(val_loader):,} "
        f"batch_size={batch_size} num_workers={num_workers}"
    )

    train_summary = train_semantic_v0(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_items=len(item_map),
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
        show_progress=show_progress,
        checkpoint_extra={
            "cfg": cfg,
            "source": source,
            "seed": seed,
            "max_len": max_len,
            "embedding_dim": int(item_features.size(1)),
            "model_hparams": {
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "dropout": dropout,
            },
            "protocol": {
                "name": "semantic_v0_source_only",
                "checkpoint_selection": "source_validation_only",
                "target_usage": "none_during_training",
            },
        },
    )

    summary_json = result_dir / f"semantic_v0_{source}_seed{seed}_train_summary.json"
    with timed("save training summary and source-val csv"):
        save_json(train_summary, summary_json)
        append_csv({
            "model": "semantic_v0",
            "stage": "source_val",
            "source": source,
            "target": "",
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "best_epoch": train_summary["best_epoch"],
            "early_stop_metric": train_summary["early_stop_metric"],
            "best_metric": train_summary["best_metric"],
            "ranking_mode": eval_ranking_mode,
            "eval_negatives": eval_negatives,
            "train_samples": len(train_samples),
            "source_val_users": len(val_samples),
            "num_items": len(item_map),
            "embedding_dim": int(item_features.size(1)),
            "total_elapsed_sec": train_summary.get("total_elapsed_sec", None),
        }, results_path)

    total_elapsed = time.perf_counter() - total_start
    print("=" * 80)
    print("[SemanticV0] Training finished")
    print(f"best_epoch={train_summary['best_epoch']}")
    print(f"best_{train_summary['early_stop_metric']}={train_summary['best_metric']:.6f}")
    print(f"saved_checkpoint={checkpoint_path}")
    print(f"saved_summary={summary_json}")
    print(f"appended_results={results_path}")
    print(f"total_wall_time={fmt_sec(total_elapsed)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
