from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

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
from isddg.models.llm_recg import LLMRecGBERT4Rec
from isddg.training.llm_recg_trainer import train_llm_recg


def cfg_get(cfg: Dict[str, Any], section: str, key: str, default: Any) -> Any:
    return cfg.get(section, {}).get(key, default)


def csv_list(x: str | None) -> List[str]:
    if not x:
        return []
    return [s.strip() for s in x.split(",") if s.strip()]


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
    print(f"[Step] {name} done in {fmt_sec(time.perf_counter() - start)}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Train official-style BERT4Rec-RecG baseline on source domain.")
    ap.add_argument("--config", default="configs/llm_recg.yaml")
    ap.add_argument("--source", default=None)
    ap.add_argument("--aux_domains", default=None, help="Comma-separated auxiliary domains for item-side generalization.")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--embedding_dir", default=None)
    ap.add_argument("--checkpoint_path", default=None)
    ap.add_argument("--results_path", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_progress", action="store_true")
    return ap.parse_args()


def load_domain_features(data_root: str, domain: str, embedding_dir: str):
    df, item_map = load_interactions(data_root, domain)
    feats = load_semantic_embeddings(
        data_root=data_root,
        domain=domain,
        item_map=item_map,
        embedding_dir=embedding_dir,
        strict=True,
    )[: len(item_map) + 1]
    feats[0] = 0.0
    return df, item_map, feats


def sample_auxiliary_embeddings(
    aux_feature_tables: List[tuple[str, torch.Tensor, int]],
    samples_per_domain: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official-style: pre-sample a fixed pool of auxiliary item embeddings."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    raws = []
    doms = []
    for domain, table, domain_id in aux_feature_tables:
        if table.size(0) <= 1:
            continue
        n = min(int(samples_per_domain), int(table.size(0) - 1))
        perm = torch.randperm(int(table.size(0) - 1), generator=gen)[:n] + 1
        raws.append(table[perm].float().cpu())
        doms.append(torch.full((n,), int(domain_id), dtype=torch.long))
        print(f"[AuxSample:{domain}] sampled_items={n:,} domain_id={domain_id}")
    if not raws:
        return torch.empty((0, 0)), torch.empty((0,), dtype=torch.long)
    return torch.cat(raws, dim=0), torch.cat(doms, dim=0)


def main():
    total_start = time.perf_counter()
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    source = args.source or cfg.get("source", "amazon_movies_and_tv")
    aux_domains = csv_list(args.aux_domains) or list(cfg.get("aux_domains", cfg.get("targets", [])))
    data_root = args.data_root or cfg.get("data_root", "./data")
    embedding_dir = args.embedding_dir or cfg_get(cfg, "data", "embedding_dir", "semantic_embeddings")
    seed = args.seed if args.seed is not None else int(cfg_get(cfg, "data", "seed", 2026))
    max_len = int(cfg_get(cfg, "data", "max_len", 50))
    min_len = int(cfg_get(cfg, "data", "min_len", 3))
    num_workers = int(cfg_get(cfg, "data", "num_workers", 0))
    train_negatives = int(cfg_get(cfg, "data", "train_negatives", 5))
    eval_negatives = int(cfg_get(cfg, "data", "eval_negatives", 100))
    ranking = cfg_get(cfg, "data", "ranking", "sampled")

    hidden_dim = int(cfg_get(cfg, "model", "hidden_dim", 256))
    num_layers = int(cfg_get(cfg, "model", "num_layers", 2))
    num_heads = int(cfg_get(cfg, "model", "num_heads", 2))
    dropout = float(cfg_get(cfg, "model", "dropout", 0.5))
    pattern_fusion = cfg_get(cfg, "model", "pattern_fusion", "residual")
    pattern_residual_weight = float(cfg_get(cfg, "model", "pattern_residual_weight", 0.5))
    init_pattern_fusion_as_residual = bool(cfg_get(cfg, "model", "init_pattern_fusion_as_residual", True))

    batch_size = int(cfg_get(cfg, "training", "batch_size", 128))
    epochs = int(cfg_get(cfg, "training", "epochs", 50))
    lr = float(cfg_get(cfg, "training", "lr", 1e-4))
    weight_decay = float(cfg_get(cfg, "training", "weight_decay", 0.0))
    early_stop_metric = cfg_get(cfg, "training", "early_stop_metric", "NDCG@10")
    early_stop_patience = int(cfg_get(cfg, "training", "early_stop_patience", 5))
    eval_every = int(cfg_get(cfg, "training", "eval_every", 1))
    grad_clip = float(cfg_get(cfg, "training", "grad_clip", 5.0))
    use_source_val_selection = bool(cfg_get(cfg, "training", "use_source_val_selection", True))

    alpha = float(cfg_get(cfg, "llm_recg", "alpha", 0.001))
    alignment_temperature = float(cfg_get(cfg, "llm_recg", "alignment_temperature", 1.0))
    aux_samples_per_domain = int(cfg_get(cfg, "llm_recg", "aux_samples_per_domain", 4096))
    num_patterns = int(cfg_get(cfg, "llm_recg", "num_sequential_patterns", 10))

    device = get_device(args.device or cfg_get(cfg, "training", "device", "auto"))
    show_progress = not args.no_progress
    set_seed(seed)

    result_dir = Path(cfg.get("paths", {}).get("result_dir", "results/mainline/llm_recg"))
    ckpt_dir = Path(cfg.get("paths", {}).get("checkpoint_dir", "artifacts/checkpoints/llm_recg"))
    result_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else ckpt_dir / f"bert4rec_recg_{source}_seed{seed}.pt"
    results_path = Path(args.results_path) if args.results_path else result_dir / "bert4rec_recg_source_val.csv"

    print("=" * 80)
    print("[BERT4Rec-RecG] Source training with item-side cross-domain generalization")
    print(f"source={source}")
    print(f"aux_domains={aux_domains}")
    print(f"data_root={data_root}")
    print(f"embedding_dir={embedding_dir}")
    print(f"seed={seed}")
    print(f"device={device}")
    print(f"checkpoint_path={checkpoint_path}")
    print(f"hidden_dim={hidden_dim} dropout={dropout} pattern_fusion={pattern_fusion}")
    print("=" * 80)

    with timed("load source interactions and embeddings"):
        df, item_map, item_features = load_domain_features(data_root, source, embedding_dir)
    with timed("group source sequences"):
        seqs = group_user_sequences(df, min_len=min_len)
    with timed("build source train/validation samples"):
        train_samples, val_samples = build_source_train_val_samples(seqs=seqs, max_len=max_len, min_prefix=1)

    print(
        f"[Source] interactions={len(df):,} users={len(seqs):,} items={len(item_map):,} "
        f"train_samples={len(train_samples):,} source_val_users={len(val_samples):,} "
        f"embedding_shape={tuple(item_features.shape)}"
    )

    aux_feature_tables = []
    for domain_id, domain in enumerate(aux_domains):
        with timed(f"load aux item metadata embeddings: {domain}"):
            _, aux_item_map, aux_feats = load_domain_features(data_root, domain, embedding_dir)
        aux_feature_tables.append((domain, aux_feats.cpu(), domain_id))
        print(
            f"[Aux:{domain}] catalog_items={len(aux_item_map):,} "
            f"embedding_shape={tuple(aux_feats.shape)} interaction_labels_not_used_for_training=True domain_id={domain_id}"
        )

    sampled_aux_embeddings, sampled_aux_domains = sample_auxiliary_embeddings(
        aux_feature_tables=aux_feature_tables,
        samples_per_domain=aux_samples_per_domain,
        seed=seed,
    )

    train_ds = PrefixDataset(train_samples, num_items=len(item_map), max_len=max_len)
    val_ds = PrefixDataset(val_samples, num_items=len(item_map), max_len=max_len)
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

    model = LLMRecGBERT4Rec(
        item_features=item_features,
        hidden_dim=hidden_dim,
        max_len=max_len,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        num_sequential_patterns=num_patterns,
        pattern_fusion=pattern_fusion,
        pattern_residual_weight=pattern_residual_weight,
        init_pattern_fusion_as_residual=init_pattern_fusion_as_residual,
    )

    train_summary = train_llm_recg(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_items=len(item_map),
        sampled_aux_embeddings=sampled_aux_embeddings,
        sampled_aux_domains=sampled_aux_domains,
        num_aux_domains=len(aux_feature_tables),
        device=device,
        checkpoint_path=checkpoint_path,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        train_negatives=train_negatives,
        eval_negatives=eval_negatives,
        eval_ranking_mode=ranking,
        early_stop_metric=early_stop_metric,
        early_stop_patience=early_stop_patience,
        eval_every=eval_every,
        grad_clip=grad_clip,
        seed=seed,
        alpha=alpha,
        alignment_temperature=alignment_temperature,
        use_source_val_selection=use_source_val_selection,
        show_progress=show_progress,
        checkpoint_extra={
            "cfg": cfg,
            "source": source,
            "aux_domains": aux_domains,
            "domain_id_protocol": "aux_domains=0..M-1, source/current=M",
        },
    )

    summary_path = result_dir / f"bert4rec_recg_{source}_seed{seed}_train_summary.json"
    save_json(train_summary, summary_path)

    best_row = {
        "model": "bert4rec_recg",
        "stage": "source_val",
        "source": source,
        "seed": seed,
        "best_epoch": train_summary["best_epoch"],
        "best_metric": train_summary["best_metric"],
        "best_train_loss": train_summary.get("best_train_loss", 0.0),
        "early_stop_metric": early_stop_metric,
        "selection_mode": train_summary.get("selection_mode", "source_val"),
        "checkpoint": str(checkpoint_path),
    }
    if train_summary.get("history"):
        best_epoch = int(train_summary["best_epoch"])
        for row in train_summary["history"]:
            if int(row.get("epoch", -1)) == best_epoch:
                best_row.update({k: v for k, v in row.items() if k.startswith("val_") or k.startswith("train_")})
                break
    append_csv(results_path, best_row)

    print("=" * 80)
    print(f"[BERT4Rec-RecG] train summary saved: {summary_path}")
    print(f"[BERT4Rec-RecG] source-val row appended: {results_path}")
    print(f"total_wall_time={fmt_sec(time.perf_counter() - total_start)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
