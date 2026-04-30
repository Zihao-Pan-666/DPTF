from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from diagnostic.data_utils import SequenceDataset, load_semantic_embeddings
from diagnostic.features import (
    build_source_only_target_features,
    load_or_build_popularity_features,
)
from diagnostic.models import FeatureSeqRec, GatedFusionSeqRec, NaiveFusionSeqRec
from diagnostic.training import append_result_csv, evaluate, train_one_model


def parse_args():
    p = argparse.ArgumentParser("SemDyn 1-week diagnostic runner")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--source", type=str, default="amazon_movies_and_tv")
    p.add_argument("--targets", type=str, default="amazon_cds_and_vinyl,steam")
    p.add_argument("--model", type=str, required=True,
                   choices=["semantic", "dynamics", "naive_fusion", "gated_fusion"])
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--hidden_units", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=2)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--train_negatives", type=int, default=5)
    p.add_argument("--eval_negatives", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--force_rebuild_popularity", action="store_true")

    # New: controls only target-domain popularity construction at evaluation time.
    # The source domain still uses full source statistics for training.
    p.add_argument("--target_popularity_mode", type=str, default="full",
                   choices=["full", "history_only", "source_only"],
                   help=(
                       "full: previous diagnostic behavior; uses full target CSV. "
                       "history_only: uses only items[:-2] from each target user. "
                       "source_only: no target popularity statistics; uses zeros/source mean."
                   ))
    p.add_argument("--source_only_strategy", type=str, default="source_mean",
                   choices=["zeros", "source_mean"],
                   help="Target feature strategy when --target_popularity_mode source_only.")

    p.add_argument("--results_path", type=str, default="outputs/diagnostic_results.csv")
    return p.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args, source_sem, source_pop):
    common = dict(hidden_units=args.hidden_units, max_len=args.max_len, num_heads=args.num_heads,
                  num_layers=args.num_layers, dropout=args.dropout)
    if args.model == "semantic":
        return FeatureSeqRec(source_sem, **common)
    if args.model == "dynamics":
        return FeatureSeqRec(source_pop, **common)
    if args.model == "naive_fusion":
        return NaiveFusionSeqRec(source_sem, source_pop, **common)
    if args.model == "gated_fusion":
        return GatedFusionSeqRec(source_sem, source_pop, **common)
    raise ValueError(args.model)


def switch_target_features(model, model_name: str, sem, pop):
    if model_name == "semantic":
        model.set_item_features(sem)
    elif model_name == "dynamics":
        model.set_item_features(pop)
    else:
        model.set_item_features(sem, pop)


def load_target_popularity(args, target: str, target_num_items: int, source_pop: torch.Tensor) -> torch.Tensor:
    if args.target_popularity_mode == "source_only":
        return build_source_only_target_features(
            n_target_items=target_num_items,
            source_features=source_pop,
            strategy=args.source_only_strategy,
        )

    target_pop = load_or_build_popularity_features(
        args.data_root,
        target,
        force=args.force_rebuild_popularity,
        mode=args.target_popularity_mode,
    )
    return target_pop[: target_num_items + 1]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"target_popularity_mode={args.target_popularity_mode}")

    source_ds = SequenceDataset(args.data_root, args.source, args.max_len)
    train_loader = DataLoader(source_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    source_sem = load_semantic_embeddings(args.data_root, args.source)

    # Keep source popularity as full source statistics for training. This makes
    # history_only/source_only results directly comparable with the previous run;
    # only target-side popularity availability is changed.
    source_pop = load_or_build_popularity_features(
        args.data_root,
        args.source,
        force=args.force_rebuild_popularity,
        mode="full",
    )

    # Ensure feature tables match dataset item cardinality.
    source_sem = source_sem[: source_ds.get_num_items() + 1]
    source_pop = source_pop[: source_ds.get_num_items() + 1]

    model = build_model(args, source_sem, source_pop)
    train_one_model(model, train_loader, source_ds.get_num_items(), device,
                    epochs=args.epochs, lr=args.lr, train_negatives=args.train_negatives)

    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    for target in targets:
        target_ds = SequenceDataset(args.data_root, target, args.max_len)
        target_loader = DataLoader(target_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        target_sem = load_semantic_embeddings(args.data_root, target)[: target_ds.get_num_items() + 1]
        target_pop = load_target_popularity(args, target, target_ds.get_num_items(), source_pop)
        target_pop = target_pop[: target_ds.get_num_items() + 1]

        switch_target_features(model, args.model, target_sem.to(device), target_pop.to(device))
        metrics = evaluate(model, target_loader, target_ds.get_num_items(), device, num_negatives=args.eval_negatives)
        row = {
            "model": args.model,
            "source": args.source,
            "target": target,
            "target_popularity_mode": args.target_popularity_mode,
            "source_only_strategy": args.source_only_strategy if args.target_popularity_mode == "source_only" else "",
            "epochs": args.epochs,
            "hidden_units": args.hidden_units,
            **metrics,
        }
        append_result_csv(args.results_path, row)
        print(row)


if __name__ == "__main__":
    main()
