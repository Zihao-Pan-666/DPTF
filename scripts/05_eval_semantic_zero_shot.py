from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from isddg.config import load_config
from isddg.utils.seed import set_seed
from isddg.utils.device import get_device
from isddg.utils.io import save_json, append_csv
from isddg.data.io import load_interactions, group_user_sequences, build_target_eval_samples
from isddg.data.dataset import PrefixDataset, collate_prefix
from isddg.features.semantic import load_semantic_embeddings
from isddg.models.backbone import FeatureBERT4Rec
from isddg.evaluation.semantic_evaluator import evaluate_semantic_ranking


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
    ap = argparse.ArgumentParser(description="Strict zero-shot evaluation for V0 Semantic-only BERT4Rec.")
    ap.add_argument("--config", default="configs/semantic_v0.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--targets", default=None, help="comma-separated target domains")
    ap.add_argument("--source", default=None)
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--embedding_dir", default="semantic_embeddings")
    ap.add_argument("--results_path", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--min_len", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--eval_negatives", type=int, default=None)
    ap.add_argument("--ranking_mode", choices=["sampled", "full"], default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    return ap.parse_args()


def _targets_from_args(args, cfg) -> List[str]:
    if args.targets:
        return [x.strip() for x in args.targets.split(",") if x.strip()]
    return list(cfg.get("targets", []))


def main():
    total_start = time.perf_counter()
    args = parse_args()
    cfg = load_config(args.config)
    data_root = args.data_root or cfg.get("data_root", "./data")
    seed = args.seed if args.seed is not None else cfg_get(cfg, "data", "seed", 2026)
    max_len = args.max_len if args.max_len is not None else cfg_get(cfg, "data", "max_len", 50)
    min_len = args.min_len if args.min_len is not None else cfg_get(cfg, "data", "min_len", 3)
    batch_size = args.batch_size if args.batch_size is not None else cfg_get(cfg, "training", "batch_size", 128)
    num_workers = args.num_workers if args.num_workers is not None else cfg_get(cfg, "data", "num_workers", 0)
    eval_negatives = args.eval_negatives if args.eval_negatives is not None else cfg_get(cfg, "data", "eval_negatives", 100)
    ranking_mode = args.ranking_mode or cfg_get(cfg, "data", "ranking", "sampled")
    source = args.source or cfg.get("source", "amazon_movies_and_tv")
    targets = _targets_from_args(args, cfg)
    device = get_device(args.device or cfg_get(cfg, "training", "device", "auto"))
    show_progress = not args.no_progress
    set_seed(seed)

    checkpoint_path = Path(args.checkpoint)
    with timed("load source checkpoint"):
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    model_hparams = ckpt.get("model_hparams", {})
    hidden_dim = int(model_hparams.get("hidden_dim", cfg_get(cfg, "model", "hidden_dim", 128)))
    num_layers = int(model_hparams.get("num_layers", cfg_get(cfg, "model", "num_layers", 2)))
    num_heads = int(model_hparams.get("num_heads", cfg_get(cfg, "model", "num_heads", 2)))
    dropout = float(model_hparams.get("dropout", cfg_get(cfg, "model", "dropout", 0.2)))

    result_dir = Path(cfg.get("paths", {}).get("result_dir", "results"))
    results_path = Path(args.results_path) if args.results_path else result_dir / "semantic_v0_zero_shot.csv"

    print("=" * 80)
    print("[SemanticV0] Strict zero-shot evaluation")
    print(f"source={source}")
    print(f"targets={targets}")
    print(f"checkpoint={checkpoint_path}")
    print(f"best_source_epoch={ckpt.get('best_epoch', -1)}")
    print(f"ranking_mode={ranking_mode}")
    print(f"eval_negatives={eval_negatives if ranking_mode == 'sampled' else 'FULL'}")
    print(f"seed={seed}")
    print(f"device={device}")
    print("=" * 80)

    for target in targets:
        target_start = time.perf_counter()
        print(f"\n[Target] {target}")

        prep = tqdm(total=6, desc=f"prepare {target}", unit="step", dynamic_ncols=True, disable=not show_progress)

        with timed(f"load target interactions: {target}"):
            df, item_map = load_interactions(data_root, target)
        prep.update(1)

        with timed("group target user sequences"):
            seqs = group_user_sequences(df, min_len=min_len)
        prep.update(1)

        with timed("build target eval samples"):
            eval_samples = build_target_eval_samples(seqs, max_len=max_len)
        prep.update(1)

        with timed("load and align target semantic embeddings"):
            item_features = load_semantic_embeddings(
                data_root=data_root,
                domain=target,
                item_map=item_map,
                embedding_dir=args.embedding_dir,
                strict=True,
            )[: len(item_map) + 1]
            item_features[0] = 0.0
        prep.update(1)

        embedding_dim = int(item_features.size(1))
        ckpt_embedding_dim = int(ckpt.get("embedding_dim", embedding_dim))
        if embedding_dim != ckpt_embedding_dim:
            raise ValueError(
                f"Embedding dim mismatch for target={target}: "
                f"target_dim={embedding_dim}, checkpoint_dim={ckpt_embedding_dim}"
            )

        with timed("initialize target-domain semantic model and load source weights"):
            model = FeatureBERT4Rec(
                item_features=item_features,
                hidden_dim=hidden_dim,
                max_len=max_len,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
            model.load_state_dict(ckpt["model_state"], strict=True)
            model.to(device)
        prep.update(1)

        with timed("build target eval DataLoader"):
            eval_ds = PrefixDataset(eval_samples, num_items=len(item_map), max_len=max_len)
            eval_loader = DataLoader(
                eval_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_prefix,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
        prep.update(1)
        prep.close()

        print(
            f"[Data:{target}] interactions={len(df):,} users={len(seqs):,} "
            f"items={len(item_map):,} eval_users={len(eval_samples):,} "
            f"eval_batches={len(eval_loader):,} embedding_shape={tuple(item_features.shape)}"
        )

        metrics = evaluate_semantic_ranking(
            model=model,
            loader=eval_loader,
            num_items=len(item_map),
            device=device,
            ranking_mode=ranking_mode,
            num_negatives=eval_negatives,
            seed=seed,
            ks=(10, 20),
            tie_policy="worst",
            show_progress=show_progress,
            progress_desc=f"eval {source} -> {target}",
        )

        row = {
            "model": "semantic_v0",
            "stage": "zero_shot",
            "source": source,
            "target": target,
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "best_source_epoch": ckpt.get("best_epoch", -1),
            "ranking_mode": ranking_mode,
            "eval_negatives": eval_negatives if ranking_mode == "sampled" else "",
            "num_target_users": len(eval_samples),
            "num_target_items": len(item_map),
            "embedding_dim": embedding_dim,
            **metrics,
            "target_total_elapsed_sec": time.perf_counter() - target_start,
        }

        metrics_json = result_dir / f"semantic_v0_{source}_to_{target}_{ranking_mode}_seed{seed}.json"
        with timed("save target metrics"):
            append_csv(row, results_path)
            save_json(row, metrics_json)

        print("-" * 80)
        print(f"[target={target}]")
        for key in ["Recall@10", "NDCG@10", "MRR@10", "Recall@20", "NDCG@20", "MRR@20"]:
            print(f"{key}: {metrics[key]:.6f}")
        print(
            f"tie_case_ratio={metrics['tie_case_ratio']:.6f} "
            f"all_equal_ratio={metrics['all_equal_ratio']:.6f} "
            f"avg_tie_items={metrics['avg_tie_items']:.3f}"
        )
        print(
            f"num_eval_users={metrics['num_eval_users']} "
            f"eval_time={fmt_sec(metrics['eval_elapsed_sec'])} "
            f"eval_speed={metrics['eval_users_per_sec']:.1f} users/s"
        )
        print(f"saved_metrics={metrics_json}")

    total_elapsed = time.perf_counter() - total_start
    print("=" * 80)
    print(f"[SemanticV0] Appended zero-shot results to {results_path}")
    print(f"total_wall_time={fmt_sec(total_elapsed)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
