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
from isddg.data.io import load_interactions, group_user_sequences, build_target_eval_samples
from isddg.data.dataset import PrefixDataset, collate_prefix
from isddg.features.semantic import load_semantic_embeddings
from isddg.models.llm_recg import LLMRecGBERT4Rec
from isddg.evaluation.llm_recg_evaluator import evaluate_llm_recg_ranking


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
    ap = argparse.ArgumentParser(description="Evaluate official-style BERT4Rec-RecG zero-shot transfer.")
    ap.add_argument("--config", default="configs/llm_recg.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--embedding_dir", default=None)
    ap.add_argument("--results_path", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--disable_patterns", action="store_true")
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


def build_source_user_history_samples(seqs, max_len: int):
    samples = []
    for s in seqs:
        items = list(s["items"])
        times = list(s["times"])
        if len(items) < 2:
            continue
        samples.append({
            "user": s["user"],
            "history": items[:-1][-max_len:],
            "history_times": times[:-1][-max_len:],
            "target": items[-1],
            "target_time": times[-1],
        })
    return samples


def main():
    total_start = time.perf_counter()
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    source = args.source or cfg.get("source", "amazon_movies_and_tv")
    targets = csv_list(args.targets) or list(cfg.get("targets", ["amazon_cds_and_vinyl"]))
    data_root = args.data_root or cfg.get("data_root", "./data")
    embedding_dir = args.embedding_dir or cfg_get(cfg, "data", "embedding_dir", "semantic_embeddings")
    seed = args.seed if args.seed is not None else int(cfg_get(cfg, "data", "seed", 2026))
    max_len = int(cfg_get(cfg, "data", "max_len", 50))
    min_len = int(cfg_get(cfg, "data", "min_len", 3))
    batch_size = int(cfg_get(cfg, "training", "batch_size", 128))
    num_workers = int(cfg_get(cfg, "data", "num_workers", 0))
    ranking = cfg_get(cfg, "data", "ranking", "sampled")
    eval_negatives = int(cfg_get(cfg, "data", "eval_negatives", 100))

    num_patterns = int(cfg_get(cfg, "llm_recg", "num_sequential_patterns", 10))
    kmeans_batch_size = int(cfg_get(cfg, "llm_recg", "pattern_kmeans_batch_size", 4096))
    pattern_max_users = int(cfg_get(cfg, "llm_recg", "pattern_max_users", 0))
    use_minibatch = bool(cfg_get(cfg, "llm_recg", "pattern_use_minibatch", False))
    use_patterns = bool(cfg_get(cfg, "llm_recg", "use_patterns_in_target", True)) and not args.disable_patterns

    device = get_device(args.device or cfg_get(cfg, "training", "device", "auto"))
    show_progress = not args.no_progress
    set_seed(seed)

    result_dir = Path(cfg.get("paths", {}).get("result_dir", "results/mainline/llm_recg"))
    ckpt_dir = Path(cfg.get("paths", {}).get("checkpoint_dir", "artifacts/checkpoints/llm_recg"))
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else ckpt_dir / f"bert4rec_recg_{source}_seed{seed}.pt"
    results_path = Path(args.results_path) if args.results_path else result_dir / "bert4rec_recg_zero_shot.csv"

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hp = ckpt.get("model_hparams", {})

    print("=" * 80)
    print("[BERT4Rec-RecG] Zero-shot evaluation")
    print(f"source={source}")
    print(f"targets={targets}")
    print(f"checkpoint={checkpoint_path}")
    print(f"best_source_epoch={ckpt.get('best_epoch', -1)}")
    print(f"use_patterns={use_patterns}")
    print(f"ranking={ranking} eval_negatives={eval_negatives}")
    print(f"device={device}")
    print("=" * 80)

    with timed("load source features and initialize model"):
        src_df, src_item_map, src_features = load_domain_features(data_root, source, embedding_dir)
        model = LLMRecGBERT4Rec(
            item_features=src_features,
            hidden_dim=int(hp.get("hidden_dim", cfg_get(cfg, "model", "hidden_dim", 256))),
            max_len=max_len,
            num_layers=int(hp.get("num_layers", cfg_get(cfg, "model", "num_layers", 2))),
            num_heads=int(hp.get("num_heads", cfg_get(cfg, "model", "num_heads", 2))),
            dropout=float(hp.get("dropout", cfg_get(cfg, "model", "dropout", 0.5))),
            num_sequential_patterns=int(hp.get("num_sequential_patterns", num_patterns)),
            pattern_fusion=str(hp.get("pattern_fusion", cfg_get(cfg, "model", "pattern_fusion", "residual"))),
            pattern_residual_weight=float(hp.get("pattern_residual_weight", cfg_get(cfg, "model", "pattern_residual_weight", 0.5))),
            init_pattern_fusion_as_residual=bool(cfg_get(cfg, "model", "init_pattern_fusion_as_residual", True)),
        )
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[Checkpoint] missing={missing} unexpected={unexpected}")
        model.to(device)
        model.eval()

    if use_patterns:
        with timed("extract source sequential patterns"):
            src_seqs = group_user_sequences(src_df, min_len=min_len)
            pattern_samples = build_source_user_history_samples(src_seqs, max_len=max_len)
            pattern_ds = PrefixDataset(pattern_samples, num_items=len(src_item_map), max_len=max_len)
            pattern_loader = DataLoader(
                pattern_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_prefix,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            patterns = model.extract_sequential_patterns(
                history_loader=pattern_loader,
                device=device,
                num_patterns=num_patterns,
                kmeans_batch_size=kmeans_batch_size,
                max_users=pattern_max_users,
                show_progress=show_progress,
                use_minibatch=use_minibatch,
            )
            print(f"[Patterns] shape={tuple(patterns.shape)} source_users={len(pattern_samples):,}")
    else:
        model.use_sequential_patterns = False

    for target in targets:
        target_start = time.perf_counter()
        print(f"\n[Target] {target}")

        with timed(f"load target interactions and embeddings: {target}"):
            tgt_df, tgt_item_map, tgt_features = load_domain_features(data_root, target, embedding_dir)
            tgt_seqs = group_user_sequences(tgt_df, min_len=min_len)
            eval_samples = build_target_eval_samples(tgt_seqs, max_len=max_len)

        if int(tgt_features.size(1)) != int(ckpt.get("embedding_dim", tgt_features.size(1))):
            raise ValueError(
                f"Embedding dim mismatch: target={target}, target_dim={tgt_features.size(1)}, "
                f"checkpoint_dim={ckpt.get('embedding_dim')}"
            )

        with timed("switch model item table to target domain"):
            model.load_new_pretrain_embeddings(tgt_features)
            model.use_sequential_patterns = bool(use_patterns and model.sequential_patterns.numel() > 0)

        eval_ds = PrefixDataset(eval_samples, num_items=len(tgt_item_map), max_len=max_len)
        eval_loader = DataLoader(
            eval_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_prefix,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        metrics = evaluate_llm_recg_ranking(
            model=model,
            loader=eval_loader,
            num_items=len(tgt_item_map),
            device=device,
            ranking_mode=ranking,
            num_negatives=eval_negatives,
            seed=seed,
            ks=(10, 20),
            tie_policy="worst",
            is_target_domain=use_patterns,
            show_progress=show_progress,
            progress_desc=f"BERT4Rec-RecG {source}->{target}",
        )

        row = {
            "model": "bert4rec_recg",
            "stage": "zero_shot",
            "source": source,
            "target": target,
            "seed": seed,
            "checkpoint": str(checkpoint_path),
            "best_source_epoch": int(ckpt.get("best_epoch", -1)),
            "ranking_mode": ranking,
            "eval_negatives": eval_negatives,
            "num_target_users": len(eval_samples),
            "num_target_items": len(tgt_item_map),
            "embedding_dim": int(tgt_features.size(1)),
            "use_source_sequential_patterns": bool(use_patterns),
            "num_sequential_patterns": int(num_patterns),
        }
        row.update(metrics)
        row["target_total_elapsed_sec"] = float(time.perf_counter() - target_start)
        row["protocol_target_interactions"] = "forward_only_no_training_no_tuning"

        metric_path = result_dir / f"bert4rec_recg_{source}_to_{target}_{ranking}_seed{seed}.json"
        save_json(row, metric_path)
        append_csv(results_path, row)

        print("-" * 80)
        print(f"[target={target}]")
        for k in ["Recall@10", "NDCG@10", "MRR@10", "Recall@20", "NDCG@20", "MRR@20"]:
            print(f"{k}: {row[k]:.6f}")
        print(f"tie_case_ratio={row['tie_case_ratio']:.6f} avg_tie_items={row['avg_tie_items']:.3f}")
        print(f"saved_metrics={metric_path}")

    print("=" * 80)
    print(f"[BERT4Rec-RecG] zero-shot results appended: {results_path}")
    print(f"total_wall_time={fmt_sec(time.perf_counter() - total_start)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
