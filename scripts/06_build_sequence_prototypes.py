from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

from isddg.config import ensure_dirs, load_config
from isddg.data.dataset import PrefixDataset, collate_prefix
from isddg.data.io import group_user_sequences, load_interactions
from isddg.data.semantic_splits import build_source_train_val_samples
from isddg.features.dynamic_feature_store import load_pt_feature_table
from isddg.features.semantic import load_semantic_embeddings
from isddg.models.backbone import FeatureBERT4Rec
from isddg.prototypes.sequence_dynamic import SequenceDynamicPrototypeBank
from isddg.utils.device import get_device
from isddg.utils.io import save_json
from isddg.utils.seed import set_seed


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


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description="Build source sequence dynamic prototype bank.")
    ap.add_argument("--config", default="configs/sequence_prototype.yaml")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--source", default=None)
    ap.add_argument("--embedding_dir", default=None)
    ap.add_argument("--semantic_checkpoint", default=None)
    ap.add_argument("--source_dynamic_table", default=None)
    ap.add_argument("--source_role_table", default=None)
    ap.add_argument("--out_path", default=None)
    ap.add_argument("--summary_path", default=None)
    ap.add_argument("--M", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--kmeans_batch_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no_progress", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    data_root = args.data_root or cfg.get("data_root", "./data")
    source = args.source or cfg.get("source", "amazon_movies_and_tv")
    seed = args.seed if args.seed is not None else cfg_get(cfg, "data", "seed", 2026)
    max_len = int(cfg_get(cfg, "data", "max_len", 50))
    min_len = int(cfg_get(cfg, "data", "min_len", 3))
    num_workers = int(cfg_get(cfg, "data", "num_workers", 0))
    embedding_dir = args.embedding_dir or cfg_get(cfg, "data", "embedding_dir", "semantic_embeddings")
    batch_size = int(args.batch_size or cfg_get(cfg, "prototype", "build_batch_size", 512))
    kmeans_batch_size = int(args.kmeans_batch_size or cfg_get(cfg, "prototype", "kmeans_batch_size", 8192))
    M = int(args.M or cfg_get(cfg, "prototype", "M", 128))
    show_progress = not args.no_progress

    paths = cfg.get("paths", {})
    semantic_checkpoint = Path(args.semantic_checkpoint or cfg_get(cfg, "semantic", "checkpoint", ""))
    dynamic_path_cfg = cfg_get(cfg, "prototype", "source_dynamic_table", "")
    role_path_cfg = cfg_get(cfg, "prototype", "source_role_table", "")
    source_dynamic_table = Path(args.source_dynamic_table or dynamic_path_cfg) if (args.source_dynamic_table or dynamic_path_cfg) else None
    source_role_table = Path(args.source_role_table or role_path_cfg) if (args.source_role_table or role_path_cfg) else None
    out_path = Path(args.out_path or paths.get("prototype_bank", f"artifacts/prototypes/sequence_dynamic_{source}_M{M}_seed{seed}.pt"))
    summary_path = Path(args.summary_path or paths.get("prototype_summary", f"results/mainline/sequence_dynamic_prototype_{source}_M{M}_seed{seed}_summary.json"))

    set_seed(seed)
    device = get_device(args.device or cfg_get(cfg, "training", "device", "auto"))

    print("=" * 80)
    print("[SequencePrototype] Build source prototype bank")
    print(f"source={source}")
    print(f"data_root={data_root}")
    print(f"embedding_dir={embedding_dir}")
    print(f"semantic_checkpoint={semantic_checkpoint}")
    print(f"source_dynamic_table={source_dynamic_table}")
    print(f"source_role_table={source_role_table}")
    print(f"M={M} batch_size={batch_size} kmeans_batch_size={kmeans_batch_size}")
    print(f"out_path={out_path}")
    print(f"device={device}")
    print("=" * 80)

    # Defensive path handling: cfg["paths"] may contain file paths. If an older
    # ensure_dirs() accidentally created a directory named *.pt or *.json, fail
    # early with a clear message instead of a low-level torch.save error.
    if out_path.exists() and out_path.is_dir():
        raise RuntimeError(
            f"Output path points to an existing directory, not a file: {out_path}\n"
            "Remove that directory first, then rerun. On PowerShell:\n"
            f"  Remove-Item -Recurse -Force '{out_path}'"
        )
    if summary_path.exists() and summary_path.is_dir():
        raise RuntimeError(
            f"Summary path points to an existing directory, not a file: {summary_path}\n"
            "Remove that directory first, then rerun. On PowerShell:\n"
            f"  Remove-Item -Recurse -Force '{summary_path}'"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    df, item_map = load_interactions(data_root, source)
    seqs = group_user_sequences(df, min_len=min_len)
    train_samples, val_samples = build_source_train_val_samples(seqs, max_len=max_len, min_prefix=1)
    if not train_samples:
        raise RuntimeError("No source training prefix samples were built.")

    sem = load_semantic_embeddings(
        data_root=data_root,
        domain=source,
        item_map=item_map,
        embedding_dir=embedding_dir,
        strict=True,
    )[: len(item_map) + 1]
    sem[0] = 0.0

    model = load_semantic_model(cfg, semantic_checkpoint, sem, max_len=max_len, device=device)

    dyn_table = None
    if source_dynamic_table is not None and source_dynamic_table.exists():
        dyn_table = load_pt_feature_table(source_dynamic_table, num_items=len(item_map)).float()
        print(f"[DynamicValue] loaded {source_dynamic_table} shape={tuple(dyn_table.shape)}")
    else:
        print("[DynamicValue] not used")

    role_table = None
    if source_role_table is not None and source_role_table.exists():
        role_table = load_pt_feature_table(source_role_table, num_items=len(item_map)).float()
        print(f"[RoleValue] loaded {source_role_table} shape={tuple(role_table.shape)}")
    else:
        print("[RoleValue] not used")

    ds = PrefixDataset(train_samples, num_items=len(item_map), max_len=max_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_prefix,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    M = min(M, len(train_samples))
    kmeans = MiniBatchKMeans(
        n_clusters=M,
        random_state=seed,
        batch_size=max(kmeans_batch_size, M * 4),
        n_init="auto",
    )

    print("[Pass 1/2] partial_fit KMeans on source prefix states")
    first = True
    for batch in tqdm(loader, desc="kmeans-fit", dynamic_ncols=True, disable=not show_progress):
        hist = batch["history"].to(device, non_blocking=True)
        keys = model(hist).detach().cpu().numpy().astype(np.float32)
        if first:
            if keys.shape[0] < M:
                raise RuntimeError(f"First batch size {keys.shape[0]} is smaller than M={M}. Increase build_batch_size or reduce M.")
            first = False
        kmeans.partial_fit(keys)

    centers = kmeans.cluster_centers_.astype(np.float32)
    hidden_dim = int(centers.shape[1])
    dyn_dim = int(dyn_table.size(1)) if dyn_table is not None else 0
    role_dim = int(role_table.size(1)) if role_table is not None else 0

    sem_sums = np.zeros((M, hidden_dim), dtype=np.float64)
    dyn_sums = np.zeros((M, dyn_dim), dtype=np.float64)
    role_sums = np.zeros((M, role_dim), dtype=np.float64)
    counts = np.zeros(M, dtype=np.int64)
    global_sem_sum = np.zeros(hidden_dim, dtype=np.float64)
    global_dyn_sum = np.zeros(dyn_dim, dtype=np.float64)
    global_role_sum = np.zeros(role_dim, dtype=np.float64)
    total_values = 0

    print("[Pass 2/2] aggregate next-item semantic/dynamic/role values")
    for batch in tqdm(loader, desc="aggregate", dynamic_ncols=True, disable=not show_progress):
        hist = batch["history"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        keys_t = model(hist)
        keys_np = keys_t.detach().cpu().numpy().astype(np.float32)
        labels = kmeans.predict(keys_np)

        sem_values = model.encode_items(target).detach().cpu().numpy().astype(np.float32)
        target_cpu = target.detach().cpu()
        dyn_values = dyn_table[target_cpu].numpy().astype(np.float32) if dyn_table is not None else None
        role_values = role_table[target_cpu].numpy().astype(np.float32) if role_table is not None else None

        for i, lab in enumerate(labels):
            lab = int(lab)
            counts[lab] += 1
            sem_sums[lab] += sem_values[i]
            global_sem_sum += sem_values[i]
            if dyn_values is not None:
                dyn_sums[lab] += dyn_values[i]
                global_dyn_sum += dyn_values[i]
            if role_values is not None:
                role_sums[lab] += role_values[i]
                global_role_sum += role_values[i]
            total_values += 1

    sem_values = sem_sums / np.maximum(counts[:, None], 1)
    dyn_values = dyn_sums / np.maximum(counts[:, None], 1) if dyn_dim > 0 else np.zeros((M, 0), dtype=np.float32)
    role_values = role_sums / np.maximum(counts[:, None], 1) if role_dim > 0 else np.zeros((M, 0), dtype=np.float32)

    empty = counts == 0
    if empty.any():
        sem_values[empty] = global_sem_sum / max(total_values, 1)
        if dyn_dim > 0:
            dyn_values[empty] = global_dyn_sum / max(total_values, 1)
        if role_dim > 0:
            role_values[empty] = global_role_sum / max(total_values, 1)

    meta = {
        "format": "sequence_dynamic_prototype_v1",
        "source": source,
        "seed": seed,
        "M": int(M),
        "num_items": int(len(item_map)),
        "num_train_prefix_samples": int(len(train_samples)),
        "num_source_val_samples": int(len(val_samples)),
        "max_len": int(max_len),
        "semantic_checkpoint": str(semantic_checkpoint),
        "source_dynamic_table": str(source_dynamic_table) if source_dynamic_table is not None else "",
        "source_role_table": str(source_role_table) if source_role_table is not None else "",
        "hidden_dim": int(hidden_dim),
        "dynamic_dim": int(dyn_dim),
        "role_dim": int(role_dim),
        "empty_clusters": int(empty.sum()),
        "min_support": int(counts.min()) if len(counts) else 0,
        "median_support": float(np.median(counts)) if len(counts) else 0.0,
        "max_support": int(counts.max()) if len(counts) else 0,
        "elapsed_sec": float(time.perf_counter() - start),
        "protocol": {
            "target_interaction_usage": "none",
            "prototype_construction": "source_train_prefixes_only",
            "value_semantic": "source next-item projected semantic vector",
            "value_dynamic": "source next-item oracle continuous dynamic vector, if provided",
            "value_role": "source next-item soft role distribution, if provided",
        },
    }

    bank = SequenceDynamicPrototypeBank(
        keys=torch.from_numpy(centers),
        semantic_values=torch.from_numpy(sem_values.astype(np.float32)),
        dynamic_values=torch.from_numpy(dyn_values.astype(np.float32)),
        role_values=torch.from_numpy(role_values.astype(np.float32)),
        counts=torch.from_numpy(counts.astype(np.int64)),
        meta=meta,
    )
    bank.save(out_path)
    save_json(meta, summary_path)

    print("=" * 80)
    print(f"[Saved] prototype_bank={out_path}")
    print(f"[Saved] summary={summary_path}")
    print(json.dumps({k: meta[k] for k in ["M", "num_train_prefix_samples", "hidden_dim", "dynamic_dim", "role_dim", "empty_clusters", "min_support", "median_support", "max_support"]}, indent=2))
    print(f"elapsed={fmt_sec(meta['elapsed_sec'])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
