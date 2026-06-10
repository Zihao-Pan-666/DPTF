from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from isddg.config import load_config, ensure_dirs
from isddg.utils.seed import set_seed
from isddg.utils.device import get_device
from isddg.data.io import load_interactions, group_user_sequences, split_source_prefix_samples
from isddg.data.dataset import PrefixDataset, collate_prefix
from isddg.features.semantic import load_semantic_embeddings
from isddg.baselines.semantic_only import build_semantic_only_model
from isddg.training.trainer import train_sequence_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/isddg_initial.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg["data"]["seed"])
    device = get_device(cfg["training"]["device"])

    source = cfg["source"]
    df, item_map = load_interactions(cfg["data_root"], source)
    seqs = group_user_sequences(df, min_len=cfg["data"]["min_len"])
    samples = split_source_prefix_samples(seqs, max_len=cfg["data"]["max_len"])
    ds = PrefixDataset(samples, num_items=len(item_map), max_len=cfg["data"]["max_len"])
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=collate_prefix)

    df, item_map = load_interactions(cfg["data_root"], source)
    sem = load_semantic_embeddings(cfg["data_root"], source, item_map=item_map)[: len(item_map) + 1]
    model = build_semantic_only_model(sem, cfg)
    train_sequence_model(
        model, loader, num_items=len(item_map), device=device,
        epochs=cfg["training"]["epochs"], lr=cfg["training"]["lr"],
        train_negatives=cfg["data"]["train_negatives"],
    )
    out = Path(cfg["paths"]["checkpoint_dir"]) / "semantic_only.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "cfg": cfg, "num_items": len(item_map)}, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
