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
from isddg.features.role_store import load_role_table, align_feature_table
from isddg.models.backbone import FeatureBERT4Rec
from isddg.models.isddg import ISDDGModel
from isddg.prototypes.bank import PrototypeBank
from isddg.training.trainer import train_sequence_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/isddg_initial.yaml")
    ap.add_argument("--semantic_checkpoint", default="artifacts/checkpoints/semantic_only.pt")
    ap.add_argument("--prototype_path", default="artifacts/prototypes/prototype_bank.pt")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    set_seed(cfg["data"]["seed"])
    device = get_device(cfg["training"]["device"])

    source = cfg["source"]
    df, item_map = load_interactions(cfg["data_root"], source)
    seqs = group_user_sequences(df, min_len=cfg["data"]["min_len"])
    samples = split_source_prefix_samples(seqs, max_len=cfg["data"]["max_len"])
    ds = PrefixDataset(samples, len(item_map), cfg["data"]["max_len"])
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=collate_prefix)

    df, item_map = load_interactions(cfg["data_root"], source)
    sem = load_semantic_embeddings(cfg["data_root"], source, item_map=item_map)[: len(item_map) + 1]
    role = align_feature_table(load_role_table(Path(cfg["paths"]["role_dir"]) / "source_role_table.pt"), len(item_map))
    bank = PrototypeBank.load(args.prototype_path)

    backbone = FeatureBERT4Rec(
        item_features=sem, role_features=role, role_alpha=cfg["model"]["role_alpha"],
        hidden_dim=cfg["model"]["hidden_dim"], max_len=cfg["data"]["max_len"],
        num_layers=cfg["model"]["num_layers"], num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
    )
    if Path(args.semantic_checkpoint).exists():
        ckpt = torch.load(args.semantic_checkpoint, map_location="cpu")
        backbone.load_state_dict(ckpt["model_state"], strict=False)

    model = ISDDGModel(
        backbone=backbone,
        role_table=role,
        prototype_keys=bank.keys,
        prototype_values=bank.values,
        top_m=cfg["prototypes"]["top_m"],
        proto_temperature=cfg["prototypes"]["temperature"],
        lambda_dyn=cfg["model"]["lambda_dyn"],
    )
    train_sequence_model(
        model, loader, num_items=len(item_map), device=device,
        epochs=cfg["training"]["epochs"], lr=cfg["training"]["lr"],
        train_negatives=cfg["data"]["train_negatives"],
    )
    out = Path(cfg["paths"]["checkpoint_dir"]) / "isddg.pt"
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg,
        "num_items": len(item_map),
        "prototype_keys": bank.keys,
        "prototype_values": bank.values,
    }, out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
