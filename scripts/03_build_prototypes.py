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
from isddg.prototypes.bank import build_prototype_bank


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/isddg_initial.yaml")
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/semantic_only.pt")
    ap.add_argument("--M", type=int, default=None)
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
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate_prefix)

    df, item_map = load_interactions(cfg["data_root"], source)
    sem = load_semantic_embeddings(cfg["data_root"], source, item_map=item_map)[: len(item_map) + 1]
    role_path = Path(cfg["paths"]["role_dir"]) / "source_role_table.pt"
    role = align_feature_table(load_role_table(role_path), len(item_map))

    backbone = FeatureBERT4Rec(
        item_features=sem, role_features=role, role_alpha=cfg["model"]["role_alpha"],
        hidden_dim=cfg["model"]["hidden_dim"], max_len=cfg["data"]["max_len"],
        num_layers=cfg["model"]["num_layers"], num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = backbone.load_state_dict(ckpt["model_state"], strict=False)
    print(f"Loaded semantic checkpoint. missing={missing}, unexpected={unexpected}")

    isddg = ISDDGModel(backbone, role_table=role, lambda_dyn=cfg["model"]["lambda_dyn"]).to(device)
    isddg.eval()
    keys, next_roles = [], []
    for batch in loader:
        hist = batch["history"].to(device)
        target = batch["target"].to(device)
        user_h = isddg.backbone(hist)
        q, _, _ = isddg.build_query_key(hist, user_h)
        keys.append(q.cpu())
        next_roles.append(role[target.cpu()])
    keys = torch.cat(keys, dim=0)
    next_roles = torch.cat(next_roles, dim=0)

    M = args.M or cfg["prototypes"]["M"]
    bank = build_prototype_bank(keys, next_roles, M=M, seed=cfg["data"]["seed"])
    out = Path(cfg["paths"]["prototype_dir"]) / "prototype_bank.pt"
    bank.save(out)
    print(f"Saved prototype bank to {out}. keys={tuple(bank.keys.shape)}, values={tuple(bank.values.shape)}")


if __name__ == "__main__":
    main()
