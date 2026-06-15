from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from isddg.config import load_config
from isddg.utils.seed import set_seed
from isddg.utils.device import get_device
from isddg.utils.io import append_csv
from isddg.data.io import load_interactions, group_user_sequences, build_target_eval_samples
from isddg.data.dataset import PrefixDataset, collate_prefix
from isddg.features.semantic import load_semantic_embeddings
from isddg.features.role_store import load_role_table, align_feature_table
from isddg.models.backbone import FeatureBERT4Rec
from isddg.models.isddg import ISDDGModel
from isddg.evaluation.evaluator import evaluate_sampled


def build_model_for_domain(cfg, checkpoint_path, domain, role_source_path, device):
    df, item_map = load_interactions(cfg["data_root"], target)
    sem = load_semantic_embeddings(cfg["data_root"], target, item_map=item_map)[: len(item_map) + 1]
    # Important: for strict zero-shot, target role priors should be predicted from text.
    # This initial skeleton reuses source role prior shape as a placeholder if target role prior is absent.
    source_role = load_role_table(role_source_path)
    K = source_role.size(-1)
    role = torch.full((len(item_map) + 1, K), 1.0 / K)
    role[0] = 0.0

    backbone = FeatureBERT4Rec(
        item_features=sem, role_features=role, role_alpha=cfg["model"]["role_alpha"],
        hidden_dim=cfg["model"]["hidden_dim"], max_len=cfg["data"]["max_len"],
        num_layers=cfg["model"]["num_layers"], num_heads=cfg["model"]["num_heads"],
        dropout=cfg["model"]["dropout"],
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = ISDDGModel(
        backbone=backbone,
        role_table=role,
        prototype_keys=ckpt.get("prototype_keys"),
        prototype_values=ckpt.get("prototype_values"),
        top_m=cfg["prototypes"]["top_m"],
        proto_temperature=cfg["prototypes"]["temperature"],
        lambda_dyn=cfg["model"]["lambda_dyn"],
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.to(device)
    return model, df, item_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/isddg_initial.yaml")
    ap.add_argument("--checkpoint", default="artifacts/checkpoints/isddg.pt")
    ap.add_argument("--targets", default=None, help="comma separated; overrides config")
    ap.add_argument("--results_path", default="results/zero_shot_results.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["seed"])
    device = get_device(cfg["training"]["device"])
    targets = [x.strip() for x in args.targets.split(",")] if args.targets else cfg["targets"]

    role_source_path = Path(cfg["paths"]["role_dir"]) / "source_role_table.pt"
    for target in targets:
        model, df, item_map = build_model_for_domain(cfg, args.checkpoint, target, role_source_path, device)
        seqs = group_user_sequences(df, min_len=cfg["data"]["min_len"])
        samples = build_target_eval_samples(seqs, max_len=cfg["data"]["max_len"])
        ds = PrefixDataset(samples, len(item_map), cfg["data"]["max_len"])
        loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=collate_prefix)
        metrics = evaluate_sampled(
            model, loader, num_items=len(item_map), device=device,
            num_negatives=cfg["data"]["eval_negatives"], seed=cfg["data"]["seed"],
        )
        row = {"model": "ISDDG_initial", "source": cfg["source"], "target": target, **metrics}
        append_csv(row, args.results_path)
        print(row)


if __name__ == "__main__":
    main()
