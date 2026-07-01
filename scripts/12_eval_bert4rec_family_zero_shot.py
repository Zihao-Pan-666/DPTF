from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from isddg.data.dataset import collate_prefix
from isddg.evaluation.bert4rec_family_evaluator import evaluate_bert4rec_family
from isddg.features.semantic import load_semantic_embeddings
from isddg.models.bert4rec_family import BERT4RecSemanticFamily
from isddg.utils.bert4rec_family_compat import (
    build_target_samples_compat,
    group_user_sequences_compat,
    load_interactions_compat,
    make_prefix_dataset,
    resolve_device_compat,
    set_seed_compat,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final zero-shot evaluation of a source-selected family checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    return parser.parse_args()


def append_csv(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(row.keys())
    if exists:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle), None)
        if header and header != fieldnames:
            raise RuntimeError(
                f"CSV schema mismatch for {path}. Existing={header}, new={fieldnames}"
            )
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    experiment = cfg["experiment"]
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    evaluation_cfg = dict(cfg["evaluation"])
    source = str(cfg["source"])
    targets = [str(x) for x in cfg["targets"]]
    data_root = str(cfg.get("data_root", "./data"))
    embedding_dir = str(data_cfg.get("embedding_dir", "semantic_embeddings"))
    seed = int(data_cfg.get("seed", 2026))
    run_name = str(experiment["run_name"])

    default_checkpoint = (
        Path(cfg["paths"]["checkpoint_dir"])
        / f"{run_name}_{source}_seed{seed}.pt"
    )
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    set_seed_compat(seed)
    device = resolve_device_compat(str(training_cfg.get("device", "auto")))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if checkpoint.get("source_domain") != source:
        raise RuntimeError(
            f"Checkpoint source={checkpoint.get('source_domain')} but config source={source}"
        )
    if checkpoint.get("best_metric_name") != evaluation_cfg["primary_metric"]:
        raise RuntimeError("Checkpoint selection metric differs from config")

    result_dir = Path(cfg["paths"]["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, object]] = []
    for target_index, target in enumerate(targets):
        interactions, user_map, item_map = load_interactions_compat(data_root, target)
        sequences = group_user_sequences_compat(
            interactions, min_len=int(data_cfg["min_len"])
        )
        samples = build_target_samples_compat(
            sequences=sequences,
            max_len=int(data_cfg["max_len"]),
            min_len=int(data_cfg["min_len"]),
        )
        item_features = load_semantic_embeddings(
            data_root=data_root,
            domain=target,
            item_map=item_map,
            embedding_dir=embedding_dir,
        )

        hparams = checkpoint["model_hparams"]
        model = BERT4RecSemanticFamily(
            item_features=item_features,
            hidden_dim=int(hparams["hidden_dim"]),
            max_len=int(hparams["max_len"]),
            num_layers=int(hparams["num_layers"]),
            num_heads=int(hparams["num_heads"]),
            dropout=float(hparams["dropout"]),
            architecture=str(hparams["architecture"]),
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)

        loader = DataLoader(
            make_prefix_dataset(
                samples,
                num_items=len(item_map),
                max_len=int(data_cfg["max_len"]),
            ),
            batch_size=int(training_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers", 0)),
            collate_fn=collate_prefix,
            pin_memory=device.type == "cuda",
        )
        metrics = evaluate_bert4rec_family(
            model=model,
            loader=loader,
            device=device,
            num_items=len(item_map),
            ranking=str(data_cfg["ranking"]),
            eval_negatives=int(data_cfg["eval_negatives"]),
            ks=tuple(int(k) for k in evaluation_cfg.get("ks", [10, 20])),
            tie_policy=str(evaluation_cfg.get("tie_policy", "worst")),
            seed=seed
            + int(evaluation_cfg.get("target_seed_offset", 20000))
            + target_index,
            max_batches=int(evaluation_cfg.get("max_batches", 0)),
        )

        row: dict[str, object] = {
            "run_name": checkpoint["run_name"],
            "mode": checkpoint["mode"],
            "architecture": hparams["architecture"],
            "source": source,
            "target": target,
            "seed": seed,
            "source_best_epoch": checkpoint["best_epoch"],
            "source_selection_metric": checkpoint["best_metric_name"],
            "source_selection_value": checkpoint["best_metric_value"],
            "ranking": data_cfg["ranking"],
            "eval_negatives": int(data_cfg["eval_negatives"]),
            "target_interactions_used_for_training": False,
            "target_interactions_used_for_model_selection": False,
        }
        for key in (
            "Recall@10",
            "Recall@20",
            "NDCG@10",
            "NDCG@20",
            "MRR@10",
            "MRR@20",
            "all_equal_ratio",
            "tie_case_ratio",
            "avg_tie_items",
            "mean_rank_1based",
        ):
            if key in metrics:
                row[key] = metrics[key]

        append_csv(result_dir / "bert4rec_family_zero_shot.csv", row)
        all_results.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    output_path = (
        result_dir / f"{checkpoint['run_name']}_{source}_seed{seed}_zero_shot.json"
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
