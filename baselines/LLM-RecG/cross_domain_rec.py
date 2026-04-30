import argparse
import logging
import os
import random
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from rec_datasets import AmazonUserSequencesDataset, SteamDataset
from utils import load_pretrained_embeddings, resolve_embedding_path
from models.model_trainer import train_model, evaluate_model_with_neg_sampling
from models.sasrec_sem import SASRec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

ALL_DOMAINS = [
    "amazon_industrial_and_scientific",
    "amazon_musical_instruments",
    "amazon_video_games",
    "steam",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_units", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--dataset_name", type=str, default="amazon_industrial_and_scientific")
    parser.add_argument("--model_path", type=str, default="./saved_ckpts/")
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--train_num_negatives", type=int, default=5)
    parser.add_argument("--eval_num_negatives", type=int, default=100)
    parser.add_argument("--force_training", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_embeddings(dataset_name):
    parquet_path = resolve_embedding_path(dataset_name)
    return load_pretrained_embeddings(parquet_path)


def initialize_dataset(dataset_name, max_len):
    data_path = f"./data/{dataset_name}/processed_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing processed data at {data_path}")

    data = pd.read_csv(data_path)

    if "amazon" in dataset_name.lower():
        return AmazonUserSequencesDataset(
            data=data,
            max_seq_length=max_len,
            dataset_name=dataset_name,
        )

    return SteamDataset(
        data=data[["UserId", "ItemId", "Timestamp"]],
        max_seq_length=max_len,
        dataset_name=dataset_name,
    )


def validate_dataset_embedding_alignment(dataset_name, dataset, embeddings):
    num_items = dataset.get_num_items()
    expected_rows = num_items + 1
    actual_rows = embeddings.size(0)

    if actual_rows != expected_rows:
        raise ValueError(
            f"[{dataset_name}] Embedding rows ({actual_rows}) != num_items + 1 ({expected_rows})."
        )

    logger.info(
        f"[{dataset_name}] Alignment OK: dataset num_items={num_items}, "
        f"embedding rows={actual_rows} (including padding row 0)"
    )


def summarize_eval_result(result_dict: Dict[str, Dict[str, float]]) -> str:
    ordered_keys = [k for k in result_dict.keys() if k != "avg"] + ["avg"]
    chunks = []
    for key in ordered_keys:
        values = result_dict[key]
        chunks.append(f"{key}: R@10={values['R10']:.4f}, N@10={values['N10']:.4f}")
    return " | ".join(chunks)


def build_zero_shot_eval_fn(
    args,
    device,
    source_embeddings: torch.Tensor,
):
    """
    Return an eval_fn(model, target_domains) closure that:
    1) runs zero-shot eval on all target domains after each epoch
    2) computes avg R@10 / avg N@10
    3) restores source embeddings after evaluation
    """

    def zero_shot_eval_fn(model, target_domains: List[str]) -> Dict[str, Dict[str, float]]:
        model.eval()
        results: Dict[str, Dict[str, float]] = {}

        with torch.no_grad():
            for target_name in target_domains:
                logger.info(f"[ZeroShot] Evaluating target domain: {target_name}")

                target_dataset = initialize_dataset(target_name, args.max_len)
                target_embs = load_embeddings(target_name)
                validate_dataset_embedding_alignment(target_name, target_dataset, target_embs)
                target_num_items = target_dataset.get_num_items()

                model.load_new_pretrain_embeddings(target_embs)

                target_loader = DataLoader(
                    target_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                )

                recall_sum, ndcg_sum, total = evaluate_model_with_neg_sampling(
                    model=model,
                    dataloader=target_loader,
                    top_k_set=[5, 10, 20],
                    num_items=target_num_items,
                    device=device,
                    num_negatives=args.eval_num_negatives,
                    is_target_domain=True,
                )

                r10 = (recall_sum[10] / max(total, 1)) * 100.0
                n10 = (ndcg_sum[10] / max(total, 1)) * 100.0
                results[target_name] = {"R10": r10, "N10": n10}

        model.load_new_pretrain_embeddings(source_embeddings)

        avg_r10 = float(np.mean([v["R10"] for v in results.values()])) if results else 0.0
        avg_n10 = float(np.mean([v["N10"] for v in results.values()])) if results else 0.0
        results["avg"] = {"R10": avg_r10, "N10": avg_n10}

        logger.info(f"[ZeroShot Summary] {summarize_eval_result(results)}")
        return results

    return zero_shot_eval_fn


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Random seed: {args.seed}")

    os.makedirs(args.model_path, exist_ok=True)

    # 1) Source domain data
    source_dataset = initialize_dataset(args.dataset_name, args.max_len)
    source_embeddings = load_embeddings(args.dataset_name)
    validate_dataset_embedding_alignment(args.dataset_name, source_dataset, source_embeddings)
    num_items = source_dataset.get_num_items()

    train_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False)

    model = SASRec(
        hidden_units=args.hidden_units,
        max_seq_length=args.max_len,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        pretrained_item_embeddings=source_embeddings
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model_save_path = os.path.join(args.model_path, f"sasrec_{args.dataset_name}_sem.pth")
    target_domains = [d for d in ALL_DOMAINS if d != args.dataset_name]

    zero_shot_eval_fn = build_zero_shot_eval_fn(
        args=args,
        device=device,
        source_embeddings=source_embeddings,
    )

    logger.info(f"Source domain: {args.dataset_name}")
    logger.info(f"Target domains: {target_domains}")

    # 2) Train or load model
    if os.path.exists(model_save_path) and not args.force_training:
        logger.info(f"Loading checkpoint: {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        logger.info("Training SASRec-Sem on the source domain...")
        train_model(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            num_items=num_items,
            early_stop_patience=args.early_stop_patience,
            model_save_path=model_save_path,
            device=device,
            train_num_negatives=args.train_num_negatives,
        )
        model.load_state_dict(torch.load(model_save_path, map_location=device))

    # 3) In-domain evaluation
    logger.info("\n" + "=" * 60)
    logger.info("[EVAL] IN-DOMAIN")
    logger.info("=" * 60)
    evaluate_model_with_neg_sampling(
        model=model,
        dataloader=eval_loader,
        top_k_set=[5, 10, 20],
        num_items=num_items,
        device=device,
        num_negatives=args.eval_num_negatives,
        is_target_domain=False,
    )

    # 4) Final zero-shot evaluation
    logger.info("\n" + "=" * 60)
    logger.info("[EVAL] ZERO-SHOT TRANSFER")
    logger.info("=" * 60)
    final_result = zero_shot_eval_fn(model, target_domains)
    logger.info(f"[FINAL] {summarize_eval_result(final_result)}")


if __name__ == "__main__":
    main()
