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

from rec_datasets import AmazonUserSequencesDataset, SteamDataset, AuxiliarySemanticSampler
from utils import load_pretrained_embeddings, resolve_embedding_path
from models.model_trainer import evaluate_model_with_neg_sampling, train_sagerec_model, train_model

from baselines.sasrec import SASRecWithDomainAlignment
from baselines.bert4rec import BERT4RecWithDomainAlignment
from baselines.unisrec import UniSRecWithDomainAlignment
from baselines.gru4rec import GRU4RecWithDomainAlignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

ALL_DOMAINS = ["amazon_movies_and_tv", "amazon_cds_and_vinyl", "steam"]


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
    parser.add_argument("--model_name", type=str, default="bert4rec", help="sasrec or bert4rec or gru4rec")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    parser.add_argument("--dataset_name", type=str, default="amazon_movies_and_tv")
    parser.add_argument("--model_path", type=str, default="./saved_ckpts/")

    # 【关键修改】：严格限定支持此 6 种模式，移出老版无用的模型
    parser.add_argument("--loss_mode", type=str, default="sage",
                        choices=["sem", "recg", "sage", "sage_no_sa", "sage_no_id", "sage_no_adagen"])

    # 早停与评测控制
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_criterion", type=str, default="zero_shot", choices=["loss", "zero_shot"])
    parser.add_argument("--check_step", type=int, default=3, help="Evaluate every N epochs")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Skip eval for first N epochs")

    # 强制评测某一个 checkpoints 的特权命令
    parser.add_argument("--eval_only", action="store_true", help="Skip training and only run evaluation")
    parser.add_argument("--eval_checkpoint", type=str, default="", help="Path to specific checkpoint to evaluate")

    parser.add_argument("--n_exps", type=int, default=8, help="Number of MoE experts for UniSRec")
    parser.add_argument("--force_training", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--num_sequential_patterns", type=int, default=20) # 大数据集专用档位
    parser.add_argument("--train_num_negatives", type=int, default=5)
    parser.add_argument("--eval_num_negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)

    # sagerec_main.py 中的 parse_args 修改项
    parser.add_argument("--lambda_g", type=float, default=0.05, help="Base regularization strength")
    parser.add_argument("--gamma_g", type=float, default=0.1, help="Decay rate for domain discrepancy")
    parser.add_argument("--beta_id", type=float, default=0.1, help="Weight for intra-domain diversity")
    parser.add_argument("--tau", type=float, default=0.2, help="Temperature for diversity entropy")

    return parser.parse_args()


def load_embeddings(dataset_name):
    return load_pretrained_embeddings(resolve_embedding_path(dataset_name))


def initialize_dataset(dataset_name, max_len):
    data_path = f"./data/{dataset_name}/processed_data.csv"
    data = pd.read_csv(data_path)
    if "amazon" in dataset_name.lower():
        return AmazonUserSequencesDataset(data=data, max_seq_length=max_len, dataset_name=dataset_name)
    return SteamDataset(data=data[["UserId", "ItemId", "Timestamp"]], max_seq_length=max_len, dataset_name=dataset_name)


def sample_auxiliary_embeddings(device, source_domain, num_samples=4096):
    aux_names = [d for d in ALL_DOMAINS if d != source_domain]
    sampled_embs, sampled_ids = [], []
    for name in aux_names:
        domain_idx = ALL_DOMAINS.index(name)
        embs = load_embeddings(name)[1:]
        indices = torch.randperm(embs.size(0))[:min(num_samples, embs.size(0))]
        sampled_embs.append(embs[indices])
        sampled_ids.append(torch.full((len(indices),), domain_idx, dtype=torch.long))
    return (torch.cat(sampled_embs, dim=0).to(device), torch.cat(sampled_ids, dim=0).to(device), aux_names)


def collect_source_train_sequences(dataset, device):
    train_sequences = [dataset[idx][0].unsqueeze(0) for idx in range(len(dataset))]
    return torch.cat(train_sequences, dim=0).to(device)


def summarize_eval_result(result_dict: Dict[str, Dict[str, float]]) -> str:
    ordered_keys = [k for k in result_dict.keys() if k != "avg"] + ["avg"]
    return " | ".join(
        [f"{key}: R@10={result_dict[key]['R10']:.4f}, N@10={result_dict[key]['N10']:.4f}" for key in ordered_keys])


def build_zero_shot_eval_fn(args, device, source_embeddings: torch.Tensor, source_train_sequences: torch.Tensor):
    def zero_shot_eval_fn(model, target_domains: List[str]) -> Dict[str, Dict[str, float]]:
        model.eval()
        # 纯 sem 模式下，切断未训练序列特征的污染
        if args.loss_mode != "sem":
            model.extract_sequential_patterns_from_source(source_train_sequences)
        results = {}
        with torch.no_grad():
            for target_name in target_domains:
                target_dataset = initialize_dataset(target_name, args.max_len)
                target_embs = load_embeddings(target_name)
                model.load_new_pretrain_embeddings(target_embs)
                target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)

                # 【修改】支持 10 和 20
                recall_sum, ndcg_sum, total = evaluate_model_with_neg_sampling(
                    model=model, dataloader=target_loader, top_k_set=[10, 20],
                    num_items=target_dataset.get_num_items(),
                    device=device, num_negatives=args.eval_num_negatives, is_target_domain=True,
                    is_sem_baseline=args.loss_mode == "sem"
                )
                results[target_name] = {
                    "R10": (recall_sum[10] / max(total, 1)) * 100.0,
                    "N10": (ndcg_sum[10] / max(total, 1)) * 100.0,
                    "R20": (recall_sum[20] / max(total, 1)) * 100.0,
                    "N20": (ndcg_sum[20] / max(total, 1)) * 100.0
                }

        model.load_new_pretrain_embeddings(source_embeddings)
        results["avg"] = {
            "R10": float(np.mean([v["R10"] for v in results.values()])) if results else 0.0,
            "N10": float(np.mean([v["N10"] for v in results.values()])) if results else 0.0,
            "R20": float(np.mean([v["R20"] for v in results.values()])) if results else 0.0,
            "N20": float(np.mean([v["N20"] for v in results.values()])) if results else 0.0
        }
        return results

    return zero_shot_eval_fn


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.model_path, exist_ok=True)

    source_dataset = initialize_dataset(args.dataset_name, args.max_len)
    source_embeddings = load_embeddings(args.dataset_name)
    num_items = source_dataset.get_num_items()

    train_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False)

    model_name_lower = args.model_name.lower()
    if model_name_lower == "sasrec":
        model = SASRecWithDomainAlignment(hidden_units=args.hidden_units, max_seq_length=args.max_len,
                                          num_heads=args.num_heads, num_layers=args.num_layers,
                                          dropout_rate=args.dropout_rate, pretrained_item_embeddings=source_embeddings,
                                          num_sequential_patterns=args.num_sequential_patterns).to(device)
    elif model_name_lower == "bert4rec":
        model = BERT4RecWithDomainAlignment(hidden_units=args.hidden_units, max_seq_length=args.max_len,
                                            num_heads=args.num_heads, num_layers=args.num_layers,
                                            dropout_rate=args.dropout_rate,
                                            pretrained_item_embeddings=source_embeddings,
                                            num_sequential_patterns=args.num_sequential_patterns).to(device)

    elif model_name_lower == "gru4rec":  # 【新增分支】
        model = GRU4RecWithDomainAlignment(
            hidden_units=args.hidden_units,
            num_layers=args.num_layers,
            dropout_rate=args.dropout_rate,
            pretrained_item_embeddings=source_embeddings,
            num_sequential_patterns=args.num_sequential_patterns,
            max_seq_length=args.max_len
        ).to(device)

    elif model_name_lower == "unisrec":
        model = UniSRecWithDomainAlignment(hidden_units=args.hidden_units, max_seq_length=args.max_len,
                                           num_heads=args.num_heads, num_layers=args.num_layers,
                                           dropout_rate=args.dropout_rate, pretrained_item_embeddings=source_embeddings,
                                           num_sequential_patterns=args.num_sequential_patterns, n_exps=args.n_exps).to(
            device)
    else:
        raise ValueError(f"Unsupported model name: {args.model_name}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model_save_path = os.path.join(
        args.model_path,
        f"{args.model_name.lower()}_{args.dataset_name}_{args.loss_mode}.pth"
    )

    # ===【新代码：初始化动态采样器】===
    current_domain_id = ALL_DOMAINS.index(args.dataset_name)
    aux_names = [d for d in ALL_DOMAINS if d != args.dataset_name]

    logger.info("Initializing AuxiliarySemanticSampler for dynamic global sampling...")
    aux_sampler = AuxiliarySemanticSampler(
        model=model,
        aux_domains=aux_names,
        all_domains_for_index=ALL_DOMAINS,  # 传入全局列表
        data_root="./data",  # 确保这里是你存放 parquet 的根目录
        aux_batch_size=args.num_samples,
        sample_mode="domain_uniform",
        verbose=True
    )

    source_train_sequences = collect_source_train_sequences(source_dataset, device)

    # === 执行训练 ===
    saved_paths = {"loss": model_save_path.replace(".pth", "_loss.pth")}
    for target in aux_names:
        saved_paths[f"test_on_{target}"] = model_save_path.replace(".pth", f"_test_on_{target}.pth")

    if args.eval_only:
        logger.info(f"Skipping training because --eval_only is set.")
    elif os.path.exists(saved_paths["loss"]) and not args.force_training:
        logger.info("Found existing checkpoints. Skipping training.")
    else:
        zero_shot_eval_fn = build_zero_shot_eval_fn(args, device, source_embeddings, source_train_sequences)
        saved_paths = train_sagerec_model(
            model=model, dataloader=train_loader, optimizer=optimizer, num_epochs=args.num_epochs,
            num_items=num_items, num_aux_domains=len(ALL_DOMAINS), current_domain_id=current_domain_id,
            aux_sampler=aux_sampler,  # 【关键】：这里将字典与Tensor换为了Sampler对象
            alpha_base=args.alpha,
            early_stop_patience=args.early_stop_patience, model_save_path=model_save_path,
            device=device, train_num_negatives=args.train_num_negatives, temperature=0.1,
            target_domains=aux_names, eval_fn=zero_shot_eval_fn,
            check_step=args.check_step, warmup_epochs=args.warmup_epochs,
            early_stop_criterion=args.early_stop_criterion,
            loss_mode = args.loss_mode,
            args=args
        )

    # === 执行独立评估阶段 ===
    logger.info("\n" + "=" * 60)
    logger.info("[EVAL] IN-DOMAIN (Based on best Training Loss)")
    logger.info("=" * 60)

    loss_ckpt = args.eval_checkpoint if args.eval_checkpoint else saved_paths["loss"]
    if os.path.exists(loss_ckpt):
        model.load_state_dict(torch.load(loss_ckpt, map_location=device))
        evaluate_model_with_neg_sampling(
            model=model, dataloader=eval_loader, top_k_set=[10, 20], num_items=num_items,
            device=device, num_negatives=args.eval_num_negatives, is_target_domain=False,
            is_sem_baseline=args.loss_mode == "sem"
        )
    else:
        logger.warning(f"Checkpoint not found for In-Domain eval: {loss_ckpt}")

    logger.info("\n" + "=" * 60)
    logger.info("[EVAL] ZERO-SHOT TRANSFER")
    logger.info("=" * 60)
    final_results = {}
    for target_name in aux_names:
        model.load_new_pretrain_embeddings(source_embeddings)

        if args.early_stop_criterion == "loss":
            target_ckpt = args.eval_checkpoint if args.eval_checkpoint else saved_paths["loss"]
            logger.info(f"\n>>> Testing on: {target_name} (Using best LOSS model)")
        else:
            target_ckpt = args.eval_checkpoint if args.eval_checkpoint else saved_paths.get(f"test_on_{target_name}")
            logger.info(f"\n>>> Strictly testing on: {target_name} (Model selected via other domains)")

        if os.path.exists(target_ckpt):
            model.load_state_dict(torch.load(target_ckpt, map_location=device))
        else:
            logger.warning(f"Checkpoint not found for Zero-Shot eval: {target_ckpt}")
            continue

        # 纯 sem 模式下，切断未训练序列特征的污染
        if args.loss_mode != "sem":
            model.extract_sequential_patterns_from_source(source_train_sequences)

        target_dataset = initialize_dataset(target_name, args.max_len)
        target_embs = load_embeddings(target_name)
        model.load_new_pretrain_embeddings(target_embs)

        target_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False)
        recall_sum, ndcg_sum, total = evaluate_model_with_neg_sampling(
            model=model, dataloader=target_loader, top_k_set=[10, 20], num_items=target_dataset.get_num_items(),
            device=device, num_negatives=args.eval_num_negatives, is_target_domain=True,
            is_sem_baseline=args.loss_mode == "sem"
        )
        final_results[target_name] = {"R10": (recall_sum[10] / max(total, 1)) * 100.0,
                                      "N10": (ndcg_sum[10] / max(total, 1)) * 100.0}

    final_results["avg"] = {
        "R10": float(np.mean([v["R10"] for v in final_results.values()])) if final_results else 0.0,
        "N10": float(np.mean([v["N10"] for v in final_results.values()])) if final_results else 0.0
    }
    logger.info(f"[FINAL REPORT] {summarize_eval_result(final_results)}")


if __name__ == "__main__":
    main()