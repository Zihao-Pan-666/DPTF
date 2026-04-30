import logging
import time
from collections import Counter
import random

import numpy as np
import torch
from tqdm import tqdm

from models.loss_func import alignment_loss_with_sampled_entropy

logger = logging.getLogger(__name__)


def bpr_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-10))


def recall_at_k(pred_items, ground_truth, k):
    return torch.tensor(ground_truth in pred_items[:k], dtype=torch.float32).item()


def ndcg_at_k(pred_items, ground_truth, k):
    if ground_truth in pred_items[:k]:
        rank = pred_items[:k].index(ground_truth) + 1
        return float(1.0 / np.log2(rank + 1))
    return 0.0


def _sample_training_negatives_fast(
    train_seq: torch.Tensor,
    pos_items: torch.Tensor,
    num_items: int,
    num_negatives: int,
    device: torch.device,
    max_rounds: int = 16,
) -> torch.Tensor:
    """
    Fast filtered negative sampling.

    Goal:
    - keep training semantics close to the current implementation:
      negatives should avoid the user's history and the current positive item
    - avoid the extremely slow Python-side full-pool construction:
      [item for item in range(1, num_items + 1) if item not in blocked]

    Strategy:
    - sample candidate negatives directly on GPU
    - reject samples that appear in history or equal the positive item
    - resample only invalid positions

    Notes:
    - This preserves the intended 'filtered negative' semantics for training.
    - It may allow duplicate negatives within the same row, which is acceptable
      for BPR training and much faster than exact without-replacement sampling.
    """
    batch_size, seq_len = train_seq.size()

    with torch.no_grad():
        neg_items = torch.randint(
            low=1,
            high=num_items + 1,
            size=(batch_size, num_negatives),
            device=device,
        )

        # Build blocked set per row: history items + current positive item
        # blocked: [B, L+1]
        blocked = torch.cat([train_seq, pos_items.unsqueeze(1)], dim=1)

        # Rejection sampling for invalid positions
        for _ in range(max_rounds):
            # invalid if sampled negative appears anywhere in blocked
            invalid = (neg_items.unsqueeze(-1) == blocked.unsqueeze(1)).any(dim=-1)

            if not invalid.any():
                break

            resampled = torch.randint(
                low=1,
                high=num_items + 1,
                size=(int(invalid.sum().item()),),
                device=device,
            )
            neg_items[invalid] = resampled

        # Very rare fallback: if some positions are still invalid after max_rounds,
        # repair them one by one with a small local loop.
        invalid = (neg_items.unsqueeze(-1) == blocked.unsqueeze(1)).any(dim=-1)
        if invalid.any():
            invalid_indices = invalid.nonzero(as_tuple=False)
            for idx in invalid_indices:
                b, n = int(idx[0].item()), int(idx[1].item())
                blocked_set = set(int(x) for x in blocked[b].tolist() if int(x) != 0)

                while True:
                    candidate = int(torch.randint(1, num_items + 1, (1,), device=device).item())
                    if candidate not in blocked_set:
                        neg_items[b, n] = candidate
                        break

    return neg_items


def train_model(
    model,
    dataloader,
    optimizer,
    num_epochs,
    num_items,
    early_stop_patience,
    model_save_path,
    device,
    train_num_negatives: int = 5,
    target_domains=None,         # 新增：例如 ["amazon_grocery_and_gourmet_food", ...]
    eval_fn=None                 # 新增：用于每 epoch zero-shot 评估的函数
):
    """Source-domain BPR training without generalization loss."""
    best_loss = float("inf")
    best_avg_r10 = -1.0
    best_zero_shot_epoch = -1
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for batch in pbar:
            train_seq, val_item, _ = batch
            train_seq = train_seq.to(device, non_blocking=True)
            val_item = val_item.to(device, non_blocking=True)

            logits = model(train_seq, is_target_domain=False)
            neg_items = _sample_training_negatives_fast(
                train_seq=train_seq,
                pos_items=val_item,
                num_items=num_items,
                num_negatives=train_num_negatives,
                device=device,
            )

            pos_logits = logits.gather(1, val_item.unsqueeze(1)).repeat(1, train_num_negatives)
            neg_logits = logits.gather(1, neg_items)
            loss = bpr_loss(pos_logits.reshape(-1), neg_logits.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss / max(pbar.n, 1):.4f}"})

        avg_loss = total_loss / max(len(dataloader), 1)
        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}] loss={avg_loss:.4f} "
            f"time={(time.time() - t0) / 60:.1f}min"
        )

        # ==========================================================
        # 综合评估、保存与早停逻辑 (Train Loss + Zero-Shot Metric)
        # ==========================================================
        improved_metric = False
        improved_loss = False

        # 1. 评估损失 (Train Loss) - 原库逻辑
        if avg_loss < best_loss:
            best_loss = avg_loss
            improved_loss = True
            loss_save_path = model_save_path.replace(".pth", "_best_loss.pth")
            torch.save(model.state_dict(), loss_save_path)
            logger.info(f"[Train] ⭐ New best training loss={best_loss:.4f}! Saved to {loss_save_path}")

        # 2. 评估指标 (Zero-Shot Metric) - 你的最佳泛化逻辑
        # if (eval_fn is not None) and (target_domains is not None) and (len(target_domains) > 0):
        #     eval_result = eval_fn(model, target_domains)
        #     avg_r10 = eval_result["avg"]["R10"]
        #     avg_n10 = eval_result["avg"]["N10"]
        #     logger.info(f"[ZeroShot][Epoch {epoch + 1}] avg R@10={avg_r10:.4f}, avg N@10={avg_n10:.4f}")
        #
        #     if avg_r10 > best_avg_r10:
        #         best_avg_r10 = avg_r10
        #         best_zero_shot_epoch = epoch + 1
        #         improved_metric = True
        #         # 将拥有最佳泛化指标的模型保存为主模型 (供主脚本最后 load 评测使用)
        #         torch.save(model.state_dict(), model_save_path)
        #         logger.info(f"[ZeroShot] ⭐ New best avg R@10={best_avg_r10:.4f}! Saved main model to {model_save_path}")
        # Optional logging only: keep zero-shot eval for observation if needed,
        # but DO NOT use it for checkpoint selection or early stopping.
        if (eval_fn is not None) and (target_domains is not None) and (len(target_domains) > 0):
            eval_result = eval_fn(model, target_domains)
            avg_r10 = eval_result["avg"]["R10"]
            avg_n10 = eval_result["avg"]["N10"]
            logger.info(
                f"[ZeroShot][Epoch {epoch + 1}] avg R@10={avg_r10:.4f}, avg N@10={avg_n10:.4f}"
            )

        # 3. 早停逻辑：
        #    - 若提供了 zero-shot eval，则只按 best_metric 早停
        #    - 若没有 eval_fn，才退回按 train loss 早停
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"[Loss] New best loss={best_loss:.4f}, model saved to {model_save_path}")
        else:
            patience_counter += 1
            logger.info(f"[Loss] No improvement. Patience: {patience_counter}/{early_stop_patience}")

        if patience_counter >= early_stop_patience:
            logger.info("Early stopping triggered.")
            break

    # if best_zero_shot_epoch != -1:
    #     logger.info(f"[ZeroShot BEST] epoch={best_zero_shot_epoch}, avg R@10={best_avg_r10:.4f}")
    # logger.info(f"Training completed. Main model (Best Metric) saved to {model_save_path}")


def resample_embeddings(sampled_embeddings, sampled_domains, batch_size, device):
    """
    Sample a balanced auxiliary mini-batch over all available auxiliary domains.
    This is the sampled approximation mentioned after Eq. (14).
    """
    resampled_embeddings = []
    resampled_domains_list = []

    unique_domains = torch.unique(sampled_domains).tolist()
    for domain_id in unique_domains:
        mask = sampled_domains == domain_id
        domain_embeddings = sampled_embeddings[mask]
        if domain_embeddings.size(0) == 0:
            continue

        sample_size = min(batch_size, domain_embeddings.size(0))
        indices = torch.randperm(
            domain_embeddings.size(0),
            device=sampled_embeddings.device,
        )[:sample_size]
        resampled_embeddings.append(domain_embeddings[indices])
        resampled_domains_list.append(
            torch.full((sample_size,), int(domain_id), dtype=torch.long, device=device)
        )

    if not resampled_embeddings:
        return (
            torch.empty((0, sampled_embeddings.size(1)), device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    return (
        torch.cat(resampled_embeddings, dim=0),
        torch.cat(resampled_domains_list, dim=0),
    )


def train_model_with_alignment(
    model,
    dataloader,
    optimizer,
    num_epochs,
    num_items,
    num_aux_domains,
    current_domain_id,
    sampled_domains,
    sampled_embeddings,
    alpha,
    early_stop_patience,
    model_save_path,
    device,
    train_num_negatives: int = 5,
    alignment_temperature: float = 0.1,
    total_item_count=None,
    target_domains=None,  # 新增
    eval_fn=None          # 新增
):
    """
    Main RecG training.

    Paper mapping:
    - recommendation term: BPR
    - item-level generalization term: sampled entropy-based alignment
    - total objective: rec_loss + align_loss
    """
    sampled_domains = sampled_domains.to(device)
    sampled_embeddings = sampled_embeddings.to(device)

    best_loss = float("inf")
    patience_counter = 0
    best_metric = -1.0

    for epoch in range(num_epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_align_loss = 0.0
        total_intra = 0.0
        total_inter = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for batch in pbar:
            train_seq, val_item, _ = batch
            train_seq = train_seq.to(device, non_blocking=True)
            val_item = val_item.to(device, non_blocking=True)

            # (1) Recommendation loss
            neg_items = _sample_training_negatives_fast(
                train_seq=train_seq,
                pos_items=val_item,
                num_items=num_items,
                num_negatives=train_num_negatives,
                device=device,
            )

            # 1)仅获取当前 Batch 的 User Representation
            user_rep = model.encode_sequence(train_seq)  # shape: [B, D]

            # 2)仅获取正负样本的 Item Embeddings
            pos_embs = model.get_item_embeddings(val_item)  # shape: [B, D]
            neg_embs = model.get_item_embeddings(neg_items)  # shape: [B, num_neg, D]

            # 3)局部点积打分
            pos_logits = (user_rep * pos_embs).sum(dim=-1).unsqueeze(1)  # shape: [B, 1]
            pos_logits = pos_logits.repeat(1, train_num_negatives)  # shape: [B, num_neg]

            neg_logits = torch.bmm(neg_embs, user_rep.unsqueeze(2)).squeeze(2)  # shape: [B, num_neg]

            rec_loss = bpr_loss(pos_logits.reshape(-1), neg_logits.reshape(-1))

            # (2) Item-level generalization loss on the FINAL item embedding space
            source_items = torch.cat([train_seq.reshape(-1), val_item], dim=0)
            source_items = torch.unique(source_items[source_items > 0])

            if source_items.numel() > 0:
                source_projected = model.project_items_for_alignment(source_items)
                source_domains = torch.full(
                    (source_projected.size(0),),
                    current_domain_id,
                    dtype=torch.long,
                    device=device,
                )

                aux_raw, aux_doms = resample_embeddings(
                    sampled_embeddings=sampled_embeddings,
                    sampled_domains=sampled_domains,
                    batch_size=train_seq.size(0),
                    device=device,
                )

                if aux_raw.numel() > 0:
                    aux_projected = model.project_raw_for_alignment(aux_raw)
                    combined_embs = torch.cat([source_projected, aux_projected], dim=0)
                    combined_doms = torch.cat([source_domains, aux_doms], dim=0)
                    align_loss, intra_div, inter_ent, beta_val = alignment_loss_with_sampled_entropy(
                        sampled_embeddings=combined_embs,
                        sampled_domains=combined_doms,
                        num_domains=num_aux_domains,
                        alpha_base=alpha,
                        temperature=alignment_temperature,
                        total_item_count=total_item_count,
                    )
                else:
                    align_loss = torch.zeros((), device=device)
            else:
                align_loss = torch.zeros((), device=device)

            total_loss_batch = rec_loss + align_loss

            # if epoch == 0 and pbar.n < 3:
            #     logger.info(
            #         f"[DEBUG] rec={rec_loss.item():.6f}, "
            #         f"align={align_loss.item():.6f}, "
            #         f"intra={intra_div.item():.6f}, "
            #         f"inter={inter_ent.item():.6f}, "
            #         f"beta={beta_val.item():.6f}"
            #     )

            optimizer.zero_grad()
            total_loss_batch.backward()

            # 【关键修复】1. 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # 【关键修复】2. Non-finite 检查，如果本批次炸了，直接跳过不更新
            has_nan = False
            for param in model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    has_nan = True
                    break

            if has_nan:
                logger.warning("发现 NaN 梯度，跳过本 Batch 更新...")
                optimizer.zero_grad()
                continue

            optimizer.step()

            total_loss += total_loss_batch.item()
            total_rec_loss += rec_loss.item()
            total_align_loss += align_loss.item()

            total_intra += intra_div.item()
            total_inter += inter_ent.item()

            steps = max(pbar.n, 1)
            pbar.set_postfix(
                {
                    "rec": f"{total_rec_loss / steps:.4f}",
                    "align": f"{total_align_loss / steps:.6f}",
                }
            )

        avg_loss = total_loss / max(len(dataloader), 1)
        avg_rec = total_rec_loss / max(len(dataloader), 1)
        avg_align = total_align_loss / max(len(dataloader), 1)
        avg_intra = total_intra / max(len(dataloader), 1)
        avg_inter = total_inter / max(len(dataloader), 1)

        logger.info(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"loss={avg_loss:.4f} "
            f"bpr={avg_rec:.4f} "
            f"align={avg_align:.6f} "
            f"intra={avg_intra:.6f} "
            f"inter={avg_inter:.6f} "
            f"time={(time.time() - t0) / 60:.1f}min"
        )

        # 新增：每个 Epoch 结束后的 Zero-shot 跨域评估与早停逻辑
        # ==========================================================
        if eval_fn is not None and target_domains is not None:
            logger.info(f"--- Epoch {epoch + 1} Zero-shot Evaluation ---")

            # 调用闭包函数进行评估，它会自动处理 patterns 提取和全目标域测试
            results = eval_fn(model, target_domains)

            # 获取所有目标域的平均 R@10 作为保存模型的依据
            current_metric = results.get("avg", {}).get("R10", 0.0)

            if current_metric > best_metric:
                best_metric = current_metric
                patience_counter = 0
                # 保存当前最优模型
                torch.save(model.state_dict(), model_save_path)
                logger.info(f"⭐ New best zero-shot Avg R@10: {best_metric:.4f}. Model saved to {model_save_path}")
            else:
                patience_counter += 1
                logger.info(f"No improvement in Avg R@10. Patience: {patience_counter} / {early_stop_patience}")

            if patience_counter >= early_stop_patience:
                logger.info("Early stopping triggered! Training finished.")
                break
        else:
            # 如果没有传入 eval_fn，退回到只保存最后一个 epoch 或原始的保存逻辑
            torch.save(model.state_dict(), model_save_path)

    logger.info(f"Training completed. Best model saved to: {model_save_path}")


def evaluate_model_with_neg_sampling(
        model,
        dataloader,
        top_k_set,
        num_items,
        device,
        num_negatives=100,
        is_target_domain=False,
):
    model.eval()

    recall_sum = {k: 0.0 for k in top_k_set}
    ndcg_sum = {k: 0.0 for k in top_k_set}
    total = 0

    domain_label = "Target" if is_target_domain else "Source"
    pbar = tqdm(dataloader, desc=f"Eval [{domain_label}]", leave=False)

    with torch.no_grad():

        for batch in pbar:

            train_seq, val_item, test_item = batch

            train_seq = train_seq.to(device, non_blocking=True)
            val_item = val_item.to(device, non_blocking=True)
            test_item = test_item.to(device, non_blocking=True)

            batch_size = test_item.size(0)

            # 构造 evaluation sequence
            eval_seq = torch.cat([train_seq[:, 1:], val_item.unsqueeze(1)], dim=1)

            candidate_list = []

            train_seq_cpu = train_seq.cpu().numpy()
            val_item_cpu = val_item.cpu().numpy()
            test_item_cpu = test_item.cpu().numpy()

            for i in range(batch_size):

                # 构建 history
                history = set(train_seq_cpu[i])
                history.add(val_item_cpu[i])
                history.discard(0)

                # ⚠️ 注意：绝对不能加入 test_item
                # history.add(test_item_cpu[i])  <-- 删除这一行

                candidate_pool = [
                    item for item in range(1, num_items + 1)
                    if item not in history
                ]

                if len(candidate_pool) >= num_negatives:
                    negative_samples = random.sample(candidate_pool, num_negatives)
                else:
                    negative_samples = candidate_pool.copy()

                # 将真实 item 加入候选集合
                candidates = negative_samples + [int(test_item_cpu[i])]

                candidate_list.append(candidates)

            candidate_tensor = torch.tensor(
                candidate_list,
                dtype=torch.long,
                device=device
            )

            scores = model.predict(
                item_seq=eval_seq,
                candidate_items=candidate_tensor,
                is_target_domain=is_target_domain,
            )

            _, top_indices = torch.topk(scores, k=max(top_k_set), dim=-1)

            top_k_items = torch.gather(candidate_tensor, 1, top_indices)

            for i in range(batch_size):

                ground_truth = int(test_item_cpu[i])
                pred_items = top_k_items[i].tolist()

                for k in top_k_set:

                    recall_sum[k] += recall_at_k(pred_items, ground_truth, k)
                    ndcg_sum[k] += ndcg_at_k(pred_items, ground_truth, k)

            total += batch_size

    for k in top_k_set:

        r_k = (recall_sum[k] / max(total, 1)) * 100
        n_k = (ndcg_sum[k] / max(total, 1)) * 100

        logger.info(
            f"[{domain_label}] Recall@{k}: {r_k:.4f}%, NDCG@{k}: {n_k:.4f}%"
        )

    return recall_sum, ndcg_sum, total



def evaluate_model(model, dataloader, top_k_set, device):
    model.eval()
    recall_sum = {k: 0.0 for k in top_k_set}
    ndcg_sum = {k: 0.0 for k in top_k_set}
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            train_seq, val_item, test_item = batch
            train_seq = train_seq.to(device, non_blocking=True)
            val_item = val_item.to(device, non_blocking=True)
            test_item = test_item.to(device, non_blocking=True)

            eval_seq = torch.cat([train_seq, val_item.unsqueeze(1)], dim=1)
            logits = model.predict(eval_seq)
            _, top_k_items = torch.topk(logits, max(top_k_set), dim=-1)

            for i in range(test_item.size(0)):
                for k in top_k_set:
                    recall_sum[k] += recall_at_k(top_k_items[i][:k], int(test_item[i].item()), k)
                    ndcg_sum[k] += ndcg_at_k(top_k_items[i][:k], int(test_item[i].item()), k)
            total += test_item.size(0)

    for k in top_k_set:
        avg_recall = recall_sum[k] / max(total, 1) * 100
        avg_ndcg = ndcg_sum[k] / max(total, 1) * 100
        logger.info(f"Test Recall@{k}: {avg_recall:.4f}%, NDCG@{k}: {avg_ndcg:.4f}%")

    return recall_sum, ndcg_sum, total


def fit_most_pop(model, dataloader):
    logger.info("Fitting MostPop model...")
    item_counts = Counter()
    for batch in dataloader:
        train_seq, _, _ = batch
        item_counts.update(int(x) for x in train_seq.flatten().tolist() if int(x) != 0)

    model.popularity = torch.zeros(model.num_items + 1)
    for item, count in item_counts.items():
        model.popularity[item] = count

    if model.popularity.sum() > 0:
        model.popularity /= model.popularity.sum()

    logger.info("MostPop model fitted.")


def evaluate_most_pop(model, dataloader, top_k_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    recall_sum = {k: 0 for k in top_k_list}
    ndcg_sum = {k: 0 for k in top_k_list}
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            train_seq, _, test_item = batch
            train_seq = train_seq.to(device)
            test_item = test_item.to(device)

            predictions = model.predict(train_seq)
            top_k_predictions = predictions.argsort(dim=-1, descending=True)

            for i in range(train_seq.size(0)):
                ground_truth = int(test_item[i].item())
                pred_items = top_k_predictions[i].tolist()
                for k in top_k_list:
                    recall_sum[k] += recall_at_k(pred_items, ground_truth, k)
                    ndcg_sum[k] += ndcg_at_k(pred_items, ground_truth, k)
            total += train_seq.size(0)

    for k in top_k_list:
        recall = recall_sum[k] / max(total, 1) * 100
        ndcg = ndcg_sum[k] / max(total, 1)
        logger.info(f"Recall@{k}: {recall:.4f}%, NDCG@{k}: {ndcg:.4f}")

    return recall_sum, ndcg_sum
