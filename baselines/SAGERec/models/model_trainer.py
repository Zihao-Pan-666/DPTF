import math
import time
import random
import logging
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.loss import unified_alignment_loss

logger = logging.getLogger(__name__)


def bpr_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    """经典 BPR Loss 计算"""
    return -torch.mean(torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-10))


def recall_at_k(pred_items, ground_truth, k):
    return torch.tensor(ground_truth in pred_items[:k], dtype=torch.float32).item()


def ndcg_at_k(pred_items, ground_truth, k):
    if ground_truth in pred_items[:k]:
        rank = pred_items[:k].index(ground_truth) + 1
        return float(1.0 / np.log2(rank + 1))
    return 0.0


def _sample_training_negatives_fast(
        train_seq: torch.Tensor, pos_items: torch.Tensor, num_items: int,
        num_negatives: int, device: torch.device, max_rounds: int = 16,
) -> torch.Tensor:
    """高效且严谨的负采样：确保负样本不在历史记录中"""
    batch_size, seq_len = train_seq.size()
    with torch.no_grad():
        neg_items = torch.randint(1, num_items + 1, size=(batch_size, num_negatives), device=device)
        blocked = torch.cat([train_seq, pos_items.unsqueeze(1)], dim=1)
        for _ in range(max_rounds):
            invalid = (neg_items.unsqueeze(-1) == blocked.unsqueeze(1)).any(dim=-1)
            if not invalid.any(): break
            resampled = torch.randint(1, num_items + 1, size=(int(invalid.sum().item()),), device=device)
            neg_items[invalid] = resampled

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


# ==========================================================
# 标准推荐模型训练
# ==========================================================
def train_model(model, train_loader, optimizer, device, num_epochs, eval_data=None,
                patience=5, save_path=None, num_neg_samples=99, metrics_at_k=[5, 10, 20]):
    best_recall10 = 0.0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            seq = batch['seq'].to(device)
            pos = batch['pos'].to(device)
            neg = batch['neg'].to(device)

            optimizer.zero_grad()
            pos_logits, neg_logits = model(seq, pos, neg)
            loss = bpr_loss(pos_logits, neg_logits)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        elapsed = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        if eval_data is not None:
            metrics = evaluate_model_with_neg_sampling(
                model, eval_data, top_k_set=metrics_at_k, num_items=model.num_items, device=device,
                num_negatives=num_neg_samples
            )
            recall10 = (metrics[0].get(10, 0.0) / metrics[2]) * 100
            print(f"  Eval -> R@10={recall10:.4f}")

            if recall10 > best_recall10:
                best_recall10 = recall10
                patience_counter = 0
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    print(f"  Model saved (R@10={best_recall10:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

    return model


# ==========================================================
# 核心：SAGERec 对齐训练器 (支持严谨负采样与动态跨域抽样)
# ==========================================================
def train_sagerec_model(
        model, dataloader, optimizer, num_epochs,
        num_items, num_aux_domains, current_domain_id,
        aux_sampler, alpha_base, early_stop_patience,
        model_save_path, device, train_num_negatives, temperature,
        target_domains, eval_fn, check_step, warmup_epochs, early_stop_criterion,
        loss_mode,args
):
    best_loss = float('inf')
    best_avg_r10 = -1.0
    patience_counter = 0

    # 自动创建多个保存路径字典，用于完美还原“留一法”验证
    saved_paths = {"loss": model_save_path.replace(".pth", "_loss.pth")}
    best_leave_one_out = {}
    if target_domains:
        for target in target_domains:
            saved_paths[f"test_on_{target}"] = model_save_path.replace(".pth", f"_test_on_{target}.pth")
            best_leave_one_out[target] = -1.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = total_bpr = total_gen = total_sic = total_id = total_omega = 0.0
        start_time = time.time()

        with tqdm(dataloader, desc=f"Epoch {epoch:02d}/{num_epochs}", leave=False, dynamic_ncols=True) as pbar:
            for step, batch in enumerate(pbar, start=1):
                if isinstance(batch, (tuple, list)):
                    seq, pos, _ = batch
                else:
                    seq, pos = batch['seq'], batch['pos']

                seq, pos = seq.to(device, non_blocking=True), pos.to(device, non_blocking=True)

                # 1. 严格过滤历史记录的负采样
                neg_items = _sample_training_negatives_fast(seq, pos, num_items, train_num_negatives, device)

                # 判断当前是否为纯 SEM 基线模式
                is_sem = (loss_mode == "sem")

                # 2. BPR 推荐损失 (传入 is_sem_baseline)
                logits = model(seq, is_target_domain=False, is_sem_baseline=is_sem)
                pos_logits = logits.gather(1, pos.unsqueeze(1)).repeat(1, train_num_negatives)
                neg_logits = logits.gather(1, neg_items)
                rec_loss = bpr_loss(pos_logits.reshape(-1), neg_logits.reshape(-1))

                # 3. 动态提取跨域辅助特征 (限制为 batch_size，防止计算灾难)
                aux_data = aux_sampler.sample(device)
                batch_size = seq.size(0)
                sampled_embeddings = aux_data["aux_raw"][:batch_size]
                sampled_domains = aux_data["aux_domain_ids"][:batch_size]

                # 【核心修复】：还原对“整个历史序列”的提取，找回真实的 Alpha 尺度！
                source_items = torch.unique(torch.cat([seq.reshape(-1), pos], dim=0))
                source_item_ids = source_items[source_items > 0]  # 过滤掉 padding 0

                if not is_sem and source_item_ids.numel() > 0 and sampled_embeddings.numel() > 0:
                    # 只有在非 sem 模式才进行投影和对齐计算
                    source_projected = model.project_items_for_alignment(source_item_ids)
                    source_sem_raw = model.get_raw_item_embeddings(source_item_ids)
                    source_domain_labels = torch.full((source_item_ids.size(0),), current_domain_id, dtype=torch.long,
                                                      device=device)

                    all_projected = torch.cat([source_projected, model.project_raw_for_alignment(sampled_embeddings)],
                                              dim=0)
                    all_raw = torch.cat([source_sem_raw, sampled_embeddings], dim=0)
                    all_domains = torch.cat([source_domain_labels, sampled_domains], dim=0)

                    gen_loss, sic_val, id_val, omega_val = unified_alignment_loss(
                        mode=args.loss_mode,
                        projected_embeddings=all_projected,
                        raw_embeddings=all_raw,
                        domain_labels=all_domains,
                        num_domains=num_aux_domains,
                        source_domain_id=current_domain_id,
                        lambda_g=args.lambda_g,
                        gamma_g=args.gamma_g,
                        beta_id=args.beta_id,
                        tau=args.tau
                    )
                else:
                    # 【关键修复】：将 0.0 改为 tensor(0.0)，防止后续 .item() 报错
                    gen_loss, sic_val, id_val, omega_val = (
                        torch.tensor(0.0, device=device, requires_grad=True),
                        torch.tensor(0.0, device=device),
                        torch.tensor(0.0, device=device),
                        args.lambda_g
                    )

                loss = rec_loss + omega_val * gen_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += loss.item()
                total_bpr += rec_loss.item()
                total_gen += gen_loss.item()
                total_sic += sic_val.item()
                total_id += id_val.item()
                total_omega += omega_val
                pbar.set_postfix({
                    'Loss': f"{total_loss/step:.4f}",
                    'BPR': f"{total_bpr/step:.4f}",
                    'Gen':f"{total_gen/step:.4f}",
                    'SIC': f"{total_sic/step:.4f}",
                    'ID': f"{total_id/step:.4f}",
                    'Omega': f"{total_omega/step:.6f}"
                })

        elapsed = time.time() - start_time
        steps = len(dataloader)
        avg_loss = total_loss / steps
        tqdm.write(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | BPR: {total_bpr / steps:.4f} | "
                   f"Gen: {total_gen / steps:.4f} | SIC: {total_sic / steps:.4f} | "
                   f"ID: {total_id/step:.4f} | Omega: {total_omega/step:.5f} | Time: {elapsed:.1f}s")

        # if epoch % 25 == 0:
        #     periodic_path = model_save_path.replace(".pth", f"_{epoch}.pth")
        #     torch.save(model.state_dict(), periodic_path)
        #     tqdm.write(f"  [Checkpoint] Periodic save at Epoch {epoch}: {periodic_path}")

        # ==================== 统一的保存与早停逻辑 ====================
        improved = False
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), saved_paths["loss"])
            improved = True
            if early_stop_criterion == 'loss':
                patience_counter = 0  # 只有真正创新低时才重置
                # tqdm.write(f"  [Save] Loss 创新低: {best_loss:.4f}, 重置早停计数。")

        # Zero-Shot 准则下的早停
        if early_stop_criterion == 'zero_shot' and epoch >= warmup_epochs and (
                epoch % check_step == 0 or epoch == num_epochs):
            tqdm.write("  --> 正在运行留一法 Zero-Shot 评估以验证早停...")
            eval_res = eval_fn(model, target_domains)
            avg_r10 = eval_res.get("avg", {}).get("R10", 0.0)

            # 更新留一法最优模型 (逻辑保持不变)
            for test_target in target_domains:
                val_domains = [d for d in target_domains if d != test_target]
                val_score = sum(eval_res[d]["R10"] for d in val_domains) / len(val_domains) if val_domains else \
                eval_res[test_target]["R10"]
                if val_score > best_leave_one_out[test_target]:
                    best_leave_one_out[test_target] = val_score
                    torch.save(model.state_dict(), saved_paths[f"test_on_{test_target}"])

            if avg_r10 > best_avg_r10:
                best_avg_r10 = avg_r10
                patience_counter = 0
            else:
                patience_counter += 1

        # Loss 准则下的早停 (修复后的计数逻辑)
        elif early_stop_criterion == 'loss':
            if not improved:
                patience_counter += 1
        # ==============================================================

        if patience_counter >= early_stop_patience:
            tqdm.write(f"\n[Early Stop] 已触发早停，结束于 Epoch {epoch}。")
            break

    return saved_paths


def evaluate_model_with_neg_sampling(model, dataloader, top_k_set, num_items, device, num_negatives=100,
                                     is_target_domain=False, is_sem_baseline=False):
    # 与之前相同，不需要改动
    model.eval()
    recall_sum = {k: 0.0 for k in top_k_set}
    ndcg_sum = {k: 0.0 for k in top_k_set}
    total = 0
    domain_label = "Target" if is_target_domain else "Source"
    pbar = tqdm(dataloader, desc=f"Eval [{domain_label}]", leave=False)

    # 【优化点 1】：在循环外预计算全量物品的 Embedding，避免重复计算
    with torch.no_grad():
        all_item_embs = model.projection_layer(model.pretrained_item_embedding.weight)  # [N, H]

        for batch in pbar:
            train_seq, val_item, test_item = batch
            train_seq, val_item, test_item = train_seq.to(device), val_item.to(device), test_item.to(device)
            batch_size = test_item.size(0)

            # 准备评估序列：[历史记录 + 验证集] -> 预测 [测试集]
            eval_seq = torch.cat([train_seq[:, 1:], val_item.unsqueeze(1)], dim=1)

            # # === 【新增调试打印：仅在第一个 Batch 打印一次】 ===
            # if total == 0:
            #     logger.info(f"DEBUG - eval_seq shape: {eval_seq.shape}")  # 预期应为 [128, 50]
            #     logger.info(f"DEBUG - 第一个样本的 eval_seq 内容: \n{eval_seq[0].cpu().numpy()}")
            #     logger.info(f"DEBUG - 验证集物品 (应为序列最后一位): {val_item[0].item()}")
            #     logger.info(f"DEBUG - 测试集物品 (Ground Truth): {test_item[0].item()}")
            # # =================================================

            # 【优化点 2】：高速负采样，不再使用列表推导式
            candidate_list = []
            test_item_np = test_item.cpu().numpy()
            train_val_history = torch.cat([train_seq, val_item.unsqueeze(1)], dim=1).cpu().numpy()

            for i in range(batch_size):
                history_set = set(train_val_history[i])
                history_set.discard(0)

                # 随机快速抽样
                neg_samples = []
                while len(neg_samples) < num_negatives:
                    sample = random.randint(1, num_items)
                    if sample not in history_set:
                        neg_samples.append(sample)
                candidate_list.append(neg_samples + [int(test_item_np[i])])

            candidate_tensor = torch.tensor(candidate_list, dtype=torch.long, device=device)  # [B, 101]

            # 获取用户表征
            # 注意：此处需微调模型代码，或直接调用模型内部编码逻辑
            user_rep = model.encode_sequence(eval_seq, is_sem_baseline=is_sem_baseline)

            # 【优化点 3】：仅对候选物品进行打分（矩阵点乘优化）
            # candidate_embs: [B, 101, H]
            candidate_embs = all_item_embs[candidate_tensor]
            # scores: [B, 101]
            scores = (user_rep.unsqueeze(1) * candidate_embs).sum(dim=-1)

            _, top_indices = torch.topk(scores, k=max(top_k_set), dim=-1)
            top_k_items = torch.gather(candidate_tensor, 1, top_indices)

            for i in range(batch_size):
                ground_truth = int(test_item_np[i])
                pred_items = top_k_items[i].tolist()
                for k in top_k_set:
                    recall_sum[k] += recall_at_k(pred_items, ground_truth, k)
                    ndcg_sum[k] += ndcg_at_k(pred_items, ground_truth, k)
            total += batch_size

    for k in top_k_set:
        r_k = (recall_sum[k] / max(total, 1)) * 100
        n_k = (ndcg_sum[k] / max(total, 1)) * 100
        logger.info(f"[{domain_label}] Recall@{k}: {r_k:.4f}%, NDCG@{k}: {n_k:.4f}%")

    return recall_sum, ndcg_sum, total