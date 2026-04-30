from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def bpr_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-10).mean()


@torch.no_grad()
def sample_negatives(seq: torch.Tensor, pos: torch.Tensor, num_items: int, num_negatives: int) -> torch.Tensor:
    device = seq.device
    out = torch.randint(1, num_items + 1, (seq.size(0), num_negatives), device=device)
    blocked = torch.cat([seq, pos.unsqueeze(1)], dim=1)
    for _ in range(10):
        invalid = (out.unsqueeze(-1) == blocked.unsqueeze(1)).any(dim=-1)
        if not invalid.any():
            break
        out[invalid] = torch.randint(1, num_items + 1, (int(invalid.sum()),), device=device)
    return out


def train_one_model(model, train_loader: DataLoader, num_items: int, device: torch.device,
                    epochs: int = 20, lr: float = 1e-4, train_negatives: int = 5):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        total, steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"train epoch {ep:02d}", leave=False)
        for seq, rel_time, val, _test in pbar:
            seq = seq.to(device)
            rel_time = rel_time.to(device)
            pos = val.to(device)
            neg = sample_negatives(seq, pos, num_items, train_negatives)
            logits = model(seq, rel_time)
            pos_logits = logits.gather(1, pos.view(-1, 1)).expand_as(neg)
            neg_logits = logits.gather(1, neg)
            loss = bpr_loss(pos_logits.reshape(-1), neg_logits.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=f"{total/max(steps,1):.4f}")
        print(f"epoch={ep:02d} loss={total/max(steps,1):.4f}")
    return model


@torch.no_grad()
def evaluate(model, data_loader: DataLoader, num_items: int, device: torch.device,
             num_negatives: int = 100, k_list: Iterable[int] = (10, 20)) -> Dict[str, float]:
    model.eval()
    sums = {f"R@{k}": 0.0 for k in k_list}
    sums.update({f"N@{k}": 0.0 for k in k_list})
    total = 0
    for seq, rel_time, _val, test in tqdm(data_loader, desc="eval", leave=False):
        seq = seq.to(device)
        rel_time = rel_time.to(device)
        test = test.to(device)
        neg = sample_negatives(seq, test, num_items, num_negatives)
        candidates = torch.cat([test.view(-1, 1), neg], dim=1)
        logits_all = model(seq, rel_time)
        scores = logits_all.gather(1, candidates)
        order = scores.argsort(dim=1, descending=True)
        ranked_items = candidates.gather(1, order)
        for i in range(seq.size(0)):
            total += 1
            gt = int(test[i].item())
            rank_list = ranked_items[i].tolist()
            for k in k_list:
                topk = rank_list[:k]
                if gt in topk:
                    sums[f"R@{k}"] += 1.0
                    rank = topk.index(gt) + 1
                    sums[f"N@{k}"] += 1.0 / math.log2(rank + 1)
    return {m: 100.0 * v / max(total, 1) for m, v in sums.items()} | {"n_users": float(total)}


def append_result_csv(path: str, row: Dict[str, object]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
