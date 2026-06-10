from __future__ import annotations

from typing import Callable
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .losses import bpr_loss
from isddg.data.dataset import NegativeSampler


def train_sequence_model(
    model,
    loader: DataLoader,
    num_items: int,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-4,
    train_negatives: int = 5,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sampler = NegativeSampler(num_items=num_items)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0
        pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False)
        for batch in pbar:
            hist = batch["history"].to(device)
            pos = batch["target"].to(device)
            neg = []
            for h, p in zip(hist.tolist(), pos.tolist()):
                forbidden = set([x for x in h if x != 0])
                forbidden.add(int(p))
                neg.append(sampler.sample(forbidden, train_negatives))
            neg = torch.tensor(neg, dtype=torch.long, device=device)
            candidates = torch.cat([pos.view(-1, 1), neg], dim=1)
            scores = model.score(hist, candidates) if hasattr(model, "score") else model(hist, candidates)
            loss = bpr_loss(scores[:, 0], scores[:, 1:])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * hist.size(0)
            n += hist.size(0)
            pbar.set_postfix(loss=total / max(n, 1))
        print(f"epoch={epoch} loss={total / max(n, 1):.6f}")
    return model
