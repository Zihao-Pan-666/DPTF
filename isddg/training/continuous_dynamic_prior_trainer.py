from __future__ import annotations
from pathlib import Path
from typing import Dict
import time
import torch
from torch import nn
from tqdm import tqdm

class ContinuousDynamicPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())

@torch.no_grad()
def predict_continuous_table(model: nn.Module, item_features: torch.Tensor, batch_size: int = 4096, device: torch.device | None = None) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    outs = []
    for s in range(0, item_features.size(0), batch_size):
        outs.append(model(item_features[s:s + batch_size].to(device)).detach().cpu())
    table = torch.cat(outs, dim=0)
    table[0] = 0.0
    return table

def _mse_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    err = y_pred - y_true
    mse = (err ** 2).mean()
    mae = err.abs().mean()
    return {"mse": float(mse.item()), "mae": float(mae.item())}

def train_continuous_dynamic_prior(
    item_features: torch.Tensor,
    target_table: torch.Tensor,
    checkpoint_path: str | Path,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    patience: int = 10,
    val_ratio: float = 0.1,
    seed: int = 2026,
    batch_size: int = 4096,
    device: torch.device | None = None,
    checkpoint_extra: Dict | None = None,
) -> Dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = item_features.float()
    y = target_table.float()
    valid = torch.arange(1, x.size(0))
    g = torch.Generator().manual_seed(seed)
    perm = valid[torch.randperm(valid.numel(), generator=g)]
    n_val = max(1, int(len(perm) * val_ratio))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = ContinuousDynamicPredictor(x.size(1), y.size(1), hidden_dim=hidden_dim, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    best, best_epoch, bad, history = float("inf"), -1, 0, []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=g)]
        total_loss, total_n = 0.0, 0
        for s in tqdm(range(0, train_idx.numel(), batch_size), desc=f"cont-dyn-prior epoch {epoch:03d}", leave=False):
            idx = train_idx[s:s + batch_size]
            pred = model(x[idx].to(device))
            loss = torch.nn.functional.mse_loss(pred, y[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.item()) * idx.numel()
            total_n += idx.numel()

        model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, val_idx.numel(), batch_size):
                idx = val_idx[s:s + batch_size]
                preds.append(model(x[idx].to(device)).detach().cpu())
        pred_val = torch.cat(preds, 0)
        val = _mse_metrics(y[val_idx], pred_val)
        row = {"epoch": epoch, "train_mse": total_loss / max(total_n, 1), "val_mse": val["mse"], "val_mae": val["mae"]}
        history.append(row)

        if val["mse"] < best:
            best, best_epoch, bad = val["mse"], epoch, 0
            payload = {
                "model_state": model.state_dict(),
                "input_dim": int(x.size(1)),
                "output_dim": int(y.size(1)),
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "best_epoch": int(best_epoch),
                "best_val_mse": float(best),
                "history": history,
                "total_elapsed_sec": time.time() - t0,
            }
            if checkpoint_extra:
                payload.update(checkpoint_extra)
            torch.save(payload, checkpoint_path)
            status = "saved"
        else:
            bad += 1
            status = f"patience={bad}/{patience}"

        print(f"[ContDynPrior][epoch={epoch:03d}] train_mse={row['train_mse']:.6f} val_mse={val['mse']:.6f} val_mae={val['mae']:.6f} best={best:.6f}@{best_epoch} {status}", flush=True)
        if bad >= patience:
            break

    return {"checkpoint_path": str(checkpoint_path), "best_epoch": best_epoch, "best_val_mse": best, "history": history, "total_elapsed_sec": time.time() - t0}

def load_continuous_predictor_from_checkpoint(path: str | Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    model = ContinuousDynamicPredictor(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        dropout=float(ckpt.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt
