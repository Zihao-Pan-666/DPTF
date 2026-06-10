from __future__ import annotations

from pathlib import Path
import json
import torch
import pandas as pd


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_torch(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def append_csv(row: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        # Align columns explicitly to prevent schema drift.
        cols = list(dict.fromkeys(list(old.columns) + list(df.columns)))
        old = old.reindex(columns=cols)
        df = df.reindex(columns=cols)
        pd.concat([old, df], ignore_index=True).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)
