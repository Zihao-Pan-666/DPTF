import ast
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import remap_processed_data_item_ids


# =========================================================
# 1. 核心序列数据集模块 (支持当前 Baseline 及所有模型常规打分)
# =========================================================

class _BaseSequenceDataset(Dataset):
    """
    IMPORTANT:
    This dataset DOES remap ItemId, but not arbitrarily.

    It strictly reproduces the SAME id construction logic used by the current
    embedding-generation script (`my_batch_sem.py`), so that:
    - sequence item ids
    - embedding tensor row ids
    - model input ids

    all live in the same model-side index space.

    Model-side convention:
    - 0 is reserved for padding
    - actual items are in 1..N
    """

    def __init__(self, data: pd.DataFrame, max_seq_length: int, dataset_name: str):
        required_columns = {"UserId", "ItemId", "Timestamp"}
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"{dataset_name} is missing required columns: {sorted(missing)}")

        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length

        df = data.copy()
        df["UserId"] = df["UserId"].astype(str)

        # Key fix:
        # map raw processed_data.csv ItemId into the SAME model index space
        # implied by the embedding-generation script.
        df, raw_to_model_map = remap_processed_data_item_ids(df, dataset_name=dataset_name)
        self.raw_to_model_map = raw_to_model_map

        df["ItemId"] = df["ItemId"].astype(int)

        if df["ItemId"].min() <= 0:
            raise ValueError(
                f"{dataset_name}: remapped ItemId must start from 1 because 0 is reserved for padding."
            )

        df = df.sort_values(["UserId", "Timestamp"]).reset_index(drop=True)
        self.data_frame = df
        self.num_items = int(df["ItemId"].max())
        self.user_sequences = self._create_user_sequences()

        unique_items = int(df["ItemId"].nunique())

        if unique_items != self.num_items:
            raise ValueError(
                f"[{dataset_name}] Remapped ItemId is not dense in 1..N. "
                f"unique_items={unique_items}, max_item_id={self.num_items}. "
                "This would break embedding alignment."
            )

    def _create_user_sequences(self) -> List[List[int]]:
        user_sequences = []
        for _, group in self.data_frame.groupby("UserId", sort=False):
            sequence = group["ItemId"].astype(int).tolist()
            if len(sequence) >= 3:
                user_sequences.append(sequence)
        return user_sequences

    def __len__(self):
        return len(self.user_sequences)

    def get_num_items(self):
        return self.num_items

    def __getitem__(self, idx):
        sequence = self.user_sequences[idx]

        # 准确划分历史、验证和测试
        train_items = sequence[:-2]
        val_item = sequence[-2]
        test_item = sequence[-1]

        # 强制将 train_seq 锁定为 max_seq_length，确保位置编码一致性
        train_seq = np.zeros(self.max_seq_length, dtype=np.int64)
        seq_len = min(len(train_items), self.max_seq_length)
        if seq_len > 0:
            train_seq[-seq_len:] = np.asarray(train_items[-seq_len:], dtype=np.int64)

        return (
            torch.tensor(train_seq, dtype=torch.long),
            torch.tensor(val_item, dtype=torch.long),
            torch.tensor(test_item, dtype=torch.long),
        )


class SteamDataset(_BaseSequenceDataset):
    def __init__(self, data, max_seq_length, dataset_name="steam"):
        super().__init__(data=data, max_seq_length=max_seq_length, dataset_name=dataset_name)


class AmazonUserSequencesDataset(_BaseSequenceDataset):
    def __init__(self, data, max_seq_length, dataset_name="amazon"):
        super().__init__(data=data, max_seq_length=max_seq_length, dataset_name=dataset_name)


# =========================================================
# 2. 辅助域语义采样模块 (支持后续 DARecG 等创新模型的动态采样对齐)
# =========================================================

def _parse_embedding_cell(x):
    """
    Robust parser for parquet embedding cell.
    Supports list / np.ndarray / stringified list.
    """
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, str):
        # e.g. "[0.1, 0.2, ...]"
        return np.asarray(ast.literal_eval(x), dtype=np.float32)
    # fallback
    return np.asarray(x, dtype=np.float32)


def load_domain_semantic_embeddings(
        domain_name: str,
        data_root: str = "./data",
        parquet_name: Optional[str] = None,
        item_id_col: str = "ItemId",
        emb_col: str = "item_text_embedding",
):
    """
    Load one domain's precomputed semantic embeddings from parquet.
    Returns:
      item_ids: torch.LongTensor [N]
      raw_embs: torch.FloatTensor [N, D0]
    """
    if parquet_name is None:
        parquet_name = f"{domain_name}_embedding_llama.parquet"

    p = Path(data_root) / domain_name / parquet_name
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {p}")

    df = pd.read_parquet(p)
    if item_id_col not in df.columns:
        raise KeyError(f"Column '{item_id_col}' not found in {p}")
    if emb_col not in df.columns:
        raise KeyError(f"Column '{emb_col}' not found in {p}")

    item_ids = torch.tensor(df[item_id_col].values, dtype=torch.long)
    raw_embs = torch.tensor(
        np.stack(df[emb_col].apply(_parse_embedding_cell).values),
        dtype=torch.float32
    )
    return item_ids, raw_embs


class AuxiliarySemanticSampler:
    """
    Runtime auxiliary sampler for Generalized Models:
      - reads cached semantic embeddings (raw)
      - samples auxiliary items from one or multiple domains
      - projects raw embeddings via model.project_raw_for_alignment(...)
    """

    def __init__(
            self,
            model,
            aux_domains: List[str],
            all_domains_for_index: List[str] = None,  # 新增一个参数接收全局列表
            data_root: str = "./data",
            parquet_name_map: Optional[Dict[str, str]] = None,
            aux_batch_size: int = 1024,
            item_id_col: str = "ItemId",
            emb_col: str = "item_text_embedding",
            sample_mode: str = "domain_uniform",  # "domain_uniform" or "global_uniform"
            verbose: bool = True,
    ):
        self.model = model
        self.aux_domains = aux_domains
        self.data_root = data_root
        self.parquet_name_map = parquet_name_map or {}
        self.aux_batch_size = aux_batch_size
        self.item_id_col = item_id_col
        self.emb_col = emb_col
        self.sample_mode = sample_mode

        self.domain_cache = []
        # each element: dict(name, item_ids, raw_embs, domain_id)

        for did, dname in enumerate(self.aux_domains):
            parquet_name = self.parquet_name_map.get(dname, None)
            item_ids, raw_embs = load_domain_semantic_embeddings(
                domain_name=dname,
                data_root=self.data_root,
                parquet_name=parquet_name,
                item_id_col=self.item_id_col,
                emb_col=self.emb_col,
            )
            actual_domain_id = all_domains_for_index.index(dname) if all_domains_for_index else did
            self.domain_cache.append({
                "name": dname,
                "domain_id": actual_domain_id, # 使用全局 ID
                "item_ids": item_ids,  # CPU tensor
                "raw_embs": raw_embs,  # CPU tensor
            })
            if verbose:
                print(f"[AuxSampler] loaded domain={dname}, n_items={item_ids.shape[0]}, emb_dim={raw_embs.shape[1]}")

        if len(self.domain_cache) == 0:
            raise RuntimeError("No auxiliary domain embeddings loaded.")

        # For global uniform sampling
        if self.sample_mode == "global_uniform":
            all_ids = []
            all_raw = []
            all_dom = []
            for pack in self.domain_cache:
                n = pack["item_ids"].shape[0]
                all_ids.append(pack["item_ids"])
                all_raw.append(pack["raw_embs"])
                all_dom.append(torch.full((n,), pack["domain_id"], dtype=torch.long))
            self._global_ids = torch.cat(all_ids, dim=0)
            self._global_raw = torch.cat(all_raw, dim=0)
            self._global_dom = torch.cat(all_dom, dim=0)

            if verbose:
                print(f"[AuxSampler] global pool size={self._global_ids.shape[0]}")

    def _sample_indices(self, n_total: int, k: int) -> torch.Tensor:
        if n_total <= k:
            return torch.arange(n_total, dtype=torch.long)
        return torch.randint(low=0, high=n_total, size=(k,), dtype=torch.long)

    @torch.no_grad()
    def sample(self, device: torch.device):
        """
        Returns dict:
          aux_raw:        [Na, D0]
          aux_projected:  [Na, D]
          aux_ids:        [Na]
          aux_domain_ids: [Na]
          aux_domain_name: str (if sampled from one domain mode)
        """
        if self.sample_mode == "domain_uniform":
            # randomly choose one domain first
            pack = self.domain_cache[np.random.randint(0, len(self.domain_cache))]
            ids = pack["item_ids"]
            raws = pack["raw_embs"]
            did = pack["domain_id"]
            dname = pack["name"]

            sel = self._sample_indices(ids.shape[0], self.aux_batch_size)
            aux_ids = ids[sel].to(device)
            aux_raw = raws[sel].to(device)
            aux_domain_ids = torch.full((aux_ids.shape[0],), did, dtype=torch.long, device=device)

            aux_projected = self.model.project_raw_for_alignment(aux_raw)

            return {
                "aux_raw": aux_raw,
                "aux_projected": aux_projected,
                "aux_ids": aux_ids,
                "aux_domain_ids": aux_domain_ids,
                "aux_domain_name": dname
            }

        elif self.sample_mode == "global_uniform":
            ids = self._global_ids
            raws = self._global_raw
            doms = self._global_dom

            sel = self._sample_indices(ids.shape[0], self.aux_batch_size)
            aux_ids = ids[sel].to(device)
            aux_raw = raws[sel].to(device)
            aux_domain_ids = doms[sel].to(device)

            aux_projected = self.model.project_raw_for_alignment(aux_raw)

            return {
                "aux_raw": aux_raw,
                "aux_projected": aux_projected,
                "aux_ids": aux_ids,
                "aux_domain_ids": aux_domain_ids,
                "aux_domain_name": "mixed"
            }

        else:
            raise ValueError(f"Unsupported sample_mode: {self.sample_mode}")


def build_aux_sampler_from_args(model, args):
    """
    Convenience builder for trainer script.
    Required args:
      - args.data_root
      - args.aux_domains (comma-separated str)  e.g. "amazon_mi,amazon_vg,steam"
    Optional args:
      - args.aux_batch_size
      - args.aux_sample_mode
      - args.item_id_col
      - args.emb_col
    """
    if not hasattr(args, "aux_domains") or args.aux_domains is None or len(args.aux_domains.strip()) == 0:
        raise ValueError("Please provide --aux_domains, e.g. --aux_domains amazon_mi,amazon_vg,steam")

    aux_domains = [x.strip() for x in args.aux_domains.split(",") if x.strip()]

    sampler = AuxiliarySemanticSampler(
        model=model,
        aux_domains=aux_domains,
        data_root=getattr(args, "data_root", "./data"),
        aux_batch_size=getattr(args, "aux_batch_size", 1024),
        item_id_col=getattr(args, "item_id_col", "ItemId"),
        emb_col=getattr(args, "emb_col", "item_text_embedding"),
        sample_mode=getattr(args, "aux_sample_mode", "domain_uniform"),
        verbose=True,
    )

    # return callable for trainer
    def _fn(device):
        return sampler.sample(device=device)

    return _fn