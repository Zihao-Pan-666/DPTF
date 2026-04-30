from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import remap_processed_data_item_ids


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
        avg_len = (
            sum(len(seq) for seq in self.user_sequences) / len(self.user_sequences)
            if self.user_sequences
            else 0.0
        )

        # print(f"[{dataset_name}] Users: {len(self.user_sequences)}")
        # print(f"[{dataset_name}] Num mapped items: {self.num_items}")
        # print(f"[{dataset_name}] Unique mapped items: {unique_items}")
        # print(f"[{dataset_name}] Avg Len: {avg_len:.2f}")

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
