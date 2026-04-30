import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch


def preprocess(dataset_path, dataset_saved_path, task):
    """
    Kept for compatibility with the original project structure.
    """
    if os.path.isdir(dataset_saved_path):
        if task == "non-overlap":
            processed_path = os.path.join(dataset_saved_path, "processed_data.csv")
            if os.path.exists(processed_path):
                return pd.read_csv(processed_path)

    if "douban" in dataset_path:
        dataset = pd.read_csv(dataset_path, sep="\t")
        max_seq_length = 200
    elif "epinions" in dataset_path:
        dataset = pd.read_csv(dataset_path)
        dataset = dataset.rename(
            columns={"user": "UserId", "item": "ItemId", "time": "Timestamp"}
        )
        max_seq_length = 200
    else:
        raise ValueError(f"Cannot recognize dataset type for {dataset_path}")

    item_stat = dataset["ItemId"].value_counts()
    user_stat = dataset["UserId"].value_counts()

    filtered_item = item_stat[item_stat >= 10].index
    filtered_user = user_stat[user_stat >= 10].index

    filtered_dataset = dataset[
        dataset["UserId"].isin(filtered_user) & dataset["ItemId"].isin(filtered_item)
    ]
    filtered_dataset = (
        filtered_dataset
        .sort_values("Timestamp")
        .groupby("UserId", group_keys=False)
        .apply(lambda x: x.tail(max_seq_length))
        .reset_index(drop=True)
    )

    os.makedirs(dataset_saved_path, exist_ok=True)
    filtered_dataset.to_csv(os.path.join(dataset_saved_path, "processed_data.csv"), index=False)
    return filtered_dataset


def label_split(dataset):
    labels = []
    for user_id, group in dataset.groupby("UserId"):
        if len(group) >= 3:
            test_label = group.iloc[-1]
            validation_label = group.iloc[-2]
            training_label = group.iloc[-3]

            labels.append(
                {
                    "userId": user_id,
                    "test_label": test_label["value"],
                    "validation_label": validation_label["value"],
                    "training_label": training_label["value"],
                }
            )
    return pd.DataFrame(labels)


def resolve_embedding_path(dataset_name: str) -> str:
    """
    Resolve the embedding parquet path for a dataset.
    """
    candidates = [
        f"./data/{dataset_name}/{dataset_name}_embedding_llama.parquet",
        f"./data/{dataset_name}/{dataset_name}_embedding_llama3.parquet",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Cannot find embedding parquet for {dataset_name}. "
        f"Tried: {candidates}"
    )


def _build_steam_parquet_id_map(data: pd.DataFrame) -> Dict[Any, int]:
    """
    Exactly match my_batch_sem.py for Steam:
    raw ItemId -> 0..N-1 according to first appearance order.
    """
    unique_item_ids = data["ItemId"].unique()
    return {old_id: new_id for new_id, old_id in enumerate(unique_item_ids, start=0)}


def _build_amazon_parquet_id_map(data: pd.DataFrame) -> Dict[Any, int]:
    """
    Exactly match my_batch_sem.py for Amazon:
    stage1: raw ItemId -> 0..N-1
    stage2: stage1 id -> 1..N
    final : raw ItemId -> 1..N
    """
    unique_item_ids = data["ItemId"].unique()
    item_id_map_stage1 = {
        old_id: new_id for new_id, old_id in enumerate(unique_item_ids, start=0)
    }
    temp_ids = data["ItemId"].map(item_id_map_stage1)
    unique_item_ids_v2 = temp_ids.unique()
    item_id_map_stage2 = {
        old_id: new_id for new_id, old_id in enumerate(unique_item_ids_v2, start=1)
    }
    item_id_map = {
        old: item_id_map_stage2[item_id_map_stage1[old]]
        for old in item_id_map_stage1
    }
    return item_id_map


def build_raw_to_model_item_id_map(data: pd.DataFrame, dataset_name: str) -> Dict[Any, int]:
    """
    Build the model-side item index mapping that is strictly consistent with the
    current embedding generation script.

    Model convention:
    - 0 is reserved for padding
    - actual items must be in 1..N

    Therefore:
    - Amazon parquet already uses 1..N -> model uses the same 1..N
    - Steam parquet uses 0..N-1 -> model uses (parquet_id + 1) -> 1..N
    """
    if "ItemId" not in data.columns:
        raise ValueError("Input dataframe must contain 'ItemId' column.")

    if "steam" in dataset_name.lower():
        parquet_id_map = _build_steam_parquet_id_map(data)
        model_id_map = {raw_id: parquet_id + 1 for raw_id, parquet_id in parquet_id_map.items()}
        return model_id_map

    parquet_id_map = _build_amazon_parquet_id_map(data)
    # Amazon parquet ids are already 1..N and directly usable by the model
    return parquet_id_map


def remap_processed_data_item_ids(
    data: pd.DataFrame,
    dataset_name: str,
    item_col: str = "ItemId",
) -> Tuple[pd.DataFrame, Dict[Any, int]]:
    """
    Remap processed_data.csv ItemId into the model's internal item index space.

    This function is the key bridge between:
    - raw processed_data.csv ItemId
    - embedding parquet ItemId
    - model input sequence item ids

    Returns:
    - remapped dataframe
    - raw_id -> model_id mapping
    """
    if item_col not in data.columns:
        raise ValueError(f"Input dataframe must contain '{item_col}' column.")

    df = data.copy()
    raw_to_model_map = build_raw_to_model_item_id_map(df, dataset_name)

    df["RawItemId"] = df[item_col]
    df[item_col] = df[item_col].map(raw_to_model_map)

    missing = int(df[item_col].isna().sum())
    if missing > 0:
        raise ValueError(
            f"[{dataset_name}] Failed to map {missing} item ids from processed_data.csv "
            f"into the model index space."
        )

    df[item_col] = df[item_col].astype(int)
    return df, raw_to_model_map


def load_pretrained_embeddings(parquet_path: str) -> torch.Tensor:
    """
    Load semantic item embeddings from parquet into the model-side embedding tensor.

    Parquet convention from current embedding-generation script:
    - Amazon parquet ItemId: 1..N
    - Steam parquet ItemId: 0..N-1

    Model convention:
    - embedding_tensor[0] is padding
    - actual items occupy 1..N

    Therefore:
    - Amazon: keep ItemId as is
    - Steam: shift parquet ItemId by +1
    """
    df = pd.read_parquet(parquet_path)

    if "ItemId" not in df.columns:
        raise ValueError(f"{parquet_path} does not contain ItemId column.")
    if "item_text_embedding" not in df.columns:
        raise ValueError(f"{parquet_path} does not contain item_text_embedding column.")

    df = df.copy().sort_values("ItemId").reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{parquet_path} is empty.")

    if df["ItemId"].isna().any():
        raise ValueError(f"{parquet_path} contains NaN ItemId values.")

    if df["ItemId"].duplicated().any():
        dup_count = int(df["ItemId"].duplicated().sum())
        raise ValueError(f"{parquet_path} contains {dup_count} duplicated ItemId values.")

    item_ids = df["ItemId"].astype(int)
    min_id = int(item_ids.min())
    max_id = int(item_ids.max())

    if min_id not in (0, 1):
        raise ValueError(
            f"{parquet_path} has unexpected ItemId range starting from {min_id}. "
            "Expected 0-based (Steam) or 1-based (Amazon) ids."
        )

    is_zero_based = (min_id == 0)
    embedding_dim = len(df["item_text_embedding"].iloc[0])
    num_items = max_id + 1 if is_zero_based else max_id

    embedding_tensor = torch.zeros((num_items + 1, embedding_dim), dtype=torch.float32)

    for _, row in df.iterrows():
        parquet_item_id = int(row["ItemId"])
        model_item_id = parquet_item_id + 1 if is_zero_based else parquet_item_id
        embedding_tensor[model_item_id] = torch.tensor(
            row["item_text_embedding"],
            dtype=torch.float32,
        )

    embedding_tensor[0] = 0.0
    return embedding_tensor


def load_user_reviews(file_path: str) -> List[Dict[str, Any]]:
    reviews = []
    with open(file_path, "r", encoding="utf-8-sig") as fp:
        for i, line in enumerate(fp, start=1):
            try:
                normalized_line = line.replace("\r\n", "\n").strip()
                reviews.append(json.loads(normalized_line))
            except json.JSONDecodeError as exc:
                print(f"Error decoding review JSON on line {i}: {exc}")
    return reviews


def load_item_metadata(file_path: str) -> List[Dict[str, Any]]:
    metadata = []
    with open(file_path, "r", encoding="utf-8-sig") as fp:
        for i, line in enumerate(fp, start=1):
            try:
                normalized_line = line.replace("\r\n", "\n").strip()
                metadata.append(json.loads(normalized_line))
            except json.JSONDecodeError as exc:
                print(f"Error decoding metadata JSON on line {i}: {exc}")
    return metadata


def _normalize_text_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(x) for x in value if x is not None)
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values() if v is not None)
    return str(value)


def preprocess_data(reviews, metadata):
    """
    Paper-aligned preprocessing:

    - filter out users/items with fewer than 10 interactions
    - exclude items lacking textual features
    - no additional time-window filtering
    - preserve Timestamp / UserId / ItemId compatibility for later embedding alignment
    """
    reviews_df = pd.DataFrame(reviews)
    metadata_df = pd.DataFrame(metadata)

    if reviews_df.empty:
        raise ValueError("reviews is empty.")
    if metadata_df.empty:
        raise ValueError("metadata is empty.")

    reviews_df = reviews_df.copy()
    metadata_df = metadata_df.copy()

    reviews_df["timestamp"] = pd.to_datetime(reviews_df["timestamp"], unit="ms", errors="coerce")
    reviews_df = reviews_df.dropna(subset=["timestamp", "parent_asin", "user_id"])

    text_columns = [col for col in ["title", "description", "features"] if col in metadata_df.columns]
    if not text_columns:
        raise ValueError("metadata must contain at least one textual field among title/description/features.")

    for col in text_columns:
        metadata_df[col] = metadata_df[col].apply(_normalize_text_field)

    metadata_df["text_blob"] = metadata_df[text_columns].fillna("").agg(" ".join, axis=1).str.strip()
    metadata_df = metadata_df[metadata_df["text_blob"].str.len() > 0]

    reviews_df = reviews_df.rename(
        columns={
            "timestamp": "Timestamp",
            "parent_asin": "ItemId",
            "user_id": "UserId",
        }
    )
    metadata_df = metadata_df.rename(columns={"parent_asin": "ItemId"})

    merged = reviews_df.merge(metadata_df, how="inner", on="ItemId")

    item_counts = merged["ItemId"].value_counts()
    user_counts = merged["UserId"].value_counts()

    valid_items = item_counts[item_counts >= 10].index
    valid_users = user_counts[user_counts >= 10].index

    merged = merged[
        merged["ItemId"].isin(valid_items) & merged["UserId"].isin(valid_users)
    ].copy()

    merged = merged.sort_values(["UserId", "Timestamp"]).reset_index(drop=True)
    return merged


def save_preprocessed_data(df: pd.DataFrame, output_file: str):
    df.to_csv(output_file, index=False)
