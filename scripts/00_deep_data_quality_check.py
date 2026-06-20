from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch


# 确保从 scripts/ 直接运行时可以找到 isddg 包
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isddg.data.io import load_interactions, find_processed_csv
from isddg.features.semantic import find_embedding_parquet, load_semantic_embeddings


def safe_float(x: Any, ndigits: int = 6):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return round(float(x), ndigits)
    except Exception:
        return x


def gini_coefficient(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[values >= 0]
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values)) / (n * values.sum()) - (n + 1) / n)


def quantile_dict(values: np.ndarray, prefix: str) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            f"{prefix}_min": 0,
            f"{prefix}_p25": 0,
            f"{prefix}_median": 0,
            f"{prefix}_p75": 0,
            f"{prefix}_p90": 0,
            f"{prefix}_p95": 0,
            f"{prefix}_max": 0,
        }
    return {
        f"{prefix}_min": safe_float(np.min(values), 4),
        f"{prefix}_p25": safe_float(np.quantile(values, 0.25), 4),
        f"{prefix}_median": safe_float(np.quantile(values, 0.50), 4),
        f"{prefix}_p75": safe_float(np.quantile(values, 0.75), 4),
        f"{prefix}_p90": safe_float(np.quantile(values, 0.90), 4),
        f"{prefix}_p95": safe_float(np.quantile(values, 0.95), 4),
        f"{prefix}_max": safe_float(np.max(values), 4),
    }


def get_parquet_schema_info(path: Path) -> Tuple[List[str], int]:
    """
    尽量只读取 parquet schema 和行数，避免一开始就把 4096 维 embedding 全部读入内存。
    """
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        rows = pf.metadata.num_rows
        return cols, rows
    except Exception:
        df = pd.read_parquet(path)
        cols = list(df.columns)
        rows = len(df)
        del df
        gc.collect()
        return cols, rows


def detect_embedding_columns(columns):
    id_candidates = [
        c for c in columns
        if c.lower() in {"itemid", "item_id", "asin", "rawitemid", "raw_item_id"}
    ]

    if "item_text_embedding" in columns:
        emb_col = "item_text_embedding"
    else:
        emb_candidates = [
            c for c in columns
            if "embedding" in c.lower()
            and c.lower() not in {"oldembeddingitemid", "old_embedding_item_id"}
        ]
        emb_col = emb_candidates[0] if emb_candidates else None

    id_col = "RawItemId" if "RawItemId" in columns else (id_candidates[0] if id_candidates else None)
    return id_col, emb_col



def check_embedding_coverage_and_values(
    data_root: str | Path,
    domain: str,
    df: pd.DataFrame,
    item_map: Dict[str, int],
    sample_missing: int = 10,
    embedding_dir: str = "semantic_embeddings",
) -> Dict[str, Any]:
    """
    检查：
    1. 当前过滤后交互中实际出现的 RawItemId 是否都能在 embedding 文件里找到；
    2. embedding 文件是否有多余物品；
    3. 对齐后的 embedding table 是否存在 NaN/Inf/零向量；
    4. 对齐后的 shape 是否等于 当前 item 数 + padding。
    """
    emb_path = find_embedding_parquet(data_root, domain, embedding_dir=embedding_dir)
    columns, parquet_rows = get_parquet_schema_info(emb_path)
    id_col, emb_col = detect_embedding_columns(columns)

    if id_col is None:
        raise ValueError(
            f"{domain}: embedding parquet must contain RawItemId/asin/product_id. "
            f"Row-order alignment is forbidden for formal data freeze. "
            f"Current columns={columns}"
        )

    used_raw_items = set(df["RawItemId"].astype(str))
    duplicate_embedding_item_ids = None
    missing_used_items = set()
    unused_embedding_items = set()
    embedding_items_count = None

    emb_ids = pd.read_parquet(emb_path, columns=[id_col])[id_col].astype(str)
    emb_item_list = emb_ids.tolist()
    emb_item_set = set(emb_item_list)

    embedding_items_count = len(emb_item_set)
    duplicate_embedding_item_ids = len(emb_item_list) - len(emb_item_set)
    missing_used_items = used_raw_items - emb_item_set
    unused_embedding_items = emb_item_set - used_raw_items

    del emb_ids
    gc.collect()

    aligned_sem = load_semantic_embeddings(
        data_root,
        domain,
        item_map=item_map,
        embedding_dir=embedding_dir,
        strict=True,
    )

    expected_rows = len(item_map) + 1
    valid = aligned_sem[1:].float()

    nan_ratio = torch.isnan(valid).float().mean().item()
    inf_ratio = torch.isinf(valid).float().mean().item()

    finite_valid = torch.nan_to_num(valid, nan=0.0, posinf=0.0, neginf=0.0)
    norms = finite_valid.norm(dim=1)

    zero_vector_ratio = (norms == 0).float().mean().item()
    norm_min = norms.min().item() if len(norms) > 0 else 0.0
    norm_mean = norms.mean().item() if len(norms) > 0 else 0.0
    norm_std = norms.std().item() if len(norms) > 1 else 0.0
    norm_max = norms.max().item() if len(norms) > 0 else 0.0

    out = {
        "embedding_path": str(emb_path),
        "embedding_parquet_rows": int(parquet_rows),
        "embedding_columns": columns,
        "embedding_id_col": id_col,
        "embedding_vector_col": emb_col,
        "embedding_items_count": embedding_items_count,
        "duplicate_embedding_item_ids": duplicate_embedding_item_ids,
        "used_raw_items": len(used_raw_items),
        "missing_used_items": len(missing_used_items),
        "unused_embedding_items": len(unused_embedding_items),
        "missing_used_item_samples": sorted(list(missing_used_items))[:sample_missing],
        "unused_embedding_item_samples": sorted(list(unused_embedding_items))[:sample_missing],
        "aligned_embedding_shape": list(aligned_sem.shape),
        "aligned_embedding_expected_shape": [expected_rows, int(aligned_sem.shape[1])],
        "aligned_embedding_shape_ok": bool(aligned_sem.shape[0] == expected_rows),
        "embedding_dim": int(aligned_sem.shape[1]),
        "nan_ratio": safe_float(nan_ratio, 8),
        "inf_ratio": safe_float(inf_ratio, 8),
        "zero_vector_ratio": safe_float(zero_vector_ratio, 8),
        "embedding_norm_min": safe_float(norm_min, 6),
        "embedding_norm_mean": safe_float(norm_mean, 6),
        "embedding_norm_std": safe_float(norm_std, 6),
        "embedding_norm_max": safe_float(norm_max, 6),
    }

    del aligned_sem, valid, finite_valid, norms
    gc.collect()
    return out



def check_sequence_and_timestamp_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    检查序列长度、时间戳、重复交互、target 是否重复出现在历史等问题。
    """
    user_len = df.groupby("UserId").size().astype(int)
    seq_lens = user_len.to_numpy()

    item_freq = df.groupby("ItemId").size().astype(int)
    item_freq_values = item_freq.to_numpy()

    # source prefix samples 与 target eval samples 的理论数量
    source_prefix_samples = int(np.maximum(seq_lens - 2, 0).sum())
    target_eval_samples = int((seq_lens >= 2).sum())

    # 重复交互
    duplicate_user_item = int(df.duplicated(["UserId", "RawItemId"]).sum())
    duplicate_user_item_timestamp = int(df.duplicated(["UserId", "RawItemId", "Timestamp"]).sum())

    # 时间戳检查
    timestamp_unique = int(df["Timestamp"].nunique())
    timestamp_min = float(df["Timestamp"].min())
    timestamp_max = float(df["Timestamp"].max())
    timestamp_span = timestamp_max - timestamp_min

    same_user_timestamp_duplicates = int(df.duplicated(["UserId", "Timestamp"]).sum())
    same_user_timestamp_duplicate_ratio = same_user_timestamp_duplicates / max(len(df), 1)

    per_user_time_unique = df.groupby("UserId")["Timestamp"].nunique()
    users_all_same_timestamp = int((per_user_time_unique <= 1).sum())
    users_all_same_timestamp_ratio = users_all_same_timestamp / max(df["UserId"].nunique(), 1)

    # 因为 load_interactions 已经排序，这里检查排序后是否仍存在非单调情况
    non_monotonic_users = 0
    for _, g in df.groupby("UserId", sort=False):
        t = g["Timestamp"].to_numpy()
        if np.any(np.diff(t) < 0):
            non_monotonic_users += 1

    # target item 是否已经在 history 里出现过
    repeated_target_in_history = 0
    total_eval_users = 0
    for _, g in df.groupby("UserId", sort=False):
        items = g["ItemId"].astype(int).tolist()
        if len(items) >= 2:
            total_eval_users += 1
            if items[-1] in set(items[:-1]):
                repeated_target_in_history += 1

    repeated_target_in_history_ratio = repeated_target_in_history / max(total_eval_users, 1)

    # 长尾分布
    n_items = len(item_freq_values)
    top_1pct_n = max(1, math.ceil(n_items * 0.01))
    top_5pct_n = max(1, math.ceil(n_items * 0.05))
    sorted_freq = np.sort(item_freq_values)[::-1]

    top_1pct_interaction_share = sorted_freq[:top_1pct_n].sum() / max(len(df), 1)
    top_5pct_interaction_share = sorted_freq[:top_5pct_n].sum() / max(len(df), 1)
    tail_item_le_2_ratio = float((item_freq_values <= 2).mean()) if n_items > 0 else 0.0
    tail_item_le_5_ratio = float((item_freq_values <= 5).mean()) if n_items > 0 else 0.0

    out = {
        "interactions": int(len(df)),
        "users": int(df["UserId"].nunique()),
        "items": int(df["ItemId"].nunique()),
        "valid_sequences_len>=3": int((seq_lens >= 3).sum()),
        "avg_seq_len": safe_float(seq_lens.mean(), 4),
        "source_prefix_samples": source_prefix_samples,
        "target_eval_samples": target_eval_samples,

        "duplicate_user_item": duplicate_user_item,
        "duplicate_user_item_ratio": safe_float(duplicate_user_item / max(len(df), 1), 8),
        "duplicate_user_item_timestamp": duplicate_user_item_timestamp,
        "duplicate_user_item_timestamp_ratio": safe_float(
            duplicate_user_item_timestamp / max(len(df), 1), 8
        ),

        "timestamp_unique": timestamp_unique,
        "timestamp_min": safe_float(timestamp_min, 4),
        "timestamp_max": safe_float(timestamp_max, 4),
        "timestamp_span": safe_float(timestamp_span, 4),
        "same_user_timestamp_duplicates": same_user_timestamp_duplicates,
        "same_user_timestamp_duplicate_ratio": safe_float(
            same_user_timestamp_duplicate_ratio, 8
        ),
        "users_all_same_timestamp": users_all_same_timestamp,
        "users_all_same_timestamp_ratio": safe_float(
            users_all_same_timestamp_ratio, 8
        ),
        "non_monotonic_users_after_sort": int(non_monotonic_users),

        "repeated_target_in_history": int(repeated_target_in_history),
        "repeated_target_in_history_ratio": safe_float(
            repeated_target_in_history_ratio, 8
        ),

        "item_gini": safe_float(gini_coefficient(item_freq_values), 6),
        "top_1pct_interaction_share": safe_float(top_1pct_interaction_share, 6),
        "top_5pct_interaction_share": safe_float(top_5pct_interaction_share, 6),
        "tail_item_le_2_ratio": safe_float(tail_item_le_2_ratio, 6),
        "tail_item_le_5_ratio": safe_float(tail_item_le_5_ratio, 6),
    }

    out.update(quantile_dict(seq_lens, "seq_len"))
    out.update(quantile_dict(item_freq_values, "item_freq"))
    return out


def build_domain_warnings(row: Dict[str, Any], is_source: bool) -> List[str]:
    warnings = []

    if row.get("valid_sequences_len>=3", 0) <= 0:
        warnings.append("no valid user sequences with length >= 3")

    if row.get("target_eval_samples", 0) < 1000:
        warnings.append("too few target evaluation users")

    if is_source and row.get("timestamp_unique", 0) <= 10:
        warnings.append("source timestamp diversity is too low for dynamic-role construction")

    if row.get("timestamp_unique", 0) <= 1:
        warnings.append("timestamps have <= 1 unique value; sequential order may be unreliable")

    if row.get("missing_used_items", 0) > 0:
        warnings.append("some used interaction items do not have semantic embeddings")

    if not row.get("aligned_embedding_shape_ok", False):
        warnings.append("aligned embedding shape does not match current item_map")

    if row.get("nan_ratio", 0) > 0:
        warnings.append("semantic embedding contains NaN values")

    if row.get("inf_ratio", 0) > 0:
        warnings.append("semantic embedding contains Inf values")

    if row.get("zero_vector_ratio", 0) > 0:
        warnings.append("semantic embedding contains zero vectors")

    if row.get("duplicate_user_item_ratio", 0) > 0.05:
        warnings.append("many duplicated user-item interactions; consider de-duplication")

    if row.get("users_all_same_timestamp_ratio", 0) > 0.5:
        warnings.append("many users have identical timestamps; sequence order may rely on file order")

    return warnings



def check_one_domain(
    data_root: str | Path,
    domain: str,
    source: str,
    sample_missing: int = 10,
    embedding_dir: str = "semantic_embeddings",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    csv_path = find_processed_csv(data_root, domain)
    df, item_map = load_interactions(data_root, domain)

    seq_info = check_sequence_and_timestamp_quality(df)
    emb_info = check_embedding_coverage_and_values(
        data_root=data_root,
        domain=domain,
        df=df,
        item_map=item_map,
        sample_missing=sample_missing,
        embedding_dir=embedding_dir,
    )

    # summary CSV 不适合保存很长的 list，但 shape 字段需要保留
    row_emb_info = {}
    for k, v in emb_info.items():
        if k in {
            "embedding_columns",
            "missing_used_item_samples",
            "unused_embedding_item_samples",
        }:
            continue

        if k in {
            "aligned_embedding_shape",
            "aligned_embedding_expected_shape",
        }:
            row_emb_info[k] = str(v)
        else:
            row_emb_info[k] = v

    row = {
        "domain": domain,
        "csv_path": str(csv_path),
        **seq_info,
        **row_emb_info,
    }

    warnings = build_domain_warnings(row, is_source=(domain == source))
    row["status"] = "PASS" if not warnings else "WARN"
    row["warnings"] = " | ".join(warnings)

    detail = {
        "domain": domain,
        "csv_path": str(csv_path),
        "sequence_and_timestamp": seq_info,
        "embedding": emb_info,
        "warnings": warnings,
    }

    # 供跨域 overlap 使用，不写入 json
    detail["_user_set"] = set(df["UserId"].astype(str))
    detail["_item_set"] = set(df["RawItemId"].astype(str))

    del df
    gc.collect()
    return row, detail



def check_cross_domain_overlap(
    source_domain: str,
    target_domains: List[str],
    details: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    source_users = details[source_domain]["_user_set"]
    source_items = details[source_domain]["_item_set"]

    rows = []
    detail_rows = []

    for target in target_domains:
        target_users = details[target]["_user_set"]
        target_items = details[target]["_item_set"]

        user_overlap = source_users & target_users
        item_overlap = source_items & target_items

        row = {
            "source": source_domain,
            "target": target,
            "source_users": len(source_users),
            "target_users": len(target_users),
            "source_items": len(source_items),
            "target_items": len(target_items),
            "user_overlap": len(user_overlap),
            "item_overlap": len(item_overlap),
            "zero_shot_ok": len(user_overlap) == 0 and len(item_overlap) == 0,
        }
        rows.append(row)
        detail_rows.append({
            **row,
            "user_overlap_samples": sorted(list(user_overlap))[:20],
            "item_overlap_samples": sorted(list(item_overlap))[:20],
        })

    return pd.DataFrame(rows), detail_rows


def make_json_safe(obj):
    if isinstance(obj, set):
        return sorted(list(obj))
    if isinstance(obj, dict):
        return {
            k: make_json_safe(v)
            for k, v in obj.items()
            if not k.startswith("_")
        }
    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--source", default="amazon_movies_and_tv")
    parser.add_argument(
        "--targets",
        default="amazon_cds_and_vinyl,amazon_industrial_and_scientific",
    )
    parser.add_argument("--out_dir", default="results/data_quality_new")
    parser.add_argument("--sample_missing", type=int, default=10)
    parser.add_argument("--embedding_dir", default="semantic_embeddings")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    domains = [args.source] + targets

    summary_rows = []
    details = {}

    print("=" * 100)
    print("Deep data quality check")
    print(f"data_root = {data_root}")
    print(f"source = {args.source}")
    print(f"targets = {targets}")
    print("=" * 100)

    for domain in domains:
        print(f"\nChecking domain: {domain}")
        row, detail = check_one_domain(
            data_root=data_root,
            domain=domain,
            source=args.source,
            sample_missing=args.sample_missing,
            embedding_dir=args.embedding_dir,
        )
        summary_rows.append(row)
        details[domain] = detail

        brief = {
            "domain": row.get("domain"),
            "status": row.get("status"),
            "interactions": row.get("interactions"),
            "users": row.get("users"),
            "items": row.get("items"),
            "avg_seq_len": row.get("avg_seq_len"),
            "target_eval_samples": row.get("target_eval_samples"),
            "aligned_embedding_shape": row.get("aligned_embedding_shape"),
            "aligned_embedding_shape_ok": row.get("aligned_embedding_shape_ok"),
            "missing_used_items": row.get("missing_used_items"),
            "unused_embedding_items": row.get("unused_embedding_items"),
            "nan_ratio": row.get("nan_ratio"),
            "inf_ratio": row.get("inf_ratio"),
            "zero_vector_ratio": row.get("zero_vector_ratio"),
            "warnings": row.get("warnings"),
        }
        print(json.dumps(brief, indent=2, ensure_ascii=False))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "domain_quality_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    cross_df, cross_details = check_cross_domain_overlap(
        source_domain=args.source,
        target_domains=targets,
        details=details,
    )
    cross_path = out_dir / "cross_domain_overlap.csv"
    cross_df.to_csv(cross_path, index=False, encoding="utf-8-sig")

    detail_json = {
        "domains": {
            d: make_json_safe(details[d])
            for d in domains
        },
        "cross_domain_overlap": cross_details,
    }
    detail_path = out_dir / "domain_quality_details.json"
    detail_path.write_text(
        json.dumps(detail_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 100)
    print("Cross-domain overlap")
    print(cross_df.to_string(index=False))
    print("=" * 100)
    print(f"Saved summary to: {summary_path}")
    print(f"Saved overlap to: {cross_path}")
    print(f"Saved details to: {detail_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
