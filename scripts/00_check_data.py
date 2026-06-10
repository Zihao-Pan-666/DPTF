from __future__ import annotations

import sys
from pathlib import Path
import argparse
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isddg.data.io import (
    load_interactions,
    group_user_sequences,
    split_source_prefix_samples,
    build_target_eval_samples,
)
from isddg.features.semantic import load_semantic_embeddings, find_embedding_parquet


def check_domain(data_root: str, domain: str, max_len: int = 50):
    df, item_map = load_interactions(data_root, domain)
    seqs = group_user_sequences(df, min_len=3)

    source_samples = split_source_prefix_samples(seqs, max_len=max_len)
    target_samples = build_target_eval_samples(seqs, max_len=max_len)

    df, item_map = load_interactions(data_root, domain)
    sem = load_semantic_embeddings(data_root, domain, item_map=item_map)
    emb_path = find_embedding_parquet(data_root, domain)
    emb_df = pd.read_parquet(emb_path)

    num_interactions = len(df)
    num_users = df["UserId"].nunique()
    num_items = len(item_map)
    num_valid_seqs = len(seqs)
    avg_seq_len = sum(len(s["items"]) for s in seqs) / max(len(seqs), 1)

    sem_rows, sem_dim = sem.shape
    embedding_count_match = (sem_rows == num_items + 1)

    timestamp_unique = df["Timestamp"].nunique()
    timestamp_valid = timestamp_unique > 1

    possible_id_cols = [
        c for c in emb_df.columns
        if c.lower() in {"itemid", "item_id", "asin", "rawitemid"}
    ]

    status = "PASS"
    warnings = []

    if num_interactions == 0 or num_users == 0 or num_items == 0:
        status = "FAIL"
        warnings.append("empty interaction/user/item")

    if num_valid_seqs == 0:
        status = "FAIL"
        warnings.append("no user sequence with length >= 3")

    if len(source_samples) == 0:
        status = "FAIL"
        warnings.append("no source prefix-next training sample")

    if len(target_samples) == 0:
        status = "FAIL"
        warnings.append("no target evaluation sample")

    if not embedding_count_match:
        status = "WARN" if status == "PASS" else status
        warnings.append(
            f"embedding rows mismatch: sem_rows={sem_rows}, expected={num_items + 1}"
        )

    if not timestamp_valid:
        status = "WARN" if status == "PASS" else status
        warnings.append("timestamps have <= 1 unique value; dynamic roles may be invalid")

    if not possible_id_cols:
        status = "WARN" if status == "PASS" else status
        warnings.append(
            "semantic parquet has no explicit item id column; item-embedding alignment relies on row order"
        )

    result = {
        "domain": domain,
        "status": status,
        "interactions": num_interactions,
        "users": num_users,
        "items": num_items,
        "valid_sequences_len>=3": num_valid_seqs,
        "avg_seq_len": round(avg_seq_len, 2),
        "source_prefix_samples": len(source_samples),
        "target_eval_samples": len(target_samples),
        "embedding_shape": tuple(sem.shape),
        "embedding_count_match": embedding_count_match,
        "embedding_id_cols": possible_id_cols,
        "timestamp_unique": timestamp_unique,
        "warnings": warnings,
    }
    return result, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--domains", required=True)
    ap.add_argument("--source", default="amazon_movies_and_tv")
    ap.add_argument("--max_len", type=int, default=50)
    args = ap.parse_args()

    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    all_results = {}
    all_dfs = {}

    for domain in domains:
        result, df = check_domain(args.data_root, domain, args.max_len)
        all_results[domain] = result
        all_dfs[domain] = df

        print("=" * 80)
        print(result)

    print("=" * 80)
    print("Cross-domain overlap check")

    source_df = all_dfs[args.source]
    source_users = set(source_df["UserId"].astype(str))
    source_items = set(source_df["RawItemId"].astype(str))

    for target in domains:
        if target == args.source:
            continue
        target_df = all_dfs[target]
        target_users = set(target_df["UserId"].astype(str))
        target_items = set(target_df["RawItemId"].astype(str))

        user_overlap = len(source_users & target_users)
        item_overlap = len(source_items & target_items)

        print({
            "source": args.source,
            "target": target,
            "user_overlap": user_overlap,
            "item_overlap": item_overlap,
            "zero_shot_warning": user_overlap > 0 or item_overlap > 0,
        })


if __name__ == "__main__":
    main()
