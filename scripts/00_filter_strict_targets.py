from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


USER_COLS = {"userid", "user_id", "user", "reviewerid", "reviewer_id", "UserId"}
ITEM_COLS = {"itemid", "item_id", "item", "asin", "product_id", "ItemId"}
TIME_COLS = {"timestamp", "time", "unixreviewtime", "unix_review_time", "Timestamp"}


def find_user_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c in USER_COLS or c.lower() in USER_COLS:
            return c
    raise ValueError(f"Cannot find user column. columns={list(df.columns)}")


def find_processed_csv(data_root: Path, domain: str) -> Path:
    candidates = [
        data_root / "processed" / f"{domain}.csv",
        data_root / domain / "processed_data.csv",
        data_root / domain / f"{domain}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find csv for {domain}. Tried: {candidates}")


def filter_target_by_source_users(
    data_root: str,
    source: str,
    targets: list[str],
    out_dir: str = "processed_strict",
) -> None:
    data_root = Path(data_root)
    out_path = data_root / out_dir
    out_path.mkdir(parents=True, exist_ok=True)

    source_path = find_processed_csv(data_root, source)
    source_df = pd.read_csv(source_path)
    source_user_col = find_user_col(source_df)
    source_users = set(source_df[source_user_col].astype(str))

    print("=" * 80)
    print(f"Source: {source}")
    print(f"source_path: {source_path}")
    print(f"source_users: {len(source_users)}")

    for target in targets:
        target_path = find_processed_csv(data_root, target)
        target_df = pd.read_csv(target_path)
        target_user_col = find_user_col(target_df)

        before_interactions = len(target_df)
        before_users = target_df[target_user_col].astype(str).nunique()

        overlap_users = set(target_df[target_user_col].astype(str)) & source_users
        filtered_df = target_df[
            ~target_df[target_user_col].astype(str).isin(source_users)
        ].copy()

        after_interactions = len(filtered_df)
        after_users = filtered_df[target_user_col].astype(str).nunique()

        save_path = out_path / f"{target}.csv"
        filtered_df.to_csv(save_path, index=False)

        print("=" * 80)
        print(f"Target: {target}")
        print(f"target_path: {target_path}")
        print(f"save_path: {save_path}")
        print(f"before_users: {before_users}")
        print(f"overlap_users_removed: {len(overlap_users)}")
        print(f"after_users: {after_users}")
        print(f"before_interactions: {before_interactions}")
        print(f"after_interactions: {after_interactions}")
        print(f"removed_interactions: {before_interactions - after_interactions}")

        if after_users == 0:
            raise RuntimeError(f"All users removed for target={target}. Check UserId format.")
        if len(overlap_users) == 0:
            print("Warning: no overlap users found; maybe already strict.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--source", default="amazon_movies_and_tv")
    parser.add_argument(
        "--targets",
        default="amazon_cds_and_vinyl,amazon_industrial_and_scientific",
    )
    parser.add_argument("--out_dir", default="processed_strict")
    args = parser.parse_args()

    targets = [x.strip() for x in args.targets.split(",") if x.strip()]
    filter_target_by_source_users(
        data_root=args.data_root,
        source=args.source,
        targets=targets,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
