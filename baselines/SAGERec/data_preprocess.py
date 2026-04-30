import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

CONFIG = {
    "data_root": "./data",
    "datasets": ["amazon_movies_and_tv", "amazon_cds_and_vinyl"],  # 替换为你实际的数据集名称
    "min_inter": 10,  # K-core 阈值 (与论文一致)
    "max_seq_len": 50,  # 最大序列长度
}


def load_amazon_interactions(data_dir: Path, domain: str) -> pd.DataFrame:
    """加载 Amazon'23 交互 CSV"""
    filepath = data_dir / f"{domain}.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"未找到交互文件: {filepath}")

    df = pd.read_csv(filepath)
    # 统一列名以适配下游代码
    if "parent_asin" in df.columns:
        df.rename(columns={"parent_asin": "ItemId"}, inplace=True)
    elif "asin" in df.columns:
        df.rename(columns={"asin": "ItemId"}, inplace=True)

    df.rename(columns={"user_id": "UserId", "timestamp": "Timestamp"}, inplace=True)
    df = df[["UserId", "ItemId", "Timestamp"]].dropna()
    df["Timestamp"] = df["Timestamp"].astype(int)
    return df


def load_amazon_metadata(data_dir: Path, domain: str) -> pd.DataFrame:
    """加载 Amazon'23 元数据 JSONL"""
    filepath = data_dir / f"meta_{domain}.jsonl"
    if not filepath.exists():
        raise FileNotFoundError(f"未找到元数据文件: {filepath}")

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {filepath.name}"):
            try:
                records.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    df_meta = pd.DataFrame(records)
    if "parent_asin" in df_meta.columns:
        df_meta.rename(columns={"parent_asin": "ItemId"}, inplace=True)
    elif "asin" in df_meta.columns:
        df_meta.rename(columns={"asin": "ItemId"}, inplace=True)

    # 确保下游 prompt 生成所需的三个核心列存在
    for col in ["title", "description", "features"]:
        if col not in df_meta.columns:
            df_meta[col] = ""

    # 特殊处理 list 类型的字段转成字符串（下游 batch_sem 期待这种格式）
    df_meta['features'] = df_meta['features'].apply(lambda x: str(x) if isinstance(x, list) else x)
    df_meta['description'] = df_meta['description'].apply(lambda x: str(x) if isinstance(x, list) else x)

    return df_meta[["ItemId", "title", "description", "features"]]


def kcore_filter(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """迭代过滤少于 K 次交互的用户和物品"""
    while True:
        prev_len = len(df)
        item_counts = df["ItemId"].value_counts()
        df = df[df["ItemId"].isin(item_counts[item_counts >= k].index)]
        user_counts = df["UserId"].value_counts()
        df = df[df["UserId"].isin(user_counts[user_counts >= k].index)]
        if len(df) == prev_len: break
    return df


def process_domain(domain: str, cfg: dict):
    print(f"\n{'=' * 50}\n[Processing] {domain}\n{'=' * 50}")
    domain_dir = Path(cfg["data_root"]) / domain

    # 1. 加载数据
    df_inter = load_amazon_interactions(domain_dir, domain)
    df_meta = load_amazon_metadata(domain_dir, domain)

    # 2. 第一次 K-core 过滤
    df_inter = kcore_filter(df_inter, k=cfg["min_inter"])

    # 3. 对齐元数据（剔除没有 Meta 信息的 ItemId）
    df_inter = df_inter[df_inter["ItemId"].isin(df_meta["ItemId"])]

    # 4. 第二次 K-core 过滤（因为对齐元数据可能导致分布打破）
    df_inter = kcore_filter(df_inter, k=cfg["min_inter"])

    # 5. 合并 Meta 文本字段到交互记录中 (这是适配 batch_sem.py 的关键)
    df_final = df_inter.merge(df_meta, on="ItemId", how="inner")

    # 6. 按用户和时间戳排序，截断过长的历史序列
    df_final = df_final.sort_values(["UserId", "Timestamp"]).reset_index(drop=True)
    # 去重：同一用户对同一物品的多次交互仅保留第一次
    df_final = df_final.drop_duplicates(subset=["UserId", "ItemId"], keep="first")
    # 截断：保留最近的 max_seq_len 次交互
    df_final = df_final.groupby("UserId", group_keys=False).apply(lambda g: g.tail(cfg["max_seq_len"]))

    # 7. 保存为最终被模型读取的 processed_data.csv
    # 注意：此时的 UserId 和 ItemId 仍是原始的 ASIN 字符串，整数映射由 batch_sem.py 和 rec_datasets.py 接管！
    out_path = domain_dir / "processed_data.csv"
    df_final.to_csv(out_path, index=False)
    print(f"  ✅ Saved to {out_path} ({len(df_final)} rows)")

    return {
        "domain": domain,
        "n_users": df_final["UserId"].nunique(),
        "n_items": df_final["ItemId"].nunique(),
        "n_inter": len(df_final),
        "avg_len": df_final.groupby("UserId").size().mean()
    }


if __name__ == "__main__":
    stats_list = [process_domain(dom, CONFIG) for dom in CONFIG["datasets"]]

    print(f"\n{'=' * 60}\n{'Dataset Statistics':^60}\n{'=' * 60}")
    print(f"{'Domain':<25} {'#Users':>8} {'#Items':>8} {'#Inter':>10} {'AvgLen':>8}")
    print("-" * 60)
    for s in stats_list:
        print(f"{s['domain']:<25} {s['n_users']:>8,} {s['n_items']:>8,} {s['n_inter']:>10,} {s['avg_len']:>8.1f}")
    print("=" * 60)