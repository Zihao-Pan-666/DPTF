import os
import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from llm2vec import LLM2Vec


# ==========================================
# 原 Repo 数据集 ID 映射逻辑封装
# ==========================================

class SteamDatasetRemapper:
    def __init__(self, data):
        # Steam 原 repo 逻辑：ItemId 从 0 开始映射
        unique_item_ids = data['ItemId'].unique()
        self.item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_item_ids, start=0)}


class AmazonDatasetRemapper:
    def __init__(self, data):
        # Amazon 原 repo 逻辑：经历两次映射，最终从 1 开始（0 留作 Padding）
        unique_item_ids = data['ItemId'].unique()
        self.item_id_map_stage1 = {old_id: new_id for new_id, old_id in enumerate(unique_item_ids, start=0)}
        temp_ids = data['ItemId'].map(self.item_id_map_stage1)

        unique_item_ids_v2 = temp_ids.unique()
        self.item_id_map_stage2 = {old_id: new_id for new_id, old_id in enumerate(unique_item_ids_v2, start=1)}

        # 合并映射关系：原始 ASIN -> 阶段1 ID -> 阶段2 ID (1..N)
        self.item_id_map = {old: self.item_id_map_stage2[self.item_id_map_stage1[old]] for old in
                            self.item_id_map_stage1}


# ==========================================
# 原 Repo Prompt 构造逻辑还原
# ==========================================

def _unwrap_list_string(x):
    """还原原 repo 处理 Amazon 字段 [1:-1] 的逻辑"""
    if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
        return x[1:-1].strip()
    return str(x)


def generate_prompt_amazon(row):
    """严格还原 Amazon 字符级模板（注意点号与空格位置）"""
    title = str(row.get('title', '')).strip()
    features = _unwrap_list_string(row.get('features', ''))
    description = _unwrap_list_string(row.get('description', ''))

    if not features or features == "nan": features = "no feature provided"
    if not description or description == "nan": description = "no description provided"

    return f"Please summarize the following item based on the provided information: title: {title}.\n Feature: {features}.\n Description: {description}"


def generate_prompt_steam(row):
    """基于 Paper 定义构造 Steam 专属模板"""
    prod = str(row.get("products", "")).strip()
    txt = str(row.get("text", "")).strip()
    return f"Please summarize the following game item based on the provided information:\nProduct: {prod}\nReview: {txt}"


# ==========================================
# 批量执行主脚本
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str,
                        default="amazon_movies_and_tv,amazon_cds_and_vinyl")
    parser.add_argument("--base_model_path", type=str, default="llm/llama-3-8B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="llm/llm2vec-llama-3-8B-Instruct-mntp")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    # 强制离线模式
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading LLM2Vec model from local paths on {device}...")

    l2v = LLM2Vec.from_pretrained(
        args.base_model_path,
        peft_model_name_or_path=args.adapter_path,
        local_files_only=True
    )
    if hasattr(l2v, "model"):
        l2v.model.to(device)

    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for ds_name in dataset_list:
        data_dir = Path(f"./data/{ds_name}")
        csv_path = data_dir / "processed_data.csv"
        if not csv_path.exists():
            print(f" [Skip] {ds_name}: File not found at {csv_path}")
            continue

        print(f"\n>>> Processing Dataset: {ds_name}")
        df = pd.read_csv(csv_path)

        # 1. 严格按域应用 ID 映射
        if "steam" in ds_name.lower():
            remapper = SteamDatasetRemapper(df)
            df['ItemId'] = df['ItemId'].map(remapper.item_id_map)
            prompt_func = generate_prompt_steam
        else:
            remapper = AmazonDatasetRemapper(df)
            df['ItemId'] = df['ItemId'].map(remapper.item_id_map)
            prompt_func = generate_prompt_amazon

        # 2. 构造去重后的 Item 表并生成 Prompt
        df_items = df.drop_duplicates(subset="ItemId", keep="first").copy()
        df_items['prompt'] = df_items.apply(prompt_func, axis=1)

        # 3. 批量编码
        prompts = df_items['prompt'].tolist()
        all_embeddings = []
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=f" Encoding {ds_name}"):
            batch = prompts[i:i + args.batch_size]
            with torch.no_grad():
                emb = l2v.encode(batch, show_progress_bar=False)
            all_embeddings.extend(emb.cpu().numpy().tolist())

        df_items['item_text_embedding'] = all_embeddings

        # 4. 保存为 Parquet 文件
        out_path = data_dir / f"{ds_name}_embedding_llama.parquet"
        df_items[['ItemId', 'prompt', 'item_text_embedding']].to_parquet(out_path, compression='snappy')
        print(f" [OK] Successfully saved: {out_path}")


if __name__ == "__main__":
    main()