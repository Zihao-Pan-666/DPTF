# -*- coding: utf-8 -*-
import os
import argparse
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from llm2vec import LLM2Vec
from rec_datasets import AmazonUserSequencesDataset

# ---- Hard offline mode (no HF network) ----
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def initialize_dataset(dataset_saved_path: Path, max_len: int):
    data = pd.read_csv(dataset_saved_path / "processed_data.csv")
    return AmazonUserSequencesDataset(data, max_len)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--dataset_name', type=str, default='amazon_industrial_and_scientific',
                        help='amazon_musical_instruments/amazon_industrial_and_scientific/amazon_video_games/steam')

    parser.add_argument('--base_model_path', type=str, default="llm/llama-3-8B-Instruct",
                        help='Local path to Llama-3-8B-Instruct folder (config.json + model shards + model.safetensors.index.json + tokenizer.json)')
    parser.add_argument('--adapter_path', type=str, default="llm/llm2vec-llama-3-8B-Instruct-mntp",
                        help='Local path to LLM2Vec adapter folder (adapter_model.safetensors + adapter_config.json + modeling_llama_encoder.py etc.)')

    parser.add_argument('--batch_size_encode', type=int, default=128)
    parser.add_argument('--max_text_len', type=int, default=256)
    parser.add_argument('--output_name', type=str, default=None,
                        help='Output parquet filename (optional). Default: <dataset>_embedding_llm2vec.parquet')
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).resolve().parent

    # 自动向上找项目根目录：以同时包含 data/ 和 llm/ 为准
    repo_root = script_dir
    for _ in range(3):  # 向上最多找3层足够了
        if (repo_root / "data").is_dir() and (repo_root / "llm").is_dir():
            break
        repo_root = repo_root.parent
    else:
        raise RuntimeError(f"Cannot locate repo root from {script_dir}. Expect 'data/' and 'llm/' under repo root.")

    dataset_dir = repo_root / "data" / args.dataset_name
    base_model_path = Path(args.base_model_path).resolve()
    adapter_path = Path(args.adapter_path).resolve()

    if not (dataset_dir / "processed_data.csv").exists():
        raise FileNotFoundError(f"processed_data.csv not found: {dataset_dir / 'processed_data.csv'}")

    if not base_model_path.is_dir():
        raise FileNotFoundError(f"base_model_path not found: {base_model_path}")
    if not adapter_path.is_dir():
        raise FileNotFoundError(f"adapter_path not found: {adapter_path}")

    # Base model sanity checks (your base is instruct + json tokenizer)
    must_base = ["config.json", "model.safetensors.index.json"]
    for f in must_base:
        if not (base_model_path / f).exists():
            raise FileNotFoundError(f"Missing in base_model_path: {base_model_path / f}")

    # Adapter sanity checks (from your screenshot list)
    must_adapter = ["adapter_config.json", "adapter_model.safetensors", "config.json"]
    for f in must_adapter:
        if not (adapter_path / f).exists():
            raise FileNotFoundError(f"Missing in adapter_path: {adapter_path / f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading LLM2Vec (LOCAL/OFFLINE)...")
    print(f"  device={device}")
    print(f"  base_model_path={base_model_path}")
    print(f"  adapter_path={adapter_path}")

    l2v = None
    load_errors = []

    # 注意：llm2vec==0.2.2 的实现里，from_pretrained 可能不接受 device
    # 我们先不传 device，加载成功后再手动迁移到 GPU

    # Attempt A: 常见参数名：base_model_name_or_path + peft_model_name_or_path
    try:
        l2v = LLM2Vec.from_pretrained(
            base_model_name_or_path=str(base_model_path),
            peft_model_name_or_path=str(adapter_path),
            local_files_only=True
        )
    except Exception as e:
        load_errors.append(("AttemptA", repr(e)))

    # Attempt B: 有些版本第一个参数就是 base path
    if l2v is None:
        try:
            l2v = LLM2Vec.from_pretrained(
                str(base_model_path),
                peft_model_name_or_path=str(adapter_path),
                local_files_only=True
            )
        except Exception as e:
            load_errors.append(("AttemptB", repr(e)))

    # Attempt C: 有些版本参数名是 adapter_model_name_or_path
    if l2v is None:
        try:
            l2v = LLM2Vec.from_pretrained(
                str(base_model_path),
                adapter_model_name_or_path=str(adapter_path),
                local_files_only=True
            )
        except Exception as e:
            load_errors.append(("AttemptC", repr(e)))

    if l2v is None:
        print("Failed to load LLM2Vec with all attempts:")
        for tag, err in load_errors:
            print(f"  {tag}: {err}")
        raise RuntimeError("LLM2Vec load failed. See errors above.")

    # ✅ 关键：加载成功后，再迁移 device（而不是在构造时传 device）
    try:
        # 有些 llm2vec 对象暴露 model/tokenizer
        if hasattr(l2v, "model") and isinstance(l2v.model, torch.nn.Module):
            l2v.model.to(device)
        elif isinstance(l2v, torch.nn.Module):
            l2v.to(device)
    except Exception as e:
        print("Warning: move to device failed, will rely on internal device handling:", e)

    # ✅ 可选：确保 tokenizer pad_token
    try:
        tok = getattr(l2v, "tokenizer", None)
        if tok is not None and tok.pad_token is None:
            tok.pad_token = tok.eos_token
    except Exception:
        pass

    # ---- Load dataset & build prompts ----
    dataset = initialize_dataset(dataset_dir, args.max_len)
    df = pd.DataFrame(dataset.data_frame)
    df = df[['ItemId', 'title', 'description', 'features']]
    df_dedup = df.drop_duplicates(subset='ItemId', keep='first').copy()

    def _unwrap_list_string(x: str) -> str:
        """
        原repo默认 features/description 可能是类似 "['a', 'b']" 的字符串；
        只有当它真的长得像 [ ... ] 时才去掉最外层括号。
        否则（比如 Steam 的日语长文本）绝不能做 [1:-1] 截断。
        """
        if not isinstance(x, str):
            return ""
        s = x.strip()
        if len(s) >= 2 and s[0] == "[" and s[-1] == "]":
            return s[1:-1].strip()
        return s

    def generate_prompt(row):
        title = row.get("title", "")
        features_raw = row.get("features", "")
        description_raw = row.get("description", "")

        features = _unwrap_list_string(features_raw) if isinstance(features_raw, str) and features_raw.strip() else ""
        description = _unwrap_list_string(description_raw) if isinstance(description_raw,
                                                                         str) and description_raw.strip() else ""

        if not features:
            features = "no feature provided"
        if not description:
            description = "no description provided"

        # ✅保持原 repo 的 prompt 结构不变（最大化可比性）
        return (
            f"Please summarize the following item based on the provided information: title: {title}.\n"
            f" Feature: {features}.\n"
            f" Description: {description}"
        )

    df_dedup['prompt'] = df_dedup.apply(generate_prompt, axis=1)
    prompts = df_dedup['prompt'].tolist()

    # ---- Encode with progress bar ----
    print(f"Encoding {len(prompts)} items... (batch={args.batch_size_encode})")
    embeddings = []

    for i in tqdm(range(0, len(prompts), args.batch_size_encode), desc="Encoding items"):
        batch = prompts[i:i + args.batch_size_encode]
        with torch.no_grad():
            emb = l2v.encode(batch, show_progress_bar=False)   # torch.Tensor [B, H]
        embeddings.extend(emb.detach().cpu().float().numpy().tolist())

    assert len(embeddings) == len(df_dedup), "Embedding count mismatch!"

    df_out = df_dedup[['ItemId', 'prompt']].copy()
    df_out['item_text_embedding'] = embeddings

    out_name = args.output_name or f"{args.dataset_name}_embedding_llama.parquet"
    out_path = dataset_dir / out_name
    df_out.to_parquet(out_path, compression='snappy')

    # ---- Minimal sanity check ----
    import numpy as np
    v = np.array(df_out.iloc[0]['item_text_embedding'], dtype=np.float32)
    print(f"Saved to: {out_path}")
    print(f"Example embedding dim={v.shape[0]} mean={v.mean():.6f} std={v.std():.6f} nan={np.isnan(v).any()} inf={np.isinf(v).any()}")


if __name__ == "__main__":
    main()
