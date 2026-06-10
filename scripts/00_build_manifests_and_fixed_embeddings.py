# scripts/00a_build_manifests_and_fixed_embeddings.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, ast, hashlib, json, math, shutil
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd

USER_COLS = {"userid", "user_id", "user", "reviewerid", "reviewer_id"}
ITEM_COLS = {"itemid", "item_id", "item", "asin", "product_id", "rawitemid", "raw_item_id"}
TIME_COLS = {"timestamp", "time", "unixreviewtime", "unix_review_time"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        lc = str(c).lower()
        if lc in USER_COLS:
            rename[c] = "UserId"
        elif lc in ITEM_COLS:
            rename[c] = "ItemId"
        elif lc in TIME_COLS:
            rename[c] = "Timestamp"
    return df.rename(columns=rename)


def read_csv_canonical(path: Path) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))
    missing = {"UserId", "ItemId", "Timestamp"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing {missing}; columns={list(df.columns)}")
    df["UserId"] = df["UserId"].astype(str)
    df["ItemId"] = df["ItemId"].astype(str)
    df["Timestamp"] = pd.to_numeric(df["Timestamp"], errors="coerce").fillna(0).astype(float)
    for c in ["rating", "title", "description", "features", "products", "text", "prompt"]:
        if c not in df.columns:
            df[c] = ""
    return df


def find_old_csv(data_root: Path, old_dir: str, domain: str) -> Path:
    candidates = [data_root / old_dir / f"{domain}.csv", data_root / domain / "processed_data.csv"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"old csv not found for {domain}: {candidates}")


def find_new_csv(data_root: Path, strict_dir: str, old_dir: str, domain: str, source: str) -> Path:
    candidates = []
    if domain != source:
        candidates.append(data_root / strict_dir / f"{domain}.csv")
    candidates += [data_root / old_dir / f"{domain}.csv", data_root / domain / "processed_data.csv"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"new csv not found for {domain}: {candidates}")


def find_old_pq(data_root: Path, emb_dir: str, domain: str) -> Path:
    candidates = [
        data_root / emb_dir / f"{domain}_embedding_llama.parquet",
        data_root / emb_dir / f"{domain}_embedding_llama3.parquet",
        data_root / emb_dir / f"{domain}_embedding.parquet",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"old parquet not found for {domain}: {candidates}")


def embedding_col(df: pd.DataFrame) -> str:
    if "item_text_embedding" in df.columns:
        return "item_text_embedding"
    cands = [c for c in df.columns if "embedding" in str(c).lower()]
    if not cands:
        raise ValueError(f"No embedding column in parquet; columns={list(df.columns)}")
    return cands[0]


def parse_vec(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def build_old_manifest_like_batch_sem_llm(csv_path: Path, domain: str) -> pd.DataFrame:
    """
    Reproduce uploaded batch_sem_LLM.py mapping:
      Amazon: raw ItemId first-appearance order in the original CSV -> 1..N
      Steam:  raw ItemId first-appearance order in the original CSV -> 0..N-1
    """
    df = read_csv_canonical(csv_path)
    raw_unique = pd.Series(df["ItemId"].dropna().astype(str).unique(), name="RawItemId")
    start = 0 if "steam" in domain.lower() else 1
    old_ids = np.arange(start, start + len(raw_unique), dtype=int)
    out = pd.DataFrame({"OldEmbeddingItemId": old_ids, "RawItemId": raw_unique.astype(str)})
    meta_cols = ["title", "description", "features", "rating", "products", "text"]
    meta = df.drop_duplicates("ItemId", keep="first")[["ItemId"] + meta_cols].rename(columns={"ItemId": "RawItemId"})
    meta["RawItemId"] = meta["RawItemId"].astype(str)
    out = out.merge(meta, on="RawItemId", how="left", validate="one_to_one")
    out.insert(0, "domain", domain)
    return out


def build_new_manifest_like_isddg_loader(csv_path: Path, domain: str) -> pd.DataFrame:
    """
    Match current isddg.data.io.load_interactions:
      sort by UserId, Timestamp; then first-appearance RawItemId -> 1..N.
    """
    df = read_csv_canonical(csv_path).sort_values(["UserId", "Timestamp"]).reset_index(drop=True)
    seen: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        raw = str(r["ItemId"])
        if raw in seen:
            continue
        cid = len(seen) + 1
        seen[raw] = cid
        rows.append({
            "domain": domain,
            "CanonicalItemId": cid,
            "RawItemId": raw,
            "title": r.get("title", ""),
            "description": r.get("description", ""),
            "features": r.get("features", ""),
            "rating": r.get("rating", ""),
            "products": r.get("products", ""),
            "text": r.get("text", ""),
        })
    return pd.DataFrame(rows)


def old_embedding_with_raw_ids(old_pq: Path, old_manifest: pd.DataFrame) -> pd.DataFrame:
    emb = pd.read_parquet(old_pq).copy()
    col = embedding_col(emb)
    if col != "item_text_embedding":
        emb = emb.rename(columns={col: "item_text_embedding"})

    if "ItemId" in emb.columns:
        emb["OldEmbeddingItemId"] = pd.to_numeric(emb["ItemId"], errors="raise").astype(int)
        emb = emb.merge(
            old_manifest[["OldEmbeddingItemId", "RawItemId"]],
            on="OldEmbeddingItemId",
            how="left",
            validate="one_to_one",
        )
    else:
        if len(emb) != len(old_manifest):
            raise ValueError(f"{old_pq}: no ItemId column and row count mismatch.")
        emb["OldEmbeddingItemId"] = old_manifest["OldEmbeddingItemId"].to_numpy()
        emb["RawItemId"] = old_manifest["RawItemId"].astype(str).to_numpy()

    if emb["RawItemId"].isna().any():
        bad = emb.loc[emb["RawItemId"].isna(), "OldEmbeddingItemId"].head(20).tolist()
        raise ValueError(f"{old_pq}: cannot map some old ItemId to RawItemId: {bad}")
    emb["RawItemId"] = emb["RawItemId"].astype(str)
    return emb


def make_fixed_parquet(new_manifest: pd.DataFrame, old_emb: pd.DataFrame, strict_numeric=True):
    cols = ["RawItemId", "OldEmbeddingItemId", "item_text_embedding"]
    if "prompt" in old_emb.columns:
        cols.append("prompt")
    small = old_emb[cols].copy()

    if small.duplicated("RawItemId").any():
        samples = small.loc[small.duplicated("RawItemId", keep=False), "RawItemId"].head(20).tolist()
        raise ValueError(f"duplicated RawItemId in old embedding: {samples}")

    fixed = new_manifest.merge(small, on="RawItemId", how="left", validate="one_to_one")
    miss = fixed["item_text_embedding"].isna().sum()
    if miss:
        samples = fixed.loc[fixed["item_text_embedding"].isna(), "RawItemId"].head(20).tolist()
        raise ValueError(f"{miss} current items missing embedding. samples={samples}")

    vecs, dims, norms = [], [], []
    nonfinite = 0
    zero = 0
    for raw, x in zip(fixed["RawItemId"], fixed["item_text_embedding"]):
        v = parse_vec(x)
        if v.ndim != 1:
            raise ValueError(f"{raw}: vector shape is {v.shape}, expected 1-D")
        dims.append(v.shape[0])
        if not np.isfinite(v).all():
            nonfinite += 1
            if strict_numeric:
                raise ValueError(f"{raw}: NaN/Inf in embedding")
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        n = float(np.linalg.norm(v))
        if n == 0:
            zero += 1
        norms.append(n)
        vecs.append(v.astype(np.float32).tolist())

    if len(set(dims)) != 1:
        raise ValueError(f"Inconsistent embedding dims: {sorted(set(dims))}")
    fixed["item_text_embedding"] = vecs

    for c in ["OldEmbeddingItemId", "prompt", "rating", "products", "text"]:
        if c not in fixed.columns:
            fixed[c] = ""

    fixed = fixed[[
        "domain", "CanonicalItemId", "RawItemId", "OldEmbeddingItemId",
        "title", "description", "features", "rating", "products", "text",
        "prompt", "item_text_embedding",
    ]].sort_values("CanonicalItemId").reset_index(drop=True)

    report = {
        "items_fixed": int(len(fixed)),
        "embedding_dim": int(dims[0]),
        "nonfinite_rows": int(nonfinite),
        "zero_rows": int(zero),
        "norm_min": float(np.min(norms)),
        "norm_mean": float(np.mean(norms)),
        "norm_max": float(np.max(norms)),
    }
    return fixed, report


def copy_to_freeze(args, reports):
    if not args.freeze_dir:
        return
    root = Path(args.freeze_dir)
    for sub in ["processed", "semantic_embeddings", "item_manifests", "reports"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

    records = []
    for r in reports:
        d = r["domain"]
        files = [
            (Path(r["new_csv"]), root / "processed" / f"{d}.csv", "processed_csv"),
            (Path(r["fixed_parquet"]), root / "semantic_embeddings" / f"{d}_embedding_llama_fixed.parquet", "fixed_parquet"),
            (Path(r["new_manifest"]), root / "item_manifests" / f"{d}_item_manifest.csv", "item_manifest"),
        ]
        for src, dst, typ in files:
            shutil.copy2(src, dst)
            records.append({"domain": d, "type": typ, "path": str(dst), "sha256": sha256_file(dst)})

    manifest = {"source_domain": args.source_domain, "records": records, "source_reports": reports}
    (root / "reports" / "freeze_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(records).to_csv(root / "reports" / "freeze_manifest.csv", index=False, encoding="utf-8-sig")


def process_domain(args, domain: str):
    data_root = Path(args.data_root)
    old_csv = find_old_csv(data_root, args.old_processed_dir, domain)
    new_csv = find_new_csv(data_root, args.strict_processed_dir, args.old_processed_dir, domain, args.source_domain)
    old_pq = find_old_pq(data_root, args.old_embedding_dir, domain)

    man_dir = data_root / args.out_manifest_dir
    emb_dir = data_root / args.out_embedding_dir
    rep_dir = data_root / args.out_report_dir
    for p in [man_dir, emb_dir, rep_dir]:
        p.mkdir(parents=True, exist_ok=True)

    old_manifest = build_old_manifest_like_batch_sem_llm(old_csv, domain)
    new_manifest = build_new_manifest_like_isddg_loader(new_csv, domain)
    old_emb = old_embedding_with_raw_ids(old_pq, old_manifest)
    fixed, extra = make_fixed_parquet(new_manifest, old_emb, strict_numeric=not args.allow_nonfinite_fix)

    old_manifest_path = man_dir / f"{domain}_old_llm_item_manifest.csv"
    new_manifest_path = man_dir / f"{domain}_item_manifest.csv"
    fixed_path = emb_dir / f"{domain}_embedding_llama_fixed.parquet"
    report_path = rep_dir / f"{domain}_embedding_fix_report.json"

    if not args.overwrite:
        for p in [old_manifest_path, new_manifest_path, fixed_path, report_path]:
            if p.exists():
                raise FileExistsError(f"{p} exists. Use --overwrite.")

    old_manifest.to_csv(old_manifest_path, index=False, encoding="utf-8-sig")
    new_manifest.to_csv(new_manifest_path, index=False, encoding="utf-8-sig")
    fixed.to_parquet(fixed_path, index=False, compression="snappy")

    old_set = set(old_manifest["RawItemId"].astype(str))
    new_set = set(new_manifest["RawItemId"].astype(str))
    report = {
        "domain": domain,
        "old_csv": str(old_csv),
        "new_csv": str(new_csv),
        "old_parquet": str(old_pq),
        "old_manifest": str(old_manifest_path),
        "new_manifest": str(new_manifest_path),
        "fixed_parquet": str(fixed_path),
        "old_items": int(len(old_manifest)),
        "new_items": int(len(new_manifest)),
        "dropped_items_after_strict_filter": int(len(old_set - new_set)),
        "dropped_item_samples": sorted(old_set - new_set)[:30],
        "new_items_not_in_old_embedding_source": int(len(new_set - old_set)),
        "new_item_samples": sorted(new_set - old_set)[:30],
        **extra,
    }
    report["sha256"] = {
        "old_csv": sha256_file(old_csv),
        "new_csv": sha256_file(new_csv),
        "old_parquet": sha256_file(old_pq),
        "old_manifest": sha256_file(old_manifest_path),
        "new_manifest": sha256_file(new_manifest_path),
        "fixed_parquet": sha256_file(fixed_path),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--domains", default="amazon_movies_and_tv,amazon_cds_and_vinyl,amazon_industrial_and_scientific")
    ap.add_argument("--source_domain", default="amazon_movies_and_tv")
    ap.add_argument("--old_processed_dir", default="processed")
    ap.add_argument("--strict_processed_dir", default="processed_strict")
    ap.add_argument("--old_embedding_dir", default="semantic_embeddings")
    ap.add_argument("--out_manifest_dir", default="item_manifests")
    ap.add_argument("--out_embedding_dir", default="semantic_embeddings_fixed")
    ap.add_argument("--out_report_dir", default="embedding_fix_reports")
    ap.add_argument("--allow_nonfinite_fix", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--freeze_dir", default="")
    args = ap.parse_args()

    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    reports = []
    for d in domains:
        print("=" * 100)
        print(f"Processing {d}")
        r = process_domain(args, d)
        reports.append(r)
        print(json.dumps({k: r[k] for k in [
            "domain", "old_items", "new_items", "dropped_items_after_strict_filter",
            "new_items_not_in_old_embedding_source", "items_fixed", "embedding_dim",
            "zero_rows", "fixed_parquet"
        ]}, indent=2, ensure_ascii=False))

    rep_dir = Path(args.data_root) / args.out_report_dir
    (rep_dir / "all_embedding_fix_reports.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(reports).to_csv(rep_dir / "all_embedding_fix_reports.csv", index=False, encoding="utf-8-sig")
    copy_to_freeze(args, reports)
    print("=" * 100)
    print(f"Done. Reports: {rep_dir}")
    if args.freeze_dir:
        print(f"Frozen data: {args.freeze_dir}")


if __name__ == "__main__":
    main()
