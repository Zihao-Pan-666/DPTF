Patch notes for scripts/00_deep_data_quality_check.py

1) Add CLI argument:
    parser.add_argument("--embedding_dir", default="semantic_embeddings")

2) Pass embedding_dir into check_one_domain and check_embedding_coverage_and_values.

3) In check_embedding_coverage_and_values, replace:
    emb_path = find_embedding_parquet(data_root, domain)
    aligned_sem = load_semantic_embeddings(data_root, domain, item_map=item_map)
with:
    emb_path = find_embedding_parquet(data_root, domain, embedding_dir=embedding_dir)
    aligned_sem = load_semantic_embeddings(
        data_root, domain, item_map=item_map, embedding_dir=embedding_dir, strict=True
    )

4) Immediately after:
    id_col, emb_col = detect_embedding_columns(columns)
add:
    if id_col is None:
        raise ValueError(
            f"{domain}: embedding parquet must contain RawItemId/ItemId/asin. "
            f"Row-order alignment is forbidden for formal data freeze."
        )

5) Add hard failures, not only warnings:
    - embedding_id_col is None
    - missing_used_items > 0
    - aligned_embedding_shape_ok is False
    - nan_ratio > 0
    - inf_ratio > 0
    - zero_vector_ratio > 0
    - source-target user_overlap > 0
    - source-target item_overlap > 0

6) Recommended final check:
    python scripts/00_deep_data_quality_check.py ^
      --data_root ./data/frozen/isddg_v1 ^
      --source amazon_movies_and_tv ^
      --targets amazon_cds_and_vinyl,amazon_industrial_and_scientific ^
      --embedding_dir semantic_embeddings ^
      --out_dir ./data/frozen/isddg_v1/reports/data_quality
