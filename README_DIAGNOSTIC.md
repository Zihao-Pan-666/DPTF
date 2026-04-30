# ZCDSR Diagnostic Minimal Code

Copy the `diagnostic/` folder and `run_diagnostic.py` into your project root `LLM4ZCDSR/`.

This minimal framework is for the 1-week diagnostic experiment:
- semantic-only baseline
- preprec-core dynamics-only baseline
- naive fusion
- gated fusion

Expected project root layout:

```text
LLM4ZCDSR/
  data/
    processed/
      amazon_movies_and_tv.csv
      amazon_cds_and_vinyl.csv
      steam.csv
    semantic_embeddings/
      amazon_movies_and_tv_embedding_llama.parquet
      amazon_cds_and_vinyl_embedding_llama.parquet
      steam_embedding_llama.parquet
    popularity_features/        # auto-created
  diagnostic/
  run_diagnostic.py
```

Examples:

```bash
python run_diagnostic.py --model semantic --source amazon_movies_and_tv --targets amazon_cds_and_vinyl,steam --epochs 20
python run_diagnostic.py --model dynamics --source amazon_movies_and_tv --targets amazon_cds_and_vinyl,steam --epochs 20
python run_diagnostic.py --model naive_fusion --source amazon_movies_and_tv --targets amazon_cds_and_vinyl,steam --epochs 20
python run_diagnostic.py --model gated_fusion --source amazon_movies_and_tv --targets amazon_cds_and_vinyl,steam --epochs 20
```

Outputs are saved to `outputs/diagnostic_results.csv`.

Notes:
- `dynamics` here is a lightweight PrepRec-core approximation: popularity percentile + coarse/fine trend + relative time embedding.
- It is deliberately independent of semantic embeddings in the first stage.
- This is not a faithful full PrepRec reproduction; it is designed to reveal whether popularity dynamics supplies an independent transferable signal.
