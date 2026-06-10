# ISDDG Initial Codebase

This is an initial, refactor-friendly codebase for **ISDDG: Item-Sequence Dual-level Dynamic Generalization Framework** for zero-shot cross-domain sequential recommendation.

It is designed to replace the earlier `diagnostic/` style scripts with a cleaner structure:

- `isddg/data`: data loading, sequence construction, negative sampling
- `isddg/features`: semantic embedding loading, causal dynamic stats, GMM soft roles
- `isddg/models`: BERT4Rec-style feature backbone, role predictor, ISDDG scorer
- `isddg/prototypes`: source-domain key-value prototype bank
- `isddg/training`: losses and simple trainers
- `isddg/evaluation`: tie-aware sampled evaluation
- `scripts`: runnable entry points for the first round of experiments

## Zero-shot Boundary

Main ISDDG experiments should **not** use target-domain aggregate statistics, target-domain training losses, or target-domain parameter updates. Target sequences are used only as online query prefixes during evaluation.

## Expected data layout

Compatible with the existing DPTF diagnostic repository:

```text
data/
├── processed/
│   ├── amazon_movies_and_tv.csv
│   ├── amazon_cds_and_vinyl.csv
│   └── steam.csv
└── semantic_embeddings/
    ├── amazon_movies_and_tv_embedding_llama.parquet
    ├── amazon_cds_and_vinyl_embedding_llama.parquet
    └── steam_embedding_llama.parquet
```

Interaction CSV columns are normalized from variants such as:

- user: `UserId`, `user_id`, `reviewerID`, ...
- item: `ItemId`, `item_id`, `asin`, ...
- time: `Timestamp`, `unixReviewTime`, ...

Semantic parquet should contain one vector column, usually `item_text_embedding`.

## Quick start

```bash
pip install -r requirements.txt

python scripts/00_check_data.py \
  --data_root ./data \
  --domains amazon_movies_and_tv,amazon_cds_and_vinyl,steam

python scripts/01_build_dynamic_roles.py \
  --data_root ./data \
  --source amazon_movies_and_tv \
  --K 4 \
  --out_dir artifacts/roles

python scripts/02_train_semantic.py \
  --config configs/isddg_initial.yaml

python scripts/03_build_prototypes.py \
  --config configs/isddg_initial.yaml \
  --checkpoint artifacts/checkpoints/semantic_only.pt

python scripts/04_train_isddg.py \
  --config configs/isddg_initial.yaml \
  --semantic_checkpoint artifacts/checkpoints/semantic_only.pt \
  --prototype_path artifacts/prototypes/prototype_bank.pt

python scripts/05_eval_zero_shot.py \
  --config configs/isddg_initial.yaml \
  --checkpoint artifacts/checkpoints/isddg.pt \
  --targets amazon_cds_and_vinyl,steam
```

## Important notes

1. The implementation is intentionally modular and conservative. It is meant for initial experiments and later refinement.
2. Strong baselines such as LLM-RecG and SAGERec are represented as integration placeholders. You should plug in the exact baseline code once you freeze their official implementation choices.
3. The evaluator shuffles candidate lists and reports a tie ratio to avoid the old diagnostic issue where equal scores can produce artificially high metrics.
