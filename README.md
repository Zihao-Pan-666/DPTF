# ISDDG Mainline: Semantic-conditioned Continuous Dynamic Generalization

This repository is organized for the current ISDDG mainline:

```text
Semantic item text embedding
        ↓
strong semantic BERT4Rec source model
        ↓
source-selected semantic score

source causal dynamic observations
        ↓
continuous dynamic feature table
        ↓
semantic-conditioned continuous dynamic prior
        ↓
candidate-aware dynamic score

final score = semantic_score + beta * dynamic_score
```

## Current method position

The mainline has been narrowed from the earlier broad prototype exploration into a cleaner, paper-oriented route:

1. **Strong semantic baseline**  
   Train a BERT4Rec-style sequential model on the source domain using item text embeddings.

2. **Semantic-conditioned continuous dynamic prior**  
   Learn `item_text_embedding -> continuous_dynamic_vector` using source-domain causal dynamic observations.

3. **Discrete dynamic role as auxiliary and explanation**  
   The old K=4 soft role table is retained as structured auxiliary supervision and diagnostic interpretation, not as the main performance vector.

4. **Late fusion for zero-shot ranking**  
   Keep the semantic sequence model and dynamic prior separate until scoring:
   `final_score = semantic_score + beta * dynamic_score`.

5. **Strict zero-shot evaluation**  
   Target-domain item text embeddings may be used at inference and for transductive item-side semantic availability, but target-domain interactions, target-domain dynamic statistics, and target-domain metrics are not used for training, early stopping, beta selection, or checkpoint selection.

## Expected data layout

The code follows the existing ISDDG data loaders:

```text
data/
├── processed/
│   ├── amazon_movies_and_tv.csv
│   ├── amazon_cds_and_vinyl.csv
│   └── amazon_industrial_and_scientific.csv        # optional
└── semantic_embeddings/
    ├── amazon_movies_and_tv_embedding_llama_fixed.parquet
    ├── amazon_cds_and_vinyl_embedding_llama_fixed.parquet
    └── amazon_industrial_and_scientific_embedding_llama_fixed.parquet
```

The embedding loader also searches common fallback names such as `*_embedding_llama.parquet`, but fixed parquet files are preferred.

## Clean output layout

```text
artifacts/
├── roles/
│   └── amazon_movies_and_tv_k4_default/
│       ├── source_role_observations.parquet
│       ├── source_role_table.pt
│       ├── role_centroids.csv
│       └── role_diagnostics.json
├── dynamics/
│   └── amazon_movies_and_tv_continuous/
│       ├── source_continuous_dynamic_table.pt
│       ├── continuous_dynamic_stats.json
│       └── pred_source_continuous_dynamic_table_amazon_movies_and_tv_seed2026.pt
└── checkpoints/
    ├── semantic_final/
    │   └── semantic_amazon_movies_and_tv_seed2026.pt
    └── continuous_dynamic/
        └── semantic_conditioned_prior_enhanced_amazon_movies_and_tv_seed2026.pt

results/
└── mainline/
    └── amazon_movies_and_tv_to_amazon_cds_and_vinyl/
        ├── semantic_zero_shot.csv
        ├── continuous_dynamic_prior_source_val.csv
        ├── continuous_late_fusion_predicted_source_val.csv
        └── continuous_late_fusion_zero_shot.csv
```

## Main workflow

### 0. Check paths

```bash
python scripts/98_check_mainline_paths.py \
  --config configs/continuous_dynamic.yaml \
  --stage before_roles
```

### 1. Build source dynamic observations and soft roles

```bash
python scripts/01_build_dynamic_roles.py \
  --config configs/dynamic_roles.yaml
```

### 2. Train or re-train the strong semantic baseline

```bash
python scripts/02_train_semantic.py \
  --config configs/semantic_final.yaml
```

Then evaluate target zero-shot semantic-only:

```bash
python scripts/05_eval_semantic_zero_shot.py \
  --config configs/semantic_final.yaml \
  --targets amazon_cds_and_vinyl
```

If you are doing quick feasibility only, you may temporarily set `semantic.checkpoint` in
`configs/continuous_dynamic.yaml` to an existing source-selected semantic checkpoint, such as an alpha0-control checkpoint.
Do not use this temporary checkpoint as the final paper baseline unless its training protocol is documented.

### 3. Train enhanced continuous dynamic prior

```bash
python scripts/02_train_continuous_dynamic_prior.py \
  --config configs/continuous_dynamic.yaml
```

Watch these source-side indicators:

```text
val_reg_mse / val_reg_mae
rank_Recall@10
rank_NDCG@10
rank_MRR@10
rank_tie_case_ratio or tie_case_ratio
best_epoch
selection_value
```

For the enhanced prior, ranking metrics are more important than tiny changes in MSE. MSE tells you whether the vector is numerically close to the source dynamic table; dynamic-only ranking tells you whether the predicted prior is useful for next-item ranking.

### 4. Tune beta on source validation only

```bash
python scripts/03_tune_continuous_late_fusion.py \
  --config configs/continuous_dynamic.yaml \
  --dynamic_source predicted
```

The clean acceptance test is:

```text
best beta > 0
source-val NDCG@10 improves over beta=0
Recall@10/20 does not collapse
tie ratio stays normal
```

### 5. Freeze beta and evaluate target zero-shot

```bash
python scripts/04_eval_continuous_late_fusion_zero_shot.py \
  --config configs/continuous_dynamic.yaml \
  --targets amazon_cds_and_vinyl
```

Do not change epoch, beta, loss weights, or checkpoint based on ACV test results.

## Minimal experiment matrix

For the first clean feasibility pass:

| Method | Purpose |
|---|---|
| semantic-only final | strong source-selected baseline |
| old MSE-only continuous prior, if preserved | continuity with previous exploration |
| enhanced weighted regression only | check if feature weighting helps |
| enhanced weighted + role auxiliary | check if role structure helps |
| enhanced weighted + role auxiliary + dynamic BPR | check if ranking-aware prior helps |
| oracle continuous dynamic, source only | upper-bound diagnosis |

## Model selection rule

Use source validation only.

Recommended main rule:

```text
Primary: source validation NDCG@10
Constraint: Recall@10/20 should not drop materially
Tie-breaker: NDCG@20, then Recall@20, then earlier epoch
```

For the dynamic prior training script, `selection_mode: dynamic_composite` is allowed as a feasibility-stage diagnostic because it uses only source validation. For the final paper protocol, document the exact source-only selection rule and keep it identical across methods.
