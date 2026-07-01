# DPTF / ISDDG

**LLM-guided Dynamic Prototype Transfer Framework for Zero-shot Cross-Domain Sequential Recommendation**

This repository contains the current ISDDG/DPTF codebase and experiment materials for zero-shot cross-domain sequential recommendation. The latest mainline has moved from early semantic-only and item-level dynamic transfer attempts toward a stronger setup:

```text
Source domain sequential training
        ↓
Strong semantic generalization backbone
        ↓
Source-selected checkpoint
        ↓
Source-induced dynamic / role / prototype signals
        ↓
Target-domain forward-only zero-shot ranking
```

The current empirical focus is:

- **Source domain:** `amazon_movies_and_tv` (AMT)
- **Primary target domain:** `amazon_cds_and_vinyl` (ACV)
- **Confirmation / boundary target domain:** `amazon_industrial_and_scientific` (AIS)
- **Evaluation protocol:** sampled-100 zero-shot evaluation
- **Model selection:** source validation only
- **Target usage:** target interactions are used only as online query prefixes during evaluation, never for training, checkpoint selection, beta selection, or hyperparameter tuning

---

## 1. Current research position

The latest three-seed results show that the project is no longer in a "find a baseline" stage. The current problem is how to test dynamic generalization on top of strong semantic transfer backbones.

The recommended mainline is:

1. Keep **BERT4Rec-SEM** as the semantic sanity baseline.
2. Use **BERT4Rec-SAGE** as the primary strong backbone for ACV.
3. Use **BERT4Rec-RecG-a0.1** as the second strong baseline.
4. Rebuild sequence dynamic prototypes in the **SAGE / RecG prefix-state space**, not in the old `semantic_final` space.
5. Keep only the dynamic / role branches for the next main experiment.
6. Fix `beta_sem = 0` for the prototype semantic branch because it overlaps with the semantic backbone and was unstable in previous validation.
7. Treat AIS as a frozen confirmation / domain-shift boundary, not as a target for repeated tuning.

In short, ISDDG should now be written and tested as a **dynamic generalization enhancement on top of strong semantic transfer**, not as a standalone item-level dynamic prior.

---

## 2. What is locked, what is active, and what is deprecated

### Locked baselines

The following baselines already have three-seed results and should remain in the main comparison table:

| Method | Role in the paper |
|---|---|
| `BERT4Rec-SEM` | Semantic sanity baseline |
| `BERT4Rec-RecG-a0.1` | Strong semantic transfer baseline |
| `BERT4Rec-SAGE` | Current strongest ACV backbone / main anchor |

### Active ISDDG route

The active route is:

```text
Strong semantic backbone
    ├── SAGE as primary backbone
    └── RecG-a0.1 as secondary backbone

Source-induced dynamic signals
    ├── continuous dynamic prior
    ├── dynamic role compatibility
    └── sequence dynamic prototype memory

Final candidate-aware ranking
    └── backbone_score + beta_dyn * dynamic_or_prototype_score
```

The next clean experiment should be:

```text
SAGE + sequence dynamic prototype
```

If it passes source validation, evaluate it on ACV without any target-side retuning. If the SAGE version works, optionally repeat the same procedure on RecG-a0.1.

### Deprecated / diagnostic-only routes

The following routes should not be expanded into large new grids:

| Route | Status | Reason |
|---|---|---|
| item-level semantic-kNN dynamic prior | diagnostic only | source-selected beta often returns to 0; item-level transfer is unstable |
| role / dynamic early fusion | stopped | can interfere with backbone representation and is hard to diagnose |
| MSE-only continuous dynamic prior | diagnostic only | lower regression error does not guarantee ranking improvement |
| old sequence prototype v1 on `semantic_final` | stopped | source NDCG@10 gain was too small; prototypes were too coarse |
| prototype semantic branch | stopped for now | overlaps with the semantic backbone; `beta_sem > 0` often hurts source validation |
| old alpha0-control checkpoint | historical only | not aligned with the current clean protocol |

These negative results are still valuable for the paper: they explain why dynamic signals must be injected through a stronger cross-domain state space rather than through naive item-level statistics.

---

## 3. Latest three-seed baseline results

### 3.1 Source validation on AMT

All checkpoints are selected using AMT source validation only.

| Method | Seeds | AMT source NDCG@10 | Min-max | Avg. best epoch |
|---|---:|---:|---:|---:|
| BERT4Rec-SEM | 3 | 0.2599 ± 0.0052 | 0.2539-0.2632 | 32.7 |
| BERT4Rec-RecG-a0.1 | 3 | 0.2576 ± 0.0071 | 0.2495-0.2625 | 37.7 |
| BERT4Rec-SAGE | 3 | 0.2598 ± 0.0091 | 0.2494-0.2657 | 47.0 |

The source validation numbers are close. The main advantage of SAGE and RecG appears after zero-shot transfer to ACV.

### 3.2 AMT → ACV zero-shot

| Method | R@10 | NDCG@10 | MRR@10 | R@20 | NDCG@20 | Mean rank |
|---|---:|---:|---:|---:|---:|---:|
| BERT4Rec-SEM | 0.4323 ± 0.0071 | 0.2361 ± 0.0047 | 0.1766 ± 0.0039 | 0.6147 | 0.2822 | 23.88 |
| BERT4Rec-RecG-a0.1 | 0.4717 ± 0.0176 | 0.2622 ± 0.0140 | 0.1984 ± 0.0128 | 0.6420 | 0.3052 | 21.94 |
| BERT4Rec-SAGE | 0.4757 ± 0.0228 | 0.2656 ± 0.0151 | 0.2016 ± 0.0126 | 0.6474 | 0.3089 | 21.75 |

Relative to BERT4Rec-SEM on ACV:

| Method | R@10 | R@20 | NDCG@10 | NDCG@20 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| BERT4Rec-RecG-a0.1 | +9.12% | +4.44% | +11.03% | +8.16% | +12.36% |
| BERT4Rec-SAGE | +10.04% | +5.33% | +12.46% | +9.49% | +14.12% |

**Takeaway:** ACV is the primary target for the next innovation experiment. SAGE is the strongest current backbone, and RecG-a0.1 is the second strong baseline.

### 3.3 AMT → AIS zero-shot

| Method | R@10 | NDCG@10 | MRR@10 | R@20 | NDCG@20 | Mean rank |
|---|---:|---:|---:|---:|---:|---:|
| BERT4Rec-SEM | 0.2669 ± 0.0097 | 0.1318 ± 0.0051 | 0.0913 ± 0.0038 | 0.4297 | 0.1728 | 33.57 |
| BERT4Rec-RecG-a0.1 | 0.2176 ± 0.0354 | 0.1035 ± 0.0198 | 0.0696 ± 0.0151 | 0.3754 | 0.1431 | 36.91 |
| BERT4Rec-SAGE | 0.2337 ± 0.0174 | 0.1136 ± 0.0091 | 0.0777 ± 0.0066 | 0.3972 | 0.1546 | 35.23 |

**Takeaway:** AIS shows a different transfer regime. SEM is more stable there, so AIS should be used as a frozen confirmation target and domain-shift discussion, not as a tuning target.

---

## 4. Repository layout

```text
DPTF/
├── artifacts/                  # Generated roles, dynamics, checkpoints, prototypes
├── baselines/                  # External / integrated baseline families
│   ├── LLM-RecG/
│   ├── PrepRec/
│   └── SAGERec/
├── configs/                    # YAML configs for semantic, dynamic-role, and dynamic-prior runs
├── data/                       # Local processed data and semantic embeddings
├── isddg/                      # Main package
│   ├── baselines/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   ├── prototypes/
│   ├── training/
│   └── utils/
├── results/                    # Result CSVs and experiment outputs
├── scripts/                    # Runnable experiment entry points
├── README.md                   # This file
├── README_continuous_dynamic.md
└── requirements.txt
```

The new `README.md` is intended to supersede older generated notes and keep the repository aligned with the latest experiment status.

---

## 5. Expected data layout

The code assumes a local data layout like this:

```text
data/
├── processed/
│   ├── amazon_movies_and_tv.csv
│   ├── amazon_cds_and_vinyl.csv
│   └── amazon_industrial_and_scientific.csv
└── semantic_embeddings/
    ├── amazon_movies_and_tv_embedding_llama_fixed.parquet
    ├── amazon_cds_and_vinyl_embedding_llama_fixed.parquet
    └── amazon_industrial_and_scientific_embedding_llama_fixed.parquet
```

Common fallback names such as `*_embedding_llama.parquet` may be supported by the loader, but the fixed parquet files should be treated as the clean default.

Interaction files should contain normalized user, item, and timestamp fields. The loaders are designed to handle common variants such as `UserId`, `user_id`, `reviewerID`, `ItemId`, `item_id`, `asin`, `Timestamp`, and `unixReviewTime`.

---

## 6. Environment setup

```bash
pip install -r requirements.txt
```

Core dependencies include:

```text
numpy
pandas
pyarrow
scikit-learn
torch
tqdm
PyYAML
```

---

## 7. Currently runnable workflow

This section describes the clean workflow supported by the current public codebase.

### 7.1 Data and path checks

```bash
python scripts/00_check_data.py \
  --data_root ./data \
  --domains amazon_movies_and_tv,amazon_cds_and_vinyl,amazon_industrial_and_scientific
```

Optional data-preparation / audit scripts currently present in `scripts/` include:

```bash
python scripts/00_build_manifests_and_fixed_embeddings.py
python scripts/00_deep_data_quality_check.py
python scripts/00_filter_strict_targets.py
```

Use `--help` for each script before running if local paths differ.

### 7.2 Build source dynamic roles

```bash
python scripts/01_build_dynamic_roles.py \
  --config configs/dynamic_role_signal.yaml
```

This stage builds source-side role observations and role tables. These are source-domain assets only.

### 7.3 Train the semantic baseline

```bash
python scripts/02_train_semantic.py \
  --config configs/semantic.yaml
```

Then evaluate semantic-only zero-shot ranking:

```bash
python scripts/05_eval_semantic_zero_shot.py \
  --config configs/semantic.yaml \
  --targets amazon_cds_and_vinyl,amazon_industrial_and_scientific
```

### 7.4 Train continuous dynamic prior v2

The v2 prior is the current ranking-aware dynamic-prior path:

```bash
python scripts/02_train_continuous_dynamic_prior_v2.py \
  --config configs/continuous_dynamic_v2.yaml
```

It combines:

```text
weighted regression
+ role auxiliary supervision
+ dynamic BPR
+ source validation ranking
```

The important source-side indicators are:

```text
val_reg_mse
val_reg_mae
rank_Recall@10
rank_NDCG@10
rank_MRR@10
tie_case_ratio
best_epoch
selection_value
```

Ranking metrics matter more than small changes in regression MSE.

### 7.5 Tune beta on source validation only

```bash
python scripts/03_tune_continuous_late_fusion.py \
  --config configs/continuous_dynamic_v2.yaml \
  --dynamic_source predicted
```

Acceptance criteria:

```text
best beta > 0
source-val NDCG@10 improves over beta=0
Recall@10 / Recall@20 do not collapse
tie ratio stays near zero
```

### 7.6 Freeze beta and evaluate target zero-shot

```bash
python scripts/04_eval_continuous_late_fusion_zero_shot.py \
  --config configs/continuous_dynamic_v2.yaml \
  --targets amazon_cds_and_vinyl,amazon_industrial_and_scientific
```

Do not change the checkpoint, beta, loss weights, or epoch based on ACV or AIS results.

---

## 8. Strong-backbone dynamic prototype plan

The current README intentionally separates "currently runnable code paths" from the next main experiment plan.

The next innovation experiment should not reuse old `semantic_final` prototypes. Instead:

1. Select the SAGE checkpoint using source validation only.
2. Export source prefix states from the selected SAGE backbone.
3. Build a source key-value memory:

```text
key   = source prefix state h_u^t
value = next-item dynamic vector and / or next-item dynamic role
```

4. Cluster source prefix states into prototype centers.
5. During target evaluation, encode the target prefix with the same selected backbone.
6. Retrieve source prototypes using the target prefix state.
7. Score candidate items using dynamic / role compatibility.
8. Keep `beta_sem = 0`; tune only the dynamic branch on source validation.

Recommended small grid:

```text
M        ∈ {128, 256}
topM     ∈ {8, 16}
beta_dyn ∈ {0, 0.005, 0.01, 0.02, 0.03, 0.05}
temperature: small fixed set, source-selected only
```

Minimum required logging for every prototype result:

```text
backbone
source checkpoint
source selection metric
prototype bank path
M
topM
temperature
beta_dyn
beta_sem
target_interactions_used_for_training=False
target_interactions_used_for_model_selection=False
tie_case_ratio
```

---

## 9. Main experiment matrix

| Method | AMT source val | ACV zero-shot | AIS confirm | Priority |
|---|---|---|---|---|
| BERT4Rec-SEM | done, 3 seeds | done, 3 seeds | done, 3 seeds | required sanity baseline |
| BERT4Rec-RecG-a0.1 | done, 3 seeds | done, 3 seeds | done, 3 seeds | required strong baseline |
| BERT4Rec-SAGE | done, 3 seeds | done, 3 seeds | done, 3 seeds | main backbone |
| SAGE + predicted dynamic prior | pending / diagnostic | run only if source-selected | frozen confirm | lightweight ablation |
| SAGE + sequence dynamic prototype | pending | main innovation experiment | frozen confirm | highest |
| SAGE + role compatibility / confidence gate | optional | optional | optional | run only if Day-4 source validation passes |
| RecG-a0.1 + sequence dynamic prototype | optional | optional replication | optional | compute permitting |
| semantic-kNN / early fusion / MSE-only | archive | no new main run | no run | diagnostic table |

---

## 10. Stop rules and success criteria

### Source validation

A dynamic enhancement should enter ACV zero-shot evaluation only if it satisfies:

```text
source-val NDCG@10 improvement ≥ +0.001 to +0.002
Recall@10 does not materially decrease
tie_case_ratio stays near zero
```

### ACV zero-shot

If `SAGE + dynamic` improves ACV NDCG@10 by less than `+0.001` over SAGE, do not overclaim it as a main contribution. Report it as weak or boundary evidence.

### AIS confirmation

AIS is not a tuning target. If AIS decreases under the ACV/source-frozen setting, keep the result and discuss domain shift.

### Invalid result warning signs

A result should not be used as a main paper result if:

```text
target metrics were used for checkpoint selection
target metrics were used for beta selection
target dynamic statistics were used for training
target zero-shot CSV does not record target usage flags
prototype eval does not print source-selected beta/checkpoint/bank
tie_case_ratio is abnormal
```

---

## 11. Suggested paper narrative

The current paper narrative should use a three-layer evidence chain.

### Layer 1: Strong baseline evidence

SAGE and RecG outperform SEM on ACV, showing that the benchmark requires strong cross-domain semantic generalization and that the paper is not built on weak baselines.

### Layer 2: Dynamic diagnostic evidence

Previous failures show that dynamic signals cannot be naively transferred through item-level semantic neighbors, early fusion, or MSE-only regression.

### Layer 3: ISDDG dynamic enhancement

ISDDG should be positioned as a source-induced dynamic / role / prototype enhancement over a strong semantic transfer backbone. The key hypothesis is that source prefix-state prototypes capture transferable transition dynamics that are not fully represented by semantic item embeddings alone.

---

## 12. Recommended figure structure

Use a two-stage method figure.

### Training / source side

```text
Shared Semantic Generalization Backbone
        ↓
Source Dynamic Role Prior Learning
        ↓
Role / Dynamic-aware Sequential Modeling
        ↓
Source Sequence Dynamic Prototype Memory
```

### Inference / target side

```text
Target Prefix Encoding
        ↓
Source Prototype Retrieval
        ↓
Candidate-aware Final Ranking
```

Keep the inference side minimal to emphasize strict zero-shot evaluation.

---

## 13. One-week execution checklist

| Day | Goal | Output |
|---|---|---|
| Day 1 | Clean and freeze the three-seed baseline tables | `baseline_3seed_main_table.csv`, experiment ledger |
| Day 2 | Lock SAGE as primary backbone and RecG-a0.1 as secondary | prefix state extractor, source selection logs |
| Day 3 | Rebuild sequence dynamic prototypes on SAGE states | `prototype_sage_M{128,256}.pt`, `build_summary.json` |
| Day 4 | Select dynamic branch on AMT source validation only | `isddg_sage_source_val.csv`, source-selected summary |
| Day 5 | Forward ACV zero-shot with frozen source-selected settings | `isddg_sage_acv_zero_shot.csv` |
| Day 6 | Fill three seeds and core ablations | ablation table and case candidates |
| Day 7 | Run AIS frozen confirmation and write diagnostics | `final_main_table.csv`, `diagnostic_negative_table.csv` |

---

## 14. Naming and logging conventions

Recommended result naming:

```text
results/
├── baseline_3seed_main_table.csv
├── isddg_sage_source_val.csv
├── isddg_sage_acv_zero_shot.csv
├── isddg_sage_ais_zero_shot.csv
├── isddg_recg_acv_zero_shot.csv
├── ablation_3seed_or_single_seed.csv
└── diagnostic_negative_table.csv
```

Recommended run-name fields:

```text
family
run_name
backbone
mode
seed
source
target
checkpoint
source_selection_metric
source_selection_value
target_interactions_used_for_training
target_interactions_used_for_model_selection
tie_case_ratio
mean_rank
```

---

## 15. Maintainer note

This README is meant to replace scattered older generated markdown notes. The project direction should now remain narrow:

```text
ACV primary
SAGE first
RecG second
SEM sanity
source-only selection
target forward-only
dynamic / role / sequence prototype as plugin enhancement
AIS frozen confirmation
```

If the next SAGE prototype experiment fails the source-validation stop rule, do not expand a large grid. Keep the strong-baseline results as the main empirical contribution and report the dynamic module as a diagnostic boundary / future-work direction.
