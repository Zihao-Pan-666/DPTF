# ISDDG protocol-matched BERT4Rec Sem / RecG / SAGE patch

This patch adds a new, isolated baseline family. It does **not** delete the
historical `semantic_final`, `llm_recg`, or `baselines/SAGERec` implementations.

## What is unified

The three primary baselines use exactly the same:

- source/target domains and semantic embedding files;
- source split builder and left-padded PrefixDataset;
- hidden size 128, 2 layers, 2 heads, dropout 0.2;
- AdamW, learning rate 1e-4, batch size 128;
- 5 history-excluding training negatives;
- deterministic sampled-100 validation;
- stable `logsigmoid` BPR;
- fixed validation candidates across epochs;
- worst-tie ranking and the same metric implementation;
- source validation NDCG@10 checkpoint selection;
- fail-fast NaN/Inf checks and gradient clipping.

Only the method-specific architecture/objective changes.

| Configuration | Architecture | Alignment objective |
|---|---|---|
| `bert4rec_sem_matched.yaml` | recommendation projection only | none |
| `bert4rec_arch0_matched.yaml` | dual projections + merge | none |
| `bert4rec_recg_matched.yaml` | dual projections + merge | official single-alpha RecG |
| `bert4rec_sage_matched.yaml` | dual projections + merge | DPTF SAGE, weighted once |

`Arch0` is a diagnostic control, not one of the three headline baselines. It
separates the effect of adding the dual-projection/merge architecture from the
effect of the alignment loss.

## Important corrections

1. **Left padding readout:** the user state is always `encoded[:, -1, :]`.
2. **Candidate scoring:** all modes score candidates through the recommendation
   projection.
3. **RecG weighting:** the implemented loss is already
   `-alpha*H_intra + alpha*N/D^3*H_inter`; it is added once, avoiding the legacy
   double multiplication by `lambda_g`.
4. **SAGE fidelity:** `L_ID`, full pairwise semantic-aware `L_SIC`, and the
   raw-center adaptive coefficient follow the public DPTF/SAGERec loss;
   `omega` is then applied exactly once by the unified objective.
5. **Strict selection:** target interactions are never used in training or
   checkpoint selection. They are opened only by the final evaluation script.
6. **NaN handling:** non-finite scores, losses, gradients, parameters, or
   validation scores abort immediately and cannot overwrite a checkpoint.
7. **Runtime:** auxiliary domains contribute only deterministic item-metadata
   pools. Each step samples 128 items/domain instead of repeatedly projecting
   the entire auxiliary catalog.

## Version compatibility

The included compatibility adapter handles the signature differences between
the earlier uploaded ISDDG snapshot and the newer public DPTF tree (dataset
constructor, interaction-loader return values, split-builder location, seed
helper, and device helper). No manual edits should be needed for those API
differences.

## Installation

From the DPTF/ISDDG project root, extract this zip and copy/merge its folders
into the repository root. These are new file names, so the existing historical
scripts remain available.

Run the protocol check:

```powershell
python scripts/10_check_bert4rec_family_configs.py
```

## Recommended execution order

### 1. Cheap smoke tests

```powershell
python scripts/11_train_bert4rec_family.py `
  --config configs/bert4rec_sem_matched.yaml `
  --epochs 1 --max-train-batches 20 --max-val-batches 10 `
  --run-name smoke_sem

python scripts/11_train_bert4rec_family.py `
  --config configs/bert4rec_recg_matched.yaml `
  --epochs 1 --max-train-batches 20 --max-val-batches 10 `
  --run-name smoke_recg

python scripts/11_train_bert4rec_family.py `
  --config configs/bert4rec_sage_matched.yaml `
  --epochs 1 --max-train-batches 20 --max-val-batches 10 `
  --run-name smoke_sage
```

Delete smoke outputs before the full runs, or keep their distinct run names.

### 2. Primary full runs

```powershell
python scripts/11_train_bert4rec_family.py --config configs/bert4rec_sem_matched.yaml
python scripts/11_train_bert4rec_family.py --config configs/bert4rec_recg_matched.yaml
python scripts/11_train_bert4rec_family.py --config configs/bert4rec_sage_matched.yaml
```

### 3. Optional architecture diagnostic

```powershell
python scripts/11_train_bert4rec_family.py --config configs/bert4rec_arch0_matched.yaml
```

Interpretation:

- `Arch0 << Sem`: first inspect dual-branch architecture/initialization; do not
  blame alpha.
- `Arch0 ~= Sem` but `RecG < Arch0`: the RecG objective/weight is the likely
  source of the penalty.
- `RecG > Arch0`: the alignment loss provides a source-validation benefit.

### 4. Pre-registered RecG alpha screening using source validation only

The default `alpha=0.003` is close to the effective `0.0025` scale produced by
the historical DPTF double-0.05 weighting, while using the corrected
single-alpha formula.

```powershell
python scripts/11_train_bert4rec_family.py `
  --config configs/bert4rec_recg_matched.yaml `
  --alpha 0.001 --run-name bert4rec_recg_a001_matched

python scripts/11_train_bert4rec_family.py `
  --config configs/bert4rec_recg_matched.yaml `
  --alpha 0.003 --run-name bert4rec_recg_a003_matched

python scripts/11_train_bert4rec_family.py `
  --config configs/bert4rec_recg_matched.yaml `
  --alpha 0.01 --run-name bert4rec_recg_a010_matched
```

Use only source validation to preselect alpha. Do not inspect a target-domain
result and then change alpha.

### 5. Final zero-shot evaluation

```powershell
python scripts/12_eval_bert4rec_family_zero_shot.py --config configs/bert4rec_sem_matched.yaml
python scripts/12_eval_bert4rec_family_zero_shot.py --config configs/bert4rec_recg_matched.yaml
python scripts/12_eval_bert4rec_family_zero_shot.py --config configs/bert4rec_sage_matched.yaml
```

For a non-default alpha run, pass its exact checkpoint:

```powershell
python scripts/12_eval_bert4rec_family_zero_shot.py `
  --config configs/bert4rec_recg_matched.yaml `
  --checkpoint artifacts/checkpoints/bert4rec_family/bert4rec_recg_a001_matched_amazon_movies_and_tv_seed2026.pt
```

## Result files

- `results/mainline/bert4rec_family/bert4rec_family_source_val.csv`
- `results/mainline/bert4rec_family/bert4rec_family_zero_shot.csv`
- one JSON summary for every source run;
- one JSON zero-shot report for every checkpoint;
- source-selected checkpoints under
  `artifacts/checkpoints/bert4rec_family/`.

## Scope

This package deliberately standardizes the **item-level** Sem/RecG/SAGE
comparison first. Sequence-pattern transfer is not enabled in these primary
configs, because enabling it only for RecG/SAGE would confound the baseline
comparison. After this family is stable, sequence-pattern transfer should be a
separate `+Pattern` ablation using the same selected item-level checkpoint.
