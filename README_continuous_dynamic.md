# Continuous Dynamic Prior V2 for ISDDG

This patch adds a new semantic-conditioned continuous dynamic prior learner.

## Added files

- `isddg/training/continuous_dynamic_prior_v2_trainer.py`
- `scripts/02_train_continuous_dynamic_prior_v2.py`
- `configs/continuous_dynamic_v2.yaml`

## Recommended workflow

```bash
python scripts/01_build_dynamic_roles.py \
  --config configs/dynamic_role_signal.yaml \
  --source amazon_movies_and_tv

python scripts/02_train_continuous_dynamic_prior_v2.py \
  --config configs/continuous_dynamic_v2.yaml

python scripts/03_tune_continuous_late_fusion.py \
  --config configs/continuous_dynamic_v2.yaml \
  --dynamic_source predicted \
  --continuous_checkpoint artifacts/checkpoints/continuous_dynamic_prior_v2_amazon_movies_and_tv_seed2026.pt

python scripts/04_eval_continuous_late_fusion_zero_shot.py \
  --config configs/continuous_dynamic_v2.yaml \
  --continuous_checkpoint artifacts/checkpoints/continuous_dynamic_prior_v2_amazon_movies_and_tv_seed2026.pt \
  --targets amazon_cds_and_vinyl,steam
```

The v2 learner uses:

```text
L = lambda_reg * weighted_regression
  + lambda_role * role_auxiliary_KL
  + lambda_bpr * dynamic_BPR
```

Checkpoint selection is source-only. If `source_val_ranking=true`, the trainer
evaluates predicted dynamic-only ranking on the source validation split and uses
`selection_mode` from the config. No target-domain interaction labels are used.
