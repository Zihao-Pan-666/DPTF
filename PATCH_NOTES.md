# Patch notes

## Added files

- `isddg/features/catalog_semantic.py`
- `isddg/models/bert4rec_family.py`
- `isddg/training/bert4rec_family_losses.py`
- `isddg/training/bert4rec_family_trainer.py`
- `isddg/evaluation/bert4rec_family_evaluator.py`
- `scripts/10_check_bert4rec_family_configs.py`
- `scripts/11_train_bert4rec_family.py`
- `scripts/12_eval_bert4rec_family_zero_shot.py`
- `scripts/run_bert4rec_family.ps1`
- four matched configuration files
- `README_BERT4REC_FAMILY.md`

## Replacement policy

All names are new. Extracting the package over the project root should not
overwrite historical baseline code. The new scripts are the authoritative path
for a fair Sem/RecG/SAGE comparison.

## Compatibility assumptions

The existing project must provide:

- `isddg.data.io.load_interactions`
- `isddg.data.io.group_user_sequences`
- `isddg.data.dataset.PrefixDataset`
- `isddg.data.dataset.collate_prefix`
- `isddg.data.semantic_splits.build_source_train_val_samples`
- `isddg.data.semantic_splits.build_target_eval_samples`
- `isddg.features.semantic.load_semantic_embeddings`

The compatibility adapter supports both historical and current names/signatures for:

- two-value or three-value `load_interactions` returns;
- `PrefixDataset` with or without a required `num_items` argument;
- `set_seed` or `set_global_seed`;
- `get_device` or `resolve_device`;
- target split builders located in `data.io` or `data.semantic_splits`.
