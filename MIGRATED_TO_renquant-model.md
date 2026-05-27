# MIGRATED to renquant-model

This repository has been merged into **renquant-model**
(https://github.com/hallovorld/renquant-model) per RFC §"Backfill Plan" P3.

- Code now lives at `renquant-model/src/renquant_model_gbdt/` and
  `renquant-model/src/renquant_model_patchtst/` (history preserved via
  git filter-repo subtree merge).
- The Scorer entry point `panel_ltr_xgboost` is now registered by
  renquant-model.
- This repo is kept read-only for rollback / archaeology. Do not commit
  new work here; open it against renquant-model instead.
