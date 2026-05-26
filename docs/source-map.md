# Source Map From Monorepo

Initial source commit:
`8f3e08d8d1ae1e402a78f4815efb59e3c7c66aa8`.

The production GBDT code should be ported in reviewed slices from:

- `backtesting/renquant_104/training_panel/`
- `scripts/train_104.py`
- `scripts/train_panel*.py`
- `scripts/eval_xgb_*.py`
- GBDT-specific calibrator scripts
- GBDT scorer runtime code currently mixed into
  `backtesting/renquant_104/kernel/panel_pipeline/`

Do not copy the full folders blindly. Each slice needs:

1. a named pipeline Task/Job owner,
2. a fixture or synthetic unit test,
3. an import-boundary check,
4. a model-ledger output contract,
5. no dependency on live execution or broker code.
