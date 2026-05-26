# renquant-model-gbdt

Production model-training repository for RenQuant's current GBDT / panel-LTR
model line.

Operating model: https://github.com/hallovorld/RenQuant/blob/main/doc/arch/subrepo-operating-model.md

Repository map: [RENQUANT_REPOS.md](RENQUANT_REPOS.md)

Local automation:

```bash
make test
make doctor
```

This repo owns training, scoring, validation, and model-ledger output for the
current production model family. It must not own live order placement,
portfolio QP execution, broker adapters, or strategy scheduling.

## Pipeline Rule

All workflows are expressed as `renquant-common` Task/Job/Pipeline chains.

The current real implementation supports the first production slice:

- XGBoost `rank:pairwise` panel-LTR via `PanelLTRModel`.
- Purged K-fold / combinatorial purged CV IC estimation.
- `train_panel_ltr_artifact()` for already-materialized panel datasets.

The trainer deliberately does not fetch market data or build alpha features.
Its dataset input must be a dict with `panel`, `group_sizes`, and
`feature_cols`; missing columns, missing groups, unsupported backends, or
embargo shorter than lookahead fail closed.

## Initial Split Source

`hallovorld/RenQuant` commit
`8f3e08d8d1ae1e402a78f4815efb59e3c7c66aa8`.

## Local Test

Until `renquant-common` is published, run tests with the adjacent local source:

```bash
PYTHONPATH=../renquant-common/src:src python -m pytest -q
```
