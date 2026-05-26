# renquant-model-gbdt

Production model-training repository for RenQuant's current GBDT / panel-LTR
model line.

This repo owns training, scoring, validation, and model-ledger output for the
current production model family. It must not own live order placement,
portfolio QP execution, broker adapters, or strategy scheduling.

## Pipeline Rule

All workflows are expressed as `renquant-common` Task/Job/Pipeline chains.

The first bootstrap commit contains a dependency-injected training pipeline
contract. The existing production code will be ported into these tasks in
small reviewed slices rather than copied wholesale.

## Initial Split Source

`hallovorld/RenQuant` commit
`8f3e08d8d1ae1e402a78f4815efb59e3c7c66aa8`.

## Local Test

Until `renquant-common` is published, run tests with the adjacent local source:

```bash
PYTHONPATH=../renquant-common/src:src python -m pytest -q
```
