# CLAUDE.md

Canonical operating model:
https://github.com/hallovorld/RenQuant/blob/main/doc/arch/subrepo-operating-model.md

Local repo map: `RENQUANT_REPOS.md`.

Branch policy: `main` is the stable interface consumed by other repos and
automation. Experiments, optimizations, and large upgrades happen on feature
branches, then merge back only after tests and integration checks pass.

## Repo Role

`renquant-model-gbdt` owns the current production GBDT / panel-LTR training
line: training, scoring, validation, calibration, and model-ledger output.

## Hard Boundaries

- Use `renquant-common` pipeline primitives for train/score/validate workflows.
- Pull data through `renquant-base-data` manifests.
- Publish model/checkpoint metadata through `renquant-artifacts` manifests.
- Do not place broker orders, implement runtime QP, own daily scheduling, or
  silently write prod artifacts outside the registry.
- Large model or feature changes use a feature branch.
- Do not delete or empty the source umbrella repo at
  `/Users/renhao/git/github/RenQuant`.

## Required Model Evidence

Every serious training run must record data/config/code fingerprints, OOS IC,
regime IC, SPY or benchmark comparison where applicable, calibration health,
and placebo/shuffle sanity. If a new number is reported, its sanity check must
be stored with it.

## Workflow

```bash
make test
make doctor
```
