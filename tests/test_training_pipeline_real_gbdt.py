from __future__ import annotations

import json
from pathlib import Path

import pytest

from renquant_model_gbdt import (
    PanelGbdtTrainingPipeline,
    TrainingContext,
    train_panel_ltr_artifact,
    validate_panel_ltr_artifact,
)

from panel_fixtures import make_easy_panel


def _dataset_manifest() -> dict:
    return {
        "dataset_id": "synthetic-panel",
        "schema_version": "unit-v1",
        "fingerprint": "sha256:data",
        "uri": "object://renquant-data/synthetic-panel.parquet",
        "asset_class": "equity",
    }


def _model_config(**overrides) -> dict:
    config = {
        "strategy": "renquant_104",
        "backend": "xgboost",
        "config_fingerprint": "sha256:config",
        "code_commit": "sha-test",
        "train_run_id": "run-real-gbdt",
        "artifact_id": "unit-panel-ltr",
        "lookahead_days": 2,
        "cv_embargo_days": 2,
        "cv_n_splits": 3,
        "num_boost_round": 20,
        "cv_num_boost_round": 12,
        "feature_cols": ["x1", "x2"],
        "xgb_params": {"max_depth": 2, "eta": 0.1},
    }
    config.update(overrides)
    return config


def test_real_gbdt_training_pipeline_writes_contract_artifact(tmp_path: Path) -> None:
    panel, group_sizes, feature_cols = make_easy_panel(n_dates=30, n_tickers=8, seed=20)

    def loader(manifest: dict):
        return {"panel": panel, "group_sizes": group_sizes, "feature_cols": feature_cols}

    ctx = TrainingContext(
        dataset_manifest=_dataset_manifest(),
        model_config=_model_config(),
        output_dir=tmp_path / "out",
    )

    PanelGbdtTrainingPipeline(
        loader,
        train_panel_ltr_artifact,
        validate_panel_ltr_artifact,
    ).run(ctx)

    assert ctx.model_artifact is not None
    assert ctx.artifact_manifest is not None
    assert ctx.model_artifact["kind"] == "panel_ltr_xgboost"
    assert ctx.model_artifact["feature_cols"] == ["x1", "x2"]
    assert ctx.model_artifact["fingerprint"].startswith("sha256:")
    assert not ctx.model_artifact["fingerprint"].startswith("sha256:sha256:")
    assert ctx.model_artifact["oos_per_fold_ic"]
    assert ctx.model_artifact["cv_embargo_days"] == 2
    assert ctx.artifact_manifest["feature_cols"] == ["x1", "x2"]
    assert ctx.artifact_manifest["kind"] == "panel_ltr_xgboost"
    assert ctx.artifact_manifest["local_artifact_path"] == ctx.model_artifact["local_artifact_path"]
    assert ctx.artifact_manifest["lookahead_days"] == 2
    assert ctx.metrics_record["panel_contract_ok"] is True
    model_path = Path(ctx.model_artifact["local_artifact_path"])
    assert model_path.exists()
    payload = json.loads(model_path.read_text())
    assert payload["train_run_id"] == "run-real-gbdt"


def test_real_gbdt_pipeline_rejects_unsupported_backend(tmp_path: Path) -> None:
    panel, group_sizes, feature_cols = make_easy_panel(n_dates=12, n_tickers=6, seed=21)
    ctx = TrainingContext(
        dataset_manifest=_dataset_manifest(),
        model_config=_model_config(backend="lightgbm"),
        output_dir=tmp_path / "out",
    )

    with pytest.raises(ValueError, match="unsupported backend"):
        PanelGbdtTrainingPipeline(
            lambda _: {"panel": panel, "group_sizes": group_sizes, "feature_cols": feature_cols},
            train_panel_ltr_artifact,
            validate_panel_ltr_artifact,
        ).run(ctx)
