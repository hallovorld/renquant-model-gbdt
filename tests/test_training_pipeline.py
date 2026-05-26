from __future__ import annotations

from pathlib import Path

import pytest

from renquant_model_gbdt import (
    PanelGbdtTrainingPipeline,
    TrainingContext,
    transform_feature_frame,
)


def test_training_pipeline_uses_common_task_job_pattern(tmp_path: Path) -> None:
    calls: list[str] = []

    def loader(manifest: dict):
        calls.append("load")
        return {"rows": [1, 2, 3], "manifest": manifest}

    def trainer(dataset, config: dict, output_dir: Path):
        calls.append("train")
        assert dataset["rows"] == [1, 2, 3]
        assert config["objective"] == "rank:pairwise"
        assert output_dir.exists()
        return {
            "artifact_id": "gbdt-fixture",
            "model_family": "gbdt-panel-ltr",
            "fingerprint": "sha256:model",
            "uri": "object://renquant-artifacts/gbdt-fixture.json",
            "promotion_status": "candidate",
            "feature_cols": ["alpha_1", "alpha_2"],
            "trained_date": "2026-05-25",
            "config_fingerprint": "sha256:config",
            "panel_shape": {"rows": 1000, "cols": 2},
            "lookahead_days": 5,
            "train_run_id": "run-1",
            "oos_mean_ic": 0.031,
            "oos_std_ic": 0.012,
            "oos_per_fold_ic": [0.02, 0.04, 0.033],
            "cv_method": "purged-walk-forward",
            "cv_embargo_days": 5,
        }, {"kind": "global_calibrator"}

    def validator(artifact: dict, dataset, config: dict):
        calls.append("validate")
        assert artifact["model_family"] == "gbdt-panel-ltr"
        return {"oos_mean_ic": 0.031, "train_ic": 0.154}

    ctx = TrainingContext(
        dataset_manifest={
            "dataset_id": "alpha158_fund_fixture",
            "fingerprint": "sha256:test",
            "schema_version": "fixture-v1",
            "uri": "object://renquant-data/alpha158_fund_fixture.parquet",
            "asset_class": "equity",
        },
        model_config={"objective": "rank:pairwise", "strategy": "renquant_104"},
        output_dir=tmp_path / "out",
    )
    result = PanelGbdtTrainingPipeline(loader, trainer, validator).run(ctx)

    assert result.ok is True
    assert result.name == "panel-gbdt-training"
    assert calls == ["load", "train", "validate"]
    assert ctx.model_artifact["artifact_id"] == "gbdt-fixture"
    assert ctx.calibration_artifact == {"kind": "global_calibrator"}
    assert ctx.artifact_manifest is not None
    assert ctx.artifact_manifest["data_fingerprint"] == "sha256:test"
    assert ctx.metrics_record["oos_mean_ic"] == pytest.approx(0.031)
    assert ctx.metrics_record["panel_contract_ok"] is True


def test_training_pipeline_requires_auditable_dataset_manifest(tmp_path: Path) -> None:
    ctx = TrainingContext(
        dataset_manifest={"dataset_id": "missing_fingerprint"},
        model_config={},
        output_dir=tmp_path / "out",
    )

    with pytest.raises(ValueError, match="data manifest missing"):
        PanelGbdtTrainingPipeline(lambda _: object(), lambda *_: ({}, {}), lambda *_: {}).run(ctx)


def test_training_pipeline_requires_strict_panel_oos_contract(tmp_path: Path) -> None:
    def loader(manifest: dict):
        return {"rows": [1], "manifest": manifest}

    def trainer(dataset, config: dict, output_dir: Path):
        return {
            "artifact_id": "bad-panel",
            "model_family": "gbdt-panel-ltr",
            "fingerprint": "sha256:model",
            "uri": "object://renquant-artifacts/bad-panel.json",
            "promotion_status": "candidate",
        }, {}

    def validator(artifact: dict, dataset, config: dict):
        return {"oos_mean_ic": 0.01}

    ctx = TrainingContext(
        dataset_manifest={
            "dataset_id": "alpha158_fund_fixture",
            "fingerprint": "sha256:test",
            "schema_version": "fixture-v1",
            "uri": "object://renquant-data/alpha158_fund_fixture.parquet",
            "asset_class": "equity",
        },
        model_config={"objective": "rank:pairwise", "strategy": "renquant_104"},
        output_dir=tmp_path / "out",
    )

    with pytest.raises(ValueError, match="panel contract failed"):
        PanelGbdtTrainingPipeline(loader, trainer, validator).run(ctx)


def test_feature_transform_applies_raw_and_panel_source_space() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {"alpha": [12.0], "roe": [120.0], "missing_source": [8.0]},
        index=["AAPL"],
    )
    feature_cols = ["alpha", "roe", "missing_source", "absent"]
    metadata = {
        "feature_means": [10.0, 100.0, 0.0, 2.0],
        "feature_stds": [2.0, 10.0, 1.0, 2.0],
        "feature_norm_kind": ["legacy_full_z", "robust_z", "identity", "legacy_full_z"],
        "feature_raw_clip_low": [0.0, 0.0, 0.0, 0.0],
        "feature_raw_clip_high": [20.0, 200.0, 10.0, 10.0],
    }

    raw = transform_feature_frame(
        frame,
        feature_cols,
        metadata,
        source_space="raw",
        clip=0,
    )
    panel = transform_feature_frame(
        frame,
        feature_cols,
        metadata,
        source_space="panel",
        clip=0,
    )

    assert raw.loc["AAPL", "alpha"] == pytest.approx(1.0)
    assert raw.loc["AAPL", "roe"] == pytest.approx(2.0)
    assert raw.loc["AAPL", "absent"] == pytest.approx(-1.0)
    assert panel.loc["AAPL", "alpha"] == pytest.approx(12.0)
    assert panel.loc["AAPL", "roe"] == pytest.approx(2.0)
