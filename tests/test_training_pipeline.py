from __future__ import annotations

from pathlib import Path

import pytest

from renquant_model_gbdt import PanelGbdtTrainingPipeline, TrainingContext


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


def test_training_pipeline_requires_auditable_dataset_manifest(tmp_path: Path) -> None:
    ctx = TrainingContext(
        dataset_manifest={"dataset_id": "missing_fingerprint"},
        model_config={},
        output_dir=tmp_path / "out",
    )

    with pytest.raises(ValueError, match="data manifest missing"):
        PanelGbdtTrainingPipeline(lambda _: object(), lambda *_: ({}, {}), lambda *_: {}).run(ctx)
