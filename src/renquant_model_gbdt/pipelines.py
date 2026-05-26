"""Pipeline contracts for GBDT panel-LTR training.

The concrete XGBoost/LightGBM/CatBoost implementation is ported behind these
interfaces in later slices. This file pins the SDLC contract first: training
is a Task/Job/Pipeline chain from renquant-common, and metrics/artifacts are
explicit outputs rather than scattered side effects.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from renquant_common import Job, Pipeline, Task
from renquant_artifacts import validate_artifact_manifest, validate_panel_artifact_contract
from renquant_base_data import validate_data_manifest


@dataclass
class TrainingContext:
    """Mutable context passed through the GBDT training pipeline."""

    dataset_manifest: dict[str, Any]
    model_config: dict[str, Any]
    output_dir: Path
    dataset: Any | None = None
    model_artifact: dict[str, Any] | None = None
    calibration_artifact: dict[str, Any] | None = None
    artifact_manifest: dict[str, Any] | None = None
    metrics_record: dict[str, Any] = field(default_factory=dict)


DatasetLoader = Callable[[dict[str, Any]], Any]
Trainer = Callable[[Any, dict[str, Any], Path], tuple[dict[str, Any], dict[str, Any]]]
Validator = Callable[[dict[str, Any], Any, dict[str, Any]], dict[str, Any]]


class ValidateManifestTask(Task):
    """Fail fast when the data contract is not auditable."""

    def run(self, ctx: TrainingContext) -> bool | None:
        validate_data_manifest(ctx.dataset_manifest)
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        return True


class LoadDatasetTask(Task):
    def __init__(self, loader: DatasetLoader) -> None:
        self.loader = loader

    def run(self, ctx: TrainingContext) -> bool | None:
        ctx.dataset = self.loader(ctx.dataset_manifest)
        return True


class TrainModelTask(Task):
    def __init__(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def run(self, ctx: TrainingContext) -> bool | None:
        if ctx.dataset is None:
            raise ValueError("dataset must be loaded before TrainModelTask")
        artifact, calibration = self.trainer(ctx.dataset, ctx.model_config, ctx.output_dir)
        ctx.model_artifact = artifact
        ctx.calibration_artifact = calibration
        return True


class ValidateModelTask(Task):
    def __init__(self, validator: Validator) -> None:
        self.validator = validator

    def run(self, ctx: TrainingContext) -> bool | None:
        if ctx.dataset is None or ctx.model_artifact is None:
            raise ValueError("dataset and model_artifact are required before validation")
        ctx.metrics_record = self.validator(ctx.model_artifact, ctx.dataset, ctx.model_config)
        return True


class BuildArtifactManifestTask(Task):
    """Convert trainer output into a registry-valid artifact manifest."""

    def run(self, ctx: TrainingContext) -> bool | None:
        if ctx.model_artifact is None:
            raise ValueError("model_artifact is required before artifact manifest build")
        required = ("artifact_id", "model_family", "fingerprint", "uri")
        missing = [key for key in required if not ctx.model_artifact.get(key)]
        if missing:
            raise ValueError(f"model_artifact missing required keys: {missing}")
        panel_contract = validate_panel_artifact_contract(
            ctx.model_artifact,
            strict=bool(ctx.model_config.get("strict_panel_contract", True)),
            runtime_config=ctx.model_config,
        )
        if not panel_contract.ok:
            raise ValueError(
                "model_artifact panel contract failed: "
                f"errors={panel_contract.errors}; warnings={panel_contract.warnings}"
            )
        ctx.metrics_record.setdefault("panel_contract_ok", panel_contract.ok)
        ctx.metrics_record.setdefault("panel_contract_details", panel_contract.details)
        manifest = {
            "artifact_id": ctx.model_artifact["artifact_id"],
            "model_family": ctx.model_artifact["model_family"],
            "strategy": ctx.model_config.get("strategy", "renquant_104"),
            "fingerprint": ctx.model_artifact["fingerprint"],
            "uri": ctx.model_artifact["uri"],
            "promotion_status": ctx.model_artifact.get("promotion_status", "candidate"),
            "metrics": dict(ctx.metrics_record),
            "data_fingerprint": ctx.dataset_manifest["fingerprint"],
            "config_fingerprint": ctx.model_config.get("config_fingerprint", "unfingerprinted"),
            "code_commit": ctx.model_config.get("code_commit", "uncommitted"),
        }
        for key in _RUNTIME_ARTIFACT_FIELDS:
            if key in ctx.model_artifact and ctx.model_artifact[key] is not None:
                manifest[key] = ctx.model_artifact[key]
        validate_artifact_manifest(manifest)
        ctx.artifact_manifest = manifest
        return True


class TrainingJob(Job):
    def __init__(self, loader: DatasetLoader, trainer: Trainer, validator: Validator) -> None:
        self._tasks = [
            ValidateManifestTask(),
            LoadDatasetTask(loader),
            TrainModelTask(trainer),
            ValidateModelTask(validator),
            BuildArtifactManifestTask(),
        ]

    @property
    def tasks(self) -> list[Task]:
        return self._tasks


class PanelGbdtTrainingPipeline(Pipeline):
    """Production GBDT training pipeline shell."""

    def __init__(self, loader: DatasetLoader, trainer: Trainer, validator: Validator) -> None:
        super().__init__([TrainingJob(loader, trainer, validator)], name="panel-gbdt-training")


_RUNTIME_ARTIFACT_FIELDS = (
    "kind",
    "feature_cols",
    "feature_columns",
    "input_feature_cols",
    "local_artifact_path",
    "artifact_path",
    "trained_date",
    "lookahead_days",
    "panel_shape",
    "cv_method",
    "cv_embargo_days",
    "train_run_id",
)
