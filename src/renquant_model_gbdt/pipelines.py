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


@dataclass
class TrainingContext:
    """Mutable context passed through the GBDT training pipeline."""

    dataset_manifest: dict[str, Any]
    model_config: dict[str, Any]
    output_dir: Path
    dataset: Any | None = None
    model_artifact: dict[str, Any] | None = None
    calibration_artifact: dict[str, Any] | None = None
    metrics_record: dict[str, Any] = field(default_factory=dict)


DatasetLoader = Callable[[dict[str, Any]], Any]
Trainer = Callable[[Any, dict[str, Any], Path], tuple[dict[str, Any], dict[str, Any]]]
Validator = Callable[[dict[str, Any], Any, dict[str, Any]], dict[str, Any]]


class ValidateManifestTask(Task):
    """Fail fast when the data contract is not auditable."""

    def run(self, ctx: TrainingContext) -> bool | None:
        required = ("dataset_id", "fingerprint", "schema_version")
        missing = [key for key in required if not ctx.dataset_manifest.get(key)]
        if missing:
            raise ValueError(f"dataset_manifest missing required keys: {missing}")
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


class TrainingJob(Job):
    def __init__(self, loader: DatasetLoader, trainer: Trainer, validator: Validator) -> None:
        self._tasks = [
            ValidateManifestTask(),
            LoadDatasetTask(loader),
            TrainModelTask(trainer),
            ValidateModelTask(validator),
        ]

    @property
    def tasks(self) -> list[Task]:
        return self._tasks


class PanelGbdtTrainingPipeline(Pipeline):
    """Production GBDT training pipeline shell."""

    def __init__(self, loader: DatasetLoader, trainer: Trainer, validator: Validator) -> None:
        super().__init__([TrainingJob(loader, trainer, validator)], name="panel-gbdt-training")
