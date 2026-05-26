"""Production trainer adapter for already-materialized panel-LTR datasets."""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from renquant_artifacts import sha256_file

from .ltr_model import PanelLTRModel
from .purged_cv import (
    CombinatorialPurgedCV,
    PurgedKFold,
    cross_validated_ic,
    cross_validated_ic_cpcv,
)


def train_panel_ltr_artifact(
    dataset: Any,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train a real XGBoost panel-LTR artifact from a materialized panel."""
    spec = _dataset_spec(dataset, config)
    _require_xgboost_backend(config)
    lookahead = _required_int(config, "lookahead_days")
    embargo = int(config.get("cv_embargo_days", lookahead))
    if embargo < lookahead:
        raise ValueError(f"cv_embargo_days={embargo} < lookahead_days={lookahead}")
    config_fingerprint = _required_str(config, "config_fingerprint")
    train_run_id = _required_str(config, "train_run_id")

    cv_result = _cross_validate(spec, config, lookahead, embargo)
    _require_finite_cv(cv_result)

    model = PanelLTRModel(
        params=dict(config.get("xgb_params", {})),
        monotone_constraints=dict(config.get("monotone_constraints", {})),
    )
    train_result = model.train(
        spec.panel,
        spec.group_sizes,
        feature_cols=spec.feature_cols,
        label_col=spec.label_col,
        weight_col=spec.weight_col,
        num_boost_round=int(config.get("num_boost_round", 100)),
    )

    artifact_id = str(config.get("artifact_id") or f"panel-ltr-{train_run_id}")
    artifact_uri = str(config.get("artifact_uri") or f"object://renquant-artifacts/{artifact_id}.json")
    output_path = output_dir / f"{artifact_id}.json"
    metadata = {
        "artifact_id": artifact_id,
        "model_family": "gbdt-panel-ltr",
        "strategy": config.get("strategy", "renquant_104"),
        "promotion_status": config.get("promotion_status", "candidate"),
        "trained_date": str(date.today()),
        "config_fingerprint": config_fingerprint,
        "panel_shape": {"rows": int(len(spec.panel)), "cols": int(len(spec.feature_cols))},
        "lookahead_days": lookahead,
        "train_run_id": train_run_id,
        "oos_mean_ic": float(cv_result["mean_ic"]),
        "oos_std_ic": float(cv_result["std_ic"]),
        "oos_per_fold_ic": [float(x) for x in cv_result["per_fold_ic"]],
        "cv_method": _cv_method_label(config),
        "cv_embargo_days": embargo,
        "training_train_ic": float(train_result["train_ic"]),
        "feature_importances": train_result.get("feature_importances", {}),
        "source_contract": "materialized_panel_v1",
    }
    model.save(output_path, metadata=metadata)
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    artifact.update({
        "fingerprint": sha256_file(output_path),
        "uri": artifact_uri,
        "local_artifact_path": str(output_path),
    })
    calibrator = {
        "artifact_id": f"{artifact_id}-calibrator",
        "kind": "identity-panel-score-calibrator",
        "promotion_status": "candidate",
        "trained_date": str(date.today()),
    }
    return artifact, calibrator


def validate_panel_ltr_artifact(
    artifact: dict[str, Any],
    dataset: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return the metrics record stamped by ``train_panel_ltr_artifact``."""
    mean_ic = float(artifact["oos_mean_ic"])
    min_ic = float(config.get("acceptance_min_oos_ic", 0.0))
    return {
        "accepted": bool(mean_ic >= min_ic),
        "oos_mean_ic": mean_ic,
        "oos_std_ic": float(artifact["oos_std_ic"]),
        "oos_per_fold_ic": list(artifact["oos_per_fold_ic"]),
        "train_ic": float(artifact.get("training_train_ic", float("nan"))),
        "cv_method": artifact["cv_method"],
        "cv_embargo_days": int(artifact["cv_embargo_days"]),
        "lookahead_days": int(artifact["lookahead_days"]),
    }


class _PanelDatasetSpec:
    def __init__(
        self,
        panel: pd.DataFrame,
        group_sizes: np.ndarray,
        feature_cols: list[str],
        label_col: str,
        weight_col: str | None,
    ) -> None:
        self.panel = panel
        self.group_sizes = group_sizes
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.weight_col = weight_col


def _dataset_spec(dataset: Any, config: dict[str, Any]) -> _PanelDatasetSpec:
    if not isinstance(dataset, dict):
        raise ValueError("GBDT dataset must be a dict with panel, group_sizes, and feature_cols")
    panel = dataset.get("panel")
    if not isinstance(panel, pd.DataFrame):
        raise ValueError("GBDT dataset missing materialized pandas panel")
    group_sizes_raw = dataset.get("group_sizes")
    if group_sizes_raw is None:
        raise ValueError("GBDT dataset missing group_sizes")
    group_sizes = np.asarray(group_sizes_raw, dtype=np.int32)
    feature_cols = list(config.get("feature_cols") or dataset.get("feature_cols") or [])
    if not feature_cols:
        raise ValueError("GBDT dataset/config missing feature_cols")
    label_col = str(config.get("label_col", dataset.get("label_col", "label")))
    weight_col = config.get("weight_col", dataset.get("weight_col", "weight"))
    required = ["date", label_col, *feature_cols]
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise ValueError(f"GBDT panel missing required columns: {missing}")
    if int(group_sizes.sum()) != len(panel):
        raise ValueError(f"sum(group_sizes)={int(group_sizes.sum())} != len(panel)={len(panel)}")
    return _PanelDatasetSpec(panel.copy(), group_sizes, feature_cols, label_col, weight_col)


def _cross_validate(
    spec: _PanelDatasetSpec,
    config: dict[str, Any],
    lookahead: int,
    embargo: int,
) -> dict[str, Any]:
    cv_method = str(config.get("cv_method", "purged")).strip().lower()
    cv_splits = int(config.get("cv_n_splits", 3))
    num_rounds = int(config.get("cv_num_boost_round", config.get("num_boost_round", 100)))
    xgb_params = dict(config.get("xgb_params", {}))
    monotone = dict(config.get("monotone_constraints", {}))
    panel = spec.panel

    class _Adapter:
        def __init__(self) -> None:
            self._model = PanelLTRModel(params=xgb_params, monotone_constraints=monotone)

        def fit(self, X, y, sample_weight=None):
            missing_idx = X.index.difference(panel.index)
            if len(missing_idx):
                raise KeyError(f"CV adapter received unknown panel index: {list(missing_idx)[:5]}")
            frame = X.copy()
            frame[spec.label_col] = y
            frame["date"] = panel.loc[X.index, "date"].values
            frame["weight"] = sample_weight if sample_weight is not None else 1.0
            frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
            groups = frame.groupby("date", sort=True).size().to_numpy(dtype=np.int32)
            self._model.train(
                frame,
                groups,
                feature_cols=list(X.columns),
                label_col=spec.label_col,
                weight_col="weight",
                num_boost_round=num_rounds,
            )

        def predict(self, X):
            return self._model.predict(X.copy()).values

    if cv_method == "cpcv":
        cv = CombinatorialPurgedCV(
            n_splits=cv_splits,
            n_test_groups=int(config.get("cv_n_test_groups", 2)),
            embargo_days=embargo,
            lookahead_days=lookahead,
        )
        return cross_validated_ic_cpcv(
            _Adapter,
            spec.panel,
            spec.feature_cols,
            spec.label_col,
            cv,
            weight_col=spec.weight_col,
        )
    if cv_method != "purged":
        raise ValueError(f"unsupported cv_method for GBDT trainer: {cv_method!r}")
    cv = PurgedKFold(n_splits=cv_splits, embargo_days=embargo, lookahead_days=lookahead)
    return cross_validated_ic(
        _Adapter,
        spec.panel,
        spec.feature_cols,
        spec.label_col,
        cv,
        weight_col=spec.weight_col,
    )


def _require_xgboost_backend(config: dict[str, Any]) -> None:
    backend = str(config.get("backend", "xgboost")).strip().lower()
    if backend != "xgboost":
        raise ValueError(f"unsupported backend for renquant-model-gbdt: {backend!r}")


def _required_str(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"model_config missing required {key}")
    return str(value)


def _required_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if value is None:
        raise ValueError(f"model_config missing required {key}")
    return int(value)


def _require_finite_cv(cv_result: dict[str, Any]) -> None:
    for key in ("mean_ic", "std_ic"):
        value = cv_result.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"CV result {key} must be finite")
    folds = cv_result.get("per_fold_ic")
    if not folds or not all(math.isfinite(float(v)) for v in folds):
        raise ValueError("CV result per_fold_ic must be non-empty and finite")


def _cv_method_label(config: dict[str, Any]) -> str:
    method = str(config.get("cv_method", "purged")).strip().lower()
    return "combinatorial-purged-walk-forward" if method == "cpcv" else "purged-walk-forward"
