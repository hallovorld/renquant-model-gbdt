"""Scorer Protocol conformance + load() integration tests."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

xgb = pytest.importorskip("xgboost")

from renquant_common import (  # noqa: E402
    ArtifactManifest,
    OOSEvidence,
    Scorer,
    load_scorer,
)

from renquant_model_gbdt.scorer import XGBoostPanelScorer, load  # noqa: E402


def _train_tiny_booster():
    dtrain = xgb.DMatrix(
        [[1.0, 0.2], [0.8, 0.1], [-1.0, 0.0], [-0.7, -0.1]],
        label=[1.0, 0.8, -1.0, -0.8],
    )
    return xgb.train(
        {
            "objective": "reg:squarederror",
            "max_depth": 1,
            "eta": 1.0,
            "nthread": 1,
            "verbosity": 0,
            "seed": 7,
        },
        dtrain,
        num_boost_round=4,
        verbose_eval=False,
    )


def _write_artifact(path: Path) -> ArtifactManifest:
    booster = _train_tiny_booster()
    payload = {
        "kind": "panel_ltr_xgboost",
        "feature_cols": ["alpha_1", "alpha_2"],
        "booster_raw_json": bytes(
            booster.save_raw(raw_format="json")
        ).decode("utf-8"),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return ArtifactManifest(
        kind="panel_ltr_xgboost",
        family="gbdt",
        artifact_uri=f"file://{path}",
        feature_fingerprint="sha256:test",
        config_fingerprint="sha256:test-cfg",
        training_data_fingerprint="sha256:test-data",
        trained_at=datetime(2026, 5, 27, tzinfo=timezone.utc),
        lookahead_days=5,
        oos_evidence=OOSEvidence(
            mean_ic=0.04,
            std_ic=0.01,
            per_fold_ic=(0.03, 0.05),
            cv_method="purged_kfold",
            embargo_days=5,
        ),
        owner_repo="renquant-model-gbdt",
    )


def test_xgboost_scorer_satisfies_protocol(tmp_path: Path) -> None:
    manifest = _write_artifact(tmp_path / "x.json")
    scorer = load(manifest)
    assert isinstance(scorer, Scorer)
    assert isinstance(scorer, XGBoostPanelScorer)
    assert scorer.feature_cols == ["alpha_1", "alpha_2"]
    assert scorer.feature_fingerprint().startswith("0") or len(
        scorer.feature_fingerprint()
    ) == 64
    rows = {"AAPL": {"alpha_1": 1.0, "alpha_2": 0.2}, "MSFT": {"alpha_1": -1.0, "alpha_2": 0.0}}
    preds = scorer.predict_rows(rows)
    assert set(preds) == {"AAPL", "MSFT"}
    assert preds["AAPL"] > preds["MSFT"]
    assert scorer.predict_variance(rows) is None


def test_load_via_entry_point_round_trip(tmp_path: Path) -> None:
    """End-to-end: common.load_scorer discovers panel_ltr_xgboost via entry point."""
    manifest = _write_artifact(tmp_path / "x.json")
    scorer = load_scorer(manifest)
    assert isinstance(scorer, Scorer)
    assert isinstance(scorer, XGBoostPanelScorer)


def test_load_rejects_kind_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    payload = {
        "kind": "patchtst_panel",  # mismatch
        "feature_cols": ["a"],
        "booster_raw_json": "x",
    }
    path.write_text(json.dumps(payload))
    manifest = _write_artifact(tmp_path / "manifest.json")
    # Swap artifact_uri to the mismatched payload
    manifest = manifest.model_copy(update={"artifact_uri": f"file://{path}"})
    with pytest.raises(ValueError, match="payload kind"):
        load(manifest)


def test_load_rejects_missing_artifact_file(tmp_path: Path) -> None:
    manifest = _write_artifact(tmp_path / "real.json")
    manifest = manifest.model_copy(
        update={"artifact_uri": f"file://{tmp_path / 'missing.json'}"}
    )
    with pytest.raises(FileNotFoundError, match="not found"):
        load(manifest)


def test_load_rejects_missing_feature_cols(tmp_path: Path) -> None:
    path = tmp_path / "x.json"
    booster = _train_tiny_booster()
    payload = {
        "kind": "panel_ltr_xgboost",
        "booster_raw_json": bytes(
            booster.save_raw(raw_format="json")
        ).decode("utf-8"),
        # feature_cols intentionally absent
    }
    path.write_text(json.dumps(payload))
    manifest = _write_artifact(tmp_path / "manifest.json")
    manifest = manifest.model_copy(update={"artifact_uri": f"file://{path}"})
    with pytest.raises(ValueError, match="feature_cols"):
        load(manifest)


def test_predict_rows_rejects_non_finite_feature(tmp_path: Path) -> None:
    manifest = _write_artifact(tmp_path / "x.json")
    scorer = load(manifest)
    with pytest.raises(ValueError, match="non-finite"):
        scorer.predict_rows(
            {"AAPL": {"alpha_1": float("nan"), "alpha_2": 0.0}}
        )
