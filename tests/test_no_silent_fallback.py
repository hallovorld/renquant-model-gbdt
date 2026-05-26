from __future__ import annotations

from pathlib import Path

import pytest

from renquant_model_gbdt import train_panel_ltr_artifact

from panel_fixtures import make_easy_panel
from test_training_pipeline_real_gbdt import _model_config


def _dataset():
    panel, group_sizes, feature_cols = make_easy_panel(n_dates=12, n_tickers=6, seed=30)
    return {"panel": panel, "group_sizes": group_sizes, "feature_cols": feature_cols}


def test_missing_feature_cols_fails_closed(tmp_path: Path) -> None:
    dataset = _dataset()
    dataset.pop("feature_cols")
    config = _model_config()
    config.pop("feature_cols")

    with pytest.raises(ValueError, match="feature_cols"):
        train_panel_ltr_artifact(dataset, config, tmp_path)


def test_missing_group_sizes_fails_closed(tmp_path: Path) -> None:
    dataset = _dataset()
    dataset.pop("group_sizes")

    with pytest.raises(ValueError, match="group_sizes"):
        train_panel_ltr_artifact(dataset, _model_config(), tmp_path)


def test_missing_date_or_label_fails_closed(tmp_path: Path) -> None:
    dataset = _dataset()
    dataset["panel"] = dataset["panel"].drop(columns=["date"])

    with pytest.raises(ValueError, match="date"):
        train_panel_ltr_artifact(dataset, _model_config(), tmp_path)

    dataset = _dataset()
    dataset["panel"] = dataset["panel"].drop(columns=["label"])
    with pytest.raises(ValueError, match="label"):
        train_panel_ltr_artifact(dataset, _model_config(), tmp_path)


def test_embargo_shorter_than_lookahead_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cv_embargo_days=1 < lookahead_days=5"):
        train_panel_ltr_artifact(
            _dataset(),
            _model_config(lookahead_days=5, cv_embargo_days=1),
            tmp_path,
        )


def test_missing_config_fingerprint_fails_closed(tmp_path: Path) -> None:
    config = _model_config()
    config.pop("config_fingerprint")

    with pytest.raises(ValueError, match="config_fingerprint"):
        train_panel_ltr_artifact(_dataset(), config, tmp_path)
