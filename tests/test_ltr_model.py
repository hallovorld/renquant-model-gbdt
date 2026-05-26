from __future__ import annotations

import numpy as np
import pytest

from renquant_model_gbdt.ltr_model import PanelLTRModel

from panel_fixtures import make_easy_panel


def test_panel_ltr_model_trains_on_easy_signal() -> None:
    panel, group_sizes, feature_cols = make_easy_panel(n_dates=28, n_tickers=8, seed=1)
    model = PanelLTRModel(params={"max_depth": 2, "eta": 0.1})

    result = model.train(panel, group_sizes, feature_cols, num_boost_round=35)

    assert model.booster is not None
    assert result["train_ic"] > 0.65
    assert result["feature_importances"]


def test_predict_before_train_raises() -> None:
    panel, _, _ = make_easy_panel(n_dates=4, n_tickers=4, seed=2)

    with pytest.raises(RuntimeError, match="before train"):
        PanelLTRModel().predict(panel)


def test_save_load_roundtrip_predictions_match(tmp_path) -> None:
    panel, group_sizes, feature_cols = make_easy_panel(n_dates=20, n_tickers=6, seed=3)
    model = PanelLTRModel(params={"max_depth": 2, "eta": 0.1})
    model.train(panel, group_sizes, feature_cols, num_boost_round=25)
    original = model.predict(panel).to_numpy()

    path = tmp_path / "model.json"
    model.save(path, metadata={"train_run_id": "unit"})
    loaded = PanelLTRModel.load(path)

    assert np.allclose(original, loaded.predict(panel).to_numpy(), atol=1e-9)


def test_bad_group_sizes_fail_before_xgboost() -> None:
    panel, _, feature_cols = make_easy_panel(n_dates=8, n_tickers=5, seed=4)

    with pytest.raises(ValueError, match="sum\\(group_sizes\\)"):
        PanelLTRModel().train(panel, np.array([1, 2, 3]), feature_cols, num_boost_round=2)


def test_missing_predict_feature_fails_closed() -> None:
    panel, group_sizes, feature_cols = make_easy_panel(n_dates=10, n_tickers=5, seed=5)
    model = PanelLTRModel()
    model.train(panel, group_sizes, feature_cols, num_boost_round=5)

    with pytest.raises(ValueError, match="missing feature"):
        model.predict(panel.drop(columns=["x2"]))
