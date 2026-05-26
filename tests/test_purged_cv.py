from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd
import pytest

from renquant_model_gbdt.purged_cv import (
    CombinatorialPurgedCV,
    PurgedKFold,
    cross_validated_ic,
    cross_validated_ic_cpcv,
)

from panel_fixtures import make_easy_panel


class PerfectModel:
    def fit(self, X, y, sample_weight=None):
        pass

    def predict(self, X):
        return X["_y_"].to_numpy()


def test_purged_kfold_each_row_in_exactly_one_test_fold() -> None:
    panel, _, _ = make_easy_panel(n_dates=30, n_tickers=4, seed=10)
    counts = np.zeros(len(panel), dtype=int)

    for _, test_idx in PurgedKFold(n_splits=5, embargo_days=0, lookahead_days=1).split(panel):
        counts[test_idx] += 1

    assert (counts == 1).all()


def test_purge_counts_trading_bars_not_calendar_days() -> None:
    panel, _, _ = make_easy_panel(n_dates=60, n_tickers=3, seed=11)
    unique_dates = sorted(set(pd.to_datetime(panel["date"]).values))
    lookahead = 10

    for train_idx, test_idx in PurgedKFold(
        n_splits=5,
        embargo_days=0,
        lookahead_days=lookahead,
    ).split(panel):
        test_dates = sorted(set(pd.to_datetime(panel.iloc[test_idx]["date"]).values))
        train_dates = set(pd.to_datetime(panel.iloc[train_idx]["date"]).values)
        test_start_pos = unique_dates.index(test_dates[0])
        for offset in range(1, lookahead + 1):
            pos = test_start_pos - offset
            if pos >= 0:
                assert unique_dates[pos] not in train_dates


def test_embargo_removes_post_fold_bars() -> None:
    panel, _, _ = make_easy_panel(n_dates=40, n_tickers=3, seed=12)
    unique_dates = sorted(set(pd.to_datetime(panel["date"]).values))

    for train_idx, test_idx in PurgedKFold(n_splits=4, embargo_days=3, lookahead_days=1).split(panel):
        test_dates = sorted(set(pd.to_datetime(panel.iloc[test_idx]["date"]).values))
        train_dates = set(pd.to_datetime(panel.iloc[train_idx]["date"]).values)
        test_end_pos = unique_dates.index(test_dates[-1])
        for offset in range(1, 4):
            pos = test_end_pos + offset
            if pos < len(unique_dates):
                assert unique_dates[pos] not in train_dates


def test_cross_validated_ic_reports_fold_metrics() -> None:
    panel, _, _ = make_easy_panel(n_dates=36, n_tickers=8, seed=13)
    panel["_y_"] = panel["label"]

    result = cross_validated_ic(
        PerfectModel,
        panel,
        feature_cols=["_y_"],
        label_col="label",
        cv=PurgedKFold(n_splits=4, embargo_days=2, lookahead_days=2),
    )

    assert result["mean_ic"] > 0.99
    assert len(result["per_fold_ic"]) == 4
    assert len(result["per_fold_ic_series"]) == 4


def test_cpcv_returns_quantiles() -> None:
    panel, _, _ = make_easy_panel(n_dates=36, n_tickers=6, seed=14)
    panel["_y_"] = panel["label"]

    result = cross_validated_ic_cpcv(
        PerfectModel,
        panel,
        feature_cols=["_y_"],
        label_col="label",
        cv=CombinatorialPurgedCV(n_splits=6, n_test_groups=2, embargo_days=1, lookahead_days=2),
    )

    assert len(result["per_fold_ic"]) == comb(6, 2)
    assert result["quantiles"]["q05"] <= result["quantiles"]["q50"] <= result["quantiles"]["q95"]


def test_purged_kfold_rejects_invalid_split_count() -> None:
    panel, _, _ = make_easy_panel(n_dates=10, n_tickers=4, seed=15)

    with pytest.raises(ValueError, match="n_splits"):
        list(PurgedKFold(n_splits=1).split(panel))
