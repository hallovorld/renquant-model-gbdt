"""XGBoost panel learning-to-rank model."""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


_THREAD_COUNT = str(os.cpu_count() or 4)
for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_key, _THREAD_COUNT)


DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "rank:pairwise",
    "eta": 0.05,
    "max_depth": 4,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,
    "alpha": 0.5,
    "tree_method": "hist",
    "nthread": int(os.environ.get("OMP_NUM_THREADS", _THREAD_COUNT)),
    "verbosity": 0,
    "seed": 42,
}


def _xgb():
    import xgboost as xgb  # noqa: PLC0415

    return xgb


def _bucketize_labels_per_group(
    y: np.ndarray,
    group_sizes: np.ndarray,
    n_buckets: int = 11,
) -> np.ndarray:
    """Map continuous labels to integer relevance buckets per date group."""
    out = np.zeros(len(y), dtype=np.int32)
    offset = 0
    for gs in group_sizes:
        gs_int = int(gs)
        if gs_int <= 0:
            continue
        slice_y = y[offset:offset + gs_int]
        ranks = np.argsort(np.argsort(slice_y, kind="stable"), kind="stable")
        buckets = (ranks * n_buckets) // gs_int if gs_int >= n_buckets else ranks
        out[offset:offset + gs_int] = np.clip(buckets, 0, n_buckets - 1).astype(np.int32)
        offset += gs_int
    return out


def _mean_ic(
    panel: pd.DataFrame,
    preds: np.ndarray,
    label_col: str,
    date_col: str = "date",
) -> float:
    for col in (date_col, label_col):
        if col not in panel.columns:
            raise KeyError(f"_mean_ic: panel missing required column {col!r}")
    df = pd.DataFrame({
        "date": panel[date_col].values,
        "pred": preds,
        "label": panel[label_col].values,
    })
    ics: list[float] = []
    for _, group in df.groupby("date", sort=False):
        y = group["label"].to_numpy()
        p = group["pred"].to_numpy()
        if (
            len(y) < 2
            or np.allclose(y, y[0], rtol=0, atol=1e-12)
            or np.allclose(p, p[0], rtol=0, atol=1e-12)
        ):
            continue
        rho, _ = spearmanr(p, y)
        if not np.isnan(rho):
            ics.append(float(rho))
    return float(np.mean(ics)) if ics else float("nan")


class PanelLTRModel:
    """XGBoost ranker over a date-grouped cross-sectional panel."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        monotone_constraints: dict[str, int] | None = None,
    ) -> None:
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.monotone_constraints = dict(monotone_constraints or {})
        self.booster: Any | None = None
        self.feature_cols: list[str] = []
        self.best_iter: int | None = None

    def train(
        self,
        panel: pd.DataFrame,
        group_sizes: np.ndarray,
        feature_cols: list[str],
        label_col: str = "label",
        weight_col: str | None = "weight",
        num_boost_round: int = 100,
        early_stopping_rounds: int | None = None,
        eval_panel: pd.DataFrame | None = None,
        eval_group_sizes: np.ndarray | None = None,
    ) -> dict[str, Any]:
        _validate_panel_inputs(panel, group_sizes, feature_cols, label_col)
        self.feature_cols = list(feature_cols)
        xgb = _xgb()
        X = panel[self.feature_cols].to_numpy(dtype=np.float32)
        y_raw = panel[label_col].to_numpy(dtype=float)
        objective = str(self.params.get("objective", "rank:pairwise")).lower()
        y = (
            _bucketize_labels_per_group(y_raw, group_sizes)
            if objective in {"rank:ndcg", "rank:map", "rank:gain"}
            else y_raw
        )

        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_group(group_sizes)
        if weight_col and weight_col in panel.columns:
            dtrain.set_weight(_group_mean_weights(panel[weight_col].to_numpy(dtype=float), group_sizes))

        deval = None
        if eval_panel is not None or eval_group_sizes is not None:
            if eval_panel is None or eval_group_sizes is None:
                raise ValueError("eval_panel and eval_group_sizes must be provided together")
            _validate_panel_inputs(eval_panel, eval_group_sizes, self.feature_cols, label_col)
            eval_y_raw = eval_panel[label_col].to_numpy(dtype=float)
            eval_y = (
                _bucketize_labels_per_group(eval_y_raw, eval_group_sizes)
                if objective in {"rank:ndcg", "rank:map", "rank:gain"}
                else eval_y_raw
            )
            deval = xgb.DMatrix(eval_panel[self.feature_cols].to_numpy(dtype=np.float32), label=eval_y)
            deval.set_group(eval_group_sizes)

        params = dict(self.params)
        if self.monotone_constraints:
            unknown = [key for key in self.monotone_constraints if key not in self.feature_cols]
            if unknown:
                raise ValueError(f"monotone constraints reference unknown features: {unknown}")
            signs = [int(self.monotone_constraints.get(col, 0)) for col in self.feature_cols]
            if any(signs):
                params["monotone_constraints"] = "(" + ",".join(str(sign) for sign in signs) + ")"

        self.booster = xgb.train(
            params,
            dtrain,
            num_boost_round=int(num_boost_round),
            verbose_eval=False,
        )
        self.best_iter = getattr(self.booster, "best_iteration", int(num_boost_round) - 1)
        train_preds = self.booster.predict(dtrain)
        result: dict[str, Any] = {
            "best_iter": self.best_iter,
            "train_ic": _mean_ic(panel, train_preds, label_col),
            "feature_importances": _named_gain_importance(self.booster, self.feature_cols),
        }
        if deval is not None and eval_panel is not None:
            result["eval_ic"] = _mean_ic(eval_panel, self.booster.predict(deval), label_col)
        return result

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        if self.booster is None:
            raise RuntimeError("PanelLTRModel.predict called before train/load")
        missing = [col for col in self.feature_cols if col not in panel.columns]
        if missing:
            raise ValueError(f"PanelLTRModel.predict missing feature columns: {missing}")
        dmatrix = _xgb().DMatrix(panel[self.feature_cols].to_numpy(dtype=np.float32))
        return pd.Series(self.booster.predict(dmatrix), index=panel.index, name="panel_score")

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        if self.booster is None:
            raise RuntimeError("PanelLTRModel.save called before train")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "kind": "panel_ltr_xgboost",
            "trained_date": str(date.today()),
            "feature_cols": list(self.feature_cols),
            "params": self.params,
            "best_iter": self.best_iter,
            "booster_raw_json": bytes(self.booster.save_raw(raw_format="json")).decode("utf-8"),
        }
        if metadata:
            payload.update({k: v for k, v in metadata.items() if k not in payload})
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PanelLTRModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(params=payload.get("params"))
        model.feature_cols = list(payload["feature_cols"])
        model.best_iter = payload.get("best_iter")
        booster = _xgb().Booster()
        booster.load_model(bytearray(payload["booster_raw_json"].encode("utf-8")))
        model.booster = booster
        return model


def _validate_panel_inputs(
    panel: pd.DataFrame,
    group_sizes: np.ndarray,
    feature_cols: list[str],
    label_col: str,
) -> None:
    if "date" not in panel.columns:
        raise ValueError("panel missing required date column")
    if label_col not in panel.columns:
        raise ValueError(f"panel missing required label column {label_col!r}")
    if not feature_cols:
        raise ValueError("feature_cols must be non-empty")
    missing_features = [col for col in feature_cols if col not in panel.columns]
    if missing_features:
        raise ValueError(f"panel missing feature columns: {missing_features}")
    if int(np.sum(group_sizes)) != len(panel):
        raise ValueError(
            f"sum(group_sizes)={int(np.sum(group_sizes))} != len(panel)={len(panel)}"
        )


def _group_mean_weights(row_weights: np.ndarray, group_sizes: np.ndarray) -> np.ndarray:
    weights = np.empty(len(group_sizes), dtype=float)
    offset = 0
    for idx, group_size in enumerate(group_sizes):
        size = int(group_size)
        weights[idx] = float(np.mean(row_weights[offset:offset + size])) if size > 0 else 1.0
        offset += size
    return weights


def _named_gain_importance(booster: Any, feature_cols: list[str]) -> dict[str, float]:
    try:
        raw = booster.get_score(importance_type="gain")
    except Exception:
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        if key.startswith("f") and key[1:].isdigit():
            idx = int(key[1:])
            if 0 <= idx < len(feature_cols):
                out[feature_cols[idx]] = float(value)
    return out
