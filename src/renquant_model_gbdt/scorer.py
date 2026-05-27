"""XGBoost panel-LTR scorer implementing the renquant-common Scorer Protocol.

This module is the production runtime adapter for the GBDT family. It is
registered via an entry point in ``pyproject.toml``::

    [project.entry-points."renquant_common.scorers"]
    panel_ltr_xgboost = "renquant_model_gbdt.scorer:load"

Consumers (``renquant-pipeline``, ``renquant-backtesting``) reach this
loader exclusively through :func:`renquant_common.load_scorer` — they must
not import this module directly. Per RFC §"Bootstrap Drift Audit" item 1,
the prior leak where ``XGBoostPanelScorer`` lived in
``renquant-pipeline/xgboost_scorer.py`` is closed by this relocation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from renquant_common import ArtifactManifest


@dataclass
class XGBoostPanelScorer:
    """Scorer impl backed by an XGBoost Booster.

    Satisfies the runtime-checkable ``renquant_common.Scorer`` Protocol:
    exposes ``feature_cols``, ``feature_fingerprint()``,
    ``predict_rows()``, ``predict_variance()``.
    """

    feature_cols: list[str]
    booster: Any
    _fingerprint: str = ""

    def feature_fingerprint(self) -> str:
        """Stable hash of the feature column list (transform v1)."""
        if not self._fingerprint:
            payload = json.dumps(
                {"feature_cols": list(self.feature_cols), "transform_version": 1},
                sort_keys=True,
            ).encode("utf-8")
            object.__setattr__(
                self, "_fingerprint", hashlib.sha256(payload).hexdigest()
            )
        return self._fingerprint

    def predict_rows(
        self, rows: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        if not rows:
            return {}
        xgb = _xgb()
        tickers = list(rows)
        matrix = [
            [
                _as_float(rows[ticker][col], ticker=ticker, col=col)
                for col in self.feature_cols
            ]
            for ticker in tickers
        ]
        preds = self.booster.predict(xgb.DMatrix(matrix))
        return {ticker: float(pred) for ticker, pred in zip(tickers, preds, strict=True)}

    def predict_variance(
        self, rows: dict[str, dict[str, float]]
    ) -> dict[str, float] | None:
        # XGBoost rank:pairwise does not produce per-row variance.
        return None


def load(manifest: ArtifactManifest) -> XGBoostPanelScorer:
    """Entry-point target invoked by ``renquant_common.load_scorer``.

    Reads the XGBoost payload referenced by ``manifest.artifact_uri`` and
    materializes a :class:`XGBoostPanelScorer`. Fails fast on missing
    files, unreadable payloads, or absent ``feature_cols`` — per
    §5.13.10, an untaken fail-closed path is dead code.
    """
    payload = _read_payload(manifest)
    if payload.get("kind") not in {None, "panel_ltr_xgboost"}:
        raise ValueError(
            f"manifest.kind={manifest.kind!r} but payload kind="
            f"{payload.get('kind')!r}"
        )
    booster_raw = payload.get("booster_raw_json")
    feature_cols = payload.get("feature_cols") or payload.get("feature_columns")
    if not booster_raw:
        raise ValueError(
            "XGBoost panel artifact missing booster_raw_json payload"
        )
    if not isinstance(feature_cols, list) or not feature_cols:
        raise ValueError(
            "XGBoost panel artifact missing non-empty feature_cols"
        )
    booster = _xgb().Booster()
    booster.load_model(bytearray(str(booster_raw).encode("utf-8")))
    return XGBoostPanelScorer(
        feature_cols=[str(col) for col in feature_cols],
        booster=booster,
    )


def _read_payload(manifest: ArtifactManifest) -> dict[str, Any]:
    path = _resolve_local_path(manifest.artifact_uri)
    if path is None:
        raise ValueError(
            f"unsupported artifact_uri scheme for local read: "
            f"{manifest.artifact_uri!r}"
        )
    if not path.exists():
        raise FileNotFoundError(f"panel artifact file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"panel artifact file must contain a JSON object: {path}"
        )
    return payload


def _resolve_local_path(uri: str) -> Path | None:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(parsed.path).expanduser()
    if parsed.scheme == "":
        return Path(uri).expanduser()
    return None


def _xgb():
    import xgboost as xgb  # noqa: PLC0415

    return xgb


def _as_float(value: Any, *, ticker: str, col: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"non-numeric feature {col!r} for {ticker}: {value!r}"
        ) from exc
    if out != out or out in (float("inf"), float("-inf")):
        raise ValueError(
            f"non-finite feature {col!r} for {ticker}: {value!r}"
        )
    return out
