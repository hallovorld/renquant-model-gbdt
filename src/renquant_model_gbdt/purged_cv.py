"""DEPRECATED — re-export shim for the canonical implementation.

``PurgedKFold`` / ``CombinatorialPurgedCV`` / IC helpers now live in
``renquant_common.purged_cv`` per RFC §"Cross-Repo Contracts → PurgedKFold".
This module re-exports them for backwards compatibility during the P3
model-repo merge and will be removed once that lands.
"""
from __future__ import annotations

import warnings

from renquant_common.purged_cv import (  # noqa: F401
    CombinatorialPurgedCV,
    PurgedKFold,
    cross_validated_ic,
    cross_validated_ic_cpcv,
    evaluate_fold_ic,
)

warnings.warn(
    "renquant_model_gbdt.purged_cv is a deprecated shim; import from "
    "renquant_common.purged_cv instead. This module will be deleted in P3.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CombinatorialPurgedCV",
    "PurgedKFold",
    "cross_validated_ic",
    "cross_validated_ic_cpcv",
    "evaluate_fold_ic",
]
