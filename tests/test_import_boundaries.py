from __future__ import annotations

import importlib
import sys


def test_model_gbdt_root_import_does_not_pull_execution_runtime() -> None:
    importlib.import_module("renquant_model_gbdt")

    forbidden_prefixes = (
        "alpaca",
        "backtesting",
        "ib_insync",
        "kernel",
        "live",
        "renquant_execution",
    )
    offenders = sorted(
        name for name in sys.modules
        if name in forbidden_prefixes or name.startswith(forbidden_prefixes)
    )
    assert offenders == []
