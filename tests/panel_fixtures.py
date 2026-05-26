from __future__ import annotations

import numpy as np
import pandas as pd


def make_easy_panel(n_dates: int = 36, n_tickers: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for date in pd.bdate_range("2024-01-01", periods=n_dates):
        for idx in range(n_tickers):
            x1 = rng.normal()
            x2 = rng.normal()
            label = x1 + 0.10 * rng.normal()
            rows.append({
                "date": date,
                "ticker": f"T{idx:03d}",
                "x1": x1,
                "x2": x2,
                "label": label,
                "weight": 1.0,
            })
    panel = pd.DataFrame(rows).sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
    group_sizes = panel.groupby("date", sort=True).size().to_numpy(dtype=np.int32)
    return panel, group_sizes, ["x1", "x2"]
