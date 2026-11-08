"""Artificial OHLCV bars, generated independently of every historical dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 1729


def synthetic_market_data() -> pd.DataFrame:
    """Five complete, ordinary NYSE sessions with invented prices and volumes.

    This is a software exercise. It does not model market microstructure or
    reproduce any empirical result. No parameter search is involved.
    """
    rng = np.random.default_rng(SEED)
    frames = []
    for day, date in enumerate(("2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10")):
        t = np.arange(390, dtype=float)
        close = 100 + day * 0.3 + 0.004 * t + 0.7 * np.sin(t / 23) + rng.normal(0, 0.02, len(t))
        opening = np.r_[close[0], close[:-1]]
        spread = 0.04 + 0.02 * (1 + np.sin(t / 31))
        frames.append(pd.DataFrame({
            "timestamp": pd.date_range(date + "T14:30:00Z", periods=390, freq="min"),
            "open": opening,
            "high": np.maximum(opening, close) + spread,
            "low": np.minimum(opening, close) - spread,
            "close": close,
            "volume": rng.integers(100, 1000, len(t)),
        }))
    return pd.concat(frames, ignore_index=True)
