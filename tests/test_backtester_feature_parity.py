from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from trading_lab.backtesting.mean_reversion import BacktesterMRCandidateFiltered
from trading_lab.features import LEGACY_REPLAY_FULL_CSV_CUMULATIVE_VWAP_V1


def legacy_prepared_frame(
    source: pd.DataFrame,
    *,
    atr_window: int,
    atr_q_low: float,
    atr_q_high: float,
    trend_fast: int,
    trend_slow: int,
) -> pd.DataFrame:
    """The pre-Stage-1B replay feature path, retained only as a parity oracle."""
    df = source.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df[df["volume"] > 0].copy().reset_index(drop=True)

    df["vwap_used"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + (gain / loss)))
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ),
    )
    df["atr"] = tr.rolling(14).mean()

    indexed = df.set_index("timestamp")
    df15 = (
        indexed.resample("15min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["close"])
    )
    df15["ema_fast"] = df15["close"].ewm(span=trend_fast, adjust=False).mean()
    df15["ema_slow"] = df15["close"].ewm(span=trend_slow, adjust=False).mean()
    df15["trend15"] = np.where(df15["ema_fast"] > df15["ema_slow"], 1, -1)
    df15["trend15_lagged"] = df15["trend15"].shift(1)
    indexed["trend15"] = (
        df15["trend15_lagged"].reindex(indexed.index, method="ffill").fillna(0).astype(int)
    )
    df = indexed.reset_index()
    df["atr_q_low_band"] = df["atr"].rolling(atr_window).quantile(atr_q_low)
    df["atr_q_high_band"] = df["atr"].rolling(atr_window).quantile(atr_q_high)
    return df


def legacy_signal(row: pd.Series) -> int:
    if pd.isna(row.atr) or row.atr <= 0:
        return 0
    if pd.isna(row.vwap_used) or pd.isna(row.rsi):
        return 0
    if pd.isna(row.trend15) or row.trend15 == 0:
        return 0
    vwap_dist = (row.close - row.vwap_used) / row.atr
    if row.trend15 == 1 and vwap_dist < -0.8 and row.rsi < 30:
        return 1
    if row.trend15 == -1 and vwap_dist > 0.8 and row.rsi > 70:
        return -1
    return 0


class BacktesterFeatureParityTests(unittest.TestCase):
    def test_legacy_replay_features_and_signal_are_preserved(self) -> None:
        timestamps = pd.date_range("2026-06-01T13:30:00Z", periods=540, freq="min")
        base = 500 + np.sin(np.arange(540) / 7) * 3 + np.arange(540) * 0.01
        source = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": base + 0.05,
                "high": base + 0.75,
                "low": base - 0.75,
                "close": base,
                "volume": np.where(np.arange(540) % 53 == 0, 0, 100 + np.arange(540) % 20),
            }
        )
        expected = legacy_prepared_frame(
            source,
            atr_window=200,
            atr_q_low=0.55,
            atr_q_high=0.95,
            trend_fast=21,
            trend_slow=50,
        )

        with patch("trading_lab.backtesting.mean_reversion.pd.read_csv", return_value=source):
            backtester = BacktesterMRCandidateFiltered(file="in_memory_bars.csv", verbose=False)

        actual = backtester.df
        self.assertEqual(len(actual), len(expected))
        self.assertTrue((actual["volume"] > 0).all())
        self.assertEqual(
            set(actual["feature_profile"]),
            {LEGACY_REPLAY_FULL_CSV_CUMULATIVE_VWAP_V1.name},
        )
        for column in ("vwap_used", "rsi", "atr", "atr_q_low_band", "atr_q_high_band", "trend15"):
            np.testing.assert_allclose(actual[column], expected[column], equal_nan=True)

        expected_signals = [legacy_signal(row) for _, row in expected.iterrows()]
        actual_signals = [backtester.signal(row) for _, row in actual.iterrows()]
        self.assertEqual(actual_signals, expected_signals)

        # The helper diagnostic folds in ATR-band eligibility; the execution loop
        # still owns that gate. Its raw_signal must therefore match the loop's
        # ATR-gated effective signal, while signal() itself remains gate-agnostic.
        in_band = actual["atr_band_result"].eq("IN_BAND")
        loop_effective_signals = np.where(in_band, actual_signals, 0).tolist()
        self.assertEqual(
            actual["raw_signal"].astype(int).tolist(),
            loop_effective_signals,
        )



if __name__ == "__main__":
    unittest.main()
