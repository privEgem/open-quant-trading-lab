from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from trading_lab.features import (
    LEGACY_LIVE_7CALENDAR_CUMULATIVE_VWAP_V1,
    LEGACY_REPLAY_FULL_CSV_CUMULATIVE_VWAP_V1,
    SESSION_REGULAR_NY_VWAP_V1,
    evaluate_raw_signal,
    generate_features,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "feature_bars.csv"


class StubNYSECalendar:
    """Small deterministic exchange schedule used without the optional package."""

    def __init__(self, sessions: dict[str, tuple[str, str]]) -> None:
        self.sessions = sessions

    def schedule(self, start_date, end_date) -> pd.DataFrame:
        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()
        rows = [
            (session_date, market_open, market_close)
            for session_date, (market_open, market_close) in self.sessions.items()
            if start <= pd.Timestamp(session_date).date() <= end
        ]
        return pd.DataFrame(
            {
                "market_open": [pd.Timestamp(market_open) for _, market_open, _ in rows],
                "market_close": [pd.Timestamp(market_close) for _, _, market_close in rows],
            },
            index=pd.DatetimeIndex([session_date for session_date, _, _ in rows]),
        )


def bars_for_timestamps(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": range(100, 100 + len(timestamps)),
            "high": range(101, 101 + len(timestamps)),
            "low": range(99, 99 + len(timestamps)),
            "close": range(100, 100 + len(timestamps)),
            "volume": 10,
        }
    )


class StrategyFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = pd.read_csv(FIXTURE_PATH)

    def test_normalization_is_stable_and_last_duplicate_wins(self) -> None:
        features = generate_features(self.fixture, LEGACY_REPLAY_FULL_CSV_CUMULATIVE_VWAP_V1)

        self.assertEqual(features["timestamp"].tolist(), sorted(features["timestamp"].tolist()))
        self.assertEqual(features["close"].tolist(), [100, 101, 103, 110])
        self.assertEqual(features["positive_volume_eligible"].tolist(), [True, False, True, True])
        self.assertEqual(features["included_bar_count"].tolist(), [1, 1, 2, 3])
        self.assertAlmostEqual(features.loc[0, "vwap_used"], 100.0)
        self.assertAlmostEqual(features.loc[1, "vwap_used"], 100.0)
        self.assertAlmostEqual(features.loc[2, "vwap_used"], 101.0)

    def test_legacy_profiles_have_distinct_explicit_metadata_but_same_formula(self) -> None:
        live = generate_features(self.fixture, LEGACY_LIVE_7CALENDAR_CUMULATIVE_VWAP_V1)
        replay = generate_features(self.fixture, LEGACY_REPLAY_FULL_CSV_CUMULATIVE_VWAP_V1)

        self.assertEqual(set(live["feature_profile"]), {LEGACY_LIVE_7CALENDAR_CUMULATIVE_VWAP_V1.name})
        self.assertEqual(set(replay["feature_profile"]), {LEGACY_REPLAY_FULL_CSV_CUMULATIVE_VWAP_V1.name})
        self.assertEqual(set(live["vwap_window"]), {"fetched_7_calendar_days"})
        self.assertEqual(set(replay["vwap_window"]), {"full_supplied_csv"})
        np.testing.assert_allclose(live["vwap_used"], replay["vwap_used"], equal_nan=True)
        self.assertTrue(replay["vwap_session_key"].isna().all())

    def test_session_vwap_resets_and_excludes_outside_regular_session(self) -> None:
        bars = pd.DataFrame(
            [
                ["2026-06-01T13:00:00Z", 90, 91, 89, 90, 100],   # 09:00 NY: excluded
                ["2026-06-01T13:30:00Z", 100, 101, 99, 100, 10],  # 09:30 NY
                ["2026-06-01T13:31:00Z", 105, 106, 104, 105, 5],  # 09:31 NY
                ["2026-06-01T20:01:00Z", 200, 201, 199, 200, 10], # 16:01 NY: excluded
                ["2026-06-02T13:30:00Z", 110, 111, 109, 110, 20], # next session
            ],
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        calendar = StubNYSECalendar(
            {
                "2026-06-01": ("2026-06-01T13:30:00Z", "2026-06-01T20:00:00Z"),
                "2026-06-02": ("2026-06-02T13:30:00Z", "2026-06-02T20:00:00Z"),
            }
        )
        with patch("trading_lab.features.mcal", object()), patch("trading_lab.features.NYSE", calendar):
            features = generate_features(bars, SESSION_REGULAR_NY_VWAP_V1)

        self.assertTrue(pd.isna(features.loc[0, "vwap_used"]))
        self.assertEqual(features.loc[1, "vwap_session_key"], "2026-06-01")
        self.assertEqual(features.loc[1, "included_bar_count"], 1)
        self.assertAlmostEqual(features.loc[2, "vwap_used"], 305 / 3)
        self.assertTrue(pd.isna(features.loc[3, "vwap_used"]))
        self.assertEqual(features.loc[4, "vwap_session_key"], "2026-06-02")
        self.assertEqual(features.loc[4, "included_bar_count"], 1)
        self.assertAlmostEqual(features.loc[4, "vwap_used"], 110.0)

    def test_session_profile_fails_closed_without_exchange_calendar_package(self) -> None:
        bars = bars_for_timestamps(["2026-06-01T13:30:00Z"])

        with patch("trading_lab.features.mcal", None), patch("trading_lab.features.NYSE", None):
            with self.assertRaisesRegex(
                RuntimeError,
                "session_regular_ny_vwap_v1 requires pandas_market_calendars",
            ):
                generate_features(bars, SESSION_REGULAR_NY_VWAP_V1)

    def test_legacy_profiles_do_not_require_exchange_calendar_package(self) -> None:
        bars = bars_for_timestamps(["2026-06-01T13:30:00Z"])

        with patch("trading_lab.features.mcal", None), patch("trading_lab.features.NYSE", None):
            for profile in (
                LEGACY_LIVE_7CALENDAR_CUMULATIVE_VWAP_V1,
                LEGACY_REPLAY_FULL_CSV_CUMULATIVE_VWAP_V1,
            ):
                features = generate_features(bars, profile)
                self.assertEqual(features.loc[0, "feature_profile"], profile.name)
                self.assertAlmostEqual(features.loc[0, "vwap_used"], 100.0)

    def test_session_profile_excludes_nyse_holiday_and_respects_early_close(self) -> None:
        calendar = StubNYSECalendar(
            {
                # July 3 is the observed Independence Day holiday in 2026.
                "2026-07-02": ("2026-07-02T13:30:00Z", "2026-07-02T17:00:00Z"),
                "2026-07-06": ("2026-07-06T13:30:00Z", "2026-07-06T20:00:00Z"),
            }
        )
        bars = bars_for_timestamps(
            [
                "2026-07-02T16:59:00Z",  # early-close session: included
                "2026-07-02T17:00:00Z",  # close boundary: excluded
                "2026-07-03T13:30:00Z",  # holiday: excluded
                "2026-07-06T13:30:00Z",  # next regular session: included
            ]
        )

        with patch("trading_lab.features.mcal", object()), patch("trading_lab.features.NYSE", calendar):
            features = generate_features(bars, SESSION_REGULAR_NY_VWAP_V1)

        self.assertEqual(features.loc[0, "vwap_session_key"], "2026-07-02")
        self.assertEqual(features.loc[0, "included_bar_count"], 1)
        self.assertTrue(pd.isna(features.loc[1, "vwap_used"]))
        self.assertTrue(pd.isna(features.loc[2, "vwap_used"]))
        self.assertEqual(features.loc[3, "vwap_session_key"], "2026-07-06")
        self.assertEqual(features.loc[3, "included_bar_count"], 1)

    def test_session_profile_uses_calendar_utc_bounds_across_dst(self) -> None:
        calendar = StubNYSECalendar(
            {
                "2026-03-06": ("2026-03-06T14:30:00Z", "2026-03-06T21:00:00Z"),
                "2026-03-09": ("2026-03-09T13:30:00Z", "2026-03-09T20:00:00Z"),
                "2026-10-30": ("2026-10-30T13:30:00Z", "2026-10-30T20:00:00Z"),
                "2026-11-02": ("2026-11-02T14:30:00Z", "2026-11-02T21:00:00Z"),
            }
        )
        bars = bars_for_timestamps(
            [
                "2026-03-06T14:30:00Z",  # before spring DST: open inclusive
                "2026-03-09T13:30:00Z",  # after spring DST: open inclusive
                "2026-10-30T13:30:00Z",  # before autumn DST: open inclusive
                "2026-11-02T14:30:00Z",  # after autumn DST: open inclusive
                "2026-11-02T21:00:00Z",  # close exclusive
            ]
        )

        with patch("trading_lab.features.mcal", object()), patch("trading_lab.features.NYSE", calendar):
            features = generate_features(bars, SESSION_REGULAR_NY_VWAP_V1)

        self.assertEqual(
            features.loc[:3, "vwap_session_key"].tolist(),
            ["2026-03-06", "2026-03-09", "2026-10-30", "2026-11-02"],
        )
        self.assertEqual(features.loc[:3, "included_bar_count"].tolist(), [1, 1, 1, 1])
        self.assertTrue(pd.isna(features.loc[4, "vwap_used"]))

    def test_indicator_columns_and_signal_diagnostic_are_present(self) -> None:
        timestamps = pd.date_range("2026-06-01T13:30:00Z", periods=240, freq="min")
        bars = pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": np.arange(240, dtype=float) + 100,
                "high": np.arange(240, dtype=float) + 101,
                "low": np.arange(240, dtype=float) + 99,
                "close": np.arange(240, dtype=float) + 100,
                "volume": 10,
            }
        )
        features = generate_features(bars, LEGACY_REPLAY_FULL_CSV_CUMULATIVE_VWAP_V1)

        expected = {
            "rsi", "atr", "atr_q_low_band", "atr_q_high_band", "atr_band_result",
            "trend15", "vwap_dist", "raw_signal", "final_signal_veto_reason",
        }
        self.assertTrue(expected.issubset(features.columns))
        self.assertIn(features.loc[239, "atr_band_result"], {"IN_BAND", "OUT_OF_BAND"})

    def test_raw_signal_diagnostic_matches_frozen_thresholds(self) -> None:
        long_row = pd.Series(
            {
                "atr": 1.0,
                "vwap_used": 101.0,
                "rsi": 29.0,
                "trend15": 1,
                "atr_q_low_band": 0.9,
                "atr_q_high_band": 1.1,
                "vwap_dist": -0.81,
            }
        )
        self.assertEqual(evaluate_raw_signal(long_row).raw_signal, 1)

        out_of_band = long_row.copy()
        out_of_band["atr_q_low_band"] = 1.1
        diagnostic = evaluate_raw_signal(out_of_band)
        self.assertEqual(diagnostic.raw_signal, 0)
        self.assertEqual(diagnostic.final_signal_veto_reason, "ATR_OUT_OF_BAND")


if __name__ == "__main__":
    unittest.main()
