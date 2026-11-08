"""Synthetic semantic examples for all preserved strategy modules, no historical runs."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from trading_lab.strategies.s1_downside_drift import BacktesterChallengerLocked
from trading_lab.strategies.x1 import c, e, f, g, h, i, l


def artificial_sessions(anchors, second_open=100.0):
    """Two full fictional sessions; day one establishes a 100.0 prior close."""
    frames = []
    for day, date in enumerate(("2025-01-06", "2025-01-07")):
        prices = np.full(390, 100.0) if not day else np.interp(np.arange(390), list(anchors), list(anchors.values()))
        openings = np.r_[prices[0] if not day else second_open, prices[:-1]]
        frame = pd.DataFrame({
            "timestamp": pd.date_range(date + "T14:30:00Z", periods=390, freq="min"),
            "open": openings, "close": prices,
            "high": np.maximum(openings, prices) + 0.01,
            "low": np.minimum(openings, prices) - 0.01,
            "volume": np.full(390, 100),
        })
        frame["vwap"] = (frame.close * frame.volume).cumsum() / frame.volume.cumsum()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class PreservedStrategiesTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "artificial.csv"

    def tearDown(self):
        self.directory.cleanup()

    def save(self, anchors, second_open=100.0):
        artificial_sessions(anchors, second_open).to_csv(self.path, index=False)
        return str(self.path)

    def test_s1_short_trigger_time_exit_and_costs(self):
        path = self.save({0: 100.0, 300: 99.0, 389: 98.8})
        engine = BacktesterChallengerLocked(path, verbose=False, commission_per_share=0.01)
        engine.run()
        self.assertGreaterEqual(len(engine.trade_log), 1)
        trade = engine.trade_log[0]
        self.assertEqual(trade["side"], -1)
        self.assertEqual(pd.Timestamp(trade["entry_ts"]).tz_convert("America/New_York").strftime("%H:%M"), "15:01")
        self.assertEqual(trade["reason"], "time_exit")
        self.assertEqual(engine.position, 0)

    def test_x1_f_gap_failure_short(self):
        path = self.save({0: 100.3, 60: 100.0, 150: 99.9, 389: 99.9}, 100.3)
        trades, summary = f.BacktesterX1GapUpFailedHoldContinuation(path, verbose=False).run()
        self.assertEqual(summary["trades"], 1)
        self.assertEqual(trades.iloc[0]["side"], "short")
        self.assertEqual(trades.iloc[0]["exit_reason"], "time")

    def test_x1_g_flat_strength_one_share_cost_accounting(self):
        path = self.save({0: 100.0, 60: 100.2, 150: 100.3, 389: 100.3})
        summary, trades = g.run_backtest(data_path=path, cost_bps=1)
        self.assertEqual(summary["trades"], 1)
        trade = trades.iloc[0]
        self.assertAlmostEqual(trade["pnl"], trade["exit_1200"] - trade["entry_1031"] - trade["entry_1031"] * 0.0002)

    def test_x1_h_flat_weakness_short(self):
        path = self.save({0: 100.0, 60: 99.8, 150: 99.7, 389: 99.7})
        result = h.BacktesterX1FlatOpenOrderlyWeaknessDrift(path, verbose=False).run()
        self.assertEqual(result["summary"]["trades"], 1)
        self.assertGreater(result["trade_log"].iloc[0]["shares"], 0)

    def test_x1_i_gap_down_recovery_long(self):
        path = self.save({0: 99.9, 15: 99.75, 45: 100.0, 180: 100.1, 389: 100.1}, 99.9)
        result = i.BacktesterX1SlightGapDownEarlyRecoveryDrift(path, verbose=False).run()
        self.assertEqual(result["trades"], 1)
        self.assertEqual(result["trade_log"].iloc[0]["side"], "long")

    def test_x1_l_shakeout_reclaim_uses_fixed_notional(self):
        path = self.save({0: 100.0, 20: 99.8, 90: 100.15, 180: 100.2, 389: 100.2})
        result = l.BacktesterX1NearFlatMorningShakeoutReclaimLong(path, verbose=False).run_backtest()
        self.assertEqual(result["summary"]["trades"], 1)
        trade = result["trade_log"].iloc[0]
        self.assertAlmostEqual(trade["shares"] * trade["entry_price"], 10_000)

    def test_comparators_reject_no_setup_without_inventing_trades(self):
        path = self.save({0: 100.0, 389: 100.0})
        for cls in (c.BacktesterX1OpeningDrivePullbackContinuation, e.BacktesterX1OvernightDamageRepairReclaim):
            with self.subTest(strategy=cls.__name__):
                result = cls(path, verbose=False).run()
                self.assertEqual(result["summary"]["trades"], 0)
                self.assertTrue(result.get("trade_log", result.get("trades")).empty)
