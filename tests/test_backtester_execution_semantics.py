from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from trading_lab.backtesting.metrics import (
    BENCHMARK_EXECUTION_METHOD_METADATA,
    _equity_df,
    _metrics,
    _trade_df,
)
from trading_lab.backtesting.mean_reversion import (
    BenchmarkSessionDataError,
    BacktesterMRCandidateFiltered,
    NYSE_EXECUTION_CALENDAR,
)


def execution_frame(
    first: dict | None = None,
    second: dict | None = None,
    *,
    timestamp: str = "2026-06-01T17:00:00Z",
) -> pd.DataFrame:
    """Create two regular-session bars with direct execution inputs."""
    first_bar = {
        "timestamp": pd.Timestamp(timestamp),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 100.0,
        "atr": 10.0,
        "atr_q_low_band": 5.0,
        "atr_q_high_band": 15.0,
    }
    second_bar = {
        **first_bar,
        "timestamp": first_bar["timestamp"] + pd.Timedelta(minutes=1),
    }
    if first:
        first_bar.update(first)
    if second:
        second_bar.update(second)
    return pd.DataFrame([first_bar, second_bar])


def execution_bars(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(timestamp),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 100.0,
                "atr": 10.0,
                "atr_q_low_band": 5.0,
                "atr_q_high_band": 15.0,
            }
            for timestamp in timestamps
        ]
    )


def calendar_close(timestamp: pd.Timestamp) -> pd.Timestamp:
    session_date = timestamp.tz_convert("America/New_York").date()
    schedule = NYSE_EXECUTION_CALENDAR.schedule(start_date=session_date, end_date=session_date)
    if schedule.empty:
        raise AssertionError(f"No NYSE session for {session_date}.")
    return pd.Timestamp(schedule.iloc[0]["market_close"])


def append_session_final_bar(frame: pd.DataFrame, **overrides) -> pd.DataFrame:
    final_bar = frame.iloc[-1].to_dict()
    final_bar["timestamp"] = calendar_close(pd.Timestamp(frame.iloc[-1].timestamp)) - pd.Timedelta(minutes=1)
    final_bar.update(overrides)
    return pd.concat([frame, pd.DataFrame([final_bar])], ignore_index=True)


def causal_entry_frame() -> pd.DataFrame:
    """Three same-session bars: signal close, next-bar open fill, final flatten."""
    return append_session_final_bar(
        execution_frame(
            {"close": 80.0},
            {"open": 150.0, "high": 155.0, "low": 151.0, "close": 155.0},
            timestamp="2026-06-01T14:00:00Z",
        ),
        open=155.0,
        high=156.0,
        low=154.0,
        close=155.0,
    )


def make_backtester(frame: pd.DataFrame, **overrides) -> BacktesterMRCandidateFiltered:
    settings = {
        "file": "synthetic_execution.csv",
        "starting_cash": 1_000.0,
        "risk_frac": 0.10,
        "rr": 1.0,
        "atr_mult": 1.0,
        "slippage_bps": 0.0,
        "commission_per_share": 0.0,
        "allowed_hours_ny": {10, 13},
        "max_position_notional_pct": 1.0,
        "verbose": False,
    }
    settings.update(overrides)
    with patch.object(
        BacktesterMRCandidateFiltered,
        "_load_and_prepare_data",
        return_value=frame.copy(),
    ):
        return BacktesterMRCandidateFiltered(**settings)


def arm_position(
    bt: BacktesterMRCandidateFiltered,
    *,
    side: int,
    entry_price: float = 100.0,
    shares: int = 10,
    stop_price: float,
    tp_price: float,
    risk_budget_dollars: float = 100.0,
    initial_stop_risk_dollars: float | None = None,
) -> None:
    bt.position = side
    bt.entry_price = entry_price
    bt.shares = shares
    bt.stop_price = stop_price
    bt.tp_price = tp_price
    bt.entry_ts = bt.df.iloc[0].timestamp
    bt.risk_budget_dollars = risk_budget_dollars
    bt.initial_stop_risk_dollars = (
        abs(entry_price - stop_price) * shares
        if initial_stop_risk_dollars is None
        else initial_stop_risk_dollars
    )


class BacktesterExecutionSemanticsTests(unittest.TestCase):
    def test_long_stop_exits_when_allowed_hours_reject_entries(self) -> None:
        bt = make_backtester(
            execution_frame({"low": 94.0}),
            allowed_hours_ny={10},
        )
        arm_position(bt, side=1, stop_price=95.0, tp_price=110.0)
        bt.signal = lambda row: 0

        bt.run()

        self.assertEqual(bt.trade_log[0]["reason"], "stop")
        self.assertEqual(bt.position, 0)

    def test_short_stop_and_tp_exit_when_short_entry_hour_rejects_entries(self) -> None:
        cases = (
            ({"high": 106.0, "low": 101.0}, "stop"),
            ({"high": 102.0, "low": 89.0}, "tp"),
        )
        for first, reason in cases:
            with self.subTest(reason=reason):
                bt = make_backtester(
                    execution_frame(first),
                    allowed_hours_ny={10},
                    short_hours_ny={10},
                )
                arm_position(bt, side=-1, stop_price=105.0, tp_price=90.0)
                bt.signal = lambda row: 0

                bt.run()

                self.assertEqual(bt.trade_log[0]["reason"], reason)
                self.assertEqual(bt.position, 0)

    def test_open_position_exits_when_atr_bands_are_unavailable_or_out_of_band(self) -> None:
        cases = (
            {"atr_q_low_band": np.nan, "atr_q_high_band": np.nan, "low": 94.0},
            {"atr": 2.0, "atr_q_low_band": 3.0, "atr_q_high_band": 5.0, "low": 94.0},
        )
        for first in cases:
            with self.subTest(first=first):
                bt = make_backtester(execution_frame(first), allowed_hours_ny={10})
                arm_position(bt, side=1, stop_price=95.0, tp_price=110.0)
                bt.signal = lambda row: 0

                bt.run()

                self.assertEqual(bt.trade_log[0]["reason"], "stop")

    def test_open_position_exits_when_next_bar_is_not_entry_eligible(self) -> None:
        bt = make_backtester(
            execution_frame({"low": 94.0}, {"open": np.nan, "volume": 0.0}),
            allowed_hours_ny={10},
        )
        arm_position(bt, side=1, stop_price=95.0, tp_price=110.0)
        bt.signal = lambda row: 0

        bt.run()

        self.assertEqual(bt.trade_log[0]["reason"], "stop")

    def test_flat_position_does_not_enter_when_entry_gates_reject_bar(self) -> None:
        cases = (
            (
                execution_frame({"atr": 2.0, "atr_q_low_band": 3.0, "atr_q_high_band": 5.0}),
                {},
            ),
            (execution_frame({"atr_q_low_band": np.nan, "atr_q_high_band": np.nan}), {}),
            (execution_frame({}, {"volume": 0.0}), {}),
            (execution_frame(), {"allowed_hours_ny": {10}}),
            (
                execution_frame(timestamp="2026-06-01T19:31:00Z"),
                {"allowed_hours_ny": {15}},
            ),
            (execution_frame(), {"long_hours_ny": {10}}),
        )
        for frame, overrides in cases:
            with self.subTest(frame=frame.to_dict("list"), overrides=overrides):
                bt = make_backtester(frame, **overrides)
                bt.signal = lambda row: 1

                bt.run()

                self.assertEqual(bt.position, 0)
                self.assertEqual(bt.trade_log, [])

    def test_mark_to_market_uses_current_close_when_entry_gates_reject_bar(self) -> None:
        bt = make_backtester(
            append_session_final_bar(execution_frame({"close": 107.0})),
            allowed_hours_ny={10},
        )
        arm_position(bt, side=1, stop_price=90.0, tp_price=120.0)
        bt.signal = lambda row: 0

        bt.run()

        self.assertEqual(bt.position, 0)
        self.assertEqual(bt.equity_curve[0], 1_070.0)

    def test_stop_wins_same_bar_stop_tp_tie(self) -> None:
        bt = make_backtester(execution_frame({"high": 111.0, "low": 94.0}))
        arm_position(bt, side=1, stop_price=95.0, tp_price=110.0)
        bt.signal = lambda row: 0

        bt.run()

        self.assertEqual(bt.trade_log[0]["reason"], "stop")
        self.assertEqual(bt.trade_log[0]["pnl"], -50.0)

    def test_same_bar_stop_exit_can_open_next_bar_entry_when_gates_pass(self) -> None:
        bt = make_backtester(
            append_session_final_bar(
                execution_frame({"low": 94.0}, timestamp="2026-06-01T14:00:00Z")
            ),
            allowed_hours_ny={10},
        )
        arm_position(bt, side=1, stop_price=95.0, tp_price=110.0)
        bt.signal = lambda row: 1

        bt.run()

        self.assertEqual(bt.trade_log[0]["reason"], "stop")
        self.assertEqual(bt.trade_log[1]["reason"], "session_close")
        self.assertEqual(bt.trade_log[1]["entry_ts"], bt.df.iloc[1].timestamp)
        self.assertEqual(bt.position, 0)

    def test_future_entry_does_not_change_prior_timestamp_equity(self) -> None:
        bt = make_backtester(causal_entry_frame())
        bt.signal = lambda row: int(row.timestamp == bt.df.iloc[0].timestamp)

        bt.run()

        self.assertEqual(bt.df.iloc[0].close, 80.0)
        self.assertEqual(bt.df.iloc[1].open, 150.0)
        self.assertEqual(bt.trade_log[0]["entry_ts"], bt.df.iloc[1].timestamp)
        self.assertEqual(bt.equity_timestamps[0], bt.df.iloc[0].timestamp)
        self.assertEqual(bt.equity_curve[0], bt.starting_cash)

    def test_future_entry_commission_does_not_reduce_prior_equity(self) -> None:
        bt = make_backtester(causal_entry_frame(), commission_per_share=0.50)
        bt.signal = lambda row: int(row.timestamp == bt.df.iloc[0].timestamp)

        bt.run()

        self.assertGreater(bt.trade_log[0]["entry_ts"], bt.equity_timestamps[0])
        self.assertGreater(bt._commission(bt.trade_log[0]["shares"]), 0.0)
        self.assertEqual(bt.equity_curve[0], bt.starting_cash)

    def test_next_bar_close_reflects_position_entered_at_its_open(self) -> None:
        bt = make_backtester(causal_entry_frame(), commission_per_share=0.50)
        bt.signal = lambda row: int(row.timestamp == bt.df.iloc[0].timestamp)

        bt.run()

        entry = bt.trade_log[0]
        self.assertEqual(entry["entry_ts"], bt.equity_timestamps[1])
        self.assertAlmostEqual(bt.equity_curve[1], 1_027.0)

    def test_exit_then_future_reentry_keeps_exit_sample_causal(self) -> None:
        frame = append_session_final_bar(
            execution_frame(
                {"low": 94.0, "close": 80.0},
                {"open": 120.0, "high": 125.0, "low": 121.0, "close": 125.0},
                timestamp="2026-06-01T14:00:00Z",
            ),
            open=125.0,
            high=126.0,
            low=124.0,
            close=125.0,
        )
        bt = make_backtester(frame, commission_per_share=2.0)
        arm_position(bt, side=1, stop_price=95.0, tp_price=110.0)
        bt.signal = lambda row: int(row.timestamp == bt.df.iloc[0].timestamp)

        bt.run()

        self.assertEqual(bt.trade_log[0]["reason"], "stop")
        self.assertEqual(bt.trade_log[0]["pnl"], -70.0)
        self.assertEqual(bt.trade_log[1]["entry_ts"], bt.df.iloc[1].timestamp)
        self.assertEqual(bt.equity_curve[0], 930.0)

    def test_equity_timestamp_causality_excludes_later_position_value(self) -> None:
        bt = make_backtester(causal_entry_frame())
        bt.signal = lambda row: int(row.timestamp == bt.df.iloc[0].timestamp)

        bt.run()

        entry = bt.trade_log[0]
        future_position_value = (
            (bt.df.iloc[0].close - entry["entry_price"]) * entry["shares"] * entry["side"]
        )
        self.assertGreater(entry["entry_ts"], bt.equity_timestamps[0])
        self.assertEqual(bt.equity_curve[0], bt.starting_cash)
        self.assertNotEqual(bt.equity_curve[0], bt.starting_cash + future_position_value)

    def test_calendar_excludes_preopen_and_includes_regular_open(self) -> None:
        frame = execution_bars(
            [
                "2026-06-01T13:00:00Z",  # 09:00 NY: outside
                "2026-06-01T13:29:00Z",  # 09:29 NY: outside
                "2026-06-01T13:30:00Z",  # 09:30 NY: regular open
                "2026-06-01T13:31:00Z",
                "2026-06-01T20:00:00Z",  # close boundary: outside
            ]
        )
        bt = make_backtester(frame)
        bt.signal = lambda row: 0

        bt.run()

        self.assertEqual(bt.execution_session_contract, "nyse_regular_session_intraday_flat_v1")
        self.assertEqual(
            bt.equity_timestamps,
            [pd.Timestamp("2026-06-01T13:30:00Z"), pd.Timestamp("2026-06-01T13:31:00Z")],
        )
        self.assertFalse(bt._execution_sessions.iloc[1].in_regular_session)
        self.assertTrue(bt._execution_sessions.iloc[2].in_regular_session)

    def test_session_close_flattens_surviving_position_with_actual_risk_fields(self) -> None:
        frame = execution_bars(["2026-06-01T19:58:00Z", "2026-06-01T19:59:00Z"])
        frame.loc[1, "close"] = 104.0
        bt = make_backtester(frame, slippage_bps=100.0)
        arm_position(bt, side=1, stop_price=90.0, tp_price=120.0)
        bt.signal = lambda row: 0

        bt.run()

        trade = bt.trade_log[0]
        self.assertEqual(trade["reason"], "session_close")
        self.assertEqual(trade["exit_ts"], pd.Timestamp("2026-06-01T19:59:00Z"))
        self.assertEqual(trade["risk_budget_dollars"], 100.0)
        self.assertEqual(trade["initial_stop_risk_dollars"], 100.0)
        self.assertAlmostEqual(trade["pnl"], 29.6)
        self.assertAlmostEqual(trade["R"], 0.296)
        self.assertEqual(bt.position, 0)
        self.assertAlmostEqual(bt.equity_curve[-1], 1_029.6)

    def test_final_bar_stop_or_tp_beats_session_close(self) -> None:
        cases = (
            ({"low": 94.0, "high": 101.0}, "stop"),
            ({"low": 99.0, "high": 111.0}, "tp"),
        )
        for final_overrides, reason in cases:
            with self.subTest(reason=reason):
                frame = execution_bars(["2026-06-01T19:58:00Z", "2026-06-01T19:59:00Z"])
                for column, value in final_overrides.items():
                    frame.loc[1, column] = value
                bt = make_backtester(frame)
                arm_position(bt, side=1, stop_price=95.0, tp_price=110.0)
                bt.signal = lambda row: 0

                bt.run()

                self.assertEqual([trade["reason"] for trade in bt.trade_log], [reason])
                self.assertEqual(bt.position, 0)

    def test_session_close_prevents_cross_date_position_carry(self) -> None:
        frame = execution_bars(
            [
                "2026-06-01T19:58:00Z",
                "2026-06-01T19:59:00Z",
                "2026-06-02T13:30:00Z",
                "2026-06-02T13:31:00Z",
            ]
        )
        bt = make_backtester(frame)
        arm_position(bt, side=1, stop_price=90.0, tp_price=120.0)
        bt.signal = lambda row: 0

        bt.run()

        self.assertEqual(bt.trade_log[0]["reason"], "session_close")
        self.assertEqual(bt.trade_log[0]["exit_ts"].date().isoformat(), "2026-06-01")
        self.assertEqual(bt.position, 0)

    def test_calendar_early_close_blocks_post_close_next_bar_entry(self) -> None:
        close = calendar_close(pd.Timestamp("2026-11-27T14:30:00Z"))
        final_minute = close - pd.Timedelta(minutes=1)
        self.assertEqual(close, pd.Timestamp("2026-11-27T18:00:00Z"))
        frame = execution_bars([str(final_minute), str(close)])
        bt = make_backtester(frame, allowed_hours_ny={12})
        bt.signal = lambda row: 1

        bt.run()

        self.assertTrue(bt._execution_sessions.iloc[0].is_final_regular_minute)
        self.assertFalse(bt._execution_sessions.iloc[1].in_regular_session)
        self.assertEqual(bt.position, 0)
        self.assertEqual(bt.trade_log, [])
        self.assertEqual(bt.equity_timestamps, [final_minute])

    def test_entry_requires_current_and_next_bar_in_same_regular_session(self) -> None:
        frame = execution_bars(["2026-06-01T19:59:00Z", "2026-06-02T13:30:00Z"])
        bt = make_backtester(frame, allowed_hours_ny={15})
        bt.signal = lambda row: 1

        bt.run()

        self.assertEqual(bt.position, 0)
        self.assertEqual(bt.trade_log, [])

    def test_equity_clock_contains_only_all_regular_session_minutes(self) -> None:
        close = calendar_close(pd.Timestamp("2026-06-01T13:30:00Z"))
        regular_minutes = pd.date_range(
            pd.Timestamp("2026-06-01T13:30:00Z"),
            close - pd.Timedelta(minutes=1),
            freq="min",
        )
        timestamps = ["2026-06-01T13:00:00Z", *[str(value) for value in regular_minutes], str(close)]
        bt = make_backtester(execution_bars(timestamps))
        bt.signal = lambda row: 0

        bt.run()

        self.assertEqual(len(regular_minutes), 390)
        self.assertEqual(bt.equity_timestamps, list(regular_minutes))
        self.assertEqual(len(bt.equity_curve), 390)
        self.assertEqual(_equity_df(bt)["timestamp"].tolist(), list(regular_minutes))

    def test_missing_final_regular_session_bar_with_open_position_fails_closed(self) -> None:
        bt = make_backtester(execution_bars(["2026-06-01T19:58:00Z", "2026-06-01T20:00:00Z"]))
        arm_position(bt, side=1, stop_price=90.0, tp_price=120.0)
        bt.signal = lambda row: 0

        with self.assertRaisesRegex(BenchmarkSessionDataError, "required final regular-session bar"):
            bt.run()

    def test_dataset_ending_with_open_position_fails_closed(self) -> None:
        bt = make_backtester(execution_bars(["2026-06-01T19:58:00Z"]))
        arm_position(bt, side=1, stop_price=90.0, tp_price=120.0)
        bt.signal = lambda row: 0

        with self.assertRaisesRegex(BenchmarkSessionDataError, "required final regular-session bar"):
            bt.run()

    def test_uncapped_entry_initial_stop_risk_uses_actual_entry_stop_and_shares(self) -> None:
        bt = make_backtester(
            append_session_final_bar(execution_frame(timestamp="2026-06-01T14:00:00Z")),
            allowed_hours_ny={10},
            max_position_notional_pct=1.0,
        )
        bt.signal = lambda row: 1

        bt.run()

        trade = bt.trade_log[0]
        self.assertEqual(trade["risk_budget_dollars"], 100.0)
        self.assertEqual(
            trade["initial_stop_risk_dollars"],
            abs(trade["entry_price"] - trade["stop_price"]) * trade["shares"],
        )
        self.assertEqual(trade["initial_stop_risk_dollars"], 100.0)

    def test_notional_cap_makes_initial_stop_risk_less_than_budget(self) -> None:
        bt = make_backtester(
            append_session_final_bar(execution_frame(timestamp="2026-06-01T14:00:00Z")),
            allowed_hours_ny={10},
            max_position_notional_pct=0.20,
        )
        bt.signal = lambda row: 1

        bt.run()

        trade = bt.trade_log[0]
        self.assertLess(trade["initial_stop_risk_dollars"], trade["risk_budget_dollars"])
        self.assertEqual(trade["initial_stop_risk_dollars"], 20.0)

    def test_completed_trade_r_uses_initial_stop_risk_without_epsilon(self) -> None:
        bt = make_backtester(execution_frame())
        arm_position(
            bt,
            side=1,
            stop_price=95.0,
            tp_price=110.0,
            initial_stop_risk_dollars=50.0,
        )

        bt._record_trade(25.0, "tp", bt.df.iloc[0].timestamp)

        trade = bt.trade_log[0]
        self.assertEqual(trade["risk_budget_dollars"], 100.0)
        self.assertEqual(trade["initial_stop_risk_dollars"], 50.0)
        self.assertEqual(trade["R"], 0.5)

    def test_exit_slippage_changes_pnl_and_r_but_not_initial_stop_risk(self) -> None:
        results = []
        for slippage_bps in (0.0, 100.0):
            bt = make_backtester(execution_frame({"low": 89.0}), slippage_bps=slippage_bps)
            arm_position(bt, side=1, stop_price=90.0, tp_price=120.0)
            bt.signal = lambda row: 0
            bt.run()
            results.append(bt.trade_log[0])

        no_slip, slipped = results
        self.assertEqual(no_slip["initial_stop_risk_dollars"], 100.0)
        self.assertEqual(slipped["initial_stop_risk_dollars"], 100.0)
        self.assertAlmostEqual(no_slip["pnl"], -100.0)
        self.assertAlmostEqual(slipped["pnl"], -109.0)
        self.assertAlmostEqual(no_slip["R"], -1.0)
        self.assertAlmostEqual(slipped["R"], -1.09)

    def test_audit_metrics_preserve_mean_median_and_sum_r(self) -> None:
        bt = make_backtester(execution_frame())
        arm_position(bt, side=1, stop_price=90.0, tp_price=120.0)
        bt._record_trade(100.0, "tp", bt.df.iloc[0].timestamp)
        bt._record_trade(-50.0, "stop", bt.df.iloc[1].timestamp)
        bt.equity_curve = [1_000.0, 1_050.0]
        bt.equity_timestamps = [bt.df.iloc[0].timestamp, bt.df.iloc[1].timestamp]

        trades = _trade_df(bt)
        equity = _equity_df(bt)
        metrics = _metrics(bt, trades, 1.0)

        self.assertIn("risk_budget_dollars", trades.columns)
        self.assertIn("initial_stop_risk_dollars", trades.columns)
        self.assertEqual(equity.columns.tolist(), ["bar_index", "timestamp", "equity"])
        self.assertEqual(equity["timestamp"].tolist(), bt.equity_timestamps)
        self.assertEqual(metrics["mean_R"], 0.25)
        self.assertEqual(metrics["median_R"], 0.25)
        self.assertEqual(metrics["sum_R"], 0.5)
        self.assertEqual(
            BENCHMARK_EXECUTION_METHOD_METADATA["execution_session_contract"],
            "nyse_regular_session_intraday_flat_v1",
        )
        self.assertEqual(
            BENCHMARK_EXECUTION_METHOD_METADATA["equity_sampling"],
            "nyse_regular_session_minutes_only",
        )
        self.assertEqual(
            BENCHMARK_EXECUTION_METHOD_METADATA["equity_entry_timing"],
            "current_bar_close_before_next_bar_open_entry",
        )


if __name__ == "__main__":
    unittest.main()
