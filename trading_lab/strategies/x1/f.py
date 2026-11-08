
"""
X1-F backtester for the frozen gap-up failed-hold short candidate.

This module exists as a preserved runnable X1 reference inside the quarantined lane and should not be treated as MAIN logic or silently retuned in place.
"""


from __future__ import annotations

from pathlib import Path
from typing import Any

import math
import pandas as pd
from zoneinfo import ZoneInfo


class BacktesterX1GapUpFailedHoldContinuation:
    """
    X1-only backtester for a gap-up failed-hold continuation short.

    Core idea:
    - require an overnight gap up from the prior regular-session close
    - require that the opening hold has already failed by 10:30 NY
    - enter short on the next bar open
    - stop above the cumulative session high / signal VWAP
    - exit at a fixed midday time if the stop never triggers
    """

    def __init__(
        self,
        file: str,
        starting_cash: float = 100_000,
        risk_frac: float = 0.0033,
        slippage_bps: float = 1.0,
        commission_per_share: float = 0.0,
        max_position_notional_pct: float = 0.10,
        gap_up_threshold_pct: float = 0.0010,
        min_open_failure_pct: float = 0.0005,
        signal_hour_ny: int = 10,
        signal_minute_ny: int = 30,
        time_exit_hour_ny: int = 12,
        time_exit_minute_ny: int = 0,
        verbose: bool = True,
    ) -> None:
        self.file = str(file)
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.risk_frac = float(risk_frac)
        self.slippage_bps = float(slippage_bps)
        self.commission_per_share = float(commission_per_share)
        self.max_position_notional_pct = float(max_position_notional_pct)
        self.gap_up_threshold_pct = float(gap_up_threshold_pct)
        self.min_open_failure_pct = float(min_open_failure_pct)
        self.signal_hour_ny = int(signal_hour_ny)
        self.signal_minute_ny = int(signal_minute_ny)
        self.time_exit_hour_ny = int(time_exit_hour_ny)
        self.time_exit_minute_ny = int(time_exit_minute_ny)
        self.verbose = bool(verbose)
        self.ny_tz = ZoneInfo("America/New_York")
        self.df = self._load_and_prepare_data()

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = self.slippage_bps / 10_000.0
        return price * (1 + slip) if side == "buy" else price * (1 - slip)

    def _load_and_prepare_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {self.file}: {missing}")

        df = df[df["volume"] > 0].copy().reset_index(drop=True)
        ts_ny = df["timestamp"].dt.tz_convert(self.ny_tz)
        df["ts_ny"] = ts_ny
        df["date_ny"] = ts_ny.dt.date.astype(str)
        df["hour_ny"] = ts_ny.dt.hour
        df["minute_ny"] = ts_ny.dt.minute

        session_mask = (
            ((df["hour_ny"] > 9) | ((df["hour_ny"] == 9) & (df["minute_ny"] >= 30)))
            & (df["hour_ny"] < 16)
        )
        df = df[session_mask].copy().reset_index(drop=True)

        pxv = df["close"] * df["volume"]
        df["session_vwap"] = pxv.groupby(df["date_ny"]).cumsum() / df["volume"].groupby(df["date_ny"]).cumsum()
        df["session_open"] = df.groupby("date_ny")["open"].transform("first")
        df["cum_high"] = df.groupby("date_ny")["high"].cummax()

        prev_close = df.groupby("date_ny")["close"].last().shift(1)
        df["prev_day_close"] = df["date_ny"].map(prev_close.to_dict())
        df["gap_from_prev_close"] = (df["session_open"] / df["prev_day_close"]) - 1.0
        df["open_failure_pct"] = 1.0 - (df["close"] / df["session_open"])
        return df.reset_index(drop=True)

    def _size_position(self, entry_price: float, stop_price: float) -> int:
        stop_dist = stop_price - entry_price
        if stop_dist <= 0:
            return 0

        risk_dollars = self.cash * self.risk_frac
        shares_by_risk = math.floor(risk_dollars / stop_dist)
        shares_by_cash = math.floor(self.cash / entry_price)
        shares_by_notional = math.floor((self.cash * self.max_position_notional_pct) / entry_price)
        return int(max(0, min(shares_by_risk, shares_by_cash, shares_by_notional)))

    def run(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        trades: list[dict[str, Any]] = []

        for date_ny, day_df in self.df.groupby("date_ny", sort=True):
            day_df = day_df.reset_index(drop=True)
            signal_rows = day_df[
                (day_df["hour_ny"] == self.signal_hour_ny)
                & (day_df["minute_ny"] == self.signal_minute_ny)
            ]
            if signal_rows.empty:
                continue

            signal_idx = int(signal_rows.index[0])
            signal = signal_rows.iloc[0]
            if pd.isna(signal["prev_day_close"]) or float(signal["prev_day_close"]) <= 0:
                continue

            if float(signal["gap_from_prev_close"]) < self.gap_up_threshold_pct:
                continue
            if float(signal["open_failure_pct"]) < self.min_open_failure_pct:
                continue
            if signal_idx + 1 >= len(day_df):
                continue

            entry_bar = day_df.iloc[signal_idx + 1]
            entry_raw = float(entry_bar["open"])
            entry_fill = self._apply_slippage(entry_raw, side="sell")
            stop_price = max(float(signal["cum_high"]), float(signal["session_vwap"]))
            shares = self._size_position(entry_fill, stop_price)
            if shares <= 0:
                continue

            exit_fill = None
            exit_reason = None
            exit_ts = None

            for j in range(signal_idx + 1, len(day_df)):
                bar = day_df.iloc[j]
                if float(bar["high"]) >= stop_price:
                    exit_fill = self._apply_slippage(stop_price, side="buy")
                    exit_reason = "stop"
                    exit_ts = bar["ts_ny"]
                    break

                if (
                    int(bar["hour_ny"]) == self.time_exit_hour_ny
                    and int(bar["minute_ny"]) == self.time_exit_minute_ny
                ):
                    exit_fill = self._apply_slippage(float(bar["open"]), side="buy")
                    exit_reason = "time"
                    exit_ts = bar["ts_ny"]
                    break

            if exit_fill is None:
                continue

            gross_pnl = (entry_fill - exit_fill) * shares
            commission = shares * self.commission_per_share * 2.0
            net_pnl = gross_pnl - commission
            self.cash += net_pnl

            trades.append(
                {
                    "date_ny": date_ny,
                    "entry_ts_ny": entry_bar["ts_ny"],
                    "exit_ts_ny": exit_ts,
                    "side": "short",
                    "gap_from_prev_close": float(signal["gap_from_prev_close"]),
                    "open_failure_pct": float(signal["open_failure_pct"]),
                    "entry_price": entry_fill,
                    "stop_price": stop_price,
                    "exit_price": exit_fill,
                    "shares": shares,
                    "exit_reason": exit_reason,
                    "gross_pnl": gross_pnl,
                    "commission": commission,
                    "net_pnl": net_pnl,
                    "cash_after_trade": self.cash,
                }
            )

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            stats = {
                "trades": 0,
                "net_pnl": 0.0,
                "return_pct": 0.0,
                "winrate": 0.0,
                "median_trade_pnl": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
            }
            return trades_df, stats

        stats = {
            "trades": int(len(trades_df)),
            "net_pnl": float(trades_df["net_pnl"].sum()),
            "return_pct": float((trades_df["cash_after_trade"].iloc[-1] / self.starting_cash - 1.0) * 100.0),
            "winrate": float((trades_df["net_pnl"] > 0).mean()),
            "median_trade_pnl": float(trades_df["net_pnl"].median()),
            "best_trade": float(trades_df["net_pnl"].max()),
            "worst_trade": float(trades_df["net_pnl"].min()),
        }
        return trades_df, stats
