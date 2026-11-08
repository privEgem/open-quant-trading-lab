
"""
X1-L backtester for the frozen near-flat shakeout-reclaim long candidate.

This module is part of the preserved X1 frozen stack and should not be read as active MAIN-side development.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo


class BacktesterX1NearFlatMorningShakeoutReclaimLong:
    """
    X1-only backtester for a near-flat morning shakeout reclaim long idea.

    Core idea:
    - overnight gap is near flat, slightly down, or only slightly up
    - the session suffers an early shakeout below the open
    - by 11:00 NY, price has reclaimed both the open and session VWAP
    - the reclaim is real but not overextended
    - enter long on the next bar open and hold for a late-morning drift attempt
    """

    def __init__(
        self,
        file: str,
        starting_cash: float = 100_000,
        slippage_bps: float = 1.0,
        commission_per_share: float = 0.0,
        gap_low_pct: float = -0.0010,
        gap_high_pct: float = 0.0015,
        signal_hour_ny: int = 11,
        signal_minute_ny: int = 0,
        min_shakeout_below_open_pct: float = 0.0010,
        min_reclaim_above_open_pct: float = 0.0005,
        max_extension_above_open_pct: float = 0.0040,
        time_exit_hour_ny: int = 12,
        time_exit_minute_ny: int = 30,
        verbose: bool = True,
    ) -> None:
        self.file = str(file)
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.slippage_bps = float(slippage_bps)
        self.commission_per_share = float(commission_per_share)
        self.gap_low_pct = float(gap_low_pct)
        self.gap_high_pct = float(gap_high_pct)
        self.signal_hour_ny = int(signal_hour_ny)
        self.signal_minute_ny = int(signal_minute_ny)
        self.min_shakeout_below_open_pct = float(min_shakeout_below_open_pct)
        self.min_reclaim_above_open_pct = float(min_reclaim_above_open_pct)
        self.max_extension_above_open_pct = float(max_extension_above_open_pct)
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
        df = df[df["volume"] > 0].copy()
        df["timestamp_ny"] = df["timestamp"].dt.tz_convert(self.ny_tz)
        df["date_ny"] = df["timestamp_ny"].dt.date
        df["hm_ny"] = df["timestamp_ny"].dt.hour * 100 + df["timestamp_ny"].dt.minute
        df = df[(df["hm_ny"] >= 930) & (df["hm_ny"] <= 1600)].reset_index(drop=True)
        return df

    def _build_trade(self, row: dict[str, Any], trades: list[dict[str, Any]]) -> None:
        trades.append(row)

    def run_backtest(self) -> dict[str, Any]:
        signal_hm = self.signal_hour_ny * 100 + self.signal_minute_ny
        exit_hm = self.time_exit_hour_ny * 100 + self.time_exit_minute_ny

        grouped = list(self.df.groupby("date_ny"))
        trades: list[dict[str, Any]] = []

        for idx in range(1, len(grouped)):
            date_ny, day_df = grouped[idx]
            prev_date, prev_df = grouped[idx - 1]

            prev_close = float(prev_df.iloc[-1]["close"])
            day_open = float(day_df.iloc[0]["open"])
            gap_pct = day_open / prev_close - 1.0

            if gap_pct < self.gap_low_pct or gap_pct > self.gap_high_pct:
                continue

            sig_rows = day_df[day_df["hm_ny"] == signal_hm]
            exit_rows = day_df[day_df["hm_ny"] == exit_hm]
            if sig_rows.empty or exit_rows.empty:
                continue

            sig_idx = sig_rows.index[0]
            local_idx = day_df.index.get_loc(sig_idx)
            if local_idx + 1 >= len(day_df):
                continue

            exit_idx = exit_rows.index[0]
            exit_local_idx = day_df.index.get_loc(exit_idx)
            if exit_local_idx <= local_idx:
                continue

            hist = day_df.iloc[: local_idx + 1]
            sig = sig_rows.iloc[0]

            shakeout_low = float(hist["low"].min())
            if shakeout_low >= day_open * (1.0 - self.min_shakeout_below_open_pct):
                continue

            sig_close = float(sig["close"])
            sig_vwap = float(sig["vwap"])
            if sig_close <= day_open * (1.0 + self.min_reclaim_above_open_pct):
                continue
            if sig_close >= day_open * (1.0 + self.max_extension_above_open_pct):
                continue
            if sig_close <= sig_vwap:
                continue

            entry_bar = day_df.iloc[local_idx + 1]
            entry_ts = entry_bar["timestamp_ny"]
            entry_price = self._apply_slippage(float(entry_bar["open"]), "buy")
            stop_price = float(hist["low"].min())

            exit_price = self._apply_slippage(float(exit_rows.iloc[0]["open"]), "sell")
            exit_ts = exit_rows.iloc[0]["timestamp_ny"]
            exit_reason = "TIME_EXIT"

            for j in range(local_idx + 1, exit_local_idx + 1):
                row = day_df.iloc[j]
                if float(row["low"]) <= stop_price:
                    exit_price = self._apply_slippage(stop_price, "sell")
                    exit_ts = row["timestamp_ny"]
                    exit_reason = "STOP_EXIT"
                    break

            shares = 10_000.0 / entry_price
            gross_pnl = (exit_price - entry_price) * shares
            commission = 2.0 * self.commission_per_share * shares
            net_pnl = gross_pnl - commission
            ret_pct = net_pnl / self.starting_cash * 100.0

            self._build_trade(
                {
                    "date_ny": str(date_ny),
                    "prev_date_ny": str(prev_date),
                    "gap_pct": gap_pct,
                    "entry_time_ny": str(entry_ts),
                    "entry_price": entry_price,
                    "exit_time_ny": str(exit_ts),
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "stop_price": stop_price,
                    "shares": shares,
                    "gross_pnl": gross_pnl,
                    "commission": commission,
                    "net_pnl": net_pnl,
                    "ret_pct": ret_pct,
                    "signal_close": sig_close,
                    "signal_vwap": sig_vwap,
                    "signal_above_open_pct": sig_close / day_open - 1.0,
                    "shakeout_below_open_pct": 1.0 - shakeout_low / day_open,
                },
                trades,
            )

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            return {
                "summary": {
                    "trades": 0,
                    "net_pnl": 0.0,
                    "return_pct": 0.0,
                    "median_trade_pnl": 0.0,
                    "win_rate": 0.0,
                    "best_trade": 0.0,
                    "worst_trade": 0.0,
                },
                "trade_log": trades_df,
            }

        summary = {
            "trades": int(len(trades_df)),
            "net_pnl": float(trades_df["net_pnl"].sum()),
            "return_pct": float(trades_df["ret_pct"].sum()),
            "median_trade_pnl": float(trades_df["net_pnl"].median()),
            "win_rate": float((trades_df["net_pnl"] > 0).mean()),
            "best_trade": float(trades_df["net_pnl"].max()),
            "worst_trade": float(trades_df["net_pnl"].min()),
        }
        return {"summary": summary, "trade_log": trades_df}
