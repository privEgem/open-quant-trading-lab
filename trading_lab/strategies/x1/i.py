
"""
X1-I backtester for the frozen slight-gap-down early-recovery long candidate.

This is preserved as a side-lane X1 reference and should be treated as read-only unless a future explicit trigger reopens it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo


class BacktesterX1SlightGapDownEarlyRecoveryDrift:
    """
    X1-only backtester for a slight-gap-down early recovery drift long.

    Core idea:
    - overnight gap is flat-to-slightly-down versus the prior regular-session close
    - by 10:15 NY, price has recovered modestly above the session open
    - price is also above session VWAP, so the weak open is being repaired rather than extended
    - the bounce off the morning low is real, but price is still not too far from the morning high
    - enter on the 10:16 NY next bar open and hold for a fixed midday drift window
    """

    def __init__(
        self,
        file: str,
        starting_cash: float = 100_000,
        risk_frac: float = 0.0033,
        slippage_bps: float = 1.0,
        commission_per_share: float = 0.0,
        max_position_notional_pct: float = 0.10,
        gap_low_pct: float = -0.0030,
        gap_high_pct: float = 0.0015,
        min_open_recovery_pct: float = 0.0002,
        max_open_recovery_pct: float = 0.0030,
        min_above_vwap_pct: float = 0.0,
        max_above_vwap_pct: float = 0.0025,
        min_bounce_from_low_pct: float = 0.0008,
        max_pullback_from_high_pct: float = 0.0080,
        signal_hour_ny: int = 10,
        signal_minute_ny: int = 15,
        time_exit_hour_ny: int = 12,
        time_exit_minute_ny: int = 30,
        verbose: bool = True,
    ) -> None:
        self.file = str(file)
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.risk_frac = float(risk_frac)
        self.slippage_bps = float(slippage_bps)
        self.commission_per_share = float(commission_per_share)
        self.max_position_notional_pct = float(max_position_notional_pct)
        self.gap_low_pct = float(gap_low_pct)
        self.gap_high_pct = float(gap_high_pct)
        self.min_open_recovery_pct = float(min_open_recovery_pct)
        self.max_open_recovery_pct = float(max_open_recovery_pct)
        self.min_above_vwap_pct = float(min_above_vwap_pct)
        self.max_above_vwap_pct = float(max_above_vwap_pct)
        self.min_bounce_from_low_pct = float(min_bounce_from_low_pct)
        self.max_pullback_from_high_pct = float(max_pullback_from_high_pct)
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

        prev_close = df.groupby("date_ny")["close"].last().shift(1)
        prev_close_map = prev_close.to_dict()
        df["prev_day_close"] = df["date_ny"].map(prev_close_map)
        df["gap_from_prev_close"] = (df["session_open"] / df["prev_day_close"]) - 1.0

        df["is_signal_bar"] = (df["hour_ny"] == self.signal_hour_ny) & (df["minute_ny"] == self.signal_minute_ny)
        return df

    def run(self) -> dict[str, Any]:
        df = self.df.reset_index(drop=True)
        idx_by_date = {d: grp.index.tolist() for d, grp in df.groupby("date_ny")}

        trades: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []
        day_features: list[dict[str, Any]] = []

        for date_key, idxs in idx_by_date.items():
            day = df.loc[idxs].copy()
            equity_curve.append({"timestamp": day.iloc[0]["timestamp"], "equity": self.cash})

            signal_mask = day["is_signal_bar"]
            if not signal_mask.any():
                continue

            signal_row = day.loc[signal_mask].iloc[0]
            signal_idx = int(signal_row.name)
            day_pos = idxs.index(signal_idx)
            pre_signal = day.loc[day.index <= signal_idx].copy()

            gap = float(day.iloc[0]["gap_from_prev_close"]) if pd.notna(day.iloc[0]["gap_from_prev_close"]) else None
            session_open = float(day.iloc[0]["session_open"])
            signal_close = float(signal_row["close"])
            signal_vwap = float(signal_row["session_vwap"])
            open_recovery = (signal_close / session_open) - 1.0
            above_vwap = (signal_close / signal_vwap) - 1.0
            bounce_from_low = (signal_close / float(pre_signal["low"].min())) - 1.0
            pullback_from_high = (float(pre_signal["high"].max()) / signal_close) - 1.0

            feature_row = {
                "date": date_key,
                "gap_from_prev_close": gap,
                "open_recovery_pct": open_recovery,
                "above_vwap_pct": above_vwap,
                "bounce_from_low_pct": bounce_from_low,
                "pullback_from_high_pct": pullback_from_high,
            }
            day_features.append(feature_row)

            if gap is None:
                continue
            if not (self.gap_low_pct <= gap <= self.gap_high_pct):
                continue
            if not (self.min_open_recovery_pct <= open_recovery <= self.max_open_recovery_pct):
                continue
            if not (self.min_above_vwap_pct <= above_vwap <= self.max_above_vwap_pct):
                continue
            if bounce_from_low < self.min_bounce_from_low_pct:
                continue
            if pullback_from_high > self.max_pullback_from_high_pct:
                continue

            if day_pos + 1 >= len(day):
                continue

            entry_row = day.iloc[day_pos + 1]
            entry_price = self._apply_slippage(float(entry_row["open"]), "buy")
            stop_price = float(pre_signal["low"].min())

            if stop_price >= entry_price:
                continue

            exit_mask = (
                (day["hour_ny"] == self.time_exit_hour_ny)
                & (day["minute_ny"] == self.time_exit_minute_ny)
            )
            if not exit_mask.any():
                continue
            exit_idx = int(day.loc[exit_mask].index[0])

            shares_by_risk = int((self.cash * self.risk_frac) // max(entry_price - stop_price, 1e-9))
            shares_by_cash = int(self.cash // entry_price)
            shares_by_notional = int((self.cash * self.max_position_notional_pct) // entry_price)
            shares = min(shares_by_risk, shares_by_cash, shares_by_notional)

            if shares <= 0:
                continue

            in_trade = day.loc[(day.index > entry_row.name) & (day.index <= exit_idx)].copy()
            stop_hit = in_trade["low"] <= stop_price

            if stop_hit.any():
                exit_bar = in_trade.loc[stop_hit].iloc[0]
                exit_price = self._apply_slippage(stop_price, "sell")
                exit_reason = "stop"
            else:
                exit_bar = day.loc[exit_idx]
                exit_price = self._apply_slippage(float(exit_bar["close"]), "sell")
                exit_reason = "time"

            gross_pnl = (exit_price - entry_price) * shares
            commission = self.commission_per_share * shares * 2.0
            pnl = gross_pnl - commission
            self.cash += pnl

            trades.append(
                {
                    "date": date_key,
                    "entry_ts": entry_row["timestamp"],
                    "exit_ts": exit_bar["timestamp"],
                    "side": "long",
                    "shares": shares,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "gap_from_prev_close": gap,
                    "open_recovery_pct": open_recovery,
                    "above_vwap_pct": above_vwap,
                    "bounce_from_low_pct": bounce_from_low,
                    "pullback_from_high_pct": pullback_from_high,
                    "pnl": pnl,
                    "cash_after": self.cash,
                }
            )
            equity_curve.append({"timestamp": exit_bar["timestamp"], "equity": self.cash})

        trade_df = pd.DataFrame(trades)
        eq_df = pd.DataFrame(equity_curve).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        day_features_df = pd.DataFrame(day_features)

        summary = {
            "candidate_name": "X1-I slight-gap-down early recovery drift long",
            "starting_cash": self.starting_cash,
            "ending_cash": self.cash,
            "net_pnl": float(self.cash - self.starting_cash),
            "return_pct": float((self.cash / self.starting_cash - 1.0) * 100.0),
            "trades": int(len(trade_df)),
            "win_rate": float((trade_df["pnl"] > 0).mean() * 100.0) if len(trade_df) else 0.0,
            "median_trade_pnl": float(trade_df["pnl"].median()) if len(trade_df) else 0.0,
            "best_trade": float(trade_df["pnl"].max()) if len(trade_df) else 0.0,
            "worst_trade": float(trade_df["pnl"].min()) if len(trade_df) else 0.0,
            "stop_exits": int((trade_df["exit_reason"] == "stop").sum()) if len(trade_df) else 0,
            "time_exits": int((trade_df["exit_reason"] == "time").sum()) if len(trade_df) else 0,
            "trade_log": trade_df,
            "equity_curve": eq_df,
            "day_features": day_features_df,
        }
        return summary
