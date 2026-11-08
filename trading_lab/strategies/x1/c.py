
"""
Comparator backtester for X1-C, the weak preserved opening-drive pullback branch.

This module remains in RESEARCH-X1 only as comparator memory and is not part of the active frozen stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo


class BacktesterX1OpeningDrivePullbackContinuation:
    """
    X1-only backtester for an opening-drive pullback continuation idea.

    Core idea:
    - define an opening range through 09:44 NY
    - require an upside breakout above that range by 10:05 NY
    - require a shallow pullback that retests the breakout area without full failure
    - require re-acceptance above the opening-range high by 10:30 NY
    - go long on the next bar open
    """

    def __init__(
        self,
        file: str,
        starting_cash: float = 100_000,
        risk_frac: float = 0.0033,
        rr: float = 1.8,
        slippage_bps: float = 1.0,
        commission_per_share: float = 0.0,
        max_position_notional_pct: float = 0.10,
        opening_range_end_hour_ny: int = 9,
        opening_range_end_minute_ny: int = 44,
        breakout_latest_hour_ny: int = 10,
        breakout_latest_minute_ny: int = 5,
        signal_hour_ny: int = 10,
        signal_minute_ny: int = 30,
        min_breakout_pct: float = 0.0010,
        pullback_min_pct: float = 0.0008,
        pullback_max_pct: float = 0.0035,
        max_or_failure_pct: float = 0.0008,
        min_session_return_signal: float = 0.0025,
        signal_recovery_fraction: float = 0.65,
        time_exit_hour_ny: int = 11,
        time_exit_minute_ny: int = 30,
        verbose: bool = True,
    ) -> None:
        self.file = str(file)
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.risk_frac = float(risk_frac)
        self.rr = float(rr)
        self.slippage_bps = float(slippage_bps)
        self.commission_per_share = float(commission_per_share)
        self.max_position_notional_pct = float(max_position_notional_pct)
        self.opening_range_end_hour_ny = int(opening_range_end_hour_ny)
        self.opening_range_end_minute_ny = int(opening_range_end_minute_ny)
        self.breakout_latest_hour_ny = int(breakout_latest_hour_ny)
        self.breakout_latest_minute_ny = int(breakout_latest_minute_ny)
        self.signal_hour_ny = int(signal_hour_ny)
        self.signal_minute_ny = int(signal_minute_ny)
        self.min_breakout_pct = float(min_breakout_pct)
        self.pullback_min_pct = float(pullback_min_pct)
        self.pullback_max_pct = float(pullback_max_pct)
        self.max_or_failure_pct = float(max_or_failure_pct)
        self.min_session_return_signal = float(min_session_return_signal)
        self.signal_recovery_fraction = float(signal_recovery_fraction)
        self.time_exit_hour_ny = int(time_exit_hour_ny)
        self.time_exit_minute_ny = int(time_exit_minute_ny)
        self.verbose = bool(verbose)
        self.ny_tz = ZoneInfo("America/New_York")
        self.df = self._load_and_prepare_data()

    def _apply_slippage(self, price: float, side: str) -> float:
        slip = self.slippage_bps / 10_000.0
        if side == "buy":
            return price * (1 + slip)
        return price * (1 - slip)

    def _commission(self, shares: int) -> float:
        return float(self.commission_per_share) * int(shares)

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
        df["session_return"] = (df["close"] / df["session_open"]) - 1.0
        df["is_signal_bar"] = (df["hour_ny"] == self.signal_hour_ny) & (df["minute_ny"] == self.signal_minute_ny)
        return df

    def _time_leq(self, df: pd.DataFrame, hour: int, minute: int) -> pd.Series:
        return (df["hour_ny"] < hour) | ((df["hour_ny"] == hour) & (df["minute_ny"] <= minute))

    def run(self) -> dict[str, Any]:
        df = self.df.reset_index(drop=True)
        idx_by_date = {d: grp.index.tolist() for d, grp in df.groupby("date_ny")}
        day_features: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        debug_events: list[dict[str, Any]] = []
        equity_curve: list[dict[str, Any]] = []

        for date_key, idxs in idx_by_date.items():
            day = df.loc[idxs].copy()
            equity_curve.append({"timestamp": day.iloc[0]["timestamp"], "equity": self.cash})

            opening_mask = self._time_leq(day, self.opening_range_end_hour_ny, self.opening_range_end_minute_ny)
            latest_breakout_mask = self._time_leq(day, self.breakout_latest_hour_ny, self.breakout_latest_minute_ny)
            signal_mask = day["is_signal_bar"]

            if not signal_mask.any():
                continue

            opening_df = day.loc[opening_mask]
            if opening_df.empty:
                continue

            signal_row = day.loc[signal_mask].iloc[0]
            signal_idx = signal_row.name

            or_high = float(opening_df["high"].max())
            or_low = float(opening_df["low"].min())

            early_df = day.loc[latest_breakout_mask]
            breakout_df = early_df.loc[early_df["high"] >= or_high * (1.0 + self.min_breakout_pct)]
            breakout_found = not breakout_df.empty

            feature = {
                "date_ny": date_key,
                "or_high": or_high,
                "or_low": or_low,
                "signal_close": float(signal_row["close"]),
                "signal_session_return": float(signal_row["session_return"]),
                "breakout_found": breakout_found,
                "is_candidate_day": False,
            }

            if not breakout_found:
                day_features.append(feature)
                continue

            breakout_idx = int(breakout_df.index[0])
            if signal_idx <= breakout_idx:
                debug_events.append({"date_ny": date_key, "reason": "signal_before_breakout"})
                day_features.append(feature)
                continue

            mid_df = day.loc[(day.index > breakout_idx) & (day.index <= signal_idx)].copy()
            if mid_df.empty:
                debug_events.append({"date_ny": date_key, "reason": "no_pullback_window"})
                day_features.append(feature)
                continue

            breakout_high = float(day.loc[breakout_idx, "high"])
            pullback_low = float(mid_df["low"].min())
            pullback_pct = (breakout_high - pullback_low) / breakout_high
            required_recovery_close = breakout_high - ((1.0 - self.signal_recovery_fraction) * (breakout_high - pullback_low))

            feature.update(
                {
                    "breakout_idx": breakout_idx,
                    "breakout_high": breakout_high,
                    "pullback_low": pullback_low,
                    "pullback_pct": pullback_pct,
                    "required_recovery_close": required_recovery_close,
                }
            )

            is_candidate_day = (
                (self.pullback_min_pct <= pullback_pct <= self.pullback_max_pct)
                and (pullback_low >= or_high * (1.0 - self.max_or_failure_pct))
                and (float(signal_row["close"]) > or_high)
                and (float(signal_row["close"]) > float(signal_row["session_vwap"]))
                and (float(signal_row["session_return"]) >= self.min_session_return_signal)
                and (float(signal_row["close"]) >= required_recovery_close)
            )
            feature["is_candidate_day"] = bool(is_candidate_day)
            day_features.append(feature)

            if not is_candidate_day:
                continue

            # Entry on next bar open
            day_pos = idxs.index(signal_idx)
            if day_pos + 1 >= len(idxs):
                debug_events.append({"date_ny": date_key, "reason": "no_next_bar"})
                continue
            next_idx = idxs[day_pos + 1]
            nxt = df.loc[next_idx]

            entry_raw = float(nxt["open"])
            stop_raw = pullback_low
            if stop_raw >= entry_raw:
                debug_events.append({"date_ny": date_key, "reason": "invalid_stop_distance"})
                continue

            stop_distance = entry_raw - stop_raw
            risk_dollars = self.cash * self.risk_frac
            shares_by_risk = risk_dollars / stop_distance
            shares_by_cash = self.cash // entry_raw
            shares_by_cap = (self.cash * self.max_position_notional_pct) // entry_raw
            shares = int(max(min(shares_by_risk, shares_by_cash, shares_by_cap), 0))
            if shares <= 0:
                debug_events.append({"date_ny": date_key, "reason": "zero_shares"})
                continue

            entry_price = self._apply_slippage(entry_raw, "buy")
            tp_price = entry_price + (self.rr * (entry_raw - stop_raw))
            exit_price = None
            exit_reason = None
            exit_ts = None

            for j in idxs:
                if j < next_idx:
                    continue
                bar = df.loc[j]
                low = float(bar["low"])
                high = float(bar["high"])
                if low <= stop_raw:
                    exit_price = self._apply_slippage(stop_raw, "sell")
                    exit_reason = "stop"
                    exit_ts = bar["timestamp"]
                    break
                if high >= tp_price:
                    exit_price = self._apply_slippage(tp_price, "sell")
                    exit_reason = "tp"
                    exit_ts = bar["timestamp"]
                    break
                if (int(bar["hour_ny"]) > self.time_exit_hour_ny) or (
                    (int(bar["hour_ny"]) == self.time_exit_hour_ny) and (int(bar["minute_ny"]) >= self.time_exit_minute_ny)
                ):
                    exit_price = self._apply_slippage(float(bar["close"]), "sell")
                    exit_reason = "time_exit"
                    exit_ts = bar["timestamp"]
                    break

            if exit_price is None:
                last_bar = df.loc[idxs[-1]]
                exit_price = self._apply_slippage(float(last_bar["close"]), "sell")
                exit_reason = "end_of_day"
                exit_ts = last_bar["timestamp"]

            pnl = (exit_price - entry_price) * shares - self._commission(shares)
            self.cash += pnl
            risk_used = max((entry_raw - stop_raw) * shares, 1e-9)
            trades.append(
                {
                    "date_ny": date_key,
                    "entry_ts": nxt["timestamp"],
                    "exit_ts": exit_ts,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "stop_price": stop_raw,
                    "tp_price": tp_price,
                    "shares": shares,
                    "pnl": pnl,
                    "R": pnl / risk_used,
                    "reason": exit_reason,
                    "pullback_pct": pullback_pct,
                    "signal_session_return": float(signal_row["session_return"]),
                    "or_high": or_high,
                }
            )

        trades_df = pd.DataFrame(trades)
        day_features_df = pd.DataFrame(day_features)
        debug_df = pd.DataFrame(debug_events)
        eq_df = pd.DataFrame(equity_curve)

        if trades_df.empty:
            summary = {
                "trades": 0,
                "candidate_days": int(day_features_df["is_candidate_day"].sum()) if not day_features_df.empty else 0,
                "net_pnl": 0.0,
                "return_pct": 0.0,
                "avg_trade_pnl": 0.0,
                "median_trade_pnl": 0.0,
                "win_rate": 0.0,
            }
        else:
            summary = {
                "trades": int(len(trades_df)),
                "candidate_days": int(day_features_df["is_candidate_day"].sum()),
                "net_pnl": float(trades_df["pnl"].sum()),
                "return_pct": float((trades_df["pnl"].sum() / self.starting_cash) * 100.0),
                "avg_trade_pnl": float(trades_df["pnl"].mean()),
                "median_trade_pnl": float(trades_df["pnl"].median()),
                "win_rate": float((trades_df["pnl"] > 0).mean()),
                "best_trade_pnl": float(trades_df["pnl"].max()),
                "worst_trade_pnl": float(trades_df["pnl"].min()),
            }

        return {
            "summary": summary,
            "trade_log": trades_df,
            "equity_curve": eq_df,
            "day_features": day_features_df,
            "debug_events": debug_df,
        }
