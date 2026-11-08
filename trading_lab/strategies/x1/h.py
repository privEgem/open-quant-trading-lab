
"""
X1-H backtester for the frozen flat-open orderly-weakness short candidate.

This module is preserved as a quarantined X1 reference and should not be interpreted as MAIN benchmark logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo


class BacktesterX1FlatOpenOrderlyWeaknessDrift:
    """
    X1-only backtester for a flat-open orderly weakness drift short.

    Core idea:
    - overnight gap is roughly flat relative to the prior regular-session close
    - by 11:00 NY, price is modestly below the session open
    - by 11:00 NY, price is also meaningfully below session VWAP
    - the bounce off the morning low is still small, so weakness looks orderly rather than repaired
    - short the next bar open and hold for a noon drift window
    """

    def __init__(
        self,
        file: str,
        starting_cash: float = 100_000,
        risk_frac: float = 0.0033,
        slippage_bps: float = 1.0,
        commission_per_share: float = 0.0,
        max_position_notional_pct: float = 0.10,
        flat_gap_abs_max_pct: float = 0.0015,
        min_open_weakness_pct: float = 0.0012,
        min_below_vwap_pct: float = 0.0005,
        max_bounce_from_low_pct: float = 0.0050,
        signal_hour_ny: int = 11,
        signal_minute_ny: int = 0,
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
        self.flat_gap_abs_max_pct = float(flat_gap_abs_max_pct)
        self.min_open_weakness_pct = float(min_open_weakness_pct)
        self.min_below_vwap_pct = float(min_below_vwap_pct)
        self.max_bounce_from_low_pct = float(max_bounce_from_low_pct)
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
            pre_low = float(pre_signal["low"].min())
            pre_high = float(pre_signal["high"].max())

            open_weakness_pct = 1.0 - (signal_close / session_open)
            below_vwap_pct = 1.0 - (signal_close / signal_vwap)
            bounce_from_low_pct = (signal_close / pre_low) - 1.0

            is_candidate_day = (
                (gap is not None)
                and (abs(gap) <= self.flat_gap_abs_max_pct)
                and (open_weakness_pct >= self.min_open_weakness_pct)
                and (below_vwap_pct >= self.min_below_vwap_pct)
                and (bounce_from_low_pct <= self.max_bounce_from_low_pct)
            )

            day_features.append(
                {
                    "date_ny": date_key,
                    "gap_from_prev_close": gap,
                    "session_open": session_open,
                    "signal_close": signal_close,
                    "signal_vwap": signal_vwap,
                    "pre_low": pre_low,
                    "pre_high": pre_high,
                    "open_weakness_pct": open_weakness_pct,
                    "below_vwap_pct": below_vwap_pct,
                    "bounce_from_low_pct": bounce_from_low_pct,
                    "is_candidate_day": bool(is_candidate_day),
                }
            )

            if not is_candidate_day:
                continue
            if day_pos + 1 >= len(idxs):
                continue

            next_idx = idxs[day_pos + 1]
            nxt = df.loc[next_idx]
            entry_raw = float(nxt["open"])
            stop_raw = pre_high
            if stop_raw <= entry_raw:
                continue

            stop_distance = stop_raw - entry_raw
            risk_dollars = self.cash * self.risk_frac
            shares_by_risk = risk_dollars / stop_distance
            shares_by_cash = self.cash // entry_raw
            shares_by_cap = (self.cash * self.max_position_notional_pct) // entry_raw
            shares = int(max(min(shares_by_risk, shares_by_cash, shares_by_cap), 0))
            if shares <= 0:
                continue

            entry_price = self._apply_slippage(entry_raw, "sell")

            exit_price = None
            exit_reason = None
            exit_ts = None

            for j in idxs:
                if j < next_idx:
                    continue
                bar = df.loc[j]
                if float(bar["high"]) >= stop_raw:
                    exit_price = self._apply_slippage(stop_raw, "buy")
                    exit_reason = "stop"
                    exit_ts = bar["timestamp"]
                    break
                if (int(bar["hour_ny"]) > self.time_exit_hour_ny) or (
                    (int(bar["hour_ny"]) == self.time_exit_hour_ny) and (int(bar["minute_ny"]) >= self.time_exit_minute_ny)
                ):
                    exit_price = self._apply_slippage(float(bar["close"]), "buy")
                    exit_reason = "time_exit"
                    exit_ts = bar["timestamp"]
                    break

            if exit_price is None:
                last_bar = day.iloc[-1]
                exit_price = self._apply_slippage(float(last_bar["close"]), "buy")
                exit_reason = "eod_fallback"
                exit_ts = last_bar["timestamp"]

            gross_pnl = (entry_price - exit_price) * shares
            commission = shares * self.commission_per_share * 2.0
            pnl = gross_pnl - commission
            self.cash += pnl

            trades.append(
                {
                    "date_ny": date_key,
                    "entry_ts": nxt["timestamp"],
                    "exit_ts": exit_ts,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "shares": shares,
                    "gross_pnl": gross_pnl,
                    "commission": commission,
                    "pnl": pnl,
                    "exit_reason": exit_reason,
                    "gap_from_prev_close": gap,
                    "open_weakness_pct": open_weakness_pct,
                    "below_vwap_pct": below_vwap_pct,
                    "bounce_from_low_pct": bounce_from_low_pct,
                    "entry_raw": entry_raw,
                    "stop_raw": stop_raw,
                    "signal_close": signal_close,
                    "signal_vwap": signal_vwap,
                }
            )

        trade_log = pd.DataFrame(trades)
        equity_curve_df = pd.DataFrame(equity_curve)
        day_features_df = pd.DataFrame(day_features)

        if trade_log.empty:
            summary = {
                "trades": 0,
                "candidate_days": int(day_features_df["is_candidate_day"].sum()) if not day_features_df.empty else 0,
                "net_pnl": 0.0,
                "return_pct": 0.0,
                "median_trade_pnl": 0.0,
                "win_rate_pct": 0.0,
                "best_trade_pnl": 0.0,
                "worst_trade_pnl": 0.0,
            }
            return {
                "summary": summary,
                "trade_log": trade_log,
                "equity_curve": equity_curve_df,
                "day_features": day_features_df,
            }

        summary = {
            "trades": int(len(trade_log)),
            "candidate_days": int(day_features_df["is_candidate_day"].sum()),
            "net_pnl": float(trade_log["pnl"].sum()),
            "return_pct": float((trade_log["pnl"].sum() / self.starting_cash) * 100.0),
            "median_trade_pnl": float(trade_log["pnl"].median()),
            "win_rate_pct": float((trade_log["pnl"] > 0).mean() * 100.0),
            "best_trade_pnl": float(trade_log["pnl"].max()),
            "worst_trade_pnl": float(trade_log["pnl"].min()),
        }

        return {
            "summary": summary,
            "trade_log": trade_log,
            "equity_curve": equity_curve_df,
            "day_features": day_features_df,
        }
