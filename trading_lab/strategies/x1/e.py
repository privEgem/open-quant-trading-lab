
"""
Comparator backtester for X1-E, the benchmark-adjacent preserved repair/reclaim branch.

This module is kept for X1 comparison memory and should not be silently moved back into active candidate status.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from zoneinfo import ZoneInfo


class BacktesterX1OvernightDamageRepairReclaim:
    """
    X1-only backtester for a gap-down morning damage-repair reclaim idea.

    Core idea:
    - require a meaningful overnight gap down from the prior regular-session close
    - require extra downside damage from the open through 10:45 NY
    - require the 10:45 NY close to reclaim session VWAP
    - require at least a partial repair of the open-to-low damage
    - allow the signal even if price is still below the session open
    - enter long on the next bar open and give the move a late-morning continuation attempt
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
        gap_down_threshold_pct: float = 0.0020,
        min_open_to_low_damage_pct: float = 0.0040,
        signal_hour_ny: int = 10,
        signal_minute_ny: int = 45,
        min_recovery_fraction: float = 0.55,
        time_exit_hour_ny: int = 12,
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
        self.gap_down_threshold_pct = float(gap_down_threshold_pct)
        self.min_open_to_low_damage_pct = float(min_open_to_low_damage_pct)
        self.signal_hour_ny = int(signal_hour_ny)
        self.signal_minute_ny = int(signal_minute_ny)
        self.min_recovery_fraction = float(min_recovery_fraction)
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
            pre_signal_low = float(pre_signal["low"].min())
            signal_close = float(signal_row["close"])
            signal_vwap = float(signal_row["session_vwap"])
            open_to_low_damage_pct = (pre_signal_low / session_open) - 1.0
            recovery_fraction = (signal_close - pre_signal_low) / max(session_open - pre_signal_low, 1e-12)

            is_candidate_day = (
                (gap is not None)
                and (gap <= -self.gap_down_threshold_pct)
                and (open_to_low_damage_pct <= -self.min_open_to_low_damage_pct)
                and (signal_close > signal_vwap)
                and (recovery_fraction >= self.min_recovery_fraction)
            )

            day_features.append(
                {
                    "date_ny": date_key,
                    "gap_from_prev_close": gap,
                    "session_open": session_open,
                    "pre_signal_low": pre_signal_low,
                    "signal_close": signal_close,
                    "signal_vwap": signal_vwap,
                    "open_to_low_damage_pct": open_to_low_damage_pct,
                    "recovery_fraction": recovery_fraction,
                    "signal_above_open": bool(signal_close > session_open),
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
            stop_raw = pre_signal_low
            if stop_raw >= entry_raw:
                continue

            stop_distance = entry_raw - stop_raw
            risk_dollars = self.cash * self.risk_frac
            shares_by_risk = risk_dollars / stop_distance
            shares_by_cash = self.cash // entry_raw
            shares_by_cap = (self.cash * self.max_position_notional_pct) // entry_raw
            shares = int(max(min(shares_by_risk, shares_by_cash, shares_by_cap), 0))
            if shares <= 0:
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
                if float(bar["low"]) <= stop_raw:
                    exit_price = self._apply_slippage(stop_raw, "sell")
                    exit_reason = "stop"
                    exit_ts = bar["timestamp"]
                    break
                if float(bar["high"]) >= tp_price:
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

            gross_pnl = (exit_price - entry_price) * shares
            commission_cost = self.commission_per_share * shares * 2
            pnl = gross_pnl - commission_cost
            self.cash += pnl

            trades.append(
                {
                    "date_ny": date_key,
                    "entry_ts": nxt["timestamp"],
                    "entry_price": entry_price,
                    "exit_ts": exit_ts,
                    "exit_price": exit_price,
                    "shares": shares,
                    "pnl": pnl,
                    "gross_pnl": gross_pnl,
                    "exit_reason": exit_reason,
                    "gap_from_prev_close": gap,
                    "session_open": session_open,
                    "pre_signal_low": pre_signal_low,
                    "signal_close": signal_close,
                    "signal_vwap": signal_vwap,
                    "open_to_low_damage_pct": open_to_low_damage_pct,
                    "recovery_fraction": recovery_fraction,
                    "stop_raw": stop_raw,
                    "entry_raw": entry_raw,
                    "tp_price": tp_price,
                }
            )

        trades_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        day_features_df = pd.DataFrame(day_features)

        summary = {
            "starting_cash": self.starting_cash,
            "ending_cash": self.cash,
            "net_pnl": self.cash - self.starting_cash,
            "return_pct": ((self.cash / self.starting_cash) - 1.0) * 100.0,
            "candidate_days": int(day_features_df["is_candidate_day"].sum()) if not day_features_df.empty else 0,
            "trades": int(len(trades_df)),
        }

        if not trades_df.empty:
            summary.update(
                {
                    "wins": int((trades_df["pnl"] > 0).sum()),
                    "losses": int((trades_df["pnl"] <= 0).sum()),
                    "win_rate_pct": float((trades_df["pnl"] > 0).mean() * 100.0),
                    "median_trade_pnl": float(trades_df["pnl"].median()),
                    "avg_trade_pnl": float(trades_df["pnl"].mean()),
                    "best_trade_pnl": float(trades_df["pnl"].max()),
                    "worst_trade_pnl": float(trades_df["pnl"].min()),
                }
            )

        if self.verbose:
            print("X1-E overnight damage repair reclaim summary")
            print(summary)

        return {
            "summary": summary,
            "trades": trades_df,
            "equity_curve": equity_df,
            "day_features": day_features_df,
        }
