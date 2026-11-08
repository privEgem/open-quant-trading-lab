from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


# Locked audit-only challenger backtester for the late-day downside-continuation idea.
# This file is intentionally separate from the main benchmark backtester so the challenger
# can be audited without silently mutating the current main branch.
class BacktesterChallengerLocked:
    def __init__(
        self,
        file,
        starting_cash=100_000,
        risk_frac=0.0033,
        slippage_bps=1.0,
        commission_per_share=0.0,
        max_position_notional_pct=0.10,
        signal_hour_ny=15,
        signal_minute_ny=0,
        exit_hour_ny=15,
        exit_minute_ny=30,
        day_return_threshold=-0.007,   # -0.7%
        below_vwap_min_bars=60,
        verbose=True,
    ):
        # ----------------------------
        # Frozen challenger parameters
        # ----------------------------
        self.file = file
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)

        self.risk_frac = float(risk_frac)
        self.slippage_bps = float(slippage_bps)
        self.commission_per_share = float(commission_per_share)
        self.max_position_notional_pct = float(max_position_notional_pct)

        # These are locked because the research lane froze them explicitly.
        self.signal_hour_ny = int(signal_hour_ny)
        self.signal_minute_ny = int(signal_minute_ny)
        self.exit_hour_ny = int(exit_hour_ny)
        self.exit_minute_ny = int(exit_minute_ny)
        self.day_return_threshold = float(day_return_threshold)
        self.below_vwap_min_bars = int(below_vwap_min_bars)

        self.ny_tz = ZoneInfo("America/New_York")
        self.verbose = bool(verbose)

        # ----------------------------
        # Position state
        # ----------------------------
        self.position = 0
        self.entry_price = 0.0
        self.shares = 0
        self.stop_price = 0.0
        self.entry_ts = None
        self.entry_risk_dollars = 0.0
        self.entry_session = None

        # This challenger allows at most one trade per day.
        self.traded_sessions = set()

        self.trades = []
        self.trade_log = []
        self.equity_curve = []

        self.df = self._load_and_prepare_data()

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
        df["date_ny"] = ts_ny.dt.date
        df["hour_ny"] = ts_ny.dt.hour
        df["minute_ny"] = ts_ny.dt.minute
        df["time_ny"] = ts_ny.dt.time

        session_mask = (
            ((df["hour_ny"] > 9) | ((df["hour_ny"] == 9) & (df["minute_ny"] >= 30)))
            & (df["hour_ny"] < 16)
        )
        df["is_regular_session"] = session_mask

        df["session_key"] = df["date_ny"].astype(str).where(df["is_regular_session"], None)

        # Session-reset VWAP is intentionally different from the benchmark branch.
        # That is part of the challenger's frozen definition.
        pxv = df["close"] * df["volume"]
        pxv = pxv.where(df["is_regular_session"], np.nan)
        vol = df["volume"].where(df["is_regular_session"], np.nan)

        df["session_vwap"] = (
            pxv.groupby(df["session_key"]).cumsum()
            / vol.groupby(df["session_key"]).cumsum()
        )

        # Regular-session open for day-return calculation.
        session_open = (
            df.loc[df["is_regular_session"]]
            .groupby("session_key")["open"]
            .transform("first")
        )
        df.loc[df["is_regular_session"], "session_open"] = session_open
        df["session_open"] = df.groupby("session_key")["session_open"].ffill()
        df["day_return_from_open"] = (df["close"] / df["session_open"]) - 1.0

        # Persistent below-VWAP run length within the same regular session.
        below_vwap = (
            df["is_regular_session"]
            & df["session_vwap"].notna()
            & (df["close"] < df["session_vwap"])
        )
        df["below_vwap"] = below_vwap

        run_lengths = np.zeros(len(df), dtype=int)
        prev_key = None
        run = 0
        for i in range(len(df)):
            key = df.at[i, "session_key"]
            if pd.isna(key) or key != prev_key:
                run = 0
            if bool(df.at[i, "below_vwap"]):
                run += 1
            else:
                run = 0
            run_lengths[i] = run
            prev_key = key
        df["below_vwap_run"] = run_lengths

        return df

    def _slip_entry(self, raw_price: float) -> float:
        return float(raw_price) * (1.0 - self.slippage_bps / 10_000.0)

    def _slip_exit(self, raw_price: float) -> float:
        return float(raw_price) * (1.0 + self.slippage_bps / 10_000.0)

    def _size_position(self, entry_price: float, stop_price: float) -> int:
        # Explanation:
        # For a short, the stop must be ABOVE the entry.
        # If that is not true, the setup has non-positive risk and we skip it.
        # The research side confirmed this as an implementation choice for a previously
        # undefined edge case, not a frozen-rule drift.
        per_share_risk = stop_price - entry_price
        if per_share_risk <= 0:
            return 0

        risk_budget = self.cash * self.risk_frac
        shares_risk = int(risk_budget // per_share_risk)

        notional_cap = self.cash * self.max_position_notional_pct
        shares_notional = int(notional_cap // max(entry_price, 1e-9))

        return max(0, min(shares_risk, shares_notional))

    def _enter_short(self, row_next: pd.Series, signal_row: pd.Series) -> None:
        raw_entry = float(row_next["open"])
        entry_price = self._slip_entry(raw_entry)

        stop_price = max(float(signal_row["high"]), float(signal_row["session_vwap"]))
        shares = self._size_position(entry_price, stop_price)
        if shares <= 0:
            return

        self.position = -1
        self.entry_price = entry_price
        self.shares = shares
        self.stop_price = stop_price
        self.entry_ts = row_next["timestamp"]
        self.entry_session = signal_row["session_key"]

        per_share_risk = self.stop_price - self.entry_price
        self.entry_risk_dollars = per_share_risk * shares

        commission = self.commission_per_share * shares
        self.cash -= commission

    def _exit_short(self, exit_row: pd.Series, reason: str, raw_exit_price: float) -> None:
        exit_price = self._slip_exit(raw_exit_price)

        pnl = (self.entry_price - exit_price) * self.shares
        commission = self.commission_per_share * self.shares
        pnl_after_cost = pnl - commission
        self.cash += pnl_after_cost

        r_multiple = pnl_after_cost / (self.entry_risk_dollars + 1e-9)

        trade = {
            "entry_ts": self.entry_ts,
            "exit_ts": exit_row["timestamp"],
            "side": -1,
            "entry_price": self.entry_price,
            "shares": self.shares,
            "stop_price": self.stop_price,
            "exit_price": exit_price,
            "pnl": pnl_after_cost,
            "R": r_multiple,
            "reason": reason,
            "session": self.entry_session,
            "side_label": "short",
            "entry_hour_ny": pd.Timestamp(self.entry_ts).tz_convert(self.ny_tz).hour if self.entry_ts is not None else np.nan,
            "exit_month_ny": pd.Timestamp(exit_row["timestamp"]).tz_convert(self.ny_tz).to_period("M").strftime("%Y-%m"),
        }
        self.trade_log.append(trade)
        self.trades.append(pnl_after_cost)

        self.position = 0
        self.entry_price = 0.0
        self.shares = 0
        self.stop_price = 0.0
        self.entry_ts = None
        self.entry_risk_dollars = 0.0
        self.entry_session = None

    def run(self) -> None:
        df = self.df

        for i in range(len(df) - 1):
            row = df.iloc[i]
            row_next = df.iloc[i + 1]

            # Exit logic first so stop/time checks happen before any hypothetical new signal.
            if self.position == -1:
                # Stop hit intrabar.
                if float(row["high"]) >= self.stop_price:
                    self._exit_short(row, reason="stop", raw_exit_price=self.stop_price)

                # 15:30 exit point: implemented as exit at the 15:30 bar open.
                # This was explicitly disclosed and accepted by the research lane.
                elif (
                    bool(row_next["is_regular_session"])
                    and int(row_next["hour_ny"]) == self.exit_hour_ny
                    and int(row_next["minute_ny"]) == self.exit_minute_ny
                ):
                    self._exit_short(row_next, reason="time_exit", raw_exit_price=float(row_next["open"]))

            # Entry logic: exactly one 15:00 NY signal evaluation, short-only.
            if self.position == 0:
                if (
                    bool(row["is_regular_session"])
                    and int(row["hour_ny"]) == self.signal_hour_ny
                    and int(row["minute_ny"]) == self.signal_minute_ny
                ):
                    session_key = row["session_key"]

                    if (
                        session_key not in self.traded_sessions
                        and float(row["day_return_from_open"]) <= self.day_return_threshold
                        and float(row["close"]) < float(row["session_vwap"])
                        and int(row["below_vwap_run"]) >= self.below_vwap_min_bars
                    ):
                        self._enter_short(row_next=row_next, signal_row=row)
                        if self.position == -1:
                            self.traded_sessions.add(session_key)

            # Mark-to-market equity for metrics.
            mtm_price = float(row["close"])
            open_pnl = 0.0
            if self.position == -1:
                open_pnl = (self.entry_price - mtm_price) * self.shares
            self.equity_curve.append(self.cash + open_pnl)

            if self.cash <= 0:
                print("Account blown up — stopping test")
                break

        if self.verbose:
            self.results()

    def results(self) -> None:
        eq = pd.Series(self.equity_curve)

        if len(eq) < 2:
            print("Not enough equity points to compute stats.")
            return

        returns = eq.pct_change().dropna()
        sharpe = np.sqrt(252 * 390) * returns.mean() / (returns.std() + 1e-9)

        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd = dd.min()

        winrate = np.mean(np.array(self.trades) > 0) if self.trades else 0.0
        avg_win = np.mean([t for t in self.trades if t > 0]) if any(t > 0 for t in self.trades) else 0.0
        avg_loss = np.mean([t for t in self.trades if t < 0]) if any(t < 0 for t in self.trades) else 0.0
        expectancy = (winrate * avg_win) + ((1 - winrate) * avg_loss) if self.trades else 0.0

        gross_profit = sum(t for t in self.trades if t > 0)
        gross_loss = abs(sum(t for t in self.trades if t < 0))
        profit_factor = gross_profit / (gross_loss + 1e-9) if self.trades else 0.0

        print("\n========= CHALLENGER LOCKED RESULTS =========")
        print(f"bps = {self.slippage_bps:.1f}")
        print("Name: 15:00 downside drift after persistent downside acceptance")
        print(f"SIGNAL TIME NY: {self.signal_hour_ny:02d}:{self.signal_minute_ny:02d}")
        print(f"EXIT TIME NY: {self.exit_hour_ny:02d}:{self.exit_minute_ny:02d}")
        print(f"DAY RETURN THRESHOLD: {self.day_return_threshold:.4f}")
        print(f"BELOW VWAP MIN BARS: {self.below_vwap_min_bars}")
        print("SIDE_MODE: short_only")
        print("VWAP MODE: session-reset VWAP")
        print("TARGET: none")
        print(f"Start Equity: ${self.starting_cash:,.2f}")
        print(f"Final Equity: ${eq.iloc[-1]:,.2f}")
        print(f"Total Return: {(eq.iloc[-1] / self.starting_cash - 1) * 100:.2f}%")
        print(f"Sharpe: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd * 100:.2f}%")
        print(f"Winrate: {winrate * 100:.2f}%")
        print(f"Avg Win: ${avg_win:,.2f} | Avg Loss: ${avg_loss:,.2f}")
        print(f"Expectancy/trade: ${expectancy:,.4f}")
        print(f"Profit Factor: {profit_factor:.3f}")
        print(f"Trades: {len(self.trades)}")
        print(f"MAX_POSITION_NOTIONAL_PCT: {self.max_position_notional_pct:.2f}")

        if self.trade_log:
            trade_df = pd.DataFrame(self.trade_log)
            print("\nExit reason breakdown:")
            print(trade_df.groupby("reason")["R"].agg(["count", "mean", "median"]))
