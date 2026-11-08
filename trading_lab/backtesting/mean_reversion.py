import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

from trading_lab.features import (
    LEGACY_REPLAY_FULL_CSV_CUMULATIVE_VWAP_V1,
    NYSE as NYSE_EXECUTION_CALENDAR,
    generate_features,
)


EXECUTION_SESSION_CONTRACT = "nyse_regular_session_intraday_flat_v1"


class BenchmarkSessionDataError(RuntimeError):
    """Raised when benchmark bars cannot satisfy intraday-flat session evidence."""


# Benchmark backtester for the current filtered asymmetric mean-reversion candidate.
class BacktesterMRCandidateFiltered:
    def __init__(
        self,
        file,
        starting_cash=100_000,
        risk_frac=0.0033,
        rr=2.5,
        atr_mult=2.5,
        slippage_bps=1.0,
        commission_per_share=0.0,
        allowed_hours_ny=None,
        no_new_entries_after_hour_ny=15,
        no_new_entries_after_minute_ny=30,
        max_position_notional_pct=0.10,
        atr_window=200,
        atr_q_low=0.55,
        atr_q_high=0.95,
        trend_fast=21,
        trend_slow=50,
        verbose=True,
        side_mode="both",
        long_hours_ny=None,
        short_hours_ny=None,
    ):
        # Store the frozen candidate settings here so research runs stay explicit and reproducible.
        self.file = file
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)

        self.risk_frac = float(risk_frac)
        self.rr = float(rr)
        self.atr_mult = float(atr_mult)
        self.slippage_bps = float(slippage_bps)
        self.commission_per_share = float(commission_per_share)

        # Keep session-hour and side-hour controls separate so broad time filters and asymmetric long/short rules are both visible.
        self.allowed_hours_ny = None if allowed_hours_ny is None else set(allowed_hours_ny)
        self.no_new_entries_after_hour_ny = int(no_new_entries_after_hour_ny)
        self.no_new_entries_after_minute_ny = int(no_new_entries_after_minute_ny)
        self.ny_tz = ZoneInfo("America/New_York")

        self.long_hours_ny = None if long_hours_ny is None else set(long_hours_ny)
        self.short_hours_ny = None if short_hours_ny is None else set(short_hours_ny)

        self.atr_window = int(atr_window)
        self.atr_q_low = float(atr_q_low)
        self.atr_q_high = float(atr_q_high)
        self.trend_fast = int(trend_fast)
        self.trend_slow = int(trend_slow)
        # Replay is pinned to the historical full-CSV cumulative VWAP contract.
        # This class is audit/backtest-only; no live path imports it.
        self.feature_profile = LEGACY_REPLAY_FULL_CSV_CUMULATIVE_VWAP_V1

        # Position state tracks the currently open trade and is reused across the bar-by-bar simulation loop.
        self.position = 0
        self.entry_price = 0.0
        self.shares = 0
        self.stop_price = 0.0
        self.tp_price = 0.0
        self.entry_ts = None
        self.risk_budget_dollars = 0.0
        self.initial_stop_risk_dollars = 0.0

        # Collect both summarized trade PnL and full trade records so later diagnostics can analyze behavior in detail.
        self.trades = []
        self.trade_log = []
        self.equity_curve = []
        self.equity_timestamps = []
        self.execution_session_contract = EXECUTION_SESSION_CONTRACT

        self.df = self._load_and_prepare_data()
        self._execution_sessions = self._build_execution_session_metadata(self.df["timestamp"])

        self.max_position_notional_pct = float(max_position_notional_pct)

        self.verbose = bool(verbose)

        self.side_mode = str(side_mode).lower()
        if self.side_mode not in {"both", "long_only", "short_only"}:
            raise ValueError(f"Invalid side_mode: {self.side_mode}")


    # ----------------------------
    # Data prep
    # ----------------------------

    # Load the minute data once, validate it, then build the same indicator set the live branch expects.
    def _load_and_prepare_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.file, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        required_cols = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {self.file}: {missing}")
        df = df[df["volume"] > 0].copy()
        df = df.reset_index(drop=True)

        # Preserve the legacy positive-volume prefilter *before* indicator generation.
        # The selected helper profile then reproduces the previous cumulative full-CSV
        # VWAP, RSI, ATR, ATR band, and completed-15-minute trend calculations.
        return generate_features(
            df,
            self.feature_profile,
            atr_window=self.atr_window,
            atr_q_low=self.atr_q_low,
            atr_q_high=self.atr_q_high,
            trend_fast=self.trend_fast,
            trend_slow=self.trend_slow,
            normalize=False,
        )

        # Resample into completed 15-minute bars so the higher-timeframe trend can be applied without lookahead.
    def _add_15m_trend_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.set_index("timestamp").copy()

        df15 = (
            x.resample("15min")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["close"])
        )

        df15["ema_fast"] = df15["close"].ewm(span=self.trend_fast, adjust=False).mean()
        df15["ema_slow"] = df15["close"].ewm(span=self.trend_slow, adjust=False).mean()
        df15["trend15"] = np.where(df15["ema_fast"] > df15["ema_slow"], 1, -1)

        # Shift the 15-minute trend by one bar so each 1-minute row only sees the most recently completed regime state.
        df15["trend15_lagged"] = df15["trend15"].shift(1)

        x["trend15"] = (
            df15["trend15_lagged"]
            .reindex(x.index, method="ffill")
            .fillna(0)
            .astype(int)
        )

        return x.reset_index()

    # ----------------------------
    # Helpers
    # ----------------------------

    # Model directional slippage simply but consistently so fills are slightly worse than the raw trigger price.
    def _apply_slippage(self, price: float, side: str) -> float:
        slip = self.slippage_bps / 10_000.0
        if side == "buy":
            return price * (1 + slip)
        return price * (1 - slip)

    def _commission(self, shares: int) -> float:
        return self.commission_per_share * shares

    def _build_execution_session_metadata(self, timestamps: pd.Series) -> pd.DataFrame:
        """Resolve each supplied bar against NYSE regular-session calendar bounds."""
        if NYSE_EXECUTION_CALENDAR is None:
            raise RuntimeError(
                "Benchmark execution requires pandas_market_calendars with the NYSE calendar."
            )

        resolved = pd.DataFrame(index=timestamps.index)
        resolved["in_regular_session"] = False
        resolved["session_key"] = pd.Series(pd.NA, index=timestamps.index, dtype="string")
        resolved["market_open"] = pd.Series(pd.NaT, index=timestamps.index, dtype="datetime64[ns, UTC]")
        resolved["market_close"] = pd.Series(pd.NaT, index=timestamps.index, dtype="datetime64[ns, UTC]")
        resolved["is_final_regular_minute"] = False

        valid = timestamps.notna()
        if not valid.any():
            return resolved

        valid_timestamps = pd.DatetimeIndex(timestamps.loc[valid])
        start_date = valid_timestamps.min().tz_convert(self.ny_tz).date()
        end_date = valid_timestamps.max().tz_convert(self.ny_tz).date()
        schedule = NYSE_EXECUTION_CALENDAR.schedule(start_date=start_date, end_date=end_date)
        if schedule.empty:
            return resolved

        opens = pd.DatetimeIndex(schedule["market_open"])
        closes = pd.DatetimeIndex(schedule["market_close"])
        positions = opens.searchsorted(valid_timestamps, side="right") - 1
        session_keys = schedule.index.strftime("%Y-%m-%d")

        for index, timestamp, position in zip(timestamps.loc[valid].index, valid_timestamps, positions):
            if position < 0 or timestamp >= closes[position]:
                continue
            resolved.at[index, "in_regular_session"] = True
            resolved.at[index, "session_key"] = session_keys[position]
            resolved.at[index, "market_open"] = opens[position]
            resolved.at[index, "market_close"] = closes[position]
            resolved.at[index, "is_final_regular_minute"] = (
                timestamp == closes[position] - pd.Timedelta(minutes=1)
            )

        return resolved

    # Stop accepting new entries after the intraday cutoff, while still allowing existing positions to exit naturally.
    def _past_entry_cutoff(self, ts: pd.Timestamp) -> bool:
        ts_ny = ts.tz_convert(self.ny_tz)
        h = ts_ny.hour
        m = ts_ny.minute
        if h > self.no_new_entries_after_hour_ny:
            return True
        if h == self.no_new_entries_after_hour_ny and m >= self.no_new_entries_after_minute_ny:
            return True
        return False

    def _reset_position_state(self) -> None:
        self.position = 0
        self.entry_price = 0.0
        self.shares = 0
        self.stop_price = 0.0
        self.tp_price = 0.0
        self.entry_ts = None
        self.risk_budget_dollars = 0.0
        self.initial_stop_risk_dollars = 0.0

    # Record each completed trade with enough detail for later diagnostics, including realized R multiple and exit reason.
    def _record_trade(self, pnl: float, exit_reason: str, exit_ts: pd.Timestamp) -> None:
        if self.initial_stop_risk_dollars <= 0:
            raise RuntimeError("Completed trade requires positive initial_stop_risk_dollars.")
        R = pnl / self.initial_stop_risk_dollars
        self.trades.append(float(pnl))
        self.trade_log.append(
            {
                "entry_ts": self.entry_ts,
                "exit_ts": exit_ts,
                "side": int(self.position),
                "entry_price": float(self.entry_price),
                "shares": int(self.shares),
                "stop_price": float(self.stop_price),
                "tp_price": float(self.tp_price),
                "risk_budget_dollars": float(self.risk_budget_dollars),
                "initial_stop_risk_dollars": float(self.initial_stop_risk_dollars),
                "pnl": float(pnl),
                "R": float(R),
                "reason": exit_reason,
            }
        )

    def _manage_open_position(self, row: pd.Series) -> None:
        """Resolve existing regular-session position risk before considering entries."""
        high = float(row.high)
        low = float(row.low)

        if self.position == 1:
            hit_stop = low <= self.stop_price
            hit_tp = high >= self.tp_price
            if hit_stop:
                exit_price = self._apply_slippage(self.stop_price, "sell")
                pnl = (exit_price - self.entry_price) * self.shares - self._commission(self.shares)
                self.cash += pnl
                self._record_trade(pnl, "stop", row.timestamp)
                self._reset_position_state()
            elif hit_tp:
                exit_price = self._apply_slippage(self.tp_price, "sell")
                pnl = (exit_price - self.entry_price) * self.shares - self._commission(self.shares)
                self.cash += pnl
                self._record_trade(pnl, "tp", row.timestamp)
                self._reset_position_state()

        elif self.position == -1:
            hit_stop = high >= self.stop_price
            hit_tp = low <= self.tp_price
            if hit_stop:
                exit_price = self._apply_slippage(self.stop_price, "buy")
                pnl = (self.entry_price - exit_price) * self.shares - self._commission(self.shares)
                self.cash += pnl
                self._record_trade(pnl, "stop", row.timestamp)
                self._reset_position_state()
            elif hit_tp:
                exit_price = self._apply_slippage(self.tp_price, "buy")
                pnl = (self.entry_price - exit_price) * self.shares - self._commission(self.shares)
                self.cash += pnl
                self._record_trade(pnl, "tp", row.timestamp)
                self._reset_position_state()

    def _flatten_session_close(self, row: pd.Series) -> None:
        """Close a surviving regular-session position on its final expected minute."""
        if self.position == 0:
            return
        if pd.isna(row.close):
            raise BenchmarkSessionDataError(
                f"Missing close evidence for required session-close exit at {row.timestamp}."
            )

        close = float(row.close)
        if self.position == 1:
            exit_price = self._apply_slippage(close, "sell")
            pnl = (exit_price - self.entry_price) * self.shares - self._commission(self.shares)
        else:
            exit_price = self._apply_slippage(close, "buy")
            pnl = (self.entry_price - exit_price) * self.shares - self._commission(self.shares)
        self.cash += pnl
        self._record_trade(pnl, "session_close", row.timestamp)
        self._reset_position_state()

    def _raise_if_missing_session_close(
        self,
        row: pd.Series,
        session: pd.Series,
        next_session: pd.Series | None,
    ) -> None:
        """Fail closed instead of carrying a position past its expected final session bar."""
        if self.position == 0:
            return
        if next_session is not None and (
            bool(next_session.in_regular_session)
            and next_session.session_key == session.session_key
        ):
            return
        expected_final = session.market_close - pd.Timedelta(minutes=1)
        raise BenchmarkSessionDataError(
            "Open position lacks required final regular-session bar at "
            f"{expected_final}; cannot carry position past {row.timestamp}."
        )

    def _entry_eligible(self, row: pd.Series, nxt: pd.Series, hour_ny: int) -> bool:
        """Apply gates that exist solely to permit or reject a new entry."""
        if pd.isna(row.atr_q_low_band) or pd.isna(row.atr_q_high_band):
            return False
        if not (row.atr >= row.atr_q_low_band and row.atr <= row.atr_q_high_band):
            return False
        if pd.isna(nxt.open) or pd.isna(row.atr) or pd.isna(nxt.volume) or float(nxt.volume) <= 0:
            return False
        if self.allowed_hours_ny is not None and hour_ny not in self.allowed_hours_ny:
            return False
        return not self._past_entry_cutoff(row.timestamp)

    def _mark_to_market(self, close: float) -> float:
        if self.position == 0:
            return self.cash
        return self.cash + ((close - self.entry_price) * self.shares * self.position)

    # ----------------------------
    # Signal
    # ----------------------------

    # The raw signal is only allowed when volatility, VWAP stretch, RSI, and higher-timeframe trend all agree.
    def signal(self, row) -> int:
        if pd.isna(row.atr) or row.atr <= 0:
            return 0
        if pd.isna(row.vwap_used) or pd.isna(row.rsi):
            return 0
        if pd.isna(row.trend15) or row.trend15 == 0:
            return 0

        vwap_dist = (row.close - row.vwap_used) / row.atr

        # Only fade weakness in an uptrend and strength in a downtrend; the trend filter is a hard veto, not a soft preference.
        if row.trend15 == 1:
            if vwap_dist < -0.8 and row.rsi < 30:
                return 1

        if row.trend15 == -1:
            if vwap_dist > 0.8 and row.rsi > 70:
                return -1

        return 0

    # Position size is constrained by stop-based risk, available cash, and the hard notional concentration cap.
    def size_position(self, price: float, atr: float) -> tuple[int, float]:
        risk_dollars = self.cash * self.risk_frac
        stop_dist = atr * self.atr_mult

        if stop_dist <= 0:
            return 0, 0.0

        shares_by_risk = risk_dollars / stop_dist
        shares_by_cash = self.cash // price
        shares_by_notional = (self.cash * self.max_position_notional_pct) // price

        shares = int(max(min(shares_by_risk, shares_by_cash, shares_by_notional), 0))
        return shares, risk_dollars

    # ----------------------------
    # Main loop
    # ----------------------------

    # Simulate the strategy one minute at a time using the current bar for signal context and the next bar open for entries.
    def run(self) -> None:
        n = len(self.df)

        for i in range(n):
            row = self.df.iloc[i]
            session = self._execution_sessions.iloc[i]
            nxt = self.df.iloc[i + 1] if i + 1 < n else None
            next_session = self._execution_sessions.iloc[i + 1] if nxt is not None else None

            # Calendar regular-session bounds exclude pre/post-market, holidays, and close-boundary bars.
            if not bool(session.in_regular_session):
                continue

            ts_ny = row.timestamp.tz_convert(self.ny_tz)
            hour_ny = ts_ny.hour
            close = float(row.close)

            # Existing risk is resolved before entry-only gates. A same-bar exit can therefore
            # still be followed by a normal next-bar-open entry when all entry gates pass.
            self._manage_open_position(row)

            # Stop/TP owns final-bar priority. A survivor must flatten at the calendar close.
            if bool(session.is_final_regular_minute):
                self._flatten_session_close(row)
            else:
                self._raise_if_missing_session_close(row, session, next_session)

            # Fix this bar's portfolio sample before a next-bar-open entry can mutate
            # cash or position state.  Equity timestamped at row.timestamp must not
            # include an entry, commission, or mark that starts at nxt.timestamp.
            self.equity_curve.append(self._mark_to_market(close))
            self.equity_timestamps.append(row.timestamp)

            # Apply side mode and asymmetric hour filters after the raw signal so long-only/short-only and hour-specific experiments stay transparent.
            # Entries
            same_regular_session = next_session is not None and (
                bool(next_session.in_regular_session)
                and next_session.session_key == session.session_key
            )
            if self.position == 0 and same_regular_session and self._entry_eligible(row, nxt, hour_ny):
                sig = self.signal(row)
                if self.side_mode == "long_only" and sig == -1:
                    sig = 0
                elif self.side_mode == "short_only" and sig == 1:
                    sig = 0

                if sig == 1 and self.long_hours_ny is not None and hour_ny not in self.long_hours_ny:
                    sig = 0
                elif sig == -1 and self.short_hours_ny is not None and hour_ny not in self.short_hours_ny:
                    sig = 0

                if sig != 0:
                    # Entries are assumed at the next bar open, then adjusted for slippage to approximate a realistic delayed fill.
                    entry_raw = float(nxt.open)
                    atr = float(row.atr)

                    shares, risk_dollars = self.size_position(entry_raw, atr)
                    if shares > 0:
                        stop_dist = atr * self.atr_mult

                        if sig == 1:
                            entry_price = self._apply_slippage(entry_raw, "buy")
                            stop_price = entry_price - stop_dist
                            tp_price = entry_price + (stop_dist * self.rr)
                        else:
                            entry_price = self._apply_slippage(entry_raw, "sell")
                            stop_price = entry_price + stop_dist
                            tp_price = entry_price - (stop_dist * self.rr)

                        initial_stop_risk_dollars = abs(entry_price - stop_price) * shares
                        if initial_stop_risk_dollars <= 0:
                            raise RuntimeError("Opened trade requires positive initial_stop_risk_dollars.")

                        # Commission is charged at entry and position state is updated only after the simulated order passes all checks.
                        self.cash -= self._commission(shares)

                        self.position = sig
                        self.entry_price = entry_price
                        self.shares = shares
                        self.stop_price = float(stop_price)
                        self.tp_price = float(tp_price)
                        self.entry_ts = nxt.timestamp
                        self.risk_budget_dollars = float(risk_dollars)
                        self.initial_stop_risk_dollars = float(initial_stop_risk_dollars)

            if self.cash <= 0:
                print("Account blown up — stopping test")
                break

        if self.verbose:
            self.results()

    # ----------------------------
    # Metrics
    # ----------------------------

    # Summarize the backtest with basic portfolio and trade statistics for quick benchmark comparison.
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

        print("\n========= FILTERED CANDIDATE MR RESULTS =========")
        print(
            f"RR: {self.rr} | ATR_MULT: {self.atr_mult} | "
            f"RISK_FRAC: {self.risk_frac} | SLIP_BPS: {self.slippage_bps}"
        )
        print(f"ATR BAND: {self.atr_q_low:.2f}-{self.atr_q_high:.2f} | TREND FAST/SLOW: {self.trend_fast}/{self.trend_slow}")
        print(f"NO NEW ENTRIES AFTER {self.no_new_entries_after_hour_ny:02d}:{self.no_new_entries_after_minute_ny:02d} NY")
        print(f"ALLOWED_HOURS_NY: {sorted(self.allowed_hours_ny) if self.allowed_hours_ny else 'ALL'}")
        print(f"Start Equity: ${self.starting_cash:,.2f}")
        print(f"Final Equity: ${eq.iloc[-1]:,.2f}")
        print(f"Total Return: {(eq.iloc[-1] / self.starting_cash - 1) * 100:.2f}%")
        print(f"Sharpe: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd * 100:.2f}%")
        print(f"Winrate: {winrate * 100:.2f}%")
        print(f"Avg Win: ${avg_win:,.2f} | Avg Loss: ${avg_loss:,.2f}")
        print(f"Expectancy/trade: ${expectancy:,.4f}")
        print(f"Trades: {len(self.trades)}")
        print(f"MAX_POSITION_NOTIONAL_PCT: {self.max_position_notional_pct:.2f}")
        print(f"SIDE_MODE: {self.side_mode}")
        print(f"LONG_HOURS_NY: {sorted(self.long_hours_ny) if self.long_hours_ny else 'ALL'}")
        print(f"SHORT_HOURS_NY: {sorted(self.short_hours_ny) if self.short_hours_ny else 'ALL'}")
