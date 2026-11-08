"""Offline equity/trade metrics extracted without any historical audit driver."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from trading_lab.backtesting.mean_reversion import BacktesterMRCandidateFiltered

def _trade_df(bt: BacktesterMRCandidateFiltered) -> pd.DataFrame:
    """Convert the benchmark trade log to a DataFrame with stable columns."""
    if not bt.trade_log:
        return pd.DataFrame(columns=[
            "entry_ts", "exit_ts", "side", "entry_price", "exit_price", "shares",
            "stop_price", "tp_price", "pnl", "R", "reason",
        ])
    df = pd.DataFrame(bt.trade_log).copy()
    if "side_label" not in df.columns and "side" in df.columns:
        df["side_label"] = df["side"].map({1: "long", -1: "short"}).fillna("unknown")
    if "entry_ts" in df.columns:
        ts = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
        df["entry_ts"] = ts
        df["entry_hour_ny"] = ts.dt.tz_convert("America/New_York").dt.hour
    if "exit_ts" in df.columns:
        ts = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")
        df["exit_ts"] = ts
        df["exit_month_ny"] = ts.dt.tz_convert("America/New_York").dt.to_period("M").astype(str)
    return df

def _equity_df(bt: BacktesterMRCandidateFiltered) -> pd.DataFrame:
    if not bt.equity_curve:
        return pd.DataFrame(columns=["bar_index", "timestamp", "equity"])
    if len(bt.equity_timestamps) != len(bt.equity_curve):
        raise ValueError("Benchmark equity timestamps must align one-to-one with equity values.")
    return pd.DataFrame({
        "bar_index": range(len(bt.equity_curve)),
        "timestamp": pd.to_datetime(bt.equity_timestamps, utc=True),
        "equity": bt.equity_curve,
    })

def _metrics(bt: BacktesterMRCandidateFiltered, trades: pd.DataFrame, bps: float) -> dict[str, Any]:
    eq = pd.Series(bt.equity_curve, dtype=float)
    if len(eq) < 2:
        sharpe = max_dd_pct = total_return_pct = 0.0
        final_equity = bt.starting_cash
    else:
        rets = eq.pct_change().dropna()
        sharpe = float((252 * 390) ** 0.5 * rets.mean() / (rets.std() + 1e-9))
        peak = eq.cummax()
        max_dd_pct = float(((eq - peak) / peak).min() * 100.0)
        final_equity = float(eq.iloc[-1])
        total_return_pct = float((final_equity / bt.starting_cash - 1.0) * 100.0)
    if trades.empty:
        winrate_pct = mean_r = median_r = sum_r = profit_factor = 0.0
        trade_count = 0
    else:
        wins = trades.loc[trades["pnl"] > 0, "pnl"]
        losses = trades.loc[trades["pnl"] < 0, "pnl"]
        gross_win = float(wins.sum()) if not wins.empty else 0.0
        gross_loss = abs(float(losses.sum())) if not losses.empty else 0.0
        winrate_pct = float(100.0 * (trades["pnl"] > 0).mean())
        mean_r = float(trades["R"].mean()) if "R" in trades.columns else 0.0
        median_r = float(trades["R"].median()) if "R" in trades.columns else 0.0
        sum_r = float(trades["R"].sum()) if "R" in trades.columns else 0.0
        profit_factor = float(gross_win / (gross_loss + 1e-9))
        trade_count = int(len(trades))
    return {
        "branch": "offline_reference", "source_file": "trading_lab/backtesting/mean_reversion.py",
        "slippage_bps": float(bps), "final_equity": final_equity,
        "total_return_pct": total_return_pct, "sharpe": sharpe,
        "max_drawdown_pct": max_dd_pct, "trades": trade_count,
        "winrate_pct": winrate_pct, "mean_R": mean_r, "median_R": median_r,
        "sum_R": sum_r, "profit_factor": profit_factor,
    }

BENCHMARK_EXECUTION_METHOD_METADATA = {
    "execution_session_contract": "nyse_regular_session_intraday_flat_v1",
    "session_calendar": "NYSE",
    "session_close_exit": "final_regular_session_bar_close_with_directional_slippage",
    "entry_requires_next_bar_same_session": True,
    "equity_sampling": "nyse_regular_session_minutes_only",
    "equity_entry_timing": "current_bar_close_before_next_bar_open_entry",
    "minute_sharpe_annualization_periods": 252 * 390,
}
