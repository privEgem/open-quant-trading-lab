
"""
X1-G backtester for the frozen flat-open orderly-strength long candidate.

This is a preserved RESEARCH-X1 runnable reference and should be read as side-lane memory, not as MAIN or promotion material.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp_ny"] = df["timestamp"].dt.tz_convert("America/New_York")
    df = df[(df["timestamp_ny"].dt.time >= pd.Timestamp("09:30").time()) &
            (df["timestamp_ny"].dt.time <= pd.Timestamp("16:00").time()) &
            (df["volume"] > 0)].copy()
    df["date_ny"] = df["timestamp_ny"].dt.date
    df["time_ny"] = df["timestamp_ny"].dt.strftime("%H:%M")
    return df

def run_backtest(cost_bps: float = 1.0, *, data_path: str | Path):
    df = load_data(data_path)
    day_groups = {d: g.reset_index(drop=True) for d, g in df.groupby("date_ny")}
    days = sorted(day_groups)

    day_rows = []
    prev_close = None
    for d in days:
        g = day_groups[d]
        open0930 = g[g["time_ny"] == "09:30"]["open"].iloc[0] if (g["time_ny"] == "09:30").any() else g.iloc[0]["open"]
        close1600 = g[g["time_ny"] == "16:00"]["close"].iloc[0] if (g["time_ny"] == "16:00").any() else g.iloc[-1]["close"]
        day_rows.append({"date": d, "open0930": open0930, "close1600": close1600, "prev_close": prev_close})
        prev_close = close1600
    day_df = pd.DataFrame(day_rows)
    day_df["gap_pct"] = (day_df["open0930"] / day_df["prev_close"] - 1.0) * 100.0

    trades = []
    for d in days:
        g = day_groups[d]
        if not {"10:30", "10:31", "12:00"}.issubset(set(g["time_ny"])):
            continue

        idx1030 = g.index[g["time_ny"] == "10:30"][0]
        row1030 = g.loc[idx1030]
        pre = g.loc[:idx1030]

        gap_pct = float(day_df.loc[day_df["date"] == d, "gap_pct"].iloc[0])
        ret_from_open = (float(row1030["close"]) / float(g.iloc[0]["open"]) - 1.0) * 100.0
        dist_vwap = (float(row1030["close"]) / float(row1030["vwap"]) - 1.0) * 100.0

        if not (-0.10 <= gap_pct <= 0.10):
            continue
        if not (0.05 <= ret_from_open <= 0.35):
            continue
        if not (dist_vwap >= 0.0):
            continue

        entry = float(g[g["time_ny"] == "10:31"]["open"].iloc[0])
        stop = float(pre["low"].min())
        exit_price = float(g[g["time_ny"] == "12:00"]["close"].iloc[0])
        low_after = float(g[(g["time_ny"] > "10:30") & (g["time_ny"] <= "12:00")]["low"].min())

        stopped = low_after <= stop
        gross = (stop - entry) if stopped else (exit_price - entry)
        costs = entry * (cost_bps / 10000.0) * 2.0
        pnl = gross - costs

        trades.append({
            "date": str(d),
            "gap_pct": gap_pct,
            "ret_from_open_1030_pct": ret_from_open,
            "dist_vwap_1030_pct": dist_vwap,
            "entry_1031": entry,
            "stop": stop,
            "exit_1200": exit_price,
            "stopped": bool(stopped),
            "pnl": pnl,
        })

    trade_df = pd.DataFrame(trades)
    summary = {
        "trades": int(len(trade_df)),
        "net_pnl": float(trade_df["pnl"].sum()) if len(trade_df) else 0.0,
        "median_trade_pnl": float(trade_df["pnl"].median()) if len(trade_df) else 0.0,
        "win_rate": float((trade_df["pnl"] > 0).mean()) if len(trade_df) else 0.0,
        "best_trade": float(trade_df["pnl"].max()) if len(trade_df) else 0.0,
        "worst_trade": float(trade_df["pnl"].min()) if len(trade_df) else 0.0,
    }
    return summary, trade_df
