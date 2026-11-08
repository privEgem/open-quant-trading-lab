"""Offline synthetic backtest using the extracted engine and unchanged signals."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from trading_lab.backtesting.mean_reversion import BacktesterMRCandidateFiltered
from trading_lab.synthetic import SEED, synthetic_market_data


def run_demo() -> dict:
    bars = synthetic_market_data()
    with tempfile.TemporaryDirectory(prefix="quant-demo-") as directory:
        csv = Path(directory) / "synthetic.csv"
        bars.to_csv(csv, index=False, float_format="%.10f")
        engine = BacktesterMRCandidateFiltered(file=str(csv), verbose=False)
        engine.run()
    equity = np.asarray(engine.equity_curve)
    drawdown = equity / np.maximum.accumulate(equity) - 1
    total_entry_commission = sum(t["shares"] * engine.commission_per_share for t in engine.trade_log)
    return {
        "data": "synthetic; software demonstration only; no empirical reproduction",
        "seed": SEED,
        "bars": len(bars),
        "trades": len(engine.trade_log),
        "starting_cash": engine.starting_cash,
        "ending_cash": round(engine.cash, 6),
        "cash_change": round(engine.cash - engine.starting_cash, 6),
        "trade_pnl_less_entry_commission": round(sum(engine.trades) - total_entry_commission, 6),
        "max_drawdown_pct": round(float(drawdown.min() * 100), 6),
        "slippage_bps_per_fill": engine.slippage_bps,
        "commission_per_share_per_fill": engine.commission_per_share,
        "feature_profile": engine.feature_profile.name,
        "execution_contract": engine.execution_session_contract,
        "open_position_at_end": engine.position,
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
