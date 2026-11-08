# API guide

The quickstarts are the supported starting point. Direct APIs are educational references with explicit inputs and deliberately different historical contracts.

```python
from trading_lab.features import generate_features, SESSION_REGULAR_NY_VWAP_V1
from trading_lab.synthetic import synthetic_market_data

bars = synthetic_market_data()
features = generate_features(bars, SESSION_REGULAR_NY_VWAP_V1)
```

Bars need `timestamp`, `open`, `high`, `low`, `close`, `volume`. Timestamps are parsed as UTC. Normalization sorts stably and keeps the last duplicate timestamp; nonpositive volume is excluded from VWAP accumulation. The baseline engine retains its historical positive-volume prefilter. Profiles must be named explicitly in feature-only use.

`BacktesterMRCandidateFiltered(file=..., verbose=False)` reads a supplied CSV and builds features. Call `run()` once on a fresh instance, then inspect `trade_log`, `equity_curve`, `equity_timestamps`, `cash` and `position`. Defaults include 1 bp directional slippage and zero per-share commission; those defaults are disclosed assumptions. The demonstration reconciles entry commissions separately from trade-log PnL.

| Module under `trading_lab.strategies` | Entry API | Result |
| --- | --- | --- |
| `s1_downside_drift` | `BacktesterChallengerLocked(file, verbose=False).run()` | Mutates instance logs/equity/cash |
| `x1.f` | `BacktesterX1GapUpFailedHoldContinuation(file, verbose=False).run()` | `(trades_dataframe, summary)` |
| `x1.g` | `run_backtest(data_path=file, cost_bps=1.0)` | `(summary, trades_dataframe)` |
| `x1.h` | `BacktesterX1FlatOpenOrderlyWeaknessDrift(file, verbose=False).run()` | Mapping with `summary`, `trade_log`, diagnostics |
| `x1.i` | `BacktesterX1SlightGapDownEarlyRecoveryDrift(file, verbose=False).run()` | Summary mapping containing `trade_log`, diagnostics |
| `x1.l` | `BacktesterX1NearFlatMorningShakeoutReclaimLong(file, verbose=False).run_backtest()` | Mapping with `summary`, `trade_log` |
| `x1.c` | `BacktesterX1OpeningDrivePullbackContinuation(file, verbose=False).run()` | Mapping with `summary`, `trade_log`, diagnostics |
| `x1.e` | `BacktesterX1OvernightDamageRepairReclaim(file, verbose=False).run()` | Mapping with `summary`, `trades`, diagnostics |

G/L additionally require a `vwap` column; no field is silently inferred by those classes. Synthetic scenario tests show a session close×volume construction. G is a one-share model, L uses fixed notional; avoid comparing raw PnL across APIs. Instantiate fresh objects for each example. All default demo inputs are local artificial files; do not pass untrusted remote locations to pandas readers.

`risk_and_safety.reconciliation` exposes normalized `PositionView`, `OrderView`, `BrokerSnapshot`, `reconcile_snapshot`, incident transitions and state serialization. Construct snapshots from fake values. Query states distinguish `OK`, `UNKNOWN`, `INCOMPLETE`; completeness flags and price shapes matter. `strategy_symbol` defaults to SPY, so tests using another fictional symbol must set it explicitly. No function submits an order.

`research_audit.experiment_record` builds, validates, persists and verifies bounded evidence records. Use relative display paths for shareable records. `evidence_identity` supplies canonical JSON hashes and explicit Git-based code descriptors. Do not call Git descriptors on an unrelated or private repository and publish the result automatically; metadata can itself be sensitive.

Research intake's CLI and human-review workflow are documented in its independent README. Its storage is a local application, not a secure shared service.
