# S1: preserving an unpromoted challenger

This is a rewritten case study from preserved project records, not a rerun or a claim of live performance.

The baseline family was filtered asymmetric intraday mean reversion. S1 preserved a distinct 15:00 New York downside-drift short: sufficient decline from the session open, persistent below-session-VWAP acceptance, next-bar entry, an initial protective stop and a 15:30 time exit. `trading_lab/strategies/s1_downside_drift.py` preserves that reference class with explicit local input and without its historical audit driver.

The Stage-1 decision was **challenge weakened; no branch change**. The challenger survived as a credible comparison, but its sparse, fragile evidence did not earn promotion. The baseline remaining the reference also did not establish a robust winner or validated profitability.

The useful lesson is the separation of a contradiction from a promotion. A challenger can reveal a fairness or benchmark-validity concern and justify an audit without winning the eventual comparison. Freeze the alternatives and compare matched inputs, opportunity clocks, costs and execution assumptions before drawing a hierarchy conclusion.

The public tests exercise an artificial S1 signal/time exit and baseline execution semantics. They do not repeat Stage 1. The original market inputs, trade tables and operational evidence are excluded. Historical S1 clock/coverage limitations remain disclosed in `docs/LIMITATIONS.md`; the reference is not a corrected production simulator.

A future independent study would need lawful data, matched window/cost conventions, adequate independent evidence and a documented question. The historical comparison remains closed in this handoff. No retuning, reopening or strategy promotion occurred during preparation.
