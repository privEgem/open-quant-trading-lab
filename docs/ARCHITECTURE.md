# Architecture and extraction boundaries

`trading_lab` is an installable Python package. It contains deterministic features, one coherent baseline simulator, pure safety classifications, evidence-record helpers and preserved independent strategy references. No module loads credentials or imports a broker SDK. Data paths are explicit arguments.

The baseline simulator consumes a CSV, prepares features, resolves NYSE sessions and simulates signal/next-bar execution. `features.py` exposes three named VWAP contracts. `backtesting/metrics.py` contains only metrics and method metadata extracted from the old audit wrapper; no historical input identities, runner or fixed output location remains.

S1 and X1 references stay separate because their execution assumptions, sizing and return APIs differ. They are preserved teaching material. The public extraction removes old file launchers and implicit dataset locations; it does not unify their economics or retune their rules.

`risk_and_safety/reconciliation.py` uses standard-library values and normalized caller-supplied snapshots. It contains no acquisition, submission, cancellation or flattening implementation. The state machine is useful for reasoning about unknown observations, coverage and incident recovery; callers must not mistake it for an operational safety system.

`research_audit` validates records and hashes. Some code-identity functions explicitly invoke read-only local Git commands on a caller-supplied repository. They are not invoked on import or by default examples. Exact checkout-byte identity and canonical Git identity remain distinct.

`tools/research-intake` retains an independent `src/` package and tests. Resources are included in its wheel. Its CLI chooses bundled defaults or an explicit configuration root, while mutable state stays in the caller's data directory. SQLite context exit now closes handles after normal commit/rollback, fixing Windows cleanup of isolated temporary catalogs.

`research/` contains newly written summaries. No raw handover, private governance prompt, market dataset, operational log or historical packet archive is bundled. Generic handoff templates are rewritten rather than copied from private conversations.
