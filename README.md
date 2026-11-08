# Open Quant Trading Lab

**v0.1.0 — MIT-licensed source release for offline research and education.**

Public display identity: `privEgem`. Repository name: `privEgem/open-quant-trading-lab`. This name identifies the project; it does not assert repository availability.

Python backtesting, quantitative-research workflows, risk-control engineering and honest research case studies. Educational and experimental: **no validated profitable strategy**, **not production-ready**, **no investment advice**.

This release preserves useful engineering and failed research without presenting a working trading business. It contains no broker connector, credentials, market dataset, deployment launcher or scheduler. Importing the toolkit does not trade, download data or create runtime state.

## Run the two offline quickstarts

Use Python 3.12 or newer. A virtual environment is recommended. From this directory, using that environment's Python:

```console
python -m pip install -e ".[dev]" -e "./tools/research-intake"
python -m trading_lab.demo
python tools/research-intake/examples/offline_demo.py
python scripts/verify_offline.py
```

Dependency installation can contact the package index. The examples and verification command run offline with generated data, fictional fixtures and mock transports. See [Getting started](docs/GETTING_STARTED.md) for Windows setup and installation without network access once dependencies are available.

The first example generates 1,950 artificial minute bars with seed 1729, uses the extracted signal/feature/engine path, charges directional slippage, and checks that no simulated position remains. The reviewed run produced six synthetic trades. These numbers demonstrate software behavior; they reproduce no historical empirical result.

The second example parses fictional Atom responses, builds a temporary SQLite catalog, preserves response provenance and produces human-review queues. Its expected pipeline warning code is 3 because the fixture intentionally contains a malformed entry. The example itself exits successfully after checking that outcome.

## What is useful here

- Baseline mean-reversion simulator with next-bar entries, entry gates separate from exit management, conservative same-bar stop priority, calendar session closure and timestamped equity.
- Explicit VWAP profiles: whole-input cumulative, legacy fetched-window and NYSE regular-session reset. Their labels describe calculation contracts, not a trading recommendation.
- Preserved S1 downside-drift reference; X1 F/G/H/I/L relative survivors and C/E comparators. Different APIs and known modeling limitations remain visible.
- Pure reconciliation classification and safety-incident state transitions, tested with fake snapshots. This is an offline design reference, not a broker or production safety certificate.
- Experiment records, artifact hashes and canonical code-identity helpers. These record evidence; they do not run a scientific study or establish alpha.
- Independently installable [research intake](tools/research-intake/README.md): bounded arXiv retrieval, versioned SQLite catalog, manifests, recovery and explicit human review. Network retrieval requires an explicit `run` command.

Read [architecture](docs/ARCHITECTURE.md), [API guide](docs/API.md), [limitations](docs/LIMITATIONS.md) and [reproducibility](docs/REPRODUCIBILITY.md) before extending the examples.

## Research outcome, without promotion

- [S1](research/stage1.md): challenge weakened; no branch change. The challenger was unpromoted; the baseline was not validated by surviving the comparison.
- [X1](research/x1.md): F/G/H/I/L are frozen relative survivors; E/C preserved comparators; K1 parked. No proven edge.
- [X2](research/x2.md): paper review improved discipline but produced no serious candidate; dormant.
- [Bauer–Chernov Stage 0](research/bauer-chernov-stage0.md): immutable primary classification `unfavorable_continuation_evidence`; the historical B.4 18-cell calibration remains unfinished.

No historical study was rerun to prepare this release. Restricted empirical inputs are excluded. Read [methodology and negative results](research/methodology-and-negative-results.md) for what a future independent study would need.

## License and handoff

[LICENSE](LICENSE) contains the owner-approved standard MIT text, copyright 2026 privEgem, for original material the owner has authority to license. The owner attests, to the best of their knowledge, that they have the rights to publish and MIT-license the included material, with no known third-party, contributor, employer or contractual restrictions. This is a good-faith owner attestation, not an independent legal audit. Dependencies are installed separately. No market-data or paper redistribution right is implied. See [source and rights](docs/SOURCE_AND_RIGHTS.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

This is intended as a limited-maintenance research/toolkit handoff. Future development is voluntary; no hosted service, trading support, ongoing strategy improvement or response time is promised. [Contributing](CONTRIBUTING.md), [security](SECURITY.md), [AI handoff](docs/AI_HANDOFF.md) and reusable [review templates](docs/templates/INDEPENDENT_REVIEW.md) explain how to continue independently.

The source archive excludes private Git history. File hashes establish byte identity; they do not establish legal ownership, scientific validity or publication authorization. [Validation summary](release/VALIDATION_SUMMARY.md) records checks and their limits.
