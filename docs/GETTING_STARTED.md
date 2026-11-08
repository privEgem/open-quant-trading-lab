# Getting started

These commands apply to v0.1.0. Included original/licensable material is distributed under MIT on the basis of the owner's good-faith rights attestation; third-party data and separately installed dependencies retain their own terms.

Requirements: Python 3.12+, standard `venv`/pip support, sufficient disk space for scientific Python dependencies. Validation was performed on Windows with Python 3.13; other supported versions/platforms were not separately tested.

On PowerShell, from the release directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]" -e "./tools/research-intake"
python -m trading_lab.demo
python tools/research-intake/examples/offline_demo.py
python scripts/verify_offline.py
```

If activation is unavailable, invoke `.venv\Scripts\python.exe` explicitly for every command. On a Unix shell, activate with `source .venv/bin/activate`. Installation is the only quickstart step that may download public packages. No credentials, market account or paid data are needed.

For an environment with dependencies and the build backend already installed, local installation can be done without index access:

```console
python -m pip install --no-index --no-deps --no-build-isolation -e . -e ./tools/research-intake
```

This command assumes dependencies are already present; it is not a promise that a blank offline computer can install packages it does not have. The tested versions are recorded in `release/requirements-tested.txt`. To prepare a disconnected machine, separately obtain compatible wheels lawfully and use a local wheelhouse.

The demos use temporary synthetic files/catalogs and clean them up. The backtest should report 1,950 bars, six synthetic trades, zero final open position and equal cash change versus reconciled trade PnL. The intake demo should report three catalog items and two malformed observations, with its expected pipeline code 3.

Intake can be installed independently from `tools/research-intake`; see its README. `python -m research_intake validate-config` uses bundled resources after wheel installation. For custom editable configuration, pass `--project-root` before the subcommand. Without a runtime override, state goes to `.research-intake-data` in the caller's current directory; `validate-config` creates no runtime directory.

The default verification script runs both suites in separate child processes with Python networking blocked. Direct toolkit tests use `python -m pytest tests`; independent intake tests use the same command from its package directory. Lint: `python -m ruff check trading_lab tests examples scripts`, then `python -m ruff check src tests examples` from the intake directory. Toolkit lint checks critical syntax/name errors; intake retains its broader original rule set.

Do not run historical research commands from another project checkout to make this example work. No external data file is expected or provided.
