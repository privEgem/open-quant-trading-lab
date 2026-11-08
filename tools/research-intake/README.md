# Research intake

An independent Python 3.12+ arXiv-intake package with a normalized SQLite catalog, source/run manifests, version tracking, conservative recovery and an authoritative human-review workflow. This package does not validate strategies or download paper PDFs. This is part of the v0.1.0 source release. Included original/licensable material uses MIT on the basis of the owner's good-faith rights attestation; this does not license retrieved metadata, abstracts or papers. See [source and rights](../../docs/SOURCE_AND_RIGHTS.md).

From this directory, using a virtual environment:

```console
python -m pip install -e ".[dev]"
python -m research_intake validate-config
python examples/offline_demo.py
python -m pytest tests
```

Installation can download dependencies. Configuration validation and the mock example require no account, contact address or network. The mock example expects three catalog items, two malformed observations and pipeline code 3; that warning is intentional. The script checks it and exits 0. Fixture names, abstracts and article IDs are fictional stand-ins.

Configuration files are available in `config/` for editing and bundled read-only in the wheel for independent installations. To use edited configuration, pass `python -m research_intake --project-root . validate-config`. The global option comes before the command. `RESEARCH_INTAKE_DATA_DIR` overrides the runtime location; otherwise it is `.research-intake-data` in the caller's current directory. Never point it at an existing private catalog for a public demo.

## Explicit online retrieval

The following is documentation of an optional network command, **not part of the quickstart or release validation**. Review the [current arXiv API terms](https://info.arxiv.org/help/api/tou.html) and [manual](https://info.arxiv.org/help/api/user-manual.html) first. Configure an appropriate contact address yourself through `RESEARCH_INTAKE_CONTACT_EMAIL` if desired; none is embedded.

```console
python -m research_intake --project-root . init-db
python -m research_intake --project-root . run --mode backfill --days 7 --max-items 20
```

`init-db` creates local state. `run` deliberately contacts the official HTTPS Atom endpoint. It uses bounded pages/items/runtime, retries and sequential request spacing. The included 3-second minimum is per process; users remain responsible for aggregate request limits across their machines. Incremental mode requires an eligible completed predecessor. A failed/incomplete run is not silently equivalent to a complete retrieval.

Retrieval preserves metadata/abstract response provenance and produces review queues. Descriptive metadata and paper/content rights are distinct. Do not redistribute stored responses or abstracts by assuming that this package's selected MIT code license covers them. No actual response is included in this candidate and no service endorsement is claimed.

## Human review and recovery

Review queue files before deciding what deserves further work. A batch must match `schemas/human_review_batch.schema.json`; use `record-reviews --input batch.json --dry-run` before applying the local write, then `record-reviews --input batch.json` and `export-reviews`. Decisions preserve provenance and version identity. Automated relevance scores do not grant a research or strategy promotion.

`recover-run --run-id ...` attempts bounded local repair of an existing run's publication state; it is not a replacement for source retrieval or an instruction to repeat a scientific experiment. Read the corresponding schemas and tests before using recovery on valuable data.

No Windows Scheduled Task, pilot launcher, background job or deployment component is included. The socket-blocked monorepo verifier is the preferred release check. Default package tests also reject real connections. SQLite context exit releases handles after commit/rollback so temporary catalogs can be removed on Windows.
