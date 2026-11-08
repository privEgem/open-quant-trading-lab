"""Mock Atom intake, SQLite provenance and human-review queue; no real requests."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

from research_intake.catalog import connect, init_database
from research_intake.config import load_config
from research_intake.paths import create_runtime_directories, resolve_paths
from research_intake.pipeline import run_intake


def run_demo() -> dict:
    root = Path(__file__).resolve().parents[1]
    body = (root / "tests/fixtures/arxiv/mixed.xml").read_bytes()
    config = load_config(root)
    config.registry["policy"]["minimum_request_interval_seconds"] = 0  # mocks only
    with tempfile.TemporaryDirectory(prefix="intake-demo-") as directory:  # noqa: SIM117
        with patch.dict(os.environ, {"RESEARCH_INTAKE_DATA_DIR": directory, "RESEARCH_INTAKE_CONTACT_EMAIL": ""}):
            paths = resolve_paths(root)
            create_runtime_directories(paths)
            init_database(paths.database)
            transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body, request=request))
            with httpx.Client(transport=transport) as client:
                code, run_id = run_intake(config, paths, "backfill", 30, 20, client=client,
                    clock=lambda: datetime(2026, 7, 12, tzinfo=timezone.utc), sleeper=lambda _: None)
            manifest = json.loads((paths.manifests / f"{run_id}.json").read_text())
            with connect(paths.database) as connection:
                items = connection.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            return {
                "transport": "httpx.MockTransport; fictional fixture records",
                "pipeline_exit_code": code,
                "expected_warning": "isolated malformed entry in each mocked query",
                "catalog_items": items,
                "malformed_observations": manifest["counts"]["malformed"],
                "incremental_predecessor_eligible": manifest["incremental_predecessor_eligible"],
                "review_queue_items": manifest["queues"]["jsonl"]["count"],
                "review_authority": "human; retrieval does not approve a strategy",
            }


if __name__ == "__main__":
    result = run_demo()
    assert result["pipeline_exit_code"] == 3
    assert result["catalog_items"] == 3
    print(json.dumps(result, indent=2, sort_keys=True))
