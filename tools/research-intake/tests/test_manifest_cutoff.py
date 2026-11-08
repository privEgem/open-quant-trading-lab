from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from research_intake.catalog import append_event, connect, init_database
from research_intake.config import load_config
from research_intake.paths import create_runtime_directories, resolve_paths
from research_intake.pipeline import run_intake
from research_intake.recovery import recover_run


def _prepared(project_root: Path, tmp_path: Path, monkeypatch, body: bytes):
    monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(tmp_path))
    config = load_config(project_root)
    config.registry["policy"]["minimum_request_interval_seconds"] = 0
    paths = resolve_paths(project_root)
    create_runtime_directories(paths)
    init_database(paths.database)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=body, request=request)
    )
    return config, paths, httpx.Client(transport=transport)


def _publication_failure_run(
    project_root: Path,
    version2_xml: bytes,
    tmp_path: Path,
    monkeypatch,
):
    config, paths, client = _prepared(
        project_root, tmp_path, monkeypatch, version2_xml
    )

    def fail_publication(*args, **kwargs):
        raise OSError("simulated manifest publication failure")

    monkeypatch.setattr("research_intake.pipeline.publish_manifest", fail_publication)
    code, run_id = run_intake(
        config,
        paths,
        "backfill",
        365,
        20,
        client=client,
        clock=lambda: datetime(2026, 7, 12, tzinfo=timezone.utc),
        sleeper=lambda _: None,
    )
    assert code == 3
    return paths, run_id


def test_publication_failure_advances_provisional_cutoff(
    project_root: Path,
    version2_xml: bytes,
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths, run_id = _publication_failure_run(
        project_root, version2_xml, tmp_path, monkeypatch
    )
    assert not (paths.manifests / f"{run_id}.json").exists()
    with connect(paths.database) as connection:
        run = connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        event = connection.execute(
            "SELECT * FROM run_events WHERE run_id=? "
            "AND event_code='MANIFEST_PUBLICATION_FAILURE'",
            (run_id,),
        ).fetchone()
        assert run["catalog_committed"] == 1
        assert run["prepared_final_status"] == "PARTIAL_SUCCESS"
        assert run["manifest_publication_state"] == "MISSING"
        assert run["manifest_event_cutoff_state"] == "PROVISIONAL"
        assert run["manifest_event_cutoff_id"] == event["event_id"]


def test_recovery_reconstructs_partial_manifest_with_failure_event(
    project_root: Path,
    version2_xml: bytes,
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths, run_id = _publication_failure_run(
        project_root, version2_xml, tmp_path, monkeypatch
    )
    code, _ = recover_run(project_root, paths, run_id)
    assert code == 0
    manifest = json.loads((paths.manifests / f"{run_id}.json").read_text())
    assert manifest["final_status"] == "PARTIAL_SUCCESS"
    assert manifest["manifest_event_cutoff_state"] == "FROZEN"
    assert any(
        event["event_code"] == "MANIFEST_PUBLICATION_FAILURE"
        for event in manifest["errors"]
    )
    with connect(paths.database) as connection:
        run = connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        assert run["status"] == "PARTIAL_SUCCESS"
        assert run["manifest_event_cutoff_state"] == "FROZEN"
        with pytest.raises(sqlite3.IntegrityError, match="cutoff is immutable"):
            connection.execute(
                "UPDATE runs SET manifest_event_cutoff_id="
                "manifest_event_cutoff_id+1 WHERE run_id=?",
                (run_id,),
            )


def test_valid_manifest_cutoff_is_database_immutable(
    project_root: Path,
    version2_xml: bytes,
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, paths, client = _prepared(
        project_root, tmp_path, monkeypatch, version2_xml
    )
    code, run_id = run_intake(
        config,
        paths,
        "backfill",
        365,
        20,
        client=client,
        clock=lambda: datetime(2026, 7, 12, tzinfo=timezone.utc),
        sleeper=lambda _: None,
    )
    assert code == 0
    with connect(paths.database) as connection:
        run = connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        assert run["manifest_event_cutoff_state"] == "FROZEN"
        with pytest.raises(sqlite3.IntegrityError, match="cutoff is immutable"):
            connection.execute(
                "UPDATE runs SET manifest_event_cutoff_id="
                "manifest_event_cutoff_id+1 WHERE run_id=?",
                (run_id,),
            )


def test_later_finalization_event_stays_outside_manifest_and_enters_receipt(
    project_root: Path,
    version2_xml: bytes,
    tmp_path: Path,
    monkeypatch,
) -> None:
    config, paths, client = _prepared(
        project_root, tmp_path, monkeypatch, version2_xml
    )
    code, run_id = run_intake(
        config,
        paths,
        "backfill",
        365,
        20,
        client=client,
        clock=lambda: datetime(2026, 7, 12, tzinfo=timezone.utc),
        sleeper=lambda _: None,
    )
    assert code == 0
    manifest_path = paths.manifests / f"{run_id}.json"
    manifest_before = json.loads(manifest_path.read_text())
    with connect(paths.database) as connection:
        event_id = append_event(
            connection,
            run_id,
            "ERROR",
            "RUN_FINALIZATION_FAILURE",
            "FINALIZATION",
            {"message": "simulated post-publication failure"},
        )
        cutoff = connection.execute(
            "SELECT manifest_event_cutoff_id FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()[0]
    assert event_id > cutoff
    assert all(
        event["event_code"] != "RUN_FINALIZATION_FAILURE"
        for event in manifest_before["errors"]
    )
    code, receipt_path = recover_run(project_root, paths, run_id)
    assert code == 0
    assert json.loads(manifest_path.read_text()) == manifest_before
    receipt = json.loads(receipt_path.read_text())
    assert any(
        event["event_code"] == "RUN_FINALIZATION_FAILURE"
        for event in receipt["post_manifest_events"]
    )
