from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx

from research_intake.catalog import connect, init_database
from research_intake.config import load_config
from research_intake.paths import create_runtime_directories, resolve_paths
from research_intake.pipeline import run_intake
from research_intake.recovery import recover_run


def completed_run(project_root: Path, mixed_xml: bytes, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(tmp_path))
    config = load_config(project_root)
    config.registry["policy"]["minimum_request_interval_seconds"] = 0
    paths = resolve_paths(project_root)
    create_runtime_directories(paths)
    init_database(paths.database)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=mixed_xml, request=request))
    client = httpx.Client(transport=transport)
    code, run_id = run_intake(config, paths, "backfill", 30, 20, client=client, clock=lambda: datetime(2026, 7, 12, tzinfo=timezone.utc), sleeper=lambda _: None)
    assert code == 3
    return paths, run_id


def test_recovery_recreates_byte_identical_queues_and_manifest(project_root: Path, mixed_xml: bytes, tmp_path: Path, monkeypatch) -> None:
    paths, run_id = completed_run(project_root, mixed_xml, tmp_path, monkeypatch)
    markdown_path = paths.markdown / f"{run_id}.md"
    jsonl_path = paths.jsonl / f"{run_id}.jsonl"
    expected_markdown = markdown_path.read_bytes()
    expected_jsonl = jsonl_path.read_bytes()
    markdown_path.unlink()
    jsonl_path.unlink()
    (paths.manifests / f"{run_id}.json").unlink()
    with connect(paths.database) as connection:
        connection.execute(
            "UPDATE runs SET status='RUNNING',incremental_predecessor_eligible=0,"
            "incremental_predecessor_reason_code='NOT_FINALIZED' WHERE run_id=?",
            (run_id,),
        )
    code, receipt_path = recover_run(project_root, paths, run_id)
    assert code == 0
    assert markdown_path.read_bytes() == expected_markdown
    assert jsonl_path.read_bytes() == expected_jsonl
    receipt = json.loads(receipt_path.read_text())
    assert receipt["manifest_classification"] == "NEWLY_RECONSTRUCTED"
    assert receipt["database_reconciliation_performed"] is True
    with connect(paths.database) as connection:
        assert connection.execute(
            "SELECT incremental_predecessor_eligible FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()[0] == 1


def test_recovery_refuses_conflicting_manifest(project_root: Path, mixed_xml: bytes, tmp_path: Path, monkeypatch) -> None:
    paths, run_id = completed_run(project_root, mixed_xml, tmp_path, monkeypatch)
    manifest_path = paths.manifests / f"{run_id}.json"
    manifest_path.write_text("{}", encoding="utf-8")
    code, receipt_path = recover_run(project_root, paths, run_id)
    assert code == 1
    receipt = json.loads(receipt_path.read_text())
    assert receipt["manifest_classification"] == "PRESENT_CONFLICTING_OR_INVALID"
    assert manifest_path.read_text() == "{}"


def test_recovery_rejects_schema_valid_manifest_with_wrong_counts(
    project_root: Path, mixed_xml: bytes, tmp_path: Path, monkeypatch
) -> None:
    paths, run_id = completed_run(project_root, mixed_xml, tmp_path, monkeypatch)
    manifest_path = paths.manifests / f"{run_id}.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["counts"]["received"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    code, receipt_path = recover_run(project_root, paths, run_id)
    assert code == 1
    receipt = json.loads(receipt_path.read_text())
    assert receipt["manifest_classification"] == "PRESENT_CONFLICTING_OR_INVALID"
    assert "durable run counts" in receipt["errors"][0]["message"]


def test_recovery_rejects_tampered_output_path(
    project_root: Path, mixed_xml: bytes, tmp_path: Path, monkeypatch
) -> None:
    paths, run_id = completed_run(project_root, mixed_xml, tmp_path, monkeypatch)
    with connect(paths.database) as connection:
        connection.execute(
            "UPDATE runs SET manifest_path=? WHERE run_id=?",
            (str(tmp_path / "unexpected.json"), run_id),
        )
    code, receipt_path = recover_run(project_root, paths, run_id)
    assert code == 1
    receipt = json.loads(receipt_path.read_text())
    assert "expected output path" in receipt["errors"][0]["message"]
    assert not (tmp_path / "unexpected.json").exists()


def test_recovery_refuses_unsupported_renderer(project_root: Path, mixed_xml: bytes, tmp_path: Path, monkeypatch) -> None:
    paths, run_id = completed_run(project_root, mixed_xml, tmp_path, monkeypatch)
    with connect(paths.database) as connection:
        connection.execute("UPDATE runs SET queue_renderer_version='future-v2' WHERE run_id=?", (run_id,))
    code, receipt_path = recover_run(project_root, paths, run_id)
    assert code == 1
    assert "UNSUPPORTED_QUEUE_RENDERER" in receipt_path.read_text()


def test_missing_manifest_with_insufficient_durable_state_fails(
    project_root: Path, mixed_xml: bytes, tmp_path: Path, monkeypatch
) -> None:
    paths, run_id = completed_run(project_root, mixed_xml, tmp_path, monkeypatch)
    (paths.manifests / f"{run_id}.json").unlink()
    with connect(paths.database) as connection:
        connection.execute(
            "UPDATE runs SET prepared_final_status=NULL WHERE run_id=?", (run_id,)
        )
    code, receipt_path = recover_run(project_root, paths, run_id)
    assert code == 1
    receipt = json.loads(receipt_path.read_text())
    assert receipt["manifest_classification"] == "MISSING_NOT_RECONSTRUCTABLE"


def test_recovery_validates_failed_source_run_without_queue_rendering(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    malformed = (
        project_root / "tests" / "fixtures" / "arxiv" / "malformed.xml"
    ).read_bytes()
    monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(tmp_path))
    config = load_config(project_root)
    config.registry["policy"]["minimum_request_interval_seconds"] = 0
    paths = resolve_paths(project_root)
    create_runtime_directories(paths)
    init_database(paths.database)
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=malformed, request=request)
        )
    )
    code, run_id = run_intake(
        config,
        paths,
        "backfill",
        30,
        20,
        client=client,
        clock=lambda: datetime(2026, 7, 12, tzinfo=timezone.utc),
        sleeper=lambda _: None,
    )
    assert code == 1
    code, receipt_path = recover_run(project_root, paths, run_id)
    assert code == 0
    receipt = json.loads(receipt_path.read_text())
    assert "skip queues because catalog transaction did not commit" in receipt[
        "actions_attempted"
    ]
