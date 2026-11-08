from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from research_intake.catalog import (
    connect,
    create_run,
    ingest_batch,
    init_database,
)
from research_intake.cli import main
from research_intake.evidence import sha256_file
from research_intake.human_review import (
    ReviewValidationError,
    canonical_json,
    decision_id_for,
    export_reviews,
    load_review_batch,
    record_review_batch,
)
from research_intake.operation_lock import OperationLock

REVIEWED_AT = "2026-07-22T10:00:00Z"
CREATED_AT = "2026-07-22T10:00:01Z"


@dataclass(frozen=True)
class Seed:
    root: Path
    database: Path
    run_id: str
    items: list[dict[str, Any]]
    queue_path: Path
    queue_hash: str


def normalized_item(
    item_id: str,
    run_id: str,
    *,
    version: str = "v1",
    title: str = "A deterministic research paper",
) -> dict[str, Any]:
    suffix = item_id[-8:]
    return {
        "identity": {
            "schema_version": 1,
            "item_id": item_id,
            "record_type": "research_paper",
            "primary_key_type": "arxiv",
            "primary_key_value": f"2607.{suffix}",
            "base_arxiv_id": f"2607.{suffix}",
            "current_arxiv_version": version,
            "doi": None,
            "doi_status": "not_provided",
        },
        "bibliographic": {
            "title": title,
            "authors": ["Ada Researcher"],
            "abstract": "A sufficiently detailed abstract for deterministic testing.",
            "language": "en",
            "primary_category": "q-fin.TR",
            "categories": ["q-fin.TR"],
            "submitted_at": "2026-07-20T00:00:00Z",
            "source_updated_at": "2026-07-20T00:00:00Z",
            "journal_reference": None,
            "comments": None,
            "license": None,
        },
        "links": {
            "source_record_url": f"https://arxiv.org/abs/2607.{suffix}{version}",
            "pdf_url": None,
            "doi_url": None,
        },
        "provenance": {
            "primary_source": "arxiv",
            "source_record_id": f"https://arxiv.org/abs/2607.{suffix}{version}",
            "discovery_query_ids": ["q"],
            "retrieved_at": "2026-07-21T00:00:00Z",
            "run_id": run_id,
            "raw_artifact_path": f"raw/{suffix}.xml",
            "raw_artifact_sha256": "1" * 64,
            "normalizer_version": "1",
        },
        "relevance": {
            "eligible": True,
            "score": 80,
            "priority": "HIGH",
            "matched_rule_ids": ["category"],
            "penalty_rule_ids": [],
            "contributions": [{"rule_id": "category", "points": 80}],
            "explanation": "Eligible deterministic fixture.",
        },
        "lifecycle": {
            "first_seen_at": "2026-07-21T00:00:00Z",
            "last_seen_at": "2026-07-21T00:00:00Z",
            "first_seen_run_id": run_id,
            "last_seen_run_id": run_id,
            "latest_source_version": True,
        },
        "review": {
            "status": "NEW",
            "priority": "HIGH",
            "reviewer": None,
            "reviewed_at": None,
            "tags": [],
            "notes": None,
        },
    }


def add_run(connection: sqlite3.Connection, database: Path, run_id: str) -> None:
    create_run(
        connection,
        {
            "run_id": run_id,
            "run_type": "backfill",
            "start_timestamp_utc": "2026-07-21T00:00:00Z",
            "interval_start_utc": "2026-07-20T00:00:00Z",
            "interval_end_utc": "2026-07-21T00:00:00Z",
            "discovery_time_basis": "SUBMITTED_DATE",
            "application_version": "0.1.0",
            "git_commit": None,
            "python_version": "3.12",
            "registry_hash": "a" * 64,
            "topics_hash": "b" * 64,
            "previous_eligible_run_id": None,
            "overlap_seconds": 1,
            "manifest_path": str(database.parent / f"{run_id}.json"),
        },
        [{"source_id": "arxiv", "query_id": "q"}],
    )


def add_item(
    connection: sqlite3.Connection,
    item: dict[str, Any],
    run_id: str,
    position: int,
) -> None:
    identity = item["identity"]
    bibliography = item["bibliographic"]
    links = item["links"]
    relevance = item["relevance"]
    lifecycle = item["lifecycle"]
    review = item["review"]
    connection.execute(
        """INSERT INTO items(
        item_id,primary_source,primary_key_type,primary_key_value,base_arxiv_id,
        current_version,doi,title,abstract,authors_json,language,primary_category,
        categories_json,submitted_at,source_updated_at,journal_reference,comments,
        license,source_record_url,pdf_url,doi_url,relevance_score,relevance_priority,
        eligible,matched_rule_ids_json,penalty_rule_ids_json,review_status,
        review_priority,reviewer,reviewed_at,tags_json,notes,first_seen_at,last_seen_at,
        first_seen_run_id,last_seen_run_id,meaningful_fingerprint,canonical_json
        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            identity["item_id"],
            "arxiv",
            "arxiv",
            identity["primary_key_value"],
            identity["base_arxiv_id"],
            identity["current_arxiv_version"],
            None,
            bibliography["title"],
            bibliography["abstract"],
            canonical_json(bibliography["authors"]),
            bibliography["language"],
            bibliography["primary_category"],
            canonical_json(bibliography["categories"]),
            bibliography["submitted_at"],
            bibliography["source_updated_at"],
            None,
            None,
            None,
            links["source_record_url"],
            None,
            None,
            relevance["score"],
            relevance["priority"],
            1,
            canonical_json(relevance["matched_rule_ids"]),
            "[]",
            review["status"],
            review["priority"],
            review["reviewer"],
            review["reviewed_at"],
            canonical_json(review["tags"]),
            review["notes"],
            lifecycle["first_seen_at"],
            lifecycle["last_seen_at"],
            run_id,
            run_id,
            "f" * 64,
            canonical_json(item),
        ),
    )
    connection.execute(
        "INSERT INTO item_revisions(item_id,arxiv_version,first_observed_run_id,"
        "raw_artifact_path,raw_artifact_sha256,source_fingerprint,normalized_json) "
        "VALUES(?,?,?,?,?,?,?)",
        (
            identity["item_id"],
            identity["current_arxiv_version"],
            run_id,
            item["provenance"]["raw_artifact_path"],
            item["provenance"]["raw_artifact_sha256"],
            "e" * 64,
            canonical_json(item),
        ),
    )
    connection.execute(
        "INSERT INTO discoveries(run_id,item_id,source_id,query_id,source_record_id,"
        "disposition,observed_version,raw_artifact_path,raw_artifact_sha256,"
        "observation_position,normalized_snapshot_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            run_id,
            identity["item_id"],
            "arxiv",
            "q",
            item["provenance"]["source_record_id"],
            "NEW",
            identity["current_arxiv_version"],
            item["provenance"]["raw_artifact_path"],
            item["provenance"]["raw_artifact_sha256"],
            position,
            canonical_json(item),
        ),
    )


def write_queue(
    root: Path,
    run_id: str,
    items: list[dict[str, Any]],
) -> tuple[Path, str]:
    path = root / "review" / "jsonl" / f"{run_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "queue_context": {
                "run_id": run_id,
                "queue_render_timestamp_utc": "2026-07-21T00:01:00Z",
                "queue_renderer_version": "markdown-jsonl-v1",
            },
            "item": item,
        }
        for item in items
    ]
    content = (
        "\n".join(canonical_json(record) for record in records) + "\n"
        if records
        else ""
    ).encode("utf-8")
    path.write_bytes(content)
    return path, hashlib.sha256(content).hexdigest()


def seed_catalog(tmp_path: Path, item_count: int = 1) -> Seed:
    root = tmp_path / "runtime"
    database = root / "state" / "catalog.sqlite3"
    run_id = "run-source"
    init_database(database)
    items = [
        normalized_item(
            "ri_" + character * 64,
            run_id,
            title=f"Paper {index + 1}",
        )
        for index, character in enumerate("abc"[:item_count])
    ]
    with connect(database) as connection:
        add_run(connection, database, run_id)
        for position, item in enumerate(items):
            add_item(connection, item, run_id, position)
    queue_path, queue_hash = write_queue(root, run_id, items)
    with connect(database) as connection:
        connection.execute(
            "UPDATE runs SET jsonl_path=?,jsonl_status='CREATED',jsonl_count=?,"
            "jsonl_size=?,jsonl_sha256=? WHERE run_id=?",
            (
                queue_path.as_posix(),
                len(items),
                queue_path.stat().st_size,
                queue_hash,
                run_id,
            ),
        )
    return Seed(root, database, run_id, items, queue_path, queue_hash)


def decision(
    seed: Seed,
    item_index: int = 0,
    *,
    status: str = "KEEP_FOR_DEEP_REVIEW",
    basis: str = "ABSTRACT_TRIAGE",
    confidence: str = "HIGH",
    expected: str = "NEW",
    reviewed_at: str = REVIEWED_AT,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "item_id": seed.items[item_index]["identity"]["item_id"],
        "source_run_id": seed.run_id,
        "expected_current_status": expected,
        "status": status,
        "review_basis": basis,
        "confidence": confidence,
        "rationale": "The evidence merits a documented follow-up review.",
        "next_action": "Read the complete paper and verify the methodology.",
        "tags": tags or [],
        "reviewed_at": reviewed_at,
        "source_queue_sha256": seed.queue_hash,
    }


def batch(seed: Seed, decisions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema": "human_review_batch",
        "version": 1,
        "batch_id": "fictional-batch-001",
        "reviewer": "Fictional Reviewer",
        "decisions": decisions or [decision(seed)],
    }


def record(
    seed: Seed,
    value: dict[str, Any],
    project_root: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    return record_review_batch(
        seed.database,
        seed.root,
        value,
        project_root / "schemas" / "normalized_item.schema.json",
        dry_run=dry_run,
        now_fn=lambda: CREATED_AT,
    )


def file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_database_initialization_adds_ledger_and_append_only_triggers(
    tmp_path: Path,
) -> None:
    database = tmp_path / "catalog.sqlite3"
    init_database(database)
    with connect(database) as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='review_decisions'"
        ).fetchone()
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(review_decisions)")
        }
        triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            )
        }
    assert {
        "review_decisions_no_update",
        "review_decisions_no_delete",
    } <= triggers
    assert {
        "decision_id",
        "batch_id",
        "item_id",
        "source_run_id",
        "item_version",
        "decision_timestamp_utc",
        "reviewer",
        "review_basis",
        "status",
        "confidence",
        "rationale",
        "next_action",
        "tags_json",
        "source_queue_path",
        "source_queue_sha256",
        "previous_status",
        "previous_reviewer",
        "previous_reviewed_at",
        "previous_tags_json",
        "previous_notes",
        "created_at_utc",
    } == columns


def test_existing_schema_version_one_catalog_gains_review_table(
    tmp_path: Path,
) -> None:
    database = tmp_path / "catalog.sqlite3"
    init_database(database)
    with connect(database) as connection:
        connection.execute("DROP TABLE review_decisions")
        assert connection.execute(
            "SELECT value FROM schema_metadata WHERE key='schema_version'"
        ).fetchone()[0] == "1"
    init_database(database)
    with connect(database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM review_decisions"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT value FROM schema_metadata WHERE key='schema_version'"
        ).fetchone()[0] == "1"


def test_dry_run_on_existing_version_one_catalog_does_not_add_table(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    with connect(seed.database) as connection:
        connection.execute("DROP TABLE review_decisions")
    before = seed.database.read_bytes()
    assert record(seed, batch(seed), project_root, dry_run=True)["would_record"] == 1
    assert seed.database.read_bytes() == before
    with connect(seed.database) as connection:
        assert not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='review_decisions'"
        ).fetchone()


def test_first_cli_record_adds_review_table_under_operation_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed = seed_catalog(tmp_path)
    with connect(seed.database) as connection:
        connection.execute("DROP TABLE review_decisions")
    monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(seed.root))
    input_path = tmp_path / "fictional-review-batch.json"
    input_path.write_text(json.dumps(batch(seed)), encoding="utf-8")
    assert main(["record-reviews", "--input", str(input_path)]) == 0
    assert json.loads(capsys.readouterr().out)["recorded"] == 1
    with connect(seed.database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM review_decisions"
        ).fetchone()[0] == 1


@pytest.mark.parametrize("basis", ["ABSTRACT_TRIAGE", "FULL_TEXT_REVIEW"])
def test_valid_decision_records_for_each_review_basis(
    tmp_path: Path, project_root: Path, basis: str
) -> None:
    seed = seed_catalog(tmp_path)
    value = batch(seed, [decision(seed, basis=basis)])
    summary = record(seed, value, project_root)
    assert summary["recorded"] == 1
    with connect(seed.database) as connection:
        row = connection.execute("SELECT * FROM review_decisions").fetchone()
    assert row["review_basis"] == basis
    assert row["status"] == "KEEP_FOR_DEEP_REVIEW"
    assert row["previous_status"] == "NEW"
    assert row["source_queue_path"] == seed.queue_path.as_posix()


def test_current_columns_and_canonical_json_remain_synchronized(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    value = batch(
        seed,
        [
            decision(
                seed,
                status="EDUCATIONAL_ONLY",
                confidence="MEDIUM",
                tags=["methodology-watchlist", "reviewed"],
            )
        ],
    )
    record(seed, value, project_root)
    with connect(seed.database) as connection:
        row = connection.execute("SELECT * FROM items").fetchone()
    canonical = json.loads(row["canonical_json"])
    assert row["review_status"] == canonical["review"]["status"] == "EDUCATIONAL_ONLY"
    assert row["reviewer"] == canonical["review"]["reviewer"] == value["reviewer"]
    assert row["reviewed_at"] == canonical["review"]["reviewed_at"] == REVIEWED_AT
    assert json.loads(row["tags_json"]) == canonical["review"]["tags"]
    assert row["notes"] == canonical["review"]["notes"]
    assert row["review_priority"] == canonical["review"]["priority"] == "HIGH"


def test_future_ingestion_preserves_review_and_history_across_version(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    value = batch(seed)
    record(seed, value, project_root)
    run_id = "run-later-version"
    raw = seed.root / "raw" / "later.xml"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"later immutable source bytes")
    updated = normalized_item(
        seed.items[0]["identity"]["item_id"],
        run_id,
        version="v2",
        title="Paper 1 revised",
    )
    updated["identity"]["primary_key_value"] = seed.items[0]["identity"][
        "primary_key_value"
    ]
    updated["identity"]["base_arxiv_id"] = seed.items[0]["identity"]["base_arxiv_id"]
    updated["provenance"]["raw_artifact_path"] = raw.as_posix()
    updated["provenance"]["raw_artifact_sha256"] = sha256_file(raw)
    updated["lifecycle"]["first_seen_run_id"] = seed.run_id
    updated["lifecycle"]["last_seen_run_id"] = run_id
    with connect(seed.database) as connection:
        add_run(connection, seed.database, run_id)
    with connect(seed.database) as connection:
        result = ingest_batch(
            connection,
            run_id,
            [
                {
                    "item": updated,
                    "query_id": "q",
                    "raw_path": raw.as_posix(),
                    "raw_hash": sha256_file(raw),
                    "source_fingerprint": "d" * 64,
                }
            ],
            "2026-07-23T00:00:00Z",
            "markdown-jsonl-v1",
            (seed.root / "review" / "markdown" / f"{run_id}.md").as_posix(),
            (seed.root / "review" / "jsonl" / f"{run_id}.jsonl").as_posix(),
            project_root / "schemas" / "normalized_item.schema.json",
        )
        item_row = connection.execute(
            "SELECT current_version,review_status,reviewer,reviewed_at,tags_json,"
            "notes,canonical_json FROM items"
        ).fetchone()
        history = connection.execute(
            "SELECT item_version,status FROM review_decisions"
        ).fetchall()
    assert result["updated"] == 1
    assert item_row["current_version"] == "v2"
    assert item_row["review_status"] == "KEEP_FOR_DEEP_REVIEW"
    assert json.loads(item_row["canonical_json"])["review"]["status"] == (
        "KEEP_FOR_DEEP_REVIEW"
    )
    assert [(row["item_version"], row["status"]) for row in history] == [
        ("v1", "KEEP_FOR_DEEP_REVIEW")
    ]


def test_same_exact_decision_is_idempotent(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    value = batch(seed)
    first = record(seed, value, project_root)
    second = record(seed, value, project_root)
    assert first["recorded"] == 1
    assert second["recorded"] == 0
    assert second["already_recorded"] == 1
    assert second["projected"][0]["outcome"] == "already_recorded"
    with connect(seed.database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM review_decisions"
        ).fetchone()[0] == 1


def test_substantive_replay_under_different_batch_is_idempotent_and_preserves_origin(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    original = batch(seed)
    first = record(seed, original, project_root)
    replay = json.loads(json.dumps(original))
    replay["batch_id"] = "fictional-batch-002"
    replay["decisions"][0]["expected_current_status"] = "REJECTED"
    second = record(seed, replay, project_root)
    assert second["recorded"] == 0
    assert second["already_recorded"] == 1
    assert second["projected"][0]["decision_id"] == first["projected"][0]["decision_id"]
    with connect(seed.database) as connection:
        rows = connection.execute(
            "SELECT decision_id,batch_id FROM review_decisions"
        ).fetchall()
    assert [(row["decision_id"], row["batch_id"]) for row in rows] == [
        (first["projected"][0]["decision_id"], original["batch_id"])
    ]


def test_historical_replay_after_later_decision_is_already_recorded(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    original = batch(seed)
    first = record(seed, original, project_root)
    later = batch(
        seed,
        [
            decision(
                seed,
                status="PARKED",
                expected="KEEP_FOR_DEEP_REVIEW",
                reviewed_at="2026-07-22T11:00:00Z",
            )
        ],
    )
    later["batch_id"] = "fictional-batch-002"
    record(seed, later, project_root)
    replay = json.loads(json.dumps(original))
    replay["batch_id"] = "fictional-batch-003"
    replay["decisions"][0]["expected_current_status"] = "DUPLICATE"
    result = record(seed, replay, project_root)
    assert result["already_recorded"] == 1
    assert result["projected"][0]["decision_id"] == first["projected"][0]["decision_id"]
    with connect(seed.database) as connection:
        history = connection.execute(
            "SELECT decision_id,batch_id FROM review_decisions "
            "ORDER BY decision_timestamp_utc,decision_id"
        ).fetchall()
        item = connection.execute(
            "SELECT review_status,reviewed_at FROM items"
        ).fetchone()
    assert len(history) == 2
    assert sum(
        row["decision_id"] == first["projected"][0]["decision_id"] for row in history
    ) == 1
    assert item["review_status"] == "PARKED"
    assert item["reviewed_at"] == "2026-07-22T11:00:00Z"


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "rationale",
            "A changed substantive rationale records another immutable event.",
        ),
        ("status", "PARKED"),
    ],
)
def test_changed_substantive_field_creates_a_different_decision(
    tmp_path: Path,
    project_root: Path,
    field: str,
    replacement: str,
) -> None:
    seed = seed_catalog(tmp_path)
    first = record(seed, batch(seed), project_root)
    changed = batch(
        seed,
        [decision(seed, expected="KEEP_FOR_DEEP_REVIEW")],
    )
    changed["batch_id"] = "fictional-batch-002"
    changed["decisions"][0][field] = replacement
    second = record(seed, changed, project_root)
    assert second["recorded"] == 1
    assert second["projected"][0]["decision_id"] != first["projected"][0]["decision_id"]
    with connect(seed.database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM review_decisions"
        ).fetchone()[0] == 2


def test_conflicting_duplicate_batch_decision_is_rejected(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    original = batch(seed)
    record(seed, original, project_root)
    conflict = batch(seed)
    conflict["decisions"][0]["rationale"] = (
        "A materially different rationale must not replace the first record."
    )
    with pytest.raises(ReviewValidationError, match="conflicting decision"):
        record(seed, conflict, project_root)
    with connect(seed.database) as connection:
        assert connection.execute(
            "SELECT rationale FROM review_decisions"
        ).fetchone()[0] == original["decisions"][0]["rationale"]


def test_duplicate_item_within_input_batch_is_rejected(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    duplicate = decision(seed)
    with pytest.raises(ReviewValidationError, match="duplicate item"):
        record(seed, batch(seed, [decision(seed), duplicate]), project_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown-item", "unknown item"),
        ("unknown-run", "unknown source run"),
        ("no-discovery", "eligible NEW or UPDATED"),
        ("wrong-supplied-hash", "does not match the catalog"),
        ("modified-queue", "bytes do not match"),
        ("absent-from-queue", "absent from"),
    ],
)
def test_catalog_and_queue_provenance_failures_are_rejected(
    tmp_path: Path,
    project_root: Path,
    mutation: str,
    message: str,
) -> None:
    seed = seed_catalog(tmp_path)
    value = batch(seed)
    if mutation == "unknown-item":
        value["decisions"][0]["item_id"] = "ri_" + "f" * 64
    elif mutation == "unknown-run":
        value["decisions"][0]["source_run_id"] = "missing-run"
    elif mutation == "no-discovery":
        with connect(seed.database) as connection:
            connection.execute("DELETE FROM discoveries")
    elif mutation == "wrong-supplied-hash":
        value["decisions"][0]["source_queue_sha256"] = "0" * 64
    elif mutation == "modified-queue":
        seed.queue_path.write_bytes(seed.queue_path.read_bytes() + b" ")
    elif mutation == "absent-from-queue":
        _, empty_hash = write_queue(seed.root, seed.run_id, [])
        value["decisions"][0]["source_queue_sha256"] = empty_hash
        with connect(seed.database) as connection:
            connection.execute(
                "UPDATE runs SET jsonl_size=0,jsonl_sha256=? WHERE run_id=?",
                (empty_hash, seed.run_id),
            )
    with pytest.raises(ReviewValidationError, match=message):
        record(seed, value, project_root)
    with connect(seed.database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM review_decisions"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT review_status FROM items"
        ).fetchone()[0] == "NEW"


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("hash-mismatch", "bytes do not match"),
        ("invalid-utf8", "cannot read source run"),
        ("invalid-jsonl", "line 1 is invalid"),
    ],
)
def test_queue_byte_validation_failure_never_mutates_history_or_projection(
    tmp_path: Path,
    project_root: Path,
    failure: str,
    message: str,
) -> None:
    seed = seed_catalog(tmp_path)
    value = batch(seed)
    if failure == "hash-mismatch":
        seed.queue_path.write_bytes(seed.queue_path.read_bytes() + b" ")
    else:
        content = b"\xff" if failure == "invalid-utf8" else b"{not-json}\n"
        queue_hash = hashlib.sha256(content).hexdigest()
        seed.queue_path.write_bytes(content)
        value["decisions"][0]["source_queue_sha256"] = queue_hash
        with connect(seed.database) as connection:
            connection.execute(
                "UPDATE runs SET jsonl_size=?,jsonl_sha256=? WHERE run_id=?",
                (len(content), queue_hash, seed.run_id),
            )
    with connect(seed.database) as connection:
        before_item = dict(connection.execute("SELECT * FROM items").fetchone())
        before_history = connection.execute(
            "SELECT COUNT(*) FROM review_decisions"
        ).fetchone()[0]
    with pytest.raises(ReviewValidationError, match=message):
        record(seed, value, project_root)
    with connect(seed.database) as connection:
        after_item = dict(connection.execute("SELECT * FROM items").fetchone())
        after_history = connection.execute(
            "SELECT COUNT(*) FROM review_decisions"
        ).fetchone()[0]
    assert after_item == before_item
    assert after_history == before_history == 0
    assert not (seed.root / "human-review").exists()


def test_queue_hash_and_parser_share_one_captured_byte_sequence(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = seed_catalog(tmp_path)
    original_read_bytes = Path.read_bytes
    queue_reads = 0

    def capture_then_replace(path: Path) -> bytes:
        nonlocal queue_reads
        captured = original_read_bytes(path)
        if path.resolve() == seed.queue_path.resolve():
            queue_reads += 1
            path.write_bytes(b"\xff")
        return captured

    monkeypatch.setattr(Path, "read_bytes", capture_then_replace)
    result = record(seed, batch(seed), project_root)
    assert result["recorded"] == 1
    assert queue_reads == 1
    with connect(seed.database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM review_decisions"
        ).fetchone()[0] == 1


def test_expected_status_mismatch_is_rejected(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    value = batch(seed, [decision(seed, expected="REVIEWING")])
    with pytest.raises(ReviewValidationError, match="expected current status mismatch"):
        record(seed, value, project_root)


def test_backdated_review_is_rejected(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    record(seed, batch(seed), project_root)
    later = batch(
        seed,
        [
            decision(
                seed,
                status="PARKED",
                expected="KEEP_FOR_DEEP_REVIEW",
                reviewed_at="2026-07-22T09:59:59Z",
            )
        ],
    )
    later["batch_id"] = "fictional-batch-002"
    with pytest.raises(ReviewValidationError, match="earlier than"):
        record(seed, later, project_root)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("status", "NEW"),
        ("review_basis", "AUTOMATIC"),
        ("confidence", "CERTAIN"),
        ("tags", ["Not-Normalized"]),
        ("tags", ["repeat", "repeat"]),
        ("reviewed_at", "2026-07-22T10:00:00+00:00"),
        ("rationale", "placeholder"),
        ("next_action", "tbd"),
    ],
)
def test_invalid_decision_fields_are_rejected_by_input_validation(
    tmp_path: Path,
    project_root: Path,
    field: str,
    invalid: Any,
) -> None:
    seed = seed_catalog(tmp_path)
    value = batch(seed)
    value["decisions"][0][field] = invalid
    input_path = tmp_path / f"invalid-{field}.json"
    input_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ReviewValidationError):
        load_review_batch(
            input_path, project_root / "schemas" / "human_review_batch.schema.json"
        )


def test_unknown_input_properties_and_untrimmed_strings_are_rejected(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    value = batch(seed)
    value["unexpected"] = True
    input_path = tmp_path / "unknown.json"
    input_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ReviewValidationError, match="Additional properties"):
        load_review_batch(
            input_path, project_root / "schemas" / "human_review_batch.schema.json"
        )
    value.pop("unexpected")
    value["reviewer"] = " padded "
    input_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ReviewValidationError, match="trimmed"):
        load_review_batch(
            input_path, project_root / "schemas" / "human_review_batch.schema.json"
        )


def test_one_invalid_decision_rolls_back_complete_multi_item_batch(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path, item_count=2)
    decisions = [decision(seed, 0), decision(seed, 1)]
    decisions[1]["source_run_id"] = "missing-run"
    with pytest.raises(ReviewValidationError, match="unknown source run"):
        record(seed, batch(seed, decisions), project_root)
    with connect(seed.database) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM review_decisions"
        ).fetchone()[0] == 0
        assert {
            row[0] for row in connection.execute("SELECT review_status FROM items")
        } == {"NEW"}


def test_dry_run_changes_neither_database_nor_filesystem(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    before = file_snapshot(seed.root)
    summary = record(seed, batch(seed), project_root, dry_run=True)
    after = file_snapshot(seed.root)
    assert summary["would_record"] == 1
    assert summary["recorded"] == 0
    assert before == after


def test_recording_does_not_modify_immutable_intake_queue(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    markdown = seed.root / "review" / "markdown" / f"{seed.run_id}.md"
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_bytes(b"# Immutable intake queue\n")
    original_jsonl = seed.queue_path.read_bytes()
    original_markdown = markdown.read_bytes()
    record(seed, batch(seed), project_root)
    assert seed.queue_path.read_bytes() == original_jsonl
    assert markdown.read_bytes() == original_markdown
    assert not (seed.root / "human-review").exists()


def test_review_decisions_update_and_delete_triggers_enforce_append_only(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    record(seed, batch(seed), project_root)
    with connect(seed.database) as connection:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute(
                "UPDATE review_decisions SET rationale='replacement'"
            )
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute("DELETE FROM review_decisions")


def test_decision_id_is_stable_for_canonical_payload_order() -> None:
    payload = {
        "status": "PARKED",
        "item_id": "ri_" + "a" * 64,
        "tags": ["one", "two"],
    }
    reordered = {
        "tags": ["one", "two"],
        "item_id": "ri_" + "a" * 64,
        "status": "PARKED",
    }
    expected = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert decision_id_for(payload) == decision_id_for(reordered) == expected


def test_tags_are_canonical_sets_for_identity_storage_and_exports(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    original = batch(seed, [decision(seed, tags=["zeta", "alpha", "middle"])])
    first = record(seed, original, project_root)
    output = seed.root / "human-review"
    export_reviews(seed.database, output)
    first_export = file_snapshot(output)
    replay = json.loads(json.dumps(original))
    replay["batch_id"] = "fictional-batch-002"
    replay["decisions"][0]["expected_current_status"] = "REJECTED"
    replay["decisions"][0]["tags"] = ["middle", "zeta", "alpha"]
    second = record(seed, replay, project_root)
    export_reviews(seed.database, output)
    second_export = file_snapshot(output)
    assert second["already_recorded"] == 1
    assert second["projected"][0]["decision_id"] == first["projected"][0]["decision_id"]
    assert second_export == first_export
    with connect(seed.database) as connection:
        item = connection.execute(
            "SELECT tags_json,canonical_json FROM items"
        ).fetchone()
        history = connection.execute(
            "SELECT tags_json FROM review_decisions"
        ).fetchall()
    expected_tags = ["alpha", "middle", "zeta"]
    assert json.loads(item["tags_json"]) == expected_tags
    assert json.loads(item["canonical_json"])["review"]["tags"] == expected_tags
    assert [json.loads(row["tags_json"]) for row in history] == [expected_tags]


def test_explicit_queued_version_allows_review_of_cataloged_older_revision(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    item_id = seed.items[0]["identity"]["item_id"]
    with connect(seed.database) as connection:
        canonical = json.loads(
            connection.execute("SELECT canonical_json FROM items").fetchone()[0]
        )
        canonical["identity"]["current_arxiv_version"] = "v2"
        connection.execute(
            "UPDATE items SET current_version='v2',canonical_json=? WHERE item_id=?",
            (canonical_json(canonical), item_id),
        )
        connection.execute(
            "INSERT INTO item_revisions(item_id,arxiv_version,first_observed_run_id,"
            "raw_artifact_path,raw_artifact_sha256,source_fingerprint,normalized_json) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                item_id,
                "v2",
                seed.run_id,
                "raw/v2.xml",
                "2" * 64,
                "3" * 64,
                canonical_json(canonical),
            ),
        )
    without_version = batch(seed)
    with pytest.raises(ReviewValidationError, match="supply item_version"):
        record(seed, without_version, project_root)
    with_version = batch(seed)
    with_version["decisions"][0]["item_version"] = "v1"
    assert record(seed, with_version, project_root)["recorded"] == 1


def test_exports_are_deterministic_and_contain_full_history(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path)
    value = batch(
        seed,
        [
            decision(
                seed,
                basis="FULL_TEXT_REVIEW",
                tags=["methodology-watchlist"],
            )
        ],
    )
    record(seed, value, project_root)
    output = seed.root / "human-review"
    export_reviews(seed.database, output)
    first = file_snapshot(output)
    export_reviews(seed.database, output)
    second = file_snapshot(output)
    assert first == second
    assert all(b"\r\n" not in content for content in first.values())
    exported = json.loads(first["index.jsonl"].decode("utf-8"))
    assert exported["review_basis"] == "FULL_TEXT_REVIEW"
    assert exported["item_title"] == "Paper 1"
    assert exported["current_version"] == "v1"
    assert exported["previous_state"]["status"] == "NEW"
    markdown = first["index.md"].decode("utf-8")
    assert "`FULL_TEXT_REVIEW`" in markdown
    assert value["decisions"][0]["rationale"] in markdown


def test_export_history_uses_true_timestamp_order_with_stable_tie_break(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path, item_count=2)
    later_fraction = decision(
        seed,
        0,
        status="PARKED",
        reviewed_at="2026-07-22T10:00:00.100000Z",
    )
    exact_second = decision(
        seed,
        1,
        status="REJECTED",
        reviewed_at="2026-07-22T10:00:00Z",
    )
    record(seed, batch(seed, [later_fraction, exact_second]), project_root)
    output = seed.root / "human-review"
    export_reviews(seed.database, output)
    records = [
        json.loads(line)
        for line in (output / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["item_id"] for record in records] == [
        seed.items[1]["identity"]["item_id"],
        seed.items[0]["identity"]["item_id"],
    ]


def test_current_deep_review_and_methodology_exports_use_catalog_projection(
    tmp_path: Path, project_root: Path
) -> None:
    seed = seed_catalog(tmp_path, item_count=3)
    decisions = [
        decision(seed, 0, status="KEEP_FOR_DEEP_REVIEW"),
        decision(
            seed,
            1,
            status="REJECTED",
            tags=["methodology-watchlist"],
        ),
        decision(seed, 2, status="PARKED"),
    ]
    record(seed, batch(seed, decisions), project_root)
    output = seed.root / "human-review"
    export_reviews(seed.database, output)
    deep = (output / "deep-review-queue.md").read_text(encoding="utf-8")
    watch = (output / "methodology-watchlist.md").read_text(encoding="utf-8")
    assert "Paper 1" in deep
    assert "Paper 2" not in deep
    assert "Paper 3" not in deep
    assert "Paper 2" in watch
    assert "Paper 1" not in watch
    assert "Paper 3" not in watch


def test_export_failure_preserves_previous_complete_export(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = seed_catalog(tmp_path)
    record(seed, batch(seed), project_root)
    output = seed.root / "human-review"
    export_reviews(seed.database, output)
    before = file_snapshot(output)
    follow_up = batch(
        seed,
        [
            decision(
                seed,
                status="PARKED",
                expected="KEEP_FOR_DEEP_REVIEW",
                reviewed_at="2026-07-22T11:00:00Z",
            )
        ],
    )
    follow_up["batch_id"] = "fictional-batch-002"
    record(seed, follow_up, project_root)
    original_replace = os.replace
    promotion_count = 0

    def fail_stage_promotion(source: str | Path, destination: str | Path) -> None:
        nonlocal promotion_count
        source_path = Path(source)
        if (
            source_path.parent.name.endswith(".tmp")
            and Path(destination).parent == output
        ):
            promotion_count += 1
            if promotion_count == 3:
                raise OSError("simulated export promotion failure")
        original_replace(source, destination)

    monkeypatch.setattr("research_intake.human_review.os.replace", fail_stage_promotion)
    with pytest.raises(OSError, match="simulated"):
        export_reviews(seed.database, output)
    assert file_snapshot(output) == before


def test_cli_exit_codes_and_summaries_are_explicit(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed = seed_catalog(tmp_path)
    with connect(seed.database) as connection:
        connection.execute("DROP TABLE review_decisions")
    monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(seed.root))
    input_path = tmp_path / "fictional-review-batch.json"
    input_path.write_text(json.dumps(batch(seed)), encoding="utf-8")
    assert main(["record-reviews", "--input", str(input_path), "--dry-run"]) == 0
    dry_summary = json.loads(capsys.readouterr().out)
    assert dry_summary["dry_run"] is True
    assert dry_summary["would_record"] == 1
    assert not (seed.root / ".research-intake.lock").exists()
    assert main(["record-reviews", "--input", str(input_path)]) == 0
    recorded_summary = json.loads(capsys.readouterr().out)
    assert recorded_summary["recorded"] == 1
    assert main(["record-reviews", "--input", str(input_path)]) == 0
    replay_summary = json.loads(capsys.readouterr().out)
    assert replay_summary["already_recorded"] == 1
    assert main(["export-reviews"]) == 0
    export_summary = json.loads(capsys.readouterr().out)
    assert export_summary["files"] == [
        "index.jsonl",
        "index.md",
        "deep-review-queue.md",
        "methodology-watchlist.md",
    ]
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{}", encoding="utf-8")
    assert main(["record-reviews", "--input", str(invalid_path)]) == 2
    assert "invalid review batch schema" in capsys.readouterr().err
    assert project_root.exists()


def test_operation_lock_blocks_overlapping_record_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed = seed_catalog(tmp_path)
    with connect(seed.database) as connection:
        connection.execute("DROP TABLE review_decisions")
    monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(seed.root))
    input_path = tmp_path / "fictional-review-batch.json"
    input_path.write_text(json.dumps(batch(seed)), encoding="utf-8")
    with OperationLock(seed.root / ".research-intake.lock"):
        assert main(["record-reviews", "--input", str(input_path)]) == 1
    assert "OPERATION_ALREADY_ACTIVE" in capsys.readouterr().err
    with connect(seed.database) as connection:
        assert not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='review_decisions'"
        ).fetchone()


def test_record_review_command_does_not_load_intake_or_make_network_call(
    tmp_path: Path,
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seed = seed_catalog(tmp_path)
    monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(seed.root))
    input_path = tmp_path / "fictional-review-batch.json"
    input_path.write_text(json.dumps(batch(seed)), encoding="utf-8")

    def prohibited(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("intake or network path was invoked")

    monkeypatch.setattr("research_intake.cli.load_config", prohibited)
    monkeypatch.setattr("research_intake.cli.contact_email", prohibited)
    monkeypatch.setattr("httpx.Client.request", prohibited)
    assert main(["record-reviews", "--input", str(input_path)]) == 0
    assert json.loads(capsys.readouterr().out)["recorded"] == 1
    assert project_root.exists()
