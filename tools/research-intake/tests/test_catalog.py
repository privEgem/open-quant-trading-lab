from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_intake.arxiv import parse_atom
from research_intake.catalog import (
    append_event,
    connect,
    create_run,
    discovery_queue_items,
    ingest_batch,
    init_database,
    latest_eligible_run,
    predecessor_decision,
)
from research_intake.config import load_config
from research_intake.evidence import sha256_file
from research_intake.normalization import normalize_observation, source_fingerprint
from research_intake.relevance import score_item


def setup_run(database: Path, run_id: str = "run1") -> None:
    init_database(database)
    with connect(database) as connection:
        create_run(connection, {"run_id": run_id, "run_type": "backfill", "start_timestamp_utc": "2026-07-12T00:00:00Z", "interval_start_utc": "2026-07-01T00:00:00Z", "interval_end_utc": "2026-07-12T00:00:00Z", "discovery_time_basis": "SUBMITTED_DATE", "application_version": "0.1.0", "git_commit": None, "python_version": "3.12", "registry_hash": "a" * 64, "topics_hash": "b" * 64, "previous_eligible_run_id": None, "overlap_seconds": 1, "manifest_path": str(database.parent / f"{run_id}.json")}, [{"source_id": "arxiv", "query_id": "q"}])


def observation(project_root: Path, xml: bytes, raw: Path, run_id: str = "run1", query: str = "q", mutate: str | None = None) -> dict[str, object]:
    raw.write_bytes(xml)
    entry = parse_atom(xml).entries[0]
    if mutate:
        entry["comments"] = mutate
    config = load_config(project_root)
    relevance = score_item(entry, config.topics, "category_unconditional")
    item = normalize_observation(entry, run_id, [query], "2026-07-12T00:00:00Z", str(raw), sha256_file(raw), relevance)
    return {"item": item, "query_id": query, "raw_path": str(raw), "raw_hash": sha256_file(raw), "source_fingerprint": source_fingerprint(entry)}


def ingest(connection, run_id: str, values: list[dict[str, object]], directory: Path) -> dict[str, int]:
    return ingest_batch(
        connection,
        run_id,
        values,
        "2026-07-12T00:01:00Z",
        "markdown-jsonl-v1",
        str(directory / f"{run_id}.md"),
        str(directory / f"{run_id}.jsonl"),
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "normalized_item.schema.json",
    )


def test_initialization_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "catalog.sqlite3"
    init_database(database)
    init_database(database)
    with connect(database) as connection:
        assert connection.execute("SELECT value FROM schema_metadata WHERE key='schema_version'").fetchone()[0] == "1"


def test_first_insert_and_exact_repeat(project_root: Path, mixed_xml: bytes, tmp_path: Path) -> None:
    database = tmp_path / "catalog.sqlite3"
    setup_run(database)
    first = observation(project_root, mixed_xml, tmp_path / "raw1.xml")
    with connect(database) as connection:
        assert ingest(connection, "run1", [first], tmp_path)["new"] == 1
        assert connection.execute("SELECT COUNT(*) FROM item_revisions").fetchone()[0] == 1
        item_row = connection.execute(
            "SELECT authors_json,categories_json,abstract,review_priority FROM items"
        ).fetchone()
        assert json.loads(item_row["authors_json"]) == ["Alice Smith", "Bob Jones"]
        assert json.loads(item_row["categories_json"]) == ["q-fin.TR", "q-fin.ST"]
        assert "market microstructure" in item_row["abstract"]
        assert item_row["review_priority"] == "HIGH"
    setup_run(database, "run2")
    second = observation(project_root, mixed_xml, tmp_path / "raw2.xml", "run2")
    with connect(database) as connection:
        assert ingest(connection, "run2", [second], tmp_path)["unchanged"] == 1
        assert connection.execute("SELECT COUNT(*) FROM item_revisions").fetchone()[0] == 1


def test_same_version_metadata_change_updates_canonical(project_root: Path, mixed_xml: bytes, tmp_path: Path) -> None:
    database = tmp_path / "catalog.sqlite3"
    setup_run(database)
    with connect(database) as connection:
        ingest(connection, "run1", [observation(project_root, mixed_xml, tmp_path / "a.xml")], tmp_path)
    setup_run(database, "run2")
    changed = observation(project_root, mixed_xml, tmp_path / "b.xml", "run2", mutate="New comments")
    with connect(database) as connection:
        assert ingest(connection, "run2", [changed], tmp_path)["updated"] == 1
        item = json.loads(connection.execute("SELECT canonical_json FROM items").fetchone()[0])
        assert item["bibliographic"]["comments"] == "New comments"
        assert connection.execute("SELECT COUNT(*) FROM item_revisions").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM discoveries").fetchone()[0] == 2


def test_higher_version_and_older_late_observation(project_root: Path, mixed_xml: bytes, version2_xml: bytes, tmp_path: Path) -> None:
    database = tmp_path / "catalog.sqlite3"
    setup_run(database)
    with connect(database) as connection:
        ingest(connection, "run1", [observation(project_root, version2_xml, tmp_path / "v2.xml")], tmp_path)
    setup_run(database, "run2")
    with connect(database) as connection:
        counts = ingest(connection, "run2", [observation(project_root, mixed_xml, tmp_path / "v1.xml", "run2")], tmp_path)
        assert counts["unchanged"] == 1
        assert connection.execute("SELECT current_version FROM items").fetchone()[0] == "v2"
        assert connection.execute("SELECT last_seen_run_id FROM items").fetchone()[0] == "run2"
        assert connection.execute("SELECT COUNT(*) FROM item_revisions").fetchone()[0] == 2
        snapshot = json.loads(
            connection.execute(
                "SELECT normalized_snapshot_json FROM discoveries WHERE run_id='run2'"
            ).fetchone()[0]
        )
        assert snapshot["lifecycle"]["latest_source_version"] is False


def test_higher_version_retains_human_review_state_in_queue_snapshot(
    project_root: Path, mixed_xml: bytes, version2_xml: bytes, tmp_path: Path
) -> None:
    database = tmp_path / "catalog.sqlite3"
    setup_run(database)
    with connect(database) as connection:
        ingest(
            connection,
            "run1",
            [observation(project_root, mixed_xml, tmp_path / "v1.xml")],
            tmp_path,
        )
        canonical = json.loads(
            connection.execute("SELECT canonical_json FROM items").fetchone()[0]
        )
        canonical["review"].update(
            {
                "status": "REJECTED",
                "priority": "LOW",
                "reviewer": "human",
                "reviewed_at": "2026-07-12T01:00:00Z",
                "tags": ["reviewed"],
                "notes": "retain this decision",
            }
        )
        connection.execute(
            "UPDATE items SET review_status='REJECTED',review_priority='LOW',"
            "reviewer='human',reviewed_at='2026-07-12T01:00:00Z',"
            "tags_json='[\"reviewed\"]',notes='retain this decision',canonical_json=?",
            (json.dumps(canonical, sort_keys=True, separators=(",", ":")),),
        )
    setup_run(database, "run2")
    with connect(database) as connection:
        ingest(
            connection,
            "run2",
            [observation(project_root, version2_xml, tmp_path / "v2.xml", "run2")],
            tmp_path,
        )
        queued = discovery_queue_items(connection, "run2")
        assert queued[0]["review"] == canonical["review"]
        assert connection.execute("SELECT review_status FROM items").fetchone()[0] == "REJECTED"


def test_shared_doi_preserves_both_items_and_warns(project_root: Path, mixed_xml: bytes, tmp_path: Path) -> None:
    database = tmp_path / "catalog.sqlite3"
    setup_run(database)
    one = observation(project_root, mixed_xml, tmp_path / "one.xml")
    two = json.loads(json.dumps(one))
    two["item"]["identity"]["item_id"] = "ri_" + "f" * 64
    two["item"]["identity"]["base_arxiv_id"] = "2607.99999"
    two["item"]["identity"]["primary_key_value"] = "2607.99999"
    two["item"]["provenance"]["source_record_id"] = "http://arxiv.org/abs/2607.99999v1"
    with connect(database) as connection:
        counts = ingest(connection, "run1", [one, two], tmp_path)
        assert counts["doi_collisions"] == 1
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2
        event = connection.execute("SELECT event_code FROM run_events").fetchone()[0]
        assert event == "DOI_COLLISION"


def test_batch_rollback_also_rolls_back_staged_collision_event(
    project_root: Path, mixed_xml: bytes, tmp_path: Path
) -> None:
    database = tmp_path / "catalog.sqlite3"
    setup_run(database)
    one = observation(project_root, mixed_xml, tmp_path / "one.xml")
    two = json.loads(json.dumps(one))
    two["item"]["identity"].update(
        {
            "item_id": "ri_" + "e" * 64,
            "base_arxiv_id": "2607.88888",
            "primary_key_value": "2607.88888",
        }
    )
    two["item"]["provenance"]["source_record_id"] = (
        "http://arxiv.org/abs/2607.88888v1"
    )
    invalid = json.loads(json.dumps(two))
    invalid["item"]["identity"].update(
        {
            "item_id": "ri_" + "d" * 64,
            "base_arxiv_id": "2607.77777",
            "primary_key_value": "2607.77777",
        }
    )
    invalid["item"]["provenance"]["source_record_id"] = (
        "http://arxiv.org/abs/2607.77777v1"
    )
    invalid["raw_hash"] = "0" * 64
    with connect(database) as connection:
        with pytest.raises(RuntimeError, match="EVIDENCE_HASH_MISMATCH"):
            ingest(connection, "run1", [one, two, invalid], tmp_path)
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM run_events").fetchone()[0] == 0


def test_evidence_mismatch_rolls_back(project_root: Path, mixed_xml: bytes, tmp_path: Path) -> None:
    database = tmp_path / "catalog.sqlite3"
    setup_run(database)
    value = observation(project_root, mixed_xml, tmp_path / "raw.xml")
    value["raw_hash"] = "0" * 64
    with connect(database) as connection:
        with pytest.raises(RuntimeError, match="EVIDENCE_HASH_MISMATCH"):
            ingest(connection, "run1", [value], tmp_path)
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0


def test_final_normalized_validation_rolls_back(
    project_root: Path, mixed_xml: bytes, tmp_path: Path
) -> None:
    database = tmp_path / "catalog.sqlite3"
    setup_run(database)
    value = observation(project_root, mixed_xml, tmp_path / "raw.xml")
    value["item"]["bibliographic"]["title"] = None
    with connect(database) as connection:
        with pytest.raises(Exception, match="None is not of type 'string'"):
            ingest(connection, "run1", [value], tmp_path)
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0


def test_append_only_events(tmp_path: Path) -> None:
    database = tmp_path / "catalog.sqlite3"
    setup_run(database)
    with connect(database) as connection:
        append_event(connection, "run1", "INFO", "TEST", "TEST")
        with pytest.raises(Exception, match="append-only"):
            connection.execute("UPDATE run_events SET event_code='OTHER'")


def test_event_context_rejects_secret_fields(tmp_path: Path) -> None:
    database = tmp_path / "catalog.sqlite3"
    setup_run(database)
    with connect(database) as connection:
        with pytest.raises(ValueError, match="prohibited secret field"):
            append_event(
                connection,
                "run1",
                "ERROR",
                "TEST",
                "TEST",
                {"request": {"Authorization": "Bearer must-not-persist"}},
            )
        assert connection.execute("SELECT COUNT(*) FROM run_events").fetchone()[0] == 0


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    database = tmp_path / "catalog.sqlite3"
    init_database(database)
    with connect(database) as connection, pytest.raises(
        Exception, match="FOREIGN KEY constraint failed"
    ):
        connection.execute(
            "INSERT INTO run_events(run_id,timestamp_utc,severity,event_code,"
            "pipeline_phase,context_json) VALUES(?,?,?,?,?,?)",
            ("missing", "2026-07-12T00:00:00Z", "ERROR", "TEST", "TEST", "{}"),
        )


@pytest.mark.parametrize(
    ("status", "exhausted", "crossed", "catalog", "expected"),
    [("COMPLETE", 1, 0, 1, True), ("COMPLETE", 0, 1, 1, True), ("INCOMPLETE", 0, 0, 1, False), ("FAILED", 0, 0, 1, False), ("COMPLETE", 1, 0, 0, False)],
)
def test_predecessor_decision(tmp_path: Path, status: str, exhausted: int, crossed: int, catalog: int, expected: bool) -> None:
    database = tmp_path / "catalog.sqlite3"
    setup_run(database)
    with connect(database) as connection:
        connection.execute("UPDATE runs SET catalog_committed=? WHERE run_id='run1'", (catalog,))
        connection.execute("UPDATE run_query_state SET completion_status=?,exhausted_available_results=?,crossed_lower_boundary=?,final_stop_reason=? WHERE run_id='run1'", (status, exhausted, crossed, "QUERY_FAILURE" if status == "FAILED" else "MAX_ITEMS_REACHED"))
        assert predecessor_decision(connection, "run1")[0] is expected


def test_newer_ineligible_run_does_not_displace_older_eligible(tmp_path: Path) -> None:
    database = tmp_path / "catalog.sqlite3"
    setup_run(database, "older")
    setup_run(database, "newer")
    with connect(database) as connection:
        connection.execute(
            "UPDATE runs SET start_timestamp_utc='2026-07-10T00:00:00Z',"
            "incremental_predecessor_eligible=1 WHERE run_id='older'"
        )
        connection.execute(
            "UPDATE runs SET start_timestamp_utc='2026-07-11T00:00:00Z',"
            "incremental_predecessor_eligible=0 WHERE run_id='newer'"
        )
        assert latest_eligible_run(connection)["run_id"] == "older"
