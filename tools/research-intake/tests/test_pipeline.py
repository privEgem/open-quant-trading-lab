from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from research_intake.catalog import connect, init_database
from research_intake.config import load_config
from research_intake.paths import create_runtime_directories, resolve_paths
from research_intake.pipeline import NoEligibleIncrementalPredecessor, run_intake


class FixedClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


def prepared(project_root: Path, tmp_path: Path, monkeypatch, body: bytes):
    monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(tmp_path))
    config = load_config(project_root)
    config.registry["policy"]["minimum_request_interval_seconds"] = 0
    paths = resolve_paths(project_root)
    create_runtime_directories(paths)
    init_database(paths.database)
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body, headers={"Content-Type": "application/atom+xml", "ETag": "fixture"}, request=request))
    return config, paths, httpx.Client(transport=transport)


def test_offline_backfill_end_to_end(project_root: Path, mixed_xml: bytes, tmp_path: Path, monkeypatch) -> None:
    config, paths, client = prepared(project_root, tmp_path, monkeypatch, mixed_xml)
    code, run_id = run_intake(config, paths, "backfill", 30, 20, client=client, clock=FixedClock(datetime(2026, 7, 12, tzinfo=timezone.utc)), sleeper=lambda _: None)
    assert code == 3  # isolated malformed item
    manifest = json.loads((paths.manifests / f"{run_id}.json").read_text())
    assert manifest["discovery"]["time_basis"] == "SUBMITTED_DATE"
    assert manifest["incremental_predecessor_eligible"] is True
    assert manifest["counts"]["malformed"] == 2
    assert manifest["configuration_hashes"] == {
        "source_registry": config.registry_hash,
        "topics": config.topics_hash,
    }
    assert manifest["queues"]["markdown"]["count"] == manifest["queues"][
        "jsonl"
    ]["count"]
    assert all(
        warning["event_code"] == "MALFORMED_ENTRY"
        for warning in manifest["warnings"]
    )
    with connect(paths.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 3
        assert connection.execute("SELECT incremental_predecessor_eligible FROM runs").fetchone()[0] == 1
        canonical = json.loads(
            connection.execute(
                "SELECT canonical_json FROM items WHERE base_arxiv_id='2607.01234'"
            ).fetchone()[0]
        )
        assert canonical["provenance"]["discovery_query_ids"] == [
            "arxiv_core_qfin_tr",
            "arxiv_secondary_qfin_st_cp",
        ]
        assert "arxiv_core_qfin_tr" in canonical["provenance"][
            "raw_artifact_path"
        ]


def test_incremental_finds_new_version_of_old_paper(project_root: Path, mixed_xml: bytes, version2_xml: bytes, tmp_path: Path, monkeypatch) -> None:
    config, paths, first_client = prepared(project_root, tmp_path, monkeypatch, mixed_xml)
    code, _ = run_intake(config, paths, "backfill", 30, 20, client=first_client, clock=FixedClock(datetime(2026, 7, 12, tzinfo=timezone.utc)), sleeper=lambda _: None)
    assert code == 3
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=version2_xml, request=request))
    second_client = httpx.Client(transport=transport)
    code, run_id = run_intake(config, paths, "incremental", None, 20, client=second_client, clock=FixedClock(datetime(2026, 7, 13, tzinfo=timezone.utc)), sleeper=lambda _: None)
    assert code == 0
    manifest = json.loads((paths.manifests / f"{run_id}.json").read_text())
    assert manifest["discovery"]["time_basis"] == "UPDATED_DATE"
    with connect(paths.database) as connection:
        assert connection.execute("SELECT current_version FROM items WHERE base_arxiv_id='2607.01234'").fetchone()[0] == "v2"


def test_incremental_interval_is_start_inclusive_and_end_exclusive(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    empty = b"""<feed xmlns='http://www.w3.org/2005/Atom'
      xmlns:opensearch='http://a9.com/-/spec/opensearch/1.1/'>
      <opensearch:totalResults>0</opensearch:totalResults></feed>"""
    config, paths, first_client = prepared(
        project_root, tmp_path, monkeypatch, empty
    )
    first_code, _ = run_intake(
        config,
        paths,
        "backfill",
        1,
        20,
        client=first_client,
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=timezone.utc)),
        sleeper=lambda _: None,
    )
    assert first_code == 0
    boundary_feed = b"""<feed xmlns='http://www.w3.org/2005/Atom'
      xmlns:arxiv='http://arxiv.org/schemas/atom'
      xmlns:opensearch='http://a9.com/-/spec/opensearch/1.1/'>
      <opensearch:totalResults>2</opensearch:totalResults>
      <entry><id>http://arxiv.org/abs/2607.40002v1</id>
        <updated>2026-07-13T00:00:00Z</updated><published>2026-01-02T00:00:00Z</published>
        <title>Execution at excluded end</title><summary>Transaction costs.</summary>
        <author><name>End</name></author><arxiv:primary_category term='q-fin.TR'/>
        <category term='q-fin.TR'/></entry>
      <entry><id>http://arxiv.org/abs/2607.40001v1</id>
        <updated>2026-07-11T00:00:00Z</updated><published>2026-01-01T00:00:00Z</published>
        <title>Liquidity at included start</title><summary>Order flow.</summary>
        <author><name>Start</name></author><arxiv:primary_category term='q-fin.TR'/>
        <category term='q-fin.TR'/></entry></feed>"""
    second_client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, content=boundary_feed, request=request
            )
        )
    )
    code, run_id = run_intake(
        config,
        paths,
        "incremental",
        None,
        20,
        client=second_client,
        clock=FixedClock(datetime(2026, 7, 13, tzinfo=timezone.utc)),
        sleeper=lambda _: None,
    )
    assert code == 0
    manifest = json.loads((paths.manifests / f"{run_id}.json").read_text())
    assert manifest["discovery"]["interval_start_inclusive"] == "2026-07-11T00:00:00Z"
    assert manifest["discovery"]["interval_end_exclusive"] == "2026-07-13T00:00:00Z"
    with connect(paths.database) as connection:
        assert [
            row[0]
            for row in connection.execute("SELECT base_arxiv_id FROM items")
        ] == ["2607.40001"]


def test_incremental_refuses_without_eligible_predecessor(project_root: Path, version2_xml: bytes, tmp_path: Path, monkeypatch) -> None:
    config, paths, client = prepared(project_root, tmp_path, monkeypatch, version2_xml)
    with pytest.raises(
        NoEligibleIncrementalPredecessor,
        match="no incremental predecessor is eligible.*bounded backfill",
    ):
        run_intake(config, paths, "incremental", None, 20, client=client, clock=FixedClock(datetime(2026, 7, 13, tzinfo=timezone.utc)), sleeper=lambda _: None)
    with connect(paths.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_structurally_failed_page_writes_failed_manifest(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    malformed = (
        project_root / "tests" / "fixtures" / "arxiv" / "malformed.xml"
    ).read_bytes()
    config, paths, client = prepared(project_root, tmp_path, monkeypatch, malformed)
    code, run_id = run_intake(
        config,
        paths,
        "backfill",
        30,
        20,
        client=client,
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=timezone.utc)),
        sleeper=lambda _: None,
    )
    assert code == 1
    manifest = json.loads((paths.manifests / f"{run_id}.json").read_text())
    assert manifest["final_status"] == "FAILED"
    assert manifest["incremental_predecessor_eligible"] is False
    assert all(
        query["final_stop_reason"] == "PAGE_FAILURE"
        for query in manifest["discovery"]["queries"]
    )


def test_queue_failure_is_partial_but_predecessor_eligible(
    project_root: Path, version2_xml: bytes, tmp_path: Path, monkeypatch
) -> None:
    config, paths, client = prepared(project_root, tmp_path, monkeypatch, version2_xml)

    def fail_queues(*args):
        raise OSError("queue disk failure")

    monkeypatch.setattr("research_intake.pipeline.publish_queues", fail_queues)
    code, run_id = run_intake(
        config,
        paths,
        "backfill",
        365,
        20,
        client=client,
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=timezone.utc)),
        sleeper=lambda _: None,
    )
    assert code == 3
    manifest = json.loads((paths.manifests / f"{run_id}.json").read_text())
    assert manifest["final_status"] == "PARTIAL_SUCCESS"
    assert manifest["incremental_predecessor_eligible"] is True
    assert any(
        event["event_code"] == "QUEUE_GENERATION_FAILURE"
        for event in manifest["errors"]
    )


def test_catalog_failure_rolls_back_and_writes_failed_manifest(
    project_root: Path, version2_xml: bytes, tmp_path: Path, monkeypatch
) -> None:
    config, paths, client = prepared(project_root, tmp_path, monkeypatch, version2_xml)

    def fail_catalog(*args, **kwargs):
        raise sqlite3.DatabaseError("simulated catalog failure")

    monkeypatch.setattr("research_intake.pipeline.ingest_batch", fail_catalog)
    code, run_id = run_intake(
        config,
        paths,
        "backfill",
        365,
        20,
        client=client,
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=timezone.utc)),
        sleeper=lambda _: None,
    )
    assert code == 1
    manifest = json.loads((paths.manifests / f"{run_id}.json").read_text())
    assert manifest["final_status"] == "FAILED"
    assert any(
        event["event_code"] == "CATALOG_TRANSACTION_FAILURE"
        for event in manifest["errors"]
    )
    with connect(paths.database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0
        assert connection.execute(
            "SELECT catalog_committed FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()[0] == 0


def test_malformed_entry_advances_pagination_offset_safely(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    def feed(entries: str) -> bytes:
        return f"""<?xml version='1.0'?>
        <feed xmlns='http://www.w3.org/2005/Atom'
          xmlns:arxiv='http://arxiv.org/schemas/atom'
          xmlns:opensearch='http://a9.com/-/spec/opensearch/1.1/'>
          <opensearch:totalResults>3</opensearch:totalResults>{entries}</feed>""".encode()

    valid_one = """<entry><id>http://arxiv.org/abs/2607.10001v1</id>
      <updated>2026-07-10T12:00:00Z</updated><published>2026-07-10T12:00:00Z</published>
      <title>Market microstructure</title><summary>Order flow.</summary>
      <author><name>A One</name></author><arxiv:primary_category term='q-fin.TR'/>
      <category term='q-fin.TR'/><link href='https://arxiv.org/abs/2607.10001v1' rel='alternate'/></entry>"""
    malformed = """<entry><id>http://arxiv.org/abs/2607.10002v1</id>
      <updated>2026-07-10T11:00:00Z</updated><published>2026-07-10T11:00:00Z</published>
      <title>Missing author</title><summary>Order flow.</summary>
      <arxiv:primary_category term='q-fin.TR'/><category term='q-fin.TR'/></entry>"""
    valid_two = """<entry><id>http://arxiv.org/abs/2607.10003v1</id>
      <updated>2026-07-10T10:00:00Z</updated><published>2026-07-10T10:00:00Z</published>
      <title>Liquidity</title><summary>Bid-ask spread.</summary>
      <author><name>B Two</name></author><arxiv:primary_category term='q-fin.TR'/>
      <category term='q-fin.TR'/><link href='https://arxiv.org/abs/2607.10003v1' rel='alternate'/></entry>"""
    monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(tmp_path))
    config = load_config(project_root)
    config.registry["policy"]["minimum_request_interval_seconds"] = 0
    paths = resolve_paths(project_root)
    create_runtime_directories(paths)
    init_database(paths.database)
    starts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        start = int(request.url.params["start"])
        starts.append(start)
        body = feed(valid_one + malformed) if start == 0 else feed(valid_two)
        return httpx.Response(200, content=body, request=request)

    code, run_id = run_intake(
        config,
        paths,
        "backfill",
        30,
        20,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=timezone.utc)),
        sleeper=lambda _: None,
    )
    assert code == 3
    assert starts == [0, 2, 0, 2]
    manifest = json.loads((paths.manifests / f"{run_id}.json").read_text())
    assert manifest["incremental_predecessor_eligible"] is True
    assert all(query["pages_succeeded"] == 2 for query in manifest["discovery"]["queries"])


def test_retry_cannot_exceed_request_limit_and_respects_retry_after(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(tmp_path))
    config = load_config(project_root)
    config.registry["policy"]["minimum_request_interval_seconds"] = 0
    config.registry["policy"]["maximum_requests_per_run"] = 1
    config.registry["policy"]["retry"]["max_attempts"] = 3
    paths = resolve_paths(project_root)
    create_runtime_directories(paths)
    init_database(paths.database)
    requests = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            503,
            content=b"temporarily unavailable",
            headers={"Retry-After": "7"},
            request=request,
        )

    code, run_id = run_intake(
        config,
        paths,
        "backfill",
        30,
        20,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=timezone.utc)),
        sleeper=sleeps.append,
    )
    assert code == 1
    assert requests == 1
    assert sleeps == [7.0]
    manifest = json.loads((paths.manifests / f"{run_id}.json").read_text())
    assert manifest["final_status"] == "FAILED"
    assert manifest["source_requests"]["arxiv_core_qfin_tr"]["attempts"] == 1
    assert Path(manifest["raw_artifacts"][0]["body_path"]).read_bytes() == b"temporarily unavailable"
    assert manifest["raw_artifacts"][0]["response"]["headers"]["Retry-After"] == "7"
    with connect(paths.database) as connection:
        assert connection.execute(
            "SELECT catalog_committed FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()[0] == 0


def test_non_descending_source_page_is_an_ineligible_gap(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    body = b"""<feed xmlns='http://www.w3.org/2005/Atom'
      xmlns:arxiv='http://arxiv.org/schemas/atom'
      xmlns:opensearch='http://a9.com/-/spec/opensearch/1.1/'>
      <opensearch:totalResults>2</opensearch:totalResults>
      <entry><id>http://arxiv.org/abs/2607.20001v1</id>
        <updated>2026-07-09T00:00:00Z</updated><published>2026-07-09T00:00:00Z</published>
        <title>Liquidity</title><summary>Order flow.</summary><author><name>A</name></author>
        <arxiv:primary_category term='q-fin.TR'/><category term='q-fin.TR'/></entry>
      <entry><id>http://arxiv.org/abs/2607.20002v1</id>
        <updated>2026-07-10T00:00:00Z</updated><published>2026-07-10T00:00:00Z</published>
        <title>Execution</title><summary>Transaction costs.</summary><author><name>B</name></author>
        <arxiv:primary_category term='q-fin.TR'/><category term='q-fin.TR'/></entry>
      </feed>"""
    config, paths, client = prepared(project_root, tmp_path, monkeypatch, body)
    code, run_id = run_intake(
        config,
        paths,
        "backfill",
        30,
        20,
        client=client,
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=timezone.utc)),
        sleeper=lambda _: None,
    )
    assert code == 1
    manifest = json.loads((paths.manifests / f"{run_id}.json").read_text())
    assert manifest["incremental_predecessor_eligible"] is False
    assert manifest["incremental_predecessor_reason_code"] == "QUERY_SCAN_GAP"
    assert any(
        event["context"].get("reason") == "ORDERING_VIOLATION"
        for event in manifest["errors"]
    )


def test_one_usable_empty_query_and_one_failed_query_is_partial(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    empty = b"""<feed xmlns='http://www.w3.org/2005/Atom'
      xmlns:opensearch='http://a9.com/-/spec/opensearch/1.1/'>
      <opensearch:totalResults>0</opensearch:totalResults></feed>"""
    monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(tmp_path))
    config = load_config(project_root)
    config.registry["policy"]["minimum_request_interval_seconds"] = 0
    config.registry["policy"]["retry"]["max_attempts"] = 1
    paths = resolve_paths(project_root)
    create_runtime_directories(paths)
    init_database(paths.database)

    def handler(request: httpx.Request) -> httpx.Response:
        if "q-fin.TR" in request.url.params["search_query"]:
            return httpx.Response(200, content=empty, request=request)
        return httpx.Response(503, content=b"failed query", request=request)

    code, run_id = run_intake(
        config,
        paths,
        "backfill",
        30,
        20,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=timezone.utc)),
        sleeper=lambda _: None,
    )
    assert code == 3
    manifest = json.loads((paths.manifests / f"{run_id}.json").read_text())
    assert manifest["final_status"] == "PARTIAL_SUCCESS"
    assert manifest["incremental_predecessor_eligible"] is False
    assert manifest["incremental_predecessor_reason_code"] == "QUERY_FAILED"
    with connect(paths.database) as connection:
        assert connection.execute(
            "SELECT catalog_committed FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()[0] == 1


def test_runtime_limit_stops_before_request_and_is_ineligible(
    project_root: Path, version2_xml: bytes, tmp_path: Path, monkeypatch
) -> None:
    config, paths, client = prepared(project_root, tmp_path, monkeypatch, version2_xml)
    config.registry["policy"]["global_maximum_runtime_seconds"] = 5
    ticks = iter([0.0, 10.0, 20.0])
    monkeypatch.setattr(
        "research_intake.pipeline.time.monotonic", lambda: next(ticks)
    )
    code, run_id = run_intake(
        config,
        paths,
        "backfill",
        365,
        20,
        client=client,
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=timezone.utc)),
        sleeper=lambda _: None,
    )
    assert code == 1
    manifest = json.loads((paths.manifests / f"{run_id}.json").read_text())
    assert manifest["incremental_predecessor_eligible"] is False
    assert manifest["incremental_predecessor_reason_code"] == "QUERY_SCAN_INCOMPLETE"
    assert all(
        query["final_stop_reason"] == "MAX_RUNTIME_REACHED"
        for query in manifest["discovery"]["queries"]
    )


def test_item_limit_before_boundary_is_partial_and_ineligible(
    project_root: Path, tmp_path: Path, monkeypatch
) -> None:
    body = b"""<feed xmlns='http://www.w3.org/2005/Atom'
      xmlns:arxiv='http://arxiv.org/schemas/atom'
      xmlns:opensearch='http://a9.com/-/spec/opensearch/1.1/'>
      <opensearch:totalResults>2</opensearch:totalResults>
      <entry><id>http://arxiv.org/abs/2607.30001v1</id>
        <updated>2026-07-10T00:00:00Z</updated><published>2026-07-10T00:00:00Z</published>
        <title>Liquidity</title><summary>Order flow.</summary><author><name>A</name></author>
        <arxiv:primary_category term='q-fin.TR'/><category term='q-fin.TR'/></entry>
      </feed>"""
    config, paths, client = prepared(project_root, tmp_path, monkeypatch, body)
    code, run_id = run_intake(
        config,
        paths,
        "backfill",
        30,
        1,
        client=client,
        clock=FixedClock(datetime(2026, 7, 12, tzinfo=timezone.utc)),
        sleeper=lambda _: None,
    )
    assert code == 3
    manifest = json.loads((paths.manifests / f"{run_id}.json").read_text())
    assert manifest["incremental_predecessor_eligible"] is False
    assert manifest["incremental_predecessor_reason_code"] == "QUERY_SCAN_INCOMPLETE"
    assert all(
        query["final_stop_reason"] == "MAX_ITEMS_REACHED"
        for query in manifest["discovery"]["queries"]
    )
