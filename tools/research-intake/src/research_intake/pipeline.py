from __future__ import annotations

import copy
import json
import platform
import sqlite3
import subprocess
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import httpx
from jsonschema.exceptions import ValidationError

from research_intake import __version__
from research_intake.arxiv import (
    MalformedPage,
    parse_atom,
    parse_timestamp,
    request_page,
    safe_user_agent,
    selected_headers,
)
from research_intake.catalog import (
    append_event,
    connect,
    create_run,
    discovery_queue_items,
    ingest_batch,
    latest_eligible_run,
    query_states,
)
from research_intake.config import LoadedConfig, contact_email
from research_intake.evidence import preserve_response
from research_intake.manifests import build_manifest, publish_manifest
from research_intake.normalization import normalize_observation, source_fingerprint
from research_intake.paths import RuntimePaths
from research_intake.queues import RENDERER_VERSION, publish_queues, render_queue_bytes
from research_intake.relevance import score_item
from research_intake.validation import validate_json


class CompleteSourceFailure(RuntimeError):
    pass


class NoEligibleIncrementalPredecessor(RuntimeError):
    pass


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _run_id(now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex


def _git_commit(root: Path) -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _small_event(database: Path, run_id: str, severity: str, code: str, phase: str, context: dict[str, Any] | None = None, source_id: str | None = None, query_id: str | None = None) -> None:
    with connect(database) as connection:
        append_event(connection, run_id, severity, code, phase, context, source_id, query_id)
        connection.execute(
            "UPDATE runs SET warning_count=(SELECT COUNT(*) FROM run_events "
            "WHERE run_id=? AND severity='WARNING'),error_count=(SELECT COUNT(*) "
            "FROM run_events WHERE run_id=? AND severity='ERROR') WHERE run_id=?",
            (run_id, run_id, run_id),
        )


def _update_query(database: Path, run_id: str, query_id: str, **values: Any) -> None:
    if not values:
        return
    assignments = ",".join(f"{key}=?" for key in values)
    with connect(database) as connection:
        connection.execute(f"UPDATE run_query_state SET {assignments} WHERE run_id=? AND query_id=?", (*values.values(), run_id, query_id))


def _record_query_limit(
    database: Path,
    run_id: str,
    source_id: str,
    query_id: str,
    stop_reason: str,
    event_code: str,
    context: dict[str, Any] | None = None,
) -> None:
    _update_query(
        database,
        run_id,
        query_id,
        completion_status="INCOMPLETE",
        stopped_early=1,
        final_stop_reason=stop_reason,
    )
    _small_event(
        database,
        run_id,
        "WARNING",
        event_code,
        "RETRIEVAL",
        context or {},
        source_id,
        query_id,
    )


def _record_request_attempt(
    database: Path, run_id: str, query_id: str, page_number: int, timestamp: str
) -> None:
    with connect(database) as connection:
        connection.execute(
            "UPDATE run_query_state SET pages_attempted=MAX(pages_attempted,?),"
            "first_request_timestamp_utc=COALESCE(first_request_timestamp_utc,?),"
            "last_request_timestamp_utc=? WHERE run_id=? AND query_id=?",
            (page_number, timestamp, timestamp, run_id, query_id),
        )


def _retry_delay(
    response: httpx.Response | None,
    base_delay: float,
    attempt: int,
    now: datetime,
) -> float:
    configured = float(base_delay * (2.0 ** (attempt - 1)))
    if response is None:
        return configured
    value = response.headers.get("Retry-After")
    if not value:
        return configured
    try:
        retry_after = float(value)
    except ValueError:
        try:
            retry_after = float((
                parsedate_to_datetime(value).astimezone(timezone.utc)
                - now.astimezone(timezone.utc)
            ).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return configured
    return float(max(configured, retry_after, 0.0))


def _artifact_path(paths: RuntimePaths, run_id: str, query_id: str, page: int, attempt: int, status: int, now: datetime) -> Path:
    return paths.raw / "arxiv" / now.strftime("%Y") / now.strftime("%m") / now.strftime("%d") / run_id / f"{query_id}_page_{page:03d}_attempt_{attempt:03d}_status_{status:03d}.xml"


def _prepare_manifest_outcome(
    database: Path,
    run_id: str,
    status: str,
    finish: str,
    counts: dict[str, Any],
) -> dict[str, Any]:
    with connect(database) as connection:
        cutoff = connection.execute(
            "SELECT COALESCE(MAX(event_id),0) FROM run_events WHERE run_id=?",
            (run_id,),
        ).fetchone()[0]
        warning_count = connection.execute(
            "SELECT COUNT(*) FROM run_events WHERE run_id=? AND severity='WARNING'",
            (run_id,),
        ).fetchone()[0]
        error_count = connection.execute(
            "SELECT COUNT(*) FROM run_events WHERE run_id=? AND severity='ERROR'",
            (run_id,),
        ).fetchone()[0]
        connection.execute(
            "UPDATE runs SET prepared_final_status=?,run_finish_timestamp_utc=?,"
            "manifest_event_cutoff_id=?,manifest_event_cutoff_state='PROVISIONAL',"
            "manifest_publication_state='MISSING',run_counts_json=?,"
            "warning_count=?,error_count=? "
            "WHERE run_id=?",
            (
                status,
                finish,
                cutoff,
                json.dumps(counts, sort_keys=True, separators=(",", ":")),
                warning_count,
                error_count,
                run_id,
            ),
        )
        return dict(
            connection.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        )


def _record_manifest_failure(
    database: Path,
    run_id: str,
    event_code: str,
    error: Exception,
    finish: str,
) -> str:
    with connect(database) as connection:
        run = connection.execute(
            "SELECT catalog_committed,manifest_path,manifest_event_cutoff_state "
            "FROM runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if not run:
            raise RuntimeError("run state disappeared while recording manifest failure")
        if run["manifest_event_cutoff_state"] == "FROZEN":
            raise RuntimeError("cannot advance a frozen manifest event cutoff")
        event_id = append_event(
            connection,
            run_id,
            "ERROR",
            event_code,
            "MANIFEST",
            {"error_type": type(error).__name__, "message": str(error)},
        )
        prepared_status = (
            "PARTIAL_SUCCESS" if run["catalog_committed"] else "FAILED"
        )
        publication_state = (
            "INVALID" if Path(run["manifest_path"]).exists() else "MISSING"
        )
        connection.execute(
            "UPDATE runs SET prepared_final_status=?,run_finish_timestamp_utc=?,"
            "manifest_event_cutoff_id=?,manifest_event_cutoff_state='PROVISIONAL',"
            "manifest_publication_state=?,warning_count=(SELECT COUNT(*) FROM "
            "run_events WHERE run_id=? AND severity='WARNING'),error_count=(SELECT "
            "COUNT(*) FROM run_events WHERE run_id=? AND severity='ERROR') WHERE run_id=?",
            (
                prepared_status,
                finish,
                event_id,
                publication_state,
                run_id,
                run_id,
                run_id,
            ),
        )
        return prepared_status


def _attempt_manifest_publication(
    config: LoadedConfig,
    paths: RuntimePaths,
    run_id: str,
    status: str,
    finish: str,
    counts: dict[str, Any],
    raw_artifacts: list[dict[str, Any]],
    clock: Callable[[], datetime],
) -> tuple[dict[str, Any] | None, str]:
    run = _prepare_manifest_outcome(
        paths.database, run_id, status, finish, counts
    )
    stage = "construction"
    try:
        with connect(paths.database) as connection:
            manifest = build_manifest(connection, run, counts, raw_artifacts)
        stage = "validation"
        manifest_schema = config.project_root / "schemas" / "run_manifest.schema.json"
        validate_json(manifest, manifest_schema)
        stage = "publication"
        publish_manifest(Path(run["manifest_path"]), manifest, manifest_schema)
        return manifest, status
    except Exception as error:
        event_code = {
            "construction": "MANIFEST_CONSTRUCTION_FAILURE",
            "validation": "MANIFEST_VALIDATION_FAILURE",
            "publication": "MANIFEST_PUBLICATION_FAILURE",
        }[stage]
        prepared_status = _record_manifest_failure(
            paths.database, run_id, event_code, error, _iso(clock())
        )
        return None, prepared_status


def _freeze_published_manifest(
    database: Path, run_id: str, manifest: dict[str, Any]
) -> None:
    with connect(database) as connection:
        cursor = connection.execute(
            "UPDATE runs SET manifest_event_cutoff_state='FROZEN',"
            "manifest_publication_state='VALID' WHERE run_id=? "
            "AND manifest_event_cutoff_id=?",
            (run_id, manifest["manifest_event_cutoff_id"]),
        )
        if cursor.rowcount != 1:
            raise RuntimeError("published manifest cutoff no longer matches durable state")


def run_intake(
    config: LoadedConfig,
    paths: RuntimePaths,
    mode: str,
    days: int | None,
    max_items: int | None,
    client: httpx.Client | None = None,
    clock: Callable[[], datetime] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> tuple[int, str]:
    now_fn = clock or (lambda: datetime.now(timezone.utc))
    start = now_fn().astimezone(timezone.utc)
    run_id = _run_id(start)
    policy = config.registry["policy"]
    item_limit = min(max_items or policy["global_maximum_items_per_run"], policy["global_maximum_items_per_run"])
    email = contact_email()
    own_client = client is None
    http = client or httpx.Client()
    raw_artifacts: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    counts: dict[str, Any] = {"received": 0, "normalized": 0, "malformed": 0, "new": 0, "updated": 0, "unchanged": 0, "excluded": 0, "automatic_deduplications": 0, "doi_collisions": 0}
    previous_id: str | None = None
    try:
        with connect(paths.database) as connection:
            previous = latest_eligible_run(connection)
            if mode == "incremental":
                if not previous:
                    raise NoEligibleIncrementalPredecessor(
                        "no incremental predecessor is eligible; perform a bounded "
                        "backfill before running incremental mode"
                    )
                previous_id = previous["run_id"]
                interval_start = datetime.fromisoformat(previous["start_timestamp_utc"].replace("Z", "+00:00")) - timedelta(seconds=policy["overlap_seconds"])
                basis = "UPDATED_DATE"
            else:
                if not days or days <= 0:
                    raise ValueError("backfill requires --days greater than zero")
                interval_start = start - timedelta(days=days)
                basis = "SUBMITTED_DATE"
            interval_end = start
            queries = [{"source_id": source["id"], "query_id": query["id"]} for source, query in config.enabled_queries]
            manifest_path = paths.manifests / f"{run_id}.json"
            create_run(
                connection,
                {"run_id": run_id, "run_type": mode, "start_timestamp_utc": _iso(start), "interval_start_utc": _iso(interval_start), "interval_end_utc": _iso(interval_end), "discovery_time_basis": basis, "application_version": __version__, "git_commit": _git_commit(config.project_root), "python_version": platform.python_version(), "registry_hash": config.registry_hash, "topics_hash": config.topics_hash, "previous_eligible_run_id": previous_id, "overlap_seconds": policy["overlap_seconds"], "manifest_path": manifest_path.as_posix()},
                queries,
            )
        request_count = 0
        last_request_monotonic: float | None = None
        runtime_started = time.monotonic()
        for source, query in config.enabled_queries:
            query_id = query["id"]
            _update_query(paths.database, run_id, query_id, completion_status="RUNNING")
            offset = 0
            query_received = 0
            query_accepted = 0
            query_malformed = 0
            previous_ordering: datetime | None = None
            page_number = 0
            complete = False
            failed = False
            while not complete and not failed:
                if counts["received"] >= item_limit:
                    _record_query_limit(
                        paths.database,
                        run_id,
                        source["id"],
                        query_id,
                        "MAX_ITEMS_REACHED",
                        "ITEM_LIMIT_REACHED",
                        {"limit": item_limit},
                    )
                    break
                if request_count >= policy["maximum_requests_per_run"]:
                    _record_query_limit(
                        paths.database,
                        run_id,
                        source["id"],
                        query_id,
                        "MAX_REQUESTS_REACHED",
                        "REQUEST_LIMIT_REACHED",
                    )
                    break
                if time.monotonic() - runtime_started >= policy["global_maximum_runtime_seconds"]:
                    _record_query_limit(
                        paths.database,
                        run_id,
                        source["id"],
                        query_id,
                        "MAX_RUNTIME_REACHED",
                        "RUNTIME_LIMIT_REACHED",
                    )
                    break
                if query_received >= query["maximum_results"]:
                    _record_query_limit(
                        paths.database,
                        run_id,
                        source["id"],
                        query_id,
                        "MAX_QUERY_RESULTS_REACHED",
                        "PAGINATION_LIMIT_REACHED",
                    )
                    break
                page_number += 1
                page_size = min(100, item_limit - counts["received"], query["maximum_results"] - query_received)
                expression = query["expression"]
                if mode == "backfill":
                    low = interval_start.strftime("%Y%m%d%H%M")
                    high = (interval_end - timedelta(microseconds=1)).strftime("%Y%m%d%H%M")
                    expression = f"({expression}) AND submittedDate:[{low} TO {high}]"
                params: dict[str, str | int] = {"search_query": expression, "start": offset, "max_results": page_size, "sortBy": "submittedDate" if mode == "backfill" else "lastUpdatedDate", "sortOrder": "descending"}
                response: httpx.Response | None = None
                limit_stop_reason: str | None = None
                for attempt in range(1, policy["retry"]["max_attempts"] + 1):
                    if request_count >= policy["maximum_requests_per_run"]:
                        limit_stop_reason = "MAX_REQUESTS_REACHED"
                        _record_query_limit(
                            paths.database,
                            run_id,
                            source["id"],
                            query_id,
                            limit_stop_reason,
                            "REQUEST_LIMIT_REACHED",
                        )
                        break
                    elapsed = time.monotonic() - runtime_started
                    if elapsed >= policy["global_maximum_runtime_seconds"]:
                        limit_stop_reason = "MAX_RUNTIME_REACHED"
                        _record_query_limit(
                            paths.database,
                            run_id,
                            source["id"],
                            query_id,
                            limit_stop_reason,
                            "RUNTIME_LIMIT_REACHED",
                        )
                        break
                    if last_request_monotonic is not None:
                        wait = policy["minimum_request_interval_seconds"] - (time.monotonic() - last_request_monotonic)
                        if wait > 0:
                            if elapsed + wait >= policy["global_maximum_runtime_seconds"]:
                                limit_stop_reason = "MAX_RUNTIME_REACHED"
                                _record_query_limit(
                                    paths.database,
                                    run_id,
                                    source["id"],
                                    query_id,
                                    limit_stop_reason,
                                    "RUNTIME_LIMIT_REACHED",
                                )
                                break
                            sleeper(wait)
                    request_count += 1
                    last_request_monotonic = time.monotonic()
                    requested_at = _iso(now_fn())
                    _record_request_attempt(
                        paths.database,
                        run_id,
                        query_id,
                        page_number,
                        requested_at,
                    )
                    remaining_runtime = max(
                        0.001,
                        policy["global_maximum_runtime_seconds"]
                        - (time.monotonic() - runtime_started),
                    )
                    try:
                        response = request_page(
                            http,
                            source["endpoint"],
                            params,
                            {
                                "User-Agent": safe_user_agent(
                                    config.registry["application"]["user_agent"],
                                    email,
                                ),
                                "Accept-Encoding": "identity",
                            },
                            min(
                                policy["request_timeout_seconds"], remaining_runtime
                            ),
                        )
                    except httpx.HTTPError as error:
                        if time.monotonic() - runtime_started >= policy[
                            "global_maximum_runtime_seconds"
                        ]:
                            limit_stop_reason = "MAX_RUNTIME_REACHED"
                            _record_query_limit(
                                paths.database,
                                run_id,
                                source["id"],
                                query_id,
                                limit_stop_reason,
                                "RUNTIME_LIMIT_REACHED",
                                {
                                    "error_type": type(error).__name__,
                                    "request_attempted": True,
                                },
                            )
                            break
                        _small_event(paths.database, run_id, "WARNING" if attempt < policy["retry"]["max_attempts"] else "ERROR", "REQUEST_RETRY" if attempt < policy["retry"]["max_attempts"] else "QUERY_FAILURE", "RETRIEVAL", {"attempt": attempt, "error_type": type(error).__name__}, source["id"], query_id)
                        if attempt < policy["retry"]["max_attempts"]:
                            delay = _retry_delay(
                                None,
                                policy["retry"]["base_delay_seconds"],
                                attempt,
                                now_fn(),
                            )
                            if time.monotonic() - runtime_started + delay >= policy["global_maximum_runtime_seconds"]:
                                limit_stop_reason = "MAX_RUNTIME_REACHED"
                                _record_query_limit(
                                    paths.database,
                                    run_id,
                                    source["id"],
                                    query_id,
                                    limit_stop_reason,
                                    "RUNTIME_LIMIT_REACHED",
                                )
                                break
                            sleeper(delay)
                            continue
                        failed = True
                        break
                    body_path = _artifact_path(paths, run_id, query_id, page_number, attempt, response.status_code, start)
                    provenance = {"schema_version": 1, "source_id": source["id"], "query_id": query_id, "request": {"method": "GET", "url": str(response.request.url), "parameters": params}, "response": {"status": response.status_code, "headers": selected_headers(response.headers)}, "retrieved_at": requested_at, "attempt_number": attempt}
                    digest, sidecar = preserve_response(response.content, body_path, provenance, config.project_root / "schemas" / "raw_response_provenance.schema.json")
                    artifact = {**provenance, "body_path": body_path.as_posix(), "body_sha256": digest, "provenance_path": sidecar.as_posix()}
                    raw_artifacts.append(artifact)
                    if response.status_code in policy["retry"]["retry_statuses"] and attempt < policy["retry"]["max_attempts"]:
                        _small_event(paths.database, run_id, "WARNING", "REQUEST_RETRY", "RETRIEVAL", {"attempt": attempt, "status": response.status_code}, source["id"], query_id)
                        delay = _retry_delay(
                            response,
                            policy["retry"]["base_delay_seconds"],
                            attempt,
                            now_fn(),
                        )
                        if time.monotonic() - runtime_started + delay >= policy["global_maximum_runtime_seconds"]:
                            limit_stop_reason = "MAX_RUNTIME_REACHED"
                            _record_query_limit(
                                paths.database,
                                run_id,
                                source["id"],
                                query_id,
                                limit_stop_reason,
                                "RUNTIME_LIMIT_REACHED",
                            )
                            break
                        sleeper(delay)
                        continue
                    if response.status_code != 200:
                        _small_event(paths.database, run_id, "ERROR", "QUERY_FAILURE", "RETRIEVAL", {"status": response.status_code}, source["id"], query_id)
                        failed = True
                    break
                if limit_stop_reason is not None:
                    break
                if failed or response is None:
                    _update_query(paths.database, run_id, query_id, completion_status="FAILED", stopped_early=1, final_stop_reason="QUERY_FAILURE")
                    break
                current_artifact = raw_artifacts[-1]
                try:
                    saved_body = Path(current_artifact["body_path"]).read_bytes()
                    page = parse_atom(saved_body)
                except MalformedPage as error:
                    _small_event(paths.database, run_id, "ERROR", "MALFORMED_PAGE", "PARSING", {"page": page_number, "message": str(error)}, source["id"], query_id)
                    _small_event(
                        paths.database,
                        run_id,
                        "ERROR",
                        "PAGE_FAILURE",
                        "PARSING",
                        {"page": page_number, "reason": "STRUCTURALLY_UNUSABLE"},
                        source["id"],
                        query_id,
                    )
                    _update_query(paths.database, run_id, query_id, completion_status="FAILED", stopped_early=1, final_stop_reason="PAGE_FAILURE")
                    failed = True
                    break
                counts["received"] += len(page.entries) + len(page.malformed_entries)
                counts["malformed"] += len(page.malformed_entries)
                query_received += len(page.entries) + len(page.malformed_entries)
                query_malformed += len(page.malformed_entries)
                for warning in page.malformed_entries:
                    _small_event(paths.database, run_id, "WARNING", "MALFORMED_ENTRY", "PARSING", warning, source["id"], query_id)
                ordering_values = [
                    parse_timestamp(
                        entry["published"] if mode == "backfill" else entry["updated"]
                    )
                    for entry in page.entries
                ]
                ordering_violation = any(
                    previous is not None and current > previous
                    for previous, current in zip(
                        (
                            [previous_ordering, *ordering_values[:-1]]
                            if ordering_values
                            else []
                        ),
                        ordering_values,
                        strict=True,
                    )
                )
                if ordering_violation:
                    _small_event(
                        paths.database,
                        run_id,
                        "ERROR",
                        "PAGE_FAILURE",
                        "PARSING",
                        {"page": page_number, "reason": "ORDERING_VIOLATION"},
                        source["id"],
                        query_id,
                    )
                    _update_query(
                        paths.database,
                        run_id,
                        query_id,
                        completion_status="FAILED",
                        stopped_early=1,
                        final_stop_reason="PAGE_FAILURE",
                        received_count=query_received,
                        malformed_entry_count=query_malformed,
                    )
                    failed = True
                    break
                if ordering_values:
                    previous_ordering = ordering_values[-1]
                crossed = False
                for entry_index, (entry, ordering) in enumerate(
                    zip(page.entries, ordering_values, strict=True)
                ):
                    if ordering < interval_start:
                        crossed = True
                        continue
                    if ordering >= interval_end:
                        continue
                    try:
                        relevance = score_item(
                            entry, config.topics, query["eligibility_mode"]
                        )
                        item = normalize_observation(
                            entry,
                            run_id,
                            [query_id],
                            current_artifact["retrieved_at"],
                            current_artifact["body_path"],
                            current_artifact["body_sha256"],
                            relevance,
                        )
                        validate_json(
                            item,
                            config.project_root
                            / "schemas"
                            / "normalized_item.schema.json",
                        )
                    except (ValueError, ValidationError) as error:
                        counts["malformed"] += 1
                        query_malformed += 1
                        _small_event(
                            paths.database,
                            run_id,
                            "WARNING",
                            "MALFORMED_ENTRY",
                            "NORMALIZATION",
                            {
                                "entry_index": entry_index,
                                "error_type": type(error).__name__,
                                "message": str(error),
                            },
                            source["id"],
                            query_id,
                        )
                        continue
                    observations.append({"item": item, "query_id": query_id, "raw_path": current_artifact["body_path"], "raw_hash": current_artifact["body_sha256"], "source_fingerprint": source_fingerprint(entry), "entry_index": entry_index})
                    query_accepted += 1
                    counts["normalized"] += 1
                returned_count = len(page.entries) + len(page.malformed_entries)
                exhausted = (
                    offset + returned_count >= page.total_results
                    or returned_count == 0
                )
                _update_query(paths.database, run_id, query_id, pages_succeeded=page_number, received_count=query_received, accepted_count=query_accepted, malformed_entry_count=query_malformed, exhausted_available_results=int(exhausted), crossed_lower_boundary=int(crossed), last_ordering_timestamp_utc=_iso(min(ordering_values)) if ordering_values else None)
                if crossed or exhausted:
                    stop_reason = "LOWER_BOUNDARY_CROSSED" if crossed else "AVAILABLE_RESULTS_EXHAUSTED"
                    _update_query(paths.database, run_id, query_id, completion_status="COMPLETE", final_stop_reason=stop_reason)
                    complete = True
                else:
                    offset += returned_count
        with connect(paths.database) as connection:
            completed_states = query_states(connection, run_id)
        if not any(state["pages_succeeded"] for state in completed_states) and any(
            state["completion_status"] in {"FAILED", "INCOMPLETE"}
            for state in completed_states
        ):
            raise CompleteSourceFailure("no usable source page was available")
        discovery_queries_by_item: dict[str, set[str]] = {}
        relevance_by_item: dict[str, dict[str, Any]] = {}
        for observation in observations:
            observation_item_id = observation["item"]["identity"]["item_id"]
            discovery_queries_by_item.setdefault(observation_item_id, set()).add(
                observation["query_id"]
            )
            candidate = observation["item"]["relevance"]
            current = relevance_by_item.get(observation_item_id)
            if current is None or (
                int(candidate["eligible"]), candidate["score"]
            ) > (int(current["eligible"]), current["score"]):
                relevance_by_item[observation_item_id] = candidate
        for observation in observations:
            observation_item_id = observation["item"]["identity"]["item_id"]
            observation["item"]["provenance"]["discovery_query_ids"] = sorted(
                discovery_queries_by_item[observation_item_id]
            )
            observation["item"]["relevance"] = copy.deepcopy(
                relevance_by_item[observation_item_id]
            )
            observation["item"]["review"]["priority"] = (
                relevance_by_item[observation_item_id]["priority"]
                if relevance_by_item[observation_item_id]["eligible"]
                else "NONE"
            )
        unique_ids = set(discovery_queries_by_item)
        counts["automatic_deduplications"] = max(0, len(observations) - len(unique_ids))
        queue_timestamp = _iso(now_fn())
        markdown_path = paths.markdown / f"{run_id}.md"
        jsonl_path = paths.jsonl / f"{run_id}.jsonl"
        with connect(paths.database) as connection:
            batch_counts = ingest_batch(
                connection,
                run_id,
                observations,
                queue_timestamp,
                RENDERER_VERSION,
                markdown_path.as_posix(),
                jsonl_path.as_posix(),
                config.project_root / "schemas" / "normalized_item.schema.json",
            )
        counts.update(batch_counts)
        status = "PARTIAL_SUCCESS" if counts["malformed"] else "SUCCESS"
        with connect(paths.database) as connection:
            states = query_states(connection, run_id)
            if any(state["completion_status"] != "COMPLETE" for state in states):
                status = (
                    "PARTIAL_SUCCESS"
                    if any(state["pages_succeeded"] for state in states)
                    else "FAILED"
                )
        try:
            with connect(paths.database) as connection:
                queue_items = discovery_queue_items(connection, run_id)
                markdown, jsonl = render_queue_bytes(queue_items, run_id, queue_timestamp, config.project_root / "schemas" / "review_queue_item.schema.json")
            artifacts = publish_queues(markdown_path, jsonl_path, markdown, jsonl)
            with connect(paths.database) as connection:
                connection.execute("UPDATE runs SET markdown_status='CREATED',markdown_count=?,markdown_size=?,markdown_sha256=?,jsonl_status='CREATED',jsonl_count=?,jsonl_size=?,jsonl_sha256=? WHERE run_id=?", (len(queue_items), artifacts["markdown"]["size"], artifacts["markdown"]["sha256"], len(queue_items), artifacts["jsonl"]["size"], artifacts["jsonl"]["sha256"], run_id))
        except Exception as error:
            status = "PARTIAL_SUCCESS"
            _small_event(paths.database, run_id, "ERROR", "QUEUE_GENERATION_FAILURE", "QUEUES", {"error_type": type(error).__name__, "message": str(error)})
            with connect(paths.database) as connection:
                connection.execute("UPDATE runs SET markdown_status=COALESCE(markdown_status,'FAILED'),jsonl_status=COALESCE(jsonl_status,'FAILED') WHERE run_id=?", (run_id,))
        manifest, status = _attempt_manifest_publication(
            config,
            paths,
            run_id,
            status,
            _iso(now_fn()),
            counts,
            raw_artifacts,
            now_fn,
        )
        if manifest is None:
            return 3 if status == "PARTIAL_SUCCESS" else 1, run_id
        eligible = bool(manifest["incremental_predecessor_eligible"])
        reason = manifest["incremental_predecessor_reason_code"]
        try:
            _freeze_published_manifest(paths.database, run_id, manifest)
            with connect(paths.database) as connection:
                connection.execute(
                    "UPDATE runs SET status=?,incremental_predecessor_eligible=?,"
                    "incremental_predecessor_reason_code=? WHERE run_id=?",
                    (status, int(eligible), reason, run_id),
                )
        except (sqlite3.Error, RuntimeError) as error:
            _small_event(paths.database, run_id, "ERROR", "RUN_FINALIZATION_FAILURE", "FINALIZATION", {"message": str(error)})
            return 1, run_id
        return (0 if status == "SUCCESS" else 3 if status == "PARTIAL_SUCCESS" else 1), run_id
    except NoEligibleIncrementalPredecessor:
        raise
    except Exception as error:
        try:
            with connect(paths.database) as connection:
                exists = connection.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if exists and not isinstance(error, CompleteSourceFailure):
                    event_code = (
                        "EVIDENCE_HASH_MISMATCH"
                        if str(error) == "EVIDENCE_HASH_MISMATCH"
                        else "CATALOG_TRANSACTION_FAILURE"
                    )
                    append_event(
                        connection,
                        run_id,
                        "ERROR",
                        event_code,
                        "PIPELINE",
                        {"error_type": type(error).__name__, "message": str(error)},
                    )
            failed_manifest, _ = _attempt_manifest_publication(
                config,
                paths,
                run_id,
                "FAILED",
                _iso(now_fn()),
                counts,
                raw_artifacts,
                now_fn,
            )
            if failed_manifest is not None:
                _freeze_published_manifest(paths.database, run_id, failed_manifest)
                with connect(paths.database) as connection:
                    connection.execute(
                        "UPDATE runs SET status='FAILED',"
                        "incremental_predecessor_eligible=?,"
                        "incremental_predecessor_reason_code=? WHERE run_id=?",
                        (
                            int(
                                failed_manifest[
                                    "incremental_predecessor_eligible"
                                ]
                            ),
                            failed_manifest[
                                "incremental_predecessor_reason_code"
                            ],
                            run_id,
                        ),
                    )
        except Exception:
            pass
        return 1, run_id
    finally:
        if own_client:
            http.close()
