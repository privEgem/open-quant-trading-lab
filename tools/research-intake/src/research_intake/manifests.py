from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research_intake.catalog import event_records, predecessor_decision, query_states
from research_intake.evidence import atomic_json


def source_request_statistics(
    states: list[dict[str, Any]],
    raw_artifacts: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    statistics: dict[str, Any] = {}
    for state in states:
        query_id = state["query_id"]
        responses = [
            artifact
            for artifact in raw_artifacts
            if artifact["query_id"] == query_id
        ]
        transport_failures = [
            event
            for event in events
            if event["query_id"] == query_id
            and (
                (
                    event["event_code"] in {"REQUEST_RETRY", "QUERY_FAILURE"}
                    and "error_type" in event["context"]
                )
                or event["context"].get("request_attempted") is True
            )
        ]
        status_counts: dict[str, int] = {}
        for response in responses:
            status = str(response["response"]["status"])
            status_counts[status] = status_counts.get(status, 0) + 1
        statistics[query_id] = {
            "attempts": len(responses) + len(transport_failures),
            "responses": len(responses),
            "transport_failures": len(transport_failures),
            "retry_events": sum(
                event["query_id"] == query_id
                and event["event_code"] == "REQUEST_RETRY"
                for event in events
            ),
            "http_status_counts": status_counts,
            "pages_attempted": state["pages_attempted"],
            "pages_succeeded": state["pages_succeeded"],
            "received": state["received_count"],
            "final_stop_reason": state["final_stop_reason"],
        }
    return statistics


def build_manifest(connection: Any, run: dict[str, Any], counts: dict[str, Any], raw_artifacts: list[dict[str, Any]], mode: str = "ORIGINAL") -> dict[str, Any]:
    cutoff = int(run["manifest_event_cutoff_id"] or 0)
    events = event_records(connection, run["run_id"], cutoff)
    warnings = [event for event in events if event["severity"] == "WARNING"]
    errors = [event for event in events if event["severity"] == "ERROR"]
    eligible, reason = predecessor_decision(connection, run["run_id"])
    start = datetime.fromisoformat(run["start_timestamp_utc"].replace("Z", "+00:00"))
    finish = datetime.fromisoformat(run["run_finish_timestamp_utc"].replace("Z", "+00:00"))
    states = query_states(connection, run["run_id"])
    return {
        "schema_version": 1,
        "run_id": run["run_id"], "run_type": run["run_type"],
        "start_timestamp_utc": run["start_timestamp_utc"], "finish_timestamp_utc": run["run_finish_timestamp_utc"],
        "duration_seconds": max(0.0, (finish - start).total_seconds()), "application_version": run["application_version"],
        "git_commit": run["git_commit"], "python_version": run["python_version"],
        "configuration_hashes": {"source_registry": run["registry_hash"], "topics": run["topics_hash"]},
        "previous_eligible_run_id": run["previous_eligible_run_id"],
        "discovery": {"time_basis": run["discovery_time_basis"], "interval_start_inclusive": run["interval_start_utc"], "interval_end_exclusive": run["interval_end_utc"], "overlap_seconds": run["overlap_seconds"], "queries": states},
        "source_requests": source_request_statistics(states, raw_artifacts, events),
        "raw_artifacts": raw_artifacts, "counts": counts,
        "queues": {"renderer_version": run["queue_renderer_version"], "render_timestamp_utc": run["queue_render_timestamp_utc"], "markdown": {"path": run["markdown_path"], "status": run["markdown_status"], "count": run["markdown_count"], "size": run["markdown_size"], "sha256": run["markdown_sha256"]}, "jsonl": {"path": run["jsonl_path"], "status": run["jsonl_status"], "count": run["jsonl_count"], "size": run["jsonl_size"], "sha256": run["jsonl_sha256"]}},
        "warnings": warnings, "errors": errors, "final_status": run["prepared_final_status"],
        "incremental_predecessor_eligible": eligible, "incremental_predecessor_reason_code": reason,
        "manifest_event_cutoff_id": cutoff, "manifest_generation_mode": mode,
        "manifest_event_cutoff_state": "FROZEN",
        "manifest_published_at_utc": (
            run["run_finish_timestamp_utc"]
            if mode == "ORIGINAL"
            else datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        ),
    }


def publish_manifest(path: Path, manifest: dict[str, Any], schema_path: Path) -> None:
    atomic_json(path, manifest, schema_path)
