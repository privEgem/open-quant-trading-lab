from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from research_intake.catalog import (
    connect,
    discovery_queue_items,
    event_records,
    predecessor_decision,
    query_states,
)
from research_intake.evidence import atomic_json, atomic_write, sha256_file
from research_intake.manifests import (
    build_manifest,
    publish_manifest,
    source_request_statistics,
)
from research_intake.paths import RuntimePaths
from research_intake.queues import RENDERER_VERSION, render_queue_bytes
from research_intake.validation import validate_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_raw_artifacts(paths: RuntimePaths, run_id: str, schema: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sidecar in sorted(paths.raw.glob(f"arxiv/*/*/*/{run_id}/*.provenance.json")):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        validate_json(record, schema)
        body = Path(record["body_path"]).resolve()
        if not body.is_relative_to(paths.raw.resolve()):
            raise RuntimeError("raw evidence path escapes the configured data root")
        if sidecar.resolve() != body.with_name(
            body.name + ".provenance.json"
        ).resolve():
            raise RuntimeError("raw evidence sidecar does not match its body path")
        if not body.exists() or sha256_file(body) != record["body_sha256"]:
            raise RuntimeError(f"evidence mismatch: {body}")
        records.append({**record, "provenance_path": sidecar.as_posix()})
    return records


def _durable_counts(
    connection: Any, run: dict[str, Any], states: list[dict[str, Any]]
) -> dict[str, int]:
    if not run.get("run_counts_json"):
        raise RuntimeError("missing durable original run counts")
    counts = cast(dict[str, int], json.loads(run["run_counts_json"]))
    required = {
        "received",
        "normalized",
        "malformed",
        "new",
        "updated",
        "unchanged",
        "excluded",
        "automatic_deduplications",
        "doi_collisions",
    }
    if set(counts) != required or any(
        not isinstance(counts[key], int) or counts[key] < 0 for key in required
    ):
        raise RuntimeError("durable original run counts are invalid")
    run_id = run["run_id"]
    discovery_count = connection.execute(
        "SELECT COUNT(*) FROM discoveries WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    distinct_items = connection.execute(
        "SELECT COUNT(DISTINCT item_id) FROM discoveries WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    if counts["received"] != sum(state["received_count"] for state in states):
        raise RuntimeError("durable received count conflicts with query state")
    if counts["malformed"] != sum(
        state["malformed_entry_count"] for state in states
    ):
        raise RuntimeError("durable malformed count conflicts with query state")
    if counts["normalized"] < discovery_count:
        raise RuntimeError("durable normalized count is below discovery rows")
    if run["catalog_committed"]:
        if sum(
            counts[key] for key in ("new", "updated", "unchanged", "excluded")
        ) != counts["normalized"]:
            raise RuntimeError("durable dispositions do not cover normalized observations")
        if counts["automatic_deduplications"] != max(
            0, counts["normalized"] - distinct_items
        ):
            raise RuntimeError("durable deduplication count is inconsistent")
        collision_events = connection.execute(
            "SELECT COUNT(*) FROM run_events WHERE run_id=? "
            "AND event_code='DOI_COLLISION'",
            (run_id,),
        ).fetchone()[0]
        if counts["doi_collisions"] != collision_events:
            raise RuntimeError("durable DOI collision count is inconsistent")
    return counts


def _validate_manifest(
    manifest: dict[str, Any],
    run: dict[str, Any],
    schema: Path,
    connection: Any,
    states: list[dict[str, Any]],
    counts: dict[str, int],
    raw_artifacts: list[dict[str, Any]],
) -> None:
    validate_json(manifest, schema)
    if manifest["run_id"] != run["run_id"] or manifest["configuration_hashes"] != {"source_registry": run["registry_hash"], "topics": run["topics_hash"]}:
        raise RuntimeError("manifest conflicts with durable run identity or configuration hashes")
    manifest_cutoff = int(manifest["manifest_event_cutoff_id"])
    durable_cutoff = int(run["manifest_event_cutoff_id"] or 0)
    if run["manifest_event_cutoff_state"] == "FROZEN":
        if manifest_cutoff != durable_cutoff:
            raise RuntimeError("manifest conflicts with frozen event cutoff")
    elif manifest_cutoff > durable_cutoff:
        raise RuntimeError("manifest cutoff exceeds durable provisional cutoff")
    expected_identity = {
        "run_type": run["run_type"],
        "start_timestamp_utc": run["start_timestamp_utc"],
        "finish_timestamp_utc": run["run_finish_timestamp_utc"],
        "final_status": run["prepared_final_status"],
    }
    for field, expected in expected_identity.items():
        if manifest[field] != expected:
            raise RuntimeError(f"manifest conflicts with durable {field}")
    discovery = manifest["discovery"]
    if (
        discovery["time_basis"] != run["discovery_time_basis"]
        or discovery["interval_start_inclusive"] != run["interval_start_utc"]
        or discovery["interval_end_exclusive"] != run["interval_end_utc"]
        or discovery["overlap_seconds"] != run["overlap_seconds"]
        or discovery["queries"] != states
    ):
        raise RuntimeError("manifest conflicts with durable discovery state")
    if manifest["counts"] != counts:
        raise RuntimeError("manifest conflicts with durable run counts")
    if sorted(manifest["raw_artifacts"], key=lambda item: item["body_path"]) != sorted(
        raw_artifacts, key=lambda item: item["body_path"]
    ):
        raise RuntimeError("manifest conflicts with preserved raw evidence")
    queues = manifest["queues"]
    if (
        queues["renderer_version"] != run["queue_renderer_version"]
        or queues["render_timestamp_utc"] != run["queue_render_timestamp_utc"]
        or queues["markdown"]["path"] != run["markdown_path"]
        or queues["markdown"]["status"] != run["markdown_status"]
        or queues["markdown"]["count"] != run["markdown_count"]
        or queues["markdown"]["sha256"] != run["markdown_sha256"]
        or queues["jsonl"]["path"] != run["jsonl_path"]
        or queues["jsonl"]["status"] != run["jsonl_status"]
        or queues["jsonl"]["count"] != run["jsonl_count"]
        or queues["jsonl"]["sha256"] != run["jsonl_sha256"]
    ):
        raise RuntimeError("manifest conflicts with durable queue state")
    expected_events = event_records(connection, run["run_id"], manifest_cutoff)
    if manifest["source_requests"] != source_request_statistics(
        states, raw_artifacts, expected_events
    ):
        raise RuntimeError("manifest conflicts with durable source request statistics")
    if manifest["warnings"] != [
        event for event in expected_events if event["severity"] == "WARNING"
    ] or manifest["errors"] != [
        event for event in expected_events if event["severity"] == "ERROR"
    ]:
        raise RuntimeError("manifest conflicts with durable event history")
    eligible, reason = predecessor_decision(connection, run["run_id"])
    if (
        manifest["incremental_predecessor_eligible"] is not eligible
        or manifest["incremental_predecessor_reason_code"] != reason
    ):
        raise RuntimeError("manifest conflicts with predecessor eligibility state")


def recover_run(project_root: Path, paths: RuntimePaths, run_id: str) -> tuple[int, Path]:
    started = _now()
    recovery_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex
    receipt_path = paths.recovery_receipts / f"{recovery_id}.json"
    classification = "MISSING_NOT_RECONSTRUCTABLE"
    actions: list[str] = []
    validated: list[dict[str, Any]] = []
    created: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    reconciled = False
    result = "FAILED"
    try:
        with connect(paths.database) as connection:
            row = connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                raise RuntimeError("target run does not exist")
            run = dict(row)
            states = query_states(connection, run_id)
            if not states:
                raise RuntimeError("missing durable query state")
            if (
                run["catalog_committed"]
                and run["queue_renderer_version"] != RENDERER_VERSION
            ):
                raise RuntimeError("UNSUPPORTED_QUEUE_RENDERER")
            manifest_path = Path(run["manifest_path"])
            expected_manifest_path = paths.manifests / f"{run_id}.json"
            if manifest_path.resolve() != expected_manifest_path.resolve():
                raise RuntimeError("manifest path escapes the run's expected output path")
            if run["catalog_committed"] and (
                Path(run["markdown_path"]).resolve()
                != (paths.markdown / f"{run_id}.md").resolve()
                or Path(run["jsonl_path"]).resolve()
                != (paths.jsonl / f"{run_id}.jsonl").resolve()
            ):
                raise RuntimeError("queue path escapes the run's expected output path")
            if manifest_path.exists():
                classification = "PRESENT_CONFLICTING_OR_INVALID"
            manifest_schema = project_root / "schemas" / "run_manifest.schema.json"
            raw = _load_raw_artifacts(
                paths,
                run_id,
                project_root / "schemas" / "raw_response_provenance.schema.json",
            )
            counts = _durable_counts(connection, run, states)
            manifest: dict[str, Any]
            if manifest_path.exists():
                actions.append("validate existing manifest")
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    _validate_manifest(
                        manifest,
                        run,
                        manifest_schema,
                        connection,
                        states,
                        counts,
                        raw,
                    )
                except Exception:
                    classification = "PRESENT_CONFLICTING_OR_INVALID"
                    raise
                classification = "ALREADY_VALID"
                validated.append({"type": "manifest", "path": manifest_path.as_posix(), "sha256": sha256_file(manifest_path)})
            else:
                actions.append("reconstruct missing manifest")
                required = ["prepared_final_status", "run_finish_timestamp_utc", "queue_render_timestamp_utc", "queue_renderer_version", "manifest_event_cutoff_id"]
                missing = [field for field in required if run.get(field) is None]
                if missing:
                    raise RuntimeError(f"missing durable manifest facts: {', '.join(missing)}")
                if run["catalog_committed"] and not raw:
                    raise RuntimeError("no validated raw response provenance is available")
                manifest = build_manifest(connection, run, counts, raw, mode="RECOVERED")
                _validate_manifest(
                    manifest,
                    run,
                    manifest_schema,
                    connection,
                    states,
                    counts,
                    raw,
                )
                publish_manifest(manifest_path, manifest, manifest_schema)
                classification = "NEWLY_RECONSTRUCTED"
                created.append({"type": "manifest", "path": manifest_path.as_posix(), "sha256": sha256_file(manifest_path)})
            connection.execute(
                "UPDATE runs SET manifest_event_cutoff_id=?,"
                "manifest_event_cutoff_state='FROZEN',"
                "manifest_publication_state='VALID' WHERE run_id=?",
                (manifest["manifest_event_cutoff_id"], run_id),
            )
            manifest_cutoff = int(manifest["manifest_event_cutoff_id"])
            post_manifest_events = [
                event
                for event in event_records(connection, run_id)
                if int(event["event_id"]) > manifest_cutoff
            ]
            if run["catalog_committed"]:
                actions.append("validate or create review queues")
                items = discovery_queue_items(connection, run_id)
                markdown, jsonl = render_queue_bytes(items, run_id, run["queue_render_timestamp_utc"], project_root / "schemas" / "review_queue_item.schema.json")
                for kind, path_text, expected in (("markdown", run["markdown_path"], markdown), ("jsonl", run["jsonl_path"], jsonl)):
                    path = Path(path_text)
                    digest = hashlib.sha256(expected).hexdigest()
                    if path.exists():
                        if path.read_bytes() != expected:
                            raise RuntimeError(f"conflicting existing {kind} queue")
                        validated.append({"type": kind, "path": path.as_posix(), "sha256": digest})
                    else:
                        atomic_write(path, expected)
                        created.append({"type": kind, "path": path.as_posix(), "sha256": digest})
            else:
                actions.append("skip queues because catalog transaction did not commit")
            if run["status"] == "RUNNING":
                actions.append("reconcile stale run row")
                connection.execute(
                    "UPDATE runs SET status=?,incremental_predecessor_eligible=?,"
                    "incremental_predecessor_reason_code=? WHERE run_id=?",
                    (manifest["final_status"], int(manifest["incremental_predecessor_eligible"]), manifest["incremental_predecessor_reason_code"], run_id),
                )
                reconciled = True
            result = "SUCCESS"
    except Exception as error:
        errors.append({"code": type(error).__name__, "message": str(error)})
    receipt = {
        "schema_version": 1, "recovery_id": recovery_id, "target_run_id": run_id,
        "start_timestamp_utc": started, "finish_timestamp_utc": _now(),
        "actions_attempted": actions, "artifacts_validated": validated, "artifacts_created": created,
        "database_reconciliation_performed": reconciled, "manifest_classification": classification,
        "post_manifest_events": post_manifest_events if "post_manifest_events" in locals() else [],
        "warnings": warnings, "errors": errors, "final_recovery_status": result,
    }
    atomic_json(receipt_path, receipt, project_root / "schemas" / "recovery_receipt.schema.json")
    return (0 if result == "SUCCESS" else 1), receipt_path
