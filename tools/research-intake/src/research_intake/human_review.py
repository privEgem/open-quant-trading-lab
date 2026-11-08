from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from jsonschema.exceptions import ValidationError

from research_intake.catalog import connect, utc_now
from research_intake.validation import validate_json

HUMAN_STATUSES = {
    "NEW",
    "REVIEWING",
    "KEEP_FOR_DEEP_REVIEW",
    "EDUCATIONAL_ONLY",
    "PARKED",
    "REJECTED",
    "DUPLICATE",
}
RECORDED_STATUSES = HUMAN_STATUSES - {"NEW"}
REVIEW_BASES = {"ABSTRACT_TRIAGE", "FULL_TEXT_REVIEW"}
CONFIDENCE_LEVELS = {"LOW", "MEDIUM", "HIGH"}
PLACEHOLDERS = {"-", "n/a", "na", "none", "placeholder", "tbd", "todo", "unknown"}
TAG_PATTERN = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
EXPORT_FILENAMES = (
    "index.jsonl",
    "index.md",
    "deep-review-queue.md",
    "methodology-watchlist.md",
)


class ReviewValidationError(ValueError):
    """Raised when a human-review batch cannot be safely recorded."""


@dataclass(frozen=True)
class QueueEvidence:
    path_text: str
    sha256: str
    items: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class PreparedDecision:
    decision_id: str
    batch_id: str
    payload: dict[str, Any]
    source_queue_path: str
    previous_status: str
    previous_reviewer: str | None
    previous_reviewed_at: str | None
    previous_tags: list[str]
    previous_notes: str | None
    canonical_item: dict[str, Any] | None
    already_recorded: bool


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def decision_id_for(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def load_review_batch(input_path: Path, schema_path: Path) -> dict[str, Any]:
    try:
        value = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReviewValidationError(f"cannot read review batch: {error}") from error
    if not isinstance(value, dict):
        raise ReviewValidationError("review batch must be a JSON object")
    batch = cast(dict[str, Any], value)
    _validate_batch_schema(batch, schema_path)
    _validate_batch_semantics(batch)
    return batch


def _validate_batch_schema(batch: dict[str, Any], schema_path: Path) -> None:
    try:
        validate_json(batch, schema_path)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as error:
        raise ReviewValidationError(f"invalid review batch schema: {error}") from error


def _validate_batch_semantics(batch: dict[str, Any]) -> None:
    for field in ("batch_id", "reviewer"):
        _require_trimmed(cast(str, batch[field]), field)
    seen_items: set[str] = set()
    decisions = cast(list[dict[str, Any]], batch["decisions"])
    for position, decision in enumerate(decisions):
        prefix = f"decisions[{position}]"
        for field in (
            "item_id",
            "source_run_id",
            "expected_current_status",
            "status",
            "review_basis",
            "confidence",
            "rationale",
            "next_action",
            "reviewed_at",
            "source_queue_sha256",
        ):
            _require_trimmed(cast(str, decision[field]), f"{prefix}.{field}")
        if "item_version" in decision:
            _require_trimmed(
                cast(str, decision["item_version"]), f"{prefix}.item_version"
            )
        item_id = cast(str, decision["item_id"])
        if item_id in seen_items:
            raise ReviewValidationError(f"duplicate item in review batch: {item_id}")
        seen_items.add(item_id)
        _parse_utc_timestamp(
            cast(str, decision["reviewed_at"]), f"{prefix}.reviewed_at"
        )
        rationale = cast(str, decision["rationale"])
        if _placeholder(rationale):
            raise ReviewValidationError(f"{prefix}.rationale is not substantive")
        next_action = cast(str, decision["next_action"])
        if _placeholder(next_action):
            raise ReviewValidationError(f"{prefix}.next_action is not explicit")
        tags = cast(list[str], decision["tags"])
        if len(tags) != len(set(tags)):
            raise ReviewValidationError(f"{prefix}.tags must be unique")
        for tag in tags:
            _require_trimmed(tag, f"{prefix}.tags")
            if tag != tag.casefold() or not TAG_PATTERN.fullmatch(tag):
                raise ReviewValidationError(
                    f"{prefix}.tags contains a non-normalized tag: {tag}"
                )


def _require_trimmed(value: str, field: str) -> None:
    if not value or value != value.strip():
        raise ReviewValidationError(f"{field} must be nonempty and trimmed")


def _placeholder(value: str) -> bool:
    compact = " ".join(value.split()).casefold()
    return compact in PLACEHOLDERS


def _parse_utc_timestamp(value: str, field: str) -> datetime:
    if not value.endswith("Z"):
        raise ReviewValidationError(f"{field} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ReviewValidationError(f"{field} is not a valid date-time") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ReviewValidationError(f"{field} must be UTC")
    return parsed


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _json_string_list(value: str, field: str) -> list[str]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ReviewValidationError(f"{field} is not valid JSON") from error
    if not isinstance(decoded, list) or not all(
        isinstance(element, str) for element in decoded
    ):
        raise ReviewValidationError(f"{field} must contain a JSON string array")
    return cast(list[str], decoded)


def _canonical_tags(tags: list[str]) -> list[str]:
    return sorted(tags)


def _load_queue(
    run_id: str,
    path_text: str,
    expected_hash: str,
    data_root: Path,
) -> QueueEvidence:
    queue_path = Path(path_text)
    if not queue_path.is_absolute():
        queue_path = data_root / queue_path
    queue_path = queue_path.resolve()
    allowed_root = (data_root / "review" / "jsonl").resolve()
    try:
        queue_path.relative_to(allowed_root)
    except ValueError as error:
        raise ReviewValidationError(
            f"source run {run_id} JSONL path is outside the queue root"
        ) from error
    if not queue_path.is_file():
        raise ReviewValidationError(f"source run {run_id} JSONL queue is missing")
    try:
        queue_bytes = queue_path.read_bytes()
    except OSError as error:
        raise ReviewValidationError(
            f"cannot read source run {run_id} JSONL queue: {error}"
        ) from error
    actual_hash = hashlib.sha256(queue_bytes).hexdigest()
    if actual_hash != expected_hash:
        raise ReviewValidationError(
            f"source run {run_id} JSONL queue bytes do not match the catalog hash"
        )
    try:
        text = queue_bytes.decode("utf-8")
    except UnicodeError as error:
        raise ReviewValidationError(
            f"cannot read source run {run_id} JSONL queue: {error}"
        ) from error
    items: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ReviewValidationError(
                f"source run {run_id} JSONL has an empty line at {line_number}"
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReviewValidationError(
                f"source run {run_id} JSONL line {line_number} is invalid"
            ) from error
        if not isinstance(record, dict):
            raise ReviewValidationError(
                f"source run {run_id} JSONL line {line_number} is not an object"
            )
        context = record.get("queue_context")
        item = record.get("item")
        if (
            not isinstance(context, dict)
            or context.get("run_id") != run_id
            or not isinstance(item, dict)
        ):
            raise ReviewValidationError(
                f"source run {run_id} JSONL line {line_number} has invalid context"
            )
        identity = item.get("identity")
        if not isinstance(identity, dict) or not isinstance(
            identity.get("item_id"), str
        ):
            raise ReviewValidationError(
                f"source run {run_id} JSONL line {line_number} has no item identity"
            )
        item_id = cast(str, identity["item_id"])
        if item_id in items:
            raise ReviewValidationError(
                f"source run {run_id} JSONL repeats item {item_id}"
            )
        items[item_id] = cast(dict[str, Any], item)
    return QueueEvidence(path_text=path_text, sha256=actual_hash, items=items)


def _assert_projection_consistent(
    item_row: sqlite3.Row, canonical_item: dict[str, Any]
) -> list[str]:
    review = canonical_item.get("review")
    if not isinstance(review, dict):
        raise ReviewValidationError(
            f"item {item_row['item_id']} canonical review object is missing"
        )
    tags = _json_string_list(
        cast(str, item_row["tags_json"]),
        f"item {item_row['item_id']} tags_json",
    )
    expected = {
        "status": item_row["review_status"],
        "priority": item_row["review_priority"],
        "reviewer": item_row["reviewer"],
        "reviewed_at": item_row["reviewed_at"],
        "tags": tags,
        "notes": item_row["notes"],
    }
    if review != expected:
        raise ReviewValidationError(
            f"item {item_row['item_id']} review projection is inconsistent"
        )
    return tags


def _substantive_payload(
    batch: dict[str, Any], decision: dict[str, Any], item_version: str
) -> dict[str, Any]:
    return {
        "item_id": decision["item_id"],
        "source_run_id": decision["source_run_id"],
        "item_version": item_version,
        "decision_timestamp_utc": decision["reviewed_at"],
        "reviewer": batch["reviewer"],
        "review_basis": decision["review_basis"],
        "status": decision["status"],
        "confidence": decision["confidence"],
        "rationale": decision["rationale"],
        "next_action": decision["next_action"],
        "tags": _canonical_tags(cast(list[str], decision["tags"])),
        "source_queue_sha256": decision["source_queue_sha256"],
    }


def _stored_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "item_id": row["item_id"],
        "source_run_id": row["source_run_id"],
        "item_version": row["item_version"],
        "decision_timestamp_utc": row["decision_timestamp_utc"],
        "reviewer": row["reviewer"],
        "review_basis": row["review_basis"],
        "status": row["status"],
        "confidence": row["confidence"],
        "rationale": row["rationale"],
        "next_action": row["next_action"],
        "tags": _canonical_tags(
            _json_string_list(
                cast(str, row["tags_json"]),
                f"decision {row['decision_id']} tags_json",
            )
        ),
        "source_queue_sha256": row["source_queue_sha256"],
    }


def _prepare_decision(
    connection: sqlite3.Connection,
    batch: dict[str, Any],
    decision: dict[str, Any],
    data_root: Path,
    queue_cache: dict[str, QueueEvidence],
    has_review_table: bool,
) -> PreparedDecision:
    item_id = cast(str, decision["item_id"])
    source_run_id = cast(str, decision["source_run_id"])
    item_row = connection.execute(
        "SELECT * FROM items WHERE item_id=?", (item_id,)
    ).fetchone()
    if item_row is None:
        raise ReviewValidationError(f"unknown item: {item_id}")
    run_row = connection.execute(
        "SELECT run_id,jsonl_path,jsonl_sha256 FROM runs WHERE run_id=?",
        (source_run_id,),
    ).fetchone()
    if run_row is None:
        raise ReviewValidationError(f"unknown source run: {source_run_id}")
    if not run_row["jsonl_path"] or not run_row["jsonl_sha256"]:
        raise ReviewValidationError(
            f"source run {source_run_id} has no recorded JSONL queue provenance"
        )
    recorded_hash = cast(str, run_row["jsonl_sha256"])
    if decision["source_queue_sha256"] != recorded_hash:
        raise ReviewValidationError(
            f"source run {source_run_id} queue hash does not match the catalog"
        )
    evidence = queue_cache.get(source_run_id)
    if evidence is None:
        evidence = _load_queue(
            source_run_id,
            cast(str, run_row["jsonl_path"]),
            recorded_hash,
            data_root,
        )
        queue_cache[source_run_id] = evidence
    queued_item = evidence.items.get(item_id)
    if queued_item is None:
        raise ReviewValidationError(
            f"item {item_id} is absent from source run {source_run_id} JSONL queue"
        )
    discovery_rows = connection.execute(
        "SELECT disposition,observed_version,normalized_snapshot_json "
        "FROM discoveries WHERE run_id=? AND item_id=? "
        "AND disposition IN ('NEW','UPDATED')",
        (source_run_id, item_id),
    ).fetchall()
    eligible_versions: set[str] = set()
    for discovery in discovery_rows:
        try:
            snapshot = json.loads(discovery["normalized_snapshot_json"])
        except json.JSONDecodeError as error:
            raise ReviewValidationError(
                f"source run {source_run_id} discovery snapshot is invalid"
            ) from error
        if (
            isinstance(snapshot, dict)
            and isinstance(snapshot.get("relevance"), dict)
            and snapshot["relevance"].get("eligible") is True
        ):
            eligible_versions.add(cast(str, discovery["observed_version"]))
    if not eligible_versions:
        raise ReviewValidationError(
            f"item {item_id} was not an eligible NEW or UPDATED discovery "
            f"in source run {source_run_id}"
        )
    identity = queued_item.get("identity")
    if not isinstance(identity, dict) or not isinstance(
        identity.get("current_arxiv_version"), str
    ):
        raise ReviewValidationError(f"queued item {item_id} has no paper version")
    queue_version = cast(str, identity["current_arxiv_version"])
    if queue_version not in eligible_versions:
        raise ReviewValidationError(
            f"queued version {queue_version} for item {item_id} does not match "
            "the eligible discovery"
        )
    explicit_version = cast(str | None, decision.get("item_version"))
    if explicit_version is not None and explicit_version != queue_version:
        raise ReviewValidationError(
            f"explicit item version for {item_id} does not match the source queue"
        )
    current_version = cast(str, item_row["current_version"])
    if explicit_version is None and queue_version != current_version:
        raise ReviewValidationError(
            f"queued version {queue_version} for item {item_id} is not current; "
            "supply item_version to review that exact queued version"
        )
    if explicit_version is not None and queue_version != current_version:
        revision = connection.execute(
            "SELECT 1 FROM item_revisions WHERE item_id=? AND arxiv_version=?",
            (item_id, queue_version),
        ).fetchone()
        if revision is None:
            raise ReviewValidationError(
                f"explicit item version {queue_version} for {item_id} "
                "has no catalog revision"
            )
    try:
        canonical_item = json.loads(item_row["canonical_json"])
    except json.JSONDecodeError as error:
        raise ReviewValidationError(
            f"item {item_id} canonical JSON is invalid"
        ) from error
    if not isinstance(canonical_item, dict):
        raise ReviewValidationError(f"item {item_id} canonical JSON is not an object")
    canonical = cast(dict[str, Any], canonical_item)
    previous_tags = _assert_projection_consistent(item_row, canonical)
    payload = _substantive_payload(batch, decision, queue_version)
    decision_id = decision_id_for(payload)
    if has_review_table:
        existing = connection.execute(
            "SELECT * FROM review_decisions WHERE decision_id=?", (decision_id,)
        ).fetchone()
        if existing is not None:
            if _stored_payload(existing) != payload:
                raise ReviewValidationError(
                    f"conflicting payload for decision ID {decision_id}"
                )
            return PreparedDecision(
                decision_id=decision_id,
                batch_id=cast(str, existing["batch_id"]),
                payload=payload,
                source_queue_path=cast(str, existing["source_queue_path"]),
                previous_status=cast(str, existing["previous_status"]),
                previous_reviewer=cast(str | None, existing["previous_reviewer"]),
                previous_reviewed_at=cast(str | None, existing["previous_reviewed_at"]),
                previous_tags=_json_string_list(
                    cast(str, existing["previous_tags_json"]),
                    f"decision {decision_id} previous_tags_json",
                ),
                previous_notes=cast(str | None, existing["previous_notes"]),
                canonical_item=None,
                already_recorded=True,
            )
        batch_conflict = connection.execute(
            "SELECT decision_id FROM review_decisions WHERE batch_id=? AND item_id=?",
            (batch["batch_id"], item_id),
        ).fetchone()
        if batch_conflict is not None:
            raise ReviewValidationError(
                f"batch {batch['batch_id']} already contains a conflicting "
                f"decision for item {item_id}"
            )
    current_status = cast(str, item_row["review_status"])
    if decision["expected_current_status"] != current_status:
        raise ReviewValidationError(
            f"expected current status mismatch for {item_id}: "
            f"expected {decision['expected_current_status']}, found {current_status}"
        )
    reviewed_at = _parse_utc_timestamp(
        cast(str, decision["reviewed_at"]), f"reviewed_at for {item_id}"
    )
    if item_row["reviewed_at"] is not None:
        current_reviewed_at = _parse_utc_timestamp(
            cast(str, item_row["reviewed_at"]),
            f"catalog reviewed_at for {item_id}",
        )
        if reviewed_at < current_reviewed_at:
            raise ReviewValidationError(
                f"reviewed_at for {item_id} is earlier than the current review"
            )
    return PreparedDecision(
        decision_id=decision_id,
        batch_id=cast(str, batch["batch_id"]),
        payload=payload,
        source_queue_path=evidence.path_text,
        previous_status=current_status,
        previous_reviewer=cast(str | None, item_row["reviewer"]),
        previous_reviewed_at=cast(str | None, item_row["reviewed_at"]),
        previous_tags=previous_tags,
        previous_notes=cast(str | None, item_row["notes"]),
        canonical_item=canonical,
        already_recorded=False,
    )


def _prepare_batch(
    connection: sqlite3.Connection,
    batch: dict[str, Any],
    data_root: Path,
    has_review_table: bool,
) -> list[PreparedDecision]:
    queue_cache: dict[str, QueueEvidence] = {}
    prepared = [
        _prepare_decision(
            connection,
            batch,
            decision,
            data_root,
            queue_cache,
            has_review_table,
        )
        for decision in cast(list[dict[str, Any]], batch["decisions"])
    ]
    return sorted(prepared, key=lambda value: (value.payload["item_id"], value.decision_id))


def _record_prepared(
    connection: sqlite3.Connection,
    prepared: list[PreparedDecision],
    normalized_schema_path: Path,
    created_at: str,
) -> None:
    for value in prepared:
        if value.already_recorded:
            continue
        payload = value.payload
        connection.execute(
            """INSERT INTO review_decisions(
            decision_id,batch_id,item_id,source_run_id,item_version,
            decision_timestamp_utc,reviewer,review_basis,status,confidence,
            rationale,next_action,tags_json,source_queue_path,source_queue_sha256,
            previous_status,previous_reviewer,previous_reviewed_at,
            previous_tags_json,previous_notes,created_at_utc
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                value.decision_id,
                value.batch_id,
                payload["item_id"],
                payload["source_run_id"],
                payload["item_version"],
                payload["decision_timestamp_utc"],
                payload["reviewer"],
                payload["review_basis"],
                payload["status"],
                payload["confidence"],
                payload["rationale"],
                payload["next_action"],
                canonical_json(payload["tags"]),
                value.source_queue_path,
                payload["source_queue_sha256"],
                value.previous_status,
                value.previous_reviewer,
                value.previous_reviewed_at,
                canonical_json(value.previous_tags),
                value.previous_notes,
                created_at,
            ),
        )
        if value.canonical_item is None:
            raise RuntimeError("new review decision has no canonical item")
        canonical_item = value.canonical_item
        review = cast(dict[str, Any], canonical_item["review"])
        review.update(
            {
                "status": payload["status"],
                "reviewer": payload["reviewer"],
                "reviewed_at": payload["decision_timestamp_utc"],
                "tags": payload["tags"],
                "notes": payload["rationale"],
            }
        )
        validate_json(canonical_item, normalized_schema_path)
        connection.execute(
            "UPDATE items SET review_status=?,reviewer=?,reviewed_at=?,"
            "tags_json=?,notes=?,canonical_json=? WHERE item_id=?",
            (
                payload["status"],
                payload["reviewer"],
                payload["decision_timestamp_utc"],
                canonical_json(payload["tags"]),
                payload["rationale"],
                canonical_json(canonical_item),
                payload["item_id"],
            ),
        )


def _summary(
    batch: dict[str, Any],
    prepared: list[PreparedDecision],
    dry_run: bool,
) -> dict[str, Any]:
    projected = []
    for value in prepared:
        if value.already_recorded:
            outcome = "already_recorded"
        elif dry_run:
            outcome = "would_record"
        else:
            outcome = "recorded"
        projected.append(
            {
                "decision_id": value.decision_id,
                "from_status": value.previous_status,
                "item_id": value.payload["item_id"],
                "outcome": outcome,
                "to_status": value.payload["status"],
            }
        )
    already = sum(value.already_recorded for value in prepared)
    return {
        "already_recorded": already,
        "batch_id": batch["batch_id"],
        "dry_run": dry_run,
        "projected": projected,
        "recorded": 0 if dry_run else len(prepared) - already,
        "total": len(prepared),
        "would_record": len(prepared) - already if dry_run else 0,
    }


def record_review_batch(
    database: Path,
    data_root: Path,
    batch: dict[str, Any],
    normalized_schema_path: Path,
    *,
    dry_run: bool = False,
    now_fn: Callable[[], str] = utc_now,
) -> dict[str, Any]:
    _validate_batch_schema(
        batch, normalized_schema_path.parent / "human_review_batch.schema.json"
    )
    _validate_batch_semantics(batch)
    with connect(database) as connection:
        has_review_table = _table_exists(connection, "review_decisions")
        if not dry_run and not has_review_table:
            raise RuntimeError("review_decisions table is not initialized")
        if not dry_run:
            connection.execute("BEGIN IMMEDIATE")
        try:
            prepared = _prepare_batch(
                connection,
                batch,
                data_root,
                has_review_table,
            )
            if not dry_run:
                created_at = now_fn()
                _parse_utc_timestamp(created_at, "created_at_utc")
                _record_prepared(
                    connection,
                    prepared,
                    normalized_schema_path,
                    created_at,
                )
                connection.commit()
            return _summary(batch, prepared, dry_run)
        except Exception:
            if not dry_run:
                connection.rollback()
            raise


def _decision_export_record(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "batch_id": row["batch_id"],
        "confidence": row["confidence"],
        "created_at_utc": row["created_at_utc"],
        "current_review": {
            "notes": row["notes"],
            "priority": row["review_priority"],
            "reviewed_at": row["reviewed_at"],
            "reviewer": row["current_reviewer"],
            "status": row["review_status"],
            "tags": _canonical_tags(
                _json_string_list(
                    cast(str, row["current_tags_json"]),
                    f"item {row['item_id']} tags_json",
                )
            ),
        },
        "current_version": row["current_version"],
        "decision_id": row["decision_id"],
        "decision_timestamp_utc": row["decision_timestamp_utc"],
        "item_id": row["item_id"],
        "item_title": row["title"],
        "item_version": row["item_version"],
        "next_action": row["next_action"],
        "previous_state": {
            "notes": row["previous_notes"],
            "reviewed_at": row["previous_reviewed_at"],
            "reviewer": row["previous_reviewer"],
            "status": row["previous_status"],
            "tags": _canonical_tags(
                _json_string_list(
                    cast(str, row["previous_tags_json"]),
                    f"decision {row['decision_id']} previous_tags_json",
                )
            ),
        },
        "rationale": row["rationale"],
        "review_basis": row["review_basis"],
        "reviewer": row["reviewer"],
        "source_queue_path": row["source_queue_path"],
        "source_queue_sha256": row["source_queue_sha256"],
        "source_run_id": row["source_run_id"],
        "source_url": row["source_record_url"],
        "status": row["status"],
        "tags": _canonical_tags(
            _json_string_list(
                cast(str, row["decision_tags_json"]),
                f"decision {row['decision_id']} tags_json",
            )
        ),
    }


def _display(value: Any) -> str:
    if value is None:
        return "none"
    return " ".join(str(value).split()).replace("`", "\\`")


def _history_markdown(records: list[dict[str, Any]]) -> bytes:
    lines = [
        "# Human Review Decision History",
        "",
        "This is a derived view of the authoritative SQLite review ledger.",
        "",
    ]
    if not records:
        lines.extend(["No human-review decisions recorded.", ""])
    current_date: str | None = None
    for record in records:
        review_date = cast(str, record["decision_timestamp_utc"])[:10]
        if review_date != current_date:
            lines.extend([f"## {review_date}", ""])
            current_date = review_date
        lines.extend(
            [
                f"### {_display(record['item_title'])}",
                "",
                f"- Decision time: `{record['decision_timestamp_utc']}`",
                f"- Decision ID: `{record['decision_id']}`",
                f"- Batch ID: `{record['batch_id']}`",
                f"- Review basis: `{record['review_basis']}`",
                f"- Reviewer: {_display(record['reviewer'])}",
                f"- Source run: `{record['source_run_id']}`",
                f"- Item ID: `{record['item_id']}`",
                f"- Paper version: `{record['item_version']}`",
                f"- Status: `{record['status']}`",
                f"- Previous status: `{record['previous_state']['status']}`",
                f"- Confidence: `{record['confidence']}`",
                f"- Tags: {_display(', '.join(record['tags']) or 'none')}",
                f"- Queue path: {_display(record['source_queue_path'])}",
                f"- Queue SHA-256: `{record['source_queue_sha256']}`",
                f"- Source: {_display(record['source_url'])}",
                f"- Rationale: {_display(record['rationale'])}",
                f"- Next action: {_display(record['next_action'])}",
                "",
            ]
        )
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def _current_queue_markdown(
    title: str, description: str, items: list[dict[str, Any]]
) -> bytes:
    lines = [
        f"# {title}",
        "",
        description,
        "",
    ]
    if not items:
        lines.extend(["No matching current items.", ""])
    for item in items:
        lines.extend(
            [
                f"## {_display(item['title'])}",
                "",
                f"- Item ID: `{item['item_id']}`",
                f"- Current version: `{item['current_version']}`",
                f"- Status: `{item['review_status']}`",
                f"- Reviewer: {_display(item['reviewer'])}",
                f"- Reviewed at: {_display(item['reviewed_at'])}",
                f"- Tags: {_display(', '.join(item['tags']) or 'none')}",
                f"- Source: {_display(item['source_record_url'])}",
                f"- Notes: {_display(item['notes'])}",
                "",
            ]
        )
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def render_review_exports(
    connection: sqlite3.Connection,
) -> dict[str, bytes]:
    decision_records: list[dict[str, Any]] = []
    if _table_exists(connection, "review_decisions"):
        rows = connection.execute(
            """SELECT d.*,
            i.title,i.source_record_url,i.current_version,i.review_status,
            i.review_priority,i.reviewer AS current_reviewer,i.reviewed_at,
            i.tags_json AS current_tags_json,i.notes,
            d.tags_json AS decision_tags_json
            FROM review_decisions AS d
            JOIN items AS i ON i.item_id=d.item_id"""
        ).fetchall()
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                _parse_utc_timestamp(
                    cast(str, row["decision_timestamp_utc"]),
                    f"decision {row['decision_id']} decision_timestamp_utc",
                ),
                row["decision_id"],
            ),
        )
        decision_records = [_decision_export_record(row) for row in ordered_rows]
    item_rows = connection.execute(
        "SELECT item_id,title,source_record_url,current_version,review_status,"
        "reviewer,reviewed_at,tags_json,notes FROM items"
    ).fetchall()
    current_items = sorted(
        [
            {
                **dict(row),
                "tags": _canonical_tags(
                    _json_string_list(
                        cast(str, row["tags_json"]),
                        f"item {row['item_id']} tags_json",
                    )
                ),
            }
            for row in item_rows
        ],
        key=lambda item: (cast(str, item["title"]).casefold(), item["item_id"]),
    )
    deep_review = [
        item
        for item in current_items
        if item["review_status"] == "KEEP_FOR_DEEP_REVIEW"
    ]
    methodology_watchlist = [
        item for item in current_items if "methodology-watchlist" in item["tags"]
    ]
    index_jsonl = (
        "\n".join(canonical_json(record) for record in decision_records) + "\n"
        if decision_records
        else ""
    ).encode("utf-8")
    return {
        "index.jsonl": index_jsonl,
        "index.md": _history_markdown(decision_records),
        "deep-review-queue.md": _current_queue_markdown(
            "Deep Review Queue",
            "Current items whose authoritative status is `KEEP_FOR_DEEP_REVIEW`.",
            deep_review,
        ),
        "methodology-watchlist.md": _current_queue_markdown(
            "Methodology Watchlist",
            "Current items tagged `methodology-watchlist` in the authoritative catalog.",
            methodology_watchlist,
        ),
    }


def _write_staged_file(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def publish_review_exports(output_root: Path, files: dict[str, bytes]) -> None:
    if set(files) != set(EXPORT_FILENAMES):
        raise RuntimeError("incomplete human-review export set")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.{uuid.uuid4().hex}.tmp"
    backup = output_root.parent / f".{output_root.name}.{uuid.uuid4().hex}.bak"
    stage.mkdir()
    backup.mkdir()
    created_output_root = False
    replaced: list[str] = []
    try:
        for name in EXPORT_FILENAMES:
            _write_staged_file(stage / name, files[name])
        if output_root.exists():
            if not output_root.is_dir() or output_root.is_symlink():
                raise RuntimeError("human-review export root is not a normal directory")
        else:
            output_root.mkdir()
            created_output_root = True
        for name in EXPORT_FILENAMES:
            target = output_root / name
            if target.exists():
                if not target.is_file() or target.is_symlink():
                    raise RuntimeError(
                        f"human-review export target is not a normal file: {name}"
                    )
                _write_staged_file(backup / name, target.read_bytes())
        try:
            for name in EXPORT_FILENAMES:
                os.replace(stage / name, output_root / name)
                replaced.append(name)
        except Exception:
            for name in reversed(replaced):
                saved = backup / name
                target = output_root / name
                if saved.exists():
                    os.replace(saved, target)
                elif target.exists():
                    target.unlink()
            raise
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        if created_output_root:
            with suppress(OSError):
                output_root.rmdir()


def export_reviews(database: Path, output_root: Path) -> dict[str, Any]:
    with connect(database) as connection:
        files = render_review_exports(connection)
    publish_review_exports(output_root, files)
    return {
        "files": list(EXPORT_FILENAMES),
        "output_root": output_root.as_posix(),
    }
