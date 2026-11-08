from __future__ import annotations

import copy
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from research_intake.evidence import sha256_file
from research_intake.identity import version_number
from research_intake.validation import validate_json

SCHEMA_VERSION = "1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class _ClosingConnection(sqlite3.Connection):
    """Commit/rollback normally, then release the database handle on context exit."""

    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, factory=_ClosingConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
 run_id TEXT PRIMARY KEY, run_type TEXT NOT NULL CHECK(run_type IN ('backfill','incremental')),
 start_timestamp_utc TEXT NOT NULL, interval_start_utc TEXT NOT NULL, interval_end_utc TEXT NOT NULL,
 discovery_time_basis TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'RUNNING', prepared_final_status TEXT,
 run_finish_timestamp_utc TEXT, application_version TEXT NOT NULL, git_commit TEXT, python_version TEXT NOT NULL,
 registry_hash TEXT NOT NULL, topics_hash TEXT NOT NULL, previous_eligible_run_id TEXT, overlap_seconds INTEGER NOT NULL,
 catalog_committed INTEGER NOT NULL DEFAULT 0, catalog_committed_at_utc TEXT,
 incremental_predecessor_eligible INTEGER NOT NULL DEFAULT 0,
 incremental_predecessor_reason_code TEXT NOT NULL DEFAULT 'NOT_FINALIZED',
 queue_render_timestamp_utc TEXT, queue_renderer_version TEXT,
 markdown_path TEXT, markdown_status TEXT, markdown_count INTEGER, markdown_size INTEGER, markdown_sha256 TEXT,
 jsonl_path TEXT, jsonl_status TEXT, jsonl_count INTEGER, jsonl_size INTEGER, jsonl_sha256 TEXT,
 manifest_path TEXT NOT NULL, manifest_event_cutoff_id INTEGER,
 manifest_event_cutoff_state TEXT NOT NULL DEFAULT 'PROVISIONAL',
 manifest_publication_state TEXT NOT NULL DEFAULT 'MISSING',
 run_counts_json TEXT,
 warning_count INTEGER NOT NULL DEFAULT 0, error_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS run_query_state (
 run_id TEXT NOT NULL REFERENCES runs(run_id), query_id TEXT NOT NULL, source_id TEXT NOT NULL,
 completion_status TEXT NOT NULL DEFAULT 'NOT_STARTED', discovery_time_basis TEXT NOT NULL,
 required_lower_boundary TEXT NOT NULL, pages_attempted INTEGER NOT NULL DEFAULT 0,
 pages_succeeded INTEGER NOT NULL DEFAULT 0, received_count INTEGER NOT NULL DEFAULT 0,
 accepted_count INTEGER NOT NULL DEFAULT 0, malformed_entry_count INTEGER NOT NULL DEFAULT 0,
 exhausted_available_results INTEGER NOT NULL DEFAULT 0, crossed_lower_boundary INTEGER NOT NULL DEFAULT 0,
 stopped_early INTEGER NOT NULL DEFAULT 0, final_stop_reason TEXT NOT NULL DEFAULT 'NOT_STARTED',
 first_request_timestamp_utc TEXT, last_request_timestamp_utc TEXT, last_ordering_timestamp_utc TEXT,
 PRIMARY KEY(run_id, query_id)
);
CREATE TABLE IF NOT EXISTS items (
 item_id TEXT PRIMARY KEY, primary_source TEXT NOT NULL, primary_key_type TEXT NOT NULL,
 primary_key_value TEXT NOT NULL, base_arxiv_id TEXT NOT NULL UNIQUE, current_version TEXT NOT NULL,
 doi TEXT, title TEXT NOT NULL, abstract TEXT NOT NULL, authors_json TEXT NOT NULL,
 language TEXT NOT NULL, primary_category TEXT NOT NULL, categories_json TEXT NOT NULL,
 submitted_at TEXT NOT NULL, source_updated_at TEXT NOT NULL,
 journal_reference TEXT, comments TEXT, license TEXT,
 source_record_url TEXT NOT NULL, pdf_url TEXT, doi_url TEXT,
 relevance_score INTEGER NOT NULL, relevance_priority TEXT NOT NULL, eligible INTEGER NOT NULL,
 matched_rule_ids_json TEXT NOT NULL, penalty_rule_ids_json TEXT NOT NULL,
 review_status TEXT NOT NULL, review_priority TEXT NOT NULL, reviewer TEXT,
 reviewed_at TEXT, tags_json TEXT NOT NULL, notes TEXT,
 first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL,
 first_seen_run_id TEXT NOT NULL REFERENCES runs(run_id), last_seen_run_id TEXT NOT NULL REFERENCES runs(run_id),
 meaningful_fingerprint TEXT NOT NULL, canonical_json TEXT NOT NULL,
 UNIQUE(primary_source, primary_key_type, primary_key_value)
);
CREATE INDEX IF NOT EXISTS idx_items_doi ON items(doi);
CREATE TABLE IF NOT EXISTS item_revisions (
 item_id TEXT NOT NULL REFERENCES items(item_id), arxiv_version TEXT NOT NULL,
 first_observed_run_id TEXT NOT NULL REFERENCES runs(run_id), raw_artifact_path TEXT NOT NULL,
 raw_artifact_sha256 TEXT NOT NULL, source_fingerprint TEXT NOT NULL, normalized_json TEXT NOT NULL,
 PRIMARY KEY(item_id, arxiv_version)
);
CREATE TABLE IF NOT EXISTS discoveries (
 discovery_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id),
 item_id TEXT NOT NULL REFERENCES items(item_id), source_id TEXT NOT NULL, query_id TEXT NOT NULL,
 source_record_id TEXT NOT NULL, disposition TEXT NOT NULL CHECK(disposition IN ('NEW','UPDATED','UNCHANGED','EXCLUDED')),
 observed_version TEXT NOT NULL, raw_artifact_path TEXT NOT NULL, raw_artifact_sha256 TEXT NOT NULL,
 observation_position INTEGER NOT NULL, normalized_snapshot_json TEXT NOT NULL,
 UNIQUE(run_id, source_id, query_id, source_record_id)
);
CREATE TABLE IF NOT EXISTS review_decisions (
 decision_id TEXT PRIMARY KEY
  CHECK(length(decision_id) = 64 AND decision_id NOT GLOB '*[^0-9a-f]*'),
 batch_id TEXT NOT NULL CHECK(length(trim(batch_id)) > 0),
 item_id TEXT NOT NULL REFERENCES items(item_id),
 source_run_id TEXT NOT NULL REFERENCES runs(run_id),
 item_version TEXT NOT NULL
  CHECK(
   substr(item_version, 1, 1) = 'v'
   AND length(item_version) > 1
   AND substr(item_version, 2) NOT GLOB '*[^0-9]*'
  ),
 decision_timestamp_utc TEXT NOT NULL CHECK(length(trim(decision_timestamp_utc)) > 0),
 reviewer TEXT NOT NULL CHECK(length(trim(reviewer)) > 0),
 review_basis TEXT NOT NULL
  CHECK(review_basis IN ('ABSTRACT_TRIAGE','FULL_TEXT_REVIEW')),
 status TEXT NOT NULL
  CHECK(status IN (
   'REVIEWING','KEEP_FOR_DEEP_REVIEW','EDUCATIONAL_ONLY',
   'PARKED','REJECTED','DUPLICATE'
  )),
 confidence TEXT NOT NULL CHECK(confidence IN ('LOW','MEDIUM','HIGH')),
 rationale TEXT NOT NULL CHECK(length(trim(rationale)) > 0),
 next_action TEXT NOT NULL CHECK(length(trim(next_action)) > 0),
 tags_json TEXT NOT NULL,
 source_queue_path TEXT NOT NULL CHECK(length(trim(source_queue_path)) > 0),
 source_queue_sha256 TEXT NOT NULL
  CHECK(
   length(source_queue_sha256) = 64
   AND source_queue_sha256 NOT GLOB '*[^0-9a-f]*'
  ),
 previous_status TEXT NOT NULL
  CHECK(previous_status IN (
   'NEW','REVIEWING','KEEP_FOR_DEEP_REVIEW','EDUCATIONAL_ONLY',
   'PARKED','REJECTED','DUPLICATE'
  )),
 previous_reviewer TEXT,
 previous_reviewed_at TEXT,
 previous_tags_json TEXT NOT NULL,
 previous_notes TEXT,
 created_at_utc TEXT NOT NULL CHECK(length(trim(created_at_utc)) > 0),
 UNIQUE(batch_id, item_id)
);
CREATE INDEX IF NOT EXISTS idx_review_decisions_chronology
 ON review_decisions(decision_timestamp_utc, decision_id);
CREATE INDEX IF NOT EXISTS idx_review_decisions_item
 ON review_decisions(item_id, decision_timestamp_utc, decision_id);
CREATE TRIGGER IF NOT EXISTS review_decisions_no_update
 BEFORE UPDATE ON review_decisions
 BEGIN SELECT RAISE(ABORT, 'review_decisions is append-only'); END;
CREATE TRIGGER IF NOT EXISTS review_decisions_no_delete
 BEFORE DELETE ON review_decisions
 BEGIN SELECT RAISE(ABORT, 'review_decisions is append-only'); END;
CREATE TABLE IF NOT EXISTS run_events (
 event_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id),
 timestamp_utc TEXT NOT NULL, severity TEXT NOT NULL CHECK(severity IN ('INFO','WARNING','ERROR')),
 event_code TEXT NOT NULL, pipeline_phase TEXT NOT NULL, source_id TEXT, query_id TEXT,
 context_json TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS run_events_no_update BEFORE UPDATE ON run_events BEGIN SELECT RAISE(ABORT, 'run_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS run_events_no_delete BEFORE DELETE ON run_events BEGIN SELECT RAISE(ABORT, 'run_events is append-only'); END;
"""

MANIFEST_CUTOFF_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS runs_manifest_cutoff_immutable
BEFORE UPDATE OF manifest_event_cutoff_id, manifest_event_cutoff_state ON runs
WHEN OLD.manifest_event_cutoff_state = 'FROZEN'
 AND (
   NEW.manifest_event_cutoff_id IS NOT OLD.manifest_event_cutoff_id
   OR NEW.manifest_event_cutoff_state != 'FROZEN'
 )
BEGIN
 SELECT RAISE(ABORT, 'published manifest cutoff is immutable');
END;
"""

ITEM_COLUMN_MIGRATIONS = {
    "abstract": "TEXT NOT NULL DEFAULT ''",
    "authors_json": "TEXT NOT NULL DEFAULT '[]'",
    "language": "TEXT NOT NULL DEFAULT 'und'",
    "primary_category": "TEXT NOT NULL DEFAULT ''",
    "categories_json": "TEXT NOT NULL DEFAULT '[]'",
    "journal_reference": "TEXT",
    "comments": "TEXT",
    "license": "TEXT",
    "source_record_url": "TEXT NOT NULL DEFAULT ''",
    "pdf_url": "TEXT",
    "doi_url": "TEXT",
    "matched_rule_ids_json": "TEXT NOT NULL DEFAULT '[]'",
    "penalty_rule_ids_json": "TEXT NOT NULL DEFAULT '[]'",
    "review_priority": "TEXT NOT NULL DEFAULT 'NONE'",
    "reviewer": "TEXT",
    "reviewed_at": "TEXT",
    "tags_json": "TEXT NOT NULL DEFAULT '[]'",
    "notes": "TEXT",
}

RUN_COLUMN_MIGRATIONS = {
    "manifest_event_cutoff_state": "TEXT NOT NULL DEFAULT 'PROVISIONAL'",
    "run_counts_json": "TEXT",
}


def init_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as connection:
        connection.executescript(SCHEMA_SQL)
        run_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(runs)")
        }
        for name, declaration in RUN_COLUMN_MIGRATIONS.items():
            if name not in run_columns:
                connection.execute(
                    f"ALTER TABLE runs ADD COLUMN {name} {declaration}"
                )
        item_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(items)")
        }
        for name, declaration in ITEM_COLUMN_MIGRATIONS.items():
            if name not in item_columns:
                connection.execute(
                    f"ALTER TABLE items ADD COLUMN {name} {declaration}"
                )
        connection.executescript(MANIFEST_CUTOFF_TRIGGER_SQL)
        existing = connection.execute("SELECT value FROM schema_metadata WHERE key='schema_version'").fetchone()
        if existing and existing[0] != SCHEMA_VERSION:
            raise RuntimeError(f"unsupported catalog schema {existing[0]}")
        connection.execute("INSERT OR IGNORE INTO schema_metadata(key,value) VALUES('schema_version',?)", (SCHEMA_VERSION,))
        connection.execute("INSERT OR IGNORE INTO schema_metadata(key,value) VALUES('application_version','0.1.0')")


def require_database(path: Path) -> None:
    if not path.exists():
        raise RuntimeError("catalog is not initialized; run init-db")
    with connect(path) as connection:
        row = connection.execute("SELECT value FROM schema_metadata WHERE key='schema_version'").fetchone()
        if not row or row[0] != SCHEMA_VERSION:
            raise RuntimeError("catalog schema is missing or incompatible")


_SECRET_EVENT_KEYS = {
    "api-key",
    "api_key",
    "apikey",
    "authorization",
    "contact-email",
    "contact_email",
    "cookie",
    "email",
    "passwd",
    "password",
    "proxy-authorization",
    "secret",
    "set-cookie",
    "token",
    "user-agent",
}


def _reject_secret_event_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = str(key).strip().casefold()
            if normalized_key in _SECRET_EVENT_KEYS:
                raise ValueError(
                    f"run event context contains prohibited secret field: {key}"
                )
            _reject_secret_event_fields(child)
    elif isinstance(value, list):
        for child in value:
            _reject_secret_event_fields(child)


def append_event(connection: sqlite3.Connection, run_id: str, severity: str, code: str, phase: str, context: dict[str, Any] | None = None, source_id: str | None = None, query_id: str | None = None) -> int:
    _reject_secret_event_fields(context or {})
    cursor = connection.execute(
        "INSERT INTO run_events(run_id,timestamp_utc,severity,event_code,pipeline_phase,source_id,query_id,context_json) VALUES(?,?,?,?,?,?,?,?)",
        (run_id, utc_now(), severity, code, phase, source_id, query_id, json.dumps(context or {}, sort_keys=True, separators=(",", ":"))),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("event insert did not return an event ID")
    return cursor.lastrowid


def latest_eligible_run(connection: sqlite3.Connection) -> sqlite3.Row | None:
    return cast(
        sqlite3.Row | None,
        connection.execute(
            "SELECT * FROM runs WHERE incremental_predecessor_eligible=1 "
            "ORDER BY start_timestamp_utc DESC LIMIT 1"
        ).fetchone(),
    )


def create_run(connection: sqlite3.Connection, values: dict[str, Any], queries: list[dict[str, Any]]) -> None:
    connection.execute(
        """INSERT INTO runs(run_id,run_type,start_timestamp_utc,interval_start_utc,interval_end_utc,discovery_time_basis,application_version,git_commit,python_version,registry_hash,topics_hash,previous_eligible_run_id,overlap_seconds,manifest_path)
        VALUES(:run_id,:run_type,:start_timestamp_utc,:interval_start_utc,:interval_end_utc,:discovery_time_basis,:application_version,:git_commit,:python_version,:registry_hash,:topics_hash,:previous_eligible_run_id,:overlap_seconds,:manifest_path)""",
        values,
    )
    for query in queries:
        connection.execute(
            "INSERT INTO run_query_state(run_id,query_id,source_id,discovery_time_basis,required_lower_boundary) VALUES(?,?,?,?,?)",
            (values["run_id"], query["query_id"], query["source_id"], values["discovery_time_basis"], values["interval_start_utc"]),
        )


def event_records(connection: sqlite3.Connection, run_id: str, cutoff: int | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM run_events WHERE run_id=?"
    params: list[Any] = [run_id]
    if cutoff is not None:
        sql += " AND event_id<=?"
        params.append(cutoff)
    sql += " ORDER BY event_id"
    return [{**dict(row), "context": json.loads(row["context_json"])} for row in connection.execute(sql, params)]


def query_states(connection: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute("SELECT * FROM run_query_state WHERE run_id=? ORDER BY query_id", (run_id,))]


def predecessor_decision(connection: sqlite3.Connection, run_id: str) -> tuple[bool, str]:
    run = connection.execute("SELECT catalog_committed FROM runs WHERE run_id=?", (run_id,)).fetchone()
    states = query_states(connection, run_id)
    if not states:
        return False, "DURABLE_STATE_INCOMPLETE"
    for state in states:
        if state["completion_status"] == "FAILED":
            if state["final_stop_reason"] in {"PAGE_FAILURE"}:
                return False, "QUERY_SCAN_GAP"
            return False, "QUERY_FAILED"
        if state["completion_status"] != "COMPLETE":
            return False, "QUERY_SCAN_INCOMPLETE"
        if not state["exhausted_available_results"] and not state["crossed_lower_boundary"]:
            return False, "QUERY_SCAN_INCOMPLETE"
    if not run or not run["catalog_committed"]:
        return False, "CATALOG_NOT_COMMITTED"
    return True, "ALL_QUERIES_COVERED"


def _meaningful_fingerprint(item: dict[str, Any]) -> str:
    import hashlib

    payload = {"identity": {"doi": item["identity"]["doi"]}, "bibliographic": item["bibliographic"], "links": item["links"], "relevance": item["relevance"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def ingest_batch(
    connection: sqlite3.Connection,
    run_id: str,
    observations: list[dict[str, Any]],
    queue_timestamp: str,
    renderer_version: str,
    markdown_path: str,
    jsonl_path: str,
    normalized_schema_path: Path,
) -> dict[str, int]:
    counts = {"new": 0, "updated": 0, "unchanged": 0, "excluded": 0, "doi_collisions": 0}
    connection.execute("BEGIN IMMEDIATE")
    try:
        seen_doi_pairs: set[tuple[str, str, str]] = set()
        for position, observation in enumerate(observations):
            raw_path = Path(observation["raw_path"])
            if not raw_path.exists() or sha256_file(raw_path) != observation["raw_hash"]:
                raise RuntimeError("EVIDENCE_HASH_MISMATCH")
            item = cast(dict[str, Any], observation["item"])
            identity = item["identity"]
            item_id = identity["item_id"]
            existing_row = connection.execute("SELECT * FROM items WHERE item_id=?", (item_id,)).fetchone()
            existing = (
                cast(dict[str, Any], json.loads(existing_row["canonical_json"]))
                if existing_row
                else None
            )
            fingerprint = _meaningful_fingerprint(item)
            observed_version = identity["current_arxiv_version"]
            current_version = existing_row["current_version"] if existing_row else None
            canonical_update = not existing_row or version_number(observed_version) >= version_number(
                cast(str, current_version)
            )
            item["lifecycle"]["latest_source_version"] = canonical_update
            if not item["relevance"]["eligible"]:
                disposition = "EXCLUDED"
            elif not existing_row:
                disposition = "NEW"
            elif canonical_update and fingerprint != existing_row["meaningful_fingerprint"]:
                disposition = "UPDATED"
            else:
                disposition = "UNCHANGED"
            counts[disposition.lower()] += 1
            if existing and canonical_update:
                item["review"] = existing["review"]
                item["lifecycle"]["first_seen_at"] = existing["lifecycle"]["first_seen_at"]
                item["lifecycle"]["first_seen_run_id"] = existing["lifecycle"]["first_seen_run_id"]
            if existing_row and not canonical_update:
                assert existing is not None
                canonical_item = copy.deepcopy(existing)
                canonical_item["lifecycle"]["last_seen_at"] = item["lifecycle"][
                    "last_seen_at"
                ]
                canonical_item["lifecycle"]["last_seen_run_id"] = run_id
                canonical_item["lifecycle"]["latest_source_version"] = True
                canonical_fingerprint = existing_row["meaningful_fingerprint"]
            else:
                canonical_item = copy.deepcopy(item)
                if (
                    existing is not None
                    and existing["lifecycle"]["last_seen_run_id"] == run_id
                ):
                    for field in (
                        "retrieved_at",
                        "raw_artifact_path",
                        "raw_artifact_sha256",
                    ):
                        canonical_item["provenance"][field] = existing["provenance"][
                            field
                        ]
                canonical_fingerprint = fingerprint
            validate_json(item, normalized_schema_path)
            validate_json(canonical_item, normalized_schema_path)
            canonical_json = json.dumps(canonical_item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            bibliography = canonical_item["bibliographic"]
            canonical_identity = canonical_item["identity"]
            links = canonical_item["links"]
            relevance = canonical_item["relevance"]
            review = canonical_item["review"]
            lifecycle = canonical_item["lifecycle"]
            item_values = {
                "item_id": item_id,
                "primary_source": "arxiv",
                "primary_key_type": "arxiv",
                "primary_key_value": identity["base_arxiv_id"],
                "base_arxiv_id": identity["base_arxiv_id"],
                "current_version": canonical_identity["current_arxiv_version"],
                "doi": canonical_identity["doi"],
                "title": bibliography["title"],
                "abstract": bibliography["abstract"],
                "authors_json": json.dumps(
                    bibliography["authors"], ensure_ascii=False, separators=(",", ":")
                ),
                "language": bibliography["language"],
                "primary_category": bibliography["primary_category"],
                "categories_json": json.dumps(
                    bibliography["categories"], ensure_ascii=False, separators=(",", ":")
                ),
                "submitted_at": bibliography["submitted_at"],
                "source_updated_at": bibliography["source_updated_at"],
                "journal_reference": bibliography["journal_reference"],
                "comments": bibliography["comments"],
                "license": bibliography["license"],
                "source_record_url": links["source_record_url"],
                "pdf_url": links["pdf_url"],
                "doi_url": links["doi_url"],
                "relevance_score": relevance["score"],
                "relevance_priority": relevance["priority"],
                "eligible": int(relevance["eligible"]),
                "matched_rule_ids_json": json.dumps(
                    relevance["matched_rule_ids"], separators=(",", ":")
                ),
                "penalty_rule_ids_json": json.dumps(
                    relevance["penalty_rule_ids"], separators=(",", ":")
                ),
                "review_status": review["status"],
                "review_priority": review["priority"],
                "reviewer": review["reviewer"],
                "reviewed_at": review["reviewed_at"],
                "tags_json": json.dumps(
                    review["tags"], ensure_ascii=False, separators=(",", ":")
                ),
                "notes": review["notes"],
                "first_seen_at": lifecycle["first_seen_at"],
                "last_seen_at": lifecycle["last_seen_at"],
                "first_seen_run_id": lifecycle["first_seen_run_id"],
                "last_seen_run_id": lifecycle["last_seen_run_id"],
                "meaningful_fingerprint": canonical_fingerprint,
                "canonical_json": canonical_json,
            }
            connection.execute(
                """INSERT INTO items(
                item_id,primary_source,primary_key_type,primary_key_value,base_arxiv_id,
                current_version,doi,title,abstract,authors_json,language,primary_category,
                categories_json,submitted_at,source_updated_at,journal_reference,comments,
                license,source_record_url,pdf_url,doi_url,relevance_score,relevance_priority,
                eligible,matched_rule_ids_json,penalty_rule_ids_json,review_status,
                review_priority,reviewer,reviewed_at,tags_json,notes,first_seen_at,last_seen_at,
                first_seen_run_id,last_seen_run_id,meaningful_fingerprint,canonical_json)
                VALUES(
                :item_id,:primary_source,:primary_key_type,:primary_key_value,:base_arxiv_id,
                :current_version,:doi,:title,:abstract,:authors_json,:language,:primary_category,
                :categories_json,:submitted_at,:source_updated_at,:journal_reference,:comments,
                :license,:source_record_url,:pdf_url,:doi_url,:relevance_score,
                :relevance_priority,:eligible,:matched_rule_ids_json,:penalty_rule_ids_json,
                :review_status,:review_priority,:reviewer,:reviewed_at,:tags_json,:notes,
                :first_seen_at,:last_seen_at,:first_seen_run_id,:last_seen_run_id,
                :meaningful_fingerprint,:canonical_json)
                ON CONFLICT(item_id) DO UPDATE SET
                current_version=excluded.current_version,doi=excluded.doi,title=excluded.title,
                abstract=excluded.abstract,authors_json=excluded.authors_json,
                language=excluded.language,primary_category=excluded.primary_category,
                categories_json=excluded.categories_json,submitted_at=excluded.submitted_at,
                source_updated_at=excluded.source_updated_at,
                journal_reference=excluded.journal_reference,comments=excluded.comments,
                license=excluded.license,source_record_url=excluded.source_record_url,
                pdf_url=excluded.pdf_url,doi_url=excluded.doi_url,
                relevance_score=excluded.relevance_score,
                relevance_priority=excluded.relevance_priority,eligible=excluded.eligible,
                matched_rule_ids_json=excluded.matched_rule_ids_json,
                penalty_rule_ids_json=excluded.penalty_rule_ids_json,
                review_status=excluded.review_status,review_priority=excluded.review_priority,
                reviewer=excluded.reviewer,reviewed_at=excluded.reviewed_at,
                tags_json=excluded.tags_json,notes=excluded.notes,
                last_seen_at=excluded.last_seen_at,last_seen_run_id=excluded.last_seen_run_id,
                meaningful_fingerprint=excluded.meaningful_fingerprint,
                canonical_json=excluded.canonical_json""",
                item_values,
            )
            connection.execute(
                "INSERT OR IGNORE INTO item_revisions(item_id,arxiv_version,first_observed_run_id,raw_artifact_path,raw_artifact_sha256,source_fingerprint,normalized_json) VALUES(?,?,?,?,?,?,?)",
                (item_id, observed_version, run_id, observation["raw_path"], observation["raw_hash"], observation["source_fingerprint"], json.dumps(item, sort_keys=True, separators=(",", ":"))),
            )
            connection.execute(
                "INSERT OR IGNORE INTO discoveries(run_id,item_id,source_id,query_id,source_record_id,disposition,observed_version,raw_artifact_path,raw_artifact_sha256,observation_position,normalized_snapshot_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, item_id, "arxiv", observation["query_id"], item["provenance"]["source_record_id"], disposition, observed_version, observation["raw_path"], observation["raw_hash"], position, json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=False)),
            )
            doi = identity["doi"]
            if doi:
                for collision in connection.execute("SELECT item_id FROM items WHERE doi=? AND item_id<>?", (doi, item_id)):
                    pair = tuple(sorted((item_id, collision["item_id"])))
                    key = (doi, pair[0], pair[1])
                    if key not in seen_doi_pairs:
                        seen_doi_pairs.add(key)
                        counts["doi_collisions"] += 1
                        append_event(connection, run_id, "WARNING", "DOI_COLLISION", "CATALOG", {"doi": doi, "item_ids": list(pair)})
        connection.execute(
            "UPDATE runs SET catalog_committed=1,catalog_committed_at_utc=?,queue_render_timestamp_utc=?,queue_renderer_version=?,markdown_path=?,jsonl_path=? WHERE run_id=?",
            (utc_now(), queue_timestamp, renderer_version, markdown_path, jsonl_path, run_id),
        )
        connection.commit()
        return counts
    except Exception:
        connection.rollback()
        raise


def discovery_queue_items(connection: sqlite3.Connection, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT item_id,disposition,normalized_snapshot_json FROM discoveries WHERE run_id=? AND disposition IN ('NEW','UPDATED') ORDER BY observation_position",
        (run_id,),
    )
    seen: set[str] = set()
    items = []
    for row in rows:
        if row["item_id"] in seen:
            continue
        seen.add(row["item_id"])
        item = json.loads(row["normalized_snapshot_json"])
        if item["relevance"]["eligible"]:
            items.append(item)
    return items
