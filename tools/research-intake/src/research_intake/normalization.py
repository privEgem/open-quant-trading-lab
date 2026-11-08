from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from research_intake.identity import item_id_for, split_arxiv_identifier


def clean_text(value: str | None) -> str | None:
    return re.sub(r"\s+", " ", value).strip() if value else None


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if candidate.startswith(prefix):
            candidate = candidate[len(prefix) :]
    return candidate or None


def source_fingerprint(record: dict[str, Any]) -> str:
    meaningful = {key: record.get(key) for key in ("title", "abstract", "authors", "categories", "primary_category", "published", "updated", "doi", "journal_ref", "comments", "license", "links")}
    return hashlib.sha256(json.dumps(meaningful, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalize_observation(
    source: dict[str, Any],
    run_id: str,
    query_ids: list[str],
    retrieved_at: str,
    raw_path: str,
    raw_hash: str,
    relevance: dict[str, Any],
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base, version = split_arxiv_identifier(source["source_record_id"])
    item_id = item_id_for(base)
    doi = normalize_doi(source.get("doi"))
    alternate = next((link.get("href") for link in source["links"] if link.get("rel") == "alternate"), source["source_record_id"])
    pdf = next((link.get("href") for link in source["links"] if link.get("title") == "pdf" or link.get("type") == "application/pdf"), None)
    old_review = existing.get("review") if existing else None
    old_lifecycle = existing.get("lifecycle") if existing else None
    review = old_review or {"status": "NEW", "priority": relevance["priority"] if relevance["eligible"] else "NONE", "reviewer": None, "reviewed_at": None, "tags": [], "notes": None}
    lifecycle = {
        "first_seen_at": old_lifecycle["first_seen_at"] if old_lifecycle else retrieved_at,
        "last_seen_at": retrieved_at,
        "first_seen_run_id": old_lifecycle["first_seen_run_id"] if old_lifecycle else run_id,
        "last_seen_run_id": run_id,
        "latest_source_version": True,
    }
    return {
        "identity": {"schema_version": 1, "item_id": item_id, "record_type": "research_paper", "primary_key_type": "arxiv", "primary_key_value": base, "base_arxiv_id": base, "current_arxiv_version": version, "doi": doi, "doi_status": "provided_by_primary_source" if doi else "not_provided"},
        "bibliographic": {"title": clean_text(source["title"]), "authors": [clean_text(author) for author in source["authors"]], "abstract": clean_text(source["abstract"]), "language": source.get("language") or "und", "primary_category": source["primary_category"], "categories": source["categories"], "submitted_at": source["published"], "source_updated_at": source["updated"], "journal_reference": clean_text(source.get("journal_ref")), "comments": clean_text(source.get("comments")), "license": source.get("license")},
        "links": {"source_record_url": alternate, "pdf_url": pdf, "doi_url": f"https://doi.org/{doi}" if doi else None},
        "provenance": {"primary_source": "arxiv", "source_record_id": source["source_record_id"], "discovery_query_ids": sorted(set(query_ids)), "retrieved_at": retrieved_at, "run_id": run_id, "raw_artifact_path": raw_path, "raw_artifact_sha256": raw_hash, "normalizer_version": "1"},
        "relevance": relevance,
        "lifecycle": lifecycle,
        "review": review,
    }
