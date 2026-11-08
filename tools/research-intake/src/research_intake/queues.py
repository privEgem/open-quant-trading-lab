from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from research_intake.evidence import atomic_write
from research_intake.validation import validate_json

RENDERER_VERSION = "markdown-jsonl-v1"
PRIORITY = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "EXCLUDED": 3}


def sorted_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        items,
        key=lambda item: (
            PRIORITY[item["relevance"]["priority"]],
            -int(item["relevance"]["score"]),
            -int(item["bibliographic"]["submitted_at"].replace("-", "").replace(":", "").replace("T", "").replace("Z", "").replace("+00:00", "").replace(".", "")[:14]),
            item["bibliographic"]["title"].casefold(),
            item["identity"]["item_id"],
        ),
    )


def render_queue_bytes(items: list[dict[str, Any]], run_id: str, render_timestamp: str, schema_path: Path) -> tuple[bytes, bytes]:
    ordered = sorted_items(items)
    priority_counts = {
        priority: sum(
            item["relevance"]["priority"] == priority for item in ordered
        )
        for priority in ("HIGH", "MEDIUM", "LOW")
    }
    lines = [
        f"# Research Intake Review Queue — {run_id}",
        "",
        f"Generated: {render_timestamp}",
        f"Items: {len(ordered)}",
        "Priority counts: "
        + ", ".join(
            f"{priority}={priority_counts[priority]}"
            for priority in ("HIGH", "MEDIUM", "LOW")
        ),
        "",
    ]
    jsonl: list[str] = []
    for item in ordered:
        bib = item["bibliographic"]
        relevance = item["relevance"]
        review = item["review"]
        lines.extend(
            [
                f"## {bib['title']}", "", f"- Item ID: `{item['identity']['item_id']}`",
                f"- Authors: {', '.join(bib['authors'])}", f"- Submitted: {bib['submitted_at']}",
                f"- Categories: {', '.join(bib['categories'])}",
                f"- Relevance: {relevance['score']} ({relevance['priority']})",
                f"- Human review status: {review['status']}",
                f"- Matched rules: {', '.join(relevance['matched_rule_ids']) or 'none'}",
                f"- Penalties: {', '.join(relevance['penalty_rule_ids']) or 'none'}",
                f"- Source: {item['links']['source_record_url']}", "", bib["abstract"], "",
            ]
        )
        queue_record = {"queue_context": {"run_id": run_id, "queue_render_timestamp_utc": render_timestamp, "queue_renderer_version": RENDERER_VERSION}, "item": item}
        validate_json(queue_record, schema_path)
        jsonl.append(json.dumps(queue_record, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    markdown_bytes = ("\n".join(lines).rstrip() + "\n").encode("utf-8")
    jsonl_bytes = (("\n".join(jsonl) + "\n") if jsonl else "").encode("utf-8")
    return markdown_bytes, jsonl_bytes


def publish_queues(markdown_path: Path, jsonl_path: Path, markdown: bytes, jsonl: bytes) -> dict[str, Any]:
    atomic_write(markdown_path, markdown)
    atomic_write(jsonl_path, jsonl)
    return {
        "markdown": {"path": markdown_path.as_posix(), "sha256": hashlib.sha256(markdown).hexdigest(), "size": len(markdown)},
        "jsonl": {"path": jsonl_path.as_posix(), "sha256": hashlib.sha256(jsonl).hexdigest(), "size": len(jsonl)},
    }
