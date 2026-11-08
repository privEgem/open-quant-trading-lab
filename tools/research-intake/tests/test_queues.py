from __future__ import annotations

import json
from pathlib import Path

from research_intake.arxiv import parse_atom
from research_intake.config import load_config
from research_intake.normalization import normalize_observation
from research_intake.queues import render_queue_bytes
from research_intake.relevance import score_item


def item(project_root: Path, mixed_xml: bytes, index: int) -> dict[str, object]:
    entry = parse_atom(mixed_xml).entries[index]
    config = load_config(project_root)
    relevance = score_item(entry, config.topics, "category_unconditional")
    return normalize_observation(entry, "run", ["q"], "2026-07-12T00:00:00Z", "raw.xml", "a" * 64, relevance)


def test_deterministic_queue_and_jsonl_context(project_root: Path, mixed_xml: bytes) -> None:
    items = [item(project_root, mixed_xml, 1), item(project_root, mixed_xml, 0)]
    first = render_queue_bytes(items, "run", "2026-07-12T01:00:00Z", project_root / "schemas" / "review_queue_item.schema.json")
    second = render_queue_bytes(list(reversed(items)), "run", "2026-07-12T01:00:00Z", project_root / "schemas" / "review_queue_item.schema.json")
    assert first == second
    records = [json.loads(line) for line in first[1].decode().splitlines()]
    assert records[0]["queue_context"]["queue_render_timestamp_utc"] == "2026-07-12T01:00:00Z"
    assert first[0].count(b"## ") == len(records)


def test_high_priority_sorts_first(project_root: Path, mixed_xml: bytes) -> None:
    high = item(project_root, mixed_xml, 0)
    low = item(project_root, mixed_xml, 1)
    markdown, _ = render_queue_bytes([low, high], "run", "2026-07-12T01:00:00Z", project_root / "schemas" / "review_queue_item.schema.json")
    assert markdown.find(high["bibliographic"]["title"].encode()) < markdown.find(low["bibliographic"]["title"].encode())
