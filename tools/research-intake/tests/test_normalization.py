from __future__ import annotations

import copy
from pathlib import Path

from research_intake.arxiv import parse_atom
from research_intake.config import load_config
from research_intake.normalization import normalize_observation
from research_intake.relevance import score_item


def _normalize(project_root: Path, source: dict[str, object]):
    config = load_config(project_root)
    relevance = score_item(source, config.topics, "category_unconditional")
    return normalize_observation(
        source,
        "run",
        ["query"],
        "2026-07-12T00:00:00Z",
        "raw.xml",
        "a" * 64,
        relevance,
    )


def test_source_doi_does_not_change_stable_item_identity(
    project_root: Path, mixed_xml: bytes
) -> None:
    provided = parse_atom(mixed_xml).entries[0]
    absent = copy.deepcopy(provided)
    absent["doi"] = None
    with_doi = _normalize(project_root, provided)
    without_doi = _normalize(project_root, absent)
    assert with_doi["identity"]["item_id"] == without_doi["identity"]["item_id"]
    assert with_doi["identity"]["doi_status"] == "provided_by_primary_source"
    assert without_doi["identity"]["doi_status"] == "not_provided"


def test_whitespace_and_optional_fields_are_normalized(
    project_root: Path, mixed_xml: bytes
) -> None:
    source = copy.deepcopy(parse_atom(mixed_xml).entries[1])
    source["title"] = "  Liquidity\n   and   Intraday Reversal  "
    source["abstract"] = "Order\nflow   and spreads."
    source["authors"] = ["  Carol   Lee "]
    normalized = _normalize(project_root, source)
    assert normalized["bibliographic"]["title"] == "Liquidity and Intraday Reversal"
    assert normalized["bibliographic"]["abstract"] == "Order flow and spreads."
    assert normalized["bibliographic"]["authors"] == ["Carol Lee"]
    assert normalized["bibliographic"]["journal_reference"] is None
    assert normalized["bibliographic"]["language"] == "und"
