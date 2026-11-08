import copy
from pathlib import Path

import pytest

from research_intake.arxiv import parse_atom
from research_intake.config import load_config
from research_intake.relevance import score_item


@pytest.mark.parametrize(
    ("index", "mode", "eligible"),
    [(0, "category_unconditional", True), (1, "topic_required", True), (2, "topic_required", False)],
)
def test_eligibility(project_root: Path, mixed_xml: bytes, index: int, mode: str, eligible: bool) -> None:
    config = load_config(project_root)
    result = score_item(parse_atom(mixed_xml).entries[index], config.topics, mode)
    assert result["eligible"] is eligible


def test_primary_score_and_caps(project_root: Path, mixed_xml: bytes) -> None:
    config = load_config(project_root)
    result = score_item(parse_atom(mixed_xml).entries[0], config.topics, "category_unconditional")
    assert result["score"] <= 100
    assert result["score"] >= 70
    assert result["priority"] == "HIGH"


def test_rule_order_is_deterministic(project_root: Path, mixed_xml: bytes) -> None:
    config = load_config(project_root)
    result = score_item(parse_atom(mixed_xml).entries[0], config.topics, "category_unconditional")
    assert result["matched_rule_ids"] == sorted(result["matched_rule_ids"])


def test_phrase_rule_ids_are_stable_when_configuration_is_reordered(
    project_root: Path, mixed_xml: bytes
) -> None:
    config = load_config(project_root)
    entry = parse_atom(mixed_xml).entries[0]
    original = score_item(entry, config.topics, "category_unconditional")
    reordered = copy.deepcopy(config.topics)
    reordered["topics"].reverse()
    for topic in reordered["topics"]:
        topic["phrases"].reverse()
    changed_order = score_item(entry, reordered, "category_unconditional")
    assert changed_order["matched_rule_ids"] == original["matched_rule_ids"]
    assert changed_order["contributions"] == original["contributions"]


def test_generic_ml_penalty(project_root: Path, mixed_xml: bytes) -> None:
    config = load_config(project_root)
    result = score_item(parse_atom(mixed_xml).entries[2], config.topics, "topic_required")
    assert "penalty.generic_ml_only" in result["penalty_rule_ids"]


def test_proprietary_and_unrelated_penalties_are_explicit(
    project_root: Path, mixed_xml: bytes
) -> None:
    config = load_config(project_root)
    proprietary = copy.deepcopy(parse_atom(mixed_xml).entries[0])
    proprietary["abstract"] = "This paper depends on a proprietary dataset."
    proprietary_result = score_item(
        proprietary, config.topics, "category_unconditional"
    )
    assert "penalty.proprietary_data" in proprietary_result["penalty_rule_ids"]
    unrelated = copy.deepcopy(parse_atom(mixed_xml).entries[2])
    unrelated["title"] = "Medical Imaging Forecast"
    unrelated["abstract"] = "A domain-specific forecast."
    unrelated_result = score_item(unrelated, config.topics, "category_unconditional")
    assert "penalty.unrelated_domain" in unrelated_result["penalty_rule_ids"]
    assert unrelated_result["eligible"] is False


def test_phrase_caps_and_priority_bands(
    project_root: Path, mixed_xml: bytes
) -> None:
    config = load_config(project_root)
    capped = copy.deepcopy(parse_atom(mixed_xml).entries[0])
    capped["title"] = (
        "Liquidity order flow execution transaction cost slippage market impact"
    )
    capped["abstract"] = (
        "Market microstructure, limit order book, bid-ask spread, time of day, "
        "walk-forward, out-of-sample, multiple testing and data leakage."
    )
    capped_result = score_item(capped, config.topics, "category_unconditional")
    assert sum(
        contribution["points"] == 12
        for contribution in capped_result["contributions"]
    ) == 3
    assert sum(
        contribution["points"] == 4
        for contribution in capped_result["contributions"]
    ) == 5
    medium = copy.deepcopy(parse_atom(mixed_xml).entries[0])
    medium["title"] = "Liquidity"
    medium["abstract"] = "A neutral study."
    medium_result = score_item(medium, config.topics, "category_unconditional")
    assert medium_result["score"] == 47
    assert medium_result["priority"] == "MEDIUM"
    low = copy.deepcopy(medium)
    low["title"] = "A neutral study"
    low_result = score_item(low, config.topics, "category_unconditional")
    assert low_result["score"] == 35
    assert low_result["priority"] == "LOW"
