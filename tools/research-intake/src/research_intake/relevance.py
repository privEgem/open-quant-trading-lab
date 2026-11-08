from __future__ import annotations

import re
import unicodedata
from typing import Any


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _matches(text: str, phrase: str) -> bool:
    target = _normalized(text)
    needle = _normalized(phrase)
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", target) is not None


def score_item(record: dict[str, Any], topics: dict[str, Any], eligibility_mode: str) -> dict[str, Any]:
    contributions: list[dict[str, Any]] = []
    matched: list[str] = []
    penalties: list[str] = []
    primary = record["primary_category"]
    if primary == "q-fin.TR":
        contributions.append({"rule_id": "category.primary.q_fin_tr", "points": 35})
    elif primary in {"q-fin.ST", "q-fin.CP"}:
        contributions.append({"rule_id": "category.primary.secondary", "points": 15})
    title_rules: list[str] = []
    abstract_rules: list[str] = []
    bonus = False
    for topic in topics["topics"]:
        for phrase in topic["phrases"]:
            base_id = f"topic.{topic['id']}.{phrase['id']}"
            if _matches(record["title"], phrase["text"]):
                title_rules.append(f"title.{base_id}")
                matched.append(f"title.{base_id}")
                bonus = bonus or bool(topic.get("explicit_bonus"))
            if _matches(record["abstract"], phrase["text"]):
                abstract_rules.append(f"abstract.{base_id}")
                matched.append(f"abstract.{base_id}")
                bonus = bonus or bool(topic.get("explicit_bonus"))
    for rule_id in sorted(title_rules)[:3]:
        contributions.append({"rule_id": rule_id, "points": 12})
    for rule_id in sorted(abstract_rules)[:5]:
        contributions.append({"rule_id": rule_id, "points": 4})
    if bonus:
        contributions.append({"rule_id": "bonus.explicit_validation_execution_friction", "points": 10})
    combined = f"{record['title']} {record['abstract']}"
    positive = bool(title_rules or abstract_rules)
    penalty_config = topics["penalties"]
    generic = any(_matches(combined, phrase) for phrase in penalty_config["generic_ml_only"]["phrases"])
    if generic and not positive:
        penalties.append(penalty_config["generic_ml_only"]["id"])
        contributions.append({"rule_id": penalties[-1], "points": -20})
    for key, points in (("proprietary_data", -15), ("unrelated_domain", -15)):
        rule = penalty_config[key]
        penalty_matches = any(_matches(combined, phrase) for phrase in rule["phrases"])
        if key == "unrelated_domain":
            penalty_matches = not positive and any(
                _matches(record["title"], phrase) for phrase in rule["phrases"]
            )
        if penalty_matches:
            penalties.append(rule["id"])
            contributions.append({"rule_id": rule["id"], "points": points})
    score = min(100, sum(int(part["points"]) for part in contributions))
    gate = eligibility_mode == "category_unconditional" or positive
    eligible = gate and score > 0
    priority = "EXCLUDED" if not eligible else "HIGH" if score >= 70 else "MEDIUM" if score >= 40 else "LOW"
    ordered = sorted(set(matched))
    ordered_penalties = sorted(set(penalties))
    explanation = "; ".join(f"{part['rule_id']} {part['points']:+d}" for part in contributions) or "no configured rule matched"
    return {
        "eligible": eligible,
        "score": score,
        "priority": priority,
        "matched_rule_ids": ordered,
        "penalty_rule_ids": ordered_penalties,
        "contributions": contributions,
        "explanation": explanation,
    }
