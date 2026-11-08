from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from jsonschema import Draft202012Validator, FormatChecker


class ConfigurationError(ValueError):
    pass


REQUIRED_TOPIC_IDS = {
    "market_microstructure",
    "intraday_mean_reversion",
    "intraday_reversal",
    "liquidity",
    "order_flow",
    "bid_ask_spreads",
    "execution",
    "transaction_costs",
    "slippage",
    "time_of_day_effects",
    "etf_index_microstructure",
    "backtest_overfitting",
    "selection_bias",
    "data_leakage",
    "out_of_sample",
    "walk_forward",
    "multiple_testing",
    "deflated_sharpe",
}


@dataclass(frozen=True)
class LoadedConfig:
    registry: dict[str, Any]
    topics: dict[str, Any]
    registry_hash: str
    topics_hash: str
    project_root: Path

    @property
    def enabled_sources(self) -> list[dict[str, Any]]:
        return [source for source in self.registry["sources"] if source["enabled"]]

    @property
    def enabled_queries(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        return [
            (source, query)
            for source in self.enabled_sources
            for query in source["queries"]
            if query["enabled"]
        ]


def _load_yaml(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = yaml.safe_load(raw)
    if not isinstance(value, dict):
        raise ConfigurationError(f"{path.name} must contain a mapping")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_schema(value: dict[str, Any], schema_path: Path, label: str) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(value),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        message = "; ".join(
            f"{'.'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in errors
        )
        raise ConfigurationError(f"invalid {label}: {message}")


def _duplicates(values: list[str]) -> list[str]:
    return sorted({value for value in values if values.count(value) > 1})


def _semantic_validate(registry: dict[str, Any], topics: dict[str, Any]) -> None:
    policy = registry["policy"]
    if policy["retry"]["max_attempts"] > policy["maximum_requests_per_run"]:
        raise ConfigurationError(
            "retry max_attempts cannot exceed maximum_requests_per_run"
        )
    if any(
        status != 429 and not 500 <= status <= 599
        for status in policy["retry"]["retry_statuses"]
    ):
        raise ConfigurationError(
            "retry statuses are limited to HTTP 429 and 5xx responses"
        )
    source_ids = [str(source["id"]) for source in registry["sources"]]
    if duplicates := _duplicates(source_ids):
        raise ConfigurationError(f"duplicate source IDs: {', '.join(duplicates)}")
    query_ids = [str(query["id"]) for source in registry["sources"] for query in source["queries"]]
    if duplicates := _duplicates(query_ids):
        raise ConfigurationError(f"duplicate query IDs: {', '.join(duplicates)}")
    for source in registry["sources"]:
        if source["access_method"] != "arxiv_atom":
            raise ConfigurationError(f"unsupported access method: {source['access_method']}")
        if source["enabled"] and source["id"] != "arxiv":
            raise ConfigurationError("arxiv is the only supported enabled source ID")
        endpoint = urlparse(source["endpoint"])
        if source["enabled"] and (
            endpoint.scheme != "https"
            or endpoint.hostname != "export.arxiv.org"
            or endpoint.path.rstrip("/") != "/api/query"
        ):
            raise ConfigurationError(
                "enabled arXiv source must use the official HTTPS API endpoint"
            )
        if source["enabled"] and not any(query["enabled"] for query in source["queries"]):
            raise ConfigurationError(f"enabled source {source['id']} has no enabled query")
        if source["enabled"]:
            collection = source["collection"]
            if not collection["metadata"] or not collection["abstracts"] or collection["pdfs"]:
                raise ConfigurationError("enabled arXiv source requires metadata/abstracts and forbids PDFs")
        for query in source["queries"]:
            if not query["enabled"]:
                continue
            categories = set(
                re.findall(r"cat:([A-Za-z0-9.-]+)", query["expression"])
            )
            expected = (
                {"q-fin.TR"}
                if query["eligibility_mode"] == "category_unconditional"
                else {"q-fin.ST", "q-fin.CP"}
            )
            if not categories or not categories.issubset(expected):
                raise ConfigurationError(
                    f"query {query['id']} contains categories outside its approved scope"
                )
    topic_ids = [str(topic["id"]) for topic in topics["topics"]]
    missing_topics = sorted(REQUIRED_TOPIC_IDS.difference(topic_ids))
    if missing_topics:
        raise ConfigurationError(
            f"missing required topic IDs: {', '.join(missing_topics)}"
        )
    phrase_ids = [
        f"{topic['id']}.{phrase['id']}"
        for topic in topics["topics"]
        for phrase in topic["phrases"]
    ]
    penalty_ids = [str(rule["id"]) for rule in topics["penalties"].values()]
    if _duplicates(topic_ids + phrase_ids + penalty_ids):
        raise ConfigurationError("duplicate topic or rule IDs")


def load_config(project_root: Path) -> LoadedConfig:
    registry, registry_hash = _load_yaml(project_root / "config" / "source_registry.yaml")
    topics, topics_hash = _load_yaml(project_root / "config" / "topics.yaml")
    _validate_schema(registry, project_root / "schemas" / "source_registry.schema.json", "source registry")
    _validate_schema(topics, project_root / "schemas" / "topics.schema.json", "topics")
    _semantic_validate(registry, topics)
    return LoadedConfig(registry, topics, registry_hash, topics_hash, project_root)


def contact_email() -> str | None:
    value = os.environ.get("RESEARCH_INTAKE_CONTACT_EMAIL", "").strip()
    if not value:
        return None
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ConfigurationError("RESEARCH_INTAKE_CONTACT_EMAIL contains control characters")
    return value
