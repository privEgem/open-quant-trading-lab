from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest

from research_intake.config import (
    ConfigurationError,
    _semantic_validate,
    _validate_schema,
    contact_email,
    load_config,
)
from research_intake.paths import resolve_paths


def test_valid_configuration(project_root: Path) -> None:
    config = load_config(project_root)
    assert len(config.enabled_queries) == 2
    assert len(config.registry_hash) == 64


def test_duplicate_source_id_rejected(project_root: Path) -> None:
    config = load_config(project_root)
    registry = copy.deepcopy(config.registry)
    registry["sources"].append(copy.deepcopy(registry["sources"][0]))
    with pytest.raises(ConfigurationError, match="duplicate source"):
        _semantic_validate(registry, config.topics)


def test_duplicate_query_id_rejected(project_root: Path) -> None:
    config = load_config(project_root)
    registry = copy.deepcopy(config.registry)
    registry["sources"][0]["queries"].append(copy.deepcopy(registry["sources"][0]["queries"][0]))
    with pytest.raises(ConfigurationError, match="duplicate query"):
        _semantic_validate(registry, config.topics)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda registry: registry.pop("application"), "application"),
        (
            lambda registry: registry["policy"].__setitem__(
                "maximum_requests_per_run", 0
            ),
            "minimum of 1",
        ),
        (
            lambda registry: registry["sources"][0]["queries"][0].__setitem__(
                "eligibility_mode", "semantic_guess"
            ),
            "is not one of",
        ),
    ],
)
def test_source_registry_schema_rejects_missing_and_invalid_values(
    project_root: Path, mutation, expected: str
) -> None:
    config = load_config(project_root)
    registry = copy.deepcopy(config.registry)
    mutation(registry)
    with pytest.raises(ConfigurationError, match=expected):
        _validate_schema(
            registry,
            project_root / "schemas" / "source_registry.schema.json",
            "source registry",
        )


def test_unsupported_access_method_rejected(project_root: Path) -> None:
    config = load_config(project_root)
    registry = copy.deepcopy(config.registry)
    registry["sources"][0]["access_method"] = "browser"
    with pytest.raises(ConfigurationError, match="unsupported access method"):
        _semantic_validate(registry, config.topics)


def test_enabled_query_cannot_expand_category_scope(project_root: Path) -> None:
    config = load_config(project_root)
    registry = copy.deepcopy(config.registry)
    registry["sources"][0]["queries"][0]["expression"] = "cat:cs.LG"
    with pytest.raises(ConfigurationError, match="outside its approved scope"):
        _semantic_validate(registry, config.topics)


def test_enabled_source_requires_official_arxiv_endpoint(project_root: Path) -> None:
    config = load_config(project_root)
    registry = copy.deepcopy(config.registry)
    registry["sources"][0]["endpoint"] = "https://example.test/api"
    with pytest.raises(ConfigurationError, match="official HTTPS"):
        _semantic_validate(registry, config.topics)


def test_disabled_query_is_not_executable(project_root: Path) -> None:
    config = load_config(project_root)
    config.registry["sources"][0]["queries"][1]["enabled"] = False
    assert [query["id"] for _, query in config.enabled_queries] == [
        "arxiv_core_qfin_tr"
    ]


def test_enabled_source_requires_query(project_root: Path) -> None:
    config = load_config(project_root)
    registry = copy.deepcopy(config.registry)
    for query in registry["sources"][0]["queries"]:
        query["enabled"] = False
    with pytest.raises(ConfigurationError, match="no enabled query"):
        _semantic_validate(registry, config.topics)


def test_required_topic_cannot_be_removed(project_root: Path) -> None:
    config = load_config(project_root)
    topics = copy.deepcopy(config.topics)
    topics["topics"] = [
        topic for topic in topics["topics"] if topic["id"] != "market_microstructure"
    ]
    with pytest.raises(ConfigurationError, match="missing required topic"):
        _semantic_validate(config.registry, topics)


def test_retry_attempts_cannot_exceed_run_request_limit(project_root: Path) -> None:
    config = load_config(project_root)
    registry = copy.deepcopy(config.registry)
    registry["policy"]["maximum_requests_per_run"] = 2
    registry["policy"]["retry"]["max_attempts"] = 3
    with pytest.raises(ConfigurationError, match="max_attempts"):
        _semantic_validate(registry, config.topics)


def test_retry_statuses_reject_success_codes(project_root: Path) -> None:
    config = load_config(project_root)
    registry = copy.deepcopy(config.registry)
    registry["policy"]["retry"]["retry_statuses"] = [200]
    with pytest.raises(ConfigurationError, match="HTTP 429 and 5xx"):
        _semantic_validate(registry, config.topics)


def test_contact_email_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RESEARCH_INTAKE_CONTACT_EMAIL", raising=False)
    assert contact_email() is None


def test_contact_email_rejects_header_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_INTAKE_CONTACT_EMAIL", "a@example.test\r\nInjected: yes")
    with pytest.raises(ConfigurationError):
        contact_email()


def test_external_data_root_is_explicit(
    project_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with tempfile.TemporaryDirectory(prefix="research-intake-external-") as directory:
        external = Path(directory)
        monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(external))
        assert resolve_paths(project_root).root == external.resolve()


def test_default_data_root_is_caller_local(
    project_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("RESEARCH_INTAKE_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_paths(project_root).root == (tmp_path / ".research-intake-data").resolve()
    assert resolve_paths(project_root).root != (project_root / "data").resolve()
