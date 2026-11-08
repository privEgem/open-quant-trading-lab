from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator


def test_all_checked_in_json_schemas_are_valid(project_root: Path) -> None:
    schema_paths = sorted((project_root / "schemas").glob("*.schema.json"))
    assert {path.name for path in schema_paths} == {
        "human_review_batch.schema.json",
        "normalized_item.schema.json",
        "raw_response_provenance.schema.json",
        "recovery_receipt.schema.json",
        "review_queue_item.schema.json",
        "run_manifest.schema.json",
        "source_registry.schema.json",
        "topics.schema.json",
    }
    for path in schema_paths:
        Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))
