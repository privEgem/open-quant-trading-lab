from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


def validate_json(value: dict[str, Any], schema_path: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    item_schema = schema.get("properties", {}).get("item", {})
    if item_schema.get("$ref") == "normalized_item.schema.json":
        schema["properties"]["item"] = json.loads(
            (schema_path.parent / "normalized_item.schema.json").read_text(encoding="utf-8")
        )
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(value)
