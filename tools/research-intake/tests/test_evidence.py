from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research_intake.arxiv import parse_atom
from research_intake.evidence import atomic_write, preserve_response


def provenance() -> dict[str, object]:
    return {"schema_version": 1, "source_id": "arxiv", "query_id": "q", "request": {"method": "GET", "url": "https://example.test", "parameters": {}}, "response": {"status": 200, "headers": {"Content-Type": "application/atom+xml"}}, "retrieved_at": "2026-07-12T00:00:00Z", "attempt_number": 1}


def test_saved_bytes_are_not_changed_by_parsing(project_root: Path, mixed_xml: bytes, tmp_path: Path) -> None:
    target = tmp_path / "response.xml"
    digest, sidecar = preserve_response(mixed_xml, target, provenance(), project_root / "schemas" / "raw_response_provenance.schema.json")
    before = target.read_bytes()
    parse_atom(before)
    assert target.read_bytes() == mixed_xml
    assert digest == hashlib.sha256(mixed_xml).hexdigest()
    assert json.loads(sidecar.read_text())["body_sha256"] == digest


def test_atomic_write_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "immutable"
    atomic_write(target, b"first")
    with pytest.raises(FileExistsError):
        atomic_write(target, b"second")
