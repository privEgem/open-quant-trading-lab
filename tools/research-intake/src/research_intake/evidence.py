from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from research_intake.validation import validate_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(131072), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: dict[str, Any], schema_path: Path | None = None) -> None:
    if schema_path:
        validate_json(value, schema_path)
    atomic_write(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode())


def preserve_response(
    body: bytes,
    body_path: Path,
    provenance: dict[str, Any],
    provenance_schema: Path,
) -> tuple[str, Path]:
    atomic_write(body_path, body)
    digest = sha256_file(body_path)
    sidecar = body_path.with_suffix(body_path.suffix + ".provenance.json")
    record = {**provenance, "body_path": body_path.as_posix(), "body_sha256": digest}
    atomic_json(sidecar, record, provenance_schema)
    return digest, sidecar
