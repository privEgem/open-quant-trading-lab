"""Fail-closed Research OS Experiment Record V0 utilities.

This module records one bounded research experiment.  It deliberately does not
claim to provide a repository-wide trial ledger, retention system, publication
workflow, alpha confirmation, or promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from numbers import Integral, Real
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "research_experiment_record_v0"
RECORD_TYPE = "research_experiment"
VALID_PHASES = frozenset({"exploratory", "confirmatory"})
VALID_STATUSES = frozenset({"completed", "invalid", "inconclusive", "failed"})


class ExperimentRecordError(ValueError):
    """Raised when experiment evidence is incomplete, malformed, or mismatched."""


def sha256_file(path: Path) -> str:
    """Return exact-byte SHA-256 for an existing regular file."""
    if not path.is_file():
        raise FileNotFoundError(f"Material file does not exist: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_json_value(value: Any) -> Any:
    """Return a deterministic JSON-compatible value or reject it clearly."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ExperimentRecordError("Non-finite numeric value is not representable in an experiment record.")
        return numeric
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExperimentRecordError("Experiment-record mapping keys must be strings.")
            normalized[key] = normalize_json_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [normalize_json_value(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )
    raise ExperimentRecordError(f"Value is not deterministically JSON serializable: {type(value)!r}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize supported data with stable key ordering and a final newline."""
    normalized = normalize_json_value(value)
    return (
        json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def describe_file(
    role: str,
    path: Path,
    *,
    display_path: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe an existing material file by role, display path, size, and hash."""
    _require_text(role, "artifact role")
    if not path.is_file():
        raise FileNotFoundError(f"Declared material {role!r} does not exist: {path}")
    descriptor: dict[str, Any] = {
        "role": role,
        "path": display_path if display_path is not None else str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if metadata is not None:
        descriptor["metadata"] = normalize_json_value(metadata)
    return descriptor


def describe_repository_code_component(
    role: str,
    path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Describe a source file using its repository-relative path."""
    resolved_root = repository_root.resolve()
    resolved_path = path.resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ExperimentRecordError(
            f"Material code component must be inside repository root: {path}"
        ) from exc
    return describe_file(role, resolved_path, display_path=relative_path.as_posix())


def collect_code_identity(
    repository_root: Path,
    material_components: Mapping[str, Path],
) -> dict[str, Any]:
    """Capture Git state and exact bytes for declared material source components."""
    resolved_root = repository_root.resolve()
    if not material_components:
        raise ExperimentRecordError("At least one material code component is required.")
    status_lines = _git(resolved_root, "status", "--porcelain", "--untracked-files=no").splitlines()
    dirty_paths = sorted({_porcelain_path(line) for line in status_lines if line.strip()})
    branch = _git(resolved_root, "branch", "--show-current") or None
    return {
        "git_head_commit": _git(resolved_root, "rev-parse", "HEAD"),
        "git_tree": _git(resolved_root, "rev-parse", "HEAD^{tree}"),
        "branch": branch,
        "dirty": bool(status_lines),
        "dirty_tracked_paths": dirty_paths,
        "material_code_components": [
            describe_repository_code_component(role, path, resolved_root)
            for role, path in material_components.items()
        ],
    }


def build_experiment_record(
    *,
    experiment_id: str,
    research_family_id: str,
    trial_id: str,
    phase: str,
    experiment_type: str,
    status: str,
    promotion_status: str,
    generated_at_utc: str,
    repository: str,
    code_identity: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
    method: Mapping[str, Any],
    trial_accounting: Mapping[str, Any],
    config: Mapping[str, Any],
    results: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    lineage: Mapping[str, Any],
    limitations: Sequence[str],
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """Build and validate one V0 record without writing it."""
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": RECORD_TYPE,
        "experiment_id": experiment_id,
        "research_family_id": research_family_id,
        "trial_id": trial_id,
        "phase": phase,
        "experiment_type": experiment_type,
        "status": status,
        "promotion_status": promotion_status,
        "generated_at_utc": generated_at_utc,
        "repository": repository,
        "code_identity": code_identity,
        "inputs": inputs,
        "method": method,
        "trial_accounting": trial_accounting,
        "config": config,
        "results": results,
        "outputs": outputs,
        "lineage": lineage,
        "limitations": limitations,
        "warnings": warnings,
    }
    return validate_experiment_record(record)


def validate_experiment_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate V0 required semantics and return normalized record data."""
    normalized = normalize_json_value(record)
    if not isinstance(normalized, dict):
        raise ExperimentRecordError("Experiment record must be a JSON object.")

    if normalized.get("schema_version") != SCHEMA_VERSION:
        raise ExperimentRecordError(f"schema_version must be {SCHEMA_VERSION!r}.")
    if normalized.get("record_type") != RECORD_TYPE:
        raise ExperimentRecordError(f"record_type must be {RECORD_TYPE!r}.")
    for field in (
        "experiment_id",
        "research_family_id",
        "trial_id",
        "experiment_type",
        "status",
        "promotion_status",
        "generated_at_utc",
        "repository",
    ):
        _require_text(normalized.get(field), field)

    if normalized.get("phase") not in VALID_PHASES:
        raise ExperimentRecordError(f"phase must be one of {sorted(VALID_PHASES)!r}.")
    if normalized.get("status") not in VALID_STATUSES:
        raise ExperimentRecordError(f"status must be one of {sorted(VALID_STATUSES)!r}.")
    _validate_utc_timestamp(normalized["generated_at_utc"])
    _validate_code_identity(normalized.get("code_identity"))
    _validate_artifacts(normalized.get("inputs"), "input", require_relative_paths=False)
    _validate_artifacts(normalized.get("outputs"), "output", require_relative_paths=True)
    _validate_method(normalized.get("method"))
    _validate_trial_accounting(normalized.get("trial_accounting"))
    _require_mapping(normalized.get("config"), "config")
    _require_mapping(normalized.get("results"), "results")
    _validate_lineage(normalized.get("lineage"))
    _validate_text_list(normalized.get("limitations"), "limitations", require_nonempty=True)
    _validate_text_list(normalized.get("warnings"), "warnings", require_nonempty=False)
    return normalized


def write_experiment_record(path: Path, record: Mapping[str, Any]) -> None:
    """Write validated record once. Existing target paths are never overwritten."""
    normalized = validate_experiment_record(record)
    payload = canonical_json_bytes(normalized)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite experiment record: {path}") from exc


def load_experiment_record(path: Path) -> dict[str, Any]:
    """Load and validate one experiment record JSON file."""
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentRecordError(f"Experiment record is not valid JSON: {path}") from exc
    return validate_experiment_record(content)


def verify_experiment_record(
    record_or_path: Mapping[str, Any] | Path,
    *,
    input_paths: Mapping[str, Path],
    output_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Rehash declared material inputs, outputs, and code components fail-closed."""
    record = (
        load_experiment_record(record_or_path)
        if isinstance(record_or_path, Path)
        else validate_experiment_record(record_or_path)
    )
    input_descriptors = {entry["role"]: entry for entry in record["inputs"]}
    missing_inputs = sorted(set(input_descriptors) - set(input_paths))
    if missing_inputs:
        raise ExperimentRecordError(f"Missing paths for declared material inputs: {missing_inputs!r}")
    for role, descriptor in input_descriptors.items():
        _verify_descriptor(input_paths[role], descriptor, f"input {role!r}")

    resolved_output_root = output_root.resolve()
    for descriptor in record["outputs"]:
        _verify_descriptor(
            _resolve_relative_path(resolved_output_root, descriptor["path"], "output"),
            descriptor,
            f"output {descriptor['role']!r}",
        )

    resolved_repository_root = repository_root.resolve()
    for descriptor in record["code_identity"]["material_code_components"]:
        _verify_descriptor(
            _resolve_relative_path(resolved_repository_root, descriptor["path"], "code component"),
            descriptor,
            f"code component {descriptor['role']!r}",
        )
    return record


def _git(repository_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
        raise ExperimentRecordError(f"Cannot capture repository identity: {message}")
    return result.stdout.strip()


def _porcelain_path(line: str) -> str:
    path = line[3:]
    if " -> " in path:
        path = path.rsplit(" -> ", maxsplit=1)[-1]
    return path


def _validate_code_identity(value: Any) -> None:
    _require_mapping(value, "code_identity")
    for field in ("git_head_commit", "git_tree"):
        _require_text(value.get(field), f"code_identity.{field}")
    branch = value.get("branch")
    if branch is not None:
        _require_text(branch, "code_identity.branch")
    if not isinstance(value.get("dirty"), bool):
        raise ExperimentRecordError("code_identity.dirty must be boolean.")
    dirty_paths = value.get("dirty_tracked_paths")
    _validate_text_list(dirty_paths, "code_identity.dirty_tracked_paths", require_nonempty=False)
    if dirty_paths != sorted(set(dirty_paths)):
        raise ExperimentRecordError("code_identity.dirty_tracked_paths must be unique and sorted.")
    _validate_artifacts(
        value.get("material_code_components"),
        "material code component",
        require_relative_paths=True,
    )


def _validate_artifacts(value: Any, label: str, *, require_relative_paths: bool) -> None:
    if not isinstance(value, list) or not value:
        raise ExperimentRecordError(f"At least one declared material {label} is required.")
    roles: set[str] = set()
    paths: set[str] = set()
    for descriptor in value:
        _require_mapping(descriptor, f"{label} descriptor")
        role = _require_text(descriptor.get("role"), f"{label}.role")
        path = _require_text(descriptor.get("path"), f"{label}.path")
        digest = _require_text(descriptor.get("sha256"), f"{label}.sha256")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ExperimentRecordError(f"{label}.sha256 must be a lower-case SHA-256 hex digest.")
        size = descriptor.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ExperimentRecordError(f"{label}.size_bytes must be a non-negative integer.")
        if "metadata" in descriptor:
            _require_mapping(descriptor["metadata"], f"{label}.metadata")
        if role in roles:
            raise ExperimentRecordError(f"Duplicate material {label} role: {role!r}")
        roles.add(role)
        if require_relative_paths:
            relative_path = Path(path)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ExperimentRecordError(f"{label}.path must be a relative non-traversing path: {path!r}")
            if path in paths:
                raise ExperimentRecordError(f"Duplicate authoritative {label} path: {path!r}")
            paths.add(path)


def _validate_method(value: Any) -> None:
    _require_mapping(value, "method")
    _require_text(value.get("method_id"), "method.method_id")


def _validate_trial_accounting(value: Any) -> None:
    _require_mapping(value, "trial_accounting")
    _require_text(value.get("scope"), "trial_accounting.scope")
    _require_text(value.get("limitation"), "trial_accounting.limitation")


def _validate_lineage(value: Any) -> None:
    _require_mapping(value, "lineage")
    _validate_text_list(value.get("parent_experiment_ids"), "lineage.parent_experiment_ids", require_nonempty=False)
    _validate_text_list(value.get("source_record_ids"), "lineage.source_record_ids", require_nonempty=False)
    supersedes = value.get("supersedes_record_id")
    if supersedes is not None:
        _require_text(supersedes, "lineage.supersedes_record_id")


def _validate_utc_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExperimentRecordError("generated_at_utc must be an ISO-8601 UTC timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ExperimentRecordError("generated_at_utc must carry a UTC offset.")


def _validate_text_list(value: Any, name: str, *, require_nonempty: bool) -> None:
    if not isinstance(value, list) or (require_nonempty and not value):
        qualifier = "non-empty " if require_nonempty else ""
        raise ExperimentRecordError(f"{name} must be a {qualifier}list of strings.")
    for item in value:
        _require_text(item, name)


def _require_mapping(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise ExperimentRecordError(f"{name} must be an object.")


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentRecordError(f"{name} is required and must be a non-empty string.")
    return value


def _resolve_relative_path(root: Path, stored_path: str, label: str) -> Path:
    candidate = (root / stored_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ExperimentRecordError(f"Declared {label} path escapes its root: {stored_path!r}") from exc
    return candidate


def _verify_descriptor(path: Path, descriptor: Mapping[str, Any], label: str) -> None:
    if not path.is_file():
        raise ExperimentRecordError(f"Declared {label} is missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != descriptor["size_bytes"]:
        raise ExperimentRecordError(
            f"Declared {label} size mismatch: expected {descriptor['size_bytes']}, got {actual_size}."
        )
    actual_digest = sha256_file(path)
    if actual_digest != descriptor["sha256"]:
        raise ExperimentRecordError(
            f"Declared {label} SHA-256 mismatch: expected {descriptor['sha256']}, got {actual_digest}."
        )
