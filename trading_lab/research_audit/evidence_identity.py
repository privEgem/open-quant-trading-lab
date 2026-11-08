"""Future-only Research OS evidence identity helpers.

This module is additive. It does not reinterpret or rewrite completed historical
experiment records. Canonical Git content is used for cross-platform material
code identity while exact checkout bytes remain separately observable.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


class EvidenceIdentityError(ValueError):
    """Raised when a future evidence identity cannot be established exactly."""


_RESERVED_STRATEGY_METADATA_KEYS = frozenset({"source_file", "data_file"})


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for identity payloads."""
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceIdentityError("Identity payload is not deterministically JSON serializable.") from exc
    return (payload + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_text(repository_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown Git error"
        raise EvidenceIdentityError(f"Git identity command failed: {message}")
    return result.stdout.strip()


def _git_bytes(repository_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise EvidenceIdentityError(f"Git identity command failed: {message or 'unknown Git error'}")
    return result.stdout


def _repository_relative_path(path: Path, repository_root: Path) -> tuple[Path, str]:
    resolved_root = repository_root.resolve()
    resolved_path = path.resolve()
    try:
        relative_path = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise EvidenceIdentityError(f"Material code component is outside repository root: {path}") from exc
    if not resolved_path.is_file():
        raise EvidenceIdentityError(f"Material code component is not a regular file: {path}")
    return resolved_path, relative_path.as_posix()


def describe_canonical_repository_code_component(
    role: str,
    path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Describe tracked material code using canonical Git content plus checkout bytes.

    The working file must be Git-equivalent to ``HEAD:<path>`` after repository
    clean filters/attributes are applied. This permits platform line-ending
    differences while rejecting semantic uncommitted changes.
    """
    if not isinstance(role, str) or not role.strip():
        raise EvidenceIdentityError("Material code role must be non-empty text.")

    resolved_root = repository_root.resolve()
    resolved_path, relative_path = _repository_relative_path(path, resolved_root)

    head = _git_text(resolved_root, "rev-parse", "HEAD")
    tree = _git_text(resolved_root, "rev-parse", "HEAD^{tree}")
    try:
        object_format = _git_text(resolved_root, "rev-parse", "--show-object-format")
    except EvidenceIdentityError:
        object_format = "sha1"

    blob_oid = _git_text(resolved_root, "rev-parse", f"HEAD:{relative_path}")
    canonical_blob = _git_bytes(resolved_root, "cat-file", "blob", blob_oid)
    canonical_blob_sha256 = hashlib.sha256(canonical_blob).hexdigest()

    working_bytes = resolved_path.read_bytes()
    working_tree_sha256 = hashlib.sha256(working_bytes).hexdigest()
    filtered_oid = _git_text(
        resolved_root,
        "hash-object",
        f"--path={relative_path}",
        str(resolved_path),
    )
    if filtered_oid != blob_oid:
        raise EvidenceIdentityError(
            f"Material code component has a semantic working-tree change relative to HEAD: {relative_path}"
        )

    return {
        "role": role.strip(),
        "path": relative_path,
        "git_head_commit": head,
        "git_tree": tree,
        "git_object_format": object_format,
        "git_blob_oid": blob_oid,
        "git_blob_sha256": canonical_blob_sha256,
        "git_blob_size_bytes": len(canonical_blob),
        "working_tree_sha256": working_tree_sha256,
        "working_tree_size_bytes": len(working_bytes),
        "git_filtered_worktree_oid": filtered_oid,
        "working_tree_matches_git_content": True,
    }


def material_code_fingerprint(components: Sequence[Mapping[str, Any]]) -> str:
    """Fingerprint canonical tracked code identity, excluding checkout-byte differences."""
    if not components:
        raise EvidenceIdentityError("At least one material code component is required.")
    material: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    for component in components:
        role = component.get("role")
        path = component.get("path")
        blob_oid = component.get("git_blob_oid")
        blob_sha256 = component.get("git_blob_sha256")
        if not all(isinstance(value, str) and value for value in (role, path, blob_oid, blob_sha256)):
            raise EvidenceIdentityError("Canonical code component is incomplete.")
        if role in seen_roles:
            raise EvidenceIdentityError(f"Duplicate material-code role: {role!r}")
        seen_roles.add(role)
        material.append(
            {
                "role": role,
                "path": path,
                "git_blob_oid": blob_oid,
                "git_blob_sha256": blob_sha256,
            }
        )
    material.sort(key=lambda item: (item["role"], item["path"]))
    return canonical_sha256(material)


def build_future_experiment_identity_metadata(
    *,
    frozen_strategy_config: Mapping[str, Any],
    strategy_parameter_keys: Iterable[str],
    execution_input: Mapping[str, Any],
    strategy_source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate frozen-origin provenance, strategy parameters, and actual input."""
    if not isinstance(frozen_strategy_config, Mapping) or not frozen_strategy_config:
        raise EvidenceIdentityError("frozen_strategy_config must be a non-empty mapping.")
    if not isinstance(execution_input, Mapping) or not execution_input:
        raise EvidenceIdentityError("execution_input must be a non-empty mapping.")
    if not isinstance(strategy_source_identity, Mapping) or not strategy_source_identity:
        raise EvidenceIdentityError("strategy_source_identity must be a non-empty mapping.")

    parameter_keys = tuple(strategy_parameter_keys)
    if not parameter_keys:
        raise EvidenceIdentityError("At least one strategy parameter key is required.")
    if len(set(parameter_keys)) != len(parameter_keys):
        raise EvidenceIdentityError("Strategy parameter keys must be unique.")
    forbidden = sorted(set(parameter_keys) & _RESERVED_STRATEGY_METADATA_KEYS)
    if forbidden:
        raise EvidenceIdentityError(f"Strategy parameter allow-list contains metadata keys: {forbidden!r}")
    missing = [key for key in parameter_keys if key not in frozen_strategy_config]
    if missing:
        raise EvidenceIdentityError(f"Frozen strategy config is missing allow-listed keys: {missing!r}")

    strategy_parameters = {key: frozen_strategy_config[key] for key in parameter_keys}
    return {
        "frozen_strategy_origin": {
            "source_identity": dict(strategy_source_identity),
            "config_sha256": canonical_sha256(dict(frozen_strategy_config)),
        },
        "strategy_parameters": strategy_parameters,
        "execution_input": dict(execution_input),
    }
