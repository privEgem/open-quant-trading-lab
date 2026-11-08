from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    raw: Path
    state: Path
    manifests: Path
    recovery_receipts: Path
    markdown: Path
    jsonl: Path

    @property
    def database(self) -> Path:
        return self.state / "catalog.sqlite3"

    @property
    def lock_file(self) -> Path:
        return self.root / ".research-intake.lock"

    @property
    def human_review(self) -> Path:
        return self.root / "human-review"


def resolve_paths(project_root: Path) -> RuntimePaths:
    configured = os.environ.get("RESEARCH_INTAKE_DATA_DIR")
    root = Path(configured).expanduser().resolve() if configured else (Path.cwd() / ".research-intake-data").resolve()
    return RuntimePaths(
        root=root,
        raw=root / "raw",
        state=root / "state",
        manifests=root / "manifests",
        recovery_receipts=root / "recovery-receipts",
        markdown=root / "review" / "markdown",
        jsonl=root / "review" / "jsonl",
    )


def create_runtime_directories(paths: RuntimePaths) -> None:
    for path in (paths.root, paths.raw, paths.state, paths.manifests, paths.recovery_receipts, paths.markdown, paths.jsonl):
        path.mkdir(parents=True, exist_ok=True)
