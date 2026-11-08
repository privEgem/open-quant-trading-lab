"""Read-only default configuration resources included in the wheel."""
from pathlib import Path


def default_project_root() -> Path:
    return Path(__file__).resolve().parent
