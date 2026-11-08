"""Extraction regressions: bundled config, offline imports and closed SQLite handles."""
import importlib
import pkgutil
import socket
import sqlite3

import pytest

import research_intake
from research_intake.catalog import connect
from research_intake.config import load_config
from research_intake.resources import default_project_root


def test_bundled_configuration_loads_without_checkout():
    config = load_config(default_project_root())
    assert config.registry["sources"][0]["endpoint"] == "https://export.arxiv.org/api/query"


def test_all_intake_imports_do_not_connect(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No network during import")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for module in pkgutil.walk_packages(research_intake.__path__, research_intake.__name__ + "."):
        if module.name != "research_intake.__main__":
            importlib.import_module(module.name)


def test_database_context_commits_then_releases_handle(tmp_path):
    path = tmp_path / "test.sqlite3"
    with connect(path) as connection:
        connection.execute("CREATE TABLE example (value INTEGER)")
        connection.execute("INSERT INTO example VALUES (7)")
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")
    with connect(path) as verify:
        assert verify.execute("SELECT value FROM example").fetchone()[0] == 7
    path.unlink()  # Windows refuses this if any connection remains open.


def test_database_context_rolls_back_on_error(tmp_path):
    path = tmp_path / "rollback.sqlite3"
    with connect(path) as connection:
        connection.execute("CREATE TABLE example (value INTEGER)")
    with pytest.raises(RuntimeError), connect(path) as connection:
        connection.execute("INSERT INTO example VALUES (7)")
        raise RuntimeError("synthetic interruption")
    with connect(path) as verify:
        assert verify.execute("SELECT COUNT(*) FROM example").fetchone()[0] == 0
