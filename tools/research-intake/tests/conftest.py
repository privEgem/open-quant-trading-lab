from __future__ import annotations

import socket
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real networking forbidden in default intake tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def mixed_xml(project_root: Path) -> bytes:
    return (project_root / "tests" / "fixtures" / "arxiv" / "mixed.xml").read_bytes()


@pytest.fixture
def version2_xml(project_root: Path) -> bytes:
    return (project_root / "tests" / "fixtures" / "arxiv" / "version2.xml").read_bytes()
