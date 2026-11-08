from __future__ import annotations

from pathlib import Path

from research_intake.catalog import connect
from research_intake.cli import main as cli_main


def main(argv):
    return cli_main(["--project-root", str(Path(__file__).resolve().parents[1]), *argv])


def test_validate_config_does_not_create_runtime_root(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "not-created"
    monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(runtime))
    assert main(["validate-config"]) == 0
    assert not runtime.exists()


def test_backfill_without_days_is_invalid_before_network(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(runtime))
    assert main(["init-db"]) == 0
    assert main(["run", "--mode", "backfill"]) == 2


def test_incremental_refusal_reports_reason_without_run_artifacts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    runtime = tmp_path / "runtime"
    monkeypatch.setenv("RESEARCH_INTAKE_DATA_DIR", str(runtime))
    assert main(["init-db"]) == 0
    capsys.readouterr()

    assert main(["run", "--mode", "incremental", "--max-items", "20"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no incremental predecessor is eligible" in captured.err
    assert "bounded backfill" in captured.err
    with connect(runtime / "state" / "catalog.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    artifact_roots = (
        runtime / "raw",
        runtime / "manifests",
        runtime / "recovery-receipts",
        runtime / "review" / "markdown",
        runtime / "review" / "jsonl",
    )
    assert all(not any(root.iterdir()) for root in artifact_roots)
