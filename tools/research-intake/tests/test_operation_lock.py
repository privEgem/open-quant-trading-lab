from pathlib import Path

import pytest

from research_intake.operation_lock import OperationAlreadyActive, OperationLock


def test_competing_operation_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "lock"
    with (
        OperationLock(path),
        pytest.raises(OperationAlreadyActive, match="OPERATION_ALREADY_ACTIVE"),
        OperationLock(path),
    ):
        pass
