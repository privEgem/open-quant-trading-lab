from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from trading_lab.research_audit.experiment_record import (
    ExperimentRecordError,
    VALID_STATUSES,
    build_experiment_record,
    canonical_json_bytes,
    describe_file,
    describe_repository_code_component,
    sha256_file,
    validate_experiment_record,
    verify_experiment_record,
    write_experiment_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentRecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        self.root = Path(self.temp.name)
        self.repository_root = self.root / "repository"
        self.repository_root.mkdir()
        self.source_path = self.repository_root / "component.py"
        self.source_path.write_bytes(b"component bytes\n")
        self.input_path = self.root / "input.csv"
        self.input_path.write_bytes(b"timestamp,close\n2026-01-01T00:00:00Z,1\n")
        self.output_root = self.root / "outputs"
        self.output_root.mkdir()
        self.output_path = self.output_root / "summary.json"
        self.output_path.write_bytes(b'{"result":"ok"}\n')

    def tearDown(self) -> None:
        self.temp.cleanup()

    def record(self) -> dict[str, object]:
        return build_experiment_record(
            experiment_id="experiment-1",
            research_family_id="family-1",
            trial_id="trial-1",
            phase="exploratory",
            experiment_type="static_replay",
            status="completed",
            promotion_status="not_promoted",
            generated_at_utc="2026-08-27T12:00:00+00:00",
            repository="example/research-project",
            code_identity={
                "git_head_commit": "a" * 40,
                "git_tree": "b" * 40,
                "branch": "feature/test",
                "dirty": False,
                "dirty_tracked_paths": [],
                "material_code_components": [
                    describe_repository_code_component("component", self.source_path, self.repository_root)
                ],
            },
            inputs=[describe_file("static_input", self.input_path, metadata={"kind": "synthetic"})],
            method={"method_id": "static_replay_profile_comparison_v1"},
            trial_accounting={
                "scope": "record_local_trial_identity_only",
                "limitation": "No repository-wide ledger or historical trial count is claimed.",
            },
            config={"slippage_bps": 1.0},
            results={"net_pnl": 12.5},
            outputs=[describe_file("summary", self.output_path, display_path="summary.json")],
            lineage={
                "parent_experiment_ids": [],
                "source_record_ids": [],
                "supersedes_record_id": None,
            },
            limitations=["Replay output is not broker/account PnL."],
        )

    def test_sha256_is_exact_and_deterministic(self) -> None:
        path = self.root / "digest.txt"
        path.write_bytes(b"abc")
        self.assertEqual(sha256_file(path), hashlib.sha256(b"abc").hexdigest())
        self.assertEqual(sha256_file(path), sha256_file(path))

    def test_normalization_and_json_are_deterministic(self) -> None:
        left = canonical_json_bytes({"z": {"b", "a"}, "a": (2, 1)})
        right = canonical_json_bytes({"a": [2, 1], "z": {"a", "b"}})
        self.assertEqual(left, right)
        self.assertEqual(left, b'{\n  "a": [\n    2,\n    1\n  ],\n  "z": [\n    "a",\n    "b"\n  ]\n}\n')

    def test_build_write_and_verify_valid_record(self) -> None:
        record_path = self.output_root / "experiment_record.json"
        write_experiment_record(record_path, self.record())
        verified = verify_experiment_record(
            record_path,
            input_paths={"static_input": self.input_path},
            output_root=self.output_root,
            repository_root=self.repository_root,
        )
        self.assertEqual(verified["experiment_id"], "experiment-1")

    def test_write_refuses_overwrite(self) -> None:
        record_path = self.output_root / "experiment_record.json"
        write_experiment_record(record_path, self.record())
        with self.assertRaises(FileExistsError):
            write_experiment_record(record_path, self.record())

    def test_missing_identity_family_and_trial_are_rejected(self) -> None:
        for field in ("experiment_id", "research_family_id", "trial_id"):
            with self.subTest(field=field):
                record = copy.deepcopy(self.record())
                record.pop(field)
                with self.assertRaises(ExperimentRecordError):
                    validate_experiment_record(record)

    def test_invalid_phase_is_rejected(self) -> None:
        record = copy.deepcopy(self.record())
        record["phase"] = "production"
        with self.assertRaises(ExperimentRecordError):
            validate_experiment_record(record)

    def test_invalid_status_is_rejected(self) -> None:
        record = copy.deepcopy(self.record())
        record["status"] = "running"
        with self.assertRaises(ExperimentRecordError):
            validate_experiment_record(record)

    def test_accepted_statuses_remain_valid(self) -> None:
        for status in VALID_STATUSES:
            with self.subTest(status=status):
                record = copy.deepcopy(self.record())
                record["status"] = status
                self.assertEqual(validate_experiment_record(record)["status"], status)

    def test_missing_method_is_rejected(self) -> None:
        record = copy.deepcopy(self.record())
        record["method"] = {}
        with self.assertRaises(ExperimentRecordError):
            validate_experiment_record(record)

    def test_missing_declared_input_is_rejected(self) -> None:
        record = copy.deepcopy(self.record())
        record["inputs"] = []
        with self.assertRaises(ExperimentRecordError):
            validate_experiment_record(record)

    def test_input_hash_mismatch_is_rejected(self) -> None:
        record = self.record()
        self.input_path.write_bytes(b"timestamp,close\n2026-01-01T00:00:00Z,2\n")
        with self.assertRaisesRegex(ExperimentRecordError, "SHA-256 mismatch"):
            verify_experiment_record(
                record,
                input_paths={"static_input": self.input_path},
                output_root=self.output_root,
                repository_root=self.repository_root,
            )

    def test_missing_declared_output_is_rejected(self) -> None:
        record = self.record()
        self.output_path.unlink()
        with self.assertRaisesRegex(ExperimentRecordError, "is missing"):
            verify_experiment_record(
                record,
                input_paths={"static_input": self.input_path},
                output_root=self.output_root,
                repository_root=self.repository_root,
            )

    def test_output_hash_mismatch_is_rejected(self) -> None:
        record = self.record()
        self.output_path.write_bytes(b'{"result":"changed"}\n')
        with self.assertRaisesRegex(ExperimentRecordError, "mismatch"):
            verify_experiment_record(
                record,
                input_paths={"static_input": self.input_path},
                output_root=self.output_root,
                repository_root=self.repository_root,
            )

    def test_duplicate_authoritative_output_paths_are_rejected(self) -> None:
        record = copy.deepcopy(self.record())
        duplicate = copy.deepcopy(record["outputs"][0])
        duplicate["role"] = "duplicate_summary"
        record["outputs"].append(duplicate)
        with self.assertRaisesRegex(ExperimentRecordError, "Duplicate authoritative"):
            validate_experiment_record(record)

    def test_unrepresentable_values_are_rejected(self) -> None:
        with self.assertRaises(ExperimentRecordError):
            canonical_json_bytes({"unsupported": object()})


if __name__ == "__main__":
    unittest.main()
