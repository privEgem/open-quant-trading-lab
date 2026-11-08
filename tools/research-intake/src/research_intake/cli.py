from __future__ import annotations

import argparse
import sys
from pathlib import Path

from research_intake.catalog import init_database, require_database
from research_intake.config import ConfigurationError, contact_email, load_config
from research_intake.human_review import (
    canonical_json,
    export_reviews,
    load_review_batch,
    record_review_batch,
)
from research_intake.operation_lock import OperationAlreadyActive, OperationLock
from research_intake.paths import create_runtime_directories, resolve_paths
from research_intake.pipeline import NoEligibleIncrementalPredecessor, run_intake
from research_intake.recovery import recover_run
from research_intake.resources import default_project_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m research_intake")
    parser.add_argument("--project-root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-config")
    commands.add_parser("init-db")
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("backfill", "incremental"), required=True)
    run.add_argument("--days", type=int)
    run.add_argument("--max-items", type=int)
    recover = commands.add_parser("recover-run")
    recover.add_argument("--run-id", required=True)
    record_reviews = commands.add_parser("record-reviews")
    record_reviews.add_argument("--input", type=Path, required=True)
    record_reviews.add_argument("--dry-run", action="store_true")
    commands.add_parser("export-reviews")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    project_root = (arguments.project_root.resolve() if arguments.project_root else default_project_root())
    try:
        paths = resolve_paths(project_root)
        if arguments.command == "validate-config":
            load_config(project_root)
            contact_email()
            print("configuration is valid")
            return 0
        if arguments.command == "record-reviews":
            require_database(paths.database)
            batch = load_review_batch(
                arguments.input,
                project_root / "schemas" / "human_review_batch.schema.json",
            )
            if arguments.dry_run:
                summary = record_review_batch(
                    paths.database,
                    paths.root,
                    batch,
                    project_root / "schemas" / "normalized_item.schema.json",
                    dry_run=True,
                )
            else:
                with OperationLock(paths.lock_file):
                    init_database(paths.database)
                    summary = record_review_batch(
                        paths.database,
                        paths.root,
                        batch,
                        project_root / "schemas" / "normalized_item.schema.json",
                    )
            print(canonical_json(summary))
            return 0
        if arguments.command == "export-reviews":
            require_database(paths.database)
            with OperationLock(paths.lock_file):
                summary = export_reviews(paths.database, paths.human_review)
            print(canonical_json(summary))
            return 0
        config = load_config(project_root)
        contact_email()
        create_runtime_directories(paths)
        with OperationLock(paths.lock_file):
            if arguments.command == "init-db":
                init_database(paths.database)
                print(paths.database)
                return 0
            require_database(paths.database)
            if arguments.command == "recover-run":
                code, receipt = recover_run(project_root, paths, arguments.run_id)
                print(receipt)
                return code
            if arguments.max_items is not None and arguments.max_items <= 0:
                raise ConfigurationError("--max-items must be greater than zero")
            if arguments.mode == "backfill" and (
                arguments.days is None or arguments.days <= 0
            ):
                raise ConfigurationError(
                    "backfill requires --days greater than zero"
                )
            if arguments.mode == "incremental" and arguments.days is not None:
                raise ConfigurationError("--days is valid only for backfill")
            code, run_id = run_intake(config, paths, arguments.mode, arguments.days, arguments.max_items)
            print(run_id)
            return code
    except (ConfigurationError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    except OperationAlreadyActive as error:
        print(str(error), file=sys.stderr)
        return 1
    except NoEligibleIncrementalPredecessor as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
