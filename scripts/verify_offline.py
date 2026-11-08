"""Run both independent test suites with networking blocked in child Python processes."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "scripts/offline_guard")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["RESEARCH_INTAKE_CONTACT_EMAIL"] = ""
    with tempfile.TemporaryDirectory(prefix="quant-tests-") as directory:
        for label, cwd in (("toolkit", root), ("intake", root / "tools/research-intake")):
            print(f"Offline suite: {label}", flush=True)
            command = [sys.executable, "-m", "pytest", "tests", "-ra", "-p", "no:cacheprovider", "--basetemp", str(Path(directory) / label)]
            result = subprocess.run(command, cwd=cwd, env=env, check=False)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
