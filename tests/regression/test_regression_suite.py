from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_regression_suite_smoke_schema() -> None:
    result = subprocess.run(
        [
            "taskset",
            "-c",
            "0-3",
            sys.executable,
            "scripts/run_regression_suite.py",
            "--smoke",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    payload = json.loads(result.stdout)
    assert payload["schema"] == "OracleBaselineRegressionAggregate"
    assert payload["schema_version"] == 1
    assert payload["mode"] == "smoke"
    assert payload["field_test_count"] >= 1
    assert payload["case_count"] >= 1
    assert Path(payload["aggregate_path"]).is_file()
