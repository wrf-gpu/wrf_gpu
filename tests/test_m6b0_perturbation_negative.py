from __future__ import annotations

import subprocess
import sys


def test_perturbation_negative_test_fails_loudly(tmp_path):
    savepoint_dir = tmp_path / "column"
    subprocess.run(
        [
            sys.executable,
            "scripts/m6b0_wrf_savepoint_extract.py",
            "--tier",
            "column",
            "--steps",
            "1",
            "--output",
            str(savepoint_dir),
        ],
        check=True,
    )

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/m6b0_perturbation_negative_test.py",
            "--savepoint",
            str(savepoint_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"caught": true' in proc.stdout
    assert '"passed": false' in proc.stdout
