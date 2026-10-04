from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _historical_artifacts import require_historical


def test_perturbation_negative_test_fails_loudly(tmp_path):
    require_historical(Path(
        "<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z/"
        "wrfout_d02_2026-05-22_00:00:00"
    ))
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
