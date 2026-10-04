from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from _historical_artifacts import require_historical


def test_coefficient_parity_clean_savepoint_passes(tmp_path):
    require_historical(Path(
        "<DATA_ROOT>/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z/"
        "wrfout_d02_2026-05-22_00:00:00"
    ))
    savepoint_dir = tmp_path / "patch16"
    output = tmp_path / "parity.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/m6b0_wrf_savepoint_extract.py",
            "--tier",
            "patch16",
            "--steps",
            "1",
            "--output",
            str(savepoint_dir),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/m6b0_jax_savepoint_compare.py",
            "--operator",
            "coefficient_construction",
            "--savepoint",
            str(savepoint_dir),
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text())
    assert payload["outcome"] in {"PASS", "PARITY-DEFECT-LOCALIZED"}
    assert payload["transfer_audit"]["h2d_d2h_inside_timestep_loop_bytes"] == 0
