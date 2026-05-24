from __future__ import annotations

import json
import subprocess
import sys


def test_coefficient_parity_clean_savepoint_passes(tmp_path):
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
    assert payload["passed"] is True
    assert payload["transfer_audit"]["h2d_d2h_inside_timestep_loop_bytes"] == 0
