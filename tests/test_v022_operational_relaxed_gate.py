from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "v022_operational_relaxed_gate.py"


def _run(args: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd, text=True, capture_output=True, check=False)


def _base_args(fixture: Path, out: Path) -> list[str]:
    return [
        "--candidate-dir",
        str(fixture / "candidate"),
        "--cpu-wrf-dir",
        str(fixture / "cpu_wrf"),
        "--strict-dir",
        str(fixture / "strict_gpu"),
        "--restart-dir",
        str(fixture / "restart_probe"),
        "--candidate-guards",
        str(fixture / "candidate_guards.json"),
        "--strict-guards",
        str(fixture / "strict_guards.json"),
        "--init",
        "2026-05-21_18:00:00",
        "--out",
        str(out),
        "--gate-mode",
        "operational-relaxed",
    ]


def test_synthetic_operational_relaxed_example_passes_24_72_120(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    out = tmp_path / "proof"

    result = _run(["--synthetic-example-root", str(fixture), "--out", str(out), "--gate-mode", "operational-relaxed"])

    assert result.returncode == 0, result.stderr
    rollup = json.loads((out / "rollup.json").read_text(encoding="utf-8"))
    assert rollup["verdict"] == "PASS"
    domain = rollup["domains"][0]
    assert domain["lead_hours"] == [24, 72, 120]
    assert domain["hard_guards"]["all_hard_guards_pass"] is True
    assert all(item["validation_band_pass"] for item in domain["lead_payloads"])
    assert all(item["tier_o_band_pass"] for item in domain["lead_payloads"])


def test_missing_120h_candidate_fails_closed(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    seed_out = tmp_path / "seed"
    assert _run(["--synthetic-example-root", str(fixture), "--out", str(seed_out)]).returncode == 0
    (fixture / "candidate" / "wrfout_d02_2026-05-26_18:00:00").unlink()

    out = tmp_path / "proof"
    result = _run(_base_args(fixture, out))

    assert result.returncode == 1
    rollup = json.loads((out / "rollup.json").read_text(encoding="utf-8"))
    assert rollup["verdict"] == "FAIL"
    assert any("lead 120 T2 validation band failed" in reason for reason in rollup["reject_reasons"])


def test_cpu_wrf_surface_rmse_band_rejects_large_t2_error(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    seed_out = tmp_path / "seed"
    assert _run(["--synthetic-example-root", str(fixture), "--out", str(seed_out)]).returncode == 0
    target = fixture / "candidate" / "wrfout_d02_2026-05-22_18:00:00"
    with Dataset(target, "a") as dataset:
        dataset.variables["T2"][0, :, :] = dataset.variables["T2"][0, :, :] + 10.0

    out = tmp_path / "proof"
    result = _run(
        [
            "--candidate-dir",
            str(fixture / "candidate"),
            "--cpu-wrf-dir",
            str(fixture / "cpu_wrf"),
            "--init",
            "2026-05-21_18:00:00",
            "--out",
            str(out),
            "--gate-mode",
            "cpu-wrf-backlog",
        ]
    )

    assert result.returncode == 1
    lead24 = json.loads((out / "d02" / "lead_024.json").read_text(encoding="utf-8"))
    assert lead24["skill_bands"]["T2"]["vs_cpu_wrf"]["rmse"] > 3.0
    assert lead24["skill_bands"]["T2"]["validation_band_pass"] is False
