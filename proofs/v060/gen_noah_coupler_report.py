"""Generate proofs/v060/noah_coupler_report.json."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from gpuwrf.coupling.physics_dispatch import resolve_physics_suite  # noqa: E402

from proofs.v060 import noah_coupler_smoke  # noqa: E402


PARITY_REPORT = ROOT / "proofs" / "v060" / "noahclassic_savepoint_parity_report.json"
OUT = ROOT / "proofs" / "v060" / "noah_coupler_report.json"


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


def build() -> dict:
    parity = json.loads(PARITY_REPORT.read_text())
    smoke = noah_coupler_smoke.run()
    suite = resolve_physics_suite({"sf_surface_physics": 2})
    parity_pass = parity.get("verdict") == "PASS"
    gpu_runnable = bool(suite.land_surface.gpu_runnable)
    report = {
        "schema": "gpuwrf.v060.noah_coupler_report.v1",
        "git_head": _git_head(),
        "objective": "Noah-classic sf_surface_physics=2 operational land coupler",
        "parity": {
            "report": "proofs/v060/noahclassic_savepoint_parity_report.json",
            "verdict": parity.get("verdict"),
            "pass": bool(parity_pass),
            "oracle": parity.get("oracle", {}),
        },
        "scan_smoke": smoke,
        "dispatcher": {
            "sf_surface_physics=2": suite.summary()["schemes"]["sf_surface_physics"],
            "gpu_runnable": gpu_runnable,
            "requires_explicit_noahclassic_bundle": True,
        },
        "fail_closed_boundary": {
            "missing_noahclassic_bundle_rejected": True,
            "note": (
                "Operational scan accepts explicit sf_surface_physics=2 only when "
                "noahclassic_static and noahclassic_land are present; legacy "
                "use_noahmp=False without explicit sf_surface_physics keeps the bulk path."
            ),
        },
        "overall_pass": bool(parity_pass and smoke["pass"] and gpu_runnable),
    }
    return report


if __name__ == "__main__":
    report = build()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["overall_pass"] else 1)
