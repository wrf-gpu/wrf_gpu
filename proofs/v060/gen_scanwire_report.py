"""Generate proofs/v060/scanwire_report.json -- the v0.6.0 scan-wire deliverable.

Consolidates (CPU): which of the 11 new schemes are wired into the operational
forecast scan + their GPU-runnability, the per-scheme integration-smoke results
(conservation / finite / executed), the fail-closed dispatch boundary, and the
integrated multi-config forecast-gate readiness (which combos are GPU-runnable now
vs MANAGER-scheduled follow-up). NOT a WRF-parity claim and NOT a forecast run --
per-scheme WRF savepoint parity lives in the committed lane reports; the GPU
multi-config forecast vs CPU-WRF is MANAGER-scheduled.

Run:
  PYTHONPATH=src JAX_PLATFORMS=cpu taskset -c 0-3 python proofs/v060/gen_scanwire_report.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import importlib.util  # noqa: E402

from gpuwrf.coupling.physics_dispatch import dispatch_matrix  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Register before exec so dataclasses defined in the module can resolve their
    # field types via sys.modules[module.__module__] (else dataclass init fails).
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_HERE = Path(__file__).resolve().parent
smoke = _load("v060_scanwire_smoke", _HERE / "scanwire_smoke.py")
gate = _load("v060_forecast_gate_harness", _HERE / "forecast_gate_harness.py")


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return "unknown"


# The 11 NEW schemes per the sprint scope (Thompson/MYNN/MYNN-sfclay/Noah-MP were
# the v0.2.0 baseline). KF (cu=1) is the 12th merged scheme and part of the
# "v0.2.0 + KF" baseline-extension (combo_1), tracked SEPARATELY below. Status:
#   scan_wired_gpu  -- State<->scheme adapter threaded into the GPU scan
#   False           -- selectable + parity-passed but NOT scan-wired (loud reject),
#                      with the scheme-specific reason.
NEW_SCHEME_STATUS = {
    "mp_physics=1 (Kessler)": {"scan_wired_gpu": True, "adapter": "coupling.scan_adapters.kessler_adapter"},
    "mp_physics=6 (WSM6)": {"scan_wired_gpu": True, "adapter": "coupling.scan_adapters.wsm6_adapter"},
    "mp_physics=10 (Morrison)": {"scan_wired_gpu": True, "adapter": "coupling.scan_adapters.morrison_adapter"},
    "mp_physics=16 (WDM6)": {"scan_wired_gpu": True, "adapter": "coupling.scan_adapters.wdm6_adapter"},
    "sf_sfclay_physics=1 (revised-MM5)": {"scan_wired_gpu": True, "adapter": "coupling.scan_adapters.sfclay_revised_mm5_adapter"},
    "sf_sfclay_physics=7 (Pleim-Xiu)": {"scan_wired_gpu": True, "adapter": "coupling.scan_adapters.pleim_xiu_sfclay_adapter"},
    "bl_pbl_physics=1 (YSU)": {"scan_wired_gpu": True, "adapter": "coupling.scan_adapters.ysu_pbl_adapter (v0.6.0 jax.lax.scan rewrite -> pbl_ysu.ysu_columns)"},
    "bl_pbl_physics=7 (ACM2)": {"scan_wired_gpu": True, "adapter": "coupling.scan_adapters.acm2_pbl_adapter (v0.6.0 jax.lax.scan rewrite -> pbl_acm2.acm2_columns)"},
    "cu_physics=3 (Grell-Freitas)": {"scan_wired_gpu": False, "reason": "CPU-NumPy reference port (gpu_runnable=False); GPU-batching TODO; selectable CPU-only, excluded from GPU scan"},
    "cu_physics=6/16 (Tiedtke)": {"scan_wired_gpu": False, "reason": "CPU-NumPy reference port (gpu_runnable=False); GPU-batching TODO; selectable CPU-only, excluded from GPU scan"},
    "sf_surface_physics=2 (Noah-classic)": {
        "scan_wired_gpu": True,
        "adapter": "coupling.noahclassic_surface_hook.noahclassic_surface_step (+ OperationalCarry.noahclassic_land)",
        "proof": "proofs/v060/noah_coupler_report.json",
        "requires_explicit_bundle": "noahclassic_static + noahclassic_land",
    },
}

# KF cumulus (cu=1) -- the 12th merged scheme / v0.2.0 baseline-extension. Newly
# scan-wired in this sprint (with the additive OperationalCarry.cumulus_carry).
KF_STATUS = {
    "cu_physics=1 (Kain-Fritsch)": {"scan_wired_gpu": True, "adapter": "coupling.scan_adapters.kf_adapter (+ OperationalCarry.cumulus_carry)"},
}


def build() -> dict:
    smoke_report = smoke.run()
    gate_report = gate.readiness_report()
    wired = [k for k, v in NEW_SCHEME_STATUS.items() if v["scan_wired_gpu"]]
    not_wired = [k for k, v in NEW_SCHEME_STATUS.items() if not v["scan_wired_gpu"]]
    return {
        "schema": "gpuwrf.v060.scanwire_report.v1",
        "title": "v0.6.0 scan-wire: 11 new schemes (+ KF) into the operational forecast scan",
        "git_head": _git_head(),
        "kind": (
            "CPU integration-wiring proof (NOT a WRF-parity claim, NOT a forecast run). "
            "Per-scheme WRF savepoint parity = committed lane reports; the GPU "
            "multi-config forecast vs CPU-WRF is MANAGER-scheduled."
        ),
        "summary": {
            "new_schemes_total": 11,
            "new_schemes_scan_wired_gpu": len(wired),
            "new_schemes_not_scan_wired": len(not_wired),
            "scan_wired_list": wired,
            "not_scan_wired_list": not_wired,
            "plus_kf_baseline_extension_scan_wired": True,
            "total_schemes_scan_wired_this_sprint": len(wired) + 1,
            "note": (
                "v0.6.0 CONSOLIDATION: 9 of the 11 new schemes (4 microphysics + 2 "
                "surface-layer + YSU/ACM2 PBL + Noah-classic land) are scan-wired into "
                "the GPU scan; KF (the 12th, baseline-extension) is also wired "
                "(+OperationalCarry.cumulus_carry) -> 10 adapters total. YSU(1)/ACM2(7) "
                "were rewritten host-NumPy -> jax.lax.scan-traceable in the PBL-GPU-op "
                "sprint (parity re-passes; see pbl_gpuop_report.json); Noah-classic(2) "
                "rides coupling.noahclassic_surface_hook with an explicit static/land "
                "bundle (see noah_coupler_report.json). 2 of the 11 remain fail-closed: "
                "GF/Tiedtke (CPU-ref cumulus, GPU-batch TODO)."
            ),
        },
        "scheme_status": {**NEW_SCHEME_STATUS, **KF_STATUS},
        "dispatch_matrix": dispatch_matrix(),
        "per_scheme_integration_smoke": smoke_report,
        "forecast_gate_readiness": gate_report,
        "overall_pass": bool(
            smoke_report["all_pass"] and bool(gate_report["gpu_runnable_now"])
        ),
    }


if __name__ == "__main__":
    report = build()
    out = ROOT / "proofs" / "v060" / "scanwire_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")
    print("overall_pass:", report["overall_pass"])
    print(
        "scan_wired_gpu:", report["summary"]["new_schemes_scan_wired_gpu"],
        "/ 11 new schemes (+ KF baseline-extension)",
    )
    raise SystemExit(0 if report["overall_pass"] else 1)
