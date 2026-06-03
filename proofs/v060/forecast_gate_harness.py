"""v0.6.0 integrated multi-config forecast gate harness (MANAGER-scheduled run).

WIRED-READY, NOT RUN HERE. This sprint prepares the harness; the single-GPU
end-to-end run vs CPU-WRF is MANAGER-scheduled (one GPU job at a time). Running
on CPU only validates the combo selections through the dispatcher (the
``--validate`` mode); the GPU forecast + CPU-WRF scoring requires ``--run`` on a
JAX GPU backend and is deliberately gated behind it.

Canonical scheme combos (only jit/vmap'd GPU-runnable schemes; the Grell-Freitas
and Tiedtke CPU-NumPy reference cumulus ports are EXCLUDED from the GPU gate per
the S0 GPU-batching-TODO note):

  combo_1  v0.2.0 baseline + KF cumulus:
           Thompson(8)/MYNN(5)/MYNN-sfclay(5)/Noah-MP(4)/KF(1) + RRTMG
  combo_2  single-moment + YSU + revised-MM5 + Noah classic + KF:
           WSM6(6)/YSU(1)/sfclayrev(1)/Noah-classic(2)/KF(1) + RRTMG
  combo_3  two-moment + ACM2 + Pleim-Xiu + Noah-MP:
           Morrison(10)/ACM2(7)/Pleim-Xiu(7)/Noah-MP(4)/no-cumulus + RRTMG

Each combo is resolved + GPU-gate-checked through coupling.physics_dispatch
(fail-closed on any non-GPU-runnable scheme). The manager runs each combo
end-to-end vs the corpus CPU-WRF d02 reference (same grid; per-lead gridpoint
paired bias/RMSE on T2/U10/V10 + diagnostics where present), reusing the
regression-gate scorer pattern in proofs/m20/continuous_gate.py.

NOTE: combo_1's surface-layer is MYNN-sfclay (sf_sfclay=5) paired with KF; combo_2
exercises the WSM6/YSU/sfclayrev/Noah-classic suite; combo_3 exercises the
two-moment/ACM2/Pleim-Xiu/Noah-MP suite. KF (cu=1) is included in combos 1-2 as
the operational GPU cumulus; combo_3 leaves cumulus off (resolved grid-scale) to
isolate the two-moment + ACM2 + PX + Noah-MP suite.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import path: this harness is run from the repo root; the package lives in src/.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from gpuwrf.coupling.physics_dispatch import (  # noqa: E402
    UnsupportedSchemeSelection,
    resolve_physics_suite,
)
from gpuwrf.io.namelist_check import validate_supported_namelist  # noqa: E402


@dataclass(frozen=True)
class ForecastCombo:
    """One canonical end-to-end scheme combination for the integrated gate."""

    combo_id: str
    description: str
    mp_physics: int
    bl_pbl_physics: int
    sf_sfclay_physics: int
    sf_surface_physics: int
    cu_physics: int
    ra_sw_physics: int = 4
    ra_lw_physics: int = 4

    def as_namelist(self) -> dict[str, int]:
        return {
            "mp_physics": self.mp_physics,
            "bl_pbl_physics": self.bl_pbl_physics,
            "sf_sfclay_physics": self.sf_sfclay_physics,
            "sf_surface_physics": self.sf_surface_physics,
            "cu_physics": self.cu_physics,
            "ra_sw_physics": self.ra_sw_physics,
            "ra_lw_physics": self.ra_lw_physics,
        }


CANONICAL_COMBOS: tuple[ForecastCombo, ...] = (
    ForecastCombo(
        combo_id="combo_1_thompson_mynn_noahmp_kf",
        description="v0.2.0 baseline (Thompson/MYNN/MYNN-sfclay/Noah-MP) + KF cumulus + RRTMG",
        mp_physics=8, bl_pbl_physics=5, sf_sfclay_physics=5, sf_surface_physics=4, cu_physics=1,
    ),
    ForecastCombo(
        combo_id="combo_2_wsm6_ysu_revisedmm5_noahclassic_kf",
        description="WSM6/YSU/revised-MM5 sfclay/Noah-classic + KF cumulus + RRTMG",
        mp_physics=6, bl_pbl_physics=1, sf_sfclay_physics=1, sf_surface_physics=2, cu_physics=1,
    ),
    ForecastCombo(
        combo_id="combo_3_morrison_acm2_pleimxiu_noahmp",
        description="Morrison/ACM2/Pleim-Xiu sfclay/Noah-MP, no cumulus + RRTMG",
        mp_physics=10, bl_pbl_physics=7, sf_sfclay_physics=7, sf_surface_physics=4, cu_physics=0,
    ),
)

# Per-lead gridpoint-paired regression envelope vs CPU-WRF, mirroring
# proofs/m20/continuous_gate.py. The manager confirms/relaxes these against the
# corpus CPU-WRF reference for each new-suite combo at the GPU run.
GATE_FIELDS_CORE = ("T2", "U10", "V10")
GATE_FIELDS_DIAG = ("Q2", "PSFC", "PBLH", "SWDOWN", "GLW", "HFX", "LH")


@dataclass
class ComboReadiness:
    combo_id: str
    description: str
    namelist: dict[str, int]
    namelist_accepted: bool
    dispatch_resolved: bool
    gpu_gate_ready: bool
    non_gpu_schemes: list[str] = field(default_factory=list)
    error: str | None = None


def validate_combo(combo: ForecastCombo) -> ComboReadiness:
    """CPU-safe readiness check: namelist accept + dispatch resolve + GPU-gate flag.

    Does NOT run a forecast. Confirms the combo passes the fail-closed namelist
    matrix and the dispatcher, and that every scheme in it is GPU-runnable (so the
    combo is admissible to the integrated GPU gate).
    """

    nl = combo.as_namelist()
    readiness = ComboReadiness(
        combo_id=combo.combo_id,
        description=combo.description,
        namelist=nl,
        namelist_accepted=False,
        dispatch_resolved=False,
        gpu_gate_ready=False,
    )
    try:
        validate_supported_namelist(nl)
        readiness.namelist_accepted = True
        suite = resolve_physics_suite(nl)
        readiness.dispatch_resolved = True
        readiness.gpu_gate_ready = suite.gpu_gate_ready
        readiness.non_gpu_schemes = list(suite.non_gpu_schemes)
    except (UnsupportedSchemeSelection, Exception) as exc:  # noqa: BLE001 - report any failure
        readiness.error = f"{type(exc).__name__}: {exc}"
    return readiness


def readiness_report() -> dict[str, Any]:
    """Build the CPU-only forecast-gate readiness object (no GPU, no forecast)."""

    combos = [validate_combo(c) for c in CANONICAL_COMBOS]
    return {
        "status": "READY_NOT_RUN",
        "note": (
            "Integrated multi-config forecast gate is WIRED. The single-GPU "
            "end-to-end run vs CPU-WRF is MANAGER-scheduled. CPU validation below "
            "confirms each combo is namelist-accepted, dispatch-resolvable, and "
            "GPU-gate-admissible (all schemes GPU-runnable). GF (cu=3) and Tiedtke "
            "(cu=6/16) are intentionally NOT in any canonical combo (CPU-reference)."
        ),
        "gate_fields": {"core": list(GATE_FIELDS_CORE), "diagnostics": list(GATE_FIELDS_DIAG)},
        "scoring": (
            "per-lead gridpoint-paired bias/RMSE vs corpus CPU-WRF d02 reference, "
            "mirroring proofs/m20/continuous_gate.py (reference = CPU-WRF, pairing = "
            "every grid cell, resolution = every lead hour)."
        ),
        "combos": [vars(r) for r in combos],
        "all_combos_gpu_gate_ready": all(r.gpu_gate_ready for r in combos),
        "manager_run_steps": [
            "1. Free the GPU (one GPU job at a time); verify CUDA context sanity.",
            "2. For each combo: build the OperationalNamelist with the combo's "
            "mp/bl/cu/sf options; thread each non-default scheme's State adapter into "
            "the operational scan (the per-scheme kernels passed savepoint parity; "
            "their scan adapters are this gate's remaining wiring).",
            "3. Run the GPU forecast over a corpus CPU-WRF d02 case (e.g. the v0.2.0 "
            "TOST run dates); emit wrfout with the new QNCLOUD/QNCCN/RAINC leaves.",
            "4. Score per-lead gridpoint-paired bias/RMSE vs the CPU-WRF reference "
            "(continuous_gate pattern) on the core + diagnostic fields.",
            "5. Record one proof JSON per combo under proofs/v060/forecast_gate/.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v0.6.0 integrated forecast gate harness")
    parser.add_argument(
        "--validate", action="store_true",
        help="CPU-only readiness check (default if neither flag given): no GPU, no forecast.",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="MANAGER-only: run the GPU forecast vs CPU-WRF. Requires a JAX GPU backend.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write the readiness JSON here.")
    args = parser.parse_args(argv)

    if args.run:
        print(
            "ERROR: --run is MANAGER-scheduled and requires a JAX GPU backend plus a "
            "corpus CPU-WRF reference case. This harness only ships the CPU readiness "
            "check (--validate). Wire the per-combo scan adapters and GPU run per the "
            "manager_run_steps in the readiness report before invoking --run.",
            file=sys.stderr,
        )
        return 2

    report = readiness_report()
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0 if report["all_combos_gpu_gate_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
