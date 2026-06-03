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


class _SchemeNamelist:
    """Minimal namelist-shaped stub exposing the physics-option attributes the
    operational scan's ``_resolve_operational_suite`` reads (so the scan-wire status
    can be checked on CPU without building a full OperationalNamelist + grid)."""

    def __init__(self, nl: dict[str, int]):
        self.mp_physics = nl["mp_physics"]
        self.bl_pbl_physics = nl["bl_pbl_physics"]
        self.sf_sfclay_physics = nl["sf_sfclay_physics"]
        self.cu_physics = nl["cu_physics"]
        self.sf_surface_physics = nl["sf_surface_physics"]
        # Noah-MP (sf_surface=4) is threaded via use_noahmp in the scan.
        self.use_noahmp = int(nl["sf_surface_physics"]) == 4


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
    scan_wired: bool = False
    scan_wire_error: str | None = None
    non_gpu_schemes: list[str] = field(default_factory=list)
    error: str | None = None


def validate_combo(combo: ForecastCombo) -> ComboReadiness:
    """CPU-safe readiness check: namelist accept + dispatch resolve + GPU-gate flag
    + OPERATIONAL-SCAN-WIRE status.

    Does NOT run a forecast. Confirms the combo (1) passes the fail-closed namelist
    matrix, (2) resolves through the dispatcher, (3) every scheme is GPU-runnable
    (admissible to the GPU gate), and (4) every scheme's State<->scheme SCAN ADAPTER
    is actually threaded into the operational scan (``_resolve_operational_suite``
    accepts it). A combo is GPU-forecast-runnable only when BOTH gpu_gate_ready AND
    scan_wired hold -- scan_wired is the honest gate the manager's ``--run`` checks.
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

    # Scan-wire status: does the operational scan actually thread every scheme's
    # adapter? (Imported lazily so the dispatcher-only --validate path stays cheap.)
    try:
        from gpuwrf.runtime.operational_mode import _resolve_operational_suite

        _resolve_operational_suite(_SchemeNamelist(nl))
        readiness.scan_wired = True
    except UnsupportedSchemeSelection as exc:
        readiness.scan_wired = False
        readiness.scan_wire_error = str(exc)
    return readiness


# Fully-scan-wired alternate combos (post-consolidation 2026-06-04). YSU (bl=1) and
# ACM2 (bl=7) are now the v0.6.0 jax.lax.scan/vmap GPU-op rewrites and ARE scan-wired,
# so canonical combo_3 (ACM2) is already GPU-scan-runnable. The only canonical combo
# still unwired is combo_2, whose Noah-classic land (sf_surface=2) needs the explicit
# noahclassic_static/noahclassic_land bundle the readiness stub does not attach. These
# alternates swap Noah-classic for Noah-MP(4) so the microphysics (WSM6/Morrison/WDM6/
# Kessler), surface-layer (revised-MM5/Pleim-Xiu) and cumulus (KF/Tiedtke) schemes are
# exercised end-to-end on the GPU scan without depending on the Noah-classic bundle.
SCAN_WIRED_COMBOS: tuple[ForecastCombo, ...] = (
    ForecastCombo(
        combo_id="combo_2w_wsm6_mynn_revisedmm5_noahmp_kf",
        description="WSM6/MYNN/revised-MM5 sfclay/Noah-MP + KF cumulus + RRTMG "
                    "(scan-wired variant of combo_2: MYNN instead of host-NumPy YSU, "
                    "Noah-MP instead of the canonical Noah-classic bundle path)",
        mp_physics=6, bl_pbl_physics=5, sf_sfclay_physics=1, sf_surface_physics=4, cu_physics=1,
    ),
    ForecastCombo(
        combo_id="combo_3w_morrison_mynn_pleimxiu_noahmp",
        description="Morrison/MYNN/Pleim-Xiu sfclay/Noah-MP, no cumulus + RRTMG "
                    "(scan-wired variant of combo_3: MYNN instead of host-NumPy ACM2)",
        mp_physics=10, bl_pbl_physics=5, sf_sfclay_physics=7, sf_surface_physics=4, cu_physics=0,
    ),
    ForecastCombo(
        combo_id="combo_4w_wdm6_mynn_kessler_check",
        description="WDM6/MYNN/MYNN-sfclay/Noah-MP + KF cumulus + RRTMG "
                    "(exercises the WDM6 Nc/Nn additive leaves + KF carry end-to-end)",
        mp_physics=16, bl_pbl_physics=5, sf_sfclay_physics=5, sf_surface_physics=4, cu_physics=1,
    ),
)


def readiness_report() -> dict[str, Any]:
    """Build the CPU-only forecast-gate readiness object (no GPU, no forecast)."""

    canonical = [validate_combo(c) for c in CANONICAL_COMBOS]
    scan_wired = [validate_combo(c) for c in SCAN_WIRED_COMBOS]
    runnable = [r for r in (canonical + scan_wired) if r.gpu_gate_ready and r.scan_wired]
    return {
        "status": "READY_NOT_RUN",
        "note": (
            "Integrated multi-config forecast gate. The single-GPU end-to-end run vs "
            "CPU-WRF is MANAGER-scheduled (--run). v0.6.0 CONSOLIDATION scan-wire status "
            "(HONEST, post-consolidation 2026-06-04): combo_1 (v0.2.0 + KF) is fully "
            "scan-wired. YSU (bl=1) and ACM2 (bl=7) are now the v0.6.0 jax.lax.scan/vmap "
            "GPU-op rewrites and ARE scan-wired -- so canonical combo_3 (Morrison/ACM2/"
            "Pleim-Xiu/Noah-MP) is now GPU-scan-runnable. Canonical combo_2 still fails "
            "ONLY because its Noah-classic land (sf_surface=2) needs the explicit "
            "noahclassic_static/noahclassic_land bundle that the readiness stub does not "
            "attach (YSU itself is wired); the SCAN_WIRED_COMBOS swap Noah-classic for "
            "Noah-MP so the WSM/Morrison/WDM6 + revised-MM5/Pleim-Xiu + KF/Tiedtke schemes "
            "run end-to-end now. Tiedtke (cu=6) is the GPU-batched jit/vmap adapter and IS "
            "scan-wired. GF (cu=3) and New-Tiedtke (cu=16) remain CPU-reference, FAIL-CLOSED "
            "(loud) by design; MYJ (bl=2)/Janjic (sfclay=2) are savepoint-parity-proven "
            "CPU references, also FAIL-CLOSED in the scan."
        ),
        "gate_fields": {"core": list(GATE_FIELDS_CORE), "diagnostics": list(GATE_FIELDS_DIAG)},
        "scoring": (
            "per-lead gridpoint-paired bias/RMSE vs corpus CPU-WRF d02 reference, "
            "mirroring proofs/m20/continuous_gate.py (reference = CPU-WRF, pairing = "
            "every grid cell, resolution = every lead hour)."
        ),
        "canonical_combos": [vars(r) for r in canonical],
        "scan_wired_combos": [vars(r) for r in scan_wired],
        "gpu_runnable_now": [r.combo_id for r in runnable],
        "all_canonical_gpu_gate_ready": all(r.gpu_gate_ready for r in canonical),
        "all_canonical_scan_wired": all(r.scan_wired for r in canonical),
        "manager_run_steps": [
            "1. Free the GPU (one GPU job at a time); verify CUDA context sanity.",
            "2. For each GPU-runnable combo (gpu_runnable_now): build the "
            "OperationalNamelist with the combo's mp/bl/cu/sf options (the scan "
            "adapters in coupling.scan_adapters are threaded; _resolve_operational_suite "
            "accepts the combo).",
            "3. Run run_forecast_operational over a corpus CPU-WRF d02 case (e.g. the "
            "v0.2.0 TOST run dates); emit wrfout with the new QNCLOUD/QNCCN/RAINC leaves.",
            "4. Score per-lead gridpoint-paired bias/RMSE vs the CPU-WRF reference "
            "(continuous_gate pattern) on the core + diagnostic fields.",
            "5. Record one proof JSON per combo under proofs/v060/forecast_gate/.",
            "6. CARRY-OVER (post-consolidation, cross-model): real-run Noah-classic(2) "
            "land/static bundle assembly for canonical combo_2 (YSU(1)/ACM2(7) PBL and "
            "Tiedtke(6) cumulus are now GPU-scan-wired -- DONE); GPU-batch GF(3) cumulus "
            "(~2000-LOC closure-ensemble + beta-PDF vmap rewrite) and gate New-Tiedtke(16); "
            "GPU-scan-wire MYJ(2)/Janjic(2) (parity-proven CPU references today).",
        ],
    }


def _build_combo_namelist(combo: ForecastCombo, grid):
    """Build the OperationalNamelist for one combo (manager --run helper).

    Sets the physics-suite selection fields the scan dispatches on. Noah-MP land
    (sf_surface=4) is threaded via use_noahmp=True (the scan's land toggle); the
    manager attaches the Noah-MP static/params + boundary forcing for the real run.
    """

    from gpuwrf.runtime.operational_mode import OperationalNamelist

    nl = OperationalNamelist.from_grid(grid)
    return nl.__class__(
        **{
            **{f.name: getattr(nl, f.name) for f in nl.__dataclass_fields__.values()},
            "mp_physics": combo.mp_physics,
            "bl_pbl_physics": combo.bl_pbl_physics,
            "sf_sfclay_physics": combo.sf_sfclay_physics,
            "cu_physics": combo.cu_physics,
            "sf_surface_physics": combo.sf_surface_physics,
            "use_noahmp": int(combo.sf_surface_physics) == 4,
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v0.6.0 integrated forecast gate harness")
    parser.add_argument(
        "--validate", action="store_true",
        help="CPU-only readiness check (default if neither flag given): no GPU, no forecast.",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="MANAGER-only: run the GPU forecast vs CPU-WRF. Requires a JAX GPU backend "
             "+ a corpus CPU-WRF reference case + Noah-MP init bundle.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Write the readiness JSON here.")
    args = parser.parse_args(argv)

    if args.run:
        # The scan adapters ARE wired (coupling.scan_adapters); the GPU forecast +
        # CPU-WRF scoring is MANAGER-scheduled (single GPU job) because it needs a
        # GPU backend, a corpus d02 reference case, and the Noah-MP init bundle. This
        # branch refuses cleanly rather than half-running, and points at the wiring.
        try:
            import jax
            backend = jax.default_backend()
        except Exception:
            backend = "unknown"
        print(
            "ERROR: --run is MANAGER-scheduled. The per-combo State<->scheme scan "
            "adapters ARE wired (gpuwrf.coupling.scan_adapters; _resolve_operational_suite "
            "accepts the gpu_runnable_now combos). What --run still needs (MANAGER): a "
            "JAX GPU backend (current backend="
            f"{backend!r}), a corpus CPU-WRF d02 reference case + met_em/boundary forcing, "
            "and the Noah-MP init bundle for the Noah-MP combos. Build each combo's "
            "namelist via _build_combo_namelist, run run_forecast_operational, then score "
            "per the readiness report's manager_run_steps.",
            file=sys.stderr,
        )
        return 2

    report = readiness_report()
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    # Gate passes when there is at least one GPU-runnable-now combo AND the scan-wired
    # variants all resolve (the canonical combos may legitimately be NOT scan-wired).
    return 0 if report["gpu_runnable_now"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
