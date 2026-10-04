#!/usr/bin/env python
"""Regenerate proofs/v060/consolidation_integration_matrix.json from the LIVE
merged registries + the multicfg operational smoke report.

This is the authoritative consolidation status matrix: it derives each accepted
physics option's status (GPU-OPERATIONAL-WIRED / PARITY-PROVEN-FAIL-CLOSED /
PASSIVE-OFF) directly from the merged source-of-truth tables
(``contracts.physics_registry`` accept sets + scheme names,
``runtime.operational_mode._SCAN_WIRED_OPTIONS`` / ``_SCAN_UNWIRED_REASON``,
``coupling.scan_adapters`` adapter tables, ``coupling.physics_dispatch``
gpu-runnability), so the matrix can never silently drift from the code. The
integration-smoke block is read back from ``multicfg_smoke_report.json``.

CPU-only; run under taskset -c 0-3 JAX_PLATFORMS=cpu JAX_ENABLE_X64=true.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from gpuwrf.contracts.physics_registry import (
    CU_SCHEMES,
    MP_SCHEMES,
    PBL_SCHEMES,
    RA_LW_SCHEMES,
    RA_SW_SCHEMES,
    SFCLAY_SCHEMES,
    SURFACE_SCHEMES,
)
from gpuwrf.coupling.physics_dispatch import scheme_entry
from gpuwrf.coupling.scan_adapters import (
    CU_SCAN_ADAPTERS,
    MP_SCAN_ADAPTERS,
    PBL_SCAN_ADAPTERS,
    SFCLAY_SCAN_ADAPTERS,
)
from gpuwrf.runtime.operational_mode import _SCAN_UNWIRED_REASON, _SCAN_WIRED_OPTIONS

HERE = Path(__file__).resolve().parent
OUT = HERE / "consolidation_integration_matrix.json"
SMOKE = HERE / "multicfg_smoke_report.json"

# (namelist-key, scheme-map, scan-adapter-table, dispatch-family) per physics axis.
_AXES = [
    ("mp_physics", MP_SCHEMES, MP_SCAN_ADAPTERS, "microphysics"),
    ("bl_pbl_physics", PBL_SCHEMES, PBL_SCAN_ADAPTERS, "pbl"),
    ("sf_sfclay_physics", SFCLAY_SCHEMES, SFCLAY_SCAN_ADAPTERS, "surface_layer"),
    ("cu_physics", CU_SCHEMES, CU_SCAN_ADAPTERS, "cumulus"),
    ("sf_surface_physics", SURFACE_SCHEMES, None, "land_surface"),
    ("ra_sw_physics", RA_SW_SCHEMES, None, "radiation"),
    ("ra_lw_physics", RA_LW_SCHEMES, None, "radiation"),
]

# Land + radiation operational-wiring is not table-driven via *_SCAN_ADAPTERS
# (land threads a carry hook; radiation is a column-endpoint held-rate driver).
# Map the wired land/radiation options explicitly to the authoritative hook so
# the matrix stays honest rather than guessing.
_LAND_WIRED = {0, 2, 4}  # 0 off, 2 Noah-classic hook, 4 Noah-MP hook
# HONESTY (close-critic FIX-NOW #1/#2, 2026-06-04): only RRTMG (ra=4) is actually
# selected by the operational radiation slot. runtime.operational_mode hardcodes the
# RRTMG held-rate RTHRATEN (rrtmg_theta_tendency = combined LW+SW); OperationalNamelist
# has NO ra_lw_physics/ra_sw_physics field and there is NO radiation-family dispatch.
# So ra_lw=1 (classic RRTM-LW) / ra_sw=1 (Dudhia-SW) are isolated-savepoint parity-proven
# + accepted but NOT operational-scan-wired -- the same fail-closed posture as MYJ/Janjic.
# (Classic RRTM-LW is moreover a host-NumPy single-column kernel that is not jit/vmap-
# traceable, so it cannot ride the device scan as-is.) They are therefore NOT counted as
# GPU-OPERATIONAL-WIRED; they read PARITY-PROVEN-FAIL-CLOSED via _RAD_NOT_SCAN_WIRED below.
_RAD_WIRED = {0, 4}      # only RRTMG (4) is the operational radiation; 0 = off/passive
_RAD_NOT_SCAN_WIRED = {  # parity-proven + accepted but NOT operational-scan-selectable
    ("ra_lw_physics", 1): (
        "classic RRTM-LW is isolated-WRF-savepoint parity-proven + accepted, but NOT "
        "operational-scan-wired: OperationalNamelist has no ra_lw_physics field, the radiation "
        "slot hardcodes RRTMG (ra=4), and the host-NumPy single-column RRTM-LW kernel is not "
        "jit/vmap-traceable for the device scan (post-0.9.0 carry-over)"
    ),
    ("ra_sw_physics", 1): (
        "Dudhia-SW is isolated-WRF-savepoint parity-proven + accepted, but NOT operational-scan-"
        "wired: OperationalNamelist has no ra_sw_physics field and the radiation slot hardcodes "
        "RRTMG (ra=4) (no radiation-family dispatch yet; post-0.9.0 carry-over)"
    ),
}


def _detail_for(key: str, opt: int, adapters, family: str) -> tuple[str, str]:
    """Return (status, detail) for one accepted option, derived from live tables."""

    if opt == 0:
        return "PASSIVE/OFF", "n/a (no-op by design)"

    wired = opt in _SCAN_WIRED_OPTIONS.get(key, ())
    if key == "sf_surface_physics":
        wired = opt in _LAND_WIRED
    if key in ("ra_sw_physics", "ra_lw_physics"):
        wired = opt in _RAD_WIRED

    if wired:
        if adapters is not None and opt in adapters:
            return "GPU-OPERATIONAL-WIRED", f"scan_adapters[{opt}]={adapters[opt].__name__}"
        if key == "sf_surface_physics":
            hook = {2: "coupling.noahclassic_surface_hook", 4: "coupling.noahmp_surface_hook"}[opt]
            return "GPU-OPERATIONAL-WIRED", hook
        if key in ("ra_sw_physics", "ra_lw_physics"):
            # Only RRTMG (opt 4) is operational (the radiation slot hardcodes it).
            owner = {
                ("ra_lw_physics", 4): "physics.rrtmg_lw (RRTMG LW)",
                ("ra_sw_physics", 4): "physics.rrtmg_sw (RRTMG SW)",
            }[(key, opt)]
            # RRTMG (opt 4): the real WRF-oracle evidence is the B3 proofs, NOT the
            # stale M5 artifacts (artifacts/m5/tier1_rrtmg_sw_parity.json pass=false
            # is superseded; see artifacts/m5/SUPERSEDED_rrtmg_see_proofs_b3.json).
            proof_ptr = (
                " | proof: proofs/b3/real_wrf_fixture_parity.json (pass, NOT self-compare; "
                "SW surface-down max_abs 0.0238 W/m2, LW 4.97e-5 W/m2) "
                "[M5 RRTMG artifacts SUPERSEDED]"
            )
            return "GPU-OPERATIONAL-WIRED", f"held-rate RTHRATEN column driver: {owner}{proof_ptr}"
        # mp=8 Thompson is wired through the existing coupler (not the table).
        entry = scheme_entry(family, opt)
        return "GPU-OPERATIONAL-WIRED", f"{entry.owner_module}.{entry.entrypoint}"

    # Radiation: ra_lw=1 / ra_sw=1 are parity-proven + accepted but NOT operational-
    # scan-wired (the slot hardcodes RRTMG ra=4). Honest fail-closed posture (close-
    # critic FIX-NOW #1/#2). Their isolated savepoint parity IS real, so they read
    # PARITY-PROVEN-FAIL-CLOSED (same bucket as MYJ/Janjic), not ACCEPTED-NOT-GATED.
    if (key, opt) in _RAD_NOT_SCAN_WIRED:
        return "PARITY-PROVEN-FAIL-CLOSED", _RAD_NOT_SCAN_WIRED[(key, opt)]

    reason = _SCAN_UNWIRED_REASON.get(f"{key}={opt}", "scan-unwired (see operational_mode)")
    # HONESTY (audit overclaim #4): a few accepted options are fail-closed but are
    # NOT individually WRF-savepoint-proven. New Tiedtke (cu=16) shares the cu=6
    # kernel but has no distinct source-path savepoint gate, so it must NOT be
    # labeled "PARITY-PROVEN". Label it as accepted/fail-closed/not-separately-gated.
    if f"{key}={opt}" in _ACCEPTED_NOT_SEPARATELY_GATED:
        return "ACCEPTED-FAIL-CLOSED (NOT separately source-gated)", reason
    return "PARITY-PROVEN-FAIL-CLOSED", reason


# Accepted options that are fail-closed but NOT individually WRF-savepoint-proven
# (they share another option's kernel / lack a distinct source-path gate). These
# must never read as "parity-proven".
_ACCEPTED_NOT_SEPARATELY_GATED = frozenset({"cu_physics=16"})


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE).decode().strip()
    except Exception:  # pragma: no cover - non-git contexts
        return "unknown"


def build() -> dict:
    per_scheme: dict[str, dict] = {}
    counts = {
        "gpu_operational_wired": 0,
        "parity_proven_fail_closed": 0,
        "accepted_fail_closed_not_separately_gated": 0,
        "passive_off": 0,
        "unknown_investigate": 0,
    }
    fail_closed: list[str] = []
    for key, schemes, adapters, family in _AXES:
        for opt in sorted(schemes):
            status, detail = _detail_for(key, opt, adapters, family)
            per_scheme[f"{key}={opt}"] = {
                "name": schemes[opt].name,
                "status": status,
                "detail": detail,
            }
            if status == "GPU-OPERATIONAL-WIRED":
                counts["gpu_operational_wired"] += 1
            elif status == "PARITY-PROVEN-FAIL-CLOSED":
                counts["parity_proven_fail_closed"] += 1
                fail_closed.append(f"{key}={opt}")
            elif status.startswith("ACCEPTED-FAIL-CLOSED"):
                counts["accepted_fail_closed_not_separately_gated"] += 1
                fail_closed.append(f"{key}={opt}")
            elif status == "PASSIVE/OFF":
                counts["passive_off"] += 1
            else:
                counts["unknown_investigate"] += 1

    smoke = json.loads(SMOKE.read_text())
    integration = {
        "report": "proofs/v060/multicfg_smoke_report.json",
        "all_pass": smoke.get("all_pass"),
        "n_configs": smoke.get("n_configs"),
        "n_run_configs": smoke.get("n_run_configs"),
        "n_run_pass": smoke.get("n_run_pass"),
        "n_fail_closed_configs": smoke.get("n_fail_closed_configs"),
        "n_fail_closed_ok": smoke.get("n_fail_closed_ok"),
        "scheme_coverage": smoke.get("scheme_coverage"),
    }

    overall = (
        counts["unknown_investigate"] == 0
        and integration["all_pass"] is True
        and integration["n_run_pass"] == integration["n_run_configs"]
        and integration["n_fail_closed_ok"] == integration["n_fail_closed_configs"]
    )

    return {
        "proof": "v060-consolidation-integration-matrix",
        "git_head": _git_head(),
        "branch": "worker/opus/v060-consolidation4",
        "base_trunk": "e998250 (trunk-0.9.0)",
        "wave1_base": "worker/opus/v060-consolidation (dcc9666)",
        "merged_branches": [
            "WAVE-1 (already in base dcc9666): v060-close (c38a3c0), v060-myj (4807e28), "
            "v060-wsm-sm (dec11a1), v060-ysu-acm2-gpuop (44aa8df), v060-gf-tiedtke-gpu (42534d8)",
            "WAVE-2 (consolidation2): worker/opus/v060-lin-mp (72a41c5)",
            "WAVE-2 (consolidation2): worker/opus/v060-boulac (8837e80)",
            "WAVE-2 (consolidation2): worker/opus/v060-radiation (49114db)",
            "WAVE-3 (consolidation3): worker/opus/v060-bmj-fix2 (fcf4346) -- BMJ cu=2 fp64-proven",
        ],
        "kind": (
            "Consolidation per-scheme operational/parity/fail-closed status matrix + "
            "integration-smoke result. Derived from the LIVE merged registries "
            "(physics_registry scheme maps, scan_adapters adapter tables, "
            "operational_mode._SCAN_WIRED_OPTIONS/_SCAN_UNWIRED_REASON, "
            "physics_dispatch) + the multicfg operational smoke "
            "(proofs/v060/multicfg_smoke_report.json). HONEST: every non-operational "
            "scheme is fail-closed (loud), never silently skipped. Fail-closed schemes "
            "that ARE individually WRF-savepoint-proven read PARITY-PROVEN-FAIL-CLOSED; "
            "New Tiedtke (cu=16), which shares the cu=6 kernel and has no distinct "
            "source-path savepoint gate, reads ACCEPTED-FAIL-CLOSED (NOT separately "
            "source-gated) so it is never mislabeled parity-proven."
        ),
        "per_scheme_status": per_scheme,
        "counts": counts,
        "fail_closed_schemes": fail_closed,
        "integration_smoke": integration,
        "carry_overs_post_0_9_0": [
            "GF (cu=3): faithful GPU-batch closure-ensemble + beta-PDF gamma (~2000-LOC dedicated sprint).",
            "New-Tiedtke (cu=16): separate WRF-source savepoint gate + GPU-batch.",
            "MYJ (bl=2) + Janjic (sf=2): GPU-scan-wire the parity-proven CPU references.",
            "Noah-classic (land=2) real-run static/land bundle assembly for canonical combo.",
            "Radiation operational scan-wiring: ra_lw=1 (classic RRTM-LW) and ra_sw=1 "
            "(Dudhia-SW) are isolated-WRF-savepoint parity-proven + accepted, but NOT "
            "operational-scan-wired (close-critic FIX #1/#2). The operational radiation slot "
            "hardcodes the RRTMG (ra=4) held-rate RTHRATEN; OperationalNamelist has no "
            "ra_lw_physics/ra_sw_physics field and there is no radiation-family dispatch. "
            "Classic RRTM-LW is additionally a host-NumPy single-column kernel not yet jit/vmap-"
            "traceable for the device scan. A radiation-family dispatch + a jit/vmap RRTM-LW "
            "rewrite is the post-0.9.0 carry-over; until then ra=1 is fail-closed (same posture "
            "as MYJ/Janjic).",
        ],
        "overall_consolidation_pass": overall,
    }


def main() -> None:
    matrix = build()
    OUT.write_text(json.dumps(matrix, indent=2) + "\n")
    c = matrix["counts"]
    print(f"per_scheme_status: {len(matrix['per_scheme_status'])} options")
    print(f"  GPU-OPERATIONAL-WIRED                 = {c['gpu_operational_wired']}")
    print(f"  PARITY-PROVEN-FAIL-CLOSED             = {c['parity_proven_fail_closed']}")
    print(f"  ACCEPTED-FAIL-CLOSED (not src-gated)  = {c['accepted_fail_closed_not_separately_gated']}")
    print(f"  PASSIVE/OFF                           = {c['passive_off']}")
    print(f"  UNKNOWN/INVESTIGATE                   = {c['unknown_investigate']}")
    print(f"  all fail-closed schemes               = {matrix['fail_closed_schemes']}")
    s = matrix["integration_smoke"]
    print(f"smoke: RUN {s['n_run_pass']}/{s['n_run_configs']} PASS; "
          f"FAIL-CLOSED {s['n_fail_closed_ok']}/{s['n_fail_closed_configs']} OK; all_pass={s['all_pass']}")
    print(f"overall_consolidation_pass = {matrix['overall_consolidation_pass']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
