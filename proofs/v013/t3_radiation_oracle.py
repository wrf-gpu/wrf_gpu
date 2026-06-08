#!/usr/bin/env python3
"""v0.13 Tier3 RADIATION-family oracle proof -- GSFC (Chou-Suarez) shortwave.

Validates the newly-operational GSFC shortwave scheme (``ra_sw_physics=2``)
against the INDEPENDENT pristine-WRF GSFC oracle: the real unmodified
``phys/module_ra_gsfcsw.F:GSFCSWRAD`` source compiled standalone by
``proofs/radiation/oracle/gsfcsw_build_and_run.sh`` (fp32 = canonical WRF REAL*4
build; fp64 = ``-fdefault-real-8`` precision audit). The JAX port is NEVER used
to create the reference; there is NO JAX-vs-JAX self-compare.

Two distinct things are proven, per pristine-WRF savepoint case (7 regimes:
clear-sky hi/lo sun, night, thick warm cloud, ice/snow marine cloud,
snow/graupel cloud, terminator):

  (A) BARE COLUMN KERNEL -- ``physics.ra_sw_gsfc.solve_gsfc_sw_column`` is run on
      the savepoint column inputs (T/p/p8w/moisture/cloud-fraction +
      COSZEN/ALBEDO/SOLCON/JULDAY/CENTER_LAT injected from the savepoint), and
      the kernel ``dT/dt`` is converted to ``RTHRATEN = max(dT/dt,0)/pi3D``
      exactly as ``GSFCSWRAD`` does. Compared to the oracle ``RTHRATEN`` plus the
      surface net SW flux ``GSW`` and the TOA upward residual ``RSWTOA`` within a
      predeclared tolerance.

  (B) WIRED OPERATIONAL PATH -- the real ``gsfc_sw_theta_tendency(state, grid,
      ...)`` coupler is run end-to-end on a real operational ``State`` built so
      its mass profiles reproduce the savepoint column (its OWN solar geometry /
      albedo / solcon derivation, which is the shared RRTMG-validated code), and
      asserted finite + physically bounded (daytime SW heating non-negative,
      magnitude < 1e-2 K/s). This proves the wired radiation-slot path a
      ``ra_sw_physics=2`` forecast actually executes stays sane.

The fp32 (canonical) and fp64 (precision-audit) oracles are both checked; the
canonical fp32 build is the WRF source-of-truth (single-precision dust flows
through the long sequential multi-band/k-distribution/cloud-overlap accumulation,
so the canonical tolerance is looser than the fp64 round-off floor).

Run (CPU dev; GPU is owned by another worker -- NEVER use it here):
    JAX_PLATFORMS=cpu PYTHONPATH=src python3 proofs/v013/t3_radiation_oracle.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.contracts.grid import (  # noqa: E402
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.state import State, _state_field_shapes  # noqa: E402
from gpuwrf.coupling.physics_couplers import (  # noqa: E402
    GRAVITY_M_S2,
    P0_PA,
    R_D_OVER_CP,
    gsfc_sw_theta_tendency,
)
from gpuwrf.physics.ra_sw_gsfc import GsfcSWColumnState, solve_gsfc_sw_column  # noqa: E402

SAVE_FP32 = ROOT / "proofs" / "radiation" / "savepoints_gsfcsw"
SAVE_FP64 = ROOT / "proofs" / "radiation" / "savepoints_gsfcsw_fp64"
REPORT = HERE / "t3_radiation_oracle.json"

CASE_IDS = (1, 2, 3, 4, 5, 6, 7)

# PREDECLARED TOLERANCES, frozen before any comparison. The canonical oracle is
# WRF REAL*4: single-precision error accumulates through 8 UV/PAR bands + 3 IR
# bands x 10 k-intervals + the recursive cloud-overlap adding, so the canonical
# RTHRATEN tolerance is a relative bound on the column-max heating rate (the
# small absolute heating in upper layers has large relative noise). The fp64
# precision-audit build matches the fp64 JAX port to deep round-off.
PREDECLARED_TOL = {
    "fp32": {
        "rthraten_abs": 1.0e-6,    # K/s absolute floor
        "rthraten_rel": 5.0e-3,    # relative on column-max heating rate
        "gsw_rel": 5.0e-3,
        "gsw_abs": 5.0e-2,         # W/m^2 absolute floor
        "rswtoa_rel": 5.0e-3,
        "rswtoa_abs": 5.0e-2,
    },
    "fp64": {
        "rthraten_abs": 1.0e-9,
        "rthraten_rel": 1.0e-6,
        "gsw_rel": 1.0e-7,
        "gsw_abs": 1.0e-4,
        "rswtoa_rel": 1.0e-7,
        "rswtoa_abs": 1.0e-4,
    },
    "sanity_max_rthraten": 1.0e-2,
}


def col(d: dict, name: str) -> np.ndarray:
    return np.asarray(d["columns"][name], dtype=np.float64)


def scalar(d: dict, name: str) -> float:
    return float(d["scalars"][name])


def _grid_for(nz: int, ny: int = 1, nx: int = 1, lat0: float = 28.0) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    terrain_height = jnp.zeros((ny, nx), dtype=jnp.float64)
    projection = Projection("lambert", lat0, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="t3-gsfc-proof", sha256="t3-gsfc-proof", shape=(ny, nx), units="m",
        projection_transform="native-wrf-lambert", max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    metrics = DycoreMetrics.flat(
        ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=5000.0, provenance="t3-gsfc-flat",
    )
    return GridSpec(projection, terrain_meta, vertical, bc, eta, terrain_height, metrics=metrics)


def _state_from_savepoint(d: dict, grid: GridSpec) -> State:
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    shapes = _state_field_shapes(grid)
    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in shapes.items()}
    T = col(d, "T")
    p = col(d, "P")
    dz = col(d, "DZ")
    exner = (np.maximum(p, 1.0) / P0_PA) ** R_D_OVER_CP
    theta = T / exner
    z_iface = np.concatenate([[0.0], np.cumsum(dz)])
    ph = z_iface * GRAVITY_M_S2

    def b3(profile):
        return jnp.broadcast_to(jnp.asarray(profile)[:, None, None], (nz, ny, nx))

    fields.update(
        theta=b3(theta), p=b3(p),
        ph=jnp.broadcast_to(jnp.asarray(ph)[:, None, None], (nz + 1, ny, nx)),
        qv=b3(col(d, "QV")), qc=b3(col(d, "QC")), qr=b3(col(d, "QR")),
        qi=b3(col(d, "QI")), qs=b3(col(d, "QS")), qg=b3(col(d, "QG")),
        t_skin=jnp.full((ny, nx), float(T[0]), dtype=jnp.float64),
        xland=jnp.full((ny, nx), 1.0, dtype=jnp.float64),
        lu_index=jnp.zeros((ny, nx), dtype=jnp.int32),
    )
    return State(**fields)


def run_case(d: dict, tol: dict):
    nz = len(col(d, "T"))
    one = jnp.ones((1,), dtype=jnp.float64)

    # (A) bare column kernel on the savepoint inputs.
    column = GsfcSWColumnState(
        T=col(d, "T")[None, :], p=col(d, "P")[None, :], p8w=col(d, "P8W")[None, :],
        qv=col(d, "QV")[None, :], qc=col(d, "QC")[None, :], qr=col(d, "QR")[None, :],
        qi=col(d, "QI")[None, :], qs=col(d, "QS")[None, :], qg=col(d, "QG")[None, :],
        dz=col(d, "DZ")[None, :], cldfra=col(d, "CLDFRA")[None, :],
        coszen=one * scalar(d, "COSZEN"), albedo=one * scalar(d, "ALBEDO"),
        solcon=one * scalar(d, "SOLCON"), julday=int(scalar(d, "JULDAY")),
        center_lat=scalar(d, "CENTER_LAT"), f_qi=True, warm_rain=False,
    )
    out = solve_gsfc_sw_column(column)
    pi = col(d, "PI")
    # WRF: RTHRATEN += max(TTEN,0)/pi3D. The kernel already returns max(.,0) dT/dt.
    kernel_rth = np.asarray(out.heating_rate)[0] / pi
    kernel_gsw = float(np.asarray(out.gsw)[0])
    kernel_rswtoa = float(np.asarray(out.rswtoa)[0])

    oracle_rth = col(d, "RTHRATEN")
    oracle_gsw = scalar(d, "GSW")
    oracle_rswtoa = scalar(d, "RSWTOA")

    scale = max(float(np.max(np.abs(oracle_rth))), tol["rthraten_abs"])
    rth_abs = float(np.max(np.abs(kernel_rth - oracle_rth)))
    rth_rel = rth_abs / scale
    rth_ok = (rth_rel <= tol["rthraten_rel"]) or (rth_abs <= tol["rthraten_abs"])

    gsw_tol = max(tol["gsw_rel"] * abs(oracle_gsw), tol["gsw_abs"])
    gsw_err = abs(kernel_gsw - oracle_gsw)
    gsw_ok = gsw_err <= gsw_tol

    rswtoa_tol = max(tol["rswtoa_rel"] * abs(oracle_rswtoa), tol["rswtoa_abs"])
    rswtoa_err = abs(kernel_rswtoa - oracle_rswtoa)
    rswtoa_ok = rswtoa_err <= rswtoa_tol

    # (B) full wired coupler path sanity.
    grid = _grid_for(nz, lat0=scalar(d, "CENTER_LAT"))
    state = _state_from_savepoint(d, grid)
    full_rth = np.asarray(
        gsfc_sw_theta_tendency(state, grid, time_utc="2019-06-21T12:00:00Z")
    )
    finite = bool(np.all(np.isfinite(full_rth)))
    max_abs_full = float(np.max(np.abs(full_rth)))
    nonneg = bool(np.min(full_rth) >= -1.0e-12)
    sanity_ok = finite and (max_abs_full <= PREDECLARED_TOL["sanity_max_rthraten"]) and nonneg

    passed = bool(rth_ok and gsw_ok and rswtoa_ok and sanity_ok)
    return passed, {
        "label": d["scalars"]["REGIME"],
        "coszen": scalar(d, "COSZEN"),
        "albedo": scalar(d, "ALBEDO"),
        "solcon": scalar(d, "SOLCON"),
        "bare_kernel": {
            "RTHRATEN": {"max_abs": rth_abs, "max_rel": rth_rel, "scale": scale,
                         "tol_abs": tol["rthraten_abs"], "tol_rel": tol["rthraten_rel"],
                         "pass": bool(rth_ok)},
            "GSW": {"kernel": kernel_gsw, "oracle": oracle_gsw, "abs_err": gsw_err,
                    "tol": gsw_tol, "pass": bool(gsw_ok)},
            "RSWTOA": {"kernel": kernel_rswtoa, "oracle": oracle_rswtoa, "abs_err": rswtoa_err,
                       "tol": rswtoa_tol, "pass": bool(rswtoa_ok)},
        },
        "full_wired_path_sanity": {
            "finite": finite, "max_abs_rthraten": max_abs_full,
            "nonnegative_daytime_sw": nonneg, "bound": PREDECLARED_TOL["sanity_max_rthraten"],
            "pass": bool(sanity_ok),
        },
        "pass": passed,
    }


def read_text_if_present(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    fp32_cases, fp64_cases = {}, {}
    fp32_pass = fp64_pass = True

    print("=== FP32 (canonical WRF REAL*4 oracle) ===")
    for cid in CASE_IDS:
        with open(SAVE_FP32 / f"gsfcsw_case_{cid}.json", encoding="utf-8") as fh:
            d = json.load(fh)
        ok, res = run_case(d, PREDECLARED_TOL["fp32"])
        fp32_cases[str(cid)] = res
        fp32_pass = fp32_pass and ok
        bk = res["bare_kernel"]
        print(f"CASE {cid} {res['label']:30s} cos={res['coszen']:.2f} -> {'PASS' if ok else 'FAIL'} | "
              f"RTH rel={bk['RTHRATEN']['max_rel']:.2e} | GSW {bk['GSW']['kernel']:.3f}/"
              f"{bk['GSW']['oracle']:.3f} err={bk['GSW']['abs_err']:.2e} | "
              f"RSWTOA err={bk['RSWTOA']['abs_err']:.2e} | sane={res['full_wired_path_sanity']['pass']}")

    print("\n=== FP64 (precision-audit -fdefault-real-8 oracle) ===")
    for cid in CASE_IDS:
        with open(SAVE_FP64 / f"gsfcsw_case_{cid}.json", encoding="utf-8") as fh:
            d = json.load(fh)
        ok, res = run_case(d, PREDECLARED_TOL["fp64"])
        fp64_cases[str(cid)] = res
        fp64_pass = fp64_pass and ok
        bk = res["bare_kernel"]
        print(f"CASE {cid} {res['label']:30s} -> {'PASS' if ok else 'FAIL'} | "
              f"RTH abs={bk['RTHRATEN']['max_abs']:.2e} rel={bk['RTHRATEN']['max_rel']:.2e} | "
              f"GSW err={bk['GSW']['abs_err']:.2e}")

    overall = bool(fp32_pass and fp64_pass)
    report = {
        "scheme": "GSFC (Chou-Suarez) shortwave (ra_sw_physics=2) -- bare kernel + operational wiring",
        "what_is_proven": (
            "the JAX port physics.ra_sw_gsfc.solve_gsfc_sw_column reproduces the "
            "pristine-WRF GSFCSWRAD RTHRATEN/GSW/RSWTOA oracle (fp32 canonical + "
            "fp64 precision-audit); and the wired operational coupler "
            "coupling.physics_couplers.gsfc_sw_theta_tendency runs finite + "
            "physically-bounded on a real State. The scheme is now operationally "
            "scan-wired in the radiation slot (ra_sw_physics=2)."
        ),
        "verdict": "PASS" if overall else "FAIL",
        "overall_pass": overall,
        "canonical_fp32_pass": bool(fp32_pass),
        "fp64_precision_audit_pass": bool(fp64_pass),
        "self_compare": False,
        "self_compare_note": (
            "The reference is the unmodified pristine WRF phys/module_ra_gsfcsw.F "
            "GSFCSWRAD source compiled standalone (proofs/radiation/oracle/"
            "gsfcsw_build_and_run.sh); the JAX port is never used to make the "
            "reference."
        ),
        "oracle": {
            "source": "$WRF_PRISTINE_ROOT/phys/module_ra_gsfcsw.F",
            "entry": "GSFCSWRAD -> sorad (Chou-Suarez multi-band delta-Eddington SW)",
            "source_unmodified": True,
            "full_wrf_exe": False,
            "fp32_savepoints": "proofs/radiation/savepoints_gsfcsw (gsfcsw_case_*.json)",
            "fp64_savepoints": "proofs/radiation/savepoints_gsfcsw_fp64 (gsfcsw_case_*.json)",
            "fp32_source_checksums": read_text_if_present(SAVE_FP32 / "gsfcsw_wrf_source_checksums.txt"),
            "fp64_source_checksums": read_text_if_present(SAVE_FP64 / "gsfcsw_wrf_source_checksums.txt"),
        },
        "wired_path": {
            "kernel": "gpuwrf.physics.ra_sw_gsfc.solve_gsfc_sw_column",
            "coupler": "gpuwrf.coupling.physics_couplers.gsfc_sw_theta_tendency",
            "column_assembler": "gpuwrf.coupling.physics_couplers._gsfc_sw_column_inputs",
            "dispatch": (
                "gpuwrf.runtime.operational_mode radiation slot dispatches "
                "ra_sw_physics=2 -> GSFC SW + RRTMG/classic-RRTM LW; default "
                "ra_sw_physics=4 -> RRTMG SW+LW (byte-unchanged)."
            ),
            "tendency_conversion": "RTHRATEN = max(dT/dt,0)/exner (WRF GSFCSWRAD: TTEN2D/pi3D)",
        },
        "default_unchanged": (
            "ra_sw_physics default is 4 (RRTMG SW+LW); the GSFC path is opt-in via "
            "ra_sw_physics=2 only. The default RRTMG dispatch is untouched."
        ),
        "jax_precision": "fp64",
        "jax_platform": jax.default_backend(),
        "git_head": git_head(),
        "predeclared_tolerances": PREDECLARED_TOL,
        "fp32_cases": fp32_cases,
        "fp64_audit_cases": fp64_cases,
    }
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("\nOVERALL:", report["verdict"])
    print("wrote", REPORT)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
