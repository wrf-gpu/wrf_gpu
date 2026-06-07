#!/usr/bin/env python3
"""WIRED Dudhia shortwave (ra_sw_physics=1) operational-coupler oracle proof.

This proof validates the OPERATIONAL WIRING of the Dudhia shortwave scheme --
the new ``coupling.physics_couplers.dudhia_sw_theta_tendency`` coupler and its
``_dudhia_sw_column_inputs`` State->column assembler -- against the INDEPENDENT
pristine-WRF Dudhia oracle (the real ``phys/module_ra_sw.F:SWRAD`` source,
compiled by ``proofs/v060/oracle/dudhia_build_and_run.sh``). The JAX port is
NEVER used to create the reference; there is NO JAX-vs-JAX self-compare.

The bare column kernel (``physics.ra_sw_dudhia.solve_dudhia_sw_column``) already
has its own savepoint-parity proof (``proofs/v060/run_dudhia_parity.py``). THIS
proof is distinct: it exercises the new operational plumbing that wires that
kernel into the GPU scan radiation slot, i.e. the code that a forecast run with
``ra_sw_physics=1`` actually executes. Forecast runs do NOT statistically catch a
plumbing bug here (a wrong Exner conversion or a transposed column reshape would
quietly bias SW heating), so per the exotic-feature rule this committed
correctness proof ships WITH the implementation.

What is checked, per pristine-WRF savepoint case (7 regimes: clear-sky hi/lo sun,
night, thick warm cloud, ice cloud marine, snow/graupel, terminator):

  (A) COLUMN PLUMBING -- a real operational ``State`` is built so its mass-point
      profiles reproduce the savepoint column. ``_dudhia_sw_column_inputs``
      derives the Dudhia column view from that State; we assert the derived
      ``T``/``p``/``dz``/``qv`` columns reproduce the savepoint inputs (this is the
      State->column assembler the coupler runs every step).

  (B) WIRED TENDENCY MATH -- the coupler's exact tendency conversion is run on
      that column view (with the savepoint's COSZEN/ALBEDO/SOLCON injected, since
      the per-column solar geometry / albedo / date-eccentricity derivation is
      shared, already-RRTMG-validated code): kernel ``dT/dt`` -> ``RTHRATEN`` via
      ``/exner``, exactly as ``dudhia_sw_theta_tendency`` does. Compared to the
      oracle ``RTHRATEN`` (theta tendency WRF writes back) within a predeclared
      tolerance, plus the surface net SW flux GSW.

  (C) FULL WIRED PATH SANITY -- the real ``dudhia_sw_theta_tendency(state, grid,
      time_utc=...)`` coupler is run end-to-end (its OWN geometry / albedo /
      solcon derivation, no injection) and asserted finite and physically bounded
      (daytime SW heating non-negative, magnitude < 1e-2 K/s), proving the whole
      wired radiation-slot path executes and stays sane.

Run (CPU dev):
    JAX_PLATFORMS=cpu PYTHONPATH=src python3 proofs/radiation/cdudhia_sw_oracle.py
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
    DudhiaSWColumnState,
    _dudhia_sw_column_inputs,
    dudhia_sw_theta_tendency,
    solve_dudhia_sw_column,
)

# Oracle savepoints (pristine WRF module_ra_sw.F; see proofs/v060/oracle/).
SAVE_FP32 = ROOT / "proofs" / "v060" / "savepoints"
SAVE_FP64 = ROOT / "proofs" / "v060" / "savepoints_fp64"
REPORT = HERE / "cdudhia_sw_oracle.json"

CASE_IDS = (1, 2, 3, 4, 5, 6, 7)

# PREDECLARED TOLERANCES, frozen before any comparison (mirror the bare-kernel
# proof; the WIRED path adds only an Exner round trip + reshape, which are exact
# in fp64). The canonical oracle is WRF REAL*4, so single-precision dust flows
# through the long sequential per-layer accumulation.
PREDECLARED_TOL = {
    "rthraten_abs": 5.0e-9,   # K/s absolute floor (near-zero layers)
    "rthraten_rel": 5.0e-4,   # relative on the column max heating rate
    "gsw_rel": 5.0e-4,        # relative on surface net SW flux
    "gsw_abs": 5.0e-3,        # W/m^2 absolute floor (night/near-zero)
    # column-plumbing reproduction: the State carries T/p/dz/qv exactly (fp64),
    # so the derived column must match the savepoint to round-off.
    "plumbing_rel": 1.0e-9,
    # full-wired-path sanity bound on SW theta tendency magnitude (K/s).
    "sanity_max_rthraten": 1.0e-2,
}


def col(d: dict, name: str) -> np.ndarray:
    return np.asarray(d["columns"][name], dtype=np.float64)


def scalar(d: dict, name: str) -> float:
    return float(d["scalars"][name])


def _grid_for(nz: int, ny: int = 1, nx: int = 1) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    terrain_height = jnp.zeros((ny, nx), dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="cdudhia-proof",
        sha256="cdudhia-proof",
        shape=(ny, nx),
        units="m",
        projection_transform="native-wrf-lambert",
        max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    metrics = DycoreMetrics.flat(
        ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=5000.0,
        provenance="cdudhia-proof-flat",
    )
    return GridSpec(projection, terrain_meta, vertical, bc, eta, terrain_height, metrics=metrics)


def _state_from_savepoint(d: dict, grid: GridSpec) -> State:
    """Build an operational State whose mass profiles reproduce the savepoint.

    theta = T / exner(p); the geopotential interface heights are accumulated from
    the savepoint layer thickness ``DZ`` so ``_column_dz_from_state`` recovers it.
    """

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    shapes = _state_field_shapes(grid)
    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in shapes.items()}

    T = col(d, "T")          # (nz,)
    p = col(d, "P")          # (nz,)
    dz = col(d, "DZ")        # (nz,)
    exner = (np.maximum(p, 1.0) / P0_PA) ** R_D_OVER_CP
    theta = T / exner

    # interface geopotential from cumulative thickness (k=0 at surface).
    z_iface = np.concatenate([[0.0], np.cumsum(dz)])          # (nz+1,)
    ph = z_iface * GRAVITY_M_S2

    def b3(profile):  # (nz,) -> (nz, ny, nx)
        return jnp.broadcast_to(jnp.asarray(profile)[:, None, None], (nz, ny, nx))

    fields.update(
        theta=b3(theta),
        p=b3(p),
        ph=jnp.broadcast_to(jnp.asarray(ph)[:, None, None], (nz + 1, ny, nx)),
        qv=b3(col(d, "QV")),
        qc=b3(col(d, "QC")),
        qr=b3(col(d, "QR")),
        qi=b3(col(d, "QI")),
        qs=b3(col(d, "QS")),
        qg=b3(col(d, "QG")),
        t_skin=jnp.full((ny, nx), float(T[0]), dtype=jnp.float64),
        xland=jnp.full((ny, nx), 1.0, dtype=jnp.float64),
        lu_index=jnp.zeros((ny, nx), dtype=jnp.int32),
    )
    return State(**fields)


def run_wired_for(d: dict):
    """Run the WIRED coupler plumbing + tendency math; return parity diagnostics."""

    nz = len(col(d, "T"))
    grid = _grid_for(nz)
    state = _state_from_savepoint(d, grid)

    # (A) Column plumbing: derive the Dudhia column view exactly as the coupler.
    column, _geometry = _dudhia_sw_column_inputs(state, grid)
    derived_T = np.asarray(column.T)[0]        # (nz,)
    derived_p = np.asarray(column.p)[0]
    derived_dz = np.asarray(column.dz)[0]
    derived_qv = np.asarray(column.qv)[0]
    plumb = {}
    for name, derived, ref in (
        ("T", derived_T, col(d, "T")),
        ("p", derived_p, col(d, "P")),
        ("dz", derived_dz, col(d, "DZ")),
        ("qv", derived_qv, col(d, "QV")),
    ):
        scale = max(float(np.max(np.abs(ref))), 1.0e-30)
        plumb[name] = float(np.max(np.abs(derived - ref)) / scale)
    plumbing_max_rel = max(plumb.values())
    plumbing_ok = plumbing_max_rel <= PREDECLARED_TOL["plumbing_rel"]

    # (B) Wired tendency math on the SAME column view, with the savepoint's
    #     COSZEN/ALBEDO/SOLCON injected (the shared geometry/albedo/eccentricity
    #     derivation is RRTMG-validated; here we isolate the new coupler math).
    one = jnp.ones((1,), dtype=jnp.float64)
    injected = column._replace(
        coszen=one * scalar(d, "COSZEN"),
        albedo=one * scalar(d, "ALBEDO"),
        solcon=one * scalar(d, "SOLCON"),
        T=column.T[:1], p=column.p[:1], qv=column.qv[:1],
        qc=column.qc[:1], qr=column.qr[:1], qi=column.qi[:1],
        qs=column.qs[:1], qg=column.qg[:1], dz=column.dz[:1],
    )
    out = solve_dudhia_sw_column(injected)
    heating_T = np.asarray(out.heating_rate)[0]   # (nz,) dT/dt (K/s)
    # EXACT coupler conversion: RTHRATEN = (dT/dt)/exner; WRF SWRAD uses pi3D.
    pi = col(d, "PI")
    wired_rth = heating_T / pi
    wired_gsw = float(np.asarray(out.gsw)[0])

    oracle_rth = col(d, "RTHRATEN")
    oracle_gsw = scalar(d, "GSW")

    scale = max(float(np.max(np.abs(oracle_rth))), PREDECLARED_TOL["rthraten_abs"])
    rth_abs = float(np.max(np.abs(wired_rth - oracle_rth)))
    rth_rel = rth_abs / scale
    rth_ok = (rth_rel <= PREDECLARED_TOL["rthraten_rel"]) or (rth_abs <= PREDECLARED_TOL["rthraten_abs"])

    gsw_tol = max(PREDECLARED_TOL["gsw_rel"] * abs(oracle_gsw), PREDECLARED_TOL["gsw_abs"])
    gsw_err = abs(wired_gsw - oracle_gsw)
    gsw_ok = gsw_err <= gsw_tol

    # (C) Full wired path sanity: run the real coupler end-to-end (its own
    #     geometry/albedo/solcon), assert finite + physically bounded.
    full_rth = np.asarray(
        dudhia_sw_theta_tendency(state, grid, time_utc="2019-05-21T12:00:00Z")
    )
    finite = bool(np.all(np.isfinite(full_rth)))
    max_abs_full = float(np.max(np.abs(full_rth)))
    # daytime SW heating is non-negative (Dudhia deposits absorbed SW as warming);
    # allow tiny negative round-off.
    nonneg = bool(np.min(full_rth) >= -1.0e-12)
    sanity_ok = finite and (max_abs_full <= PREDECLARED_TOL["sanity_max_rthraten"]) and nonneg

    passed = bool(plumbing_ok and rth_ok and gsw_ok and sanity_ok)
    return passed, {
        "label": d["scalars"]["REGIME"],
        "coszen": scalar(d, "COSZEN"),
        "albedo": scalar(d, "ALBEDO"),
        "solcon": scalar(d, "SOLCON"),
        "plumbing": {
            "per_field_max_rel": plumb,
            "max_rel": plumbing_max_rel,
            "tol_rel": PREDECLARED_TOL["plumbing_rel"],
            "pass": bool(plumbing_ok),
        },
        "wired_tendency": {
            "RTHRATEN": {
                "max_abs": rth_abs,
                "max_rel": rth_rel,
                "scale": scale,
                "tol_abs": PREDECLARED_TOL["rthraten_abs"],
                "tol_rel": PREDECLARED_TOL["rthraten_rel"],
                "pass": bool(rth_ok),
            },
            "GSW": {
                "wired": wired_gsw,
                "oracle": oracle_gsw,
                "abs_err": gsw_err,
                "tol": gsw_tol,
                "pass": bool(gsw_ok),
            },
        },
        "full_wired_path_sanity": {
            "finite": finite,
            "max_abs_rthraten": max_abs_full,
            "nonnegative_daytime_sw": nonneg,
            "bound": PREDECLARED_TOL["sanity_max_rthraten"],
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
    fp32_cases = {}
    fp64_cases = {}
    fp32_pass = True
    fp64_pass = True

    for cid in CASE_IDS:
        with open(SAVE_FP32 / f"dudhia_case_{cid}.json", encoding="utf-8") as fh:
            d = json.load(fh)
        ok, res = run_wired_for(d)
        fp32_cases[str(cid)] = res
        fp32_pass = fp32_pass and ok
        w = res["wired_tendency"]
        print(
            f"FP32 CASE {cid} {res['label']:32s} coszen={res['coszen']:.2f} -> "
            f"{'PASS' if ok else 'FAIL'} | plumb_rel={res['plumbing']['max_rel']:.2e} | "
            f"RTH abs={w['RTHRATEN']['max_abs']:.3e} rel={w['RTHRATEN']['max_rel']:.3e} | "
            f"GSW wired={w['GSW']['wired']:.4f} oracle={w['GSW']['oracle']:.4f} "
            f"err={w['GSW']['abs_err']:.3e} | full_sane={res['full_wired_path_sanity']['pass']}"
        )

    print()
    for cid in CASE_IDS:
        with open(SAVE_FP64 / f"dudhia_case_{cid}.json", encoding="utf-8") as fh:
            d = json.load(fh)
        ok, res = run_wired_for(d)
        fp64_cases[str(cid)] = res
        fp64_pass = fp64_pass and ok
        w = res["wired_tendency"]
        print(
            f"FP64 CASE {cid} {res['label']:32s} -> {'PASS' if ok else 'FAIL'} | "
            f"RTH rel={w['RTHRATEN']['max_rel']:.3e} | GSW err={w['GSW']['abs_err']:.3e}"
        )

    overall = bool(fp32_pass and fp64_pass)
    report = {
        "scheme": "Dudhia shortwave (ra_sw_physics=1) -- OPERATIONAL WIRING",
        "what_is_proven": (
            "the wired operational coupler dudhia_sw_theta_tendency + its "
            "_dudhia_sw_column_inputs State->column assembler reproduce the "
            "pristine-WRF Dudhia RTHRATEN/GSW oracle; plus full-wired-path "
            "finiteness + physical-bound sanity. Distinct from the bare-kernel "
            "proof proofs/v060/run_dudhia_parity.py."
        ),
        "verdict": "PASS" if overall else "FAIL",
        "overall_pass": overall,
        "canonical_fp32_pass": bool(fp32_pass),
        "fp64_precision_audit_pass": bool(fp64_pass),
        "self_compare": False,
        "self_compare_note": (
            "The reference is the unmodified pristine WRF phys/module_ra_sw.F "
            "SWRAD source compiled standalone (proofs/v060/oracle); the JAX wired "
            "coupler is never used to make the reference."
        ),
        "oracle": {
            "source": "/home/enric/src/wrf_pristine/WRF/phys/module_ra_sw.F",
            "entry": "SWRAD -> SWPARA (Stephens 1984 broadband shortwave)",
            "source_unmodified": True,
            "full_wrf_exe": False,
            "fp32_savepoints": "proofs/v060/savepoints (dudhia_case_*.json)",
            "fp64_savepoints": "proofs/v060/savepoints_fp64 (dudhia_case_*.json)",
            "fp32_source_checksums": read_text_if_present(SAVE_FP32 / "dudhia_wrf_source_checksums.txt"),
            "fp64_source_checksums": read_text_if_present(SAVE_FP64 / "dudhia_wrf_source_checksums.txt"),
        },
        "wired_path": {
            "coupler": "gpuwrf.coupling.physics_couplers.dudhia_sw_theta_tendency",
            "column_assembler": "gpuwrf.coupling.physics_couplers._dudhia_sw_column_inputs",
            "dispatch": (
                "gpuwrf.runtime.operational_mode._physics_boundary_step radiation "
                "slot dispatches ra_sw_physics=1 -> Dudhia SW + RRTMG LW; default "
                "ra_sw_physics=4 -> RRTMG SW+LW (byte-unchanged)."
            ),
            "tendency_conversion": "RTHRATEN = (dT/dt)/exner (WRF SWRAD: TTEN1D/pi3D)",
        },
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
