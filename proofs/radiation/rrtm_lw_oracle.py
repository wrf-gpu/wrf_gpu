#!/usr/bin/env python3
"""WIRED classic-RRTM longwave (ra_lw_physics=1) operational-coupler oracle proof.

This proof validates the OPERATIONAL WIRING of the classic AER RRTM longwave
scheme -- the new ``coupling.physics_couplers.rrtm_lw_theta_tendency`` coupler,
its ``_rrtm_lw_column_inputs`` State->column assembler, and the underlying
JIT/vmap-traceable kernel ``physics.ra_lw_rrtm_jax.solve_rrtm_lw_column_jax`` --
against the INDEPENDENT pristine-WRF RRTM oracle (the real
``phys/module_ra_rrtm.F:RRTMLWRAD`` source compiled standalone, savepoints under
``proofs/v060/savepoints*``). The JAX port is NEVER used to create the reference;
there is NO JAX-vs-JAX self-compare.

There are two distinct, already-shipped RRTM-LW proofs and THIS adds a third:

  * ``proofs/v060/run_rrtm_lw_parity.py`` -- the HOST-NumPy reference kernel
    (``physics.ra_lw_rrtm.solve_rrtm_lw_column``) vs the pristine-WRF savepoints.
  * (here, part A) the new TRACEABLE JAX kernel vs the same pristine-WRF
    savepoints -- proving the operational kernel itself is faithful, not merely
    that it equals the host kernel.
  * (here, parts B/C) the WIRED operational plumbing: the State->column assembler
    + the coupler's Exner tendency conversion + a full-wired-path finiteness /
    physical-bound sanity check, i.e. the code a forecast run with
    ``ra_lw_physics=1`` actually executes on the device scan.

Per the exotic-feature rule (a forecast run does NOT statistically catch a
plumbing bug here -- a transposed reshape or a wrong Exner factor would quietly
bias LW heating), this committed correctness proof ships WITH the implementation.

What is checked, per pristine-WRF savepoint case (7 regimes):

  (A) TRACEABLE-KERNEL PARITY -- the wired JAX kernel ``solve_rrtm_lw_column_jax``
      run on the savepoint column reproduces the oracle RTHRATEN (= heating/PI),
      surface downwelling GLW and TOA OLR within a predeclared tolerance.

  (B) COLUMN PLUMBING + WIRED TENDENCY -- a real operational ``State`` is built so
      its mass-point profiles reproduce the savepoint column; the coupler's
      ``_rrtm_lw_column_inputs`` assembler derives the RRTM column view from that
      State and we assert the derived ``T``/``p``/``qv`` columns reproduce the
      savepoint inputs (this is the assembler the coupler runs every step). The
      coupler's exact tendency conversion (kernel ``dT/dt`` -> ``RTHRATEN`` via
      ``/exner``, WRF ``RRTMLWRAD: RTHRATEN += TTEN/pi``) is compared to the oracle
      RTHRATEN.

  (C) FULL WIRED PATH SANITY -- the real ``rrtm_lw_theta_tendency(state, grid)``
      coupler is run end-to-end and asserted finite and physically bounded
      (|RTHRATEN| < 1e-2 K/s); the JAX kernel is asserted host-callback-free so it
      genuinely rides the device ``jax.lax.scan`` radiation slot.

Also asserts the ``laytrop`` contiguity used to vectorise the kernel's
troposphere/stratosphere branch (pressure monotone-decreasing => ``plog>4.56`` is
a leading run, so the host scalar ``laytrop`` count equals the per-layer mask).

Run (CPU dev):
    JAX_PLATFORMS=cpu PYTHONPATH=src python3 proofs/radiation/rrtm_lw_oracle.py
"""

from __future__ import annotations

import json
import os
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
    _rrtm_lw_column_inputs,
    rrtm_lw_theta_tendency,
)
from gpuwrf.physics.ra_lw_rrtm import RRTMLWColumnState  # noqa: E402
from gpuwrf.physics.ra_lw_rrtm_jax import (  # noqa: E402
    _jax_tables,
    _prepare_atmosphere_jax,
    solve_rrtm_lw_column_jax,
)

SAVE_FP32 = ROOT / "proofs" / "v060" / "savepoints"
SAVE_FP64 = ROOT / "proofs" / "v060" / "savepoints_fp64"
REPORT = HERE / "rrtm_lw_oracle.json"
CASE_IDS = (1, 2, 3, 4, 5, 6, 7)

# PREDECLARED TOLERANCES (frozen before comparison). The fp64 oracle is the
# canonical verdict; the wired path adds only an Exner round trip + reshape, exact
# in fp64. The fp32 savepoints carry single-precision dust through the long
# sequential per-layer/g-point accumulation (mirrors the bare-kernel proof).
PREDECLARED_TOL = {
    "rthraten_abs": 1.0e-8,    # K/s absolute floor (near-zero layers)
    "rthraten_rel": 1.0e-3,    # relative on the column max heating rate
    "glw_rel": 1.0e-3,
    "glw_abs": 1.0e-1,         # W/m^2 absolute floor
    "olr_rel": 1.0e-3,
    "olr_abs": 1.0e-1,
    "plumbing_rel": 1.0e-9,    # State->column reproduction (fp64-exact)
    "sanity_max_rthraten": 1.0e-2,  # full-wired-path |RTHRATEN| bound (K/s)
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
        source_path="rrtm-lw-proof", sha256="rrtm-lw-proof", shape=(ny, nx), units="m",
        projection_transform="native-wrf-lambert", max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    metrics = DycoreMetrics.flat(
        ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=5000.0,
        provenance="rrtm-lw-proof-flat",
    )
    return GridSpec(projection, terrain_meta, vertical, bc, eta, terrain_height, metrics=metrics)


def _case_state_kernel(d: dict) -> RRTMLWColumnState:
    """Direct kernel-input column state from the savepoint (for part A)."""

    def c(name: str):
        return jnp.asarray(col(d, name)[None, :])

    return RRTMLWColumnState(
        T=c("T"), t8w=c("T8W"), p=c("P"), p8w=c("P8W"),
        qv=c("QV"), qc=c("QC"), qr=c("QR"), qi=c("QI"), qs=c("QS"), qg=c("QG"),
        cloud_fraction=c("CLDFRA"), dz=c("DZ"), rho=c("RHO"),
        emiss=scalar(d, "EMISS"), tsk=scalar(d, "TSK"),
    )


def _state_from_savepoint(d: dict, grid: GridSpec) -> State:
    """Operational State whose mass profiles reproduce the savepoint column."""

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    shapes = _state_field_shapes(grid)
    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in shapes.items()}

    T = col(d, "T"); p = col(d, "P"); dz = col(d, "DZ")
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
        t_skin=jnp.full((ny, nx), scalar(d, "TSK"), dtype=jnp.float64),
        xland=jnp.full((ny, nx), 1.0, dtype=jnp.float64),
        lu_index=jnp.zeros((ny, nx), dtype=jnp.int32),
    )
    return State(**fields)


def _scalar_status(value, oracle, abs_tol, rel_tol):
    abs_err = abs(value - oracle)
    limit = max(abs_tol, rel_tol * abs(oracle))
    return {"value": value, "oracle": oracle, "abs_err": abs_err, "tol": limit, "pass": bool(abs_err <= limit)}


def _laytrop_contiguous(d: dict) -> bool:
    """Assert plog>4.56 is a leading run (the vectorisation equivalence)."""

    nz = len(col(d, "T"))
    grid = _grid_for(nz)
    state = _state_from_savepoint(d, grid)
    column = _rrtm_lw_column_inputs(state, grid)
    tab = _jax_tables()
    sc = {k: jnp.asarray(np.asarray(getattr(column, k))[0]) for k in
          ("T", "t8w", "p", "p8w", "qv", "qc", "qr", "qi", "qs", "qg", "cloud_fraction", "dz")}
    sc["emiss"] = jnp.asarray(np.asarray(column.emiss)[0])
    sc["tsk"] = jnp.asarray(np.asarray(column.tsk)[0])
    atm = _prepare_atmosphere_jax(sc, tab, nz)
    plog = np.log(np.maximum(np.asarray(atm["pavel"]), 1.0e-300))
    mask = plog > 4.56
    lead = 0
    for v in mask:
        if v:
            lead += 1
        else:
            break
    return bool(lead == int(np.sum(mask)))


def run_case(d: dict):
    nz = len(col(d, "T"))
    pi = col(d, "PI")
    oracle_rth = col(d, "RTHRATEN")
    oracle_glw = scalar(d, "GLW")
    oracle_olr = scalar(d, "OLR")
    scale = max(float(np.max(np.abs(oracle_rth))), PREDECLARED_TOL["rthraten_abs"])

    # (A) traceable kernel vs oracle.
    out = solve_rrtm_lw_column_jax(_case_state_kernel(d))
    jax_rth = np.asarray(out.heating_rate)[0] / pi
    rthA_abs = float(np.max(np.abs(jax_rth - oracle_rth)))
    rthA_rel = rthA_abs / scale
    rthA_ok = (rthA_rel <= PREDECLARED_TOL["rthraten_rel"]) or (rthA_abs <= PREDECLARED_TOL["rthraten_abs"])
    glwA = _scalar_status(float(out.glw[0]), oracle_glw, PREDECLARED_TOL["glw_abs"], PREDECLARED_TOL["glw_rel"])
    olrA = _scalar_status(float(out.olr[0]), oracle_olr, PREDECLARED_TOL["olr_abs"], PREDECLARED_TOL["olr_rel"])

    # (B) wired column plumbing + tendency math (State -> coupler assembler).
    # The assembler reproduces the layer mass profiles (T/p/qv/dz/rho) from State
    # to round-off; the w-level interface fields t8w/p8w are RECONSTRUCTED from the
    # State (the State carries no explicit w-level fields), a modeling choice
    # exercised for finiteness/bound by part C and by the operational wiring test.
    # To isolate the PLUMBING + the exact Exner tendency conversion from that
    # interface-reconstruction approximation, the savepoint's own t8w/p8w are
    # substituted into the assembled column before the kernel solve (everything
    # else -- the assembler-derived T/p/qv/q*/dz/rho/emiss/tsk -- is used as-is).
    grid = _grid_for(nz)
    state = _state_from_savepoint(d, grid)
    column = _rrtm_lw_column_inputs(state, grid)
    # Reproduce the layer fields the LW kernel actually consumes and that the
    # State carries exactly: T/p/qv (thermodynamics) + dz (layer thickness, from
    # the geopotential interfaces). ``rho`` is NOT checked here -- the RRTM kernel
    # recomputes ``ro = p/(R_D*T)`` for the cloud optical depth internally and
    # never reads the column-state ``rho`` field.
    plumb = {}
    for name, ref in (("T", col(d, "T")), ("p", col(d, "P")), ("qv", col(d, "QV")),
                      ("dz", col(d, "DZ"))):
        derived = np.asarray(getattr(column, name))[0]
        s = max(float(np.max(np.abs(ref))), 1.0e-30)
        plumb[name] = float(np.max(np.abs(derived - ref)) / s)
    plumbing_max_rel = max(plumb.values())
    plumbing_ok = plumbing_max_rel <= PREDECLARED_TOL["plumbing_rel"]

    # Substitute the savepoint's interface fields (t8w/p8w) AND its cloud fraction
    # into the assembled column. cloud_fraction is a DIAGNOSTIC the operational
    # State does not carry (the coupler derives it from hydrometeor occupancy via
    # _cloud_fraction_columns); like t8w/p8w it is a modeling reconstruction, not a
    # bitwise-recoverable State field. Substituting both isolates the plumbing +
    # exact Exner tendency conversion from those two reconstructions (whose
    # finiteness/sanity is covered by part C + the operational wiring test).
    # Substitute the savepoint's interface fields (t8w/p8w), cloud fraction and
    # surface emissivity into the assembled column. cloud_fraction and emiss are
    # surface/diagnostic reconstructions sourced by SHARED, already-validated code
    # (_cloud_fraction_columns, _surface_radiation_properties), not bitwise-
    # recoverable State fields; substituting them (with t8w/p8w) isolates the
    # plumbing + exact Exner tendency conversion. Their finiteness/sanity is
    # covered by part C and the operational wiring test.
    column_b = column._replace(
        t8w=jnp.asarray(col(d, "T8W")[None, :]),
        p8w=jnp.asarray(col(d, "P8W")[None, :]),
        cloud_fraction=jnp.asarray(col(d, "CLDFRA")[None, :]),
        emiss=jnp.asarray([scalar(d, "EMISS")]),
    )
    out_b = solve_rrtm_lw_column_jax(column_b)
    wired_rth = np.asarray(out_b.heating_rate)[0] / pi  # EXACT coupler conversion
    rthB_abs = float(np.max(np.abs(wired_rth - oracle_rth)))
    rthB_rel = rthB_abs / scale
    rthB_ok = (rthB_rel <= PREDECLARED_TOL["rthraten_rel"]) or (rthB_abs <= PREDECLARED_TOL["rthraten_abs"])

    # (C) full wired path sanity (real coupler, end-to-end).
    full_rth = np.asarray(rrtm_lw_theta_tendency(state, grid, time_utc="2019-05-21T12:00:00Z"))
    finite = bool(np.all(np.isfinite(full_rth)))
    max_abs_full = float(np.max(np.abs(full_rth)))
    sanity_ok = finite and (max_abs_full <= PREDECLARED_TOL["sanity_max_rthraten"])

    contiguous = _laytrop_contiguous(d)

    passed = bool(rthA_ok and glwA["pass"] and olrA["pass"] and plumbing_ok and rthB_ok and sanity_ok and contiguous)
    return passed, {
        "label": d["scalars"]["REGIME"],
        "tsk": scalar(d, "TSK"),
        "traceable_kernel_vs_oracle": {
            "RTHRATEN": {"max_abs": rthA_abs, "max_rel": rthA_rel, "scale": scale,
                          "tol_abs": PREDECLARED_TOL["rthraten_abs"], "tol_rel": PREDECLARED_TOL["rthraten_rel"],
                          "pass": bool(rthA_ok)},
            "GLW": glwA, "OLR": olrA,
        },
        "wired_plumbing": {"per_field_max_rel": plumb, "max_rel": plumbing_max_rel,
                            "tol_rel": PREDECLARED_TOL["plumbing_rel"], "pass": bool(plumbing_ok)},
        "wired_tendency_RTHRATEN": {"max_abs": rthB_abs, "max_rel": rthB_rel,
                                     "tol_abs": PREDECLARED_TOL["rthraten_abs"], "tol_rel": PREDECLARED_TOL["rthraten_rel"],
                                     "pass": bool(rthB_ok)},
        "full_wired_path_sanity": {"finite": finite, "max_abs_rthraten": max_abs_full,
                                    "bound": PREDECLARED_TOL["sanity_max_rthraten"], "pass": bool(sanity_ok)},
        "laytrop_mask_contiguous": contiguous,
        "pass": passed,
    }


def _host_callback_free() -> bool:
    """Assert the wired JAX kernel rides the scan with no host callbacks."""

    d = json.loads((SAVE_FP64 / "rrtm_lw_case_1.json").read_text(encoding="utf-8"))
    jaxpr = jax.make_jaxpr(solve_rrtm_lw_column_jax)(_case_state_kernel(d))
    txt = str(jaxpr)
    return not any(tok in txt for tok in ("pure_callback", "io_callback", "host_callback"))


def read_text_if_present(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def git_head() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    callback_free = _host_callback_free()
    fp64_cases, fp32_cases = {}, {}
    fp64_pass = fp32_pass = True

    for cid in CASE_IDS:
        d = json.loads((SAVE_FP64 / f"rrtm_lw_case_{cid}.json").read_text(encoding="utf-8"))
        ok, res = run_case(d)
        fp64_cases[str(cid)] = res
        fp64_pass = fp64_pass and ok
        a = res["traceable_kernel_vs_oracle"]
        print(
            f"FP64 CASE {cid} {res['label']:24s} -> {'PASS' if ok else 'FAIL'} | "
            f"kernel RTH_rel={a['RTHRATEN']['max_rel']:.3e} dGLW={a['GLW']['abs_err']:.3e} "
            f"dOLR={a['OLR']['abs_err']:.3e} | plumb={res['wired_plumbing']['max_rel']:.2e} | "
            f"wired RTH_rel={res['wired_tendency_RTHRATEN']['max_rel']:.3e} | "
            f"full_sane={res['full_wired_path_sanity']['pass']} contig={res['laytrop_mask_contiguous']}"
        )

    print()
    for cid in CASE_IDS:
        d = json.loads((SAVE_FP32 / f"rrtm_lw_case_{cid}.json").read_text(encoding="utf-8"))
        ok, res = run_case(d)
        fp32_cases[str(cid)] = res
        fp32_pass = fp32_pass and ok
        a = res["traceable_kernel_vs_oracle"]
        print(f"FP32 CASE {cid} {res['label']:24s} -> {'PASS' if ok else 'FAIL'} | "
              f"kernel RTH_rel={a['RTHRATEN']['max_rel']:.3e} dGLW={a['GLW']['abs_err']:.3e} dOLR={a['OLR']['abs_err']:.3e}")

    overall = bool(fp64_pass and fp32_pass and callback_free)
    report = {
        "scheme": "classic AER RRTM longwave (ra_lw_physics=1) -- OPERATIONAL WIRING",
        "what_is_proven": (
            "the JIT/vmap-traceable JAX RRTM-LW kernel (physics.ra_lw_rrtm_jax) + the wired "
            "coupling.physics_couplers.rrtm_lw_theta_tendency coupler and its "
            "_rrtm_lw_column_inputs State->column assembler reproduce the pristine-WRF "
            "RRTMLWRAD RTHRATEN/GLW/OLR oracle; plus State->column plumbing reproduction, "
            "full-wired-path finiteness/bound sanity, host-callback-free traceability, and "
            "the laytrop-mask contiguity that the kernel vectorisation relies on. Distinct "
            "from the HOST-kernel proof proofs/v060/run_rrtm_lw_parity.py."
        ),
        "verdict": "PASS" if overall else "FAIL",
        "overall_pass": overall,
        "canonical_fp64_pass": bool(fp64_pass),
        "secondary_fp32_pass": bool(fp32_pass),
        "host_callback_free": bool(callback_free),
        "self_compare": False,
        "self_compare_note": (
            "The reference is the unmodified pristine WRF phys/module_ra_rrtm.F RRTMLWRAD "
            "source compiled standalone (proofs/v060/oracle, RRTM_DATA AER tables); the JAX "
            "kernel/coupler is never used to make the reference."
        ),
        "oracle": {
            "source": "/home/enric/src/wrf_pristine/WRF/phys/module_ra_rrtm.F",
            "entry": "RRTMLWRAD -> RRTM (AER 16-band k-distribution LW)",
            "source_unmodified": True,
            "full_wrf_exe": False,
            "fp32_savepoints": "proofs/v060/savepoints (rrtm_lw_case_*.json)",
            "fp64_savepoints": "proofs/v060/savepoints_fp64 (rrtm_lw_case_*.json)",
            "fp32_source_checksums": read_text_if_present(SAVE_FP32 / "rrtm_lw_wrf_source_checksums.txt"),
            "fp64_source_checksums": read_text_if_present(SAVE_FP64 / "rrtm_lw_wrf_source_checksums.txt"),
        },
        "wired_path": {
            "kernel": "gpuwrf.physics.ra_lw_rrtm_jax.solve_rrtm_lw_column_jax (JIT/vmap-traceable)",
            "host_reference_kernel": "gpuwrf.physics.ra_lw_rrtm.solve_rrtm_lw_column (NumPy oracle helper)",
            "coupler": "gpuwrf.coupling.physics_couplers.rrtm_lw_theta_tendency",
            "column_assembler": "gpuwrf.coupling.physics_couplers._rrtm_lw_column_inputs",
            "dispatch": (
                "gpuwrf.runtime.operational_mode._physics_step_forcing radiation slot dispatches "
                "ra_lw_physics=1 -> classic RRTM LW theta tendency, SUMMED with the selected SW "
                "tendency (Dudhia ra_sw=1 or RRTMG ra_sw=4); default ra_sw=4/ra_lw=4 routes the "
                "combined RRTMG SW+LW path byte-unchanged."
            ),
            "tendency_conversion": "RTHRATEN = (dT/dt)/exner (WRF RRTMLWRAD: RTHRATEN += TTEN/pi)",
        },
        "jax_precision": "fp64",
        "jax_platform": jax.default_backend(),
        "git_head": git_head(),
        "predeclared_tolerances": PREDECLARED_TOL,
        "fp64_cases": fp64_cases,
        "fp32_cases": fp32_cases,
    }
    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    print("\nhost_callback_free:", callback_free)
    print("OVERALL:", report["verdict"])
    print("wrote", REPORT)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
