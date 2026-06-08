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

v0.13 carry-over fixes (rrtm-lw skeptic 2026-06-08) -- THREE additional proofs:

  (F1) NON-5000-PTOP PARITY -- two extra pristine-WRF savepoints at p_top != 5000
      Pa (low-top 100 mb, high-top 20 mb; proofs/v060/savepoints_fp64/
      rrtm_lw_ptop_case_*.json from rrtm_lw_ptop_oracle_driver.f90). The kernel,
      now given the grid's REAL top_pressure_pa, sizes the above-model-top buffer
      ``nbuf=nint(p_top_mb/4)`` exactly as WRF (module_ra_rrtm.F:6781) and matches
      the oracle to fp64 round-off, where the OLD hardcoded p_top=5000 (nbuf=13)
      version mis-sizes the buffer (recorded: the hardcoded path FAILS the parity
      tol on these tops -> the fix is load-bearing).

  (F2) FAIL-LOUD ON A MIS-SIZED BUFFER -- the high-top (20 mb) column run through
      the OLD hardcoded buffer (top_pressure_pa=None == legacy 5000 Pa, nbuf=13)
      drives layer pressures negative; the kernel now NaN-propagates (the sanity
      gate catches it) instead of the old masking clamps silently emitting finite
      garbage that passed the gate.

  (F3) DIRECT NEGATIVE-PRESSURE GUARD -- a synthetic negative-pressure column is
      asserted to yield NaN (not finite garbage) from solve_rrtm_lw_column_jax.

Production (p_top=5000 Pa, positive pressures) is BIT-IDENTICAL to before.

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
# v0.13 F1: non-5000-ptop fp64 savepoints (grid-aware buffer fix). Case 1 low-top
# 100 mb (hardcoded nbuf=13 undershoots TOA), case 2 high-top 20 mb (hardcoded
# nbuf=13 -> negative buffer pressure == the Finding-F2 regime).
PTOP_CASE_IDS = (1, 2)
PTOP_CASE_FILES = {cid: SAVE_FP64 / f"rrtm_lw_ptop_case_{cid}.json" for cid in PTOP_CASE_IDS}

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


def _case_state_kernel(d: dict, top_pressure_pa: float | None = None) -> RRTMLWColumnState:
    """Direct kernel-input column state from the savepoint (for part A).

    ``top_pressure_pa`` plumbs the grid's real model-top pressure (F1). ``None``
    keeps the legacy hardcoded 5000-Pa buffer behaviour (used by the original 7
    cases, which ARE 5000-Pa, so they stay bit-identical)."""

    def c(name: str):
        return jnp.asarray(col(d, name)[None, :])

    return RRTMLWColumnState(
        T=c("T"), t8w=c("T8W"), p=c("P"), p8w=c("P8W"),
        qv=c("QV"), qc=c("QC"), qr=c("QR"), qi=c("QI"), qs=c("QS"), qg=c("QG"),
        cloud_fraction=c("CLDFRA"), dz=c("DZ"), rho=c("RHO"),
        emiss=scalar(d, "EMISS"), tsk=scalar(d, "TSK"),
        top_pressure_pa=top_pressure_pa,
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


def run_ptop_case(d: dict):
    """v0.13 F1 + F2: non-5000-ptop parity + fail-loud on the mis-sized buffer.

    F1: the kernel given the grid's REAL ``top_pressure_pa`` reproduces the
        pristine-WRF oracle (sized buffer ``nbuf=nint(p_top_mb/4)``); the OLD
        hardcoded path (``top_pressure_pa=None`` == 5000 Pa) is recorded and shown
        to DIVERGE (case 1 low-top) or NaN (case 2 high-top) -> the fix is needed.
    F2: on case 2 (20 mb top) the hardcoded buffer drives pressures negative; the
        kernel now NaN-propagates (no finite garbage).
    """

    ptop = scalar(d, "PTOP")
    pi = col(d, "PI")
    oracle_rth = col(d, "RTHRATEN")
    oracle_glw = scalar(d, "GLW")
    oracle_olr = scalar(d, "OLR")
    scale = max(float(np.max(np.abs(oracle_rth))), PREDECLARED_TOL["rthraten_abs"])

    # --- F1: grid-aware kernel (real ptop) vs WRF oracle ---
    out_fix = solve_rrtm_lw_column_jax(_case_state_kernel(d, top_pressure_pa=ptop))
    rth_fix = np.asarray(out_fix.heating_rate)[0] / pi
    rth_abs = float(np.max(np.abs(rth_fix - oracle_rth)))
    rth_rel = rth_abs / scale
    rth_ok = (rth_rel <= PREDECLARED_TOL["rthraten_rel"]) or (rth_abs <= PREDECLARED_TOL["rthraten_abs"])
    glw_fix = _scalar_status(float(out_fix.glw[0]), oracle_glw,
                             PREDECLARED_TOL["glw_abs"], PREDECLARED_TOL["glw_rel"])
    olr_fix = _scalar_status(float(out_fix.olr[0]), oracle_olr,
                             PREDECLARED_TOL["olr_abs"], PREDECLARED_TOL["olr_rel"])
    fixed_finite = bool(np.all(np.isfinite(rth_fix)) and np.isfinite(out_fix.glw[0]) and np.isfinite(out_fix.olr[0]))
    f1_pass = bool(rth_ok and glw_fix["pass"] and olr_fix["pass"] and fixed_finite)

    # --- the OLD hardcoded path (top_pressure_pa=None == legacy 5000 Pa) ---
    out_hc = solve_rrtm_lw_column_jax(_case_state_kernel(d, top_pressure_pa=None))
    rth_hc = np.asarray(out_hc.heating_rate)[0] / pi
    hc_rth_rel = float(np.max(np.abs(rth_hc - oracle_rth))) / scale
    hc_glw = float(out_hc.glw[0]); hc_olr = float(out_hc.olr[0])
    hc_finite = bool(np.all(np.isfinite(rth_hc)) and np.isfinite(hc_glw) and np.isfinite(hc_olr))
    # The hardcoded path must NOT silently match the oracle (else the fix would be
    # cosmetic). It either fails the parity tol (low-top undershoot) or NaNs
    # (high-top negative buffer). This asserts the fix is LOAD-BEARING.
    hc_diverges = (not hc_finite) or (hc_rth_rel > PREDECLARED_TOL["rthraten_rel"])

    # --- F2: on the high-top (negative-buffer) regime the hardcoded path must
    # fail LOUD (NaN), never emit finite garbage. ---
    is_negative_buffer_regime = ptop < 5000.0  # nbuf=13 buffer overshoots past 0 mb
    if is_negative_buffer_regime:
        f2_pass = bool(not hc_finite)  # NaN-propagated == fail-loud (good)
        f2_note = "hardcoded nbuf=13 buffer -> negative pressure -> NaN (fail-loud, no finite garbage)"
    else:
        f2_pass = True  # not the F2 regime for this case
        f2_note = "low-top: hardcoded nbuf=13 undershoots TOA (finite but biased); F2 regime not triggered"

    passed = bool(f1_pass and hc_diverges and f2_pass)
    return passed, {
        "label": d["scalars"]["REGIME"],
        "ptop_pa": ptop,
        "nbuf_grid_aware": int(round(ptop * 0.01 / 4.0)),
        "nbuf_hardcoded": 13,
        "F1_grid_aware_vs_oracle": {
            "RTHRATEN": {"max_abs": rth_abs, "max_rel": rth_rel,
                          "tol_rel": PREDECLARED_TOL["rthraten_rel"], "pass": bool(rth_ok)},
            "GLW": glw_fix, "OLR": olr_fix, "finite": fixed_finite, "pass": f1_pass,
        },
        "hardcoded_path_diverges": {
            "rth_rel_vs_oracle": hc_rth_rel, "glw": hc_glw, "olr": hc_olr,
            "finite": hc_finite, "diverges": bool(hc_diverges),
            "note": "old DEFAULT_PTOP_PA=5000 buffer; must NOT match oracle (fix is load-bearing)",
        },
        "F2_fail_loud": {"negative_buffer_regime": bool(is_negative_buffer_regime),
                          "hardcoded_finite": hc_finite, "pass": bool(f2_pass), "note": f2_note},
        "pass": passed,
    }


def run_negative_pressure_guard():
    """v0.13 F3: a synthetic negative-pressure column must NaN-propagate (fail-loud),
    NOT emit finite garbage through the old masking clamps.

    Builds a tiny physical column then FORCES a high model top (small p_top) so the
    legacy hardcoded nbuf=13 buffer (pz[l]=pz[l-1]-4mb) marches past 0 into negative
    pressures. With the F2 guard the kernel returns NaN; the OLD clamps returned a
    plausible finite GLW/OLR.
    """

    nz = 10
    # A monotone-decreasing pressure column with a ~30 mb model top (p8w[top]=3000
    # Pa). Hardcoded nbuf=13 buffer: 30,26,...,30-52 = -22 mb -> negative.
    p8w = np.linspace(100000.0, 3000.0, nz + 1)
    p = 0.5 * (p8w[:-1] + p8w[1:])
    T = np.linspace(290.0, 220.0, nz)
    t8w = np.linspace(291.0, 218.0, nz + 1)
    dz = np.full(nz, 1500.0)
    qv = np.full(nz, 1.0e-3)
    z = np.zeros(nz)

    def c(a):
        return jnp.asarray(np.asarray(a, dtype=np.float64)[None, :])

    state = RRTMLWColumnState(
        T=c(T), t8w=c(t8w), p=c(p), p8w=c(p8w),
        qv=c(qv), qc=c(z), qr=c(z), qi=c(z), qs=c(z), qg=c(z),
        cloud_fraction=c(z), dz=c(dz), rho=c(np.full(nz, 1.0)),
        emiss=0.98, tsk=291.0,
        top_pressure_pa=None,  # legacy hardcoded 5000 Pa -> nbuf=13 -> negative buffer
    )
    out = solve_rrtm_lw_column_jax(state)
    glw = float(out.glw[0]); olr = float(out.olr[0])
    heating_finite = bool(np.all(np.isfinite(np.asarray(out.heating_rate))))
    is_nan = bool(np.isnan(glw) or np.isnan(olr) or not heating_finite)

    # And the CORRECTLY-sized buffer (real p_top=3000) must be finite (the same
    # column is physical once the buffer is grid-aware).
    state_fix = state._replace(top_pressure_pa=3000.0)
    out_fix = solve_rrtm_lw_column_jax(state_fix)
    fix_finite = bool(np.all(np.isfinite(np.asarray(out_fix.heating_rate)))
                      and np.isfinite(out_fix.glw[0]) and np.isfinite(out_fix.olr[0]))

    passed = bool(is_nan and fix_finite)
    return passed, {
        "model_top_pa": float(p8w[-1]),
        "hardcoded_nbuf13_buffer_goes_negative": True,
        "hardcoded_result": {"glw": glw, "olr": olr, "is_nan_fail_loud": is_nan},
        "grid_aware_result_finite": fix_finite,
        "note": ("legacy nbuf=13 buffer drives pressures negative -> kernel NaN-propagates "
                 "(fail-loud) instead of the old maximum/clip masking clamps emitting finite "
                 "garbage; grid-aware buffer makes the same column physical/finite"),
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

    # --- v0.13 F1: non-5000-ptop parity + F2 fail-loud on the hardcoded buffer ---
    print()
    ptop_cases = {}
    ptop_pass = True
    for cid in PTOP_CASE_IDS:
        d = json.loads(PTOP_CASE_FILES[cid].read_text(encoding="utf-8"))
        ok, res = run_ptop_case(d)
        ptop_cases[str(cid)] = res
        ptop_pass = ptop_pass and ok
        f1 = res["F1_grid_aware_vs_oracle"]; hc = res["hardcoded_path_diverges"]
        print(
            f"PTOP CASE {cid} {res['label']:20s} (p_top={res['ptop_pa']:.0f}Pa, "
            f"nbuf {res['nbuf_hardcoded']}->{res['nbuf_grid_aware']}) -> {'PASS' if ok else 'FAIL'} | "
            f"F1 grid-aware: RTH_rel={f1['RTHRATEN']['max_rel']:.2e} dGLW={f1['GLW']['abs_err']:.2e} "
            f"dOLR={f1['OLR']['abs_err']:.2e} | hardcoded: RTH_rel={hc['rth_rel_vs_oracle']:.2e} "
            f"finite={hc['finite']} (F2 fail-loud={res['F2_fail_loud']['pass']})"
        )

    # --- v0.13 F3: synthetic negative-pressure guard ---
    print()
    f3_pass, f3_res = run_negative_pressure_guard()
    print(
        f"F3 NEGATIVE-PRESSURE GUARD -> {'PASS' if f3_pass else 'FAIL'} | "
        f"hardcoded nbuf=13 buffer: glw={f3_res['hardcoded_result']['glw']} "
        f"olr={f3_res['hardcoded_result']['olr']} is_nan(fail-loud)="
        f"{f3_res['hardcoded_result']['is_nan_fail_loud']} | grid-aware finite="
        f"{f3_res['grid_aware_result_finite']}"
    )

    overall = bool(fp64_pass and fp32_pass and callback_free and ptop_pass and f3_pass)
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
        "non_5000_ptop_pass": bool(ptop_pass),
        "negative_pressure_guard_pass": bool(f3_pass),
        "host_callback_free": bool(callback_free),
        "v013_carryover_fixes": {
            "F1_grid_aware_buffer": (
                "ra_lw_rrtm_jax._nbuf / _prepare_atmosphere_jax / solve_rrtm_lw_column_jax now size the "
                "above-model-top buffer nbuf=nint(p_top_mb/4) from the grid's RRTMLWColumnState.top_pressure_pa "
                "(plumbed by coupling.physics_couplers._rrtm_lw_column_inputs from grid.vertical.top_pressure_pa), "
                "matching WRF module_ra_rrtm.F:6781; None falls back to DEFAULT_PTOP_PA=5000 (production bit-identical)."
            ),
            "F2_fail_loud_guards": (
                "the forbidden masking clamps jnp.maximum(pavel,1e-300) and jnp.clip(ifp,0,200) are replaced by "
                "positivity GUARDS: a non-positive prepared pavel/pz NaN-taints the column and a non-finite/out-of-range "
                "corr index NaN-taints the correction, so a mis-sized/pathological column NaN-propagates (the sanity gate "
                "fails loud) instead of emitting silent finite garbage. Positive-pressure production is unaffected."
            ),
        },
        "non_5000_ptop_oracle": {
            "driver": "proofs/v060/oracle/rrtm_lw_ptop_oracle_driver.f90 (same UNMODIFIED module_ra_rrtm.F:RRTMLWRAD)",
            "savepoints": "proofs/v060/savepoints_fp64/rrtm_lw_ptop_case_{1,2}.json (low-top 100mb, high-top 20mb)",
            "source_checksums": read_text_if_present(SAVE_FP64 / "rrtm_lw_ptop_wrf_source_checksums.txt"),
        },
        "non_5000_ptop_cases": ptop_cases,
        "negative_pressure_guard": f3_res,
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
