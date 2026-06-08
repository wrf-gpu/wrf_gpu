"""v0.13 CONSOLIDATED OPERATIONAL FUNCTIONAL SMOKE.

Goal (v0.13 "completely functional" gate): every physics option this port
advertises as OPERATIONAL -- i.e. one whose State adapter is threaded into the
operational scan (``runtime.operational_mode._SCAN_WIRED_OPTIONS`` +
``coupling.scan_adapters`` + ``coupling.physics_dispatch``) -- must ACTUALLY RUN
in the integration and produce finite, physical output, AND must measurably
MUTATE its expected fields (it truly ran, not a silent no-op).

POSITIONING: OPERATIONAL = wired into the scan and runs end-to-end here.
REFERENCE-ONLY (oracle infra only, fail-closed in the scan) is intentionally NOT
covered -- those are validated by their per-scheme oracle proofs, not by this
integration gate. This module is the self-documenting cross-check that the
OPERATIONAL surface is genuinely functional.

AUTHORITATIVE ENTRYPOINT
------------------------
Microphysics / surface-layer / PBL / cumulus / radiation are exercised through
``operational_mode._physics_step_forcing`` -- the EXACT per-step physics block the
operational scan body runs (dispatcher-selected mp -> surface-layer/land -> PBL ->
GWDO -> cumulus -> SW/LW radiation, in WRF physics-driver call order), gated first
by the real fail-closed authority ``_resolve_operational_suite``. Land-surface
(Noah-MP / Noah-classic) is exercised through the EXACT coupler steps the scan
calls (``noahmp_surface_step`` / ``noahclassic_surface_step``).

DISCIPLINE
----------
* NO masking / clamps / self-compare / synthetic happy-path. A scheme that breaks
  or is inert is reported as a real "not fully functional" finding (xfail with a
  precise reason), never silently skipped.
* CPU-only (the operational physics block is JIT-traceable on CPU == lowerable on
  the GPU scan). Tiny idealized columns keep the whole module fast (~<2 min on
  4 CPU cores).

Run (CPU, cores 28-31, NO GPU):
  JAX_PLATFORMS=cpu PYTHONPATH=src TF_CPP_MIN_LOG_LEVEL=3 taskset -c 28-31 \
      python -m pytest tests/test_v013_operational_smoke.py -q
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.contracts.grid import (
    BCMetadata,
    DycoreMetrics,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.physics_registry import (
    ACCEPTED_BL_PBL_PHYSICS,
    ACCEPTED_CU_PHYSICS,
    ACCEPTED_MP_PHYSICS,
    ACCEPTED_RA_LW_PHYSICS,
    ACCEPTED_RA_SW_PHYSICS,
    ACCEPTED_SF_SFCLAY_PHYSICS,
)
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.coupling.scan_adapters import (
    CU_SCAN_ADAPTERS,
    CU_STATELESS_SCAN_ADAPTERS,
    MP_SCAN_ADAPTERS,
    PBL_SCAN_ADAPTERS,
    SFCLAY_SCAN_ADAPTERS,
)
from gpuwrf.runtime.operational_mode import (
    DEFAULT_BL_PBL_PHYSICS,
    DEFAULT_MP_PHYSICS,
    OperationalNamelist,
    UnsupportedSchemeSelection,
    _SCAN_WIRED_OPTIONS,
    _physics_step_forcing,
    _resolve_operational_suite,
)
from gpuwrf.runtime.operational_state import initial_operational_carry

ROOT = Path(__file__).resolve().parents[1]
TIME_UTC = "2019-05-21T12:00:00Z"  # local noon over the Canary projection -> SW active

P0_PA, R_D, C_P, GRAVITY = 1.0e5, 287.0, 1004.0, 9.80665


# ============================================================================
# Authoritative OPERATIONAL set per category (derived from the scan-wiring; the
# disabled option 0 is excluded, it means "no scheme runs in that slot").
# sf_surface is special: it is wired via the use_noahmp toggle (Noah-MP=4) and
# the explicit Noah-classic (=2) hook, NOT via _SCAN_WIRED_OPTIONS, so it is
# enumerated explicitly below.
# ============================================================================
OPERATIONAL_MP = tuple(o for o in _SCAN_WIRED_OPTIONS["mp_physics"] if o != 0)
OPERATIONAL_BL = tuple(o for o in _SCAN_WIRED_OPTIONS["bl_pbl_physics"] if o != 0)
OPERATIONAL_SF_SFCLAY = tuple(o for o in _SCAN_WIRED_OPTIONS["sf_sfclay_physics"] if o != 0)
OPERATIONAL_CU = tuple(o for o in _SCAN_WIRED_OPTIONS["cu_physics"] if o != 0)
OPERATIONAL_RA_SW = tuple(o for o in _SCAN_WIRED_OPTIONS["ra_sw_physics"] if o != 0)
OPERATIONAL_RA_LW = tuple(o for o in _SCAN_WIRED_OPTIONS["ra_lw_physics"] if o != 0)
OPERATIONAL_SF_SURFACE = (2, 4)  # Noah-classic (explicit hook) + Noah-MP (use_noahmp)


# ============================================================================
# Minimal idealized grid + state builders (the validated b2 C-grid pattern: a
# deep, moist, windy column that gives every scheme something to act on).
# ============================================================================
def _grid(nz: int = 24, ny: int = 4, nx: int = 4) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, nx, ny)
    terrain_meta = TerrainProvenance(
        source_path="v013-op-smoke", sha256="v013-op-smoke", shape=(ny, nx), units="m",
        projection_transform="native-wrf-lambert", max_elevation_m=0.0,
        coastline_sanity_check_passed=True,
    )
    vertical = VerticalCoord("hybrid_eta", nz, 5000.0, eta)
    bc = BCMetadata("ideal", (), 1, "linear", True)
    metrics = DycoreMetrics.flat(
        ny=ny, nx=nx, nz=nz, eta_levels=eta, top_pressure_pa=5000.0, provenance="v013-op-smoke-flat",
    )
    return GridSpec(projection, terrain_meta, vertical, bc, eta, jnp.zeros((ny, nx)), metrics=metrics)


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    """CPU-allocated zero tendencies (bypasses the GPU-only ``Tendencies.zeros``)."""

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    z = lambda shape: jnp.zeros(shape, dtype=jnp.float64)  # noqa: E731
    return Tendencies(
        z((nz, ny, nx + 1)), z((nz, ny + 1, nx)), z((nz + 1, ny, nx)),
        z((nz, ny, nx)), z((nz, ny, nx)), z((nz, ny, nx)), z((nz + 1, ny, nx)), z((ny, nx)),
    )


def _base_state(grid: GridSpec, *, dz_m: float = 300.0, seed: int = 3) -> State:
    """Deep, moist, windy stratiform-ish column (b2 pattern, deterministic)."""

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    rng = np.random.default_rng(seed)
    fields = {n: jnp.zeros(s, dtype=jnp.float64) for n, s in _state_field_shapes(grid).items()}
    z_iface = np.arange(nz + 1) * dz_m
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    theta_col = 300.0 + 0.004 * z_mid
    p_col = P0_PA * (1.0 - GRAVITY * z_mid / (C_P * 290.0)) ** (C_P / R_D)

    def m3(base, noise):
        return jnp.asarray(base[:, None, None] + noise * rng.standard_normal((nz, ny, nx)), dtype=jnp.float64)

    fields["theta"] = m3(theta_col, 0.3)
    fields["p"] = m3(p_col, 50.0)
    fields["p_total"] = fields["p"]
    fields["qv"] = jnp.clip(m3(0.012 * np.exp(-z_mid / 3000.0), 5.0e-4), 0.0, None)
    fields["qc"] = jnp.clip(m3(np.where(z_mid < 4000.0, 4.0e-4, 0.0), 2.0e-5), 0.0, None)
    fields["qr"] = jnp.clip(m3(np.where((z_mid > 500.0) & (z_mid < 3000.0), 1.0e-4, 0.0), 1.0e-5), 0.0, None)
    fields["qi"] = jnp.clip(m3(np.where(z_mid > 6000.0, 5.0e-5, 0.0), 1.0e-6), 0.0, None)
    fields["qs"] = jnp.clip(m3(np.where(z_mid > 5000.0, 3.0e-5, 0.0), 1.0e-6), 0.0, None)
    fields["Ni"] = jnp.clip(m3(np.where(z_mid > 6000.0, 5.0e3, 0.0), 1.0e2), 0.0, None)
    fields["Nr"] = jnp.clip(m3(np.where((z_mid > 500.0) & (z_mid < 3000.0), 1.0e4, 0.0), 1.0e2), 0.0, None)
    fields["Ns"] = jnp.clip(m3(np.where(z_mid > 5000.0, 5.0e3, 0.0), 1.0e2), 0.0, None)
    fields["Nc"] = jnp.clip(m3(np.where(z_mid < 4000.0, 1.0e8, 0.0), 1.0e6), 0.0, None)
    fields["Nn"] = jnp.clip(m3(np.full(nz, 1.0e8), 1.0e6), 0.0, None)
    fields["u"] = jnp.asarray(6.0 + 0.5 * rng.standard_normal((nz, ny, nx + 1)), dtype=jnp.float64)
    fields["v"] = jnp.asarray(-2.0 + 0.5 * rng.standard_normal((nz, ny + 1, nx)), dtype=jnp.float64)
    fields["w"] = jnp.asarray(0.05 * rng.standard_normal((nz + 1, ny, nx)), dtype=jnp.float64)
    fields["qke"] = jnp.full((nz, ny, nx), 0.4, dtype=jnp.float64)
    ph = jnp.asarray(np.broadcast_to(GRAVITY * z_iface[:, None, None], (nz + 1, ny, nx)), dtype=jnp.float64)
    fields["ph"] = ph
    fields["ph_total"] = ph
    xland = np.ones((ny, nx))
    xland[:, nx // 2:] = 2.0
    fields["xland"] = jnp.asarray(xland, dtype=jnp.float64)
    fields["t_skin"] = jnp.asarray(np.where(xland > 1.5, 299.5, 304.0), dtype=jnp.float64)
    fields["soil_moisture"] = jnp.full((ny, nx), 0.3, dtype=jnp.float64)
    fields["mavail"] = jnp.where(jnp.asarray(xland) > 1.5, 1.0, 0.4).astype(jnp.float64)
    fields["roughness_m"] = jnp.where(jnp.asarray(xland) > 1.5, 2.85e-3, 0.15).astype(jnp.float64)
    fields["ustar"] = jnp.full((ny, nx), 0.3, dtype=jnp.float64)
    fields["rhosfc"] = jnp.full((ny, nx), 1.15, dtype=jnp.float64)
    fields["mu_total"] = jnp.full((ny, nx), 1.0e5, dtype=jnp.float64)
    fields["mu"] = jnp.full((ny, nx), 9.0e4, dtype=jnp.float64)
    fields["lu_index"] = jnp.zeros((ny, nx), dtype=jnp.int32)
    return State(**fields)


def _convective_state(grid: GridSpec, *, dz_m: float = 400.0, seed: int = 5) -> State:
    """Conditionally-unstable, near-saturated, warm-SST column so the deep-cumulus
    closures (KF/BMJ/Grell-Freitas) actually TRIGGER (a stratiform b2 profile does
    not, which would make a cumulus mutation assertion a false negative)."""

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    rng = np.random.default_rng(seed)
    fields = {n: jnp.zeros(s, dtype=jnp.float64) for n, s in _state_field_shapes(grid).items()}
    z_iface = np.arange(nz + 1) * dz_m
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    theta_col = 298.0 + 0.0035 * z_mid
    p_col = P0_PA * (1.0 - GRAVITY * z_mid / (C_P * 295.0)) ** (C_P / R_D)

    def m3(base, noise):
        return jnp.asarray(base[:, None, None] + noise * rng.standard_normal((nz, ny, nx)), dtype=jnp.float64)

    fields["theta"] = m3(theta_col, 0.2)
    fields["p"] = m3(p_col, 30.0)
    fields["p_total"] = fields["p"]
    qv = np.where(z_mid < 2000.0, 0.018, 0.016 * np.exp(-(z_mid - 2000.0) / 3000.0))
    fields["qv"] = jnp.clip(m3(qv, 3.0e-4), 1.0e-6, None)
    fields["qc"] = jnp.clip(m3(np.where(z_mid < 3000.0, 3.0e-4, 0.0), 2.0e-5), 0.0, None)
    fields["u"] = jnp.asarray(4.0 + 0.3 * rng.standard_normal((nz, ny, nx + 1)), dtype=jnp.float64)
    fields["v"] = jnp.asarray(1.0 + 0.3 * rng.standard_normal((nz, ny + 1, nx)), dtype=jnp.float64)
    fields["w"] = jnp.asarray(0.2 + 0.05 * rng.standard_normal((nz + 1, ny, nx)), dtype=jnp.float64)
    fields["qke"] = jnp.full((nz, ny, nx), 0.6, dtype=jnp.float64)
    ph = jnp.asarray(np.broadcast_to(GRAVITY * z_iface[:, None, None], (nz + 1, ny, nx)), dtype=jnp.float64)
    fields["ph"] = ph
    fields["ph_total"] = ph
    fields["xland"] = jnp.ones((ny, nx), dtype=jnp.float64)
    fields["t_skin"] = jnp.full((ny, nx), 303.0, dtype=jnp.float64)
    fields["soil_moisture"] = jnp.full((ny, nx), 0.35, dtype=jnp.float64)
    fields["mavail"] = jnp.full((ny, nx), 0.8, dtype=jnp.float64)
    fields["roughness_m"] = jnp.full((ny, nx), 0.1, dtype=jnp.float64)
    fields["ustar"] = jnp.full((ny, nx), 0.4, dtype=jnp.float64)
    fields["rhosfc"] = jnp.full((ny, nx), 1.15, dtype=jnp.float64)
    fields["mu_total"] = jnp.full((ny, nx), 1.0e5, dtype=jnp.float64)
    fields["mu"] = jnp.full((ny, nx), 9.0e4, dtype=jnp.float64)
    fields["lu_index"] = jnp.zeros((ny, nx), dtype=jnp.int32)
    # qv_flux drives the Tiedtke surface-moisture forcing; seed a realistic value.
    fields["qv_flux"] = jnp.full((ny, nx), 1.0e-4, dtype=jnp.float64)
    return State(**fields)


def _namelist(grid: GridSpec, *, dt_s: float = 20.0, **over) -> OperationalNamelist:
    """Build a CPU operational namelist (from_grid's Tendencies.zeros is GPU-only)."""

    base = OperationalNamelist.from_grid(grid, dt_s=dt_s, tendencies=_cpu_tendencies(grid))
    return dataclasses.replace(
        base, time_utc=TIME_UTC, run_physics=True, radiation_cadence_steps=1, **over
    )


def _maxabs(a) -> float:
    return float(np.max(np.abs(np.asarray(a))))


def _changed(after, before, *, atol: float = 0.0) -> bool:
    return not np.allclose(np.asarray(after), np.asarray(before), atol=atol)


def _all_finite(state: State) -> bool:
    for leaf in jax.tree_util.tree_leaves(state):
        a = np.asarray(leaf)
        if np.issubdtype(a.dtype, np.floating) and not np.all(np.isfinite(a)):
            return False
    return True


# ============================================================================
# 0. The operational set is self-consistent (adapter table == scan-wired set).
# ============================================================================
def test_operational_set_is_consistent_with_adapter_tables() -> None:
    """Every scan-wired non-zero option (except the default-coupler-backed ones)
    has a concrete adapter; conversely every adapter is in the scan-wired set."""

    # microphysics: mp=8 is the default thompson_adapter (not in MP_SCAN_ADAPTERS).
    assert set(MP_SCAN_ADAPTERS) | {DEFAULT_MP_PHYSICS} >= set(OPERATIONAL_MP)
    assert set(MP_SCAN_ADAPTERS) <= set(OPERATIONAL_MP)
    # PBL: bl=5 MYNN + bl=2 MYJ are routed explicitly in the step (not in the table).
    assert set(PBL_SCAN_ADAPTERS) | {DEFAULT_BL_PBL_PHYSICS, 2} >= set(OPERATIONAL_BL)
    assert set(PBL_SCAN_ADAPTERS) <= set(OPERATIONAL_BL)
    # surface layer: sf=5 MYNN-sfclay (default) + sf=2 Janjic are routed explicitly.
    assert set(SFCLAY_SCAN_ADAPTERS) | {5, 2} >= set(OPERATIONAL_SF_SFCLAY)
    assert set(SFCLAY_SCAN_ADAPTERS) <= set(OPERATIONAL_SF_SFCLAY)
    # cumulus: every wired cumulus option has an adapter.
    assert set(CU_SCAN_ADAPTERS) == set(OPERATIONAL_CU)
    assert set(CU_STATELESS_SCAN_ADAPTERS) <= set(CU_SCAN_ADAPTERS)


# ============================================================================
# 1. MICROPHYSICS -- every operational mp option runs in the step and condenses
#    moisture (qv changes). mp neighbours (PBL/cumulus/radiation) disabled so the
#    qv change is attributable to microphysics.
# ============================================================================
# mp=14 (WDM5) was scan-wired (MP_SCAN_ADAPTERS[14], _SCAN_WIRED_OPTIONS) but
# MISSING from the physics-dispatch routable table (coupling.physics_dispatch._MP_ENTRIES),
# so _resolve_operational_suite REJECTED it (advertised-operational but unroutable).
# CLOSED 2026-06-08: added _mp_entry(14, microphysics_wdm5, wdm5_physics_tendency) to
# _MP_ENTRIES. WDM5 now routes + runs functionally, so the gap set is empty.
_MP_DISPATCH_GAP: set[int] = set()


@pytest.mark.parametrize("mp", OPERATIONAL_MP)
def test_microphysics_operational_runs_and_mutates(mp: int) -> None:
    grid = _grid()
    state = _base_state(grid)
    nml = _namelist(grid, mp_physics=mp, bl_pbl_physics=0, sf_sfclay_physics=0, cu_physics=0)

    if mp in _MP_DISPATCH_GAP:
        pytest.xfail(
            f"mp_physics={mp} (WDM5) is scan-wired (MP_SCAN_ADAPTERS[{mp}], "
            "_SCAN_WIRED_OPTIONS) but MISSING from coupling.physics_dispatch._MP_ENTRIES "
            "-> _resolve_operational_suite (the operational fail-closed authority) "
            "REJECTS it, so it is advertised-operational but cannot run. Add the "
            "_mp_entry(14, 'gpuwrf.physics.microphysics_wdm5', 'wdm5_physics_tendency') "
            "dispatch row to close this."
        )

    _resolve_operational_suite(nml)  # operational authority must accept it
    carry = initial_operational_carry(state)
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    after = forcing.state
    assert _all_finite(after), f"mp={mp} produced a non-finite field"
    # microphysics condenses/evaporates -> moisture must change (not a no-op).
    assert _changed(after.qv, state.qv), f"mp={mp} did not mutate qv (silent no-op)"


# ============================================================================
# 2. PBL -- every operational PBL option runs and mixes momentum (u or v change).
#    mp/cumulus/radiation disabled so the u/v change is attributable to the PBL.
#    bl=2 (MYJ) is mandatorily paired with sf=2 (Janjic Eta); covered as a pair.
# ============================================================================
_PBL_SFCLAY_PAIR = {1: 1, 2: 2, 5: 5, 7: 1, 8: 1, 99: 1}
_PBL_TKE_SCHEMES = {2, 5, 8}  # also carry a prognostic TKE (qke) update


@pytest.mark.parametrize("bl", OPERATIONAL_BL)
def test_pbl_operational_runs_and_mixes(bl: int) -> None:
    grid = _grid()
    state = _base_state(grid)
    sf = _PBL_SFCLAY_PAIR[bl]
    nml = _namelist(
        grid, mp_physics=0, bl_pbl_physics=bl, sf_sfclay_physics=sf, cu_physics=0, use_noahmp=False
    )
    _resolve_operational_suite(nml)
    carry = initial_operational_carry(state)
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    after = forcing.state
    assert _all_finite(after), f"bl={bl} produced a non-finite field"
    # PBL mixes momentum: u or v must change (mp is off, so this is the PBL).
    assert _changed(after.u, state.u) or _changed(after.v, state.v), (
        f"bl={bl} did not mutate u/v (silent no-op)"
    )
    if bl in _PBL_TKE_SCHEMES:
        assert _changed(after.qke, state.qke), f"bl={bl} (TKE scheme) did not update qke"


# ============================================================================
# 3. SURFACE LAYER -- every operational sfclay option runs and writes the
#    kinematic surface-flux handles (ustar / theta_flux). mp/PBL/cumulus disabled.
#    sf=2 (Janjic) is mandatorily paired with bl=2 (MYJ); covered as a pair.
# ============================================================================
_SFCLAY_PARAM = tuple(o for o in OPERATIONAL_SF_SFCLAY if o != 2)


@pytest.mark.parametrize("sf", _SFCLAY_PARAM)
def test_sfclay_operational_runs_and_writes_fluxes(sf: int) -> None:
    grid = _grid()
    state = _base_state(grid)
    nml = _namelist(
        grid, mp_physics=0, bl_pbl_physics=0, sf_sfclay_physics=sf, cu_physics=0, use_noahmp=False
    )
    _resolve_operational_suite(nml)
    carry = initial_operational_carry(state)
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    after = forcing.state
    assert _all_finite(after), f"sf={sf} produced a non-finite field"
    assert _changed(after.ustar, state.ustar) or _changed(after.theta_flux, state.theta_flux), (
        f"sf={sf} did not write surface-flux handles (silent no-op)"
    )


def test_myj_pair_operational_runs_and_mutates() -> None:
    """bl=2 (MYJ PBL) + sf=2 (Janjic Eta sfclay) is the mandatory MYJ pair; it
    covers both the bl=2 and sf=2 operational options. The pair must mix momentum,
    update the TKE carry (qke), and write a real ustar."""

    grid = _grid()
    state = _base_state(grid)
    nml = _namelist(
        grid, mp_physics=0, bl_pbl_physics=2, sf_sfclay_physics=2, cu_physics=0, use_noahmp=False
    )
    suite = _resolve_operational_suite(nml)
    assert suite.pbl.option == 2 and suite.surface_layer.option == 2
    carry = initial_operational_carry(state)
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    after = forcing.state
    assert _all_finite(after), "MYJ pair produced a non-finite field"
    assert _changed(after.u, state.u) or _changed(after.v, state.v), "MYJ did not mix momentum"
    assert _changed(after.qke, state.qke), "MYJ did not update the TKE carry (qke)"


@pytest.mark.parametrize("bl,sf", [(2, 5), (5, 2), (2, 1), (1, 2)])
def test_myj_pairing_fails_closed_when_unpaired(bl: int, sf: int) -> None:
    """MYJ(2)/Janjic(2) must be selected together; a half-pair fails closed."""

    grid = _grid()
    nml = _namelist(grid, bl_pbl_physics=bl, sf_sfclay_physics=sf, use_noahmp=False)
    with pytest.raises(UnsupportedSchemeSelection):
        _resolve_operational_suite(nml)


# ============================================================================
# 4. CUMULUS -- the deep-convective closures must TRIGGER on a conditionally
#    unstable column and produce convective precip (rainc_acc > 0). mp/PBL off and
#    a larger dt so the cumulus step accumulates a measurable adjustment.
# ============================================================================
# cu=6 (modified Tiedtke) is scan-wired (CU_SCAN_ADAPTERS[6], runs finite), but its
# operational adapter HARD-ZEROES the QVFTEN/QVPBLTEN large-scale moisture-convergence
# forcing (documented carry-over in scan_adapters.tiedtke_adapter). The Tiedtke kernel
# triggers only when fed nonzero QVFTEN (verified directly), so with the forcing zeroed
# it never activates in the operational scan -> wired-but-INERT. Reported, not masked.
_CU_TRIGGERING = tuple(o for o in OPERATIONAL_CU if o != 6)


@pytest.mark.parametrize("cu", _CU_TRIGGERING)
def test_cumulus_operational_triggers_and_rains(cu: int) -> None:
    grid = _grid()
    state = _convective_state(grid)
    nml = _namelist(grid, dt_s=300.0, mp_physics=0, bl_pbl_physics=0, sf_sfclay_physics=0, cu_physics=cu)
    _resolve_operational_suite(nml)
    carry = initial_operational_carry(state)
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    after = forcing.state
    assert _all_finite(after), f"cu={cu} produced a non-finite field"
    # Deep convection fires on the unstable column: convective precip + heating.
    assert _maxabs(after.rainc_acc) > 0.0, f"cu={cu} did not trigger (no convective precip)"
    assert _changed(after.theta, state.theta), f"cu={cu} did not apply a convective tendency"


def test_cumulus_tiedtke_runs_finite_but_is_inert_carry_over() -> None:
    """cu=6 (Tiedtke) is scan-wired and runs finite, but is INERT in the operational
    scan because the adapter zeroes its QVFTEN moisture-convergence forcing. This is
    a documented carry-over surfaced honestly (xfail), not a silent skip."""

    grid = _grid()
    state = _convective_state(grid)
    nml = _namelist(grid, dt_s=900.0, mp_physics=0, bl_pbl_physics=0, sf_sfclay_physics=0, cu_physics=6)
    _resolve_operational_suite(nml)
    carry = initial_operational_carry(state)
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    after = forcing.state
    # It genuinely RUNS (finite, JIT-traceable) -- this much is functional.
    assert _all_finite(after), "cu=6 produced a non-finite field"
    if _maxabs(after.rainc_acc) <= 0.0 and not _changed(after.theta, state.theta):
        pytest.xfail(
            "cu_physics=6 (modified Tiedtke) is scan-wired (CU_SCAN_ADAPTERS[6]) and runs "
            "finite, but the operational adapter hard-zeroes QVFTEN/QVPBLTEN (the "
            "large-scale moisture-convergence forcing the Tiedtke closure triggers on), so "
            "it is INERT even on a strongly convective column. The kernel itself triggers "
            "when fed nonzero QVFTEN -> the gap is the operational coupling, not the kernel. "
            "Thread a real QVFTEN forcing into tiedtke_adapter to make cu=6 functional."
        )


# ============================================================================
# 5. RADIATION -- every operational SW/LW option runs and produces a finite,
#    NONZERO radiative heating rate (RTHRATEN). mp/PBL/sfclay/cumulus disabled so
#    the held RTHRATEN is purely the selected SW+LW pair.
# ============================================================================
@pytest.mark.parametrize("ra_sw", OPERATIONAL_RA_SW)
def test_ra_sw_operational_runs_and_heats(ra_sw: int) -> None:
    grid = _grid()
    state = _base_state(grid)
    nml = _namelist(
        grid, mp_physics=0, bl_pbl_physics=0, sf_sfclay_physics=0, cu_physics=0, ra_sw_physics=ra_sw
    )
    _resolve_operational_suite(nml)
    carry = initial_operational_carry(state)
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=True)
    rth = np.asarray(forcing.carry.rthraten)
    assert np.all(np.isfinite(rth)), f"ra_sw={ra_sw} produced a non-finite RTHRATEN"
    assert _maxabs(rth) > 0.0, f"ra_sw={ra_sw} produced a zero RTHRATEN (no heating)"


@pytest.mark.parametrize("ra_lw", OPERATIONAL_RA_LW)
def test_ra_lw_operational_runs_and_cools(ra_lw: int) -> None:
    grid = _grid()
    state = _base_state(grid)
    nml = _namelist(
        grid, mp_physics=0, bl_pbl_physics=0, sf_sfclay_physics=0, cu_physics=0, ra_lw_physics=ra_lw
    )
    _resolve_operational_suite(nml)
    carry = initial_operational_carry(state)
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=True)
    rth = np.asarray(forcing.carry.rthraten)
    assert np.all(np.isfinite(rth)), f"ra_lw={ra_lw} produced a non-finite RTHRATEN"
    assert _maxabs(rth) > 0.0, f"ra_lw={ra_lw} produced a zero RTHRATEN (no LW exchange)"


def test_sw_and_lw_are_selected_independently() -> None:
    """The SW and LW selectors compose independently (changing one changes RTHRATEN)."""

    grid = _grid()
    state = _base_state(grid)

    def rth(ra_sw, ra_lw):
        nml = _namelist(
            grid, mp_physics=0, bl_pbl_physics=0, sf_sfclay_physics=0, cu_physics=0,
            ra_sw_physics=ra_sw, ra_lw_physics=ra_lw,
        )
        carry = initial_operational_carry(state)
        return np.asarray(_physics_step_forcing(carry, nml, 0.0, run_radiation=True).carry.rthraten)

    base = rth(4, 4)
    assert not np.allclose(base, rth(1, 4)), "changing ra_sw did not change RTHRATEN"
    assert not np.allclose(base, rth(4, 1)), "changing ra_lw did not change RTHRATEN"


# ============================================================================
# 6. LAND SURFACE -- the two operational LSMs run through the EXACT coupler steps
#    the scan invokes and advance their land carry + write finite surface fluxes.
# ============================================================================
_NOAHMP_TABLE_DIR = Path("/home/enric/src/wrf_pristine/WRF/run")
_HAVE_MPTABLE = (_NOAHMP_TABLE_DIR / "MPTABLE.TBL").exists()
_NOAHCLASSIC_SAVEPOINTS = ROOT / "proofs" / "v060" / "savepoints_noahclassic.json"


def _land_grid(nz: int = 12, ny: int = 2, nx: int = 3) -> GridSpec:
    return _grid(nz=nz, ny=ny, nx=nx)


def _land_state(grid: GridSpec) -> State:
    """All-land column for the LSM coupler steps (xland=1 everywhere)."""

    nz, ny, nx = grid.nz, grid.ny, grid.nx
    fields = {n: jnp.zeros(s, dtype=jnp.float64) for n, s in _state_field_shapes(grid).items()}
    z_iface = np.arange(nz + 1) * 300.0
    z_mid = 0.5 * (z_iface[:-1] + z_iface[1:])
    fields["theta"] = jnp.broadcast_to(jnp.asarray(300.0 + 0.004 * z_mid)[:, None, None], (nz, ny, nx))
    fields["p"] = jnp.broadcast_to(
        jnp.asarray(P0_PA * (1.0 - GRAVITY * z_mid / (C_P * 290.0)) ** (C_P / R_D))[:, None, None], (nz, ny, nx)
    )
    fields["p_total"] = fields["p"]
    fields["qv"] = jnp.full((nz, ny, nx), 8.0e-3, dtype=jnp.float64)
    fields["u"] = jnp.full((nz, ny, nx + 1), 4.0, dtype=jnp.float64)
    fields["v"] = jnp.full((nz, ny + 1, nx), 1.0, dtype=jnp.float64)
    ph = jnp.broadcast_to(jnp.asarray(GRAVITY * z_iface)[:, None, None], (nz + 1, ny, nx))
    fields["ph"] = ph
    fields["ph_total"] = ph
    fields["xland"] = jnp.ones((ny, nx), dtype=jnp.float64)
    fields["t_skin"] = jnp.full((ny, nx), 298.0, dtype=jnp.float64)
    fields["soil_moisture"] = jnp.full((ny, nx), 0.3, dtype=jnp.float64)
    fields["mavail"] = jnp.full((ny, nx), 0.6, dtype=jnp.float64)
    fields["roughness_m"] = jnp.full((ny, nx), 0.08, dtype=jnp.float64)
    fields["ustar"] = jnp.full((ny, nx), 0.3, dtype=jnp.float64)
    fields["rhosfc"] = jnp.full((ny, nx), 1.15, dtype=jnp.float64)
    fields["lu_index"] = jnp.zeros((ny, nx), dtype=jnp.int32)
    return State(**fields)


@pytest.mark.skipif(not _HAVE_MPTABLE, reason="pristine WRF MPTABLE not available (Noah-MP tables)")
def test_sf_surface_noahmp_operational_runs_and_advances_land() -> None:
    """sf_surface_physics=4 (Noah-MP) via the EXACT scan coupler noahmp_surface_step."""

    from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic
    from gpuwrf.coupling.noahmp_surface_hook import noahmp_surface_step
    from gpuwrf.physics.noahmp.tables import load_noahmp_parameters

    grid = _land_grid()
    state = _land_state(grid)
    ny, nx = grid.ny, grid.nx
    shape = (ny, nx)
    zero = jnp.zeros(shape, dtype=jnp.float64)
    land = NoahMPLandState(
        tslb=jnp.broadcast_to(jnp.asarray(295.0), (NSOIL, ny, nx)),
        smois=jnp.broadcast_to(jnp.asarray(0.20), (NSOIL, ny, nx)),
        sh2o=jnp.broadcast_to(jnp.asarray(0.20), (NSOIL, ny, nx)),
        smcwtd=jnp.broadcast_to(jnp.asarray(0.20), shape),
        isnow=jnp.zeros(shape, dtype=jnp.int32),
        tsno=jnp.broadcast_to(jnp.asarray(273.0), (NSNOW, ny, nx)),
        snice=jnp.broadcast_to(zero, (NSNOW, ny, nx)), snliq=jnp.broadcast_to(zero, (NSNOW, ny, nx)),
        zsnso=jnp.broadcast_to(
            jnp.asarray([0.0, 0.0, 0.0, -0.05, -0.25, -0.7, -1.5]).reshape(NSNOW + NSOIL, 1, 1),
            (NSNOW + NSOIL, ny, nx),
        ),
        snowh=zero, sneqv=zero, sneqvo=zero, tauss=zero,
        albold=jnp.broadcast_to(jnp.asarray(0.2), shape),
        tv=jnp.broadcast_to(jnp.asarray(296.0), shape), tg=jnp.broadcast_to(jnp.asarray(295.0), shape),
        tah=jnp.broadcast_to(jnp.asarray(295.0), shape), eah=jnp.broadcast_to(jnp.asarray(2000.0), shape),
        canliq=zero, canice=zero, fwet=zero, lai=jnp.broadcast_to(jnp.asarray(2.0), shape),
        sai=jnp.broadcast_to(jnp.asarray(0.5), shape),
        cm=jnp.broadcast_to(jnp.asarray(0.01), shape), ch=jnp.broadcast_to(jnp.asarray(0.01), shape),
        t_skin=jnp.broadcast_to(jnp.asarray(295.0), shape), qsfc=jnp.broadcast_to(jnp.asarray(0.01), shape),
        znt=jnp.broadcast_to(jnp.asarray(0.05), shape), emiss=jnp.broadcast_to(jnp.asarray(0.97), shape),
        albedo=jnp.broadcast_to(jnp.asarray(0.2), shape), sfcrunoff=zero, udrunoff=zero,
    )
    static = NoahMPStatic(
        ivgtyp=jnp.full(shape, 5, dtype=jnp.int32), isltyp=jnp.full(shape, 6, dtype=jnp.int32),
        xland=jnp.ones(shape), landmask=jnp.ones(shape), lakemask=zero,
        lu_index=jnp.full(shape, 5, dtype=jnp.int32), tbot=jnp.broadcast_to(jnp.asarray(290.0), shape),
        dzs=jnp.asarray([0.05, 0.20, 0.45, 0.80]), zsoil=jnp.asarray([-0.05, -0.25, -0.7, -1.5]),
        lat=jnp.broadcast_to(jnp.asarray(28.0), shape), dx_m=3000.0,
        parameters=load_noahmp_parameters(_NOAHMP_TABLE_DIR),
        shdmax=jnp.broadcast_to(jnp.asarray(0.3), shape), shdfac=jnp.broadcast_to(jnp.asarray(0.3), shape),
    )

    class _Rad:
        soldn = jnp.full(shape, 600.0)
        lwdn = jnp.full(shape, 350.0)
        cosz = jnp.full(shape, 0.6)

    class _Clock:
        julian = 142.0
        yearlen = 365.0

    state_out, land_out = noahmp_surface_step(state, land, static, 90.0, radiation=_Rad(), clock=_Clock())
    assert _all_finite(state_out), "Noah-MP step produced a non-finite State field"
    for leaf in jax.tree_util.tree_leaves(land_out):
        assert np.all(np.isfinite(np.asarray(leaf))), "Noah-MP land carry has a non-finite leaf"
    # Noah-MP wrote real surface fluxes + re-derived CH + advanced the ground temperature.
    assert _maxabs(state_out.theta_flux) > 0.0, "Noah-MP did not write a surface theta flux"
    assert not np.allclose(np.asarray(land_out.ch), 0.01), "Noah-MP did not re-derive land CH (seed kept)"
    assert _changed(land_out.tg, land.tg), "Noah-MP did not advance the ground temperature carry"


@pytest.mark.skipif(
    not _NOAHCLASSIC_SAVEPOINTS.exists(),
    reason="proofs/v060/savepoints_noahclassic.json (WRF-derived Noah-classic bundle) not available",
)
def test_sf_surface_noahclassic_operational_runs_and_advances_land() -> None:
    """sf_surface_physics=2 (Noah classic) via the EXACT scan coupler
    noahclassic_surface_step, with the WRF-savepoint-derived static/land bundle."""

    from gpuwrf.coupling.noahclassic_surface_hook import (
        NoahClassicLandState,
        NoahClassicRadiation,
        NoahClassicStatic,
        noahclassic_surface_step,
    )
    from gpuwrf.physics.lsm_noah_classic import NoahClassicParams

    grid = _land_grid()
    state = _land_state(grid)
    ny, nx = grid.ny, grid.nx

    def tile(value, dtype=jnp.float64):
        return jnp.full((ny, nx), value, dtype=dtype)

    def tile4(values):
        arr = jnp.asarray(values, dtype=jnp.float64)
        return jnp.broadcast_to(arr, (ny, nx, arr.shape[0]))

    data = json.loads(_NOAHCLASSIC_SAVEPOINTS.read_text())
    col = next(c for c in data["columns"] if c["name"] == "daytime_veg10")
    rp = col["wrf"]["redprm"]
    snow = col["wrf"]["snow_in"]
    zero = jnp.zeros((ny, nx), dtype=jnp.float64)
    params = NoahClassicParams(
        bexp=tile(rp["bexp"]), dksat=tile(rp["dksat"]), dwsat=tile(rp["dwsat"]), psisat=tile(rp["psisat"]),
        quartz=tile(rp["quartz"]), f1=tile(rp["f1"]), smcmax=tile(rp["smcmax"]), smcwlt=tile(rp["smcwlt"]),
        smcref=tile(rp["smcref"]), smcdry=tile(rp["smcdry"]), kdt=tile(rp["kdt"]), frzx=tile(rp["frzx"]),
        slope=tile(rp["slope"]), snup=tile(rp["snup"]), salp=tile(rp["salp"]), czil=tile(rp["czil"]),
        sbeta=tile(rp["sbeta"]), csoil=tile(rp["csoil"]), fxexp=tile(rp["fxexp"]), zbot=tile(rp["zbot"]),
        cfactr=tile(rp["cfactr"]), cmcmax=tile(rp["cmcmax"]), rsmax=tile(rp["rsmax"]), topt=tile(rp["topt"]),
        rgl=tile(rp["rgl"]), hs=tile(rp["hs"]), rsmin=tile(rp["rsmin"]), lvcoef=tile(rp["lvcoef"]),
        nroot=tile(int(rp["nroot"]), dtype=jnp.int32), rtdis=tile4(rp["rtdis"]), alb=tile(rp["alb"]),
        embrd=tile(rp["embrd"]), xlai=tile(rp["xlai"]), z0brd=tile(rp["z0brd"]), shdfac=tile(rp["shdfac"]),
        is_urban=jnp.full((ny, nx), bool(col["vegtyp"] == col["isurban"])),
    )
    smav = (tile4(col["wrf"]["smc_in"]) - params.smcwlt[..., None]) / (params.smcmax - params.smcwlt)[..., None]
    static = NoahClassicStatic(
        params=params, zsoil=tile4(col["zsoil"]), sldpth=tile4(col["sldpth"]),
        snoalb=tile(col["state_in"]["snoalb"]), tbot=tile(col["tbot"]),
        solnet_albedo=tile(col["state_in"]["albbck"]), lwdn_emissivity=tile(col["state_in"]["emiss"]),
    )
    land = NoahClassicLandState(
        t1=tile(col["wrf"]["t1_in"]), stc=tile4(col["wrf"]["stc_in"]), smc=tile4(col["wrf"]["smc_in"]),
        sh2o=tile4(col["wrf"]["sh2o_in"]), cmc=tile(snow["cmc"]), sneqv=tile(snow["sneqv"]),
        snowh=tile(snow["snowh"]), sncovr=tile(snow["sncovr"]), snotime1=tile(col["state_in"]["snotime1"]),
        ribb=tile(col["wrf"]["chcm_in"]["ribb"]), flx4=zero, fvb=zero, fbur=zero, fgsn=zero,
        smcrel=smav, xlaidyn=params.xlai, hfx=zero, qfx=zero, lh=zero, grdflx=zero,
    )
    rad = NoahClassicRadiation(
        soldn=tile(col["wrf"]["forcing"]["soldn"]), lwdn=tile(col["wrf"]["forcing"]["glw"]),
        cosz=jnp.ones((ny, nx), dtype=jnp.float64),
    )

    state_out, land_out = noahclassic_surface_step(state, land, static, 90.0, radiation=rad)
    assert _all_finite(state_out), "Noah-classic step produced a non-finite State field"
    for leaf in jax.tree_util.tree_leaves(land_out):
        assert np.all(np.isfinite(np.asarray(leaf))), "Noah-classic land carry has a non-finite leaf"
    # Noah-classic wrote a real surface theta flux, advanced the soil column + skin temp.
    assert _maxabs(state_out.theta_flux) > 0.0, "Noah-classic did not write a surface theta flux"
    assert _changed(state_out.t_skin, state.t_skin), "Noah-classic did not advance the skin temperature"
    assert _changed(land_out.stc, land.stc), "Noah-classic did not advance the soil temperature column"
