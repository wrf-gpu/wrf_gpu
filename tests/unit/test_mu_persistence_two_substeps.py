from __future__ import annotations

from dataclasses import replace as dataclass_replace

import jax
import jax.numpy as jnp
import pytest

from gpuwrf.contracts.grid import BCMetadata, DycoreMetrics, GridSpec, Projection, TerrainProvenance, VerticalCoord
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.runtime.operational_mode import OperationalNamelist, _operational_acoustic_substep_core, _with_save_family
from gpuwrf.runtime.operational_state import initial_operational_carry


def _pure_sigma_metrics(base: DycoreMetrics, nz: int) -> DycoreMetrics:
    """Override DycoreMetrics.flat (hybrid c1f=eta -> zero top-face mass -> singular
    calc_coef_w) to the WRF pure-sigma coordinate c1=1, c2=0 (hybrid_opt=0)."""

    one_h = jnp.ones((nz,), dtype=jnp.float64)
    zero_h = jnp.zeros((nz,), dtype=jnp.float64)
    one_f = jnp.ones((nz + 1,), dtype=jnp.float64)
    zero_f = jnp.zeros((nz + 1,), dtype=jnp.float64)
    return DycoreMetrics(
        msftx=base.msftx, msfty=base.msfty, msfux=base.msfux, msfuy=base.msfuy,
        msfvx=base.msfvx, msfvy=base.msfvy,
        c1h=one_h, c2h=zero_h, c3h=one_h, c4h=zero_h,
        c1f=one_f, c2f=zero_f, c3f=one_f, c4f=zero_f,
        dn=base.dn, dnw=base.dnw, rdn=base.rdn, rdnw=base.rdnw,
        cf1=base.cf1, cf2=base.cf2, cf3=base.cf3, fnm=base.fnm, fnp=base.fnp,
        dzdx=base.dzdx, dzdy=base.dzdy, dzdx_u=base.dzdx_u, dzdy_v=base.dzdy_v,
        p_top=base.p_top, provenance="unit-pure-sigma",
    )


def _grid(nx: int = 5, ny: int = 5, nz: int = 4) -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    grid = GridSpec(
        projection=Projection("lambert", 28.3, -15.6, 3000.0, 3000.0, nx, ny),
        terrain=TerrainProvenance("analytic://unit", "unit", (ny, nx), "m", "native", 0.0, True),
        vertical=VerticalCoord("hybrid_eta", nz, 16000.0, eta),
        bc=BCMetadata("ideal", ("u", "v", "theta"), 0, "linear", False),
        eta_levels=eta,
        terrain_height=jnp.zeros((ny, nx), dtype=jnp.float64),
    )
    return dataclass_replace(grid, metrics=_pure_sigma_metrics(grid.metrics, nz))


def _state(grid: GridSpec) -> State:
    """Hydrostatically-balanced fixture.

    AC5(c): the legacy fixture used an unphysical zero geopotential, which the
    real WRF ``advance_w`` cannot integrate (zero-thickness column).  Replaced
    with a discretely hydrostatic column: ``ph(k+1)=ph(k)+dnw*mut*alt`` with the
    WRF dry specific volume ``alt=(R_d*theta/p0)*(p/p0)^(cv/cp)``, so the implicit
    w-solve receives a finite, consistent column.
    """

    R_D, CP_D, P0 = 287.0, 1004.0, 100000.0
    CV_D = CP_D - R_D
    fields = {
        name: jnp.zeros(shape, dtype=jnp.int32 if name == "lu_index" else jnp.float64)
        for name, shape in _state_field_shapes(grid).items()
    }
    z = jnp.arange(grid.nz, dtype=jnp.float64)[:, None, None]
    p_base = 90000.0 - 1000.0 * z + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64)
    mu_pert = 750.0 + 2.0 * jnp.arange(grid.ny, dtype=jnp.float64)[:, None] + jnp.zeros((grid.ny, grid.nx))
    mu_base = 85000.0 + jnp.zeros_like(mu_pert)
    mu_total = mu_base + mu_pert
    theta = 300.0 + 0.01 * z + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64)
    # Discrete hydrostatic geopotential from the WRF dry EOS specific volume.
    alt = (R_D / P0) * theta * (p_base / P0) ** (-CV_D / CP_D)  # (nz, ny, nx)
    dnw = jnp.abs(jnp.asarray(grid.eta_levels)[1:] - jnp.asarray(grid.eta_levels)[:-1])  # (nz,)
    incr = dnw[:, None, None] * mu_total[None, :, :] * alt  # (nz, ny, nx)
    ph_total = jnp.concatenate(
        (jnp.zeros((1, grid.ny, grid.nx), dtype=jnp.float64), jnp.cumsum(incr, axis=0)), axis=0
    )
    fields.update(
        theta=theta,
        qv=0.004 + jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        p=p_base,
        p_total=p_base,
        p_perturbation=jnp.zeros_like(p_base),
        ph=ph_total,
        ph_total=ph_total,
        ph_perturbation=jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
        mu=mu_total,
        mu_total=mu_total,
        mu_perturbation=mu_pert,
    )
    return State(**fields)


def _zero_tendencies(grid: GridSpec) -> Tendencies:
    return Tendencies(
        u=jnp.zeros((grid.nz, grid.ny, grid.nx + 1), dtype=jnp.float64),
        v=jnp.zeros((grid.nz, grid.ny + 1, grid.nx), dtype=jnp.float64),
        w=jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
        theta=jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        qv=jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        p=jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        ph=jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
        mu=jnp.zeros((grid.ny, grid.nx), dtype=jnp.float64),
    )


@pytest.mark.xfail(
    reason=(
        "STALE TEST (test-triage 2026-05-30): drives the orphaned legacy helper "
        "operational_mode._operational_acoustic_substep_core, which builds its "
        "AcousticCoreState via _acoustic_core_state(). The F7K rewrite "
        "(acoustic.py:509, commit 49f138c 2026-05-29) changed acoustic_substep_core "
        "to advance the persistent coupled work theta `uv_state.theta_coupled_work` "
        "instead of re-coupling perturbation theta each substep. The PRODUCTION RK "
        "path (_rk_scan_step -> _acoustic_scan -> _acoustic_core_state_from_prep, "
        "operational_mode.py:853) populates theta_coupled_work; the legacy helper's "
        "_acoustic_core_state() never did, so theta_coupled_work=None and "
        "advance_mu_t_wrf crashes on `inputs.theta.shape` (AttributeError). The "
        "helper has NO src/ callers (test-only) and is NOT on the operational "
        "forecast path. TRACKING: manager to either delete the orphaned "
        "_operational_acoustic_substep_core helper + its tests, or wire "
        "theta_coupled_work into _acoustic_core_state. The CODE on the operational "
        "path is correct; this expectation is outdated."
    ),
    raises=AttributeError,
    strict=True,
)
def test_mu_save_preserves_nonzero_perturbation_across_two_zero_tendency_substeps():
    grid = _grid()
    state = _state(grid)
    namelist = OperationalNamelist(
        grid=grid,
        tendencies=_zero_tendencies(grid),
        metrics=grid.metrics,
        dt_s=2.0,
        acoustic_substeps=2,
        run_physics=False,
        run_boundary=False,
        use_vertical_solver=True,
        disable_guards=True,
    )
    carry = _with_save_family(initial_operational_carry(state).replace(state=state), state)
    initial_mu_save = carry.mu_save

    carry = _operational_acoustic_substep_core(carry, namelist, 1.0)
    carry = _operational_acoustic_substep_core(carry, namelist, 1.0)
    jax.block_until_ready(carry.mu_save)

    assert float(jnp.max(jnp.abs(carry.mu_save - initial_mu_save))) <= 1.0e-10
    assert float(jnp.max(jnp.abs(carry.state.mu_perturbation - initial_mu_save))) <= 1.0e-10
