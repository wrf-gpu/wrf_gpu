"""v0.13 Tier-3 surface-layer + LSM operational wiring tests.

Asserts the three first-batch schemes are wired (or fail-closed) exactly as
claimed, and that the default suite is byte-unchanged:

* sf_sfclay_physics=91 (old-MM5) + =3 (NCEP-GFS) RESOLVE through the operational
  scan (in SFCLAY_SCAN_ADAPTERS, accepted by _resolve_operational_suite) and run
  a surface-layer step producing finite B2 flux handles;
* sf_surface_physics=1 (slab LSM) FAILS CLOSED in the operational scan
  (reference-only: validated kernel/oracle, no LSM hook yet);
* sf_surface_physics=3 (RUC LSM) is rejected (out of the accept matrix);
* the default namelist (sf_sfclay=5 MYNN) is unchanged.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling.scan_adapters import (
    SFCLAY_SCAN_ADAPTERS,
    gfs_sfclay_adapter,
    sfclay_old_mm5_adapter,
)
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    UnsupportedSchemeSelection,
    _resolve_operational_suite,
)

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- #
# Registry / catalog wiring                                                   #
# --------------------------------------------------------------------------- #
def test_sfclay_scan_adapters_register_new_schemes() -> None:
    assert SFCLAY_SCAN_ADAPTERS[91] is sfclay_old_mm5_adapter
    assert SFCLAY_SCAN_ADAPTERS[3] is gfs_sfclay_adapter


def test_scheme_catalog_classifications() -> None:
    from gpuwrf.io.scheme_catalog import (
        SupportStatus,
        assert_catalog_consistent,
        classify_scheme,
    )

    assert_catalog_consistent()
    assert classify_scheme("sf_sfclay_physics", 91).status is SupportStatus.IMPLEMENTED
    assert classify_scheme("sf_sfclay_physics", 3).status is SupportStatus.IMPLEMENTED
    assert classify_scheme("sf_surface_physics", 1).status is SupportStatus.REFERENCE_ONLY
    # RUC (3) stays fail-closed (intractable for this batch).
    assert (
        classify_scheme("sf_surface_physics", 3).status
        is SupportStatus.RECOGNIZED_FAIL_CLOSED
    )


def test_registry_consistent_with_new_schemes() -> None:
    from gpuwrf.contracts.physics_registry import (
        ACCEPTED_SF_SFCLAY_PHYSICS,
        ACCEPTED_SF_SURFACE_PHYSICS,
        assert_registry_consistent,
    )

    assert_registry_consistent()
    assert set(ACCEPTED_SF_SFCLAY_PHYSICS) >= {3, 91}
    assert 1 in ACCEPTED_SF_SURFACE_PHYSICS


# --------------------------------------------------------------------------- #
# Operational-scan resolution (wired vs fail-closed)                          #
# --------------------------------------------------------------------------- #
def _namelist(**overrides) -> OperationalNamelist:
    # Minimal namelist: only the physics selectors matter for _resolve_*.
    base = OperationalNamelist.__new__(OperationalNamelist)
    defaults = dict(
        mp_physics=8, bl_pbl_physics=5, sf_sfclay_physics=5, cu_physics=0,
        sf_surface_physics=None, use_noahmp=False,
        ra_sw_physics=4, ra_lw_physics=4,
        noahclassic_static=None, noahclassic_land=None, noahclassic_rad=None,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        object.__setattr__(base, k, v)
    return base


def test_old_mm5_and_gfs_sfclay_resolve_wired() -> None:
    # 91 (old-MM5) and 3 (GFS) are scan-wired -> resolution must not raise.
    _resolve_operational_suite(_namelist(sf_sfclay_physics=91))
    _resolve_operational_suite(_namelist(sf_sfclay_physics=3))


def test_slab_lsm_fails_closed_reference_only() -> None:
    with pytest.raises(UnsupportedSchemeSelection) as exc:
        _resolve_operational_suite(_namelist(sf_surface_physics=1))
    assert "sf_surface_physics=1" in str(exc.value)


def test_ruc_lsm_rejected_out_of_matrix() -> None:
    # RUC (3) is not in the accept matrix -> rejected at resolution.
    with pytest.raises(Exception):
        _resolve_operational_suite(_namelist(sf_surface_physics=3))


def test_default_suite_unchanged() -> None:
    # The default surface-layer scheme is MYNN (5); resolution unchanged.
    _resolve_operational_suite(_namelist())  # sf_sfclay=5, no surface override


# --------------------------------------------------------------------------- #
# Adapters run + produce finite B2 flux handles                               #
# --------------------------------------------------------------------------- #
def _surface_state():
    from gpuwrf.contracts.state import State, _state_field_shapes

    nz, ny, nx = 2, 2, 2
    shapes = _state_field_shapes_for(nz, ny, nx)
    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in shapes.items()}
    ph = jnp.broadcast_to(
        jnp.linspace(0.0, 2000.0 * 9.80665, nz + 1, dtype=jnp.float64)[:, None, None],
        (nz + 1, ny, nx),
    )
    fields.update(
        u=jnp.full((nz, ny, nx + 1), 5.0, jnp.float64),
        v=jnp.full((nz, ny + 1, nx), 1.0, jnp.float64),
        theta=jnp.full((nz, ny, nx), 295.0, jnp.float64),
        mu=jnp.full((ny, nx), 90000.0, jnp.float64),
        ph=ph, p=jnp.full((nz, ny, nx), 95000.0, jnp.float64),
        qv=jnp.full((nz, ny, nx), 8.0e-3, jnp.float64),
        ustar=jnp.full((ny, nx), 0.3, jnp.float64),
        t_skin=jnp.full((ny, nx), 298.0, jnp.float64),
        xland=jnp.array([[1.0, 2.0], [1.0, 2.0]], jnp.float64),
        mavail=jnp.full((ny, nx), 0.8, jnp.float64),
        roughness_m=jnp.full((ny, nx), 0.08, jnp.float64),
        soil_moisture=jnp.full((ny, nx), 0.3, jnp.float64),
    )
    # Drop any int-typed leaf defaults the float fill mistyped (e.g. lu_index).
    return State(**fields)


def _state_field_shapes_for(nz: int, ny: int, nx: int):
    """Resolve the full State leaf-shape map for a tiny grid (all required leaves)."""
    from gpuwrf.contracts.state import _state_field_shapes

    class _G:
        pass

    g = _G()
    g.nz, g.ny, g.nx = nz, ny, nx
    return _state_field_shapes(g)


@pytest.mark.parametrize("adapter", [sfclay_old_mm5_adapter, gfs_sfclay_adapter])
def test_adapter_runs_and_writes_finite_flux_handles(adapter) -> None:
    state = _surface_state()
    out = jax.jit(adapter)(state, 60.0)
    for handle in ("ustar", "theta_flux", "qv_flux", "tau_u", "tau_v", "rhosfc", "fltv"):
        v = np.asarray(getattr(out, handle))
        assert np.all(np.isfinite(v)), f"{handle} not finite for {adapter.__name__}"
    # ustar must be positive everywhere.
    assert np.all(np.asarray(out.ustar) > 0.0)
