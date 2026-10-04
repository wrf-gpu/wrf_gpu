"""B39: WRF MYNN surface-layer carry (MOL/HFX/QFX/QSFC/PBLH, DX) reaches SFCLAY.

The real-column fidelity gate is ``mynn_sl_oracle_gate.py`` (pristine
module_sf_mynn.F, all PROD columns). These CPU tests pin the plumbing:
producers write WRF's carried values, views hand them to the surface layer,
the RK step keeps them, and force_fp64 never upcasts the REAL-locked leaves.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.contracts import precision as precision_contract
from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import SURFACE_LAYER_CARRY_LEAVES
from gpuwrf.contracts.state import State
from gpuwrf.coupling.noahmp_surface_hook import _build_column_view
from gpuwrf.physics.noahmp_coupler import noahmp_surface_adapter
from gpuwrf.physics.surface_constants import CP_D
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics

GWDO_HISTORY_LEAVES = getattr(precision_contract, "GWDO_DIAGNOSTIC_LEAVES", ())
assert GWDO_HISTORY_LEAVES in ((), ("dtaux3d", "dtauy3d", "dusfcg", "dvsfcg"))

_COUPLER_TEST = Path(__file__).resolve().parents[2] / "test_noahmp_coupler.py"
_spec = importlib.util.spec_from_file_location("b39_noahmp_fixture", _COUPLER_TEST)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def _cpu_state(*, gwd_opt=0):
    import gpuwrf.contracts.state as state_module
    import jax

    device = state_module._gpu_device
    state_module._gpu_device = lambda: jax.devices("cpu")[0]
    try:
        grid = GridSpec.canary_3km_template()
        return State.zeros(grid, gwd_opt=gwd_opt) if GWDO_HISTORY_LEAVES else State.zeros(grid)
    finally:
        state_module._gpu_device = device


@pytest.mark.parametrize("gwd_opt", (0, 1) if GWDO_HISTORY_LEAVES else (0,))
def test_leaves_are_real_locked_zero_seeded_and_survive_force_fp64(gwd_opt):
    from gpuwrf.runtime.operational_mode import _PHYSICS_NON_DRY_REPLACE_FIELDS, _enforce_operational_precision

    state = _cpu_state(gwd_opt=gwd_opt)
    mynn_history = getattr(precision_contract, "MYNN_DIAGNOSTIC_LEAVES", ())
    assert mynn_history in ((), ("el_pbl", "maxmf", "maxwidth", "ztop_plume"))
    real_leaves = SURFACE_LAYER_CARRY_LEAVES + mynn_history + GWDO_HISTORY_LEAVES
    assert State.__slots__[71:] == real_leaves
    for name in real_leaves:
        leaf = getattr(state, name)
        assert name in _PHYSICS_NON_DRY_REPLACE_FIELDS
        if name in GWDO_HISTORY_LEAVES and gwd_opt == 0:
            assert leaf is None, name
            continue
        shape = state.qke.shape if name in ("el_pbl", "dtaux3d", "dtauy3d") else state.xland.shape
        assert leaf.dtype == jnp.float32 and leaf.shape == shape
        assert float(jnp.abs(leaf).max()) == 0.0
        assert name in _PHYSICS_NON_DRY_REPLACE_FIELDS
    enforced = _enforce_operational_precision(state, force_fp64=True)
    assert enforced.theta.dtype == jnp.float64
    for name in real_leaves:
        if getattr(state, name) is None:
            assert getattr(enforced, name) is None, name
        else:
            assert getattr(enforced, name).dtype == jnp.float32, name
    # producers write f64 values; State.replace keeps the seeded REAL dtype
    written = state.replace(hfx=jnp.ones(state.xland.shape, jnp.float64))
    assert written.hfx.dtype == jnp.float32
    for name in mynn_history + GWDO_HISTORY_LEAVES:
        if getattr(state, name) is None:
            continue  # inactive GWDO is checked as None above; gwd_opt=1 checks all REAL writes
        shape = getattr(state, name).shape
        written = state.replace(**{name: jnp.ones(shape, jnp.float64)})
        assert getattr(written, name).dtype == jnp.float32, name
        np.testing.assert_array_equal(np.asarray(getattr(written, name)), np.ones(shape, np.float32))


def test_column_view_passes_carry_and_grid_dx():
    state = _cpu_state()
    values = {name: jnp.full(state.xland.shape, 1.0 + i, jnp.float32)
              for i, name in enumerate(SURFACE_LAYER_CARRY_LEAVES)}
    view = _build_column_view(state.replace(**values), _GridWithDx(9000.0))
    for name, value in values.items():
        np.testing.assert_array_equal(np.asarray(getattr(view, name)), np.asarray(value))
    assert view.dx_m == 9000.0
    assert _build_column_view(state).dx_m is None


class _GridWithDx:
    class projection:
        dx_m = 9000.0

    metrics = None

    def __init__(self, dx):
        self.projection.dx_m = dx


def test_wstar_inputs_are_deletion_sensitive():
    state, *_ = _fixture._build()
    n = state.xland.shape
    warm = state.replace(hfx=jnp.full(n, 300.0), qfx=jnp.full(n, 1.0e-4), pblh=jnp.full(n, 1500.0),
                         mol=jnp.full(n, -0.5), qsfc=jnp.full(n, 0.012), dx_m=9000.0)
    cold = surface_layer_with_diagnostics(state)
    hot = surface_layer_with_diagnostics(warm)
    # WSTAR (fluxc>0) and VSGD (dx 9 km) raise the bulk wind -> different BR/ust.
    assert not np.allclose(np.asarray(hot.br), np.asarray(cold.br))
    assert not np.allclose(np.asarray(hot.fluxes.ustar), np.asarray(cold.fluxes.ustar))


@pytest.mark.skipif(not _fixture.HAVE_TABLES, reason="pristine WRF MPTABLE not available")
def test_noahmp_adapter_writes_wrf_carry_semantics():
    state, land, static, rad, clock = _fixture._build()
    n = state.xland.shape
    zeros = jnp.zeros(n)
    state = state.replace(mol=zeros, hfx=zeros, qfx=zeros, qsfc=zeros, pblh=jnp.full(n, 800.0))
    diag = surface_layer_with_diagnostics(state)
    state_out, land_out, blended = noahmp_surface_adapter(
        state, land, static, radiation=rad, clock=clock, dt=90.0)
    is_land = np.asarray(state.xland).reshape(-1) < 1.5
    rho = np.asarray(diag.fluxes.rhosfc).reshape(-1)
    qx = np.asarray(state.qv)[..., 0].reshape(-1)
    # MOL is sfclay's; HFX/QFX are the BLENDED physical fluxes the kinematic
    # handles were built from (Noah-MP over land, sfclay over water).
    np.testing.assert_array_equal(np.asarray(state_out.mol), np.asarray(diag.mol))
    np.testing.assert_allclose(np.asarray(state_out.hfx).reshape(-1),
                               np.asarray(blended.theta_flux).reshape(-1) * rho * CP_D * (1.0 + 0.84 * qx),
                               rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(np.asarray(state_out.qfx).reshape(-1),
                               np.asarray(blended.qv_flux).reshape(-1) * rho, rtol=1e-12, atol=1e-15)
    np.testing.assert_array_equal(np.asarray(state_out.hfx).reshape(-1)[~is_land],
                                  np.asarray(diag.hfx).reshape(-1)[~is_land])
    # QSFC: Noah-MP QSFC1D over land, sfclay over water.
    qsfc = np.asarray(state_out.qsfc).reshape(-1)
    np.testing.assert_array_equal(qsfc[is_land], np.asarray(land_out.qsfc).reshape(-1)[is_land])
    np.testing.assert_array_equal(qsfc[~is_land], np.asarray(diag.qsfc).reshape(-1)[~is_land])
    # PBLH is not a surface output: unchanged by the surface step.
    np.testing.assert_array_equal(np.asarray(state_out.pblh), np.asarray(state.pblh))


def test_mynn_adapters_store_the_pblh_their_solve_computes():
    """B39: both operational MYNN adapters keep the GET_PBLH their step computes."""

    import inspect

    from gpuwrf.coupling import physics_couplers as C

    for adapter in (C.mynn_adapter, C.mynn_adapter_with_source_leaves):
        source = inspect.getsource(adapter)
        assert "step_mynn_pbl_column_with_pblh(" in source
        assert "_with_mynn_pblh(" in source
    state = _cpu_state()
    stored = C._with_mynn_pblh(state, jnp.arange(state.xland.size, dtype=jnp.float64))
    assert stored.pblh.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(stored.pblh).reshape(-1), np.arange(state.xland.size, dtype=np.float32))


def test_pre_b39_pickled_state_unpickles_with_real_zero_leaves():
    """integrate IT21: States pickled before the B39 slots existed must still load."""

    import pickle

    state = _cpu_state()
    blob = pickle.dumps(state)
    # Simulate a pre-B39 pickle: drop the five slot entries from the slot-state dict.
    reduced = state.__reduce_ex__(2)
    dict_state, slot_state = reduced[2]
    old_slot_state = {k: v for k, v in slot_state.items() if k not in SURFACE_LAYER_CARRY_LEAVES}
    restored = State.__new__(State)
    restored.__setstate__((dict_state, old_slot_state))
    for name in SURFACE_LAYER_CARRY_LEAVES:
        leaf = getattr(restored, name)
        shape = state.qke.shape if name == "el_pbl" else state.xland.shape
        assert leaf.dtype == jnp.float32 and leaf.shape == shape
        assert float(jnp.abs(leaf).max()) == 0.0
    np.testing.assert_array_equal(np.asarray(restored.theta), np.asarray(state.theta))
    roundtrip = pickle.loads(blob)
    assert roundtrip.active_field_names() == state.active_field_names()
