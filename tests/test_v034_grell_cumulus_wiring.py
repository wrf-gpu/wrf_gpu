"""v0.3.4 o1-grell: operational wiring of cu=5 (Grell-3D) / cu=93 (Grell-Devenyi).

Coupled smoke through the real operational physics step (``_physics_step_forcing``)
on a 12x12 convective grid (G3/GD act only >= 4 points from a non-periodic
boundary; the ideal smoke grid is periodic, so both schemes compute everywhere
and G3's conv_grell_spread3d writes the 4..7 interior): finite state, convective
rain, theta/qv tendencies applied, untouched outside the WRF write region; cudt>0
fails closed with a named reason; the adapter routes to the WRF kernels.
CPU only.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.coupling import scan_adapters  # noqa: E402
from gpuwrf.runtime.operational_mode import (  # noqa: E402
    UnsupportedSchemeSelection,
    _physics_step_forcing,
    _resolve_operational_suite,
)
from gpuwrf.runtime.operational_state import initial_operational_carry  # noqa: E402

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402

import test_v013_operational_smoke as smoke  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proofs" / "v034"))
import grell_oracle_io as gio  # noqa: E402

NZ = 44


def _grell_state(grid):
    """Smoke State carrying realistic deep/dry-upper/marginal soundings (the
    oracle generator's), sub-saturated and with a stratosphere -- the shallow
    supersaturated b2 smoke column gives G3/GD no cloud top / no valid k22."""
    state = smoke._convective_state(grid)
    tile = gio.make_tile(11, nx=grid.nx, ny=grid.ny, kx=NZ,
                         regimes=("deep", "deep_dryupper", "marginal"))
    lev = lambda n: tile["f3"][n].transpose(1, 2, 0)  # (nx,k,ny) -> (k,ny,nx)
    t, q, pi, p = (lev(n)[:NZ] for n in ("t", "q", "pi", "p"))
    theta_m = (t / pi) * (1.0 + 461.6 / 287.0 * q)
    zf = np.concatenate([np.zeros((1, grid.ny, grid.nx)), np.cumsum(lev("dz8w")[:NZ], axis=0)])
    ph = jnp.asarray(9.81 * zf)
    return state.replace(theta=jnp.asarray(theta_m), qv=jnp.asarray(q), p=jnp.asarray(p),
                         p_total=jnp.asarray(p), ph=ph, ph_total=ph,
                         qc=jnp.zeros_like(state.qc), w=jnp.asarray(lev("w")))


def _setup(cu):
    grid = smoke._grid(nz=NZ, ny=12, nx=12)
    state = _grell_state(grid)
    nml = smoke._namelist(grid, dt_s=60.0, mp_physics=0, bl_pbl_physics=0,
                          sf_sfclay_physics=0, cu_physics=cu)
    return grid, state, nml


@pytest.mark.parametrize("cu", (5, 93))
def test_grell_operational_step_triggers_and_rains(cu):
    _grid, state, nml = _setup(cu)
    _resolve_operational_suite(nml)
    carry = initial_operational_carry(state)
    after = _physics_step_forcing(carry, nml, 0.0, run_radiation=False).state
    assert smoke._all_finite(after), f"cu={cu} produced a non-finite field"
    rain = np.asarray(after.rainc_acc) - np.asarray(state.rainc_acc)
    assert rain.max() > 0.0, f"cu={cu} did not rain"
    assert (rain >= 0.0).all()
    dth = np.abs(np.asarray(after.theta) - np.asarray(state.theta)).max(axis=0)
    assert dth.max() > 0.0, f"cu={cu} applied no theta tendency"
    if cu == 5:
        # conv_grell_spread3d writes RTHCUTEN only on ids+4..ide-5 (even periodic).
        ring = np.ones((12, 12), bool)
        ring[4:8, 4:8] = False
        assert dth[ring].max() == 0.0, "G3 theta tendency leaked outside WRF's write region"


def test_grell_cudt_cadence_fails_closed():
    _grid, _state, nml = _setup(5)
    nml = dataclasses.replace(nml, cumulus_cadence_steps=5, cudt_minutes=5.0)
    with pytest.raises(UnsupportedSchemeSelection) as excinfo:
        _resolve_operational_suite(nml)
    assert "cudt>0" in str(excinfo.value)


@pytest.mark.parametrize("cu, kernel", ((5, "g3drv_tile"), (93, "grelldrv_tile")))
def test_adapter_routes_to_wrf_kernel(cu, kernel, monkeypatch):
    """Deletion-sensitive routing: the adapter must call the WRF-port kernel."""
    from gpuwrf.physics import _grell_cup_jax as G

    calls = []
    real = getattr(G, kernel)

    def spy(**kw):
        calls.append(kernel)
        return real(**kw)

    monkeypatch.setattr(G, kernel, spy)
    grid, state, _nml = _setup(cu)
    scan_adapters.CU_SCAN_ADAPTERS[cu](state, 60.0, grid)
    assert calls == [kernel]
