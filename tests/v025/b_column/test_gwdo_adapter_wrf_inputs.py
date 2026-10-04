"""B42: gwdo_adapter (default AND native) reproduces WRF's per-step u/v face increment.

Fixture (build_gwdo_adapter_fixture.py): real CPU-WRF PROD d01 State crops
(crop j 19-34, i 103-118 = strongest u AND v drag; 00z: 248 drag columns, 12z: 26) and the pristine bl_gwdo_run
tendencies (REAL, WRF phy_prep inputs) mapped to the C-grid with add_a2c_u/v.
The pre-B42 default adapter (log-linear p8w, g = 9.80665) misses by up to 8-12 %
of the maximum increment; WRF inputs leave REAL-vs-f64 rounding only.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling.physics_couplers import build_gwdo_statics_from_wrf_fields, gwdo_adapter

FIX = Path(__file__).resolve().parent / "fixtures" / "gwdo_prod_d01_adapter_crops.npz"
STATIC = ("VAR", "CON", "OA1", "OA2", "OA3", "OA4", "OL1", "OL2", "OL3", "OL4")


class _State:
    def __init__(self, **leaves):
        self.__dict__.update(leaves)

    def replace(self, **kw):
        return _State(**{**self.__dict__, **kw})


@pytest.mark.parametrize("native", ["0", "1"])
@pytest.mark.parametrize("case", ["night", "day"])
def test_adapter_increment_matches_pristine_wrf(case, native, monkeypatch):
    if native == "1" and importlib.util.find_spec("gpuwrf.kernels.phys_gwdo_column") is None:
        pytest.skip("native GWDO kernel not in this tree")
    monkeypatch.setenv("GPUWRF_GWDO_NATIVE_REAL", native)
    z = np.load(FIX)
    f = lambda n: z[f"{case}_{n}"]
    st = [f(f"st_{n}") for n in STATIC]
    statics = build_gwdo_statics_from_wrf_fields(
        *st, dx_m=float(f("dx")), sina=f("st_SINALPHA"), cosa=f("st_COSALPHA"))
    grid = SimpleNamespace(metrics=SimpleNamespace(fnm=jnp.asarray(f("fnm"), jnp.float64),
                                                   fnp=jnp.asarray(f("fnp"), jnp.float64)))
    state = _State(**{k: jnp.asarray(f(k), jnp.float64) for k in ("theta", "qv", "p", "ph", "u", "v")})
    out = gwdo_adapter(state, float(f("dt")), statics, grid)
    du = np.asarray(out.u, np.float64) - np.asarray(state.u, np.float64)
    dv = np.asarray(out.v, np.float64) - np.asarray(state.v, np.float64)
    eu, ev = f("expected_du"), f("expected_dv")
    assert np.abs(eu).max() > 0.02 and np.abs(ev).max() > 0.02  # real drag in the crop (m/s)
    assert np.isfinite(du).all() and np.isfinite(dv).all()
    assert np.abs(du - eu).max() <= 5e-5 * np.abs(eu).max(), np.abs(du - eu).max()
    assert np.abs(dv - ev).max() <= 5e-5 * np.abs(ev).max(), np.abs(dv - ev).max()
