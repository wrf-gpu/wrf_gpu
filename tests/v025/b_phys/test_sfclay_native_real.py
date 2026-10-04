"""BP47: native WRF-REAL MYNN surface layer (GPUWRF_SFCLAY_NATIVE_REAL).

Fidelity gate = mynn_sl_oracle_gate.py vs the REAL*4 pristine oracle on all
PROD columns; these CPU tests pin dtype/S2 behaviour and the default path.
"""
from __future__ import annotations

import importlib.util
import os
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics import surface_layer as SL
from gpuwrf.physics.fp32.surface_layer_real import native_real_enabled, surface_layer_with_diagnostics_real

_COUPLER_TEST = Path(__file__).resolve().parents[2] / "test_noahmp_coupler.py"
_spec = importlib.util.spec_from_file_location("bp47_noahmp_fixture", _COUPLER_TEST)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)


def _warm_state():
    state, *_ = _fixture._build()
    n = state.xland.shape
    return state.replace(hfx=jnp.full(n, 150.0), qfx=jnp.full(n, 5.0e-5), pblh=jnp.full(n, 900.0),
                         mol=jnp.full(n, -0.2), qsfc=jnp.full(n, 0.011), dx_m=9000.0)


def test_flag_default_off_and_default_entry_is_f64_impl(monkeypatch):
    import gpuwrf.physics.fp32.surface_layer_real as real_mod

    # Off unless set; the v0.3 release defaults set it (opt-out =0 / GPUWRF_FAST_DEFAULTS=0 -> f64 entry).
    assert native_real_enabled() is (os.environ.get("GPUWRF_SFCLAY_NATIVE_REAL", "0") == "1")
    monkeypatch.setattr(real_mod, "_NATIVE_REAL", False)  # the legacy entry is the f64 impl
    state = _warm_state()
    public = SL.surface_layer_with_diagnostics(state)
    impl = SL._surface_layer_impl(state, False, jnp.float64)
    for a, b in zip(jax.tree.leaves(public), jax.tree.leaves(impl)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert public.hfx.dtype == jnp.float64


def test_native_outputs_real_and_close_to_f64():
    state = _warm_state()
    real = surface_layer_with_diagnostics_real(state)
    ref = SL._surface_layer_impl(state, False, jnp.float64)
    for name in ("hfx", "lh", "u10", "v10", "t2", "q2", "zol", "mol", "psim", "psih", "br", "znt", "qsfc"):
        a, b = getattr(real, name), getattr(ref, name)
        assert a.dtype == jnp.float32, name
        assert np.all(np.isfinite(np.asarray(a))), name
        np.testing.assert_allclose(np.asarray(a, np.float64), np.asarray(b), rtol=2e-4, atol=2e-5, err_msg=name)
    assert real.fluxes.ustar.dtype == jnp.float32


def test_native_lowered_hlo_has_no_f64_arithmetic():
    from jaxlib import _jax

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from precision_inventory import inventory

    state = _warm_state()
    options = _jax.HloPrintOptions()
    options.print_metadata = True
    options.print_large_constants = False
    # The fixture view is not a pytree: close over it (inputs become f64 constants).
    hlo = (jax.jit(lambda: surface_layer_with_diagnostics_real(state)).lower()
           .compiler_ir("hlo").as_hlo_module().to_string(options))
    _, nodes = inventory(hlo, Path.cwd())
    compute = Counter(n["opcode"] for n in nodes if n["category"] == "compute")
    assert not compute, compute
