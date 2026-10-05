"""Independent literal contract for the LW12 fused-glue default (b-diff #15).

GPUWRF_DYN_GLUE_FUSED = "momuvn_rhsph_uvn_pin": the rhs_ph REAL stencil kernel (PROD branch: order >= 4,
specified, non-hydrostatic, rigid lid) plus the fused large-step u/v PGF + Coriolis + curvature kernel
on nested-child domains only (the root keeps XLA), with row-major pinned operands (BD72), plus (v0.3.2,
BD82 on the P2 layout pin) the fused u and v momentum-advection stencils on nested children only (w and
the root keep XLA). The value is pinned literally, it must parse to exactly those dispatch parts, and
both opt-outs must hold.
"""
import os

import pytest

from gpuwrf import _fast_defaults as defaults

KEY = "GPUWRF_DYN_GLUE_FUSED"
EXPECTED = "momuvn_rhsph_uvn_pin"


@pytest.fixture(autouse=True)
def _restore_environ():
    """apply_fast_path_defaults() sets EVERY release switch in os.environ; restore exactly so
    no fast-path flag leaks into later tests of the session (CPU pytest stays legacy)."""
    saved_env, saved_status = dict(os.environ), dict(defaults.FAST_DEFAULTS_STATUS)
    yield
    os.environ.clear()
    os.environ.update(saved_env)
    defaults.FAST_DEFAULTS_STATUS.clear()
    defaults.FAST_DEFAULTS_STATUS.update(saved_status)


def test_lw12_glue_default_is_the_registered_selection():
    assert defaults.FAST_PATH_DEFAULTS.get(KEY) == EXPECTED


def test_lw12_glue_default_dispatches_rhsph_and_nested_only_pinned_uv(monkeypatch):
    import jax.numpy as jnp
    from gpuwrf.dynamics.core import rhs_ph
    from gpuwrf.dynamics.core import rk_addtend_dry as rk
    from gpuwrf.kernels import dyn_real_fp32 as real

    monkeypatch.setenv(KEY, defaults.FAST_PATH_DEFAULTS[KEY])
    monkeypatch.setenv("GPUWRF_DYN_REAL_ALL", defaults.FAST_PATH_DEFAULTS["GPUWRF_DYN_REAL_ALL"])
    assert real.glue_parts() == frozenset({"momuvn", "rhsph", "uvn", "pin"})
    assert rk.large_step_uv_nested_only()
    # momentum advection: u and v fused on nested children only; the root and w keep the XLA path
    assert real.glue_mom("u", True) and real.glue_mom("v", True)
    assert not real.glue_mom("u", False) and not real.glue_mom("v", False)
    assert not real.glue_mom("w", True) and not real.glue_mom("w", None)
    x = jnp.zeros((2, 3, 4), jnp.float32)
    # PROD rhs_ph call: h_sca_adv_order 5, specified/nested, non-hydrostatic, gw, rigid lid.
    assert rhs_ph._rhs_ph_fused(5, True, True, True, True, x, x)
    assert not rhs_ph._rhs_ph_fused(5, True, True, True, False, x, x)


@pytest.mark.parametrize("optout", ["0", ""])
def test_lw12_glue_default_keeps_explicit_optout(monkeypatch, optout):
    monkeypatch.setenv("GPUWRF_FAST_DEFAULTS", "1")
    monkeypatch.setenv(KEY, optout)
    defaults.apply_fast_path_defaults()
    assert os.environ.get(KEY) == optout


def test_lw12_glue_default_applies_when_unset(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.setenv("GPUWRF_FAST_DEFAULTS", "1")
    defaults.apply_fast_path_defaults()
    assert os.environ.get(KEY) == EXPECTED


def test_lw12_master_optout_leaves_glue_unset(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.setenv("GPUWRF_FAST_DEFAULTS", "0")
    defaults.apply_fast_path_defaults()
    assert KEY not in os.environ
