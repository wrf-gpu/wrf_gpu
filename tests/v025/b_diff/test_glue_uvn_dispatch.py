"""GPUWRF_DYN_GLUE_FUSED part parsing and the nested-only large-step u/v kernel dispatch (uvn).

Fidelity of the kernel itself is the pristine REAL4 gate in probe_dyn_real.py (uv family:
horizontal_pressure_gradient + coriolis + curvature, lid on/off, cq mutant). These tests pin
the dispatch: parts parse, 'uvn' enables the fused kernel on NESTED children only (the root
keeps the XLA path: the fused kernel flips the root step layout), 'pin' adds the row-major
operand constraint, and the default (unset) path is untouched.
"""
import inspect

import jax
import pytest

from gpuwrf.dynamics.core import rk_addtend_dry as rk
from gpuwrf.kernels import dyn_real_fp32 as real
from gpuwrf.runtime import operational_mode as op


@pytest.mark.parametrize("value,parts", [
    ("0", set()), ("", set()), ("1", {"mom", "uv"}), ("uvn_pin", {"uvn", "pin"}),
    ("mom_omega_rhsph_uv2", {"mom", "omega", "rhsph", "uv2"}),
])
def test_glue_parts(monkeypatch, value, parts):
    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", value)
    assert real.glue_parts() == frozenset(parts)


@pytest.mark.parametrize("value,nested_only", [("uvn_pin", True), ("uvn", True), ("uv", False), ("uv_uvn", False), ("0", False)])
def test_nested_only_predicate(monkeypatch, value, nested_only):
    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", value)
    assert rk.large_step_uv_nested_only() is nested_only


def test_pin_rows_only_with_pin_part(monkeypatch):
    import jax.numpy as jnp
    x = jnp.ones((2, 3, 4), jnp.float32)
    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", "uvn")
    assert real.pin_rows((x,))[0] is x
    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", "uvn_pin")
    assert real.pin_rows((x,))[0] is not x


def test_augment_gates_fused_uv_on_nested_children():
    source = inspect.getsource(op._augment_large_step_tendencies)
    assert "large_step_uv_fused_enabled(u_t, v_t, base_state) and (" in source
    assert "_acoustic_lateral_bc_flags(namelist)[2] or not large_step_uv_nested_only()" in source


# --- L1' 'momuvn' (BD80): u+v momentum-advection stencils on NESTED children only -------------------------

@pytest.mark.parametrize("value,field,nested,expected", [
    ("rhsph_uvn_pin", "u", True, False), ("0", "v", True, False), ("1", "v", False, True),
    ("mom", "u", False, True), ("mom", "w", None, True), ("momuv", "u", False, True), ("momuv", "w", True, False),
    ("momuvn_rhsph_uvn_pin", "u", True, True), ("momuvn_rhsph_uvn_pin", "v", True, True),
    ("momuvn_rhsph_uvn_pin", "u", False, False), ("momuvn_rhsph_uvn_pin", "w", True, False), ("momuvn", "v", None, False),
])
def test_glue_mom_parts(monkeypatch, value, field, nested, expected):
    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", value)
    assert real.glue_mom(field, nested) is expected


def test_momentum_advection_gets_static_nested_flag():
    source = inspect.getsource(op._augment_large_step_tendencies)
    for field in ("u", "v"):
        assert (f"advect_{field}_flux(\n            haloed.{field}, vel, rdx=1.0 / dx, rdy=1.0 / dy, "
                "nested=_acoustic_lateral_bc_flags(namelist)[2],") in source


def _pallas_names(jaxpr):
    names = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            info = eqn.params.get("name", eqn.params.get("name_and_src_info"))
            names.append(getattr(info, "name", info))
        for param in eqn.params.values():  # nested jit / control-flow bodies
            for sub in (param if isinstance(param, (tuple, list)) else (param,)):
                inner = getattr(sub, "jaxpr", sub)
                if eqn.primitive.name != "pallas_call" and hasattr(inner, "eqns"):
                    names += _pallas_names(inner)
    return names


@pytest.mark.parametrize("value,nested,expected", [
    ("momuvn_rhsph_uvn_pin", True, ["b_diff_advect_u_fp32", "b_diff_advect_v_fp32"]),
    ("momuvn_rhsph_uvn_pin", False, []),
    ("rhsph_uvn_pin", True, []),
    ("mom", False, ["b_diff_advect_u_fp32", "b_diff_advect_v_fp32"]),
])
def test_momuvn_dispatch(monkeypatch, value, nested, expected):
    """The fused u/v stencils run exactly when the part enables them for this domain (E114: assert the
    fast path ran, fresh lambdas per case so no trace cache hides a flag flip, E140)."""
    import jax.numpy as jnp
    from gpuwrf.dynamics.flux_advection import CoupledVelocities, advect_u_flux, advect_v_flux
    monkeypatch.setenv("GPUWRF_DYN_GLUE_FUSED", value)
    nz, ny, nx = 6, 7, 9
    z = lambda *shape: jnp.full(shape, 0.5, jnp.float32)
    vel = CoupledVelocities(ru=z(nz, ny, nx), rv=z(nz, ny, nx), rom=z(nz + 1, ny, nx), specified=True,
                            ru_full=z(nz, ny, nx + 1), rv_full=z(nz, ny + 1, nx))
    kw = dict(rdx=1 / 3000., rdy=1 / 3000., rdzw=z(nz), fzm=z(nz), fzp=z(nz), nested=nested)
    names = []
    names += _pallas_names(jax.make_jaxpr(lambda q: advect_u_flux(q, vel, **kw))(z(nz, ny, nx + 1)).jaxpr)
    names += _pallas_names(jax.make_jaxpr(lambda q: advect_v_flux(q, vel, **kw))(z(nz, ny + 1, nx)).jaxpr)
    assert names == expected
