"""#2 (mass-opus MO07): the specified ROOT relaxes normal winds by WRF's tendency only.

WRF relaxes u/v in the relax zone once, through relax_bdy_dry -> ru_tendf/rv_tendf (RK1-frozen,
dyn_em/solve_em.F:943-945, module_bc_em.F:161) that rk_addtend_dry adds every stage; advance_uv
skips the spec face and spec_bdyupdate pins it (solve_em.F:1361-1381).  The port additionally
blended the relax rows toward the boundary work target on EVERY acoustic substep
(boundary_apply.apply_normal_bdy_work relax_rows=True) -> double relaxation.  Pristine-WRF mimic
of that blend on the Swiss case (mass-opus MO07): relax-zone normal U/V RMSE 0.32 m/s at h1,
interior 10 m wind slowdown -0.16...-0.19 m/s from h1 to h24.  This test proves on the actual
production own-step (CPU abstract trace, legacy and native fp32 acoustic) that the root keeps the
spec-face pin but no relax-row blend, rejects the pre-fix stage config, and that nested children
(already tendency-only through the frozen WRF bundle) keep their config.
"""
from dataclasses import fields

import jax
import jax.numpy as jnp
import pytest
from prod_inputs import prod_domains

from gpuwrf.coupling import boundary_apply
from gpuwrf.dynamics.core import acoustic
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.domain_tree import DomainTree


def _tree():
    hierarchy, bundles, _, _, _, carries = prod_domains()
    return DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False), carries


def _trace_root_step(monkeypatch, *, drop_stage_flag=False):
    """relax_rows of every normal-wind boundary call and the stage configs of one root own-step."""

    tree, carries = _tree()
    nml, carry = tree.domains["d01"].namelist, carries["d01"]
    seen = {"legacy": [], "native": [], "stage": []}
    legacy, native = acoustic.apply_normal_bdy_work, boundary_apply.apply_normal_bdy_work
    config_cls = op.AcousticCoreConfig

    def spy(path, real):
        def call(*args, **kwargs):
            seen[path].append(bool(kwargs.get("relax_rows", True)))
            return real(*args, **kwargs)
        return call

    def stage_config(**kwargs):
        if drop_stage_flag:  # mutant: the pre-fix _acoustic_scan stage config
            kwargs.pop("specified_relax_tendency")
        cfg = config_cls(**kwargs)
        seen["stage"].append(cfg)
        return cfg

    # legacy acoustic binds the helper at import; the native adapter imports it per call
    monkeypatch.setattr(acoustic, "apply_normal_bdy_work", spy("legacy", legacy))
    monkeypatch.setattr(boundary_apply, "apply_normal_bdy_work", spy("native", native))
    monkeypatch.setattr(op, "AcousticCoreConfig", stage_config)
    shapes = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), carry)
    clock = op.build_clock_base(nml)
    try:
        jax.clear_caches()
        jax.eval_shape(lambda c: op._advance_chunk_fori(
            c, nml, jnp.asarray(1, jnp.int32), clock, n_steps=1,
            cadence=int(nml.radiation_cadence_steps)), shapes)
    finally:
        jax.clear_caches()
    assert jax.devices()[0].platform == "cpu"
    return nml, seen


@pytest.mark.parametrize("native_fp32", ["0", "1"])
def test_root_own_step_pins_spec_face_without_relax_row_blend(monkeypatch, native_fp32):
    monkeypatch.setenv("GPUWRF_DYN_FP32", native_fp32)
    nml, seen = _trace_root_step(monkeypatch)
    assert op._specified_bdy_cadence_active(nml) and not op._nested_frozen_wrf_boundary_active(nml)
    assert seen["stage"] and all(cfg.specified_relax_tendency for cfg in seen["stage"])
    # the selected acoustic path ran (E114) and every normal-wind boundary call is spec-face only
    ran = "native" if native_fp32 == "1" else "legacy"
    assert seen[ran] and not any(seen["legacy"] + seen["native"])


def test_pre_fix_stage_config_is_rejected(monkeypatch):
    """Deletion mutant (E39): without the stage flag the root blends the relax rows again."""

    monkeypatch.setenv("GPUWRF_DYN_FP32", "0")
    _, seen = _trace_root_step(monkeypatch, drop_stage_flag=True)
    assert seen["legacy"] and all(seen["legacy"])


def test_nested_children_keep_their_boundary_config():
    """WRF children get relax_bdy_dry the same way (solve_em.F:943 specified .or. nested); the port's
    frozen-bundle child already applies the tendency only and never sets the root stage flag."""

    tree, _ = _tree()
    child = tree.domains["d02"].namelist
    assert op._nested_frozen_wrf_boundary_active(child)
    assert not op._specified_bdy_cadence_active(child)
    assert "specified_relax_tendency" in {f.name for f in fields(acoustic.AcousticCoreConfig)}
    assert acoustic.AcousticCoreConfig.__dataclass_fields__["specified_relax_tendency"].default is False
