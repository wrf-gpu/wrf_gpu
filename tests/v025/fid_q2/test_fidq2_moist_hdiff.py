"""fid-q2: WRF diff_opt=1 horizontal mixing of moist species and QNI/QNR (CPU, abstract trace + real operands).

Pristine ``rk_scalar_tend`` (module_em.F:1380-1390) adds ``horizontal_diffusion('m', scalar, scalar_tends,
mut, ..., xkmhd)`` at RK1 for every moist species and every other scalar; solve_em passes ``grid%muts`` and
``grid%xkhh`` (solve_em.F:2299-2314, :2869-2884) and rk_update_scalar reuses the term in all RK stages.
The port lacked it, which dried/warmed the lowest levels over steep terrain (alisios twin verdict LW9:
Q2/RH2 over land). Numerical evidence vs a pristine serial WRF d01 run: moist_hdiff_evidence.json here.
"""
import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling.boundary_apply import NESTED_BOUNDARY_SCALAR_SPECIES
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.domain_tree import DomainTree

CASE = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")
WRF_EM = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_em.F")
WRF_SOLVE = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/solve_em.F")


@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    """Fresh product load of PROD d01+d02 on CPU (no stale pickle, E138); ~3 min."""
    if not (CASE / "namelist.input").is_file():
        pytest.skip("PROD case not mounted")
    from gpuwrf.contracts import state as state_contract
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains

    root = tmp_path_factory.mktemp("prod")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(state_contract, "_gpu_device", lambda: jax.devices("cpu")[0])
        hierarchy, bundles, _, _, _, carries = _load_domains(
            NestedPipelineConfig(CASE, root / "out", root / "proof", hours=1, max_dom=2), ("d01", "d02"))
        tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
        cases = {}
        for domain in ("d01", "d02"):
            nml, carry = tree.domains[domain].namelist, carries[domain]
            if op._h_diabatic_pair_enabled(nml) and carry.h_diabatic is None:
                carry = carry.replace(h_diabatic=jnp.zeros_like(carry.state.theta))
            cases[domain] = (nml, carry)
        yield cases


def _trace(nml, carry):
    """Abstract one-step trace; returns the recorded seam calls."""
    rec = {"helper": [], "hdiff": [], "finished": [], "root_sc": []}
    originals = {
        "helper": op._diffopt1_scalar_horizontal_diffusion,
        "hdiff": op.horizontal_diffusion_coord_scalar_tendency,
        "finish": op._carry_from_finished_stage,
        "root_sc": op._root_scalar_stage_tendencies,
    }

    def helper(reference, xkhh, muts, namelist):
        rec["helper"].append((reference, xkhh, muts))
        return originals["helper"](reference, xkhh, muts, namelist)

    def hdiff(field, xkhh, mass, **kwargs):
        rec["hdiff"].append((field, xkhh, mass, kwargs.get("base_3d") is not None))
        return originals["hdiff"](field, xkhh, mass, **kwargs)

    def finish(*args, **kwargs):
        out = originals["finish"](*args, **kwargs)
        rec["finished"].append(out.state.mu_total)
        return out

    def root_sc(*args, **kwargs):
        rec["root_sc"].append((tuple(args[1]), kwargs.get("sc_tendencies")))
        return originals["root_sc"](*args, **kwargs)

    shapes = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), carry)
    op._diffopt1_scalar_horizontal_diffusion = helper
    op.horizontal_diffusion_coord_scalar_tendency = hdiff
    op._carry_from_finished_stage = finish
    op._root_scalar_stage_tendencies = root_sc
    try:
        jax.clear_caches()
        jax.eval_shape(lambda c: op._advance_chunk_fori(
            c, nml, jnp.asarray(1, jnp.int32), op.build_clock_base(nml), n_steps=1,
            cadence=int(nml.radiation_cadence_steps)), shapes)
    finally:
        op._diffopt1_scalar_horizontal_diffusion = originals["helper"]
        op.horizontal_diffusion_coord_scalar_tendency = originals["hdiff"]
        op._carry_from_finished_stage = originals["finish"]
        op._root_scalar_stage_tendencies = originals["root_sc"]
        jax.clear_caches()
    return rec


def test_pristine_rk_scalar_tend_mixes_every_scalar_with_xkhh_and_muts():
    if not WRF_EM.exists():
        pytest.skip("pristine WRF source unavailable")
    em = WRF_EM.read_text()
    body = em[em.index("SUBROUTINE rk_scalar_tend"): em.index("END SUBROUTINE rk_scalar_tend")]
    rk1 = body[body.index("rk_step_1: IF( rk_step == 1 ) THEN"):]
    assert rk1.index("diff_opt1 : IF (config_flags%diff_opt .eq. 1) THEN") < rk1.index(
        "CALL horizontal_diffusion ( 'm', scalar(ims,kms,jms,im)") < rk1.index("CALL sixth_order_diffusion")
    assert "IF (.not. mix2_off)" in rk1
    solve = WRF_SOLVE.read_text()
    for start in (solve.index("CALL rk_scalar_tend (  im, im"), solve.index("CALL rk_scalar_tend ( is, is")):
        call = solve[start: start + 2000]
        assert "grid%muts" in call and "grid%xkhh" in call


@pytest.mark.parametrize("domain", ["d01", "d02"])
def test_step_mixes_every_scalar_once_with_rk1_muts_and_theta_xkhh(prod, domain):
    nml, carry = prod[domain]
    assert jax.devices()[0].platform == "cpu"
    assert int(nml.diff_opt) == 1 and int(nml.km_opt) == 4
    rec = _trace(nml, carry)
    assert len(rec["finished"]) == int(nml.rk_order)
    # Formed once per step (RK1), from the post-acoustic RK1 total mass grid%muts.
    assert len(rec["helper"]) == 1
    reference, xkhh, muts = rec["helper"][0]
    assert muts is rec["finished"][0]
    theta_calls = [c for c in rec["hdiff"] if c[3]]
    scalar_calls = [c for c in rec["hdiff"] if not c[3]]
    assert len(theta_calls) == 1 and theta_calls[0][1] is xkhh  # the same grid%xkhh as theta
    assert [c[0] for c in scalar_calls] == [getattr(reference, n) for n in NESTED_BOUNDARY_SCALAR_SPECIES]
    assert all(c[1] is xkhh for c in scalar_calls)


def test_root_sc_tend_carries_mixing_without_sixth_order(prod):
    nml, carry = prod["d01"]
    no6 = dataclasses.replace(nml, diff_6th_opt=0)
    rec = _trace(no6, carry)
    assert len(rec["helper"]) == 1
    assert len(rec["root_sc"]) == int(nml.rk_order)
    for species, sc in rec["root_sc"]:
        assert sc is not None and set(species) <= set(sc)


def test_nested_sc_tend_carries_mixing_without_sixth_order(prod):
    nml, carry = prod["d02"]
    rec = _trace(dataclasses.replace(nml, diff_6th_opt=0), carry)
    assert len(rec["helper"]) == 1


def test_mixing_matches_literal_wrf_on_real_prod_operands(prod):
    from tests.test_v0234_nested_scalar_diffusion_source_repair import _literal_nested_scalar

    nml, carry = prod["d01"]
    haloed = op.apply_halo(carry.state, op.halo_spec(nml.grid))
    *_, xkhh = op._diffopt1_dry_forward_tendencies(
        haloed, nml, base_state=carry.base_state, return_scalar_xkhh=True)
    mu = haloed.mu_total
    out = op._diffopt1_scalar_horizontal_diffusion(haloed, xkhh, mu, nml)
    assert tuple(out) == NESTED_BOUNDARY_SCALAR_SPECIES
    m = nml.metrics
    mass = np.asarray(m.c1h)[:, None, None] * np.asarray(mu)[None] + np.asarray(m.c2h)[:, None, None]
    maps = tuple(np.asarray(getattr(m, n), np.float64) for n in ("msftx", "msfty", "msfux", "msfuy", "msfvx", "msfvy"))
    dx, dy = float(nml.grid.projection.dx_m), float(nml.grid.projection.dy_m)
    for name in NESTED_BOUNDARY_SCALAR_SPECIES:
        expected = _literal_nested_scalar(getattr(haloed, name), xkhh, mass, maps, dx, dy)
        actual = np.asarray(out[name], np.float64)
        scale = np.abs(expected).max()
        if name == "qv":
            assert scale > 0.0  # PROD init has no condensate on d01: only qv is non-vacuous here
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-9 * scale)
    # Real-terrain signature: the lowest level over >1000 m land gains moisture (pristine WRF does too).
    hgt = np.asarray(nml.grid.terrain_height)
    high = hgt >= 1000.0
    assert high.any()
    qv_tend = np.asarray(out["qv"])[0] / mass[0]
    assert qv_tend[high].mean() > 0.0


def test_shipping_fp32_kernel_mixing_matches_literal_wrf_on_real_prod_operands(prod, monkeypatch):
    """Release path: dyn_diff_fp32 horizontal_diffusion_fp32(name='m') (Pallas, interpret on CPU)."""
    from gpuwrf.dynamics import explicit_diffusion as ed
    from gpuwrf.kernels import dyn_diff_fp32 as kd
    from tests.test_v0234_nested_scalar_diffusion_source_repair import _literal_nested_scalar

    nml, carry = prod["d01"]
    haloed = op.apply_halo(carry.state, op.halo_spec(nml.grid))
    *_, xkhh = op._diffopt1_dry_forward_tendencies(
        haloed, nml, base_state=carry.base_state, return_scalar_xkhh=True)
    calls = []
    kernel = kd.horizontal_diffusion_fp32

    def interpreted(*args, **kwargs):
        calls.append(kwargs.get("name"))
        return kernel(*args, **{**kwargs, "interpret": True})

    monkeypatch.setattr(ed, "_NATIVE_DIFFUSION_FP32", True)
    monkeypatch.setattr(kd, "horizontal_diffusion_fp32", interpreted)
    mu = haloed.mu_total
    out = op._diffopt1_scalar_horizontal_diffusion(haloed, xkhh, mu, nml)
    assert calls == ["m"] * len(NESTED_BOUNDARY_SCALAR_SPECIES)  # the fp32 kernel really ran (E114)
    m = nml.metrics
    mass = np.asarray(m.c1h)[:, None, None] * np.asarray(mu)[None] + np.asarray(m.c2h)[:, None, None]
    maps = tuple(np.asarray(getattr(m, n), np.float64) for n in ("msftx", "msfty", "msfux", "msfuy", "msfvx", "msfvy"))
    expected = _literal_nested_scalar(np.asarray(haloed.qv, np.float64), np.asarray(xkhh, np.float64), mass, maps,
                                      float(nml.grid.projection.dx_m), float(nml.grid.projection.dy_m))
    actual = np.asarray(out["qv"])
    assert actual.dtype == np.float32
    scale = np.abs(expected).max()
    assert scale > 0.0
    np.testing.assert_allclose(actual.astype(np.float64), expected, rtol=0.0, atol=1e-5 * scale)
    assert not actual[:, 0, :].any() and not actual[:, -1, :].any()
    assert not actual[:, :, 0].any() and not actual[:, :, -1].any()
