"""fid-q2 RE03: GPUWRF_NEST_O3_FROM_PARENT — a nest radiates with the parent's held o3rad, forced down (CPU).

WRF o3input=2 interpolates the CAM ozone only on domain 1 at its radiation calls (module_radiation_driver.F:1803);
o3rad is Registry rdf=(p2c) (Registry.EM_COMMON:1264), so after EVERY parent step the force-down sets the nest's
o3rad = interp_fcn SINT(parent o3rad) level-wise (inc/nest_forcedown_interp.inc:265-280, imask 1), and the nest's
radiation uses that held field; d03 receives d02's copy.  Oracle: pristine WRF V4.7.1 oznini/ozn_time_int/ozn_p_int
+ interp_fcn on real WN3 0227 (proofs/nest_o3/, fixture nest_o3_forcedown_0227_v1.npz).
"""
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.nesting import nest_o3
from gpuwrf.nesting.interp import build_sint_weights
from gpuwrf.nesting.nest_o3 import NEST_O3_FROM_PARENT_ENV, force_child_o3rad

FIXTURE = Path(__file__).with_name("fixtures") / "nest_o3_forcedown_0227_v1.npz"
SHA256 = "18105327238bf3c314f42b03bd67e3e5b3dd05574fd11fa2ec0df6b62822d6ff"
CASE = Path("<USER_HOME>/wrf_gpu2_lanes/wn3/cases/20260227_18z_a1")


def _fixture():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == SHA256
    with np.load(FIXTURE, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


@pytest.mark.parametrize("child", ["d02", "d03"])
def test_force_child_o3rad_reproduces_pristine_wrf_forcedown(child):
    fx = _fixture()
    crop = fx[f"{child}_parent_crop"]
    (ip, jp), (cny, cnx) = fx[f"{child}_ip_jp"], fx[f"{child}_shape"]
    jj, ii, expected = fx[f"{child}_j"], fx[f"{child}_i"], fx[f"{child}_expected"]
    w = build_sint_weights(parent_grid_ratio=3, i_parent_start=int(ip), j_parent_start=int(jp),
                           parent_ny=crop.shape[1], parent_nx=crop.shape[2], child_ny=int(cny), child_nx=int(cnx))
    got = np.asarray(jax.jit(lambda f: force_child_o3rad(f, w, parent_grid_ratio=3))(jnp.asarray(crop)))
    assert got.dtype == np.float32 and got.shape == (crop.shape[0], cny, cnx)
    sampled = got[:, jj, ii]
    ulp = np.abs(sampled.astype(np.float64) - expected) / np.spacing(np.abs(expected)).astype(np.float64)
    assert np.isfinite(sampled).all() and ulp.max() <= 8.0, ulp.max()
    assert (sampled == expected).mean() >= 0.75
    # discriminating: WRF's field is SINT, not a containing-parent-cell copy nor the parent profile itself
    containing = crop[:, (jj // 3) + int(jp) - 1, (ii // 3) + int(ip) - 1]
    assert (np.abs(containing.astype(np.float64) - expected) / expected).max() > 1e-3


def load_nested(tmp: Path):
    """Product _load_domains (WN3 0227 d01+d02, CPU) with the flag on -> (bundles, carries, tree)."""
    from gpuwrf.contracts import state as state_contract
    from gpuwrf.integration import nested_pipeline as pipeline
    from gpuwrf.runtime.domain_tree import DomainTree

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(NEST_O3_FROM_PARENT_ENV, "1")
        mp.setattr(state_contract, "_gpu_device", lambda: jax.devices("cpu")[0])
        cfg = pipeline.NestedPipelineConfig(CASE, tmp / "out", tmp / "proof", hours=3, max_dom=2)
        hierarchy, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01", "d02"))
        tree = DomainTree.from_domains(hierarchy, bundles)
    return bundles, carries, tree


def test_batched_carry_force_maps_each_case_like_the_single_force():
    """nested_pipeline._batched_force (B > 1) vmaps the same per-case SINT over the leading case axis."""
    from dataclasses import dataclass, replace
    from types import SimpleNamespace

    from gpuwrf.nesting.nest_o3 import force_child_carry_o3rad

    @dataclass(frozen=True)
    class Carry:
        o3rad: object

        def replace(self, **kw):
            return replace(self, **kw)

    fx = _fixture()
    crop = fx["d02_parent_crop"]
    (ip, jp), (cny, cnx) = fx["d02_ip_jp"], fx["d02_shape"]
    w = build_sint_weights(parent_grid_ratio=3, i_parent_start=int(ip), j_parent_start=int(jp),
                           parent_ny=crop.shape[1], parent_nx=crop.shape[2], child_ny=int(cny), child_nx=int(cnx))
    parents = jnp.stack([jnp.asarray(crop), jnp.asarray(crop * np.float32(1.5))])
    child0 = jnp.zeros((2, crop.shape[0], int(cny), int(cnx)), jnp.float32)
    got = force_child_carry_o3rad(Carry(child0), Carry(parents), SimpleNamespace(mass=w), parent_grid_ratio=3,
                                  batched=True).o3rad
    assert got.shape == child0.shape and got.dtype == jnp.float32
    for b in range(2):
        one = force_child_carry_o3rad(Carry(child0[b]), Carry(parents[b]), SimpleNamespace(mass=w),
                                      parent_grid_ratio=3).o3rad
        np.testing.assert_array_max_ulp(np.asarray(got[b]), np.asarray(one), maxulp=4)
    sampled = np.asarray(got[0])[:, fx["d02_j"], fx["d02_i"]]
    np.testing.assert_array_max_ulp(sampled, fx["d02_expected"].astype(np.float32), maxulp=8)
    # absent leaf on either side: the released carry passes through untouched
    plain = Carry(None)
    assert force_child_carry_o3rad(plain, Carry(parents), SimpleNamespace(mass=w), parent_grid_ratio=3) is plain


@pytest.fixture(scope="module")
def nested(tmp_path_factory):
    if not (CASE / "wrfinput_d02").is_file():
        pytest.skip("WN3 0227 case inputs unavailable")
    return load_nested(tmp_path_factory.mktemp("nest_o3"))


def _spy(monkeypatch):
    from gpuwrf.runtime import operational_mode as om

    calls = []

    def fake(state, grid=None, **kw):
        calls.append(kw)
        return jnp.zeros(state.theta.shape, jnp.float32), fake.held_diag

    monkeypatch.setattr(om, "rrtmg_theta_tendency", fake)
    return calls, fake


def _cam_columns(carry, namelist):
    from gpuwrf.coupling.physics_couplers import rrtmg_ozone_columns
    from gpuwrf.runtime import operational_mode as om

    return rrtmg_ozone_columns(carry.state, namelist.grid, time_utc=namelist.time_utc, lead_seconds=0.0,
                               clock_base=om._rad_clock_base(om.build_clock_base(namelist)),
                               radiation_static=namelist.radiation_static, land_state=carry.noahmp_land)


def _refresh(carry, namelist, run_radiation):
    from gpuwrf.runtime import operational_mode as om

    return om._refresh_rrtmg_driver(carry, namelist, 0.0, run_radiation, om.build_clock_base(namelist))


def test_nested_carries_seed_a_held_real_o3rad(nested):
    _bundles, carries, _tree = nested
    for name, carry in carries.items():
        assert carry.o3rad is not None, name
        assert carry.o3rad.dtype == jnp.float32 and carry.o3rad.shape == carry.state.theta.shape
        assert not np.asarray(carry.o3rad).any()  # WRF: filled by the root's first radiation call / first force-down


def test_root_refreshes_o3rad_only_at_its_radiation_calls(nested, monkeypatch):
    bundles, carries, _tree = nested
    ns, root = bundles["d01"].namelist, carries["d01"]
    assert bool(ns.boundary_config.force_geopotential)  # the root runs its own CAM call
    calls, fake = _spy(monkeypatch)
    fake.held_diag = root.radiation_diagnostics
    held = _refresh(root, ns, False)
    assert not calls and held.o3rad is root.o3rad  # between radiation calls WRF holds o3rad
    new = _refresh(root, ns, True)
    own = np.asarray(_cam_columns(root, ns))  # what the OFF path's radiation interpolates itself
    assert np.array_equal(np.asarray(new.o3rad), np.moveaxis(own, -1, 0))
    assert np.array_equal(np.asarray(calls[-1]["ozone_vmr_override"]), own)  # radiation consumed the stored field
    assert own.min() > 0.0


def test_forcedown_sets_child_o3rad_from_parent_and_child_radiates_with_it(nested, monkeypatch):
    from gpuwrf.runtime.domain_tree import _operational_force

    bundles, carries, tree = nested
    calls, fake = _spy(monkeypatch)
    fake.held_diag = carries["d01"].radiation_diagnostics
    root = _refresh(carries["d01"], bundles["d01"].namelist, True)
    (edge,) = tree.children("d01")
    forced = _operational_force(edge, root, carries["d02"])
    ratio = int(edge.parent_grid_ratio)
    got = np.asarray(forced.o3rad)
    compiled = nest_o3._compiled_force(edge.weights.mass, ratio, False)(root.o3rad)
    assert np.array_equal(got, np.asarray(compiled)) and got.min() > 0
    # eager SINT agrees to XLA:CPU whole-jit fusion ulps (E186); G1 binds the SINT itself to pristine WRF
    eager = np.asarray(force_child_o3rad(root.o3rad, edge.weights.mass, parent_grid_ratio=ratio))
    np.testing.assert_array_max_ulp(got, eager, maxulp=4)
    # only the o3rad leaf differs from the released force-down
    plain = _operational_force(edge, root.replace(o3rad=None), carries["d02"])
    for name in forced.__dataclass_fields__:
        if name not in ("o3rad",):
            a, b = jax.tree.leaves(getattr(forced, name)), jax.tree.leaves(getattr(plain, name))
            assert len(a) == len(b) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(a, b)), name
    ns2 = bundles["d02"].namelist
    assert not bool(ns2.boundary_config.force_geopotential)
    fake.held_diag = forced.radiation_diagnostics
    child = _refresh(forced, ns2, True)
    assert child.o3rad is forced.o3rad  # a nest never recomputes o3rad
    used = np.asarray(calls[-1]["ozone_vmr_override"])
    assert np.array_equal(used, np.moveaxis(np.asarray(forced.o3rad), 0, -1))
    # the nest's own CAM profile (the released behaviour) is a different field (RE02: up to 10 % low levels)
    own = np.asarray(_cam_columns(forced, ns2))
    assert np.abs(own - used).max() / used.max() > 1e-4


def test_fused_cascade_forces_child_o3rad_like_the_eager_force(nested, monkeypatch):
    """The default fused leaf cascade (PROD d01->d02, WN3 d02->d03) forces o3rad at the same seam as the eager path."""
    from gpuwrf.runtime import domain_tree as dt

    bundles, carries, tree = nested
    ns, ns2 = bundles["d01"].namelist, bundles["d02"].namelist
    (edge,) = tree.children("d01")
    parent = carries["d01"].replace(o3rad=jnp.moveaxis(_cam_columns(carries["d01"], ns), -1, 0))
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "0")
    monkeypatch.setattr(dt, "_advance_chunk", lambda carry, *_a, **_k: carry)  # isolate the force-down seam
    program = dt._build_fused_cascade_program(
        parent_name="d01", parent_namelist=ns, parent_cadence=int(ns.radiation_cadence_steps),
        child_names=("d02",), child_namelists=(ns2,), child_weights=(edge.weights,),
        child_bdy_widths=(int(tree.domains["d02"].state.u_bdy.shape[2]),),
        child_ratios=(int(edge.parent_grid_ratio),), child_cadences=(int(ns2.radiation_cadence_steps),),
    )
    _parent_new, (child_new,) = program(parent, (carries["d02"],), 1, (1,))
    eager = np.asarray(dt._operational_force(edge, parent, carries["d02"]).o3rad)
    got = np.asarray(child_new.o3rad)
    assert got.dtype == np.float32 and got.min() > 0.0
    np.testing.assert_array_max_ulp(got, eager, maxulp=4)


def test_fused_force_reads_the_post_step_parent(nested, monkeypatch):
    """WRF forces the nest's o3rad AFTER the parent's step (med_nest_force after the parent solve), so the fused
    cascade must read the POST-step parent.  The stubbed parent step changes o3rad (as a parent radiation call does):
    the forced child must be SINT(post-step parent), not SINT(pre-step parent) (critic b-core O3R, E39)."""
    from gpuwrf.runtime import domain_tree as dt

    bundles, carries, tree = nested
    ns, ns2 = bundles["d01"].namelist, bundles["d02"].namelist
    (edge,) = tree.children("d01")
    parent = carries["d01"].replace(o3rad=jnp.moveaxis(_cam_columns(carries["d01"], ns), -1, 0))
    parent_shape = parent.state.theta.shape
    monkeypatch.setenv("GPUWRF_NESTED_AOT", "0")

    def parent_step_refreshes_o3(carry, *_a, **_k):  # isolate the force seam; only the parent step changes o3rad
        if carry.state.theta.shape == parent_shape:
            return carry.replace(o3rad=carry.o3rad * jnp.float32(2.0))
        return carry

    monkeypatch.setattr(dt, "_advance_chunk", parent_step_refreshes_o3)
    program = dt._build_fused_cascade_program(
        parent_name="d01", parent_namelist=ns, parent_cadence=int(ns.radiation_cadence_steps),
        child_names=("d02",), child_namelists=(ns2,), child_weights=(edge.weights,),
        child_bdy_widths=(int(tree.domains["d02"].state.u_bdy.shape[2]),),
        child_ratios=(int(edge.parent_grid_ratio),), child_cadences=(int(ns2.radiation_cadence_steps),),
    )
    _parent_new, (child_new,) = program(parent, (carries["d02"],), 1, (1,))
    post = np.asarray(dt._operational_force(edge, parent.replace(o3rad=parent.o3rad * jnp.float32(2.0)),
                                            carries["d02"]).o3rad)
    pre = np.asarray(dt._operational_force(edge, parent, carries["d02"]).o3rad)
    got = np.asarray(child_new.o3rad)
    assert np.isfinite(got).all() and np.isfinite(post).all() and np.isfinite(pre).all()  # E200
    np.testing.assert_array_max_ulp(got, post, maxulp=4)
    assert np.abs(got - pre).max() / np.abs(pre).max() > 0.5  # discriminating: the pre-step field is half


def test_override_reaches_both_rrtmg_column_states(nested):
    from gpuwrf.coupling.physics_couplers import _rrtmg_column_inputs
    from gpuwrf.runtime import operational_mode as om

    bundles, carries, _tree = nested
    ns, carry = bundles["d02"].namelist, carries["d02"]
    marker = jnp.full(carry.state.theta.shape[1:] + carry.state.theta.shape[:1], 3.25e-7, jnp.float32)
    sw, lw, *_ = _rrtmg_column_inputs(carry.state, ns.grid, time_utc=ns.time_utc, lead_seconds=0.0,
                                      clock_base=om._rad_clock_base(om.build_clock_base(ns)),
                                      radiation_static=ns.radiation_static, land_state=carry.noahmp_land,
                                      ozone_vmr_override=marker)
    assert np.array_equal(np.asarray(sw.ozone_vmr), np.asarray(marker))
    assert np.array_equal(np.asarray(lw.ozone_vmr), np.asarray(marker))
