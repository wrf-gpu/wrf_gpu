"""B43 / BD92: GPUWRF_ROOT_SCALAR_BDY_RK1 — specified-root moist/scalar lateral forcing in WRF's RK cadence.

Pristine solve_em.F on a specified root with have_bcs_moist/have_bcs_scalar = .false. (Registry default):
QV gets relax_bdy_scalar + spec_bdy_scalar into moist_tend at rk_step 1 (:2345-2380), resident for all three
rk_update_scalar calls, and spec_bdy_final after microphysics (:4714-4733); QC..QG/QNI/QNR get flow_dep_bdy after
every stage's rk_update_scalar (:2426-2438, :2995-3015).  The released port relaxes qv at END of step on the
post-physics value, which leaves the relax zone drier by one step's moistening (mass-opus 01:13Z: d01 rings 1-4 QV
-.00424 g/kg at tau1, 80x the CPU pair).

The chain test runs the unchanged pristine routines (pristine_boundary.py) against the candidate helpers on REAL
PROD d01 operands (wrfinput fields, the real coupled wrfbdy_d01 records for WRF, the production leaves from the
current wrfbdy loader for the port).  The dispatch test traces the real PROD d01 production own-step.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "b_core")]

from gpuwrf.coupling import boundary_apply as bd
from gpuwrf.runtime import operational_mode as op

PROD = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")
F32 = np.float32
DT, CADENCE = 54.0, 21600.0
SIDE_VARS = (("BXS", 0), ("BXE", 1), ("BYS", 2), ("BYE", 3))
# Pre-registered (BD92 r2): WRF's own response to a 1-ulp move of its REAL records reaches 33 ulp of max|qv| in the
# relax rows; the port on production leaves must stay inside that; the released pass misses by ~900.  The flow_dep
# species copy the interior, so their band error must not exceed the chain's own interior rounding (cells B43 does not
# touch: same _apply_moisture_large_step arithmetic in both arms) by more than 10 %.
RELAX_ULP_OF_MAX, SPEC_ULP_OF_MAX, FLOW_NOISE_FACTOR = 33.0, 4.0, 1.1


@dataclasses.dataclass(frozen=True)
class _St:
    fields: dict

    def __getattr__(self, name):
        try:
            return self.__dict__["fields"][name]
        except KeyError:
            return None

    def replace(self, **kw):
        return _St({**self.fields, **kw})


def _finite_or_fail(*arrays):
    for a in arrays:
        if not np.all(np.isfinite(np.asarray(a, np.float64))):
            raise FloatingPointError("non-finite operand in the B43 gate")


def _ulp_of_max(got, want):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    _finite_or_fail(got, want)
    return np.abs(got - want) / float(np.spacing(F32(max(np.abs(want).max(), 1e-30))))


def test_gate_rejects_non_finite():
    good = np.ones((2, 3, 3), F32)
    for bad in (np.nan, np.inf, -np.inf):
        broken = good.copy(); broken[1, 1, 1] = bad
        with pytest.raises(FloatingPointError):
            _ulp_of_max(broken, good)
        with pytest.raises(FloatingPointError):
            _ulp_of_max(good, broken)


def test_species_split_follows_solve_em():
    off = bd.BoundaryConfig(have_bcs_moist=False, have_bcs_scalar=False)
    assert bd.root_scalar_rk1_split(off, ("qv", "qc", "qr", "Ni", "Nr")) == (("qv",), ("qc", "qr", "Ni", "Nr"))
    moist = bd.BoundaryConfig(have_bcs_moist=True, have_bcs_scalar=False)
    assert bd.root_scalar_rk1_split(moist, ("qv", "qc", "Ni")) == (("qv", "qc"), ("Ni",))
    with pytest.raises(ValueError):
        bd.root_scalar_rk1_split(bd.BoundaryConfig(), ("qv",))
    with pytest.raises(NotImplementedError):
        bd.root_scalar_rk1_split(off, ("qv", "nwfa"))


def test_record_bracket_keeps_the_old_interval_on_the_record_step():
    """solve_em.F:378 advances dtbc before the step; the step ending on a record still uses the old interval."""
    rec = jnp.asarray(np.arange(3 * 2, dtype=F32).reshape(3, 2) ** 2 + 1.0)
    value, rate = bd._root_record_value_rate_real4(rec, CADENCE, CADENCE)
    np.testing.assert_array_equal(np.asarray(rate), (np.asarray(rec[1]) - np.asarray(rec[0])) / F32(CADENCE))
    np.testing.assert_array_equal(np.asarray(value), np.asarray(rec[0]) + F32(CADENCE) * np.asarray(rate))
    value, rate = bd._root_record_value_rate_real4(rec, CADENCE + DT, CADENCE)
    np.testing.assert_array_equal(np.asarray(rate), (np.asarray(rec[2]) - np.asarray(rec[1])) / F32(CADENCE))


def test_spec_bdy_scalar_overwrites_the_spec_zone():
    base = jnp.full((2, 6, 7), 3.0, jnp.float32)
    bdy = jnp.full((2, 6, 7), 0.25, jnp.float32)
    out = op._root_sc_with_boundary({"qv": base, "qc": base}, {"qv": bdy}, 1)
    ring = np.zeros((6, 7), bool); ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    np.testing.assert_array_equal(np.asarray(out["qv"])[:, ring], 0.25)
    np.testing.assert_array_equal(np.asarray(out["qv"])[:, ~ring], 3.25)
    np.testing.assert_array_equal(np.asarray(out["qc"]), np.asarray(base))


def _raw_records(path, nz):
    from netCDF4 import Dataset
    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        b = {s: np.asarray(ds.variables[f"QVAPOR_{s}"][:], F32) for s, _ in SIDE_VARS}
        t = {s: np.asarray(ds.variables[f"QVAPOR_BT{s[1:]}"][:], F32) for s, _ in SIDE_VARS}
    width = b["BXS"].shape[1]; side_len = max(v.shape[-1] for v in b.values())
    val = np.zeros((b["BXS"].shape[0], 4, width, nz, side_len), F32); ten = np.zeros_like(val)
    for s, i in SIDE_VARS:
        n = b[s].shape[-1]
        val[:, i, :, :, :n] = b[s][:, :, :nz]; ten[:, i, :, :, :n] = t[s][:, :, :nz]
    return val, ten


def _lap(f):
    pad = np.pad(np.asarray(f, np.float64), ((0, 0), (1, 1), (1, 1)), mode="edge")
    return pad[:, 2:, 1:-1] + pad[:, :-2, 1:-1] + pad[:, 1:-1, 2:] + pad[:, 1:-1, :-2] - 4 * pad[:, 1:-1, 1:-1]


def _shaped(seed, field, mass, frac):
    f = np.asarray(field, np.float64); rng = np.random.default_rng(seed)
    lap = _lap(mass * f); lap = lap / max(np.abs(lap).max(), 1e-30)
    base = lap + 0.5 * rng.standard_normal(f.shape) + 0.2
    return (base / np.abs(base).max() * frac * np.abs(f).max() * mass.mean() / DT).astype(F32)


@pytest.fixture(scope="module")
def prod_operands():
    if not (PROD / "wrfbdy_d01").exists():
        pytest.skip(f"PROD inputs unavailable: {PROD}")
    return _load_prod_operands()


def _load_prod_operands():
    from netCDF4 import Dataset
    from gpuwrf.integration.d02_replay import load_wrfbdy_boundary_leaves
    with Dataset(PROD / "wrfinput_d01") as ds:
        ds.set_auto_mask(False)
        g = lambda n: np.asarray(ds.variables[n][0], np.float64)
        qv, mub, mu = g("QVAPOR"), g("MUB"), g("MU")
        u, v, c1h, c2h, c1f, c2f, msfty = g("U"), g("V"), g("C1H"), g("C2H"), g("C1F"), g("C2F"), g("MAPFAC_MY")
        msfuy, msfvx = g("MAPFAC_UY"), g("MAPFAC_VX")
    nz, ny, nx = qv.shape
    run = SimpleNamespace(path=PROD, namelist={"time_control": {"interval_seconds": int(CADENCE)}, "dynamics": {}},
                          wrfinput_file=lambda d: PROD / f"wrfinput_{d}")
    leaves, _ = load_wrfbdy_boundary_leaves(
        run, SimpleNamespace(nx=nx, ny=ny, nz=nz), domain="d01", mu_total=mu + mub, mub=mub,
        metrics=SimpleNamespace(c1h=c1h, c2h=c2h, c1f=c1f, c2f=c2f, msfuy=msfuy, msfvx=msfvx))
    # wrfinput carries no hydrometeors: a real-structured cloud field (rings included) for the flow_dep species
    qc = (1e-4 * (qv / qv.max()) * (1.0 + np.sin(np.arange(nx) / 7.0))[None, None, :]).astype(F32)
    return dict(qv=qv.astype(F32), qc=qc, mu_t=(mu + mub).astype(F32), u=u.astype(F32), v=v.astype(F32),
                c1h=c1h, c2h=c2h, msfty=msfty, leaves={k: np.asarray(leaves[k]) for k in ("qv_bdy", "mu_bdy", "mub_bdy")},
                val_ten=_raw_records(PROD / "wrfbdy_d01", nz))


@pytest.fixture(scope="module")
def pristine_lib(tmp_path_factory):
    fc = os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    if not shutil.which(fc):
        pytest.skip(f"gfortran for the pristine WRF oracle not found: {fc}")
    from pristine_boundary import build
    return build(tmp_path_factory.mktemp("pristine_b43"))


def _chain(op_, lib, *, released=False):
    """One large step: (pristine WRF finals, port finals) for qv (relaxed) and qc (flow_dep)."""
    from pristine_boundary import Oracle
    qv, qc, mu_t = op_["qv"], op_["qc"], op_["mu_t"]
    nz, ny, nx = qv.shape
    cfg = bd.BoundaryConfig(update_cadence_s=CADENCE, have_bcs_moist=False, have_bcs_scalar=False)
    relaxed, flow = bd.root_scalar_rk1_split(cfg, ("qv", "qc"))
    mass_t = op_["c1h"][:, None, None] * mu_t.astype(np.float64)[None] + op_["c2h"][:, None, None]
    fields = dict(qv=qv, qc=qc)
    D = {n: _shaped(10 + i, fields[n], mass_t, 2e-4) for i, n in enumerate(fields)}
    P = {n: _shaped(40 + i, fields[n], mass_t, 5e-4) for i, n in enumerate(fields)}
    kap = {n: 1e-3 * np.abs(fields[n]).max() * mass_t.mean() / DT / np.abs(_lap(mass_t * fields[n])).max() for n in fields}
    advect = lambda n, q: (kap[n] * _lap(mass_t * np.asarray(q, np.float64))).astype(F32)
    dmu = _lap(mu_t[None].astype(np.float64))[0]; dmu = dmu / np.abs(dmu).max() * 30.0
    mu_s = [(mu_t + (s + 1) / 3.0 * dmu).astype(F32) for s in range(3)]
    ru_s = [(op_["u"] * (1 + 0.1 * s) - (0.3 if s == 1 else 0.0)).astype(F32) for s in range(3)]
    rv_s = [(op_["v"] * (1 - 0.1 * s) + (0.3 if s == 2 else 0.0)).astype(F32) for s in range(3)]
    dt_rk = (DT / 3.0, DT / 2.0, DT)
    mass3 = op_["c1h"][:, None, None] * mu_s[2].astype(np.float64)[None] + op_["c2h"][:, None, None]
    val, ten = op_["val_ten"]
    lead = DT; lower = 0
    wrf = {}
    for n in fields:
        o = Oracle(lib, (nz, ny, nx), dt=DT, dtbc=lead, dts=DT / 3.0)
        o.set("c1h", op_["c1h"]); o.set("c2h", op_["c2h"]); o.set("msftx", op_["msfty"]); o.set("msfty", op_["msfty"])
        o.set("mu", mu_t); o.set("mut", mu_t); o.set("scalar", fields[n])
        o.set("scalar_tend", (D[n].astype(np.float64) + P[n]).astype(F32))
        if n in relaxed:
            o.boundary("scalar", val[lower], ten[lower]); o.run("relax_bdy_scalar"); o.run("spec_bdy_scalar")
        o.set("sc_tend", o.field("scalar_tend", (nz, ny, nx)))
        o.set("scalar_2", fields[n]); o.set("scalar_1", fields[n]); o.set("mu_old", mu_t)
        o.set("mu_base", np.zeros_like(mu_t)); o.set("h_tendency", np.zeros_like(qv)); o.set("z_tendency", np.zeros_like(qv))
        q = fields[n]
        for s in range(3):
            o.set("advect_tend", advect(n, q)); o.set("mu_new", mu_s[s]); o.s[2] = F32(dt_rk[s])
            o.run("rk_update_scalar", tag=s + 1)
            q = o.field("scalar_2", (nz, ny, nx))
            if n in flow:
                o.set("field", q); o.set("ru", ru_s[s]); o.set("rv", rv_s[s]); o.run("flow_dep_bdy")
                q = o.field("field", (nz, ny, nx)); o.set("scalar_2", q)
        if n in relaxed:
            o.set("field", q); o.set("mu", mu_s[2]); o.run("spec_bdy_final", tag=4)
            q = o.field("field", (nz, ny, nx))
        wrf[n] = q

    real = lambda a: jnp.asarray(a, jnp.float32)
    metrics = SimpleNamespace(c1h=real(op_["c1h"]), c2h=real(op_["c2h"]), msfty=real(op_["msfty"]))
    ref = _St({"qv": real(qv), "qc": real(qc), "mu_total": real(mu_t), **{k: real(a) for k, a in op_["leaves"].items()}})
    sc = {n: real(D[n]) for n in fields}
    if not released:
        sc = op._root_sc_with_boundary(sc, bd.root_scalar_boundary_tendencies(ref, lead, metrics, DT, cfg, relaxed), 1)
    state = ref
    species = tuple(fields)
    for s in range(3):
        q_t = op._root_scalar_stage_tendencies(tuple(real(advect(n, np.asarray(getattr(state, n)))) for n in species),
                                               species, state, cfg, metrics.msfty, bounded=True, sc_tendencies=sc)
        state = op._apply_moisture_large_step(state.replace(mu_total=real(mu_s[s])), ref, q_tendencies=q_t,
                                              dt_rk=dt_rk[s], metrics=metrics, species=species)
        if not released:
            state = state.replace(**{n: bd.flow_dep_bdy(getattr(state, n), real(ru_s[s]), real(rv_s[s]), cfg) for n in flow})
    # operational_mode: post-RK physics increment, then (flag ON) the RK3-signed flow_dep, MP (skips the spec zone)
    state = state.replace(**{n: getattr(state, n) + real(DT * P[n] / mass3) for n in species})
    if released:
        port = dict(qv=np.asarray(bd._apply_3d(state.qv, state.qv_bdy, lead, DT, cfg)),
                    qc=np.asarray(bd.flow_dep_bdy(state.qc, real(op_["u"]), real(op_["v"]), cfg)))
    else:
        state = state.replace(**{n: bd.flow_dep_bdy(getattr(state, n), real(ru_s[2]), real(rv_s[2]), cfg) for n in flow})
        state = bd.root_scalar_spec_final(state, lead, metrics, cfg, relaxed)
        port = dict(qv=np.asarray(state.qv), qc=np.asarray(state.qc))
    return wrf, port


def _zones(shape):
    ny, nx = shape[-2:]
    yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    bdist = np.minimum(np.minimum(yy, ny - 1 - yy), np.minimum(xx, nx - 1 - xx))
    return bdist < 1, (bdist >= 1) & (bdist < 5), bdist >= 5


def test_root_chain_matches_pristine_wrf(prod_operands, pristine_lib):
    wrf, port = _chain(prod_operands, pristine_lib)
    spec, relax, interior = _zones(wrf["qv"].shape)
    err = _ulp_of_max(port["qv"], wrf["qv"])
    assert err[:, spec].max() <= SPEC_ULP_OF_MAX, err[:, spec].max()
    assert err[:, relax].max() <= RELAX_ULP_OF_MAX, err[:, relax].max()
    qc = _ulp_of_max(port["qc"], wrf["qc"])
    noise = qc[:, interior].max()
    assert qc[:, spec | relax].max() <= FLOW_NOISE_FACTOR * noise, (qc[:, spec | relax].max(), noise)


def test_released_end_of_step_pass_fails_the_same_gate(prod_operands, pristine_lib):
    wrf, port = _chain(prod_operands, pristine_lib, released=True)
    spec, relax, interior = _zones(wrf["qv"].shape)
    err = _ulp_of_max(port["qv"], wrf["qv"])
    assert err[:, relax].max() > 10 * RELAX_ULP_OF_MAX, err[:, relax].max()
    qc = _ulp_of_max(port["qc"], wrf["qc"])
    assert qc[:, spec].max() > 10 * FLOW_NOISE_FACTOR * qc[:, interior].max()


def _trace_root_step(monkeypatch, flag):
    from prod_inputs import prod_domains
    from gpuwrf.runtime.domain_tree import DomainTree

    hierarchy, bundles, _, _, _, carries = prod_domains()
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    nml, carry = tree.domains["d01"].namelist, carries["d01"]
    counts: dict[str, int] = {}

    def spy(module, name):
        real = getattr(module, name)

        def call(*args, **kwargs):
            counts[name] = counts.get(name, 0) + 1
            return real(*args, **kwargs)
        monkeypatch.setattr(module, name, call)

    spy(op, "root_scalar_boundary_tendencies"); spy(op, "_root_sc_with_boundary")
    spy(op, "flow_dep_bdy"); spy(bd, "root_scalar_spec_final")
    monkeypatch.setenv("GPUWRF_ROOT_SCALAR_BDY_RK1", flag)
    shapes = jax.tree.map(lambda v: jax.ShapeDtypeStruct(v.shape, v.dtype), carry)
    clock = op.build_clock_base(nml)
    try:
        jax.clear_caches()
        out = jax.eval_shape(lambda c: op._advance_chunk_fori(
            c, nml, jnp.asarray(1, jnp.int32), clock, n_steps=1, cadence=int(nml.radiation_cadence_steps)), shapes)
    finally:
        jax.clear_caches()
    assert jax.devices()[0].platform == "cpu"
    return nml, counts, [str(leaf) for leaf in jax.tree.leaves(out)]


def test_root_own_step_dispatch_on_and_off(monkeypatch):
    nml, on, avals_on = _trace_root_step(monkeypatch, "1")
    assert op._root_scalar_bdy_rk1_active(nml)
    _relaxed, flow = bd.root_scalar_rk1_split(nml.boundary_config, op._advected_scalar_species(nml) + ("Ni", "Nr"))
    # E114: the fast path ran — one RK1 boundary build, one merge per stage, flow_dep per stage + post-physics, one pin
    assert on.get("root_scalar_boundary_tendencies") == 1, on
    assert on.get("_root_sc_with_boundary") == 3, on
    assert on.get("flow_dep_bdy") == 4 * len(flow) and flow, on
    assert on.get("root_scalar_spec_final") == 1, on
    nml, off, avals_off = _trace_root_step(monkeypatch, "0")
    assert not op._root_scalar_bdy_rk1_active(nml) and off == {}, off
    assert avals_on == avals_off


def test_end_of_step_seam_pins_qv_like_spec_bdy_final(pristine_lib):
    """The production end-of-step pass (apply_lateral_boundaries with root_scalar_rk1) on the real PROD d01 carry state:
    a spec-zone qv that differs from the record target is re-pinned exactly like pristine spec_bdy_final; relax zone,
    interior and the flow_dep species are left untouched (review-b 02:09Z)."""
    from pristine_boundary import Oracle
    from prod_inputs import prod_domains
    from gpuwrf.runtime.domain_tree import DomainTree

    hierarchy, bundles, _, _, _, carries = prod_domains()
    nml = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False).domains["d01"].namelist
    state = carries["d01"].state
    cfg = nml.boundary_config
    species = tuple(n for n in ("qv", "qc", "qr", "qi", "qs", "qg", "Ni", "Nr") if getattr(state, n) is not None)
    relaxed, flow = bd.root_scalar_rk1_split(cfg, species)
    nz, ny, nx = state.qv.shape
    spec, _relax, _interior = _zones(state.qv.shape)
    qv_old = np.asarray(state.qv, np.float64) * np.where(spec, 1.01, 1.0)[None]
    st = state.replace(qv=jnp.asarray(qv_old, state.qv.dtype))
    out = bd.apply_lateral_boundaries(st, DT, DT, cfg, nml.metrics, dry_spec_only=True, positivity_floor=False,
                                      root_scalar_rk1=(relaxed, flow))
    val, ten = _raw_records(PROD / "wrfbdy_d01", nz)
    o = Oracle(pristine_lib, (nz, ny, nx), dt=DT, dtbc=DT, dts=DT / 3.0)
    o.set("c1h", np.asarray(nml.metrics.c1h)); o.set("c2h", np.asarray(nml.metrics.c2h))
    o.set("msf", np.asarray(nml.metrics.msfty)); o.set("mu", np.asarray(state.mu_total))
    o.boundary("scalar", val[0], ten[0]); o.set("field", qv_old); o.run("spec_bdy_final", tag=4)
    want = o.field("field", (nz, ny, nx))
    got = np.asarray(out.qv, np.float64)
    assert _ulp_of_max(got, want)[:, spec].max() <= SPEC_ULP_OF_MAX
    assert _ulp_of_max(qv_old, want)[:, spec].max() > 100 * SPEC_ULP_OF_MAX  # the perturbation is visible to the gate
    np.testing.assert_array_equal(got[:, ~spec], qv_old[:, ~spec])
    for name in flow:
        got_flow, old_flow = np.asarray(getattr(out, name)), np.asarray(getattr(st, name))
        _finite_or_fail(got_flow, old_flow)  # assert_array_equal treats matching NaNs as equal (E200)
        np.testing.assert_array_equal(got_flow, old_flow)
