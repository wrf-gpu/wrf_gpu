"""BD95: GPUWRF_NEST_SCALAR_SPEC_FINAL — WRF ``spec_bdy_final`` for nested moist/scalar species.

Pristine solve_em.F ends every step on a nested domain with spec_bdy_final for every moist species
(:4714-4733, ``IF im == P_QV .OR. config_flags%nested``) and every scalar (``IF config_flags%nested``):
the spec zone becomes REAL ``(bdy + dtbc*bdy_tend)/(c1h*muts + c2h)`` (share/module_bc.F:2066-2216), so
the ring has no memory.  The released live nest skipped it, and ring 0 kept residues: subnormal "dust" and
negative qc/qr/Nr where CPU-WRF has exact 0 (F1P d02/d03).

Operands are REAL d03 values of the WN3 0227 CPU-WRF reference: the ring-0 rows of two hourly frames give
the coupled record pair (real hydrometeor structure, exact zeros), the frames give muts/c1h/c2h.  The
pristine routine runs unchanged (pristine_boundary.py, nested=True).  CPU RN32 emulation is exact for normal
results; subnormal results are masked and counted (E117).
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "b_core")]

from gpuwrf.coupling import boundary_apply as bd

CPU = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
FRAMES = ("2026-02-28_17:00:00", "2026-02-28_18:00:00")
WRF_NAMES = (("qv", "QVAPOR"), ("qc", "QCLOUD"), ("qr", "QRAIN"), ("qi", "QICE"), ("qs", "QSNOW"),
             ("qg", "QGRAUP"), ("Ni", "QNICE"), ("Nr", "QNRAIN"))
F32 = np.float32
CHILD_DT, CADENCE = 6.0, 18.0  # WN3 d03 dt and its parent (d02) step = the live-nest record interval
DUST = F32(1.401298464324817e-45)


@dataclass(frozen=True)
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
            raise FloatingPointError("non-finite operand in the BD95 gate")


def _bitwise_normal(got, want, cells):
    """Bitwise on normal results; subnormal pristine results are masked (E117) and counted.

    E200: both FULL arrays must be finite before any mask is applied."""
    _finite_or_fail(got, want)
    got, want = np.asarray(got, F32)[:, cells], np.asarray(want, F32)[:, cells]
    normal = (want == 0) | (np.abs(want) >= np.finfo(F32).tiny)
    return int((~normal).sum()), int((got.view(np.uint32) != want.view(np.uint32))[normal].sum())


def test_gate_rejects_non_finite():
    """NaN/+-Inf inside AND outside the compared cells fail the gate, in either operand."""
    good = np.ones((2, 4, 4), F32)
    cells, _ = _zones(4, 4)  # ring 0 compared, the 2x2 interior not
    for bad in (np.nan, np.inf, -np.inf):
        for where in ((1, 0, 2), (1, 2, 2)):  # spec-zone cell, interior cell
            assert cells[where[1:]] == (where[1] == 0)
            broken = good.copy(); broken[where] = bad
            with pytest.raises(FloatingPointError):
                _bitwise_normal(broken, good, cells)
            with pytest.raises(FloatingPointError):
                _bitwise_normal(good, broken, cells)
            with pytest.raises(FloatingPointError):
                _finite_or_fail(broken)


def _zones(ny, nx):
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    ring = np.minimum(np.minimum(y, ny - 1 - y), np.minimum(x, nx - 1 - x))
    return ring == 0, ring


def _mass(c1, c2, mu):
    return (c1[:, None, None] * mu[None]).astype(F32) + c2[:, None, None]  # numpy REAL: each op rounded


@pytest.fixture(scope="module")
def operands():
    if not (CPU / f"wrfout_d03_{FRAMES[1]}").exists():
        pytest.skip(f"WN3 0227 CPU-WRF reference unavailable: {CPU}")
    from netCDF4 import Dataset

    frames = []
    for stamp in FRAMES:
        with Dataset(CPU / f"wrfout_d03_{stamp}") as ds:
            ds.set_auto_mask(False)
            frames.append({key: np.asarray(ds.variables[key][0], F32)
                           for key in ("MU", "MUB", "C1H", "C2H", *(w for _, w in WRF_NAMES))})
    for fr in frames:  # E200: every raw operand finite before any mass/coupling arithmetic
        _finite_or_fail(*fr.values())
    nz, ny, nx = frames[0]["QVAPOR"].shape
    side_len = max(nx, ny) + 1
    c1, c2 = frames[0]["C1H"], frames[0]["C2H"]
    records = {}
    for name, wrf in WRF_NAMES:
        rec = np.zeros((2, 4, 1, nz, side_len), F32)
        for t, fr in enumerate(frames):
            coupled = fr[wrf] * _mass(c1, c2, fr["MU"] + fr["MUB"])  # REAL mass_weight of WRF's own ring
            for side, strip in enumerate((coupled[:, :, 0], coupled[:, :, -1], coupled[:, 0, :], coupled[:, -1, :])):
                rec[t, side, 0, :, :strip.shape[-1]] = strip
        # X-side corner entries are never read by WRF (Y sides own the corners): make them visible to the gate.
        rec[:, :2, :, :, 0] *= F32(1.5); rec[:, :2, :, :, ny - 1] *= F32(1.5)
        rec[:, :2, :, :, 0] += F32(1e-6); rec[:, :2, :, :, ny - 1] += F32(1e-6)
        records[name] = rec
    muts = (frames[1]["MU"] + frames[1]["MUB"]).astype(F32) * F32(1.0 + 3e-6)  # end-of-step mass != record mass
    fields = {name: frames[1][wrf] for name, wrf in WRF_NAMES}
    _finite_or_fail(c1, c2, muts, *records.values(), *fields.values())
    return dict(nz=nz, ny=ny, nx=nx, c1=c1, c2=c2, muts=muts, records=records, fields=fields)


@pytest.fixture(scope="module")
def pristine_lib(tmp_path_factory):
    fc = os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    if not shutil.which(fc):
        pytest.skip(f"gfortran for the pristine WRF oracle not found: {fc}")
    from pristine_boundary import build
    return build(tmp_path_factory.mktemp("pristine_bd95"))


def _wrf_rate(rec):
    """share/interp_fcn.F::bdy_interp1: REAL difference, REAL*8 rdt, REAL tendency."""
    return ((rec[1] - rec[0]).astype(np.float64) * (1.0 / CADENCE)).astype(F32)


def _state(op_, fields):
    st = {name: jnp.asarray(fields[name]) for name, _ in WRF_NAMES}
    st.update({f"{name}_bdy": jnp.asarray(op_["records"][name]) for name, _ in WRF_NAMES})
    st["mu_total"] = jnp.asarray(op_["muts"])
    return _St(st)


def _metrics(op_):
    return SimpleNamespace(c1h=jnp.asarray(op_["c1"]), c2h=jnp.asarray(op_["c2"]))


def _cfg():
    return bd.BoundaryConfig(update_cadence_s=CADENCE, force_geopotential=False,
                             nested_frozen_wrf_boundary_bundle=True)


def _pristine(lib, op_, name, field, lead):
    from pristine_boundary import Oracle
    nz, ny, nx = op_["nz"], op_["ny"], op_["nx"]
    rec = op_["records"][name]
    o = Oracle(lib, (nz, ny, nx), nested=True, dt=CHILD_DT, dtbc=lead, dts=CHILD_DT / 3.0)
    o.set("c1h", op_["c1"]); o.set("c2h", op_["c2"]); o.set("msf", np.ones((ny, nx), F32)); o.set("mu", op_["muts"])
    o.boundary("scalar", rec[0], _wrf_rate(rec)); o.set("field", field); o.run("spec_bdy_final", tag=4)
    want = o.field("field", (nz, ny, nx))
    _finite_or_fail(want)
    return want


@pytest.mark.parametrize("position", (1, 2, 3))
def test_operator_matches_pristine_spec_bdy_final(operands, pristine_lib, position):
    """Every species, every dtbc position of a parent interval: bitwise on the spec zone, untouched elsewhere."""
    lead = position * CHILD_DT
    out = bd.nested_scalar_spec_final(_state(operands, operands["fields"]), lead, _metrics(operands), _cfg())
    spec, _ = _zones(operands["ny"], operands["nx"])
    moved = 0
    for name, _ in WRF_NAMES:
        want = _pristine(pristine_lib, operands, name, operands["fields"][name], lead)
        got = np.asarray(getattr(out, name))
        _finite_or_fail(got, want, operands["fields"][name])  # before assert_array_equal (matching NaNs pass it)
        masked, wrong = _bitwise_normal(got, want, spec)
        assert wrong == 0, (name, position, wrong)
        assert masked == 0, (name, position, masked)  # real d03 rows: no subnormal pristine results
        np.testing.assert_array_equal(got[:, ~spec], operands["fields"][name][:, ~spec])
        moved += int((want[:, spec] != operands["fields"][name][:, spec]).sum())
    assert moved > 1000  # the pin is visible: muts and dtbc differ from the frame the field came from


def test_dust_and_negative_residues_become_exact_zero(operands, pristine_lib):
    """The BD95 symptom: ring-0 cells whose records are 0 hold +-1.4e-45 / -9e-38 before the pin, exact 0 after."""
    spec, _ = _zones(operands["ny"], operands["nx"])
    fields, n_dust = {}, 0
    for name, _ in WRF_NAMES:
        f = operands["fields"][name].copy()
        zero = (operands["records"][name][1, 2, 0] == 0)  # S-side record 0 at t1 -> ring row y=0
        row = f[:, 0, :]
        sign = np.where(np.arange(row.size).reshape(row.shape) % 2 == 0, F32(1), F32(-1))
        dust = np.where(np.arange(row.size).reshape(row.shape) % 3 == 0, F32(-9e-38), sign * DUST)
        row[zero[:, :row.shape[1]]] = dust[zero[:, :row.shape[1]]]
        n_dust += int(zero[:, :row.shape[1]].sum())
        fields[name] = f
    assert n_dust > 1000
    out = bd.nested_scalar_spec_final(_state(operands, fields), CADENCE, _metrics(operands), _cfg())
    for name, _ in WRF_NAMES:
        got = np.asarray(getattr(out, name))
        want = _pristine(pristine_lib, operands, name, fields[name], CADENCE)
        _finite_or_fail(got, want, fields[name])
        zero = (operands["records"][name][1, 2, 0][:, :operands["nx"]] == 0) & (operands["records"][name][0, 2, 0][:, :operands["nx"]] == 0)
        assert np.all(want[:, 0, :][zero] == 0) and np.all(got[:, 0, :][zero] == 0), name
        assert not np.any(np.signbit(got[:, 0, :][zero])), name
        assert _bitwise_normal(got, want, spec)[1] == 0, name


def test_seam_off_is_identity_and_on_pins_only_the_spec_zone(operands, monkeypatch):
    """Production seam (apply_lateral_boundaries, live-nest end-of-step branch): OFF returns the state object
    unchanged; ON pins the spec zone and leaves relax zone + interior bitwise untouched."""
    st = _state(operands, operands["fields"])
    cfg = _cfg()
    monkeypatch.delenv("GPUWRF_NEST_SCALAR_SPEC_FINAL", raising=False)
    off = bd.apply_lateral_boundaries(st, CHILD_DT, CHILD_DT, cfg, _metrics(operands), dry_spec_only=True,
                                      positivity_floor=False)
    assert off is st
    monkeypatch.setenv("GPUWRF_NEST_SCALAR_SPEC_FINAL", "1")
    calls = []
    real = bd.nested_scalar_spec_final

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(bd, "nested_scalar_spec_final", spy)
    on = bd.apply_lateral_boundaries(st, CHILD_DT, CHILD_DT, cfg, _metrics(operands), dry_spec_only=True,
                                     positivity_floor=False)
    assert calls == [1]  # E114: the fast path ran
    direct = real(st, CHILD_DT, _metrics(operands), cfg)
    spec, _ = _zones(operands["ny"], operands["nx"])
    for name, _ in WRF_NAMES:
        got, ref = np.asarray(getattr(on, name)), operands["fields"][name]
        _finite_or_fail(got, ref)
        np.testing.assert_array_equal(got[:, ~spec], ref[:, ~spec])
        _finite_or_fail(getattr(direct, name))
        np.testing.assert_array_equal(got.view(np.uint32), np.asarray(getattr(direct, name)).view(np.uint32))
    with pytest.raises(ValueError):
        bd.apply_lateral_boundaries(st, CHILD_DT, CHILD_DT, cfg, None, dry_spec_only=True, positivity_floor=False)


def _trace_nest_step(monkeypatch, flag):
    from prod_inputs import prod_domains
    from gpuwrf.runtime import operational_mode as op
    from gpuwrf.runtime.domain_tree import DomainTree

    hierarchy, bundles, _, _, _, carries = prod_domains()
    tree = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False)
    nml, carry = tree.domains["d02"].namelist, carries["d02"]
    counts: dict[str, int] = {}
    real = bd.nested_scalar_spec_final

    def spy(*args, **kwargs):
        counts["nested_scalar_spec_final"] = counts.get("nested_scalar_spec_final", 0) + 1
        return real(*args, **kwargs)
    monkeypatch.setattr(bd, "nested_scalar_spec_final", spy)
    monkeypatch.setenv("GPUWRF_NEST_SCALAR_SPEC_FINAL", flag)
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


def test_nest_own_step_dispatch_on_and_off(monkeypatch):
    """Real PROD d02 (live nest) own-step trace: one pin per step with the key, none without; same avals."""
    from gpuwrf.runtime import operational_mode as op
    nml, on, avals_on = _trace_nest_step(monkeypatch, "1")
    assert op._nested_frozen_wrf_boundary_active(nml)
    assert on.get("nested_scalar_spec_final") == 1, on
    nml, off, avals_off = _trace_nest_step(monkeypatch, "0")
    assert off == {}, off
    assert avals_on == avals_off
