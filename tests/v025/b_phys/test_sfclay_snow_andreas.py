"""WRF Andreas snow roughness and the entry-Noah snow-depth seam (BP80)."""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.physics import surface_layer as SL
from gpuwrf.physics.fp32 import surface_layer_real as REAL

HERE = Path(__file__).resolve().parent
PROOFS = HERE.parents[2] / "proofs/v090"
PRISTINE_SHA = "86395534a6c9bfc79dcad50094bce290eff05756777a95794b2673795f9761c3"
FIXTURE = HERE / "fixtures/sfclay_snow_swiss.json"
THRESHOLDS = {"ust": .01, "hfx": .02, "lh": .03, "qsfc": .01, "br": .05,
              "zol": .05, "mol": .05, "psim": .03, "psih": .03, "rmol": .05,
              "u10": .01, "v10": .01, "t2": .001, "th2": .001,
              "regime": 0., "znt": .001, "q2": .01}
IN_COLS = ["u", "v", "t1d", "qv", "p1d", "dz8w", "rho", "u1d2", "v1d2", "dz2w",
           "mavail", "pblh", "xland", "tsk", "psfcpa", "qcg", "snowh", "znt",
           "ust", "mol", "qsfc", "hfx", "qfx"]
OUT_COLS = ["ust", "mol", "rmol", "zol", "regime", "psim", "psih", "br",
            "flhc", "flqc", "hfx", "qfx", "lh", "qsfc", "qgh", "chs", "chs2",
            "cqs2", "ch", "wspd", "gz1oz0", "u10", "v10", "th2", "t2", "q2",
            "cpm", "wstar", "qstar", "znt"]


@pytest.fixture(scope="module")
def pristine(tmp_path_factory):
    """Fresh byte-identical WRF module and thin public-routine wrappers."""
    source = PROOFS / "module_sf_mynn_pristine.f90"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == PRISTINE_SHA
    out = tmp_path_factory.mktemp("andreas_oracle")
    compiler = os.environ.get("FC", "<USER_HOME>/miniconda3/envs/wrfbuild/bin/gfortran")
    assert Path(compiler).exists(), "pristine WRF oracle compiler required"
    env = dict(os.environ, GF=compiler, OUT_DIR=str(out), ORACLE_CPUS="24,25,28,29")
    subprocess.run(["bash", str(PROOFS / "build_oracle.sh")], env=env,
                   check=True, capture_output=True, text=True)
    wrapper = out / "andreas_wrapper.f90"
    wrapper.write_text("""subroutine bp_andreas(visc,ust,zt,zq) bind(C)
use iso_c_binding
use module_sf_mynn, only: Andreas_2002
implicit none
real(c_float), value :: visc,ust
real(c_float), intent(out) :: zt,zq
call Andreas_2002(0.1_c_float,visc,ust,zt,zq)
end subroutine bp_andreas
""")
    subprocess.run([compiler, "-O2", "-shared", "-fPIC", "-ffree-line-length-none",
                    str(PROOFS / "module_model_constants.f90"), str(source), str(wrapper),
                    "-o", str(out / "andreas.so")], cwd=out, check=True, capture_output=True)
    lib = ctypes.CDLL(str(out / "andreas.so"))
    lib.bp_andreas.argtypes = [ctypes.c_float, ctypes.c_float,
                              ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    return out / "mynn_oracle", lib


def cases():
    data = json.loads(FIXTURE.read_text())
    assert data["scope"] == "real Swiss h1 paired surface-layer columns"
    rows = data["columns"]
    depths = np.asarray([r["snowh"] for r in rows])
    assert np.count_nonzero(depths >= .1) >= 8
    assert np.count_nonzero(depths < .1) >= 8
    return rows, data["dx_m"]


def view(rows):
    from gpuwrf.coupling.noahmp_surface_hook import _NoahMPColumnView
    n = len(rows)
    g = lambda k: jnp.asarray([r[k] for r in rows]).reshape(n, 1)
    levels = lambda k1, k2: jnp.stack([g(k1), g(k2)], axis=-1)
    pressure = levels("p1d", "p1d")
    temperature = levels("t1d", "t1d")
    theta = temperature * (100000. / pressure) ** (287. / 1004.5)
    return _NoahMPColumnView(
        u=levels("u", "u1d2"), v=levels("v", "v1d2"), theta=theta,
        qv=levels("qv", "qv"), qc=jnp.zeros_like(theta), p=pressure,
        dz=levels("dz8w", "dz2w"), t_skin=g("tsk"), soil_moisture=g("mavail"),
        xland=g("xland"), lakemask=jnp.zeros((n, 1)), mavail=g("mavail"),
        roughness_m=g("znt"), ustar=g("ust"), t_air=g("t1d"), psfc=g("psfcpa"),
        rho=levels("rho", "rho"), mol=g("mol"), hfx=g("hfx"), qfx=g("qfx"),
        qsfc=g("qsfc"), pblh=g("pblh"), dx_m=3000.)


def oracle(exe, rows, dx):
    # Decimalize the same REAL32 operands used by operational WRF.
    lines = [f"{len(rows)} 2 1 0 0 0 {dx}"]
    lines += [" ".join(repr(float(np.float32(row[k]))) for k in IN_COLS) for row in rows]
    p = subprocess.run([str(exe)], input="\n".join(lines) + "\n", capture_output=True,
                       text=True, check=True)
    values = [line.split() for line in p.stdout.splitlines() if line and not line.startswith("#")]
    assert len(values) == len(rows)
    return {k: np.asarray([float(v[j + 1]) for v in values]) for j, k in enumerate(OUT_COLS)}


def diagnostics_arrays(diag):
    result = {k: np.asarray(getattr(diag, k)).reshape(-1)
              for k in THRESHOLDS if k != "ust"}
    result["ust"] = np.asarray(diag.fluxes.ustar).reshape(-1)
    result["q2"] = np.asarray(diag.q2).reshape(-1)
    result["cqs2"] = np.asarray(diag.cqs2).reshape(-1)
    return result


def assert_surface_parity(actual, expected):
    failures = []
    for name, limit in THRESHOLDS.items():
        delta = np.abs(actual[name] - expected[name])
        relative = delta / np.maximum(np.abs(expected[name]), 1.e-12)
        bad = (relative > limit) & (delta >= 1.e-6)
        if np.any(bad):
            failures.append((name, int(np.count_nonzero(bad)), float(relative.max())))
    assert not failures, failures


@pytest.mark.parametrize("native", [False, True])
def test_real_swiss_snow_and_bare_match_pristine(pristine, monkeypatch, native):
    rows, dx = cases()
    monkeypatch.setattr(REAL, "_NATIVE_REAL", native)
    state = view(rows)._replace(dx_m=dx)
    depth = jnp.asarray([r["snowh"] for r in rows]).reshape(len(rows), 1)
    result = SL.surface_layer_with_diagnostics(state, snowh=depth)
    assert_surface_parity(diagnostics_arrays(result), oracle(pristine[0], rows, dx))
    for leaf in jax.tree.leaves(result):
        assert np.isfinite(np.asarray(leaf)).all()


def test_andreas_lengths_match_unmodified_real_routine(pristine):
    rows, _ = cases()
    # Real snow-column UST/T plus a labelled cap stress. The full SL gate uses
    # real geometry/forcing; this routine gate isolates all coefficients/Zq.
    visc = np.asarray([1.326e-5 * (1 + 6.542e-3 * (r["t1d"] - 273.15)
                       + 8.301e-6 * (r["t1d"] - 273.15) ** 2
                       - 4.84e-9 * (r["t1d"] - 273.15) ** 3) for r in rows], np.float32)
    ust = np.asarray([r["ust"] for r in rows], np.float32)
    visc = np.r_[visc, np.float32(1.326e-5)]
    ust = np.r_[ust, np.float32(3.)]
    expected = []
    for nu, u in zip(visc, ust):
        zt, zq = ctypes.c_float(), ctypes.c_float()
        pristine[1].bp_andreas(float(nu), float(u), ctypes.byref(zt), ctypes.byref(zq))
        expected.append([zt.value, zq.value])
    zt, zq = SL._andreas_2002(jnp.asarray(visc), jnp.asarray(ust))
    np.testing.assert_allclose(np.stack([zt, zq], axis=-1), expected, rtol=5.e-6, atol=1.e-10)
    assert np.any(np.asarray(zt) != np.asarray(zq))


def test_noah_adapter_reads_entry_snow_depth(monkeypatch):
    from gpuwrf.physics import noahmp_coupler as N
    spec = importlib.util.spec_from_file_location("bp80_noah_fixture", HERE.parents[1] / "test_noahmp_coupler.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    state, land, static, radiation, clock = fixture._build()
    depth = jnp.full_like(land.snowh, .24)
    land = land.replace(snowh=depth)

    class Captured(Exception):
        pass

    def capture(_state, *, first_timestep=False, snowh=None):
        assert first_timestep is True
        assert snowh is depth
        raise Captured

    monkeypatch.setattr(N, "surface_layer_with_diagnostics", capture)
    with pytest.raises(Captured):
        N.noahmp_surface_adapter(state, land, static, radiation=radiation,
                                clock=clock, dt=18., first_timestep=True)


@pytest.mark.parametrize("native", [False, True])
def test_public_entry_forwards_snow_depth(monkeypatch, native):
    marker, depth = object(), object()
    monkeypatch.setattr(REAL, "_NATIVE_REAL", native)

    def capture(state, first, dtype, *, snowh=None):
        assert snowh is depth
        assert first is True
        assert dtype == (jnp.float32 if native else jnp.float64)
        return marker

    monkeypatch.setattr(SL, "_surface_layer_impl", capture)
    assert SL.surface_layer_with_diagnostics(object(), first_timestep=True, snowh=depth) is marker


def test_snow_native_has_no_float64_compute():
    import sys
    sys.path.insert(0, str(HERE))
    from precision_inventory import inventory
    rows, dx = cases()
    state = view(rows)._replace(dx_m=dx)
    depth = jnp.asarray([r["snowh"] for r in rows]).reshape(len(rows), 1)
    hlo = jax.jit(lambda st, snow: REAL.surface_layer_with_diagnostics_real(st, snowh=snow)).lower(
        state, depth).compiler_ir("hlo").as_hlo_text()
    _, nodes = inventory(hlo, Path.cwd())
    compute = [n for n in nodes if n["category"] == "compute"]
    assert not compute, compute


@pytest.mark.parametrize("native", [False, True])
def test_real_canary_explicit_zero_snow_is_byte_unchanged(monkeypatch, native):
    fixture = json.loads((HERE / "fixtures/sfclay_snow_canary.json").read_text())
    monkeypatch.setattr(REAL, "_NATIVE_REAL", native)
    for arm in fixture["arms"].values():
        rows = arm["columns"]
        assert all(row["snowh"] == 0. for row in rows)
        state = view(rows)._replace(dx_m=arm["dx_m"])
        implicit = SL.surface_layer_with_diagnostics(state)
        explicit = SL.surface_layer_with_diagnostics(state, snowh=jnp.zeros_like(state.xland))
        for before, after in zip(jax.tree.leaves(implicit), jax.tree.leaves(explicit)):
            assert np.asarray(before).dtype == np.asarray(after).dtype
            assert np.asarray(before).tobytes() == np.asarray(after).tobytes()
