"""M2 fused vertical-implicit CPU gates (sprint 2026-09-17, contract step 3).

Pre-registered accuracy gate (declared BEFORE any device run, per contract):

* The fused kernels replicate the reference OPERATION ORDER, but two different
  XLA compilation contexts (reference vs Pallas-interpreter) make independent
  FMA-contraction / folding decisions.  Measured repeatedly on this tree:
  residuals are <= 2 ulp of the output scale (see the econ hoist result
  ``proofs/v025/econ/compare_e1_applied_anchor_VIOLATION.json`` on main: the
  same phenomenon drifts full RK-step outputs much further).  Bitwise equality
  is therefore claimed ONLY where it is actually observed stable (structural
  zeros, ``a`` rows in most configurations) and NOT required by the gate.
* GATE (all outputs, all fixtures):
  ``max |fused - reference| <= 1e-12 * max(1, max |reference|)``.
  This sits ~6 orders above the observed ulp residuals (<= 5e-14 on w, whose
  scale is ~1e1) and ~9 orders below any real reordering bug (a single sign
  error during development showed up at 2e-1; a top-face double-count showed
  up at 5e-3).  The bitwise fraction is reported per test with a loose
  >=50% tripwire: empirically contraction flips up to ~17% of cells by 1 ulp
  on some outputs, so bitwise equality is NOT a valid gate here.
* The device bake-off inherits exactly this envelope (no escalation).

Fixtures: random small grids (16x16 / 8x16 columns, all >= TX=128) plus the
production FAST d01 shape 44x70x120, and the REAL FAST advance-w savepoint
coefficients (a/alpha/gamma from ``proofs/v025/m0/pallas_fast_savepoint_*``)
for the solve-only comparison.  A dedicated ill-conditioned fixture pins the
cancellation behaviour of the Thomas sweeps.
"""

from __future__ import annotations

import os
import types

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients  # noqa: E402
from gpuwrf.dynamics.core.advance_w import advance_w_wrf  # noqa: E402
from gpuwrf.kernels.fused_vertical_implicit import (  # noqa: E402
    TX,
    advance_w_pallas,
    calc_coef_w_pallas,
)


@pytest.fixture(autouse=True)
def _legacy_fp64_dycore(monkeypatch):
    """These tests pin the fp64 legacy dycore; GPUWRF_DYN_REAL_ALL (WRF REAL consumers)
    has its own pristine REAL4 gates (tests/v025/b_diff). Flag-aware for the RC default flip."""
    monkeypatch.setenv("GPUWRF_DYN_REAL_ALL", "0")
    monkeypatch.setenv("GPUWRF_CARRY_REAL_ALL", "0")  # the v0.3 release pairs them; CARRY_REAL_ALL alone is refused


#: Pre-registered envelope multiplier (see module docstring).
TOL_ULP_SCALE = 1.0e-12
#: Loose bitwise-fraction tripwire.  EMPIRICAL: compiler contraction flips up
#: to ~17% of cells by 1 ulp on some outputs (a rows), so 0.9 failed while the
#: envelope passed comfortably.  The ENVELOPE is the reorder gate (a genuine
#: reordering blows it by ~9 orders); this only catches gross pattern shifts.
BITWISE_FRACTION_FLOOR = 0.5


def _reject_misconfigured_env() -> None:
    if os.environ.get("GPUWRF_ADVANCE_W_SAFE_FLOORS", "0") == "1":
        pytest.fail("GPUWRF_ADVANCE_W_SAFE_FLOORS=1 is outside the fused kernels' contract")


def _gate(name: str, ref, got) -> float:
    """Assert the pre-registered envelope; return the bitwise fraction."""

    r = np.asarray(ref)
    g = np.asarray(got)
    assert r.shape == g.shape
    assert np.isfinite(g).all(), f"{name}: non-finite fused output"
    scale = max(1.0, float(np.abs(r).max()))
    d = float(np.abs(r - g).max())
    frac = float(np.mean(r == g))
    assert d <= TOL_ULP_SCALE * scale, (
        f"{name}: max|d|={d:.3e} > {TOL_ULP_SCALE:.0e}*{scale:.3e}"
    )
    assert frac >= BITWISE_FRACTION_FLOOR, f"{name}: bitwise fraction {frac:.4f} < floor"
    return frac


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _metrics(nz: int, seed: int):
    rng = np.random.default_rng(seed)
    return dict(
        c1h=jnp.asarray(1.0 + 0.05 * rng.standard_normal(nz)),
        c2h=jnp.asarray(np.linspace(0.0, 100.0, nz)),
        c1f=jnp.asarray(1.0 + 0.05 * rng.standard_normal(nz + 1)),
        c2f=jnp.asarray(np.linspace(0.0, 100.0, nz + 1)),
        rdn=jnp.asarray(40.0 + 2.0 * rng.standard_normal(nz)),
        rdnw=jnp.asarray(40.0 + 2.0 * rng.standard_normal(nz)),
    )


def _coef_inputs(nz: int, ny: int, nx: int, seed: int, *, dry: bool):
    rng = np.random.default_rng(seed)
    mut = jnp.asarray(1e4 * (1.0 + 0.05 * rng.standard_normal((ny, nx))))
    if dry:
        cqw = jnp.ones((nz + 1, ny, nx)).at[0].set(0.0).at[nz].set(0.0)
    else:
        cqw = jnp.asarray(0.5 + np.abs(rng.standard_normal((nz + 1, ny, nx))))
    c2a = jnp.asarray(1.0 + 0.1 * rng.standard_normal((nz, ny, nx)))
    return mut, cqw, c2a


def _advance_kwargs(nz: int, ny: int, nx: int, seed: int, **cfg):
    """Production-shaped random fixture in the style of tests/test_v015."""

    rng = np.random.default_rng(seed)
    f3 = lambda nl: jnp.asarray(1.0 + 0.01 * rng.standard_normal((nl, ny, nx)))
    f2 = lambda: jnp.asarray(1.0 + 0.01 * rng.standard_normal((ny, nx)))
    kw = dict(
        w=f3(nz + 1), rw_tend=jnp.asarray(0.001 * f3(nz + 1)),
        ww=jnp.asarray(0.01 * f3(nz + 1)),
        u=jnp.asarray(1.0 + 0.01 * rng.standard_normal((nz, ny, nx + 1))),
        v=jnp.asarray(1.0 + 0.01 * rng.standard_normal((nz, ny + 1, nx))),
        mu_work=jnp.asarray(0.01 * f2()), mut=jnp.asarray(1e4 * f2()),
        muave=jnp.asarray(0.01 * f2()), muts=jnp.asarray(1e4 * f2()),
        t_2ave=f3(nz), t_2=f3(nz), t_1=f3(nz),
        ph=f3(nz + 1), ph_1=f3(nz + 1), phb=jnp.asarray(1e3 * f3(nz + 1)),
        ph_tend=jnp.asarray(0.001 * f3(nz + 1)), ht=jnp.asarray(100.0 * f2()),
        c2a=f3(nz), cqw=f3(nz + 1), alt=f3(nz),
        a=jnp.asarray(0.1 * f3(nz + 1)), alpha=jnp.asarray(0.9 * f3(nz + 1)),
        gamma=jnp.asarray(0.1 * f3(nz + 1)),
        c1h=jnp.asarray(np.linspace(1.0, 0.1, nz)),
        c2h=jnp.asarray(np.linspace(0.0, 100.0, nz)),
        c1f=jnp.asarray(np.linspace(1.0, 0.1, nz + 1)),
        c2f=jnp.asarray(np.linspace(0.0, 100.0, nz + 1)),
        rdnw=jnp.asarray(np.full(nz, 44.0)), rdn=jnp.asarray(np.full(nz, 44.0)),
        fnm=jnp.asarray(np.full(nz, 0.5)), fnp=jnp.asarray(np.full(nz, 0.5)),
        cf1=jnp.float64(1.5), cf2=jnp.float64(-0.5), cf3=jnp.float64(0.0),
        msftx=f2(), msfty=f2(), w_save=jnp.asarray(0.01 * f3(nz + 1)),
        rdx=1.0 / 9000.0, rdy=1.0 / 9000.0, dts=5.4, epssm=0.5,
        dampcoef=0.2, zdamp=5000.0,
    )
    kw.update(cfg)
    return kw


# --------------------------------------------------------------------------- #
# kernel A: calc_coef_w                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nz,seed,dry", [
    (8, 3, True), (8, 4, False), (44, 5, True), (2, 7, False), (1, 6, True),
])
def test_calc_coef_w_envelope(nz: int, seed: int, dry: bool) -> None:
    _reject_misconfigured_env()
    ny = nx = 16
    m = _metrics(nz, seed)
    mut, cqw, c2a = _coef_inputs(nz, ny, nx, seed, dry=dry)
    ref = calc_coef_w_wrf_coefficients(mut, types.SimpleNamespace(**m), dt=5.4,
                                       epssm=0.5, top_lid=False, cqw=cqw, c2a=c2a)
    got = calc_coef_w_pallas(mut, m["c1h"], m["c2h"], m["c1f"], m["c2f"],
                             m["rdn"], m["rdnw"], dt=5.4, epssm=0.5,
                             top_lid=False, cqw=cqw, c2a=c2a)
    for name, r, g in zip(("a", "alpha", "gamma"), ref, got):
        frac = _gate(f"calc_coef_w nz={nz} seed={seed} dry={dry} {name}", r, g)
        print(f"calc_coef_w nz={nz} {name}: bitwise fraction {frac:.4f}")


def test_calc_coef_w_fast_shape_and_lid() -> None:
    _reject_misconfigured_env()
    nz, ny, nx = 44, 70, 120
    m = _metrics(nz, 8)
    mut, cqw, c2a = _coef_inputs(nz, ny, nx, 8, dry=True)
    for top_lid in (False, True):
        ref = calc_coef_w_wrf_coefficients(mut, types.SimpleNamespace(**m), dt=5.4,
                                           epssm=0.5, top_lid=top_lid, cqw=cqw,
                                           c2a=c2a)
        got = calc_coef_w_pallas(mut, m["c1h"], m["c2h"], m["c1f"], m["c2f"],
                                 m["rdn"], m["rdnw"], dt=5.4, epssm=0.5,
                                 top_lid=top_lid, cqw=cqw, c2a=c2a)
        for name, r, g in zip(("a", "alpha", "gamma"), ref, got):
            _gate(f"calc_coef_w FAST lid={top_lid} {name}", r, g)


# --------------------------------------------------------------------------- #
# kernel B: advance_w                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cfg", [
    dict(damp_opt=3, w_damping=1, top_lid=True),
    dict(damp_opt=0, w_damping=0, top_lid=False),
    dict(damp_opt=3, w_damping=0, top_lid=False),
    dict(damp_opt=0, w_damping=1, top_lid=True),
])
def test_advance_w_envelope_small(cfg: dict) -> None:
    _reject_misconfigured_env()
    kw = _advance_kwargs(8, 16, 16, 11, **cfg)
    ref = advance_w_wrf(**kw)
    got = advance_w_pallas(**kw)
    for name, r, g in zip(("w", "ph", "t_2ave"), ref, got):
        _gate(f"advance_w {cfg} {name}", r, g)


def test_advance_w_envelope_fast_shape() -> None:
    """Production FAST d01 shape, production configuration (damp3 + w_damp)."""

    _reject_misconfigured_env()
    kw = _advance_kwargs(44, 70, 120, 15, damp_opt=3, w_damping=1, top_lid=False)
    ref = advance_w_wrf(**kw)
    got = advance_w_pallas(**kw)
    for name, r, g in zip(("w", "ph", "t_2ave"), ref, got):
        _gate(f"advance_w FAST {name}", r, g)


def test_advance_w_jit_trace_matches_eager() -> None:
    """The jit path (how production would call it) must equal the eager path."""

    _reject_misconfigured_env()
    kw = _advance_kwargs(8, 16, 16, 11, damp_opt=3, w_damping=1, top_lid=False)
    eager = advance_w_pallas(**kw)
    is_arr = lambda v: hasattr(v, "shape") and v.shape != ()
    scal = {k: v for k, v in kw.items() if not is_arr(v)}
    arr = {k: v for k, v in kw.items() if is_arr(v)}
    traced = jax.jit(lambda **a: advance_w_pallas(**a, **scal))(**arr)
    for name, r, t in zip(("w", "ph", "t_2ave"), eager, traced):
        assert np.array_equal(np.asarray(r), np.asarray(t)), name


def test_advance_w_real_savepoint_coefficients() -> None:
    """Solve-only check on the REAL FAST a/alpha/gamma savepoint (if present).

    Uses the immutable real FAST savepoint prepared for the §11 Pallas spike
    (state_source_is_real).  The savepoint rhs is reused as the w field; the
    remaining advance_w inputs keep the synthetic fixture (the solve is what
    the real coefficients condition).  Skipped cleanly if <DATA_ROOT> is absent.
    """

    _reject_misconfigured_env()
    payload = "<DATA_ROOT>/wrf_gpu2/v025/m0/pallas/advance_w_fast_v1.npz"
    if not os.path.exists(payload):
        pytest.skip("real FAST savepoint not mounted")
    with np.load(payload, allow_pickle=False) as archive:
        a = jnp.asarray(archive["a"])
        alpha = jnp.asarray(archive["alpha"])
        gamma = jnp.asarray(archive["gamma"])
        rhs = jnp.asarray(archive["rhs"])
    nzf, ny, nx = (int(s) for s in rhs.shape)
    kw = _advance_kwargs(nzf - 1, ny, nx, 21, damp_opt=3, w_damping=1, top_lid=False)
    kw["a"], kw["alpha"], kw["gamma"], kw["w"] = a, alpha, gamma, rhs
    ref = advance_w_wrf(**kw)
    got = advance_w_pallas(**kw)
    for name, r, g in zip(("w", "ph", "t_2ave"), ref, got):
        _gate(f"advance_w real-savepoint {name}", r, g)


def test_advance_w_ill_conditioned_columns() -> None:
    """Near-cancellation columns: the envelope must hold where it matters.

    alpha close to the pivot reciprocal (a+gamma ~ 1) makes the Thomas sweep
    cancellation-heavy; this is the fixture class the 3.4e-6 fp32 envelope in
    cancellation_map.json comes from.  The fused kernels must stay inside the
    fp64 value-scaled envelope here.
    """

    _reject_misconfigured_env()
    nz, ny, nx = 16, 16, 16
    kw = _advance_kwargs(nz, ny, nx, 31, damp_opt=0, w_damping=0, top_lid=False)
    rng = np.random.default_rng(99)
    a = jnp.asarray(-0.9 - 0.05 * np.abs(rng.standard_normal((nz + 1, ny, nx))))
    gamma = jnp.asarray(-0.9 - 0.05 * np.abs(rng.standard_normal((nz + 1, ny, nx))))
    diag = 1.0 + np.abs(np.asarray(a)) + np.abs(np.asarray(gamma))
    alpha = jnp.asarray(1.0 / diag + 1e-13 * rng.standard_normal((nz + 1, ny, nx)))
    kw["a"], kw["alpha"], kw["gamma"] = a, alpha, gamma
    ref = advance_w_wrf(**kw)
    got = advance_w_pallas(**kw)
    for name, r, g in zip(("w", "ph", "t_2ave"), ref, got):
        _gate(f"advance_w ill-conditioned {name}", r, g)


# --------------------------------------------------------------------------- #
# structure / contract gates                                                    #
# --------------------------------------------------------------------------- #
def test_single_pallas_dispatch_per_operator() -> None:
    """Structural launch claim, CPU-verifiable half: each fused wrapper is ONE
    pallas_call dispatch (the device half — actual kernel count — is the
    bake-off harness's job).  The reference operators lower to a long fusion
    chain plus two scan ``while`` loops per advance_w call; the fused jaxpr
    must contain exactly one pallas primitive and no scan/while of its own.
    """

    kw = _advance_kwargs(8, 16, 16, 11, damp_opt=3, w_damping=1, top_lid=False)
    is_arr = lambda v: hasattr(v, "shape") and v.shape != ()
    scal = {k: v for k, v in kw.items() if not is_arr(v)}
    arr = {k: v for k, v in kw.items() if is_arr(v)}

    fused_jaxpr = jax.make_jaxpr(lambda **a: advance_w_pallas(**a, **scal))(**arr)
    names = [str(eq.primitive) for eq in fused_jaxpr.eqns]
    n_pallas = sum(1 for n in names if "pallas" in n)
    n_scan = sum(1 for n in names if n in ("scan", "while"))
    assert n_pallas == 1, f"expected exactly 1 pallas dispatch, got {n_pallas}"
    assert n_scan == 0, f"fused wrapper must carry no scan/while, got {n_scan}"

    m = _metrics(8, 3)
    mut, cqw, c2a = _coef_inputs(8, 16, 16, 3, dry=True)
    coef_jaxpr = jax.make_jaxpr(
        lambda mut, cqw, c2a: calc_coef_w_pallas(
            mut, m["c1h"], m["c2h"], m["c1f"], m["c2f"], m["rdn"], m["rdnw"],
            dt=5.4, epssm=0.5, top_lid=False, cqw=cqw, c2a=c2a)
    )(mut, cqw, c2a)
    names = [str(eq.primitive) for eq in coef_jaxpr.eqns]
    assert sum(1 for n in names if "pallas" in n) == 1
    assert not any(n in ("scan", "while") for n in names)


def test_module_source_has_no_host_transfers() -> None:
    """Static scan: no device->host idiom in the fused module (CPU proxy)."""

    import re
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "src/gpuwrf/kernels/fused_vertical_implicit.py"
    text = path.read_text()
    idioms = (
        r"\.item\(", r"jax\.device_get\(", r"(?<![A-Za-z_.])np\.asarray\(",
        r"block_until_ready\(", r"\.to_py\(", r"float\(\s*(?:jnp|jax)",
    )
    for pattern in idioms:
        assert not re.search(pattern, text), f"host-transfer idiom present: {pattern}"


def test_unsupported_configurations_raise() -> None:
    _reject_misconfigured_env()
    kw = _advance_kwargs(8, 16, 16, 11, damp_opt=0, w_damping=0, top_lid=False)
    with pytest.raises(ValueError, match="w_damping"):
        advance_w_pallas(**{**kw, "w_damping": 2})
    with pytest.raises(ValueError, match="w_damp_on"):
        advance_w_pallas(**{**kw, "w_damping": 1, "w_damp_on": 0.5})


def test_rejects_safe_floors_env(monkeypatch) -> None:
    monkeypatch.setenv("GPUWRF_ADVANCE_W_SAFE_FLOORS", "1")
    kw = _advance_kwargs(8, 16, 16, 11, damp_opt=0, w_damping=0, top_lid=False)
    with pytest.raises(RuntimeError, match="SAFE_FLOORS"):
        advance_w_pallas(**kw)


def test_window_geometry_requirements() -> None:
    """NCOL >= TX is a hard requirement; tiny grids must fail loudly."""

    _reject_misconfigured_env()
    kw = _advance_kwargs(8, 8, 8, 11, damp_opt=0, w_damping=0, top_lid=False)
    assert 8 * 8 < TX
    with pytest.raises(ValueError, match="columns"):
        advance_w_pallas(**kw)
    m = _metrics(8, 3)
    mut, cqw, c2a = _coef_inputs(8, 8, 8, 3, dry=True)
    with pytest.raises(ValueError, match="columns"):
        calc_coef_w_pallas(mut, m["c1h"], m["c2h"], m["c1f"], m["c2f"],
                           m["rdn"], m["rdnw"], dt=5.4, cqw=cqw, c2a=c2a)
