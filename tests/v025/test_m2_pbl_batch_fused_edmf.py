"""M2 family-#2 CPU gates: fused EDMF plume kernel vs the XLA reference.

Pre-registered accuracy gate (declared in
``src/gpuwrf/kernels/fused_edmf_plume.py`` BEFORE the first fused-path
execution, per contract and per the FMA-contraction precedent
``proofs/v025/econ/...VIOLATION.json`` and the sibling M2 bake-off test
``test_m2_fused_vertical_implicit.py``):

* Two different compilation contexts (XLA reference vs Pallas interpreter /
  later Triton device) make independent FMA-contraction / folding decisions
  at mul->add sites (the ``_qsat_blend`` Horner chains are exposed).
* GATE (all outputs, all fixtures):
  ``max |fused - reference| <= 1e-12 * max(1, max |reference|)``.
  Sits ~6 orders above observed ulp residuals and ~9 below any real
  reordering bug (the sibling bake-off measured reorders at 2e-1..5e-3 and
  contraction noise at <=5e-14).
* The bitwise fraction is REPORTED per fixture with the sibling's >=0.5
  tripwire (compiler contraction flips cells by 1 ulp; bitwise equality is
  NOT a valid gate across compilation contexts).  On the CPU interpreter the
  fused kernel measured FULLY bitwise on first bring-up; the envelope stays
  the gate for the device bake-off.

Structural gates: with ``GPUWRF_EDMF_FUSED_PLUME=1`` the compiled
``dmp_mf_columns`` contains ZERO while ops (the 42x16 nest lives inside the
one Pallas call); the reference path contains exactly the two whiles of the
dispatch tree (see ``test_m2_pbl_batch_dispatch_tree.py``).

Scope note: the production shape (B=8400, nz=44) is exercised at reduced B
here (the interpreter is a Python-loop reference executor); device
validation at production shape happens at integration (contract G4).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _var in (
    "GPUWRF_EDMF_FUSED_PLUME",
    "GPUWRF_MYNN_COND_NITER",
    "GPUWRF_MYNN_COND_UNROLL",
    "GPUWRF_MYNN_EDMF_LEVEL_UNROLL",
    "GPUWRF_MYNN_COLUMN_TILING",
    "GPUWRF_MYNN_SGS_CLOUD",
):
    os.environ.pop(_var, None)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from gpuwrf.physics import mynn_edmf, mynn_pbl  # noqa: E402
from gpuwrf.physics.mynn_pbl import MynnPBLColumnState  # noqa: E402

ENVELOPE = 1.0e-12
BITWISE_FRACTION_FLOOR = 0.5

_OUTPUTS = (
    "s_aw", "s_awqv", "s_awqt", "s_awqc", "s_awthl", "s_awu", "s_awv",
    "edmf_a", "edmf_qc", "edmf_qt", "edmf_thl", "maxmf", "psig_w", "active",
)


def _dmp_args(B: int, nz: int, seed: int, mode: str = "realistic") -> dict:
    """Column fixture for ``dmp_mf_columns`` (physically plausible profiles)."""
    rng = np.random.default_rng(seed)
    prof = lambda f, s: f + s * rng.random((B, nz))
    col = lambda f, s: f + s * rng.random((B,))
    zw = np.concatenate([np.zeros((B, 1)), np.cumsum(prof(30.0, 2.0), axis=-1)], axis=-1)
    if mode == "inactive":  # no surface buoyancy flux -> EDMF never activates
        fltv, flt, flq = col(0.0005, 0.0001), col(0.0005, 0.0001), col(1e-6, 1e-7)
    else:
        fltv, flt, flq = col(0.06, 0.01), col(0.05, 0.01), col(1e-5, 1e-6)
    moist = 1.0 if mode == "realistic" else 0.05
    return dict(
        sqw=jnp.asarray(prof(0.010, 0.002 * moist)),
        sqv=jnp.asarray(prof(0.009, 0.002 * moist)),
        sqc=jnp.asarray(prof(1e-4, 1e-5)),
        u=jnp.asarray(prof(3.0, 1.0)), v=jnp.asarray(prof(1.0, 0.5)),
        w=jnp.asarray(prof(0.5, 0.2)),
        th=jnp.asarray(prof(300.0, 2.0)), thl=jnp.asarray(prof(299.0, 2.0)),
        thv=jnp.asarray(prof(301.0, 2.0)),
        tk=jnp.asarray(prof(280.0, 3.0)), qke=jnp.asarray(prof(0.4, 0.1)),
        p=jnp.asarray(prof(90000.0, -300.0)), exner=jnp.asarray(prof(0.95, 0.01)),
        rho=jnp.asarray(prof(1.1, 0.05)), dz=jnp.asarray(prof(30.0, 2.0)),
        zw=jnp.asarray(zw),
        ust=jnp.asarray(col(0.2, 0.02)),
        flt=jnp.asarray(flt), fltv=jnp.asarray(fltv),
        flq=jnp.asarray(flq), flqv=jnp.asarray(flq),
        pblh=jnp.asarray(col(900.0, 50.0)), ts=jnp.asarray(col(290.0, 2.0)),
        dx=1000.0,
        xland=jnp.asarray(col(1.0, 0.4) if mode != "water" else col(2.0, 0.3)),
        dt=10.0,
    )


def _run_both(args: dict):
    os.environ["GPUWRF_EDMF_FUSED_PLUME"] = "1"
    try:
        fused = mynn_edmf.dmp_mf_columns(**args)
    finally:
        os.environ["GPUWRF_EDMF_FUSED_PLUME"] = "0"
    ref = mynn_edmf.dmp_mf_columns(**args)
    return ref, fused


def _compare(ref, fused):
    worst, bits, n = 0.0, 0, 0
    for k in _OUTPUTS:
        a, b = np.asarray(ref[k]), np.asarray(fused[k])
        denom = max(1.0, float(np.max(np.abs(a))))
        worst = max(worst, float(np.max(np.abs(a - b))) / denom)
        bits += int(np.count_nonzero(a == b))
        n += a.size
    return worst, bits / n


@pytest.mark.parametrize(
    "B,nz,seed,mode",
    [
        (6, 12, 7, "realistic"),
        (6, 12, 11, "dry"),
        (6, 12, 13, "inactive"),
        (6, 12, 17, "water"),
        (15, 8, 19, "realistic"),  # pad-exercising: 120 lanes -> pad 8
        (129, 44, 23, "realistic"),  # production nz, pad-exercising lanes
        (129, 44, 29, "dry"),
    ],
)
def test_envelope_gate_fused_vs_reference(B, nz, seed, mode) -> None:
    ref, fused = _run_both(_dmp_args(B, nz, seed, mode))
    worst, bitfrac = _compare(ref, fused)
    assert worst <= ENVELOPE, f"envelope exceeded: {worst:.3e} (mode={mode})"
    assert bitfrac >= BITWISE_FRACTION_FLOOR, (
        f"bitwise fraction {bitfrac:.3f} below tripwire (mode={mode})"
    )


def test_jitted_fused_matches_jitted_reference() -> None:
    args = _dmp_args(16, 16, 31, "realistic")
    os.environ["GPUWRF_EDMF_FUSED_PLUME"] = "1"
    try:
        fused = jax.jit(lambda **kw: mynn_edmf.dmp_mf_columns(**kw))(**args)
    finally:
        os.environ["GPUWRF_EDMF_FUSED_PLUME"] = "0"
    ref = jax.jit(lambda **kw: mynn_edmf.dmp_mf_columns(**kw))(**args)
    worst, _ = _compare(ref, fused)
    assert worst <= ENVELOPE, f"jitted envelope exceeded: {worst:.3e}"


def _jaxpr_prims(jaxpr, seen=None):
    """Primitive names recursively (incl. scan/while/cond subjaxprs)."""
    names = [e.primitive.name for e in jaxpr.eqns]
    for e in jaxpr.eqns:
        for p in e.params.values():
            sub = getattr(p, "jaxpr", None)
            if sub is not None and hasattr(sub, "eqns"):
                names += _jaxpr_prims(sub)
    return names


def test_fused_jaxpr_has_no_runtime_loops() -> None:
    """The collapse, structurally: the 42x16 scan/while nest -> ONE pallas op.

    Probed at jaxpr level (backend-independent).  HLO-level while census is
    NOT valid here: interpret-mode pallas_call lowers its own interpreter
    driver loops into the jit HLO on CPU.
    """
    args = _dmp_args(4, 12, 37, "realistic")
    # NOTE: JAX's trace cache keys on the callable object, and the env gate is
    # read at trace time -- fresh wrappers force fresh traces per branch.
    wrap = lambda: (lambda **kw: mynn_edmf.dmp_mf_columns(**kw))
    os.environ["GPUWRF_EDMF_FUSED_PLUME"] = "1"
    try:
        fused_jaxpr = jax.make_jaxpr(wrap(), return_shape=True)(**args)[0]
    finally:
        os.environ["GPUWRF_EDMF_FUSED_PLUME"] = "0"
    ref_jaxpr = jax.make_jaxpr(wrap(), return_shape=True)(**args)[0]

    ref_prims = _jaxpr_prims(ref_jaxpr)
    fused_prims = _jaxpr_prims(fused_jaxpr)
    assert "scan" in ref_prims, "reference lost the plume scan"
    assert "scan" not in fused_prims and "while" not in fused_prims, (
        f"fused jaxpr still carries runtime loops: "
        f"{sorted(set(fused_prims) & {'scan', 'while'})}"
    )
    assert any("pallas" in n for n in fused_prims), (
        f"fused jaxpr has no pallas primitive: {sorted(set(fused_prims))}"
    )


def _column_state(B: int, nz: int, seed: int) -> MynnPBLColumnState:
    """Physically plausible MYNN column state (pattern of
    tests/test_v015_mynn_sgs_cloud.py::_column)."""
    rng = np.random.default_rng(seed)
    dz = np.full((B, nz), 200.0)
    zmid = np.cumsum(dz, axis=1) - 0.5 * dz
    p = 1.0e5 * np.exp(-zmid / 8000.0)
    theta = 290.0 + 0.003 * zmid
    exner = (p / 1.0e5) ** (2.0 / 7.0)
    t = theta * exner
    es = 610.78 * np.exp(17.27 * (t - 273.15) / (t - 35.85))
    qsat = 0.622 * es / np.maximum(p - es, 1.0)
    qv = np.maximum(0.5 * qsat, 1.0e-6)
    mk = lambda a: jnp.asarray(a, jnp.float64)
    ones = lambda f, s: mk(f + s * rng.random((B, nz)))
    return MynnPBLColumnState(
        u=ones(3.0, 0.5), v=ones(1.0, 0.5), w=ones(0.3, 0.2),
        theta=mk(theta), qv=mk(qv), tke=mk(0.2 + 0.1 * rng.random((B, nz))),
        p=mk(p), rho=mk(1.1 - 1e-4 * zmid), dz=mk(dz),
        km=ones(1.0, 0.5), kh=ones(1.0, 0.5), el=ones(20.0, 5.0),
    )


_STATE_LEAVES = ("u", "v", "theta", "qv", "tke", "km", "kh", "el", "qsq")


def test_step_level_fused_matches_reference_bitwise_envelope() -> None:
    """End-to-end PBL step (sgs cloud + EDMF on): default vs fused path."""
    B, nz = 4, 20
    state = _column_state(B, nz, 41)

    def step():
        # unjitted impl: same numerics as the jitted entry (the jitted wrapper
        # only caches the trace); the env gate is read at trace time.
        return mynn_pbl._step_mynn_pbl_impl_with_pblh(
            state, 10.0, debug=False, surface=None, edmf=True, dx=1000.0)

    os.environ["GPUWRF_EDMF_FUSED_PLUME"] = "1"
    try:
        fused_state, fused_pblh = step()
    finally:
        os.environ["GPUWRF_EDMF_FUSED_PLUME"] = "0"
    ref_state, ref_pblh = step()

    worst = 0.0
    for k in _STATE_LEAVES:
        a, b = np.asarray(getattr(ref_state, k)), np.asarray(getattr(fused_state, k))
        denom = max(1.0, float(np.max(np.abs(a))))
        worst = max(worst, float(np.max(np.abs(a - b))) / denom)
    worst = max(
        worst,
        float(jnp.max(jnp.abs(ref_pblh - fused_pblh))) / max(1.0, float(jnp.max(ref_pblh))),
    )
    assert worst <= ENVELOPE, f"step-level envelope exceeded: {worst:.3e}"


def test_cond_niter_rollback_tracks_under_fused_kernel() -> None:
    """GPUWRF_MYNN_COND_NITER=50 (WRF hard cap rollback) must track in the
    fused path too -- the kernel calls the reference ``_condensation_edmf``
    verbatim, which reads the env at trace time."""
    args = _dmp_args(6, 14, 43, "realistic")
    os.environ["GPUWRF_MYNN_COND_NITER"] = "50"
    try:
        ref, fused = _run_both(args)
    finally:
        os.environ.pop("GPUWRF_MYNN_COND_NITER", None)
    worst, _ = _compare(ref, fused)
    assert worst <= ENVELOPE, f"niter=50 envelope exceeded: {worst:.3e}"
