"""Focused + adversarial tests for the v0234 momentum sixth-order diffusion fix.

Sprint `2026-07-18-v0234-s1-dyn-attribution-fable5`.  CPU-only: JAX is pinned
to the CPU platform before import and no GPU is queried, locked, or used.

Covers (contract correction lane §3):
* WRF ownership rectangles: exact zeros in rings 0-2 (u/v/w) and w top/bottom.
* No-wrap adversarial fixture: boundary-localized signal must not reach the
  opposite side (the replaced periodic operator fails exactly this).
* Literal Fortran triple-loop reference equality (masses, msf, limiter, coef).
* Coefficient scaling in dt and diff_6th_factor.
* Monotonic limiter branch behavior.
* F-B on retained real data: the production JAX lane equals the sprint's
  independent NumPy WRF-replica (which is itself validated by the F-A2
  collapse against authentic in-run WRF L1 dumps, not by self-comparison).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def _lane():
    """Lazy model import: keeps pytest collection JAX-free so co-collected
    import-hygiene tests (e.g. the Kimi discriminator's) stay valid."""

    from gpuwrf.dynamics.explicit_diffusion import (
        sixth_order_diffusion_tendency,
        wrf_sixth_order_uvw_tendf,
    )

    return sixth_order_diffusion_tendency, wrf_sixth_order_uvw_tendf

REPO = Path(__file__).resolve().parents[1]

DT = 6.0
FACTOR = 0.12
COEF = FACTOR * 0.015625 / (2.0 * DT)

NZ, NY, NX = 3, 12, 13  # mass grid; u (NZ,NY,NX+1), v (NZ,NY+1,NX), w (NZ+1,NY,NX)


def _rng_fields(seed: int = 7):
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 8.0, (NZ, NY, NX + 1))
    v = rng.normal(0.0, 8.0, (NZ, NY + 1, NX))
    w = rng.normal(0.0, 2.0, (NZ + 1, NY, NX))
    mut = rng.uniform(45000.0, 90000.0, (NY, NX))
    c1h = rng.uniform(0.6, 1.0, NZ)
    c2h = rng.uniform(0.0, 3000.0, NZ)
    c1f = rng.uniform(0.6, 1.0, NZ + 1)
    c2f = rng.uniform(0.0, 3000.0, NZ + 1)
    msf = {
        "msfux": rng.uniform(0.99, 1.01, (NY, NX + 1)),
        "msfuy": rng.uniform(0.99, 1.01, (NY, NX + 1)),
        "msfvx": rng.uniform(0.99, 1.01, (NY + 1, NX)),
        "msfvy": rng.uniform(0.99, 1.01, (NY + 1, NX)),
        "msftx": rng.uniform(0.99, 1.01, (NY, NX)),
        "msfty": rng.uniform(0.99, 1.01, (NY, NX)),
    }
    return u, v, w, mut, c1h, c2h, c1f, c2f, msf


def _call(u, v, w, mut, c1h, c2h, c1f, c2f, msf, *, dt=DT, factor=FACTOR, monotonic=True):
    _, wrf_sixth_order_uvw_tendf = _lane()
    ue, ve, we = wrf_sixth_order_uvw_tendf(
        u, v, w, mut,
        c1h=c1h, c2h=c2h, c1f=c1f, c2f=c2f,
        msfux=msf["msfux"], msfuy=msf["msfuy"],
        msfvx=msf["msfvx"], msfvy=msf["msfvy"],
        msftx=msf["msftx"], msfty=msf["msfty"],
        dt=dt, diff_6th_factor=factor, monotonic=monotonic,
    )
    return np.asarray(ue), np.asarray(ve), np.asarray(we)


# ---------------------------------------------------------------------------
# Literal Fortran reference (module_big_step_utilities_em.F:6321-6633,
# specified/nested bounds, diff_6th_slopeopt=0), 0-based translation.
# ---------------------------------------------------------------------------

def _df_pair(f, k, j, i, axis):
    if axis == 2:
        s = f[k, j]
        p0 = 10.0 * (s[i] - s[i - 1]) - 5.0 * (s[i + 1] - s[i - 2]) + (s[i + 2] - s[i - 3])
        p1 = 10.0 * (s[i + 1] - s[i]) - 5.0 * (s[i + 2] - s[i - 1]) + (s[i + 3] - s[i - 2])
        g0, g1 = s[i] - s[i - 1], s[i + 1] - s[i]
    else:
        s = f[k, :, i]
        p0 = 10.0 * (s[j] - s[j - 1]) - 5.0 * (s[j + 1] - s[j - 2]) + (s[j + 2] - s[j - 3])
        p1 = 10.0 * (s[j + 1] - s[j]) - 5.0 * (s[j + 2] - s[j - 1]) + (s[j + 3] - s[j - 2])
        g0, g1 = s[j] - s[j - 1], s[j + 1] - s[j]
    return p0, p1, g0, g1


def _lim(p0, p1, g0, g1, monotonic):
    if monotonic:
        if p0 * g0 <= 0.0:
            p0 = 0.0
        if p1 * g1 <= 0.0:
            p1 = 0.0
    return p0, p1


def _reference(u, v, w, mut, c1h, c2h, c1f, c2f, msf, monotonic=True):
    m = lambda c1, c2, k, j, i: c1[k] * mut[j, i] + c2[k]
    ue = np.zeros_like(u)
    for k in range(u.shape[0]):
        for j in range(3, u.shape[1] - 3):
            for i in range(3, u.shape[2] - 3):
                p0, p1, g0, g1 = _df_pair(u, k, j, i, 2)
                p0, p1 = _lim(p0, p1, g0, g1, monotonic)
                tx = COEF * msf["msfux"][j, i] * (
                    m(c1h, c2h, k, j, i) * p1 - m(c1h, c2h, k, j, i - 1) * p0
                )
                p0, p1, g0, g1 = _df_pair(u, k, j, i, 1)
                p0, p1 = _lim(p0, p1, g0, g1, monotonic)
                mu0 = 0.25 * (
                    m(c1h, c2h, k, j - 1, i - 1) + m(c1h, c2h, k, j - 1, i)
                    + m(c1h, c2h, k, j, i - 1) + m(c1h, c2h, k, j, i)
                )
                mu1 = 0.25 * (
                    m(c1h, c2h, k, j, i - 1) + m(c1h, c2h, k, j, i)
                    + m(c1h, c2h, k, j + 1, i - 1) + m(c1h, c2h, k, j + 1, i)
                )
                ty = COEF * msf["msfuy"][j, i] * (mu1 * p1 - mu0 * p0)
                ue[k, j, i] = (tx + ty) / msf["msfuy"][j, i]
    ve = np.zeros_like(v)
    for k in range(v.shape[0]):
        for j in range(3, v.shape[1] - 3):
            for i in range(3, v.shape[2] - 3):
                p0, p1, g0, g1 = _df_pair(v, k, j, i, 1)
                p0, p1 = _lim(p0, p1, g0, g1, monotonic)
                ty = COEF * msf["msfvy"][j, i] * (
                    m(c1h, c2h, k, j, i) * p1 - m(c1h, c2h, k, j - 1, i) * p0
                )
                p0, p1, g0, g1 = _df_pair(v, k, j, i, 2)
                p0, p1 = _lim(p0, p1, g0, g1, monotonic)
                mu0 = 0.25 * (
                    m(c1h, c2h, k, j - 1, i - 1) + m(c1h, c2h, k, j, i - 1)
                    + m(c1h, c2h, k, j - 1, i) + m(c1h, c2h, k, j, i)
                )
                mu1 = 0.25 * (
                    m(c1h, c2h, k, j - 1, i) + m(c1h, c2h, k, j, i)
                    + m(c1h, c2h, k, j - 1, i + 1) + m(c1h, c2h, k, j, i + 1)
                )
                tx = COEF * msf["msfvx"][j, i] * (mu1 * p1 - mu0 * p0)
                ve[k, j, i] = (tx + ty) / msf["msfvx"][j, i]
    we = np.zeros_like(w)
    for k in range(1, w.shape[0] - 1):
        for j in range(3, w.shape[1] - 3):
            for i in range(3, w.shape[2] - 3):
                p0, p1, g0, g1 = _df_pair(w, k, j, i, 2)
                p0, p1 = _lim(p0, p1, g0, g1, monotonic)
                mu0 = 0.5 * (m(c1f, c2f, k, j, i - 1) + m(c1f, c2f, k, j, i))
                mu1 = 0.5 * (m(c1f, c2f, k, j, i) + m(c1f, c2f, k, j, i + 1))
                tx = COEF * msf["msftx"][j, i] * (mu1 * p1 - mu0 * p0)
                p0, p1, g0, g1 = _df_pair(w, k, j, i, 1)
                p0, p1 = _lim(p0, p1, g0, g1, monotonic)
                mu0 = 0.5 * (m(c1f, c2f, k, j - 1, i) + m(c1f, c2f, k, j, i))
                mu1 = 0.5 * (m(c1f, c2f, k, j, i) + m(c1f, c2f, k, j + 1, i))
                ty = COEF * msf["msfty"][j, i] * (mu1 * p1 - mu0 * p0)
                we[k, j, i] = (tx + ty) / msf["msfty"][j, i]
    return ue, ve, we


def test_ownership_rings_exact_zero():
    fields = _rng_fields()
    ue, ve, we = _call(*fields)
    for arr in (ue, ve, we):
        ny, nx = arr.shape[1], arr.shape[2]
        assert np.all(arr[:, :3, :] == 0.0) and np.all(arr[:, ny - 3 :, :] == 0.0)
        assert np.all(arr[:, :, :3] == 0.0) and np.all(arr[:, :, nx - 3 :] == 0.0)
    assert np.all(we[0] == 0.0) and np.all(we[-1] == 0.0)
    assert np.any(ue != 0.0) and np.any(ve != 0.0) and np.any(we != 0.0)


def test_no_wrap_adversarial_boundary_localized_signal():
    u, v, w, mut, c1h, c2h, c1f, c2f, msf = _rng_fields()
    u_loc = np.zeros_like(u)
    u_loc[:, :, :3] = u[:, :, :3]  # signal ONLY in the three west columns
    ue, _, _ = _call(u_loc, np.zeros_like(v), np.zeros_like(w), mut, c1h, c2h, c1f, c2f, msf)
    # Owned faces with stencil reach into cols 0-2 are i in [3, 5] only.
    assert np.any(ue[:, :, 3:6] != 0.0)
    assert np.all(ue[:, :, 6:] == 0.0), "west-localized signal leaked eastward"
    # The replaced periodic operator wraps the same signal into the east side.
    sixth_order_diffusion_tendency, _ = _lane()
    periodic = np.asarray(
        sixth_order_diffusion_tendency(u_loc, dt=DT, diff_6th_factor=FACTOR)
    )
    assert np.any(periodic[:, :, -3:] != 0.0), (
        "adversarial fixture lost its discriminating power"
    )


def test_literal_fortran_reference_equality():
    for monotonic in (True, False):
        u, v, w, mut, c1h, c2h, c1f, c2f, msf = _rng_fields(seed=11)
        got = _call(u, v, w, mut, c1h, c2h, c1f, c2f, msf, monotonic=monotonic)
        want = _reference(u, v, w, mut, c1h, c2h, c1f, c2f, msf, monotonic=monotonic)
        for g, r in zip(got, want, strict=True):
            scale = max(1.0, float(np.abs(r).max()))
            assert float(np.abs(g - r).max()) <= 1e-12 * scale


def test_coefficient_scaling():
    fields = _rng_fields(seed=3)
    base = _call(*fields, monotonic=False)
    half_dt = _call(*fields, dt=DT / 2.0, monotonic=False)
    dbl_f = _call(*fields, factor=2.0 * FACTOR, monotonic=False)
    for b, h, d in zip(base, half_dt, dbl_f, strict=True):
        scale = max(1.0, float(np.abs(b).max()))
        assert float(np.abs(h - 2.0 * b).max()) <= 1e-12 * scale
        assert float(np.abs(d - 2.0 * b).max()) <= 1e-12 * scale


def test_monotonic_limiter_zeroes_up_gradient_fluxes():
    u, v, w, mut, c1h, c2h, c1f, c2f, msf = _rng_fields(seed=5)
    on = _call(u, v, w, mut, c1h, c2h, c1f, c2f, msf, monotonic=True)
    off = _call(u, v, w, mut, c1h, c2h, c1f, c2f, msf, monotonic=False)
    assert any(np.any(a != b) for a, b in zip(on, off, strict=True))


def test_runtime_hoists_one_uvw_bundle_before_all_rk_stages_without_observer():
    """The source-correct lane must be formed once from the step origin.

    A per-stage recomputation would silently reintroduce one of the four
    attributed defects even if the standalone stencil remained correct.
    """

    import inspect

    from gpuwrf.runtime.operational_mode import _rk_scan_step

    source = inspect.getsource(_rk_scan_step)
    build = source.index("rk1_forward_diff6_uvw = None")
    stages = source.index("def advance_stage", build)
    consume = source.index(
        "frozen_diff6_uvw_tendencies=rk1_forward_diff6_uvw", stages
    )
    assert source.count("wrf_sixth_order_uvw_tendf(") == 1
    assert build < stages < consume
    for forbidden in (
        "device_get",
        "pure_callback",
        "debug.callback",
        "io_callback",
        "host_callback",
    ):
        assert forbidden not in source[build:stages]


def test_frozen_uvw_bundle_has_production_shapes_and_is_reused_at_rk123():
    """Exercise the runtime fold, not only the standalone stencil oracle."""

    import dataclasses

    from gpuwrf.dynamics.advection import apply_halo, halo_spec
    from gpuwrf.dynamics.explicit_diffusion import wrf_sixth_order_uvw_tendf
    from gpuwrf.runtime.operational_mode import _augment_large_step_tendencies
    from tests.dynamics.test_diffopt1_smagorinsky_integration import (
        _build_grid,
        _build_state,
        _namelist,
    )

    grid = _build_grid(ny=12, nx=14, nz=5, dx=3000.0)
    state = apply_halo(_build_state(grid), halo_spec(grid))
    active = _namelist(
        grid,
        diff_6th_opt=2,
        diff_6th_factor=FACTOR,
    )
    frozen = wrf_sixth_order_uvw_tendf(
        state.u,
        state.v,
        state.w,
        state.mu_total,
        c1h=active.metrics.c1h,
        c2h=active.metrics.c2h,
        c1f=active.metrics.c1f,
        c2f=active.metrics.c2f,
        msfux=active.metrics.msfux,
        msfuy=active.metrics.msfuy,
        msfvx=active.metrics.msfvx,
        msfvy=active.metrics.msfvy,
        msftx=active.metrics.msftx,
        msfty=active.metrics.msfty,
        dt=float(active.dt_s),
        diff_6th_factor=float(active.diff_6th_factor),
        monotonic=True,
    )
    assert tuple(value.shape for value in frozen) == (
        state.u.shape,
        state.v.shape,
        state.w.shape,
    )

    disabled = dataclasses.replace(active, diff_6th_opt=0)
    for rk_step in (1, 2, 3):
        without = _augment_large_step_tendencies(
            state,
            disabled.tendencies,
            disabled,
            rk_step=rk_step,
        )
        with_frozen = _augment_large_step_tendencies(
            state,
            active.tendencies,
            active,
            rk_step=rk_step,
            frozen_diff6_uvw_tendencies=frozen,
            # Keep theta out of this momentum-only discriminator.
            frozen_diff6_theta_tendency=np.zeros_like(state.theta),
        )
        for name, expected in zip(("u", "v", "w"), frozen, strict=True):
            actual = np.asarray(getattr(with_frozen, name) - getattr(without, name))
            np.testing.assert_allclose(
                actual,
                np.asarray(expected),
                rtol=2.0e-15,
                atol=2.0e-10,
            )


def test_diff6_disabled_static_path_ignores_nan_frozen_bundle_bit_exact():
    """The new kwarg cannot contaminate released/idealized diff6-off paths."""

    from gpuwrf.dynamics.advection import apply_halo, halo_spec
    from gpuwrf.runtime.operational_mode import _augment_large_step_tendencies
    from tests.dynamics.test_diffopt1_smagorinsky_integration import (
        _build_grid,
        _build_state,
        _namelist,
    )

    grid = _build_grid(ny=9, nx=11, nz=4, dx=3000.0)
    state = apply_halo(_build_state(grid), halo_spec(grid))
    disabled = _namelist(grid, diff_6th_opt=0)
    baseline = _augment_large_step_tendencies(
        state,
        disabled.tendencies,
        disabled,
        rk_step=3,
    )
    nan_bundle = tuple(
        np.full(value.shape, np.nan, dtype=np.asarray(value).dtype)
        for value in (state.u, state.v, state.w)
    )
    guarded = _augment_large_step_tendencies(
        state,
        disabled.tendencies,
        disabled,
        rk_step=3,
        frozen_diff6_uvw_tendencies=nan_bundle,
    )
    for name in ("u", "v", "w", "theta", "mu"):
        np.testing.assert_array_equal(
            np.asarray(getattr(guarded, name)),
            np.asarray(getattr(baseline, name)),
        )


@pytest.mark.skipif(
    not Path(
        "<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/runs/control/run/namelist.input"
    ).exists(),
    reason="canonical control namelist not mounted",
)
def test_canonical_arm_uses_wrf_default_slopeopt_zero():
    """The implemented literal lane intentionally covers slopeopt=0 only."""

    namelist = Path(
        "<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/runs/control/run/namelist.input"
    ).read_text()
    registry = Path(
        "<DATA_ROOT>/canairy_meteo/artifacts/wrf_src/WRF/Registry/Registry.EM_COMMON"
    ).read_text()
    assert "diff_6th_slopeopt" not in namelist
    assert (
        "rconfig   integer diff_6th_slopeopt       namelist,dynamics"
        "     max_domains    0" in registry
    )


RUN_DIR = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
    "corrected_ni_rca_max_22c2bd7a/"
    "nested_stage_omega_transport_470e6111_dycore_suboperator_ladder1"
)
RUNS = Path("<DATA_ROOT>/wrf_gpu2/v0234_dycore_suboperator_kimi/runs")


@pytest.mark.skipif(
    not (RUN_DIR.exists() and RUNS.exists()),
    reason="retained ladder evidence not mounted",
)
def test_fb_production_lane_equals_wrf_replica_on_retained_control():
    from netCDF4 import Dataset

    from scripts import v0234_dycore_internal_split_kimi as split
    from scripts import v0234_first_interval_momentum_wrf_reassemble as reassemble
    from scripts import v0234_s1_dyn_attribution_fable5 as fab

    ranks = reassemble.load_ranks(RUNS / "control/momsp_dumps")
    u = reassemble.reassemble3d("sp1_entry__u", 1, ranks)
    v = reassemble.reassemble3d("sp1_entry__v", 1, ranks)
    mut = reassemble.reassemble2d("sp2_pbl__mut", 1, ranks)
    with Dataset(RUNS / "control/run/wrfinput_d03") as ds:
        c = {
            name: np.asarray(ds.variables[var][0], dtype=np.float64)
            for name, var in (
                ("msfux", "MAPFAC_UX"), ("msfuy", "MAPFAC_UY"),
                ("msfvx", "MAPFAC_VX"), ("msfvy", "MAPFAC_VY"),
                ("msftx", "MAPFAC_MX"), ("msfty", "MAPFAC_MY"),
            )
        }
        c1h = np.asarray(ds.variables["C1H"][:], dtype=np.float64).reshape(-1)
        c2h = np.asarray(ds.variables["C2H"][:], dtype=np.float64).reshape(-1)
        c1f = np.asarray(ds.variables["C1F"][:], dtype=np.float64).reshape(-1)
        c2f = np.asarray(ds.variables["C2F"][:], dtype=np.float64).reshape(-1)

    _, wrf_sixth_order_uvw_tendf = _lane()
    w = np.zeros((c1f.size, mut.shape[0], mut.shape[1]))
    ue, ve, we = wrf_sixth_order_uvw_tendf(
        u, v, w, mut,
        c1h=c1h, c2h=c2h, c1f=c1f, c2f=c2f,
        msfux=c["msfux"], msfuy=c["msfuy"], msfvx=c["msfvx"], msfvy=c["msfvy"],
        msftx=c["msftx"], msfty=c["msfty"],
        dt=split.DT_D03, diff_6th_factor=0.12, monotonic=True,
    )
    ref_u = fab.wrf_diff6_u_effective(u, mut, c1h, c2h, c["msfux"], c["msfuy"])
    ref_v = fab.wrf_diff6_v_effective(v, mut, c1h, c2h, c["msfvx"], c["msfvy"])
    for got, ref in ((np.asarray(ue), ref_u), (np.asarray(ve), ref_v)):
        scale = float(np.abs(ref).max())
        assert scale > 0.0
        assert float(np.abs(got - ref).max()) <= 1e-10 * scale
    assert np.all(np.asarray(we) == 0.0)
