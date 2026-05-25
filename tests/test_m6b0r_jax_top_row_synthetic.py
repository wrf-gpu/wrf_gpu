"""Synthetic regression for JAX ``calc_coef_w`` top-row ``a``/``b`` index split.

Pins the M6B0-R follow-up fix in ``src/gpuwrf/dynamics/acoustic_wrf.py:638-641,664``
that splits the previously shared ``top_denom`` into ``top_denom_a``
(uses ``c1f[nz-1]``) and ``top_denom_b`` (uses ``c1f[nz]``) per WRF source
``dyn_em/module_small_step_em.F:624-628`` (top ``a`` row) and ``:644-648``
(top ``b`` row).

The Canary M6B0-R dataset has ``c1f[nz-1] = c1f[nz] = 0`` so the previous shared
denominator silently matched the ``b``-row formula and the ``a``-row top
coefficient was multiplied by zero on both sides. This test uses a synthetic
``c1f`` profile where ``c1f[nz-1] = 0.5`` and ``c1f[nz] = 0.0`` so the two
denominators are arithmetically distinguishable and a regression would be
immediately observable.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from gpuwrf.contracts.grid import DycoreMetrics
from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients


GRAVITY_M_S2 = 9.80665


def _synthetic_metrics(nz: int = 6, ny: int = 1, nx: int = 1) -> DycoreMetrics:
    """Builds a DycoreMetrics with a non-monotone c1f profile at the top face.

    Setting ``c1f[nz-1] = 0.5`` and ``c1f[nz] = 0.0`` guarantees the two
    candidate top-row denominators differ; with ``c2f`` distinct as well the
    test can read which denominator was used straight from the output.
    """

    eta_levels = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    base = DycoreMetrics.flat(
        ny=ny,
        nx=nx,
        nz=nz,
        eta_levels=eta_levels,
        top_pressure_pa=5000.0,
        provenance="m6b0r-synthetic-top-row",
    )
    c1f = np.asarray(base.c1f).copy()
    c2f = np.asarray(base.c2f).copy()
    c1f[nz - 1] = 0.5
    c1f[nz] = 0.0
    # Make c2f distinct at the top two faces so c2f-only contributions also
    # split correctly between the two rows.
    c2f[nz - 1] = 1000.0
    c2f[nz] = 2000.0
    return dataclasses.replace(
        base,
        c1f=jnp.asarray(c1f, dtype=jnp.float64),
        c2f=jnp.asarray(c2f, dtype=jnp.float64),
    )


def _expected_a_top(mut: np.ndarray, metrics: DycoreMetrics, *, dt: float, epssm: float, top_lid: bool) -> np.ndarray:
    """Recomputes ``a[nz]`` from first principles using WRF :626 indices.

    WRF source (``module_small_step_em.F:622-626``)::

        k = kde-1
        a(i,kde,j) = -2*cof*rdnw(kde-1)**2 * c2a(i,kde-1,j) * lid_flag
                     / ((c1h(k)*MUT+c2h(k)) * (c1f(k)*MUT+c2f(k)))

    In 0-indexed Python with ``nz = kde-1``, ``c1h[k]=c1h[nz-1]`` and
    ``c1f[k]=c1f[nz-1]``.
    """

    nz = int(metrics.c1h.shape[0])
    mut_arr = np.asarray(mut, dtype=np.float64)
    c1h = np.asarray(metrics.c1h, dtype=np.float64)
    c2h = np.asarray(metrics.c2h, dtype=np.float64)
    c1f = np.asarray(metrics.c1f, dtype=np.float64)
    c2f = np.asarray(metrics.c2f, dtype=np.float64)
    rdnw = np.asarray(metrics.rdnw, dtype=np.float64)
    cof = (0.5 * float(dt) * GRAVITY_M_S2 * (1.0 + float(epssm))) ** 2
    lid_flag = 0.0 if top_lid else 1.0
    denom_a = (c1h[nz - 1] * mut_arr + c2h[nz - 1]) * (c1f[nz - 1] * mut_arr + c2f[nz - 1])
    return -2.0 * cof * rdnw[nz - 1] ** 2 * 1.0 * lid_flag / denom_a


def _expected_b_top(mut: np.ndarray, metrics: DycoreMetrics, *, dt: float, epssm: float) -> np.ndarray:
    """Recomputes ``b_top`` from first principles using WRF :646 indices.

    WRF source (``module_small_step_em.F:644-646``)::

        k = kde
        b = 1. + 2*cof*rdnw(kde-1)**2 * c2a(i,kde-1,j)
                / ((c1h(k-1)*MUT+c2h(k-1)) * (c1f(k)*MUT+c2f(k)))

    In 0-indexed Python with ``nz = kde-1``, ``c1h[k-1]=c1h[nz-1]`` and
    ``c1f[k]=c1f[nz]``.
    """

    nz = int(metrics.c1h.shape[0])
    mut_arr = np.asarray(mut, dtype=np.float64)
    c1h = np.asarray(metrics.c1h, dtype=np.float64)
    c2h = np.asarray(metrics.c2h, dtype=np.float64)
    c1f = np.asarray(metrics.c1f, dtype=np.float64)
    c2f = np.asarray(metrics.c2f, dtype=np.float64)
    rdnw = np.asarray(metrics.rdnw, dtype=np.float64)
    cof = (0.5 * float(dt) * GRAVITY_M_S2 * (1.0 + float(epssm))) ** 2
    denom_b = (c1h[nz - 1] * mut_arr + c2h[nz - 1]) * (c1f[nz] * mut_arr + c2f[nz])
    return 1.0 + 2.0 * cof * rdnw[nz - 1] ** 2 * 1.0 / denom_b


def test_top_a_row_uses_c1f_at_nz_minus_one():
    """``a[nz]`` must match the WRF :626 formula (uses ``c1f[nz-1]``).

    With ``c1f[nz-1] != c1f[nz]`` the previous shared ``top_denom`` would have
    landed on the ``b``-row denominator and produced a numerically distinct
    value; this test asserts equality with the ``a``-row formula and also
    asserts the buggy ``b``-row denominator would have given a different result
    (i.e. the fixture is sensitive enough to expose the bug).
    """

    metrics = _synthetic_metrics()
    mut = np.asarray([[90_000.0]], dtype=np.float64)
    dt = 6.0
    epssm = 0.1
    a, _alpha, _gamma = calc_coef_w_wrf_coefficients(
        jnp.asarray(mut),
        metrics,
        dt=dt,
        epssm=epssm,
        top_lid=False,
    )
    expected = _expected_a_top(mut, metrics, dt=dt, epssm=epssm, top_lid=False)
    np.testing.assert_allclose(np.asarray(a[-1]), expected, rtol=1e-12, atol=0.0)
    assert np.max(np.abs(expected)) > 0.0, "fixture must produce non-zero a[nz]"

    # Sensitivity guard: the c1f[nz] (buggy) denominator must differ from the
    # canonical c1f[nz-1] denominator, otherwise this regression is vacuous.
    nz = int(metrics.c1h.shape[0])
    c1h = np.asarray(metrics.c1h, dtype=np.float64)
    c2h = np.asarray(metrics.c2h, dtype=np.float64)
    c1f = np.asarray(metrics.c1f, dtype=np.float64)
    c2f = np.asarray(metrics.c2f, dtype=np.float64)
    rdnw = np.asarray(metrics.rdnw, dtype=np.float64)
    cof = (0.5 * dt * GRAVITY_M_S2 * (1.0 + epssm)) ** 2
    denom_buggy = (c1h[nz - 1] * mut + c2h[nz - 1]) * (c1f[nz] * mut + c2f[nz])
    buggy = -2.0 * cof * rdnw[nz - 1] ** 2 / denom_buggy
    assert not np.allclose(buggy, expected), (
        "Fixture is insensitive — c1f[nz-1] and c1f[nz] yield indistinguishable "
        "top-row a values; the regression cannot expose the bug."
    )


def test_top_b_row_uses_c1f_at_nz():
    """``b_top`` (implied via ``alpha[nz]``) must match WRF :646 (uses ``c1f[nz]``).

    Setting ``top_lid=True`` zeroes ``a[nz]`` so the alpha recurrence collapses
    to ``alpha[nz] = 1 / b_top``, which lets the test read ``b_top`` directly
    from the output and confirm the ``c1f[nz]`` denominator was used.
    """

    metrics = _synthetic_metrics()
    mut = np.asarray([[90_000.0]], dtype=np.float64)
    dt = 6.0
    epssm = 0.1
    a, alpha, _gamma = calc_coef_w_wrf_coefficients(
        jnp.asarray(mut),
        metrics,
        dt=dt,
        epssm=epssm,
        top_lid=True,
    )
    # With top_lid=True the source pins lid_flag=0, so a[nz] == 0 and
    # alpha[nz] = 1/b_top exactly.
    np.testing.assert_allclose(np.asarray(a[-1]), 0.0, atol=0.0)
    b_top_implied = 1.0 / np.asarray(alpha[-1])
    expected = _expected_b_top(mut, metrics, dt=dt, epssm=epssm)
    np.testing.assert_allclose(b_top_implied, expected, rtol=1e-12, atol=0.0)


def test_top_row_denominators_are_distinct():
    """Direct guard on the split: ``top_denom_a != top_denom_b`` for this fixture.

    Recomputes both denominators in numpy and asserts they differ, which is
    the precondition for the bug to be observable at all. Without this, both
    halves of the split would coincide and ``test_top_a_row_uses_c1f_at_nz_minus_one``
    could not distinguish them.
    """

    metrics = _synthetic_metrics()
    mut = np.asarray([[90_000.0]], dtype=np.float64)
    nz = int(metrics.c1h.shape[0])
    c1h = np.asarray(metrics.c1h, dtype=np.float64)
    c2h = np.asarray(metrics.c2h, dtype=np.float64)
    c1f = np.asarray(metrics.c1f, dtype=np.float64)
    c2f = np.asarray(metrics.c2f, dtype=np.float64)
    mass_h_top = c1h[nz - 1] * mut + c2h[nz - 1]
    mass_f_top_a = c1f[nz - 1] * mut + c2f[nz - 1]
    mass_f_top_b = c1f[nz] * mut + c2f[nz]
    top_denom_a = mass_h_top * mass_f_top_a
    top_denom_b = mass_h_top * mass_f_top_b
    assert not np.allclose(top_denom_a, top_denom_b), (
        f"Synthetic metrics must produce distinct top-row denominators; "
        f"got top_denom_a={top_denom_a}, top_denom_b={top_denom_b}"
    )
