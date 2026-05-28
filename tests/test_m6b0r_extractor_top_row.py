"""Regression tests for the M6B0-R WRF savepoint extractor top-row coefficients.

These tests pin two latent bugs identified in the
``2026-05-24-m6b0r-reproducer-audit`` sprint and fixed in the
``2026-05-24-m6b0r-extractor-fix-hygiene`` sprint:

1. ``lid_flag`` must be conditional on the WRF ``top_lid`` namelist flag
   (WRF ``module_small_step_em.F:619-620``), not hardcoded to ``1.0``.
2. The top ``a`` row denominator must use ``c1f(kde-1)`` per
   ``module_small_step_em.F:626``, while the top ``b`` row denominator
   uses ``c1f(kde)`` per ``module_small_step_em.F:646``. The previous
   single shared ``denom_top`` matched only the ``b`` row.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR_PATH = REPO_ROOT / "scripts" / "m6b0r_wrf_savepoint_extract.py"


def _load_extractor():
    """Imports the extractor script as a module without executing ``main``.

    The script depends on ``netCDF4`` for ``_load_state`` but the calc kernel
    and ``_resolve_top_lid`` are dataset-free; we can exercise them as
    library functions in unit tests.
    """

    spec = importlib.util.spec_from_file_location("m6b0r_extractor_under_test", EXTRACTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("m6b0r_extractor_under_test", module)
    spec.loader.exec_module(module)
    return module


EXTRACTOR = _load_extractor()


def _synthetic_state(nz: int = 6, ny: int = 1, nx: int = 1) -> dict[str, object]:
    """Returns a state dict with distinct, easily-traceable c1h/c1f profiles.

    Using monotone integer-derived profiles makes any index slippage in the
    top-row computation arithmetically detectable: e.g. with the c1f values
    below, ``c1f[nz-1] = 2.5`` while ``c1f[nz] = 3.0``, so a wrong index
    leaves a ratio-shaped fingerprint in ``a[nz]``.
    """

    rng = np.random.default_rng(seed=42)
    mut = 90_000.0 + 1_000.0 * rng.standard_normal((ny, nx))
    c1h = np.linspace(0.5, 0.75, nz)
    c2h = np.linspace(0.5, 0.25, nz)
    c1f = np.linspace(0.5, 0.8, nz + 1)
    c2f = np.linspace(0.5, 0.2, nz + 1)
    rdn = np.linspace(50.0, 60.0, nz)
    rdnw = np.linspace(45.0, 55.0, nz)
    theta = 300.0 * np.ones((nz, ny, nx))
    return {
        "theta": theta,
        "mut": mut,
        "c1h": c1h,
        "c2h": c2h,
        "c1f": c1f,
        "c2f": c2f,
        "rdn": rdn,
        "rdnw": rdnw,
        "attrs": {"top_lid": False},
    }


def _expected_top_a(state: dict[str, object], *, top_lid: bool, dts: float, epssm: float, g: float) -> np.ndarray:
    """Recomputes ``a[nz]`` from first principles using the canonical WRF indices."""

    nz = int(np.asarray(state["theta"]).shape[0])
    mut = np.asarray(state["mut"], dtype=np.float64)
    c1h = np.asarray(state["c1h"], dtype=np.float64)
    c2h = np.asarray(state["c2h"], dtype=np.float64)
    c1f = np.asarray(state["c1f"], dtype=np.float64)
    c2f = np.asarray(state["c2f"], dtype=np.float64)
    rdnw = np.asarray(state["rdnw"], dtype=np.float64)
    cof = (0.5 * dts * g * (1.0 + epssm)) ** 2
    lid_flag = 0.0 if top_lid else 1.0
    # WRF :626 uses c1h(kde-1)=c1h[nz-1] and c1f(kde-1)=c1f[nz-1].
    denom_a = (c1h[nz - 1] * mut + c2h[nz - 1]) * (c1f[nz - 1] * mut + c2f[nz - 1])
    return -2.0 * cof * rdnw[nz - 1] ** 2 * 1.0 * lid_flag / denom_a


def _expected_b_top(state: dict[str, object], *, dts: float, epssm: float, g: float) -> np.ndarray:
    """Recomputes ``b_top`` denominator-driven coefficient using canonical WRF indices."""

    nz = int(np.asarray(state["theta"]).shape[0])
    mut = np.asarray(state["mut"], dtype=np.float64)
    c1h = np.asarray(state["c1h"], dtype=np.float64)
    c2h = np.asarray(state["c2h"], dtype=np.float64)
    c1f = np.asarray(state["c1f"], dtype=np.float64)
    c2f = np.asarray(state["c2f"], dtype=np.float64)
    rdnw = np.asarray(state["rdnw"], dtype=np.float64)
    cof = (0.5 * dts * g * (1.0 + epssm)) ** 2
    # WRF :646 uses c1h(kde-1)=c1h[nz-1] and c1f(kde)=c1f[nz].
    denom_b = (c1h[nz - 1] * mut + c2h[nz - 1]) * (c1f[nz] * mut + c2f[nz])
    return 1.0 + 2.0 * cof * rdnw[nz - 1] ** 2 * 1.0 / denom_b


class TestLidFlag:
    """WRF ``module_small_step_em.F:619-620``: ``IF(top_lid) lid_flag=0``."""

    def test_top_lid_true_zeros_a_top_row(self):
        state = _synthetic_state()
        state["attrs"]["top_lid"] = True
        coeffs = EXTRACTOR._wrf_calc_coef_w(state, dts=6.0, epssm=0.1)
        # With top_lid=True -> lid_flag=0 -> a[nz] must be identically zero.
        assert np.allclose(coeffs["a"][-1], 0.0), (
            "top_lid=True should drive lid_flag=0 and zero out a[nz]; got non-zero values"
        )

    def test_top_lid_false_produces_canonical_a_top_row(self):
        state = _synthetic_state()
        state["attrs"]["top_lid"] = False
        coeffs = EXTRACTOR._wrf_calc_coef_w(state, dts=6.0, epssm=0.1)
        expected = _expected_top_a(state, top_lid=False, dts=6.0, epssm=0.1, g=9.80665)
        np.testing.assert_allclose(coeffs["a"][-1], expected, rtol=0.0, atol=0.0)
        # And it must NOT be zero — guards against a regression that
        # accidentally sets lid_flag=0 always.
        assert np.max(np.abs(coeffs["a"][-1])) > 0.0

    def test_explicit_kwarg_overrides_state_attrs(self):
        state = _synthetic_state()
        state["attrs"]["top_lid"] = False
        coeffs = EXTRACTOR._wrf_calc_coef_w(state, dts=6.0, epssm=0.1, top_lid=True)
        assert np.allclose(coeffs["a"][-1], 0.0)


class TestTopRowDenomIndex:
    """WRF ``:626`` (top ``a``) vs ``:646`` (top ``b``) ``c1f`` indices differ."""

    def test_top_a_row_uses_c1f_at_kde_minus_one(self):
        state = _synthetic_state()
        state["attrs"]["top_lid"] = False
        coeffs = EXTRACTOR._wrf_calc_coef_w(state, dts=6.0, epssm=0.1)
        expected_a_top = _expected_top_a(state, top_lid=False, dts=6.0, epssm=0.1, g=9.80665)
        np.testing.assert_allclose(coeffs["a"][-1], expected_a_top, rtol=0.0, atol=0.0)

    def test_top_a_row_with_wrong_c1f_index_would_differ(self):
        """Sanity check: ``c1f[nz-1]`` and ``c1f[nz]`` produce distinguishable outputs.

        Without this distinction the regression test on
        ``test_top_a_row_uses_c1f_at_kde_minus_one`` would be vacuous because
        the two indices would happen to give the same number.
        """

        state = _synthetic_state()
        canonical = _expected_top_a(state, top_lid=False, dts=6.0, epssm=0.1, g=9.80665)
        # Recompute with the buggy index to confirm the fixture is sensitive.
        nz = int(np.asarray(state["theta"]).shape[0])
        mut = np.asarray(state["mut"], dtype=np.float64)
        c1h = np.asarray(state["c1h"], dtype=np.float64)
        c2h = np.asarray(state["c2h"], dtype=np.float64)
        c1f = np.asarray(state["c1f"], dtype=np.float64)
        c2f = np.asarray(state["c2f"], dtype=np.float64)
        rdnw = np.asarray(state["rdnw"], dtype=np.float64)
        cof = (0.5 * 6.0 * 9.80665 * (1.0 + 0.1)) ** 2
        denom_bug = (c1h[nz - 1] * mut + c2h[nz - 1]) * (c1f[nz] * mut + c2f[nz])
        buggy = -2.0 * cof * rdnw[nz - 1] ** 2 / denom_bug
        assert not np.allclose(canonical, buggy), (
            "Fixture is insensitive — c1f[nz-1] and c1f[nz] would yield "
            "indistinguishable top-row a values, defeating the regression."
        )

    def test_top_b_row_uses_c1f_at_kde(self):
        """``b_top`` is implicit in ``alpha[nz]``: ``alpha[nz] = 1/(b_top - a[nz]*gamma[nz-1])``.

        With ``top_lid=True`` we get ``a[nz]=0`` and the alpha relation reduces
        to ``alpha[nz] = 1/b_top`` exactly, which lets us read the
        ``b_top`` denominator (which uses ``c1f[nz]``) from the output.
        """

        state = _synthetic_state()
        state["attrs"]["top_lid"] = True
        coeffs = EXTRACTOR._wrf_calc_coef_w(state, dts=6.0, epssm=0.1)
        # a[nz] is zero -> alpha[nz] = 1 / b_top
        b_top_implied = 1.0 / np.asarray(coeffs["alpha"][-1])
        expected_b_top = _expected_b_top(state, dts=6.0, epssm=0.1, g=9.80665)
        np.testing.assert_allclose(b_top_implied, expected_b_top, rtol=1e-12, atol=0.0)


class TestResolveTopLid:
    """``_resolve_top_lid`` parsing of wrfout/namelist fall-through chain."""

    def test_explicit_wrfout_attribute_boolean_true(self):
        class FakeDS:
            TOP_LID = True

        assert EXTRACTOR._resolve_top_lid(FakeDS()) is True

    def test_explicit_wrfout_attribute_boolean_false(self):
        class FakeDS:
            TOP_LID = False

        assert EXTRACTOR._resolve_top_lid(FakeDS()) is False

    def test_explicit_wrfout_attribute_string_true(self):
        class FakeDS:
            TOP_LID = "T"

        assert EXTRACTOR._resolve_top_lid(FakeDS()) is True

    def test_explicit_wrfout_attribute_string_false(self):
        class FakeDS:
            TOP_LID = "F"

        assert EXTRACTOR._resolve_top_lid(FakeDS()) is False

    def test_canary_namelist_fallback_returns_false(self):
        """When the wrfout omits ``TOP_LID``, the sibling ``namelist.output``
        of the canonical Canary d02 source run pins ``TOP_LID= 11*F``."""

        class FakeDS:
            pass

        assert EXTRACTOR._resolve_top_lid(FakeDS()) is False
