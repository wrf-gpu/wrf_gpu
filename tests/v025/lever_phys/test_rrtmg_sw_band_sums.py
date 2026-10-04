"""lever-phys LP02: fused SW band kernel (GPUWRF_RRTMG_SW_BAND_SUMS) on the WRF-REAL path.

Per band one Pallas launch forms the cloud optics, Eddington reftra and BOTH WRF
``vrtqdr_sw`` streams (total sky and clear sky) and emits g-summed fluxes.  Gates:
the frozen WRF tier-1 SW oracle through the fused path (with a spy that the kernel
ran, E114), and same-input agreement with the current per-g-point quadrature path
on a cloudy fixture including a night column (WRF dorrsw: exact zero SW).
CPU runs use Pallas interpret mode.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from gpuwrf.kernels import rad_sw_band_sums as B
from gpuwrf.kernels import rad_sw_quadrature as Q
from gpuwrf.physics import rrtmg_lw as lw
from gpuwrf.physics import rrtmg_sw as sw
from gpuwrf.validation import tier1_rrtmg
from gpuwrf.validation.tier1_rrtmg import load_sw_fixture_state

FLAG = "GPUWRF_RRTMG_SW_BAND_SUMS"


@pytest.fixture
def real_path(monkeypatch):
    """Both radiation fast flags (WRF-REAL entry); Pallas kernels interpreted on CPU."""
    monkeypatch.setattr(sw, "_FUSED_QUADRATURE", True)
    monkeypatch.setattr(lw, "_FUSED_TRANSFER", True)
    assert sw._real_entry()
    calls = []
    quad, sums, allb = Q.vertical_quadrature, B.sw_band_flux_sums, B.sw_allband_flux_sums
    monkeypatch.setattr(Q, "vertical_quadrature", lambda *a, **k: quad(*a, **{**k, "interpret": True}))
    monkeypatch.setattr(B, "sw_band_flux_sums",
                        lambda *a, **k: calls.append(1) or sums(*a, **{**k, "interpret": True}))
    monkeypatch.setattr(B, "sw_allband_flux_sums",
                        lambda *a, **k: calls.append(2) or allb(*a, **{**k, "interpret": True}))
    sw.solve_rrtmg_sw_column.clear_cache()
    yield calls
    sw.solve_rrtmg_sw_column.clear_cache()


def _solve(monkeypatch, state, flag):
    monkeypatch.setenv(FLAG, flag)
    sw.solve_rrtmg_sw_column.clear_cache()
    return jax.block_until_ready(sw.solve_rrtmg_sw_column(state, debug=False, with_clear_sky=True))


@pytest.mark.parametrize("mode", ["1", "2", "3"])
def test_sw_band_sums_pass_wrf_tier1(monkeypatch, tmp_path, real_path, mode):
    monkeypatch.setenv(FLAG, mode)
    monkeypatch.setattr(tier1_rrtmg, "solve_rrtmg_sw_column", sw.solve_rrtmg_sw_column)
    record = tier1_rrtmg.run_tier1_sw(tmp_path / "sw_band_sums.json")
    assert real_path and set(real_path) == {2 if mode == "2" else 1}, "fused SW band kernel was not dispatched"
    assert record["pass"], record


@pytest.mark.parametrize("mode", ["1", "2", "3"])
def test_sw_band_sums_match_quadrature_path(monkeypatch, real_path, mode):
    state, _ = load_sw_fixture_state()
    # Night column (WRF RRTMG_SWRAD dorrsw=.false. -> all SW fluxes zero).
    coszen = np.asarray(state.coszen).copy()
    coszen[1] = 0.0
    state = state.replace(coszen=jax.numpy.asarray(coszen, dtype=state.coszen.dtype))
    off = _solve(monkeypatch, state, "0")
    assert not real_path
    on = _solve(monkeypatch, state, mode)
    assert real_path and set(real_path) == {2 if mode == "2" else 1}
    # The fixture must exercise clouds: total and clear streams differ.
    gap = np.max(np.abs(np.asarray(off.flux_down, np.float64) - np.asarray(off.clear_flux_down, np.float64)))
    assert gap > 1.0, gap
    for field, rel in (("clear_flux_down", 1e-6), ("clear_flux_up", 1e-6), ("flux_down", 2e-4),
                       ("flux_up", 2e-4), ("heating_rate", 2e-4), ("surface_direct", 1e-6),
                       ("surface_down", 2e-4), ("surface_up", 2e-4)):
        a = np.asarray(getattr(off, field), np.float64)
        b = np.asarray(getattr(on, field), np.float64)
        assert a.shape == b.shape and np.all(np.isfinite(b)), field
        scale = max(float(np.max(np.abs(a))), 1e-30)
        assert float(np.max(np.abs(a - b))) <= rel * scale, (field, float(np.max(np.abs(a - b))), scale)
        assert np.all(b[1] == 0.0), (field, "night column must be exactly zero")


@pytest.mark.parametrize("mode", ["1", "2", "3"])
def test_sw_band_sums_dark_block_is_exact_zero(monkeypatch, real_path, mode):
    """A kernel block with no sunlit column takes the dorrsw branch: exact zeros, OFF-identical."""
    state, _ = load_sw_fixture_state()
    state = state.replace(coszen=jax.numpy.zeros_like(state.coszen))
    off = _solve(monkeypatch, state, "0")
    on = _solve(monkeypatch, state, mode)
    assert real_path and set(real_path) == {2 if mode == "2" else 1}
    for field in ("flux_down", "flux_up", "clear_flux_down", "clear_flux_up", "heating_rate",
                  "surface_direct", "surface_down", "surface_up", "surface_diffuse_fraction"):
        a, b = np.asarray(getattr(off, field)), np.asarray(getattr(on, field))
        assert np.all(b == 0.0), field
        assert a.tobytes() == b.tobytes(), field


def test_sw_allband_launch_equals_band_scan(monkeypatch, real_path):
    """Modes 2 (one launch, band-order XLA sum) and 3 (unrolled bands) reproduce mode 1 (band scan + carry)."""
    state, _ = load_sw_fixture_state()
    one = _solve(monkeypatch, state, "1")
    two = _solve(monkeypatch, state, "2")
    three = _solve(monkeypatch, state, "3")
    assert set(real_path) == {1, 2}
    # Same per-band solve; taumol is fused differently (stacked vs switch branch), so
    # flux-difference fields (heating) see round-off cancellation.
    for field, rel in (("flux_down", 2e-6), ("flux_up", 2e-6), ("clear_flux_down", 2e-6), ("clear_flux_up", 2e-6),
                       ("surface_direct", 2e-6), ("heating_rate", 2e-4)):
        a = np.asarray(getattr(one, field), np.float64)
        scale = max(float(np.max(np.abs(a))), 1e-30)
        for other in (two, three):
            b = np.asarray(getattr(other, field), np.float64)
            assert float(np.max(np.abs(a - b))) <= rel * scale, (field, float(np.max(np.abs(a - b))))
