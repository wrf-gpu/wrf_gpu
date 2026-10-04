"""GWDO WRF REAL column kernel (GPUWRF_GWDO_NATIVE_REAL) vs pristine bl_gwdo_run.

Fixture: real CPU-WRF PROD d01 columns (2026-07-26 00z night / 12z day) built
with WRF phy_prep formulas; expected tendencies from the unmodified pristine
routine compiled REAL (tests/v025/b_column/build_gwdo_oracle.py). Pre-registered
gate (lane notes): max |err| <= 1e-5 x field max, 0 columns above 1e-3 relative
column error, 0 non-finite. CPU runs the Pallas interpreter.

Column rule as amended at review (review-gwdo, manager 2026-10-03 03:40Z): a column
fails only if its error exceeds BOTH 1e-3 x its own max AND 1e-4 x the field max.
Justification: on GPU one 12z PROD column (4673, a single-level 1.1e-6 m/s^2 drag,
4e-5 of the field max) misses 1e-3 relative by GPU fp32 rounding of a nearly
cancelling taup difference (abs 1.2e-9 m/s^2; the existing f32 XLA path on GPU gives
the same value); the reviewer's 1-ulp input-perturbation ensemble exceeds 1e-3 in
34 % of runs on that ill-conditioned column, so 1e-3 relative alone is below the
problem's conditioning there.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.kernels import phys_gwdo_column as K

FIX = Path(__file__).resolve().parent / "fixtures" / "gwdo_prod_d01_columns.npz"


def _case(case):
    z = np.load(FIX)
    pre = f"{case}_"
    cols = {k[len(pre) + 3:]: z[k] for k in z.files if k.startswith(pre + "in_")}
    stat = {k[len(pre) + 3:]: z[k] for k in z.files if k.startswith(pre + "st_")}
    wrf = {k[len(pre) + 4:]: z[k] for k in z.files if k.startswith(pre + "wrf_")}
    return cols, stat, wrf, float(z[pre + "dt"])


@pytest.mark.parametrize("case", ["night", "day", "tile", "stress"])
def test_native_kernel_matches_pristine_wrf(case):
    cols, stat, wrf, dt = _case(case)
    got = K.gwdo_tendencies_native(cols, stat, dt, interpret=True)
    active = 0
    for field in ("rublten", "rvblten", "dusfcg", "dvsfcg"):
        ref = wrf[field].astype(np.float64)
        out = np.asarray(got[field], np.float64)
        assert np.isfinite(out).all()
        err = np.abs(out - ref)
        assert err.max() <= 1e-5 * np.abs(ref).max(), (field, err.max())
        colref = np.abs(ref).max(axis=0) if ref.ndim == 2 else np.abs(ref)
        colerr = err.max(axis=0) if ref.ndim == 2 else err
        bad = (colerr > 1e-3 * np.maximum(colref, 1e-6)) & (colerr > 1e-4 * np.abs(ref).max())
        assert not np.any(bad), (field, np.flatnonzero(bad))
        active = max(active, int((colref > 0).sum()))
    # "tile": day columns 3968-4223 = two whole 128-lane tiles of the full domain, kept
    # aligned so per-tile loop bounds see the real mix of mountain-top levels.
    # "stress": the active day columns with the pristine routine at deltim = 540 s
    # (10x PROD), where the dtfac critical-level limiter binds in 28 columns.
    assert active >= (5 if case == "tile" else 100)


def test_native_kernel_is_real_only():
    cols, stat, _, dt = _case("day")
    text = jax.jit(lambda c, s: K.gwdo_tendencies_native(c, s, dt, interpret=True)).lower(
        {k: jnp.asarray(v) for k, v in cols.items()},
        {k: jnp.asarray(v) for k, v in stat.items()}).as_text()
    assert "f64" not in text


def test_flag_routes_tendencies(monkeypatch):
    from gpuwrf.coupling import physics_couplers as C
    calls = []
    monkeypatch.setattr(K, "gwdo_tendencies_from_state", lambda *a: calls.append(a) or "native")
    monkeypatch.setenv("GPUWRF_GWDO_NATIVE_REAL", "1")
    assert C.gwdo_tendencies("s", 54.0, "st", "g") == "native" and len(calls) == 1
    monkeypatch.setenv("GPUWRF_GWDO_NATIVE_REAL", "0")
    with pytest.raises(Exception):
        C.gwdo_tendencies("s", 54.0, "st", "g")  # legacy path runs on the (bogus) State
    assert len(calls) == 1
