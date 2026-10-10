"""CAM ``radabs`` (+ trcab/trcabn) JAX port vs the pristine-WRF true-caller oracle CAM01 (210 real/augmented columns).

Oracle: ``proofs/cam_rad`` (pristine ``radclwmx`` call sequence; every radabs operand and both outputs dumped).
Gate: max relative error <= 1e-11 on all finite entries (relative to max(|ref|, tiny)) and an identical ``inf``
(1.e20) marker pattern.
A precision mutant (``r293`` folded in double instead of WRF's float32 ``1./293.``) and a deletion mutant (trace gases
dropped from the non-adjacent sum) must both FAIL the same gate.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

import gpuwrf  # noqa: F401  (enables jax x64)
import jax
import jax.numpy as jnp

from gpuwrf.physics import ra_cam_lw_abs as mod
from gpuwrf.physics.ra_cam_common import load_cam_abs_tables

FIXTURE = Path(os.environ.get("CAM01_FIXTURE", Path(__file__).resolve().parents[1] / "data" / "fixtures" / "cam01-compact-v1.npz"))
RTOL = 1e-11
TINY = 1e-300

_ARGS = ("pbr", "pnm", "co2em", "co2eml", "tplnka", "s2c", "tcg", "w", "h2otr", "plco2", "plh2o", "co2t", "tint",
         "tlayr", "plol", "plos", "pmln", "piln", "ucfc11", "ucfc12", "un2o0", "un2o1", "uch4", "uco211", "uco212",
         "uco213", "uco221", "uco222", "uco223", "uptype", "bn2o0", "bn2o1", "bch4", "abplnk1", "abplnk2", "plh2ob",
         "wb")


def _inputs(d):
    out = []
    for name in _ARGS:
        if name == "pmln":
            out.append(np.log(d["r8l_pmid"]))
        elif name == "piln":
            out.append(np.log(d["r8l_pint"]))
        else:
            out.append(d["lwi_" + name])
    return [jnp.asarray(np.asarray(x, np.float64)) for x in out]


@pytest.fixture(scope="module")
def oracle():
    if not FIXTURE.exists():
        pytest.skip(f"CAM01 fixture missing: {FIXTURE}")
    d = np.load(FIXTURE)
    return _inputs(d), np.asarray(d["lwi_abstot"]), np.asarray(d["lwi_absnxt"])


@pytest.fixture(scope="module")
def tables():
    try:
        return load_cam_abs_tables()
    except (FileNotFoundError, OSError) as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CAM_ABS_DATA not available: {exc}")


def _run(args, tables):
    fn = jax.jit(lambda *a: mod.radabs(*a, tables))  # fresh lambda -> fresh trace (module globals read at trace time)
    t0 = time.perf_counter()
    abstot, absnxt = fn(*args)
    abstot.block_until_ready()
    t1 = time.perf_counter()
    abstot2, absnxt2 = fn(*args)
    absnxt2.block_until_ready()
    t2 = time.perf_counter()
    return np.asarray(abstot), np.asarray(absnxt), t1 - t0, t2 - t1


def _errors(got, ref):
    """(max abs err, max rel err, inf patterns identical) over finite non-marker entries."""

    marker = ref == mod.INF
    same_marker = bool(np.array_equal(marker, got == mod.INF))
    ok = ~marker
    if not np.all(np.isfinite(got[ok])):
        return np.inf, np.inf, same_marker
    diff = np.abs(got[ok] - ref[ok])
    rel = diff / np.maximum(np.abs(ref[ok]), TINY)
    return float(diff.max()), float(rel.max()), same_marker


def test_radabs_matches_pristine_wrf(oracle, tables):
    args, ref_tot, ref_nxt = oracle
    abstot, absnxt, t_first, t_warm = _run(args, tables)
    assert abstot.shape == ref_tot.shape and absnxt.shape == ref_nxt.shape
    e_tot = _errors(abstot, ref_tot)
    e_nxt = _errors(absnxt, ref_nxt)
    print(f"\nradabs CAM01 {ref_tot.shape[0]} cols: abstot max abs {e_tot[0]:.3e} rel {e_tot[1]:.3e}; "
          f"absnxt max abs {e_nxt[0]:.3e} rel {e_nxt[1]:.3e}; inf markers {int((ref_tot == mod.INF).sum())}; "
          f"jit+first run {t_first:.2f} s, warm run {t_warm * 1e3:.1f} ms")
    assert e_tot[2], "abstot inf (1.e20) marker pattern differs"
    assert e_nxt[2], "absnxt inf (1.e20) marker pattern differs"
    assert e_tot[1] <= RTOL, f"abstot max rel err {e_tot[1]:.3e} > {RTOL}"
    assert e_nxt[1] <= RTOL, f"absnxt max rel err {e_nxt[1]:.3e} > {RTOL}"


def test_radabs_mutants_fail(oracle, tables, monkeypatch):
    """The gate is deletion- and precision-sensitive: each mutant must break parity."""

    args, ref_tot, ref_nxt = oracle

    # (1) precision mutant: r293 = 1/293 in double instead of the float32-folded default-REAL constant.
    monkeypatch.setattr(mod, "R293", 1.0 / 293.0)
    abstot, absnxt, _, _ = _run(args, tables)
    e_tot, e_nxt = _errors(abstot, ref_tot), _errors(absnxt, ref_nxt)
    print(f"\nmutant r293(double): abstot rel {e_tot[1]:.3e}, absnxt rel {e_nxt[1]:.3e}")
    assert e_tot[1] > RTOL and e_nxt[1] > RTOL
    monkeypatch.undo()

    # (2) deletion mutant: trace-gas absorptivity (trcab) dropped from the non-adjacent sum.
    monkeypatch.setattr(mod, "_trcab", lambda *a, **k: 0.0)
    abstot, absnxt, _, _ = _run(args, tables)
    e_tot, e_nxt = _errors(abstot, ref_tot), _errors(absnxt, ref_nxt)
    print(f"mutant no-trcab: abstot rel {e_tot[1]:.3e}, absnxt rel {e_nxt[1]:.3e}")
    assert e_tot[1] > RTOL
    assert e_nxt[1] <= RTOL  # nearest layer untouched by this mutant
