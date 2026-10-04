"""v025 acoustic-while-collapse sprint: zolrib trace-time unroll prototype gates.

The sprint contract (2026-09-18-v0250-acoustic-while-collapse) FALSIFIED its own
premise: the profiled 60.9 us ``loop_or_select_fusion`` is the body fusion of
``surface_layer._zolrib``'s fixed-point z/L iteration (MYNN surface layer), not
the acoustic while-condition (see the sprint WORKER_REPORT.md).  The prototype
here unrolls that loop at trace time WITHOUT touching ``physics/**``.

Pre-registered gates (declared in
``.agent/sprints/2026-09-18-v0250-acoustic-while-collapse/zolrib_unroll_prototype.py``
BEFORE first run, per the E1/FMA precedent
``proofs/v025/econ/compare_e1_applied_anchor_VIOLATION.json``):

- G0: production ``_zolrib`` outputs are finite on the sprint fixtures.
- G1 numerics: PRIMARY bitwise equality of unrolled vs production; pre-registered
  fallback (used and PASSED at prototype time on CPU): max|diff| <= 1e-13*scale,
  all finite.  A bitwise miss is expected from XLA FMA-contraction differences
  between loop and straight-line fusion contexts (frozen-mask identity is
  additionally asserted by run_gates.py's anchored transcription comparison).
- G2 HLO: reference compiles to exactly one ``while`` (with s64 trip counter);
  unrolled compiles to zero ``while`` ops and zero s64 references.
"""

from __future__ import annotations

import os
import sys

import jax.numpy as jnp
import pytest

_SPRINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".agent", "sprints", "2026-09-18-v0250-acoustic-while-collapse",
)
sys.path.insert(0, _SPRINT)

from zolrib_unroll_prototype import (  # noqa: E402
    bitwise_equal,
    compiled_census,
    fixture,
    zolrib_unrolled,
)

from gpuwrf.physics.surface_layer import _li_etal_2010, _zolrib  # noqa: E402


@pytest.fixture(scope="module")
def fixtures():
    fx = fixture()
    fx_warm = dict(fx)
    fx_warm["zol1_seed"] = _li_etal_2010(fx["ri"], fx["za"] / fx["z0"], fx["z0"] / fx["zt"]) * 0.9
    return fx, fx_warm


def test_g0_reference_outputs_finite(fixtures):
    fx, fx_warm = fixtures
    assert bool(jnp.all(jnp.isfinite(_zolrib(**fx))))
    assert bool(jnp.all(jnp.isfinite(_zolrib(**fx_warm))))


def test_g1_unrolled_matches_production_within_pre_registered_gate(fixtures):
    fx, fx_warm = fixtures
    un_cold = zolrib_unrolled(**fx)[0]
    un_warm = zolrib_unrolled(**fx_warm)[0]
    ref_cold = _zolrib(**fx)
    ref_warm = _zolrib(**fx_warm)

    if bitwise_equal(ref_cold, un_cold) and bitwise_equal(ref_warm, un_warm):
        return  # primary bitwise gate

    # pre-registered fallback: bounded diff + finite outputs
    max_diff = float(jnp.max(jnp.abs(ref_cold - un_cold)))
    scale = float(max(1.0, jnp.max(jnp.abs(ref_cold))))
    assert max_diff <= 1.0e-13 * scale, (max_diff, scale)
    assert bool(jnp.all(jnp.isfinite(un_cold))) and bool(jnp.all(jnp.isfinite(un_warm)))


def test_g2_unrolled_has_no_while_and_no_counter(fixtures):
    fx, _ = fixtures
    cen_ref = compiled_census(
        lambda ri, za, z0, zt, logz0, logzt: _zolrib(ri, za, z0, zt, logz0, logzt), fx
    )
    cen_un = compiled_census(
        lambda ri, za, z0, zt, logz0, logzt: zolrib_unrolled(ri, za, z0, zt, logz0, logzt)[0], fx
    )
    assert cen_ref["while_ops"] == 1
    assert cen_un["while_ops"] == 0
    # the while carries an s64 trip counter; the unroll must not
    assert cen_un["s64_refs"] == 0
