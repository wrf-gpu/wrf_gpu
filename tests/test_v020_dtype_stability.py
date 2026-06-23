"""v0.20 S1 dtype-stability gate (CPU-only).

The fp32 program (S3-S8) moves dtype boundaries aggressively. This gate is the
tripwire: it FAILS the moment a hot-path step silently changes a float leaf's
precision (f32->f64 or f64->f32) between carry-in and carry-out -- the exact leak
that produced the 1.1x "convert-scatter" trap. On the bit-identical
``fp64_default`` path the hot step must be dtype-stable; the test also proves the
detector FIRES on an injected promotion (so a future silent leak cannot pass).

Run (CPU, no GPU):
  JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest tests/test_v020_dtype_stability.py -q
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from gpuwrf.ic_generators.idealized import build_warm_bubble_setup
from gpuwrf.profiling.dtype_audit import (
    DtypePromotionError,
    assert_dtype_stable,
    count_converts,
    count_converts_for,
    diff_float_dtypes,
    dtype_histogram,
    named_dtypes,
)
from gpuwrf.runtime.operational_mode import _advance_chunk, _initial_carry_for_run


def _case():
    """Balanced warm-bubble operational case on CPU + a single fp64 hot step."""

    setup = build_warm_bubble_setup(require_gpu=False)
    carry = _initial_carry_for_run(setup.state, setup.namelist)
    nml = setup.namelist
    cadence = max(1, int(nml.radiation_cadence_steps))

    def step(c):
        return _advance_chunk(c, nml, jnp.asarray(1, dtype=jnp.int32), n_steps=1, cadence=cadence)

    return carry, step


# --------------------------------------------------------------------------- #
# 1. fp64_default hot step is dtype-stable (the bit-identical contract).       #
# --------------------------------------------------------------------------- #
def test_fp64_default_advance_chunk_is_dtype_stable():
    carry, step = _case()
    # Must NOT raise: every float leaf keeps its precision across the hot step.
    assert_dtype_stable(step, carry, label="_advance_chunk(fp64_default)")
    # The carry is overwhelmingly float64 on the default path.
    hist = dtype_histogram(carry)
    assert hist.get("float64", 0) > 0, f"expected float64 leaves on fp64 default, got {hist}"


def test_named_per_field_dtype_report_covers_prognostics():
    carry, _ = _case()
    report = named_dtypes(carry, prefix="carry")
    # The per-field report names real prognostic leaves and reports fp64 for them.
    for name in ("carry.state.theta", "carry.state.p_total", "carry.state.ph_total", "carry.state.mu_total"):
        assert name in report, f"{name} missing from per-field dtype report"
        assert report[name] == "float64", f"{name} should be float64 on default path, got {report[name]}"


# --------------------------------------------------------------------------- #
# 2. The detector FIRES on an injected silent promotion (both directions).     #
# --------------------------------------------------------------------------- #
def test_injected_f64_to_f32_downcast_is_caught():
    carry, step = _case()

    def leaky_step(c):
        out = step(c)
        # Silent f64->f32 downcast on a prognostic leaf (the exact storage-downcast
        # contamination S2/S4 must never let slip onto the fp64 default path).
        bad_state = out.state.replace(theta=out.state.theta.astype(jnp.float32), _cast=False)
        return out.replace(state=bad_state)

    with pytest.raises(DtypePromotionError, match=r"float64 -> float32"):
        assert_dtype_stable(leaky_step, carry, label="injected-downcast")


def test_injected_f32_to_f64_upcast_is_caught():
    # The detector must flag an f32 -> f64 upcast (the other silent-promotion
    # direction). Tested on a minimal pytree so the upcast is isolated -- routing
    # an f32 leaf through the real fori_loop dycore would trip JAX's OWN carry
    # dtype-stability check first (a useful second layer, exercised below).
    example = {"a": jnp.ones((4,), jnp.float32), "b": jnp.ones((4,), jnp.float64)}

    def upcast(d):
        return {"a": d["a"].astype(jnp.float64), "b": d["b"]}

    changed = diff_float_dtypes(example, jax.eval_shape(upcast, example))
    assert any(b == "float32" and a == "float64" for (b, a) in changed.values()), changed
    with pytest.raises(DtypePromotionError, match=r"float32 -> float64"):
        assert_dtype_stable(upcast, example, label="injected-upcast")


def test_fori_loop_rejects_dtype_unstable_carry_second_layer():
    # Defense-in-depth: JAX's own fori_loop/scan carry check ALSO rejects a
    # dtype-unstable hot step (carry-in f32 leaf -> carry-out f64). Feeding an f32
    # prognostic into the real dycore step raises a carry-type TypeError, so a
    # silent promotion cannot even compile, independent of the explicit auditor.
    carry, step = _case()
    f32_state = carry.state.replace(theta=carry.state.theta.astype(jnp.float32), _cast=False)
    f32_carry = carry.replace(state=f32_state)
    with pytest.raises(TypeError, match=r"carry"):
        jax.eval_shape(step, f32_carry)


# --------------------------------------------------------------------------- #
# 3. The XLA convert-counter sees converts and their direction.               #
# --------------------------------------------------------------------------- #
def test_convert_counter_detects_explicit_convert_direction():
    # f64 -> f32 explicit convert must be counted with the right transition.
    counts = count_converts_for(lambda x: x.astype(jnp.float32) + jnp.float32(1.0), jnp.ones((8,), jnp.float64))
    assert counts.total >= 1
    assert counts.f64_to_f32 >= 1, counts.as_dict()


def test_convert_counter_pure_fp64_has_no_float_converts():
    # A pure fp64 elementwise program performs no f32<->f64 converts.
    counts = count_converts_for(lambda x: x * 2.0 + 1.0, jnp.ones((8,), jnp.float64))
    assert counts.f32_to_f64 == 0 and counts.f64_to_f32 == 0, counts.as_dict()


def test_convert_counter_parses_stablehlo_and_classic_text():
    stablehlo = (
        '%0 = "stablehlo.convert"(%arg0) : (tensor<4x4xf32>) -> tensor<4x4xf64>\n'
        '%1 = "stablehlo.convert"(%0) : (tensor<4x4xf64>) -> tensor<4x4xf32>\n'
    )
    c = count_converts(stablehlo)
    assert c.f32_to_f64 == 1 and c.f64_to_f32 == 1, c.as_dict()
