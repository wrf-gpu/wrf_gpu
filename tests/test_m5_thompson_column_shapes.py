from __future__ import annotations

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.thompson_column import ThompsonColumnState, density_from_pressure_temperature, step_thompson_column
from gpuwrf.physics.thompson_column_debug_stripped import step_thompson_column_debug_stripped
from gpuwrf.profiling.budget import compiled_text
from gpuwrf.validation.tier1_thompson import load_fixture_state


def test_step_preserves_pytree_shapes_and_fp64_dtype():
    state, dt, _ = load_fixture_state()
    out = step_thompson_column(state, dt, debug=False)
    for name in ThompsonColumnState.__slots__:
        before = getattr(state, name)
        after = getattr(out, name)
        assert after.shape == before.shape
        assert after.dtype == jnp.float64
        assert np.all(np.isfinite(np.asarray(after)))


def test_negative_hydrometeor_inputs_clip_to_zero():
    qv = jnp.ones((1, 4), dtype=jnp.float64) * 0.004
    T = jnp.ones((1, 4), dtype=jnp.float64) * 285.0
    p = jnp.ones((1, 4), dtype=jnp.float64) * 90000.0
    rho = density_from_pressure_temperature(p, T, qv)
    neg = jnp.ones((1, 4), dtype=jnp.float64) * -1.0e-6
    state = ThompsonColumnState(qv, neg, neg, neg, neg, neg, neg, neg, T, p, rho)
    out = step_thompson_column(state, 60.0, debug=False)
    for field in ("qc", "qr", "qi", "qs", "qg", "Ni", "Nr"):
        assert np.all(np.asarray(getattr(out, field)) >= 0.0)


def test_debug_false_hlo_has_no_debug_assert_ops():
    state, dt, _ = load_fixture_state()
    prod = compiled_text(step_thompson_column.lower(state, dt, debug=False).compile()).lower()
    stripped = compiled_text(step_thompson_column_debug_stripped.lower(state, dt).compile()).lower()
    for token in ("is-finite", "isfinite", "debug.callback"):
        assert token not in prod
        assert token not in stripped
    assert abs(prod.count("fusion(") - stripped.count("fusion(")) == 0


def test_hlo_diff_artifact_empty_when_present():
    path = Path("artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff")
    if not path.exists():
        return
    assert path.stat().st_size == 0
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
