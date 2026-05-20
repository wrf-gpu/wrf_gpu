from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from gpuwrf.physics.mynn_pbl import MynnPBLColumnState, step_mynn_pbl_column
from gpuwrf.validation.tier1_mynn import load_fixture_state


def test_mynn_step_preserves_shapes_and_fp64_dtype():
    state, dt, _ = load_fixture_state()
    out = step_mynn_pbl_column(state, dt, debug=False)
    for name in MynnPBLColumnState.__slots__:
        before = getattr(state, name)
        after = getattr(out, name)
        assert after.shape == before.shape
        assert after.dtype == jnp.float64
        assert np.all(np.isfinite(np.asarray(after)))


def test_mynn_tke_floor_and_qv_nonnegative_on_fixture():
    state, dt, _ = load_fixture_state()
    out = step_mynn_pbl_column(state, dt, debug=False)
    assert np.min(np.asarray(out.tke)) > 0.0
    assert np.min(np.asarray(out.qv)) >= 0.0


def test_mynn_hlo_diff_artifact_empty_when_present():
    path = Path("artifacts/m5/hlo_dump/mynn_pbl_debug_vs_stripped.diff")
    if not path.exists():
        return
    assert path.stat().st_size == 0
