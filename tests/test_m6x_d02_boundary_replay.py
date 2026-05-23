from __future__ import annotations

from pathlib import Path

import jax
import pytest

from gpuwrf.integration.d02_replay import DEFAULT_REPLAY_RUN_DIR, ReplayConfig, run_replay_proof


def _has_gpu() -> bool:
    return any(device.platform == "gpu" for device in jax.devices())


def test_d02_boundary_replay_10_step_smoke(tmp_path):
    if not _has_gpu():
        pytest.skip("d02 replay smoke requires a visible JAX GPU")
    if not Path(DEFAULT_REPLAY_RUN_DIR).exists():
        pytest.skip(f"Gen2 d02 replay run not present: {DEFAULT_REPLAY_RUN_DIR}")

    payload = run_replay_proof(
        run_dir=DEFAULT_REPLAY_RUN_DIR,
        output_fields_path=tmp_path / "d02_replay_fields.npz",
        replay_config=ReplayConfig(duration_s=10.0, dt_s=1.0, n_acoustic=4, radiation_cadence_steps=10),
        trace_dir=tmp_path / "trace_d02_replay_10_steps",
        include_static_audit=False,
    )

    assert payload["steps"] == 10
    assert payload["first_nonfinite_step"] is None
    assert payload["diagnostics"]["all_state_leaves_finite"] is True
    assert payload["transfer_audit"]["trace"]["post_init_total_bytes"] == 0
    for field, shapes in payload["comparison"]["shapes"].items():
        assert shapes["forecast"] == shapes["reference"], field
