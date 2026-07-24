"""Focused restart/activation tests for the CPU-only moist-theta bisection."""

from __future__ import annotations

import json
from types import SimpleNamespace

import jax
import numpy as np

from scripts import v0234_moist_theta_activation_bisection as runner
from scripts import v0234_moist_theta_interface_cpu_ab as common


def _carry(*, h: float, active: str | None = None):
    fields = {
        name: np.zeros((2, 2, 2), dtype=np.float64)
        for name in set(runner.HEALTH_FIELDS) | set(runner.HYDROMETEORS)
    }
    fields["theta"].fill(300.0)
    if active is not None:
        fields[active][0, 0, 0] = 1.0e-12
    return SimpleNamespace(
        state=SimpleNamespace(**fields),
        h_diabatic=np.full((2, 2, 2), h, dtype=np.float64),
    )


def test_activation_is_exact_phase_reservoir_positivity_not_fitted_h_threshold() -> None:
    rounding = runner._activation_stats(np, _carry(h=1.0e-14))
    assert not rounding["real_activation"]
    assert rounding["hydrometeor_positive_total"] == 0
    assert rounding["h_diabatic"]["nonzero_count"] == 8
    assert (
        rounding["h_diabatic"]["classification"]
        == "ROUNDING_ONLY_WITH_ZERO_PHASE_RESERVOIRS"
    )

    nonroundoff = runner._activation_stats(np, _carry(h=1.0e-3))
    assert nonroundoff["real_activation"]
    assert nonroundoff["h_diabatic"]["nonroundoff"]
    assert nonroundoff["h_diabatic"]["classification"] == "PHYSICAL_NONROUNDOFF_H"

    active = runner._activation_stats(np, _carry(h=1.0e-14, active="qc"))
    assert active["real_activation"]
    assert active["hydrometeor_positive_total"] == 1
    assert active["hydrometeors"]["qc"]["positive_count"] == 1
    assert active["h_diabatic"]["classification"] == "PHYSICAL_PHASE_COUPLED"


def test_invalid_species_and_nonfinite_selected_state_fail_health() -> None:
    carry = _carry(h=0.0)
    carry.state.qi[0, 0, 0] = -1.0
    carry.state.theta[0, 0, 0] = np.nan
    stats = runner._activation_stats(np, carry)
    assert not stats["all_hydrometeors_finite_nonnegative"]
    assert not stats["selected_state_finite"]


def test_checkpoint_pickle_manifest_roundtrip_and_recovery(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path)
    value = {
        "state": np.arange(12, dtype=np.float64).reshape(3, 4),
        "clock": np.asarray(7, dtype=np.int32),
    }
    row = runner._persist_carry(common, jax, np, value, role="checkpoint", step=7)
    assert row["reread_identity"]
    assert row["all_leaves_finite"]
    assert row["step"] == 7
    recovered = runner._load_persisted(common, jax, np, row)
    np.testing.assert_array_equal(recovered["state"], value["state"])
    np.testing.assert_array_equal(recovered["clock"], value["clock"])

    # An idempotent retry authenticates and reuses the exact existing object.
    again = runner._persist_carry(common, jax, np, value, role="checkpoint", step=7)
    assert again == row


def test_checkpoint_tamper_is_fail_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "RUN_DIR", tmp_path)
    value = {"state": np.ones((2, 2), dtype=np.float64)}
    row = runner._persist_carry(common, jax, np, value, role="checkpoint", step=1)
    path = runner.Path(row["manifest_path"])
    payload = json.loads(path.read_text())
    payload["sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    try:
        runner._load_persisted(common, jax, np, row)
    except RuntimeError as exc:
        assert "manifest file hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered checkpoint was accepted")


def test_history_contract_is_sequential_bounded_and_gpu_free() -> None:
    history = runner._history_template(runner.REQUIRED_ENV, runner.Path("/candidate"))
    assert history["bound"] == {
        "first_step": 1,
        "last_step": 200,
        "last_sim_time_seconds": 1200.0,
        "checkpoint_every_steps": 25,
        "rationale": "Step 200 is d03 00:20 and has authenticated CPU-WRF and Retry20 anchors.",
    }
    assert history["activation_predicate"]["monotonicity_assumed"] is False
    assert history["activation_predicate"]["scan"] == "every completed step sequentially"
    assert history["scope"]["gpu_queries"] == 0
    assert history["scope"]["model_changes"] == 0


def test_canonical_hash_ignores_only_its_own_proof_member() -> None:
    payload = {"schema": "x", "value": 3}
    digest = runner._canonical(payload)
    assert runner._canonical({**payload, "proof_sha256": digest}) == digest
    assert runner._canonical({**payload, "value": 4}) != digest


def test_source_chain_accepts_any_physical_hydrometeor_mass_not_only_cloud_water() -> None:
    common_fields = [
        "theta",
        "qv",
        "p_perturbation",
        "ph_perturbation",
        "u",
        "v",
        "w",
    ]
    assert runner._source_effect_reaches_thermo_pressure_momentum(
        [*common_fields, "qi", "Ni"]
    )
    assert not runner._source_effect_reaches_thermo_pressure_momentum(
        [*common_fields, "Ni"]
    )
