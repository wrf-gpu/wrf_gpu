from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.runtime.domain_tree import run_domain_tree_callbacks
from scripts import v0234_v10_rootcause_step200 as target


@dataclass(frozen=True)
class _Carry:
    value: int


def _fresh_authorization(nonce: str = "1" * 64) -> dict:
    payload = {
        "schema": "gpuwrf.v0234.v10-rootcause-step200-manager-authorization.v1",
        "verdict": "MANAGER_GPU_AUTHORIZED",
        "manager_pane": "0:1",
        "nonce": nonce,
        "namespace": f"v0234_gpt_v10_rootcause_{nonce[:16]}_step200",
        "lock_label": target.LOCK_LABEL,
        "candidate_model_commit": target.CANDIDATE_COMMIT,
        "candidate_src_gpuwrf_tree": target.CANDIDATE_SRC_TREE,
        "authorized_arm_count": 1,
        "authorized_model_processes": 1,
        "root_step_upper_bound": 23,
        "d03_steps_beyond_200_authorized": False,
        "baseline_rerun_authorized": False,
        "horizon_extension_authorized": False,
        "existing_cpu_reference_only": True,
        "autotune_authority": {
            "policy": "read-only authenticated completed reference pin",
            "path": str(target.AUTOTUNE_PIN),
            "sha256": target.AUTOTUNE_PIN_SHA256,
        },
        "scientific_gate": {
            "V_strict_less_than": target.REFERENCE_RMSE["V"],
            "V10_strict_less_than": target.REFERENCE_RMSE["V10"],
            "other_strict_fields_no_worse_than_existing_reference": True,
            "finite_and_static_identity_required": True,
        },
    }
    payload["proof_sha256"] = target._canonical(payload)
    return payload


def test_fresh_manager_authorization_is_self_hashed_and_exact() -> None:
    payload = _fresh_authorization()
    assert target._validate_authorization_payload(payload) == payload["proof_sha256"]
    assert payload["candidate_model_commit"] == target.CANDIDATE_COMMIT


def test_authorization_rejects_consumed_nonce_and_namespace_drift() -> None:
    consumed = _fresh_authorization(next(iter(target.CONSUMED_NONCES)))
    with pytest.raises(RuntimeError, match="already consumed"):
        target._validate_authorization_payload(consumed)

    drifted = _fresh_authorization()
    drifted["namespace"] = "wrong"
    drifted["proof_sha256"] = target._canonical(drifted)
    with pytest.raises(RuntimeError, match="authorization mismatch"):
        target._validate_authorization_payload(drifted)


def test_step200_classifier_requires_both_winds_and_all_frozen_fields() -> None:
    improved = {field: value * 0.99 for field, value in target.REFERENCE_RMSE.items()}
    static = {field: True for field in target.STATIC_FIELDS}
    assert target.classify_step200(improved, finite=True, static_identity=static)[
        "passed"
    ]

    for field in ("V", "V10"):
        red = dict(improved)
        red[field] = target.REFERENCE_RMSE[field]
        decision = target.classify_step200(red, finite=True, static_identity=static)
        assert not decision["passed"]
        assert not decision[f"{field}_strict_improvement"]

    frozen_red = dict(improved)
    frozen_red["U"] = target.REFERENCE_RMSE["U"] + 1.0e-12
    decision = target.classify_step200(
        frozen_red, finite=True, static_identity=static
    )
    assert not decision["passed"]
    assert set(decision["frozen_field_regressions"]) == {"U"}


def test_step200_classifier_fails_closed_on_finite_or_static_regression() -> None:
    improved = {field: value * 0.99 for field, value in target.REFERENCE_RMSE.items()}
    static = {field: True for field in target.STATIC_FIELDS}
    assert not target.classify_step200(
        improved, finite=False, static_identity=static
    )["passed"]
    static["HGT"] = False
    assert not target.classify_step200(
        improved, finite=True, static_identity=static
    )["passed"]


def test_output_callback_stops_only_after_exact_synchronous_step200_write() -> None:
    calls = []

    def writer(name, step, carry):
        calls.append((name, step, carry))
        return {"written": True}

    output = target._Step200Output(writer)
    try:
        output("d03", 200, "carry")
    except target._Step200Reached:
        pass
    else:
        raise AssertionError("exact stop signal was not raised")
    assert calls == [("d03", 200, "carry")]
    assert output.calls[0]["own_step"] == 200


def test_scheduler_dispatches_no_d03_step_beyond_authorized_alarm() -> None:
    hierarchy = DomainHierarchy.from_edges(
        ("d01", "d02", "d03"),
        (
            DomainNest("d01", "d02", 3, 30, 20),
            DomainNest("d02", "d03", 3, 10, 10),
        ),
        max_dom=3,
    )
    advances: list[tuple[str, int, int]] = []
    writes: list[tuple[str, int, int]] = []

    def advance(name: str, carry: _Carry, start: int, n_steps: int) -> _Carry:
        advances.append((name, int(start), int(n_steps)))
        return _Carry(carry.value + int(n_steps))

    def force(_edge, _parent: _Carry, child: _Carry) -> _Carry:
        return child

    def writer(name: str, step: int, carry: _Carry):
        writes.append((name, int(step), carry.value))
        return {"written": True}

    with pytest.raises(target._Step200Reached):
        run_domain_tree_callbacks(
            hierarchy,
            {name: _Carry(0) for name in hierarchy.order},
            root_steps=23,
            advance=advance,
            force=force,
            output=target._Step200Output(writer),
            output_alarm_steps={"d03": (200,)},
            block_between=False,
        )

    d03_advances = [row for row in advances if row[0] == "d03"]
    assert max(start + n_steps - 1 for _, start, n_steps in d03_advances) == 200
    assert all(start <= 200 for _, start, _ in d03_advances)
    assert writes == [("d03", 200, 200)]


def test_launcher_is_one_locked_candidate_process_with_no_extension() -> None:
    launcher = (
        target.SPRINT / "v10-h5-step200-repaired-ready.sh"
    ).read_text()
    assert launcher.count("scripts/with_gpu_lock.sh") == 1
    assert launcher.count("-m scripts.v0234_v10_rootcause_step200") == 1
    assert "--label v0234-gpt-v10-rootcause" in launcher
    assert "2700s" in launcher
    assert "reference|pinned" not in launcher
    assert "root_steps=23" not in launcher
    assert "--authorization" in launcher
    assert "refusing permanently consumed nonce" in launcher
    assert "GPUWRF_V10_ROOTCAUSE_NONCE=\"$NONCE\"" in launcher


@pytest.mark.parametrize(
    "filename",
    (
        "GPU_STEP200_AUTHORIZATION.json",
        "GPU_STEP200_AUTHORIZATION_8520924a5f97acc1.json",
        "GPU_STEP200_AUTHORIZATION_ca1df30181823579.json",
    ),
)
def test_historical_authorization_is_rejected_as_consumed(filename: str) -> None:
    historical = json.loads(
        (target.SPRINT / filename).read_text()
    )
    with pytest.raises(RuntimeError, match="already consumed"):
        target._validate_authorization_payload(historical)


def test_cold_step0_anchor_requires_non_qke_identity_and_exact_historical_drift() -> None:
    cold = {
        name: np.asarray([index, index + 0.5], dtype=np.float64)
        for index, name in enumerate(
            ("mavail", "qke", "roughness_m", "theta"), start=1
        )
    }
    retained = {name: value.copy() for name, value in cold.items()}
    for name in target.EXPECTED_RETAINED_STEP0_MISMATCHES:
        retained[name] = retained[name] + 1.0
    expected = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": target._array_sha256(np, value),
            "finite": True,
        }
        for name, value in cold.items()
    }
    proof = {"repaired_candidate": {"cold_state": {"arrays": expected}}}
    row = target._validate_cold_step0_anchor(
        np, cold, retained, proof, cold["qke"]
    )
    assert row["gpu_cold_vs_cpu_candidate_mismatched"] == []
    assert row["qke_backend_admission"]["passed"]
    assert tuple(row["historical_retained_mismatches"]) == (
        target.EXPECTED_RETAINED_STEP0_MISMATCHES
    )

    drifted = {name: value.copy() for name, value in cold.items()}
    drifted["theta"] += 1.0
    with pytest.raises(RuntimeError, match="non_qke=.*theta"):
        target._validate_cold_step0_anchor(
            np, drifted, retained, proof, cold["qke"]
        )


def test_cold_step0_anchor_admits_only_bounded_qke_backend_delta() -> None:
    cold = {
        name: np.asarray([index, index + 0.5], dtype=np.float64)
        for index, name in enumerate(
            ("mavail", "qke", "roughness_m", "theta"), start=1
        )
    }
    retained = {name: value.copy() for name, value in cold.items()}
    for name in target.EXPECTED_RETAINED_STEP0_MISMATCHES:
        retained[name] = retained[name] + 1.0
    expected = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": target._array_sha256(np, value),
            "finite": True,
        }
        for name, value in cold.items()
    }
    proof = {"repaired_candidate": {"cold_state": {"arrays": expected}}}
    cpu_qke = cold["qke"].copy()

    bounded = {name: value.copy() for name, value in cold.items()}
    bounded["qke"] += 1.0e-7
    row = target._validate_cold_step0_anchor(
        np, bounded, retained, proof, cpu_qke
    )
    assert row["gpu_cold_vs_cpu_candidate_mismatched"] == ["qke"]
    assert row["non_qke_byte_mismatches"] == []
    assert row["qke_backend_admission"]["passed"]

    over_envelope = {name: value.copy() for name, value in cold.items()}
    over_envelope["qke"] += target.QKE_BACKEND_MAX_ABS_LIMIT * 2.0
    with pytest.raises(RuntimeError, match="qke_envelope=False"):
        target._validate_cold_step0_anchor(
            np, over_envelope, retained, proof, cpu_qke
        )
