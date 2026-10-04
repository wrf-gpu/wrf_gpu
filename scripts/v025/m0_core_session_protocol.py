#!/usr/bin/env python3
"""JAX-free Amendment-6 control plane for the one M0-CORE device session.

This module contains no device implementation.  It is the fail-closed state
machine that the manager-owned lock holder calls: one content-addressed receipt
is spent once, W1 cold compilation receives the pre-registered two-of-three
decision, and only a definitive W1 pass may reach W2 and W3.  The CPU sprint
executes this control plane with injected CPU stubs; future device evidence
must come from the real lock-holder callbacks.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import gpu_window_registry as window_registry


SCHEMA = "wrf_gpu2.v025.m0.core_session_protocol.v2"
SESSION_LABEL = window_registry.SESSION_LABEL
GLOBAL_DEADLINE_SECONDS = 4_500.0
COLD_THRESHOLD_SECONDS = 600.0
MAX_COLD_PROCESSES = 3
REQUIRED_MANAGERS = ("0:2", "0:3")
ACCELERATOR_ROOTS = ("jax", "jaxlib", "gpuwrf")
#: The fresh 12-rank CPU-WRF arm runs before receipt spend and lock acquisition.
CPU_PREFLIGHT_TIMEOUT_SECONDS = 420.0
#: C1 only verifies the pre-staged CPU output against W1 and prepares the pair.
C1_COMPARATOR_LOCK_HOLD_SECONDS = 60.0
#: The C1 sub-gate that Amendment 5 requires between W1 and W2.  Both halves
#: must be green or every later stage is mechanically unreachable.
C1_STAGE = "C1_SAME_RESULT_AND_FRESH_CPU_COMPARATOR"
C1_COMPONENTS = (
    "w1_same_result_wrfout_binding_verify_file",
    "fresh_12_rank_cpu_wrf_comparator",
)
HELD_STAGE_RESERVATIONS: tuple[tuple[str, float], ...] = (
    ("W1_COLD_DECISION", 1_800.0),
    ("W1_CACHED_AND_CORRECTNESS", 360.0),
    (C1_STAGE, C1_COMPARATOR_LOCK_HOLD_SECONDS),
    ("W2_PROFILED_CAPTURE", 1_400.0),
    ("W3_CLEAN_MATCHED_ARM", 600.0),
)
HELD_RESERVATION_SECONDS = sum(seconds for _, seconds in HELD_STAGE_RESERVATIONS)
W2_W3_REQUIRED_SECONDS = (
    dict(HELD_STAGE_RESERVATIONS)["W2_PROFILED_CAPTURE"]
    + dict(HELD_STAGE_RESERVATIONS)["W3_CLEAN_MATCHED_ARM"]
)
HELD_HEADROOM_SECONDS = GLOBAL_DEADLINE_SECONDS - HELD_RESERVATION_SECONDS


class SessionRefusal(RuntimeError):
    """The receipt, cold discriminator, or stage graph failed closed."""


def validate_held_budget() -> dict[str, Any]:
    """Validate the whole admissible worst branch before coordination/spend."""

    expected = {
        "W1_COLD_DECISION": 1_800.0,
        "W1_CACHED_AND_CORRECTNESS": 360.0,
        C1_STAGE: 60.0,
        "W2_PROFILED_CAPTURE": 1_400.0,
        "W3_CLEAN_MATCHED_ARM": 600.0,
    }
    observed = dict(HELD_STAGE_RESERVATIONS)
    if observed != expected:
        raise SessionRefusal(
            f"held-session reservations changed: {observed!r} != {expected!r}"
        )
    total = sum(observed.values())
    if total > GLOBAL_DEADLINE_SECONDS:
        raise SessionRefusal(
            f"held-session reservations require {total:g}s but the global "
            f"cap is {GLOBAL_DEADLINE_SECONDS:g}s"
        )
    headroom = GLOBAL_DEADLINE_SECONDS - total
    if total != 4_220.0 or headroom != 280.0:
        raise SessionRefusal(
            f"Amendment-6 budget must be 4220s + 280s headroom, got "
            f"{total:g}s + {headroom:g}s"
        )
    return {
        "status": "PASS",
        "reservations": observed,
        "reservation_seconds": total,
        "global_deadline_seconds": GLOBAL_DEADLINE_SECONDS,
        "headroom_seconds": headroom,
        "worst_branch": "three threshold-censored cold attempts",
    }


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def assert_accelerator_free() -> None:
    contaminated = sorted(
        name
        for name in sys.modules
        if any(
            name == root or name.startswith(f"{root}.")
            for root in ACCELERATOR_ROOTS
        )
    )
    if contaminated:
        raise SessionRefusal(
            "M0-CORE session control process imported accelerator roots: "
            f"{contaminated[:12]}"
        )


def session_contract() -> dict[str, Any]:
    """Return the deterministic receipt and stage-graph contract."""

    budget = validate_held_budget()
    payload = {
        "schema": SCHEMA,
        "label": SESSION_LABEL,
        "global_deadline_seconds": GLOBAL_DEADLINE_SECONDS,
        "held_budget": budget,
        "cpu_preflight": {
            "position": "before receipt spend and canonical lock acquisition",
            "timeout_seconds": CPU_PREFLIGHT_TIMEOUT_SECONDS,
            "fresh_12_rank_cpu_wrf": True,
            "device_action": False,
        },
        "canonical_lock_acquisitions": 1,
        "receipt": {
            "required_managers": list(REQUIRED_MANAGERS),
            "affirmative_verbatim_required": True,
            "content_addressed": True,
            "single_use": True,
            "spent_at_authorization": True,
            "nonqueueing": True,
            "relinquish_immediately_on_manager_request": True,
            "invalid_after": [
                "preemption",
                "process_leak",
                "identity_change",
                "global_deadline",
            ],
        },
        "cold_protocol": {
            "threshold_seconds": COLD_THRESHOLD_SECONDS,
            "maximum_processes": MAX_COLD_PROCESSES,
            "unique_empty_normal_cache_per_process": True,
            "decision": "two PASS_COLD or two FAIL_COLD",
            "third_process": "only if first two disagree",
            "first_cold_miss_suppresses_later_cold_attempts": False,
        },
        "stage_order": [
            "W1_COLD_DECISION",
            "W1_CACHED_AND_CORRECTNESS",
            C1_STAGE,
            "W2_PROFILED_CAPTURE",
            "W3_CLEAN_MATCHED_ARM",
            "LOCK_RELEASE",
            "POSTLOCK_CPU_ANALYSIS",
            "POSTLOCK_M0_CORE_FINALIZER",
        ],
        "c1_subgate": {
            "position": "between W1_CACHED_AND_CORRECTNESS and W2_PROFILED_CAPTURE",
            "components": list(C1_COMPONENTS),
            "both_required": True,
            "suppresses_on_failure": ["W2", "W3"],
            "runs_inside_the_one_held_lock": True,
            "expected_lock_hold_seconds": C1_COMPARATOR_LOCK_HOLD_SECONDS,
            "coordination_disclosure": (
                "the fresh 12-rank CPU-WRF arm runs before coordination; C1 "
                "spends at most 60 s in-lock verifying the hash-bound output "
                "and preparing the exact W2/W3 pair"
            ),
        },
        "suppression": {
            "definitive_w1_failure": ["W1_CACHED_AND_CORRECTNESS", "W2", "W3"],
            "c1_wrfout_binding_failure": ["W2", "W3"],
            "c1_comparator_failure": ["W2", "W3"],
            "safety_correctness_identity_failure": ["every_later_stage"],
            "receipt_reuse_or_spend_order_failure": ["every_stage"],
            "global_deadline_exceeded": ["every_later_stage"],
            "preemption_or_process_leak": ["every_later_stage"],
            "first_cold_miss": [],
        },
        "machine_authority": {
            "fields": ["cpu_closure_gate", "milestone_partition"],
            "never": "top_level_status_string",
            "rule": (
                "automation keys M0-CORE acceptance on the structured gate "
                "fields only; the descriptive status string is not a gate"
            ),
        },
        "deferred": {
            "native_pallas": "M2_BACKEND_BAKEOFF_ENTRY",
            "compiler_peak_live_buffers":
                "M1_SAME_SET_CANDIDATE_DIAGNOSIS_OR_M2_BACKEND_BAKEOFF",
            "ncu_fields": "M2",
        },
    }
    payload["contract_sha256"] = canonical_sha256(payload)
    return payload


def validate_receipt_shape(receipt: dict[str, Any]) -> dict[str, Any]:
    """Validate dual-manager, content-addressed, single-use receipt semantics."""

    assert_accelerator_free()
    if receipt.get("window") != SESSION_LABEL:
        raise SessionRefusal("receipt does not authorize the exact M0-CORE session")
    replies = receipt.get("replies")
    if not isinstance(replies, dict):
        raise SessionRefusal("receipt reply map is missing")
    for manager in REQUIRED_MANAGERS:
        reply = replies.get(manager)
        if (
            not isinstance(reply, dict)
            or reply.get("affirmative") is not True
            or not str(reply.get("verbatim", "")).strip()
        ):
            raise SessionRefusal(
                f"receipt lacks affirmative verbatim approval from {manager}"
            )
    if receipt.get("spent") is not None:
        raise SessionRefusal("receipt is already spent")
    fingerprint_payload = {
        "window": receipt["window"],
        "requested_at_utc": receipt.get("requested_at_utc", ""),
        "replies": {
            manager: {
                "affirmative": True,
                "verbatim": str(replies[manager]["verbatim"]),
                "received_at_utc": str(
                    replies[manager].get("received_at_utc", "")
                ),
            }
            for manager in REQUIRED_MANAGERS
        },
    }
    fingerprint = canonical_sha256(fingerprint_payload)
    supplied = receipt.get("fingerprint")
    if supplied is not None and supplied != fingerprint:
        raise SessionRefusal("receipt content address differs from its content")
    return {
        "status": "PASS",
        "label": SESSION_LABEL,
        "fingerprint": fingerprint,
        "managers": list(REQUIRED_MANAGERS),
        "single_use": True,
    }


def _classify_attempt(attempt: dict[str, Any], index: int) -> str:
    required = {"process_id", "cache_path", "readiness_seconds", "threshold_stop"}
    if not required.issubset(attempt):
        raise SessionRefusal(
            f"cold attempt {index} lacks {sorted(required - set(attempt))}"
        )
    process_id = attempt["process_id"]
    cache_path = str(attempt["cache_path"])
    readiness = attempt["readiness_seconds"]
    threshold_stop = attempt["threshold_stop"]
    if (
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
        or not cache_path
        or not isinstance(threshold_stop, bool)
    ):
        raise SessionRefusal(f"cold attempt {index} identity/censoring is invalid")
    if readiness is None:
        if threshold_stop is not True:
            raise SessionRefusal(
                f"cold attempt {index} has no readiness and was not stopped at 600s"
            )
        return "FAIL_COLD"
    if (
        isinstance(readiness, bool)
        or not isinstance(readiness, (int, float))
        or not math.isfinite(float(readiness))
        or float(readiness) < 0.0
    ):
        raise SessionRefusal(f"cold attempt {index} readiness is invalid")
    if threshold_stop:
        raise SessionRefusal(
            f"cold attempt {index} claims readiness after a threshold stop"
        )
    if float(readiness) > COLD_THRESHOLD_SECONDS:
        raise SessionRefusal(
            f"cold attempt {index} exceeded 600s without threshold censoring"
        )
    return "PASS_COLD"


def classify_cold_attempts(
    attempts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen two-of-three threshold-censored cold decision."""

    assert_accelerator_free()
    if not 1 <= len(attempts) <= MAX_COLD_PROCESSES:
        raise SessionRefusal("cold protocol requires one to three attempts")
    process_ids = [attempt.get("process_id") for attempt in attempts]
    cache_paths = [str(attempt.get("cache_path", "")) for attempt in attempts]
    if len(set(process_ids)) != len(process_ids):
        raise SessionRefusal("cold attempts did not use unique processes")
    if len(set(cache_paths)) != len(cache_paths):
        raise SessionRefusal("cold attempts did not use unique empty caches")
    classifications = [
        _classify_attempt(attempt, index)
        for index, attempt in enumerate(attempts, start=1)
    ]
    if len(classifications) == 3 and classifications[0] == classifications[1]:
        raise SessionRefusal(
            "third cold process ran although the first two already decided"
        )
    passes = classifications.count("PASS_COLD")
    failures = classifications.count("FAIL_COLD")
    if passes >= 2:
        decision = "PASS"
    elif failures >= 2:
        decision = "FAIL"
    else:
        decision = "CONTINUE"
    if decision == "CONTINUE" and len(classifications) == 3:
        raise SessionRefusal("three cold attempts did not produce a decision")
    return {
        "status": "PASS",
        "decision": decision,
        "classifications": classifications,
        "passes": passes,
        "failures": failures,
        "attempts_used": len(classifications),
        "third_required": (
            len(classifications) == 2
            and classifications[0] != classifications[1]
        ),
        "threshold_seconds": COLD_THRESHOLD_SECONDS,
    }


@dataclass(frozen=True)
class SessionStageResult:
    name: str
    status: str
    detail: Any


#: The frozen post-cold stage order.  C1 sits between W1 and W2 so that no
#: profiled or clean device stage is reachable until the same-result wrfout and
#: the fresh CPU-WRF comparator are both green.
POST_COLD_STAGE_ORDER = (
    "W1_CACHED_AND_CORRECTNESS",
    C1_STAGE,
    "W2_PROFILED_CAPTURE",
    "W3_CLEAN_MATCHED_ARM",
)

#: What each stage failure makes mechanically unreachable.
LATER_STAGES = {
    "W1_CACHED_AND_CORRECTNESS": [C1_STAGE, "W2", "W3"],
    C1_STAGE: ["W2", "W3"],
    "W2_PROFILED_CAPTURE": ["W3"],
    "W3_CLEAN_MATCHED_ARM": [],
}


class SessionDeadline:
    """One monotonic global deadline shared by every stage of the session."""

    def __init__(
        self,
        *,
        budget_seconds: float = GLOBAL_DEADLINE_SECONDS,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if (
            isinstance(budget_seconds, bool)
            or not isinstance(budget_seconds, (int, float))
            or not math.isfinite(float(budget_seconds))
            or not 0.0 < float(budget_seconds) <= GLOBAL_DEADLINE_SECONDS
        ):
            raise SessionRefusal(
                "session deadline budget must be in (0, "
                f"{GLOBAL_DEADLINE_SECONDS:g}] seconds"
            )
        self._clock = clock or (lambda: __import__("time").monotonic_ns())
        self.budget_seconds = float(budget_seconds)
        self.started_ns = int(self._clock())

    def elapsed_seconds(self) -> float:
        return (int(self._clock()) - self.started_ns) / 1e9

    def remaining_seconds(self) -> float:
        return self.budget_seconds - self.elapsed_seconds()

    def check(self, stage: str) -> None:
        remaining = self.remaining_seconds()
        if remaining <= 0.0:
            raise SessionRefusal(
                f"global {self.budget_seconds:g}s session deadline exceeded "
                f"before {stage}; every later stage is suppressed"
            )

    def check_after(self, stage: str) -> None:
        """Reject a stage that returned only after the absolute deadline."""

        remaining = self.remaining_seconds()
        if remaining <= 0.0:
            raise SessionRefusal(
                f"global {self.budget_seconds:g}s session deadline exceeded "
                f"during {stage}; every later stage is suppressed"
            )

    def require(self, stage: str, needed_seconds: float) -> float:
        """Refuse a stage that cannot finish inside the global deadline."""

        self.check(stage)
        remaining = self.remaining_seconds()
        if float(needed_seconds) > remaining:
            raise SessionRefusal(
                f"{stage} needs {float(needed_seconds):g}s but only "
                f"{remaining:.3f}s of the global {self.budget_seconds:g}s "
                "session deadline remain; no extension path exists"
            )
        return remaining

    def timeout_for(self, stage: str, stage_cap_seconds: float) -> float:
        """Clamp one blocking wait to the absolute remaining session time."""

        if (
            isinstance(stage_cap_seconds, bool)
            or not isinstance(stage_cap_seconds, (int, float))
            or not math.isfinite(float(stage_cap_seconds))
            or float(stage_cap_seconds) <= 0.0
        ):
            raise SessionRefusal(f"{stage} has an invalid timeout cap")
        self.check(stage)
        return min(float(stage_cap_seconds), max(0.0, self.remaining_seconds()))


def execute_session_graph(
    *,
    cold_runner: Callable[[int], dict[str, Any]],
    stage_runner: Callable[[str], Any],
    deadline: SessionDeadline | None = None,
) -> dict[str, Any]:
    """Execute the prospective graph with injected manager-owned callbacks.

    Callback failures are safety failures.  A threshold-censored cold miss is
    data, not an exception, so it never suppresses the next required cold
    discriminator.  ``C1_STAGE`` runs before ``"W2_PROFILED_CAPTURE"`` and
    ``"W3_CLEAN_MATCHED_ARM"``; if it raises, both are unreachable.
    """

    assert_accelerator_free()
    attempts: list[dict[str, Any]] = []
    results: list[SessionStageResult] = []
    decision: dict[str, Any] | None = None
    for index in range(1, MAX_COLD_PROCESSES + 1):
        try:
            if deadline is not None:
                deadline.require(f"W1_COLD_{index}", COLD_THRESHOLD_SECONDS)
            attempt = cold_runner(index)
            if deadline is not None:
                deadline.check_after(f"W1_COLD_{index}")
            attempts.append(attempt)
            decision = classify_cold_attempts(attempts)
        except Exception as exc:
            return {
                "schema": SCHEMA,
                "status": "BLOCKED",
                "cold_attempts": attempts,
                "cold_decision": decision,
                "stages": [
                    result.__dict__ for result in results
                ],
                "first_failure": {
                    "stage": f"W1_COLD_{index}",
                    "kind": "SAFETY_OR_IDENTITY_FAILURE",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                "suppressed": [
                    "W1_CACHED_AND_CORRECTNESS",
                    C1_STAGE,
                    "W2",
                    "W3",
                ],
            }
        results.append(
            SessionStageResult(
                name=f"W1_COLD_{index}",
                status=decision["classifications"][-1],
                detail=attempt,
            )
        )
        if decision["decision"] != "CONTINUE":
            break
    assert decision is not None
    if decision["decision"] == "FAIL":
        return {
            "schema": SCHEMA,
            "status": "BLOCKED",
            "cold_attempts": attempts,
            "cold_decision": decision,
            "stages": [result.__dict__ for result in results],
            "first_failure": {
                "stage": "W1_COLD_DECISION",
                "kind": "DEFINITIVE_W1_FAILURE",
            },
            "suppressed": [
                "W1_CACHED_AND_CORRECTNESS",
                C1_STAGE,
                "W2",
                "W3",
            ],
        }
    for stage in POST_COLD_STAGE_ORDER:
        try:
            if deadline is not None:
                deadline.check(stage)
            detail = stage_runner(stage)
            if deadline is not None:
                deadline.check_after(stage)
        except Exception as exc:
            later = LATER_STAGES[stage]
            return {
                "schema": SCHEMA,
                "status": "BLOCKED",
                "cold_attempts": attempts,
                "cold_decision": decision,
                "stages": [result.__dict__ for result in results],
                "first_failure": {
                    "stage": stage,
                    "kind": (
                        "C1_SUBGATE_FAILURE"
                        if stage == C1_STAGE
                        else "SAFETY_CORRECTNESS_OR_IDENTITY_FAILURE"
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                },
                "suppressed": later,
            }
        results.append(SessionStageResult(stage, "PASS", detail))
    executed = [result.name for result in results]
    if executed[-len(POST_COLD_STAGE_ORDER):] != list(POST_COLD_STAGE_ORDER):
        raise SessionRefusal("session stage order was not the frozen order")
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "cold_attempts": attempts,
        "cold_decision": decision,
        "stages": [result.__dict__ for result in results],
        "stage_order_executed": executed,
        "first_failure": None,
        "suppressed": [],
        "lock_acquisitions_required": 1,
        "receipt_spends_required": 1,
        "postlock_analysis_required": True,
        "elapsed_seconds": (
            deadline.elapsed_seconds() if deadline is not None else None
        ),
    }


__all__ = [
    "C1_COMPARATOR_LOCK_HOLD_SECONDS",
    "C1_COMPONENTS",
    "C1_STAGE",
    "COLD_THRESHOLD_SECONDS",
    "CPU_PREFLIGHT_TIMEOUT_SECONDS",
    "GLOBAL_DEADLINE_SECONDS",
    "HELD_HEADROOM_SECONDS",
    "HELD_RESERVATION_SECONDS",
    "HELD_STAGE_RESERVATIONS",
    "LATER_STAGES",
    "MAX_COLD_PROCESSES",
    "POST_COLD_STAGE_ORDER",
    "SESSION_LABEL",
    "SessionDeadline",
    "SessionRefusal",
    "W2_W3_REQUIRED_SECONDS",
    "assert_accelerator_free",
    "classify_cold_attempts",
    "execute_session_graph",
    "session_contract",
    "validate_held_budget",
    "validate_receipt_shape",
]
