#!/usr/bin/env python3
"""Single closed registry for every coordination-spending window label.

Reimplemented from the useful registry subset in rejected candidate
``a5e8f5d8``.  Registration is deliberately data, not a permissive predicate:
all receipt validation, lock validation, and spend paths consume this object.

Two namespaces live here and they are **disjoint by construction**:

``REGISTERED_WINDOWS``
    Coordination labels.  Exactly one of these is ever the canonical lock /
    receipt authority for a held session.

``SESSION_STAGE_IDENTITIES``
    Stage identities inside one already-authorized session (``W1``/``W2``/
    ``W3``).  A stage identity names *which part of the graph is running*; it
    is never a lock label and never a receipt window.

The live refusal recorded by ``bb1b1cd5`` happened because the reachable child
compared the exported session lock label with a stage identity.  The
disjointness invariant below is what makes that comparison impossible rather
than merely discouraged: a stage identity can never satisfy
:func:`assert_session_label`, and a coordination label can never satisfy
:func:`assert_stage_identity`.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SCHEMA = "wrf_gpu2.v025.m0.gpu_window_registry.v2"
STAGE_SCHEMA = "wrf_gpu2.v025.m0.session_stage_identity_registry.v1"
HELD_LAUNCH_MODE = "held"
LEGACY_LAUNCH_MODE = "legacy"
LAUNCH_MODES: tuple[str, ...] = (HELD_LAUNCH_MODE, LEGACY_LAUNCH_MODE)


class WindowLabelNotRegistered(RuntimeError):
    """A coordination label is outside the frozen registry."""


class StageIdentityNotRegistered(RuntimeError):
    """A stage identity is outside the frozen held-session stage graph."""


def _guarded_exception(
    exc_type: type[BaseException],
    authority_guard: str,
    message: str,
) -> BaseException:
    """Build an exception whose enforcing refusal site is mechanically named."""

    exception = exc_type(message)
    setattr(exception, "authority_guard", authority_guard)
    return exception


REGISTERED_WINDOWS: Mapping[str, str] = {
    "baseline-census": "M0 baseline census contract",
    "pallas-sm120": "M0/M2 native-backend bake-off contract",
    "reproduction": "M0 published-measurement reproduction contract",
    "m0-core-w1-w2-w3-session": "M0 Amendment 5/6 one-session receipt",
    "m0-autotune0-qualification": "m0_window_parent W1 stage",
    "m0-autotune0-profiled-census": "m0_window_parent W2 stage",
    "m0-autotune0-clean-matched-arm": "m0_window_parent W3 stage",
}
WINDOW_LABELS: tuple[str, ...] = tuple(REGISTERED_WINDOWS)
SESSION_LABEL = "m0-core-w1-w2-w3-session"

#: The frozen stage identities of the one held M0-CORE session, each bound to
#: the exact run ID and per-stage window configuration it may ever carry.  This
#: is the *independent* binding the child re-checks: the parent may not tell a
#: child that ``W1`` means some other run.
SESSION_STAGE_IDENTITIES: Mapping[str, Mapping[str, str]] = {
    "W1": {
        "run_id": "m0-autotune0-qualify-20260728-r3",
        "window_label": "m0-autotune0-qualification",
    },
    "W2": {
        "run_id": "m0-autotune0-profiled-20260728-r3",
        "window_label": "m0-autotune0-profiled-census",
    },
    "W3": {
        "run_id": "m0-autotune0-clean-20260728-r3",
        "window_label": "m0-autotune0-clean-matched-arm",
    },
}
STAGE_IDENTITIES: tuple[str, ...] = tuple(SESSION_STAGE_IDENTITIES)

#: Structural invariant, asserted at import: the two namespaces never overlap.
#: Without this, a future edit could register ``W1`` as a coordination label and
#: silently re-enable the exact live defect this registry now forbids.
_OVERLAP = set(REGISTERED_WINDOWS) & set(SESSION_STAGE_IDENTITIES)
if _OVERLAP:
    raise RuntimeError(
        "coordination labels and stage identities must be disjoint; "
        f"overlapping names: {sorted(_OVERLAP)}"
    )
del _OVERLAP


def is_registered(label: Any) -> bool:
    return isinstance(label, str) and label in REGISTERED_WINDOWS


def is_stage_identity(value: Any) -> bool:
    return isinstance(value, str) and value in SESSION_STAGE_IDENTITIES


def assert_registered(
    label: Any,
    *,
    context: str,
    exc_type: type[BaseException] = WindowLabelNotRegistered,
) -> str:
    if not is_registered(label):
        raise _guarded_exception(
            exc_type,
            "session_label_is_registered_coordination_label",
            (
                f"{context}: unknown window label {label!r} is not registered; "
                f"registered labels are {WINDOW_LABELS}"
            ),
        )
    return str(label)


def assert_lock_label(
    lock_label: Any,
    *,
    expected: Any,
    context: str,
    exc_type: type[BaseException] = WindowLabelNotRegistered,
) -> str:
    observed = assert_registered(
        lock_label, context=f"{context} exported lock", exc_type=exc_type
    )
    wanted = assert_registered(
        expected, context=f"{context} expected", exc_type=exc_type
    )
    if observed != wanted:
        raise exc_type(
            f"{context}: exported lock label {observed!r} does not equal "
            f"authorized label {wanted!r}"
        )
    return observed


def assert_session_label(
    label: Any,
    *,
    context: str,
    exc_type: type[BaseException] = WindowLabelNotRegistered,
) -> str:
    """Accept only a coordination label that may authorize a lock/receipt.

    A stage identity is refused *by name* rather than falling through to the
    generic "unknown label" message, because that is the exact live confusion
    under repair and the refusal must say so.
    """

    if is_stage_identity(label):
        raise _guarded_exception(
            exc_type,
            "session_label_is_not_a_stage_identity",
            (
                f"{context}: {label!r} is a held-session stage identity, not a "
                "coordination label; a stage name may never authorize a canonical "
                f"lock or receipt. Session labels are {WINDOW_LABELS}"
            ),
        )
    return assert_registered(label, context=context, exc_type=exc_type)


def assert_stage_identity(
    stage_identity: Any,
    *,
    expected_run_id: Any,
    context: str,
    exc_type: type[BaseException] = StageIdentityNotRegistered,
) -> Mapping[str, str]:
    """Accept only a frozen stage identity carrying its own bound run ID.

    Returns the frozen per-stage configuration so callers bind against registry
    data instead of re-deriving it.
    """

    if is_registered(stage_identity):
        raise _guarded_exception(
            exc_type,
            "stage_identity_is_not_a_coordination_label",
            (
                f"{context}: {stage_identity!r} is a coordination label, not a "
                "stage identity; the session authority may never be used as a "
                f"stage name. Stage identities are {STAGE_IDENTITIES}"
            ),
        )
    if not is_stage_identity(stage_identity):
        raise _guarded_exception(
            exc_type,
            "stage_identity_is_registered",
            (
                f"{context}: unknown stage identity {stage_identity!r}; registered "
                f"stage identities are {STAGE_IDENTITIES}"
            ),
        )
    configuration = SESSION_STAGE_IDENTITIES[str(stage_identity)]
    if expected_run_id != configuration["run_id"]:
        raise _guarded_exception(
            exc_type,
            "stage_identity_bound_to_run_id",
            (
                f"{context}: stage {stage_identity!r} is bound to run ID "
                f"{configuration['run_id']!r}, not {expected_run_id!r}"
            ),
        )
    return configuration


def assert_launch_context(
    *,
    launch_mode: Any,
    stage_identity: Any,
    run_id: Any,
    coordination_label: Any,
    context: str,
    exc_type: type[BaseException] = WindowLabelNotRegistered,
) -> Mapping[str, str]:
    """Bind launch mode, stage, run ID, and the one allowable label.

    Namespace membership checks performed separately are insufficient: the
    rejected repair accepted every registered coordination label for every
    stage.  This function is the single relation used by both parent and child.
    """

    label = assert_session_label(
        coordination_label,
        context=f"{context} coordination label",
        exc_type=exc_type,
    )
    configuration = assert_stage_identity(
        stage_identity,
        expected_run_id=run_id,
        context=f"{context} stage identity",
        exc_type=exc_type,
    )
    if launch_mode not in LAUNCH_MODES:
        raise _guarded_exception(
            exc_type,
            "launch_mode_is_registered",
            (
                f"{context}: unknown launch mode {launch_mode!r}; "
                f"registered launch modes are {LAUNCH_MODES}"
            ),
        )
    allowable_label = (
        SESSION_LABEL
        if launch_mode == HELD_LAUNCH_MODE
        else configuration["window_label"]
    )
    if label != allowable_label:
        raise _guarded_exception(
            exc_type,
            "launch_context_allows_exact_coordination_label",
            (
                f"{context}: launch mode {launch_mode!r}, stage "
                f"{stage_identity!r}, and run ID {run_id!r} allow coordination "
                f"label {allowable_label!r}, not {label!r}"
            ),
        )
    return {
        "launch_mode": str(launch_mode),
        "stage_identity": str(stage_identity),
        "run_id": str(configuration["run_id"]),
        "coordination_label": label,
        "allowable_coordination_label": str(allowable_label),
        "window_label": str(configuration["window_label"]),
    }


def registry_fingerprint() -> str:
    """Content address of the coordination-label namespace.

    Deliberately unchanged in content: recorded Review-10 evidence pins this
    value, and the stage-identity namespace is a *separate* namespace with its
    own address below.
    """

    encoded = json.dumps(
        {"schema": SCHEMA, "windows": dict(REGISTERED_WINDOWS)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def stage_registry_fingerprint() -> str:
    """Content address of the disjoint held-session stage-identity namespace."""

    encoded = json.dumps(
        {
            "schema": STAGE_SCHEMA,
            "stage_identities": {
                stage: dict(configuration)
                for stage, configuration in SESSION_STAGE_IDENTITIES.items()
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "REGISTERED_WINDOWS",
    "SCHEMA",
    "HELD_LAUNCH_MODE",
    "LAUNCH_MODES",
    "LEGACY_LAUNCH_MODE",
    "SESSION_LABEL",
    "SESSION_STAGE_IDENTITIES",
    "STAGE_IDENTITIES",
    "STAGE_SCHEMA",
    "StageIdentityNotRegistered",
    "WINDOW_LABELS",
    "WindowLabelNotRegistered",
    "assert_lock_label",
    "assert_launch_context",
    "assert_registered",
    "assert_session_label",
    "assert_stage_identity",
    "is_registered",
    "is_stage_identity",
    "registry_fingerprint",
    "stage_registry_fingerprint",
]
