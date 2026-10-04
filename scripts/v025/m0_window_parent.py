#!/usr/bin/env python3
"""JAX-free parent for the three Amendment-4 manager device windows.

This process performs only filesystem/provenance checks, spends one fresh
receipt for one whole window, verifies the already-held canonical lock, and
launches authorized children.  It never imports JAX or ``gpuwrf``.  All
post-processing helpers in this module are likewise accelerator-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import m0_stale_pair as stale_pair  # noqa: E402
import m0_host_rss_sampler as hrs  # noqa: E402
import m0_vram_sampler as mvs  # noqa: E402
import baseline_census as bc  # noqa: E402
import m0_parent_death_guard as pdg  # noqa: E402
import m0_exact_boundary_contract as exact_contract  # noqa: E402
import gpu_window_registry as registry  # noqa: E402
import run_gpu_arm as gpu_auth  # noqa: E402
import step1_driver as step1  # noqa: E402
import wrf_source_authority as wsa  # noqa: E402


SCHEMA = "wrf_gpu2.v025.m0.manager_window.v1"
# Kept in lockstep with the child's constant by a focused test; a parent that
# still emitted v1/v2 would omit ``session_label``/``launch_mode`` and be refused by exact field-set
# equality rather than silently reinterpreted.
HANDOFF_SCHEMA = "wrf_gpu2.v025.m0.authorized_child_handoff.v3"
STATUS = "CPU_C1_C2_GREEN_GPU_WINDOWS_MISSING"
CHILD = SCRIPT_DIR / "m0_exact_boundary_child.py"
PARENT_DEATH_GUARD = SCRIPT_DIR / "m0_parent_death_guard.py"
FAST_RUN_DIR = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms/fastbind_r1")
RAW_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw/autotune0-three-window-r5")
PAIR_CACHE_ROOT = Path(
    "<DATA_ROOT>/wrf_gpu2/v025/m0/raw/autotune0-qualified-pair-cache-r5"
)
PROOF_ROOT = REPO / "proofs/v025/m0/autotune0_three_window_r5"
QUALIFICATION = PROOF_ROOT / "autotune0_qualification.json"
LEDGER = (
    REPO
    / ".agent/sprints/2026-07-27-v0250-m0-setup/gpu_coordination_spent.json"
)
EVIDENCE_ENV = (
    "GPUWRF_M0_EVIDENCE",
    "GPUWRF_M0_EVIDENCE_PATH",
    "GPUWRF_M0_RUN_ID",
    "GPUWRF_M0_SOURCE_SHA256",
    "GPUWRF_M0_CONFIG_SHA256",
    "GPUWRF_M0_INPUT_MANIFEST_SHA256",
    "GPUWRF_M0_DEVICE_UUID",
)
EXPECTED_DEVICE_UUID = "GPU-cec852c8-a76b-e606-d152-2f2a16e0c6b5"
SESSION_WINDOW_DEADLINES = {
    "W1": 2_160.0,
    "W2": 1_400.0,
    "W3": 600.0,
}
QUALIFICATION_FIELDS = {
    "schema",
    "status",
    "run_id",
    "cache_seed_path",
    "cache_seed_sha256",
    "cold_readiness_seconds",
    "cached_readiness_seconds",
    "warm_integration_seconds",
    "fast_pair_path",
    "fast_pair_sha256",
    "w1_exact_boundary",
    "pair_completeness",
    "selected_cold_stage",
    "selected_cold_attempt_index",
    "integration_clock",
}
WINDOWS = {
    "W1": {
        "label": "m0-autotune0-qualification",
        "run_id": "m0-autotune0-qualify-20260728-r3",
        "deadline_seconds": 1500.0,
        "overhead_seconds": 120.0,
        "cache_path": RAW_ROOT
        / "m0-autotune0-qualify-20260728-r3/unique-empty-cache",
        "result": PROOF_ROOT / "m0-autotune0-qualify-20260728-r3.window.json",
    },
    "W2": {
        "label": "m0-autotune0-profiled-census",
        "run_id": "m0-autotune0-profiled-20260728-r3",
        "deadline_seconds": 1400.0,
        "overhead_seconds": 180.0,
        "cache_path": PAIR_CACHE_ROOT / "m0-autotune0-profiled-20260728-r3",
        "identity_path": PROOF_ROOT
        / "prepared_profiler_pair_r3/profiled_identity.json",
        "result": PROOF_ROOT / "m0-autotune0-profiled-20260728-r3.window.json",
    },
    "W3": {
        "label": "m0-autotune0-clean-matched-arm",
        "run_id": "m0-autotune0-clean-20260728-r3",
        "deadline_seconds": 1500.0,
        "overhead_seconds": 180.0,
        "cache_path": PAIR_CACHE_ROOT / "m0-autotune0-clean-20260728-r3",
        "identity_path": PROOF_ROOT
        / "prepared_profiler_pair_r3/clean_identity.json",
        "result": PROOF_ROOT / "m0-autotune0-clean-20260728-r3.window.json",
    },
}


class WindowRefusal(RuntimeError):
    """A prerequisite or postcondition blocked the manager-owned window."""


class WindowTimeout(WindowRefusal):
    """A child hit its frozen timeout and was killed at the bar.

    This is the threshold-censoring event Amendment 5 defines for a cold
    compile: it is *data* for the two-of-three decision, not a safety failure.
    Every other refusal remains a hard failure that suppresses later stages.
    """


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _atomic_json_no_replace(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise WindowRefusal(f"refusing to replace manager-window artifact: {path}")
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_json(path: Path, *, what: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise WindowRefusal(f"{what} is missing: {path}")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WindowRefusal(f"{what} is malformed: {exc}") from exc
    if not isinstance(value, dict):
        raise WindowRefusal(f"{what} is not a JSON object")
    return value


def _assert_parent_accelerator_free() -> None:
    contaminated = sorted(
        name for name in sys.modules
        if name == "jax"
        or name.startswith("jax.")
        or name == "jaxlib"
        or name.startswith("jaxlib.")
        or name == "gpuwrf"
        or name.startswith("gpuwrf.")
    )
    if contaminated:
        raise WindowRefusal(
            "manager parent was contaminated by accelerator imports before "
            f"authorization: {contaminated[:12]}"
        )


def inspect_window(window_id: str) -> dict[str, Any]:
    """Return the full real command graph without reading any prerequisite."""

    if window_id not in WINDOWS:
        raise WindowRefusal(
            f"unknown window {window_id!r}; expected {sorted(WINDOWS)}"
        )
    window = WINDOWS[window_id]
    run_root = RAW_ROOT / window["run_id"]
    common = {
        "python": sys.executable,
        "child": str(CHILD),
        "window": window_id,
        "run_id": window["run_id"],
        "run_dir": str(FAST_RUN_DIR),
        "cache_path": str(window["cache_path"]),
        "xla_flags": "--xla_gpu_autotune_level=0",
        "parent_death_guard": str(PARENT_DEATH_GUARD),
        "parent_death_signal": "SIGTERM",
        "parent_death_cleanup": "SIGKILL_FULL_CHILD_PROCESS_GROUP",
    }
    if window_id == "W1":
        stages = [
            {
                "name": "cold_empty_cache_readiness",
                "mode": "compile-only",
                "fresh_process": True,
                "timeout_seconds": 600.0,
                **common,
            },
            {
                "name": "cached_readiness_and_warm_integration",
                "mode": "clean",
                "fresh_process": True,
                "timeout_seconds": 360.0,
                **common,
            },
        ]
    elif window_id == "W2":
        stages = [
            {
                "name": "profiled_cached_readiness_and_integration",
                "mode": "profiled",
                "fresh_process": True,
                "timeout_seconds": 1100.0,
                "nsys": True,
                "same_process_allocator": True,
                "external_residency_sampler": True,
                **common,
            },
            {
                "name": "profiled_artifact_capture_integrity",
                "mode": "parent-gate",
                "gpu": False,
                "timeout_seconds": 120.0,
                "condition": "JAX-free parent gate after nsys child exit",
                **common,
            },
        ]
    else:
        stages = [
            {
                "name": "clean_cached_readiness_and_integration",
                "mode": "clean",
                "fresh_process": True,
                "timeout_seconds": 360.0,
                **common,
            },
            {
                "name": "profiled_clean_exact_identity_gate",
                "mode": "parent-gate",
                "gpu": False,
                "timeout_seconds": 60.0,
                "condition": "JAX-free exact identity/output gate",
                **common,
            },
        ]
    return {
        "schema": "wrf_gpu2.v025.m0.manager_window_inspection.v1",
        "status": STATUS,
        "inspection_only": True,
        "files_read": [],
        "files_written": [],
        "receipts_read": [],
        "receipts_consumed": [],
        "device_imports": [],
        "device_queries": [],
        "window": window_id,
        "label": window["label"],
        "run_id": window["run_id"],
        "deadline_seconds": window["deadline_seconds"],
        "overhead_seconds": window["overhead_seconds"],
        "stages": stages,
        "result": str(window["result"]),
    }


def _validate_qualification() -> dict[str, Any]:
    qualification = _load_json(QUALIFICATION, what="W1 qualification")
    if set(qualification) != QUALIFICATION_FIELDS:
        raise WindowRefusal("W1 qualification fields are incomplete/unrecognized")
    if (
        qualification["schema"]
        != "wrf_gpu2.v025.m0.autotune0_qualification.v1"
        or qualification["status"] != "AUTOTUNE0_QUALIFIED"
        or qualification["run_id"] != WINDOWS["W1"]["run_id"]
        or qualification["cache_seed_path"] != str(WINDOWS["W1"]["cache_path"])
        or qualification["integration_clock"]
        != "synchronized-integration-only-excluding-compile-cache-load-and-io"
    ):
        raise WindowRefusal("W1 qualification does not bind the exact boundary/cache")
    selected_index = qualification["selected_cold_attempt_index"]
    if (
        isinstance(selected_index, bool)
        or not isinstance(selected_index, int)
        or not 1 <= selected_index <= 3
        or qualification["selected_cold_stage"]
        != f"cold_empty_cache_readiness_{selected_index}"
    ):
        raise WindowRefusal("W1 qualification selected cold stage is invalid")
    for field, limit in (
        ("cold_readiness_seconds", 600.0),
        ("cached_readiness_seconds", 60.0),
        ("warm_integration_seconds", 300.0),
    ):
        value = qualification[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) <= limit
        ):
            raise WindowRefusal(f"W1 qualification {field} fails <= {limit:g}s")
    identity = mvs.directory_tree_identity(WINDOWS["W1"]["cache_path"])
    if identity["sha256"] != qualification["cache_seed_sha256"]:
        raise WindowRefusal("qualified W1 cache content changed")
    return qualification


def _validate_pair_prerequisite(window_id: str) -> dict[str, Any]:
    window = WINDOWS[window_id]
    qualification = _validate_qualification()
    identity_path = Path(window["identity_path"])
    identity = mvs.load_identity(
        identity_path, expected_run_id=str(window["run_id"])
    )
    mvs.validate_identity_against_workload(
        identity,
        source_root=REPO / "src/gpuwrf",
        run_dir=FAST_RUN_DIR,
        hours=1,
    )
    cache = step1.validate_prepared_cache(
        Path(window["cache_path"]), qualification["cache_seed_sha256"]
    )
    try:
        stale_pair.reject_stale_prepared_pair(
            run_id=str(window["run_id"]),
            identity_path=identity_path,
            prepared_cache_sha256=qualification["cache_seed_sha256"],
        )
    except stale_pair.StalePairError as exc:
        raise WindowRefusal(str(exc)) from exc
    if window_id == "W3":
        profiled = _load_json(
            Path(WINDOWS["W2"]["result"]), what="profiled-first W2 result"
        )
        if profiled.get("status") != "OK":
            raise WindowRefusal("W3 is forbidden until W2 profiled result is OK")
    return {
        "qualification": qualification,
        "identity": identity,
        "identity_path": str(identity_path),
        "cache": cache,
    }


def validate_pre_authorization(window_id: str) -> dict[str, Any]:
    """Check every non-receipt prerequisite before a receipt can be spent."""

    _assert_parent_accelerator_free()
    if window_id not in WINDOWS:
        raise WindowRefusal(f"unknown window {window_id!r}")
    window = WINDOWS[window_id]
    if os.path.lexists(window["result"]):
        raise WindowRefusal(f"unique result already exists: {window['result']}")
    run_root = RAW_ROOT / str(window["run_id"])
    if os.path.lexists(run_root):
        raise WindowRefusal(f"unique raw run root already exists: {run_root}")
    if window_id == "W1":
        if os.path.lexists(window["cache_path"]):
            raise WindowRefusal(
                f"W1 cold cache must not exist before authorization: "
                f"{window['cache_path']}"
            )
        return {
            "unique_raw_root_absent": True,
            "unique_empty_cache_absent": True,
            "pair_prerequisite": None,
        }
    return {
        "unique_raw_root_absent": True,
        "unique_empty_cache_absent": None,
        "pair_prerequisite": _validate_pair_prerequisite(window_id),
    }


def authorise_window(
    window_id: str,
    *,
    receipt_path: Path,
    ledger_path: Path = LEDGER,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate and spend one receipt after the zero-wait lock is held."""

    _assert_parent_accelerator_free()
    window = WINDOWS[window_id]
    label = str(window["label"])
    authorization = gpu_auth.authorise(
        label,
        receipt_path=Path(receipt_path),
        env=env,
        ledger_path=ledger_path,
    )
    return {
        "window": window_id,
        "label": window["label"],
        "receipt_path": str(receipt_path),
        "receipt_fingerprint": authorization["receipt_fingerprint"],
        "receipt_spent_at_utc":
            authorization["spend_record"]["spent_at_utc"],
        "spend_record": authorization["spend_record"],
        "canonical_lock": authorization["lock"],
    }


def cold_dump_flags(dump_dir: Path) -> tuple[str, ...]:
    """The only extra XLA flags any stage may ever add.

    Both are in JAX 0.10's ``xla_flags_to_exclude_from_cache_key`` and are
    zeroed in ``_hash_serialized_compile_options``, so enabling them on the cold
    compile cannot change the persistent-cache key that the cached stage must
    hit.  Nothing else may be appended: an arbitrary flag would change the
    compiled program and silently break the frozen exact-boundary identity.
    """

    return (
        f"--xla_dump_to={Path(dump_dir)}",
        "--xla_dump_hlo_as_text",
    )


def _validate_extra_xla_flags(flags: Sequence[str]) -> tuple[str, ...]:
    seen_dump = False
    seen_text = False
    for raw in flags:
        flag = str(raw)
        if flag == "--xla_dump_hlo_as_text":
            if seen_text:
                raise WindowRefusal("duplicate --xla_dump_hlo_as_text flag")
            seen_text = True
            continue
        if flag.startswith("--xla_dump_to="):
            value = flag.removeprefix("--xla_dump_to=")
            if seen_dump or not value or not Path(value).is_absolute():
                raise WindowRefusal(
                    "--xla_dump_to requires one non-empty absolute path"
                )
            seen_dump = True
            continue
        if flag:
            raise WindowRefusal(
                f"refusing non-cache-neutral extra XLA flag {flag!r}; only the "
                "dump flags proven excluded from the JAX cache key are allowed"
            )
    return tuple(str(flag) for flag in flags)


def _base_child_environment(
    cache_path: Path,
    *,
    extra_xla_flags: Sequence[str] = (),
) -> dict[str, str]:
    authority_path = Path(str(os.environ.get(wsa.AUTHORITY_ENV_VAR, "")))
    if not authority_path.is_file():
        raise WindowRefusal("WRF source authority artifact is not bound")
    try:
        authority = wsa.validate_source_authority(
            json.loads(authority_path.read_text(encoding="utf-8")),
            require_env_match=False,
        )
    except (OSError, json.JSONDecodeError, wsa.SourceAuthorityRefusal) as exc:
        raise WindowRefusal(f"WRF source authority refused: {exc}") from exc
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "JAX_PLATFORMS",
            "CUDA_VISIBLE_DEVICES",
            "XLA_FLAGS",
            *wsa.ROOT_ENV_VARS,
            *EVIDENCE_ENV,
        }
    }
    environment.update(wsa.child_environment_binding(authority))
    environment[wsa.AUTHORITY_ENV_VAR] = str(authority_path)
    flags = " ".join(
        ("--xla_gpu_autotune_level=0", *_validate_extra_xla_flags(extra_xla_flags))
    )
    environment.update(
        {
            "XLA_FLAGS": flags,
            "GPUWRF_JAX_CACHE": "1",
            "GPUWRF_JAX_CACHE_DIR": str(Path(cache_path).resolve()),
            # Prepared W2/W3 directories hard-link immutable W1 seed entries.
            # Disable every known write-on-hit path: no LRU atime sidecars,
            # eviction, lock-file updates, or XLA autotune-cache updates.
            # Cache misses still create private files in the arm directory and
            # are rejected later by the prepared-cache re-hash.
            "GPUWRF_JAX_CACHE_LOCK": "0",
            "JAX_COMPILATION_CACHE_MAX_SIZE": "-1",
            "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "none",
            "GPUWRF_XLA_AUTOTUNE_CACHE": "0",
        }
    )
    return environment


def session_label_of(authorization: dict[str, Any]) -> str:
    """The coordination label this authorization actually spent and holds.

    Both reachable paths already record it: the held session records
    ``m0-core-w1-w2-w3-session``, the legacy per-window path records that
    window's own coordination label.  Neither is ever a stage identity, and the
    registry refuses one if a future edit tries.
    """

    return registry.assert_session_label(
        authorization.get("label"),
        context="authorized child session label",
        exc_type=WindowRefusal,
    )


def launch_mode_of(
    authorization: dict[str, Any],
    *,
    stage_identity: str,
) -> str:
    """Derive held-versus-legacy mode from the authorization being consumed.

    Live held-session authorizations name ``M0_CORE_SESSION``.  Legacy
    per-window authorizations name their exact stage.  A few pre-existing CPU
    process-lifecycle fixtures omit that descriptive field; in that case the
    disjoint label relation uniquely determines the mode and is still validated
    by :func:`authorized_launch_context`.
    """

    scope = authorization.get("window")
    label = authorization.get("label")
    if scope == "M0_CORE_SESSION":
        return registry.HELD_LAUNCH_MODE
    if scope == stage_identity:
        return registry.LEGACY_LAUNCH_MODE
    if scope is None:
        return (
            registry.HELD_LAUNCH_MODE
            if label == registry.SESSION_LABEL
            else registry.LEGACY_LAUNCH_MODE
        )
    raise WindowRefusal(
        f"authorized child scope {scope!r} is neither held-session authority "
        f"nor legacy stage {stage_identity!r}"
    )


def authorized_launch_context(
    authorization: dict[str, Any],
    *,
    stage_identity: str,
    run_id: str,
) -> dict[str, str]:
    """Resolve and enforce the complete parent-side launch authority relation."""

    label = authorization.get("label")
    launch_mode = launch_mode_of(
        authorization,
        stage_identity=stage_identity,
    )
    context = registry.assert_launch_context(
        launch_mode=launch_mode,
        stage_identity=stage_identity,
        run_id=run_id,
        coordination_label=label,
        context="authorized child launch context",
        exc_type=WindowRefusal,
    )
    return dict(context)


def assert_stage_registry_consistency() -> dict[str, Any]:
    """Require the parent window table and the frozen stage registry to agree.

    The child binds ``W1/W2/W3`` to a run ID from the registry, independently
    of anything the parent asserts.  That independence is only useful if drift
    between the two is loud, so this is checked on the reachable launch path
    rather than left to a review.
    """

    if set(WINDOWS) != set(registry.SESSION_STAGE_IDENTITIES):
        raise WindowRefusal(
            "parent window table and stage-identity registry disagree: "
            f"{sorted(WINDOWS)} != {sorted(registry.SESSION_STAGE_IDENTITIES)}"
        )
    for stage_identity, configuration in registry.SESSION_STAGE_IDENTITIES.items():
        window = WINDOWS[stage_identity]
        if str(window["run_id"]) != configuration["run_id"]:
            raise WindowRefusal(
                f"{stage_identity} run ID {window['run_id']!r} does not match "
                f"the frozen registry value {configuration['run_id']!r}"
            )
        if str(window["label"]) != configuration["window_label"]:
            raise WindowRefusal(
                f"{stage_identity} window label {window['label']!r} does not "
                f"match the frozen registry value {configuration['window_label']!r}"
            )
    return {
        "stage_identities": list(registry.STAGE_IDENTITIES),
        "stage_registry_fingerprint": registry.stage_registry_fingerprint(),
        "coordination_registry_fingerprint": registry.registry_fingerprint(),
    }


def _handoff_payload(
    *,
    session_label: str,
    launch_mode: str = registry.HELD_LAUNCH_MODE,
    window_id: str,
    stage: str,
    run_id: str,
    authorization: dict[str, Any],
) -> dict[str, Any]:
    token = str(os.environ.get("GPUWRF_GPU_LOCK_TOKEN", ""))
    if not token:
        raise WindowRefusal("held canonical lock exported no token")
    return {
        "schema": HANDOFF_SCHEMA,
        "launch_mode": launch_mode,
        "session_label": session_label,
        "window": window_id,
        "stage": stage,
        "run_id": run_id,
        "parent_pid": os.getpid(),
        "receipt_fingerprint": authorization["receipt_fingerprint"],
        "receipt_spent_at_utc": authorization["receipt_spent_at_utc"],
        "lock_token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "child_nonce": uuid.uuid4().hex,
    }


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _kill_process_group(
    process: subprocess.Popen[Any],
    *,
    process_group: int | None = None,
) -> None:
    """Sweep the captured child group even after its leader has exited."""

    if process_group is None:
        try:
            process_group = os.getpgid(process.pid)
        except ProcessLookupError:
            return
    for signal_number, wait_seconds in (
        (signal.SIGTERM, 0.25),
        (signal.SIGKILL, 2.0),
    ):
        try:
            os.killpg(process_group, signal_number)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if not _process_group_exists(process_group):
                return
            time.sleep(0.02)


def _require_process_group_empty(process_group: int, *, stage: str) -> None:
    if _process_group_exists(process_group):
        raise WindowRefusal(
            f"{stage} process group {process_group} still has descendants "
            "after kill/reap"
        )


def _launch_authorized_child(
    *,
    window_id: str,
    stage: str,
    run_id: str,
    mode: str,
    cache_path: Path,
    run_root: Path,
    authorization: dict[str, Any],
    timeout_seconds: float,
    profiled: bool = False,
    sampler_identity: Any | None = None,
    command_builder: Callable[[Path, Path], list[str]] | None = None,
    extra_xla_flags: Sequence[str] = (),
    deadline: Any | None = None,
    register_process_group: Callable[[int, str], None] | None = None,
    complete_process_group: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Launch one fresh process and require its immutable result artifact."""

    # The lock/receipt authority and the stage identity are resolved here, once,
    # from two disjoint namespaces.  Before this repair the stage identity was
    # handed to the child as its expected lock label, which is the exact live
    # refusal recorded by ``bb1b1cd5``.
    assert_stage_registry_consistency()
    launch_context = authorized_launch_context(
        authorization,
        stage_identity=window_id,
        run_id=run_id,
    )
    session_label = launch_context["coordination_label"]
    launch_mode = launch_context["launch_mode"]
    stage_root = run_root / stage
    stage_root.mkdir(parents=True, exist_ok=False)
    result_path = stage_root / "exact_boundary.json"
    allocator_path = stage_root / "forecast_allocator.json"
    handoff_path = stage_root / "authorization.json"
    handoff = _handoff_payload(
        session_label=session_label,
        launch_mode=launch_mode,
        window_id=window_id,
        stage=stage,
        run_id=run_id,
        authorization=authorization,
    )
    _atomic_json_no_replace(handoff_path, handoff)
    if command_builder is None:
        inner = [
            sys.executable,
            str(CHILD),
            "--launch-mode",
            launch_mode,
            "--session-label",
            session_label,
            "--window",
            window_id,
            "--stage",
            stage,
            "--run-id",
            run_id,
            "--mode",
            mode,
            "--handoff",
            str(handoff_path),
            "--result",
            str(result_path),
            "--run-dir",
            str(FAST_RUN_DIR),
            "--hours",
            "1",
        ]
        if mode != "compile-only":
            inner.extend(
                ["--wrfout-output-dir", str(stage_root / "wrfout")]
            )
        if profiled:
            inner.extend(["--allocator-sidecar", str(allocator_path)])
    else:
        inner = command_builder(handoff_path, result_path)

    command = (
        step1.nsys_command(inner, out_root=stage_root)
        if profiled
        else inner
    )
    environment = _base_child_environment(
        cache_path, extra_xla_flags=extra_xla_flags
    )
    lock_owner_pid = os.getpid()
    environment[pdg.LOCK_OWNER_PID_ENV] = str(lock_owner_pid)
    guarded_command = [
        sys.executable,
        str(PARENT_DEATH_GUARD),
        "--",
        *command,
    ]
    log_path = stage_root / "child.log"
    sampler = None
    host_sampler = None
    log_stream = None
    process = None
    process_group = None
    group_completed = False
    launched_at_utc = None
    finished_at_utc = None
    try:
        if profiled:
            if sampler_identity is None:
                raise WindowRefusal("profiled stage requires sampler identity")
            sampler = mvs.LockOwnerResidencySampler(
                identity=sampler_identity,
                allocator_sidecar=allocator_path,
                output_sidecar=stage_root / "lock_owner_total_residency.json",
                stderr_path=stage_root / "residency_sampler.log",
            )
            # Establish the selected-device no-context baseline and make both
            # persistent telemetry streams ready before the forecast can create
            # a context. Pending process samples are classified after attach.
            sampler.start()
        log_stream = log_path.open("x", encoding="utf-8")
        launch_ns = time.monotonic_ns()
        launched_at_utc = datetime.now(timezone.utc).isoformat()
        environment["GPUWRF_M0_PARENT_LAUNCH_NS"] = str(launch_ns)
        process = subprocess.Popen(
            guarded_command,
            cwd=REPO,
            env=environment,
            start_new_session=True,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # start_new_session=True makes PID == PGID.  Capturing it directly
        # closes the fast-exit race between Popen and os.getpgid().
        process_group = process.pid
        if register_process_group is not None:
            register_process_group(process_group, stage)

        def finish_group() -> None:
            nonlocal group_completed
            if group_completed or process_group is None:
                return
            _kill_process_group(process, process_group=process_group)
            try:
                process.wait(timeout=5.0)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
            _require_process_group_empty(process_group, stage=stage)
            if complete_process_group is not None:
                complete_process_group(process_group, stage)
            group_completed = True

        host_sampler = hrs.ProcessTreeRssSampler(
            root_pid=process.pid,
            output_path=stage_root / "host_process_tree_rss.json",
            run_id=run_id,
            stage=stage,
            cache_path=cache_path,
        )
        host_sampler.start()
        if profiled:
            assert sampler is not None
            sampler.attach(
                root_pid=process.pid,
                process_group_id=process_group,
            )
        try:
            wait_timeout = (
                deadline.timeout_for(stage, timeout_seconds)
                if deadline is not None
                else timeout_seconds
            )
            returncode = process.wait(timeout=wait_timeout)
            finished_at_utc = datetime.now(timezone.utc).isoformat()
            if deadline is not None:
                deadline.check_after(stage)
        except subprocess.TimeoutExpired as exc:
            child_pid = process.pid
            finish_group()
            if sampler is not None:
                sampler.abort()
            if host_sampler is not None:
                host_sampler.abort()
            timeout = WindowTimeout(
                f"{stage} exceeded its frozen {timeout_seconds:g}s timeout"
            )
            # Threshold censoring needs the identity of the stopped process, so
            # the cold classifier can prove three attempts were three processes.
            timeout.child_pid = child_pid
            timeout.stage_root = stage_root
            timeout.cache_path = Path(cache_path)
            timeout.timeout_seconds = float(timeout_seconds)
            raise timeout from exc
    except BaseException:
        if process is not None:
            if process_group is not None:
                _kill_process_group(process, process_group=process_group)
                try:
                    process.wait(timeout=5.0)
                except (subprocess.TimeoutExpired, ChildProcessError):
                    pass
                _require_process_group_empty(process_group, stage=stage)
                if complete_process_group is not None and not group_completed:
                    complete_process_group(process_group, stage)
                group_completed = True
        if sampler is not None:
            sampler.abort()
        if host_sampler is not None:
            host_sampler.abort()
        raise
    finally:
        if log_stream is not None:
            log_stream.flush()
            os.fsync(log_stream.fileno())
            log_stream.close()

    sampler_payload = None
    host_rss_payload = None
    try:
        if sampler is not None:
            if returncode == 0:
                sampler_payload = sampler.finish()
            else:
                sampler.abort()
        if host_sampler is not None:
            if returncode == 0:
                host_rss_payload = host_sampler.finish()
            else:
                host_sampler.abort()
    finally:
        # A successful leader is not proof that its descendants exited.  The
        # group lookup is normally already gone and therefore free; otherwise
        # this closes the success-with-orphan variant of the nonzero f1 defect.
        if process is not None:
            if process_group is not None and not group_completed:
                _kill_process_group(process, process_group=process_group)
                _require_process_group_empty(process_group, stage=stage)
                if complete_process_group is not None:
                    complete_process_group(process_group, stage)
                group_completed = True
    if returncode != 0:
        # The process leader can exit nonzero while a SIGTERM-trapping
        # grandchild remains alive in its session.  Sweep the captured PGID
        # unconditionally before surfacing the failure.
        raise WindowRefusal(
            f"{stage} child returned {returncode}; log tail: "
            f"{log_path.read_text(errors='replace')[-2000:]}"
        )
    result = _load_json(result_path, what=f"{stage} exact-boundary result")
    if result.get("status") != "OK":
        raise WindowRefusal(f"{stage} did not emit status OK")
    timing = result.get("timing") or {}
    if timing.get("parent_process_launch_monotonic_ns") != launch_ns:
        raise WindowRefusal(f"{stage} did not bind the parent launch endpoint")
    wrfout = result.get("wrfout")
    if mode == "compile-only":
        if wrfout is not None:
            raise WindowRefusal("compile-only stage unexpectedly published wrfout")
    else:
        if not isinstance(wrfout, dict) or wrfout.get("status") != "PASS":
            raise WindowRefusal(f"{stage} lacks exact-result wrfout publication")
        try:
            exact_contract.validate_wrfout_binding(
                result, verify_file=True
            )
        except exact_contract.ContractViolation as exc:
            raise WindowRefusal(
                f"{stage} exact-result wrfout proof is invalid: {exc}"
            ) from exc
        output_path = Path(str(wrfout.get("final_wrfout_path", "")))
        expected_parent = (stage_root / "wrfout").resolve()
        if (
            output_path.parent.resolve() != expected_parent
            or not output_path.is_file()
            or sha256_file(output_path) != wrfout.get("final_wrfout_sha256")
            or wrfout.get("result_exact_value_sha256")
            != result["result"]["exact_value_sha256"]
        ):
            raise WindowRefusal(
                f"{stage} wrfout path/hash/result binding is invalid"
            )
    return {
        "name": stage,
        "status": "OK",
        "fresh_process": True,
        "pid": result.get("pid"),
        "started_at_utc": launched_at_utc,
        "finished_at_utc": finished_at_utc,
        "command": guarded_command,
        "command_sha256": hashlib.sha256(
            "\0".join(guarded_command).encode("utf-8")
        ).hexdigest(),
        "device_child_command": command,
        "device_child_command_sha256": hashlib.sha256(
            "\0".join(command).encode("utf-8")
        ).hexdigest(),
        "parent_death_guard": {
            "path": str(PARENT_DEATH_GUARD),
            "lock_owner_pid": lock_owner_pid,
            "pdeath_signal": "SIGTERM",
            "descendant_cleanup_signal": "SIGKILL",
        },
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "result": result,
        "final_wrfout_path": (
            wrfout.get("final_wrfout_path") if isinstance(wrfout, dict) else None
        ),
        "final_wrfout_sha256": (
            wrfout.get("final_wrfout_sha256") if isinstance(wrfout, dict) else None
        ),
        "wrfout_binding_sha256": (
            wrfout.get("binding_sha256") if isinstance(wrfout, dict) else None
        ),
        "sampler": sampler_payload,
        "host_rss": host_rss_payload,
        "host_rss_path": (
            str(stage_root / "host_process_tree_rss.json")
            if host_rss_payload is not None
            else None
        ),
        "host_rss_sha256": (
            sha256_file(stage_root / "host_process_tree_rss.json")
            if host_rss_payload is not None
            else None
        ),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def _boundary_identity(result: dict[str, Any]) -> dict[str, Any]:
    call = result["result"]["call"]
    production = result["result"]["production_binding"]
    return {
        "argument_identity": call["argument_identity"],
        "lowered_program_sha256": call["lowered_program_sha256"],
        "jit_identity": production["jit_identity"],
        "case_preparation_sequence": production["case_preparation_sequence"],
        "wrapper_preparation_sequence": production[
            "wrapper_preparation_sequence"
        ],
    }


def _require_timing(
    result: dict[str, Any],
    *,
    readiness_max: float,
    integration_max: float | None,
) -> None:
    timing = result["result"]["timing"]
    readiness = timing.get("readiness_seconds")
    if (
        isinstance(readiness, bool)
        or not isinstance(readiness, (int, float))
        or not 0.0 < float(readiness) <= readiness_max
    ):
        raise WindowRefusal(
            f"{result['name']} readiness {readiness!r} fails <= {readiness_max:g}s"
        )
    if integration_max is None:
        if timing.get("integration_seconds") is not None:
            raise WindowRefusal("compile-only stage unexpectedly invoked integration")
    else:
        integration = timing.get("integration_seconds")
        if (
            isinstance(integration, bool)
            or not isinstance(integration, (int, float))
            or not 0.0 < float(integration) <= integration_max
        ):
            raise WindowRefusal(
                f"{result['name']} integration {integration!r} fails "
                f"<= {integration_max:g}s"
            )
        if (
            timing["integration_start_monotonic_ns"]
            < timing["child_executable_ready_monotonic_ns"]
        ):
            raise WindowRefusal("integration began before executable readiness")
        if timing.get("derived_by_phase_subtraction") is not False:
            raise WindowRefusal("timing was derived by forbidden phase subtraction")


def _require_profiled_capture_artifacts(
    profiled: dict[str, Any],
) -> dict[str, Any]:
    """Bind the profiled child, nsys report, allocator, and residency artifacts."""

    result_path = Path(str(profiled.get("result_path", "")))
    stage_root = result_path.parent
    source_rep = stage_root / "step1_autotune_off.nsys-rep"
    allocator_path = Path(
        str(
            profiled["result"]["instrumentation"].get(
                "allocator_sidecar", ""
            )
        )
    )
    residency_path = stage_root / "lock_owner_total_residency.json"
    required = {
        "exact_boundary": result_path,
        "source_rep": source_rep,
        "allocator": allocator_path,
        "lock_owner_total_residency": residency_path,
    }
    for role, path in required.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise WindowRefusal(f"profiled {role} artifact is missing/empty: {path}")
    if sha256_file(result_path) != profiled.get("result_sha256"):
        raise WindowRefusal("profiled exact-boundary artifact changed after child exit")
    residency = _load_json(
        residency_path, what="profiled lock-owner residency sidecar"
    )
    if residency != profiled.get("sampler"):
        raise WindowRefusal(
            "profiled lock-owner residency artifact differs from accepted sampler"
        )
    return {
        role: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for role, path in required.items()
    }


def _run_w1(
    *,
    run_root: Path,
    authorization: dict[str, Any],
) -> list[dict[str, Any]]:
    cache = Path(WINDOWS["W1"]["cache_path"])
    cold = _launch_authorized_child(
        window_id="W1",
        stage="cold_empty_cache_readiness",
        run_id=WINDOWS["W1"]["run_id"],
        mode="compile-only",
        cache_path=cache,
        run_root=run_root,
        authorization=authorization,
        timeout_seconds=600.0,
    )
    _require_timing(cold, readiness_max=600.0, integration_max=None)
    if not cache.is_dir() or not any(cache.iterdir()):
        raise WindowRefusal("cold compile did not populate the unique cache")
    cached = _launch_authorized_child(
        window_id="W1",
        stage="cached_readiness_and_warm_integration",
        run_id=WINDOWS["W1"]["run_id"],
        mode="clean",
        cache_path=cache,
        run_root=run_root,
        authorization=authorization,
        timeout_seconds=360.0,
    )
    _require_timing(cached, readiness_max=60.0, integration_max=300.0)
    if _boundary_identity(cold) != _boundary_identity(cached):
        raise WindowRefusal("cold and cached processes did not resolve one exact call")
    return [cold, cached]


def _run_w2(
    *,
    run_root: Path,
    authorization: dict[str, Any],
    prerequisite: dict[str, Any],
    deadline: Any | None = None,
    register_process_group: Callable[[int, str], None] | None = None,
    complete_process_group: Callable[[int, str], None] | None = None,
) -> list[dict[str, Any]]:
    identity = prerequisite["pair_prerequisite"]["identity"]
    profiled = _launch_authorized_child(
        window_id="W2",
        stage="profiled_cached_readiness_and_integration",
        run_id=WINDOWS["W2"]["run_id"],
        mode="profiled",
        cache_path=Path(WINDOWS["W2"]["cache_path"]),
        run_root=run_root,
        authorization=authorization,
        timeout_seconds=1100.0,
        profiled=True,
        sampler_identity=identity,
        deadline=deadline,
        register_process_group=register_process_group,
        complete_process_group=complete_process_group,
    )
    _require_timing(profiled, readiness_max=60.0, integration_max=300.0)
    instrumentation = profiled["result"]["instrumentation"]
    if (
        instrumentation.get("mode") != "profiled"
        or instrumentation.get("nvtx_range")
        != "GPUWRF_M0_FORECAST_INTEGRATION"
        or profiled["result"].get("allocator") is None
        or profiled.get("sampler") is None
    ):
        raise WindowRefusal("profiled child lacks exact range/allocator/residency")
    artifacts = _require_profiled_capture_artifacts(profiled)
    return [
        profiled,
        {
            "name": "profiled_artifact_capture_integrity",
            "status": "PASS",
            "gpu": False,
            "artifacts": artifacts,
            "postlock_export_and_census_status": "MISSING",
        },
    ]


def _run_w3(
    *,
    run_root: Path,
    authorization: dict[str, Any],
    deadline: Any | None = None,
    register_process_group: Callable[[int, str], None] | None = None,
    complete_process_group: Callable[[int, str], None] | None = None,
) -> list[dict[str, Any]]:
    clean = _launch_authorized_child(
        window_id="W3",
        stage="clean_cached_readiness_and_integration",
        run_id=WINDOWS["W3"]["run_id"],
        mode="clean",
        cache_path=Path(WINDOWS["W3"]["cache_path"]),
        run_root=run_root,
        authorization=authorization,
        timeout_seconds=360.0,
        deadline=deadline,
        register_process_group=register_process_group,
        complete_process_group=complete_process_group,
    )
    _require_timing(clean, readiness_max=60.0, integration_max=300.0)
    if clean["result"]["instrumentation"] != {
        "mode": "clean",
        "nvtx_range": None,
        "allocator_sidecar": None,
        "clean_has_profiler_range": False,
        "clean_has_allocator_read": False,
        "invocation_helper":
            "m0_exact_boundary_child._invoke_and_synchronize",
    }:
        raise WindowRefusal("clean arm contains profiler/resource instrumentation")

    w2 = _load_json(Path(WINDOWS["W2"]["result"]), what="profiled W2 result")
    profiled_stages = [
        stage for stage in w2.get("stages", [])
        if stage.get("name") == "profiled_cached_readiness_and_integration"
    ]
    if len(profiled_stages) != 1:
        raise WindowRefusal("W2 result has no unique profiled boundary stage")
    profiled_result = _load_json(
        Path(profiled_stages[0]["result_path"]),
        what="profiled exact-boundary result",
    )
    if (
        _boundary_identity(clean)
        != {
            "argument_identity": profiled_result["call"]["argument_identity"],
            "lowered_program_sha256":
                profiled_result["call"]["lowered_program_sha256"],
            "jit_identity": profiled_result["production_binding"]["jit_identity"],
            "case_preparation_sequence":
                profiled_result["production_binding"]["case_preparation_sequence"],
            "wrapper_preparation_sequence":
                profiled_result["production_binding"][
                    "wrapper_preparation_sequence"
                ],
        }
    ):
        raise WindowRefusal("profiled and clean arms did not resolve one exact call")
    if (
        clean["result"]["result"]["exact_value_sha256"]
        != profiled_result["result"]["exact_value_sha256"]
    ):
        raise WindowRefusal("profiled and clean output digests differ")
    pair_gate = {
        "name": "profiled_clean_exact_identity_gate",
        "status": "PASS",
        "gpu": False,
        "profiled_result_path": profiled_stages[0]["result_path"],
        "profiled_result_sha256": profiled_stages[0]["result_sha256"],
        "clean_result_path": clean["result_path"],
        "clean_result_sha256": clean["result_sha256"],
        "exact_boundary_identity_sha256": canonical_sha256(
            _boundary_identity(clean)
        ),
        "result_sha256": clean["result"]["result"]["exact_value_sha256"],
    }

    return [
        clean,
        pair_gate,
    ]


def run_manager_window(
    *,
    window_id: str,
    receipt_path: Path,
    result_path: Path | None = None,
    ledger_path: Path = LEDGER,
) -> dict[str, Any]:
    """Execute one real window graph after all preconditions pass."""

    prerequisite = validate_pre_authorization(window_id)
    authorization = authorise_window(
        window_id,
        receipt_path=receipt_path,
        ledger_path=ledger_path,
    )
    window = WINDOWS[window_id]
    run_root = RAW_ROOT / str(window["run_id"])
    started_ns = time.monotonic_ns()
    try:
        run_root.mkdir(parents=True, exist_ok=False)
        if window_id == "W1":
            stages = _run_w1(run_root=run_root, authorization=authorization)
        elif window_id == "W2":
            stages = _run_w2(
                run_root=run_root,
                authorization=authorization,
                prerequisite=prerequisite,
            )
        else:
            stages = _run_w3(run_root=run_root, authorization=authorization)
        status = "OK"
        error = None
    except Exception as exc:
        status = "FAILED"
        error = {"type": type(exc).__name__, "message": str(exc)}
        stages = locals().get("stages", [])
    finished_ns = time.monotonic_ns()
    elapsed = (finished_ns - started_ns) / 1e9
    if elapsed > float(window["deadline_seconds"]):
        status = "FAILED"
        error = {
            "type": "WindowDeadlineExceeded",
            "message": (
                f"{window_id} elapsed {elapsed:.3f}s > "
                f"{window['deadline_seconds']:.0f}s"
            ),
        }
    payload = {
        "schema": SCHEMA,
        "status": status,
        "cpu_preparation_status": STATUS,
        "window": window_id,
        "label": window["label"],
        "run_id": window["run_id"],
        "device_touched": True,
        "authorization": authorization,
        "pre_authorization": prerequisite,
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": finished_ns,
        "elapsed_seconds": elapsed,
        "deadline_seconds": window["deadline_seconds"],
        "stages": stages,
        "error": error,
        "post_analysis_import_policy": "JAX-free",
    }
    target = Path(result_path or window["result"])
    try:
        _atomic_json_no_replace(target, payload)
    except Exception as exc:
        payload["status"] = "FAILED"
        payload["artifact_write_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "target": str(target),
        }
    return payload


# --------------------------------------------------------------------------- #
# Amendment-5 held session: one receipt, one lock, W1 -> C1 -> W2 -> W3         #
# --------------------------------------------------------------------------- #
def _write_window_payload(
    *,
    window_id: str,
    stages: list[dict[str, Any]],
    authorization: dict[str, Any],
    prerequisite: dict[str, Any],
    started_ns: int,
    result_path: Path,
    session: dict[str, Any],
) -> dict[str, Any]:
    """Publish the same manager-window artifact the post-lock path consumes."""

    window = WINDOWS[window_id]
    finished_ns = time.monotonic_ns()
    elapsed = (finished_ns - started_ns) / 1e9
    held_session = bool(session.get("held_session"))
    deadline_seconds = (
        SESSION_WINDOW_DEADLINES[window_id]
        if held_session
        else float(window["deadline_seconds"])
    )
    payload = {
        "schema": SCHEMA,
        "status": "OK",
        "cpu_preparation_status": STATUS,
        "window": window_id,
        "label": window["label"],
        "run_id": window["run_id"],
        "device_touched": True,
        "authorization": authorization,
        "pre_authorization": prerequisite,
        "started_monotonic_ns": started_ns,
        "finished_monotonic_ns": finished_ns,
        "elapsed_seconds": elapsed,
        "deadline_seconds": deadline_seconds,
        "stages": stages,
        "error": None,
        "post_analysis_import_policy": "JAX-free",
        "session": session,
    }
    if elapsed > deadline_seconds:
        raise WindowRefusal(
            f"{window_id} elapsed {elapsed:.3f}s > "
            f"{deadline_seconds:.0f}s"
        )
    _atomic_json_no_replace(Path(result_path), payload)
    return payload


def run_session_cold_attempts(
    *,
    run_root: Path,
    authorization: dict[str, Any],
    session: Any,
    deadline: Any,
    register_process_group: Callable[[int, str], None] | None = None,
    complete_process_group: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """Run the prospective two-of-three cold protocol inside the held session.

    Each attempt is a fresh process with its own unique empty normal cache and
    the unchanged 600 s bar.  A threshold stop is recorded as censored data and
    handed to ``classify_cold_attempts``; it never raises, so a first cold miss
    cannot suppress the discriminator that decides the candidate.
    """

    attempts: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    decision: dict[str, Any] | None = None
    for index in range(1, session.MAX_COLD_PROCESSES + 1):
        deadline.require(f"W1_COLD_{index}", session.COLD_THRESHOLD_SECONDS)
        cache = run_root / f"cold-attempt-{index}" / "unique-empty-cache"
        if os.path.lexists(cache):
            raise WindowRefusal(f"cold attempt {index} cache already exists: {cache}")
        stage = f"cold_empty_cache_readiness_{index}"
        try:
            record = _launch_authorized_child(
                window_id="W1",
                stage=stage,
                run_id=WINDOWS["W1"]["run_id"],
                mode="compile-only",
                cache_path=cache,
                run_root=run_root,
                authorization=authorization,
                timeout_seconds=session.COLD_THRESHOLD_SECONDS,
                deadline=deadline,
                register_process_group=register_process_group,
                complete_process_group=complete_process_group,
            )
        except WindowTimeout as exc:
            attempts.append(
                {
                    "process_id": int(getattr(exc, "child_pid", 0) or 0),
                    "cache_path": str(cache),
                    "readiness_seconds": None,
                    "threshold_stop": True,
                }
            )
            records.append(
                {
                    "name": stage,
                    "status": "FAIL_COLD",
                    "threshold_stop": True,
                    "cache_path": str(cache),
                    "detail": str(exc),
                }
            )
        else:
            _require_timing(record, readiness_max=600.0, integration_max=None)
            if not cache.is_dir() or not any(cache.iterdir()):
                raise WindowRefusal(
                    f"cold attempt {index} did not populate its unique cache"
                )
            readiness = record["result"]["timing"]["readiness_seconds"]
            child_pid = record.get("pid")
            if (
                isinstance(child_pid, bool)
                or not isinstance(child_pid, int)
                or child_pid <= 0
            ):
                raise WindowRefusal(
                    f"cold attempt {index} child did not report a usable pid"
                )
            attempts.append(
                {
                    "process_id": child_pid,
                    "cache_path": str(cache),
                    "readiness_seconds": float(readiness),
                    "threshold_stop": False,
                }
            )
            record = dict(record)
            record["cache_path"] = str(cache)
            record["cold_classification"] = "PASS_COLD"
            records.append(record)
        decision = session.classify_cold_attempts(attempts)
        if decision["decision"] != "CONTINUE":
            break
    if decision is None or decision["decision"] == "CONTINUE":
        raise WindowRefusal("cold protocol ended without a two-of-three decision")
    if decision["decision"] == "FAIL":
        raise WindowRefusal(
            "changed autotune-0 candidate definitively FAILED the frozen "
            "two-of-three 600s cold protocol; W1/C1/W2/W3 are suppressed"
        )
    qualifying = next(
        (
            index
            for index, attempt in enumerate(attempts)
            if attempt["readiness_seconds"] is not None
        ),
        None,
    )
    if qualifying is None:
        raise WindowRefusal("cold decision PASSed without a qualifying attempt")
    return {
        "decision": decision,
        "attempts": attempts,
        "stage_records": records,
        "qualified_cache_path": attempts[qualifying]["cache_path"],
        "qualified_attempt_index": qualifying + 1,
        "selected_cold_stage": records[qualifying]["name"],
    }


def run_session_w1(
    *,
    authorization: dict[str, Any],
    session: Any,
    deadline: Any,
    prerequisite: dict[str, Any],
    result_path: Path | None = None,
    register_process_group: Callable[[int, str], None] | None = None,
    complete_process_group: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """W1 inside the held session: cold decision, then the cached exact arm."""

    _assert_parent_accelerator_free()
    window = WINDOWS["W1"]
    run_root = RAW_ROOT / str(window["run_id"])
    started_ns = time.monotonic_ns()
    run_root.mkdir(parents=True, exist_ok=False)
    cold = run_session_cold_attempts(
        run_root=run_root,
        authorization=authorization,
        session=session,
        deadline=deadline,
        register_process_group=register_process_group,
        complete_process_group=complete_process_group,
    )
    # Every downstream frozen check (qualification manifest, W2/W3 prepared-pair
    # prerequisite, stale-pair refusal) binds the seed cache at exactly one
    # path. The qualifying attempt's cache is therefore promoted to that path by
    # rename, and the move is proven content-preserving: directory_tree_identity
    # hashes paths relative to the root, so an unchanged digest across the move
    # is a mechanical no-mutation proof, not an assertion.
    source_cache = Path(cold["qualified_cache_path"])
    cache = Path(window["cache_path"])
    before_identity = mvs.directory_tree_identity(source_cache)
    if os.path.lexists(cache):
        raise WindowRefusal(f"qualified cache seed path already exists: {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    os.rename(source_cache, cache)
    after_identity = mvs.directory_tree_identity(cache)
    if after_identity["sha256"] != before_identity["sha256"]:
        raise WindowRefusal("qualified cache content changed while being promoted")
    cold["qualified_cache_source_path"] = str(source_cache)
    cold["qualified_cache_path"] = str(cache)
    cold["qualified_cache_sha256"] = after_identity["sha256"]
    deadline.require("W1_CACHED_AND_CORRECTNESS", 360.0)
    cached = _launch_authorized_child(
        window_id="W1",
        stage="cached_readiness_and_warm_integration",
        run_id=str(window["run_id"]),
        mode="clean",
        cache_path=cache,
        run_root=run_root,
        authorization=authorization,
        timeout_seconds=360.0,
        deadline=deadline,
        register_process_group=register_process_group,
        complete_process_group=complete_process_group,
    )
    _require_timing(cached, readiness_max=60.0, integration_max=300.0)
    qualifying_record = cold["stage_records"][cold["qualified_attempt_index"] - 1]
    if _boundary_identity(qualifying_record) != _boundary_identity(cached):
        raise WindowRefusal("cold and cached processes did not resolve one exact call")
    stages = [
        *cold["stage_records"],
        cached,
    ]
    payload = _write_window_payload(
        window_id="W1",
        stages=stages,
        authorization=authorization,
        prerequisite=prerequisite,
        started_ns=started_ns,
        result_path=Path(result_path or window["result"]),
        session={
            "held_session": True,
            "cold_decision": cold["decision"],
            "cold_attempts": cold["attempts"],
            "qualified_cache_path": cold["qualified_cache_path"],
            "qualified_cache_source_path": cold["qualified_cache_source_path"],
            "qualified_cache_sha256": cold["qualified_cache_sha256"],
            "qualified_attempt_index": cold["qualified_attempt_index"],
            "selected_cold_stage": cold["selected_cold_stage"],
            "compiler_dump_flags": [],
            "peak_live_buffers": "MISSING_DEFERRED_TO_M1_M2",
        },
    )
    return {
        "payload": payload,
        "cold": cold,
        "cached": cached,
        "cache_path": str(cache),
    }


def run_session_window(
    *,
    window_id: str,
    authorization: dict[str, Any],
    session: Any,
    deadline: Any,
    result_path: Path | None = None,
    register_process_group: Callable[[int, str], None] | None = None,
    complete_process_group: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    """W2 or W3 inside the held session, using the one already-spent receipt."""

    _assert_parent_accelerator_free()
    if window_id not in {"W2", "W3"}:
        raise WindowRefusal(f"session window must be W2 or W3; got {window_id!r}")
    prerequisite = validate_pre_authorization(window_id)
    window = WINDOWS[window_id]
    deadline.require(
        f"{window_id}_DEVICE_STAGES", SESSION_WINDOW_DEADLINES[window_id]
    )
    run_root = RAW_ROOT / str(window["run_id"])
    started_ns = time.monotonic_ns()
    run_root.mkdir(parents=True, exist_ok=False)
    if window_id == "W2":
        stages = _run_w2(
            run_root=run_root,
            authorization=authorization,
            prerequisite=prerequisite,
            deadline=deadline,
            register_process_group=register_process_group,
            complete_process_group=complete_process_group,
        )
    else:
        stages = _run_w3(
            run_root=run_root,
            authorization=authorization,
            deadline=deadline,
            register_process_group=register_process_group,
            complete_process_group=complete_process_group,
        )
    return {
        "payload": _write_window_payload(
            window_id=window_id,
            stages=stages,
            authorization=authorization,
            prerequisite=prerequisite,
            started_ns=started_ns,
            result_path=Path(result_path or window["result"]),
            session={"held_session": True},
        ),
        "stages": stages,
    }


def _post_w1_qualification(
    *,
    w1_result_path: Path,
    fast_pair_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    w1 = _load_json(w1_result_path, what="W1 manager result")
    pair = _load_json(fast_pair_path, what="fresh post-W1 FAST pair")
    if w1.get("status") != "OK":
        raise WindowRefusal("W1 result is not OK")
    if (
        pair.get("schema") != "wrf_gpu2.v025.m0.fast_case_qualification.v1"
        or pair.get("status") not in {"PASS", "FAST_QUALIFIED"}
    ):
        raise WindowRefusal("fresh post-W1 FAST qualification is not green")
    stages = {stage["name"]: stage for stage in w1.get("stages", [])}
    session = w1.get("session") or {}
    selected_index = session.get("qualified_attempt_index")
    selected_stage = session.get("selected_cold_stage")
    if (
        isinstance(selected_index, bool)
        or not isinstance(selected_index, int)
        or not 1 <= selected_index <= 3
        or selected_stage != f"cold_empty_cache_readiness_{selected_index}"
    ):
        raise WindowRefusal("W1 result lacks one canonical selected cold attempt")
    cold = stages.get(selected_stage)
    cached = stages.get("cached_readiness_and_warm_integration")
    if not cold or not cached:
        raise WindowRefusal("W1 result lacks exact cold/cached stages")
    cold_result = _load_json(Path(cold["result_path"]), what="W1 cold result")
    cached_result = _load_json(Path(cached["result_path"]), what="W1 cached result")
    binding = pair.get("w1_exact_boundary") or {}
    expected_binding = {
        "run_id": WINDOWS["W1"]["run_id"],
        "cold_result_sha256": cold["result_sha256"],
        "cached_result_sha256": cached["result_sha256"],
        "gpu_result_sha256":
            cached_result["result"]["exact_value_sha256"],
        "gpu_wrfout_path":
            cached_result["wrfout"]["final_wrfout_path"],
        "gpu_wrfout_sha256":
            cached_result["wrfout"]["final_wrfout_sha256"],
        "gpu_wrfout_binding_sha256":
            cached_result["wrfout"]["binding_sha256"],
    }
    if binding != expected_binding:
        raise WindowRefusal(
            "fresh FAST qualification is not hash-bound to this W1 exact boundary"
        )
    completeness = pair.get("pair_completeness") or {}
    required_pair_flags = (
        "fresh_cpu_arm_this_invocation",
        "fresh_gpu_arm_this_invocation",
        "comparator_result_present",
        "provenance_present",
        "arms_non_overlapping",
    )
    if (
        completeness.get("completeness_percent") != 100
        or any(completeness.get(name) is not True for name in required_pair_flags)
    ):
        raise WindowRefusal("fresh FAST pair is incomplete")
    _require_timing(cold, readiness_max=600.0, integration_max=None)
    _require_timing(cached, readiness_max=60.0, integration_max=300.0)
    economy = pair.get("economy_gates") or {}
    expected_economy = {
        "cold_compile_seconds": (
            cold_result["timing"]["readiness_seconds"],
            600.0,
        ),
        "cached_load_seconds": (
            cached_result["timing"]["readiness_seconds"],
            60.0,
        ),
        "gpu_arm_seconds": (
            cached_result["timing"]["integration_seconds"],
            300.0,
        ),
    }
    for name, (expected, maximum) in expected_economy.items():
        measured = (economy.get(name) or {}).get("value")
        if (
            isinstance(measured, bool)
            or not isinstance(measured, (int, float))
            or float(measured) != float(expected)
            or not 0.0 < float(measured) <= maximum
        ):
            raise WindowRefusal(
                f"fresh FAST economy gate {name} is not bound to W1"
            )
    for name, maximum in (
        ("cpu_arm_seconds", 300.0),
        ("pair_seconds_excl_lock_wait", 600.0),
    ):
        measured = (economy.get(name) or {}).get("value")
        if (
            isinstance(measured, bool)
            or not isinstance(measured, (int, float))
            or not 0.0 < float(measured) <= maximum
        ):
            raise WindowRefusal(f"fresh FAST economy gate {name} is invalid")
    cache_identity = mvs.directory_tree_identity(WINDOWS["W1"]["cache_path"])
    qualification = {
        "schema": "wrf_gpu2.v025.m0.autotune0_qualification.v1",
        "status": "AUTOTUNE0_QUALIFIED",
        "run_id": WINDOWS["W1"]["run_id"],
        "cache_seed_path": str(WINDOWS["W1"]["cache_path"]),
        "cache_seed_sha256": cache_identity["sha256"],
        "cold_readiness_seconds":
            cold_result["timing"]["readiness_seconds"],
        "cached_readiness_seconds":
            cached_result["timing"]["readiness_seconds"],
        "warm_integration_seconds":
            cached_result["timing"]["integration_seconds"],
        "fast_pair_path": str(fast_pair_path),
        "fast_pair_sha256": sha256_file(fast_pair_path),
        "w1_exact_boundary": expected_binding,
        "pair_completeness": {
            name: completeness[name] for name in required_pair_flags
        }
        | {"completeness_percent": 100},
        "selected_cold_stage": selected_stage,
        "selected_cold_attempt_index": selected_index,
        "integration_clock":
            "synchronized-integration-only-excluding-compile-cache-load-and-io",
    }
    _atomic_json_no_replace(output_path, qualification)
    return qualification


def _unique_stage(window: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [
        stage for stage in window.get("stages", [])
        if stage.get("name") == name
    ]
    if len(matches) != 1:
        raise WindowRefusal(
            f"window has {len(matches)} stages named {name!r}; expected one"
        )
    return matches[0]


def _load_bound_child_result(
    stage: dict[str, Any],
    *,
    what: str,
) -> dict[str, Any]:
    path = Path(str(stage.get("result_path", "")))
    child = _load_json(path, what=what)
    if child.get("status") != "OK":
        raise WindowRefusal(f"{what} is not OK")
    if sha256_file(path) != stage.get("result_sha256"):
        raise WindowRefusal(f"{what} hash differs from the manager-window record")
    return child


def _matched_profiler_post(
    *,
    profiled: dict[str, Any],
    clean: dict[str, Any],
) -> dict[str, Any]:
    """Build and gate the exact profiled/clean pair without importing JAX."""

    profiled_wrapper = {"result": profiled}
    clean_wrapper = {"result": clean}
    profiled_identity = _boundary_identity(profiled_wrapper)
    clean_identity = _boundary_identity(clean_wrapper)
    if profiled_identity != clean_identity:
        raise WindowRefusal("post-lock profiled/clean exact identities differ")
    if (
        profiled["result"]["exact_value_sha256"]
        != clean["result"]["exact_value_sha256"]
    ):
        raise WindowRefusal("post-lock profiled/clean output digests differ")

    def arm_binding(result: dict[str, Any]) -> dict[str, str]:
        timing = result["timing"]
        case = result["case"]
        return {
            "workload_identity_sha256": canonical_sha256(
                _boundary_identity({"result": result})
            ),
            "integration_scope_sha256": canonical_sha256(
                {
                    "definition": timing["integration_definition"],
                    "hours": result["call"]["hours"],
                    "compiled_invocation_count":
                        result["call"]["compiled_invocation_count"],
                }
            ),
            "event_mix_sha256": canonical_sha256(
                {
                    "domain": case["domain"],
                    "hours": case["hours"],
                    "boundary_record_cadence_s":
                        case["boundary_record_cadence_s"],
                    "boundary_window_cadence_s":
                        case["boundary_window_cadence_s"],
                    "result_semantics": result["result"]["semantics"],
                    "result_sha256":
                        result["result"]["exact_value_sha256"],
                }
            ),
        }

    profiled_binding = arm_binding(profiled)
    clean_binding = arm_binding(clean)
    if profiled_binding != clean_binding:
        raise WindowRefusal("post-lock profiled/clean matched bindings differ")
    matched = {
        "schema": "wrf_gpu2.v025.m0.profiler_matched_pair.v1",
        "order": [profiled["run_id"], clean["run_id"]],
        "profiled": {
            "run_id": profiled["run_id"],
            "seconds": profiled["timing"]["integration_seconds"],
            "timing_region": "integration-only",
            "instrumentation": "nsys",
            **profiled_binding,
        },
        "unprofiled": {
            "run_id": clean["run_id"],
            "seconds": clean["timing"]["integration_seconds"],
            "timing_region": "integration-only",
            "instrumentation": "none",
            **clean_binding,
        },
    }
    gate = bc.profiler_gate(matched)
    if gate.get("status") != "OK":
        raise WindowRefusal(
            "post-lock profiler perturbation gate is incomplete: "
            f"{gate.get('reason')}"
        )
    return {
        "matched_pair": matched,
        "profiler_perturbation": gate,
        "exact_identity_sha256": canonical_sha256(profiled_identity),
        "result_sha256": profiled["result"]["exact_value_sha256"],
    }


def post_analysis(
    *,
    phase: str,
    w1_result: Path | None = None,
    fast_pair: Path | None = None,
    output: Path | None = None,
    lock_release_proof: Path | None = None,
    session_proof: Path | None = None,
    pair_post: Path | None = None,
    w3_result: Path | None = None,
    w1_session_result: Path | None = None,
) -> dict[str, Any]:
    """Run CPU/JAX-free post-lock qualification or matched-pair checks."""

    _assert_parent_accelerator_free()
    if lock_release_proof is None:
        raise WindowRefusal("post-analysis requires lock-release provenance")
    import m0_postlock_census as postlock

    analysis_started_ns = time.monotonic_ns()
    result_path = Path(w1_result or WINDOWS[phase]["result"])
    release = postlock.validate_lock_release_proof(
        release_path=lock_release_proof,
        window_result_path=result_path,
        analysis_started_monotonic_ns=analysis_started_ns,
        expected_window=phase,
    )
    session_binding = None
    if release.get("schema") == postlock.SESSION_RELEASE_SCHEMA:
        if session_proof is None:
            raise WindowRefusal(
                "Amendment-6 post-analysis requires the held-session root"
            )
        session_path = Path(session_proof)
        if (
            str(session_path.resolve()) != release.get("session_proof_path")
            or sha256_file(session_path) != release.get("session_proof_sha256")
        ):
            raise WindowRefusal(
                "post-analysis session root differs from the release proof"
            )
        session_binding = {
            "path": str(session_path.resolve()),
            "sha256": sha256_file(session_path),
        }
    if phase == "W1":
        if w1_result is None or fast_pair is None or output is None:
            raise WindowRefusal("W1 post-analysis requires result, pair, and output")
        return _post_w1_qualification(
            w1_result_path=w1_result,
            fast_pair_path=fast_pair,
            output_path=output,
        )
    if phase == "W2":
        if output is None:
            raise WindowRefusal("W2 post-analysis requires an output path")
        return postlock.analyze_w2(
            w2_result_path=result_path,
            release_path=lock_release_proof,
            output_path=output,
            export_root=output.parent / f"{output.stem}.exports",
            pair_post_path=pair_post,
            w3_result_path=w3_result,
            w1_result_path=w1_session_result,
            session_proof_path=session_proof,
        )
    if phase == "W3":
        w3 = _load_json(
            Path(w1_result or WINDOWS["W3"]["result"]), what="W3 manager result"
        )
        if w3.get("status") != "OK":
            raise WindowRefusal("W3 manager result is not OK")
        w2 = _load_json(Path(WINDOWS["W2"]["result"]), what="W2 manager result")
        if w2.get("status") != "OK":
            raise WindowRefusal("W2 manager result is not OK")
        profiled_stage = _unique_stage(
            w2, "profiled_cached_readiness_and_integration"
        )
        clean_stage = _unique_stage(
            w3, "clean_cached_readiness_and_integration"
        )
        inline_gate = _unique_stage(
            w3, "profiled_clean_exact_identity_gate"
        )
        if inline_gate.get("status") != "PASS":
            raise WindowRefusal("W3 inline exact identity gate is not PASS")
        matched = _matched_profiler_post(
            profiled=_load_bound_child_result(
                profiled_stage, what="W2 profiled exact-boundary result"
            ),
            clean=_load_bound_child_result(
                clean_stage, what="W3 clean exact-boundary result"
            ),
        )
        payload = {
            "schema": "wrf_gpu2.v025.m0.pair_post_analysis.v1",
            "status": "PROFILED_CLEAN_IDENTITY_AND_PERTURBATION_CONFIRMED",
            "window_result_sha256": sha256_file(
                Path(w1_result or WINDOWS["W3"]["result"])
            ),
            "window_bindings": {
                "W2": {
                    "run_id": w2["run_id"],
                    "path": str(Path(WINDOWS["W2"]["result"]).resolve()),
                    "sha256": sha256_file(Path(WINDOWS["W2"]["result"])),
                    "child_path": profiled_stage["result_path"],
                    "child_sha256": profiled_stage["result_sha256"],
                },
                "W3": {
                    "run_id": w3["run_id"],
                    "path": str(
                        Path(w1_result or WINDOWS["W3"]["result"]).resolve()
                    ),
                    "sha256": sha256_file(
                        Path(w1_result or WINDOWS["W3"]["result"])
                    ),
                    "child_path": clean_stage["result_path"],
                    "child_sha256": clean_stage["result_sha256"],
                },
            },
            "session_root": session_binding,
            **matched,
            "jax_imported": False,
        }
        payload["pair_post_sha256"] = canonical_sha256(payload)
        if output is not None:
            _atomic_json_no_replace(output, payload)
        return payload
    raise WindowRefusal(f"unknown post-analysis phase {phase!r}")


__all__ = [
    "STATUS",
    "WINDOWS",
    "WindowRefusal",
    "authorise_window",
    "inspect_window",
    "post_analysis",
    "run_manager_window",
    "validate_pre_authorization",
]
