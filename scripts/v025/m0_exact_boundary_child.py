#!/usr/bin/env python3
"""Authorized child for Amendment-4 exact forecast executable measurements.

The module is deliberately accelerator-free at import time.  The CLI consumes a
single-use parent authorization handoff and re-checks the canonical lock before
``run_boundary`` imports JAX or ``gpuwrf``.  CPU proof code may call
``run_boundary(..., expected_platform="cpu", cpu_device_adapter=True)`` only
after importing ``cpu_guard``.

No model equations live here.  The forecast call is assembled from the exact
production case/preparation functions and the exact private production JIT.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import os
import pickle
import re
import stat
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Stdlib-only, accelerator-free: the closed label/stage namespaces must be
# available to refuse a malformed authorization before anything else happens.
import gpu_window_registry as registry  # noqa: E402

SCHEMA = "wrf_gpu2.v025.m0.exact_executable_boundary.v1"
# v3 carries launch mode as well as the session lock/receipt authority separately
# from the stage identity.  The field set is compared for exact equality, so a
# v1/v2 handoff --
# the shape that let a stage name reach ``expected_label`` in the live refusal
# recorded by ``bb1b1cd5`` -- is refused rather than reinterpreted.
HANDOFF_SCHEMA = "wrf_gpu2.v025.m0.authorized_child_handoff.v3"

#: The declared inventory of guards enforced between a parent authorization and
#: the first accelerator import.  ``m0_authority_guard_inventory`` derives the
#: other side from actual tagged refusal sites (including every refusal inside
#: ``run_gpu_arm.check_canonical_lock``); tests require empty set difference in
#: both directions and a real-CLI mutation for every member.
SESSION_AUTHORITY_HARD_GATES: tuple[str, ...] = (
    "session_label_is_not_a_stage_identity",
    "session_label_is_registered_coordination_label",
    "stage_identity_is_not_a_coordination_label",
    "stage_identity_is_registered",
    "stage_identity_bound_to_run_id",
    "launch_mode_is_registered",
    "launch_context_allows_exact_coordination_label",
    "handoff_is_not_a_symlink",
    "handoff_is_openable",
    "handoff_is_a_regular_file",
    "handoff_is_valid_json",
    "handoff_is_a_json_object",
    "handoff_field_set_exact",
    "handoff_schema_exact",
    "launch_mode_matches_handoff",
    "session_label_matches_handoff",
    "stage_identity_matches_handoff",
    "stage_name_matches_handoff",
    "run_id_matches_handoff",
    "parent_pid_is_integer",
    "parent_is_ancestor",
    "parent_launch_endpoint_is_integer",
    "parent_launch_endpoint_is_positive",
    "receipt_fingerprint_nonempty",
    "receipt_spent_timestamp_is_iso8601",
    "canonical_lock_wrapper_is_held",
    "canonical_lock_environment_is_complete",
    "canonical_lock_holder_file_exists",
    "holder_record_is_one_canonical_record",
    "holder_record_token_equals_exported_token",
    "exported_lock_label_equals_session_label",
    "holder_record_label_equals_session_label",
    "lock_token_sha256_matches_handoff",
    "handoff_consumed_exactly_once",
    "handoff_consumption_is_atomic",
)

# Runtime prefixes are paired with semantic guard IDs here, but completeness is
# not trusted to this table: the committed mechanical enumerator derives all
# seven WindowNotAuthorised raise sites from ``check_canonical_lock`` and
# requires exact anchor equality with these entries.
LOCK_REFUSAL_GUARDS: tuple[tuple[str, str], ...] = (
    (
        "GPUWRF_GPU_LOCK_HELD is not set:",
        "canonical_lock_wrapper_is_held",
    ),
    (
        "the lock environment is incomplete",
        "canonical_lock_environment_is_complete",
    ),
    (
        "lock holder file ",
        "canonical_lock_holder_file_exists",
    ),
    (
        "the holder file is not one exact canonical-wrapper record:",
        "holder_record_is_one_canonical_record",
    ),
    (
        "the holder file does not carry this process's lock token:",
        "holder_record_token_equals_exported_token",
    ),
    (
        "the exported canonical-lock label differs from the exact authorized label:",
        "exported_lock_label_equals_session_label",
    ),
    (
        "the holder record does not carry the exact authorized label:",
        "holder_record_label_equals_session_label",
    ),
)
STATUS_OK = "OK"
RANGE_NAME = "GPUWRF_M0_FORECAST_INTEGRATION"
PRODUCTION_TREE = "a6885ceded260df2f5777d7366d75a5d38947cb7"
FAST_RUN_DIR = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/cpu_arms/fastbind_r1")
MODES = frozenset({"compile-only", "clean", "profiled"})

# Filled from canonical ASTs in production tree a6885ced.  These are values, not
# self-derived expectations: replacing a production helper with a same-name
# wrapper therefore fails before lowering.
EXPECTED_FUNCTION_AST_SHA256 = {
    "gpuwrf.runtime.operational_mode._assert_nonzero_initial_mu_total":
        "4dc8690a0f6181a24f0817242501d9a7599db372bca7677b013327a1ea96ecb1",
    "gpuwrf.runtime.operational_mode._operational_scan_state":
        "f0609a83ad8c7ee7421e4155a13632299495c9466ab38109ff2db2326785e1c1",
    "gpuwrf.runtime.operational_mode._dealias_pytree_buffers":
        "9e3bc0fd9be3ad694805578db05fc9bb7409646649cee82c9ff902e2f1257265",
    "gpuwrf.runtime.operational_mode._run_forecast_operational_jit":
        "85712542d47ada3716ea6343d5a46409957f5f70c8608f6a9e8f9f77fa0ae123",
    "gpuwrf.integration.daily_pipeline._build_real_case":
        "aff14597c9760987836e08382bf6f9d4a0f240ba69ae6caaceb4a6efc909a143",
    "gpuwrf.integration.daily_pipeline._capture_boundary_leaves":
        "e02ce49fe0c62fff014f7679dc8b3e709e8b9dc6c6f746acbdb01b5f49266613",
    "gpuwrf.integration.daily_pipeline._boundary_window_cadence_s":
        "23702a1d4d5154f523ed3dd9ec19d43fc11376ffe600b1f959595ea4c75e72db",
    "gpuwrf.integration.daily_pipeline._rewindow_boundary_leaves":
        "c85f42ce6e33237f3df20613ef8cc9a78cd95589ebaa880b65e86806726e83d4",
}

WRAPPER_PREPARATION_SEQUENCE = (
    "gpuwrf.runtime.operational_mode._assert_nonzero_initial_mu_total",
    "gpuwrf.runtime.operational_mode._operational_scan_state",
    "gpuwrf.runtime.operational_mode._dealias_pytree_buffers",
)
CASE_PREPARATION_SEQUENCE = (
    "gpuwrf.integration.daily_pipeline._build_real_case",
    "gpuwrf.integration.daily_pipeline._capture_boundary_leaves",
    "gpuwrf.integration.daily_pipeline._boundary_window_cadence_s",
    "gpuwrf.integration.daily_pipeline._rewindow_boundary_leaves",
)
JIT_FUNCTION = "gpuwrf.runtime.operational_mode._run_forecast_operational_jit"


class BoundaryRefusal(RuntimeError):
    """The exact-boundary contract was not satisfied before device import."""


def _authority_refusal(authority_guard: str, reason: str) -> BoundaryRefusal:
    """Build a refusal tagged for mechanical source-inventory derivation."""

    refusal = BoundaryRefusal(reason)
    refusal.authority_guard = authority_guard
    return refusal


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def callable_ast_sha256(function: Callable[..., Any]) -> str:
    """Hash a callable's canonical AST, independent of comments/line numbers."""

    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _function_record(fqname: str, function: Callable[..., Any]) -> dict[str, str]:
    expected_module, expected_name = fqname.rsplit(".", 1)
    actual = {
        "module": str(getattr(function, "__module__", "")),
        "qualname": str(getattr(function, "__qualname__", "")),
        "ast_sha256": callable_ast_sha256(function),
    }
    expected_hash = EXPECTED_FUNCTION_AST_SHA256[fqname]
    if actual["module"] != expected_module:
        raise BoundaryRefusal(
            f"production function identity changed for {fqname}: "
            f"module={actual['module']!r}"
        )
    if actual["qualname"].split(".")[-1] != expected_name:
        raise BoundaryRefusal(
            f"production function identity changed for {fqname}: "
            f"qualname={actual['qualname']!r}"
        )
    if actual["ast_sha256"] != expected_hash:
        raise BoundaryRefusal(
            f"production function AST changed for {fqname}: "
            f"{actual['ast_sha256']} != {expected_hash}"
        )
    return {"fqname": fqname, **actual}


def _verified_product_functions() -> dict[str, Any]:
    """Import and bind every exact production function used by this boundary."""

    from gpuwrf.integration import daily_pipeline as daily
    from gpuwrf.runtime import operational_mode as operational

    functions = {
        "gpuwrf.runtime.operational_mode._assert_nonzero_initial_mu_total":
            operational._assert_nonzero_initial_mu_total,
        "gpuwrf.runtime.operational_mode._operational_scan_state":
            operational._operational_scan_state,
        "gpuwrf.runtime.operational_mode._dealias_pytree_buffers":
            operational._dealias_pytree_buffers,
        "gpuwrf.runtime.operational_mode._run_forecast_operational_jit":
            operational._run_forecast_operational_jit,
        "gpuwrf.integration.daily_pipeline._build_real_case":
            daily._build_real_case,
        "gpuwrf.integration.daily_pipeline._capture_boundary_leaves":
            daily._capture_boundary_leaves,
        "gpuwrf.integration.daily_pipeline._boundary_window_cadence_s":
            daily._boundary_window_cadence_s,
        "gpuwrf.integration.daily_pipeline._rewindow_boundary_leaves":
            daily._rewindow_boundary_leaves,
    }
    identities = {
        name: _function_record(name, function)
        for name, function in functions.items()
    }
    return {"functions": functions, "identities": identities, "daily": daily}


def _cpu_state_zeros_adapter(jax):
    """Temporarily make production ``State.zeros`` select the visible CPU.

    Production initialization intentionally rejects a non-GPU backend in
    ``contracts.state._gpu_device``.  Amendment-4's CPU semantic gate still
    needs the exact production case builder.  Redirecting only that selector
    preserves every state value and every subsequent production preparation
    function while making the builder executable on CPU.
    """

    from gpuwrf.contracts import state as state_module

    platforms = sorted({device.platform for device in jax.devices()})
    if platforms != ["cpu"]:
        raise BoundaryRefusal(
            "the State.zeros CPU adapter is legal only with exactly one CPU "
            f"platform set; observed {platforms}"
        )
    original = state_module._gpu_device
    state_module._gpu_device = lambda: jax.devices("cpu")[0]
    return state_module, original


def load_fast_call_arguments(
    *,
    run_dir: Path = FAST_RUN_DIR,
    hours: float = 1.0,
    cpu_device_adapter: bool = False,
) -> tuple[tuple[Any, Any, float], dict[str, Any], dict[str, Any]]:
    """Build the first production forecast call's exact raw arguments.

    This follows ``daily_pipeline._run_forecast_sequence`` at the first segment:
    build the real case, capture its full boundary timeline, and rewindow it at
    global segment start zero before the public forecast wrapper is called.
    """

    import jax

    verified = _verified_product_functions()
    functions = verified["functions"]
    daily = verified["daily"]

    adapter_record = {
        "enabled": bool(cpu_device_adapter),
        "scope": (
            "gpuwrf.contracts.state._gpu_device only, during _build_real_case"
            if cpu_device_adapter
            else "none"
        ),
    }
    adapter = None
    if cpu_device_adapter:
        adapter = _cpu_state_zeros_adapter(jax)
    try:
        config = daily.DailyPipelineConfig(
            run_id=str(Path(run_dir).resolve()),
            run_root=Path(run_dir).resolve().parent,
            hours=1,
            domain="d01",
        )
        case, resolved_run_dir = functions[
            "gpuwrf.integration.daily_pipeline._build_real_case"
        ](config)
    finally:
        if adapter is not None:
            state_module, original = adapter
            state_module._gpu_device = original

    state = case.state
    capture = functions[
        "gpuwrf.integration.daily_pipeline._capture_boundary_leaves"
    ](state, case.namelist)
    cadence = functions[
        "gpuwrf.integration.daily_pipeline._boundary_window_cadence_s"
    ](case.namelist)
    record_cadence = float(
        (case.metadata.get("boundary") or {}).get("interval_seconds") or cadence
    )
    if capture:
        state = functions[
            "gpuwrf.integration.daily_pipeline._rewindow_boundary_leaves"
        ](
            state,
            capture,
            segment_start_s=0.0,
            record_cadence_s=record_cadence,
            window_s=cadence,
        )
    metadata = {
        "run_dir": str(Path(resolved_run_dir).resolve()),
        "domain": "d01",
        "hours": float(hours),
        "segment_start_s": 0.0,
        "segment_hours": float(hours),
        "boundary_leaf_names": sorted(capture),
        "boundary_record_cadence_s": record_cadence,
        "boundary_window_cadence_s": float(cadence),
        "case_metadata": case.metadata,
        "cpu_device_adapter": adapter_record,
        "case_preparation_sequence": list(CASE_PREPARATION_SEQUENCE),
    }
    # These objects are intentionally retained only in-process.  They are the
    # exact production case authority consumed by the post-integration writer;
    # they never enter the JSON payload or the lowered call.
    verified["case"] = case
    verified["config"] = config
    return (state, case.namelist, float(hours)), metadata, verified


def prepare_exact_call(
    raw_arguments: tuple[Any, Any, float],
    verified: dict[str, Any],
) -> tuple[tuple[Any, Any, float], list[dict[str, str]]]:
    """Apply the exact public-wrapper preparation in its production order."""

    state, namelist, hours = raw_arguments
    functions = verified["functions"]

    assert_nonzero = functions[
        "gpuwrf.runtime.operational_mode._assert_nonzero_initial_mu_total"
    ]
    scan_state = functions[
        "gpuwrf.runtime.operational_mode._operational_scan_state"
    ]
    dealias = functions[
        "gpuwrf.runtime.operational_mode._dealias_pytree_buffers"
    ]

    assert_nonzero(state)
    prepared_state = scan_state(state, namelist)
    prepared_state = dealias(prepared_state)
    sequence = [
        verified["identities"][name]
        for name in WRAPPER_PREPARATION_SEQUENCE
    ]
    return (prepared_state, namelist, hours), sequence


def _leaf_semantics(jax, value: Any) -> dict[str, Any]:
    leaves, treedef = jax.tree_util.tree_flatten(value)
    records: list[dict[str, Any]] = []
    for index, leaf in enumerate(leaves):
        shape = tuple(int(item) for item in getattr(leaf, "shape", ()))
        dtype = str(getattr(leaf, "dtype", type(leaf).__name__))
        records.append({"index": index, "shape": list(shape), "dtype": dtype})
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "treedef": _treedef_identity(treedef),
        "leaf_count": len(leaves),
        "leaves": records,
    }


def _serialized_treedef(treedef: Any) -> bytes:
    """Serialize static PyTree auxiliaries without address-bearing ``repr``.

    JAX's human-readable ``str(PyTreeDef)`` delegates to the repr of custom
    auxiliary objects.  Production ``OperationalNamelist`` deliberately uses
    identity-hashable ``_StaticHolder`` objects, whose default repr contains a
    process-local address even when the wrapped value is ``None`` and JAX
    considers two definitions equal.  PyTreeDef's protocol-buffer serializer
    rejects custom nodes.  Pickle is the JAX-recommended serializer for that
    case and records the holders' slots/values rather than their debug repr.

    The digest is intentionally environment-bound (Python/JAX versions are
    recorded in the terminal proof) and is compared only between matched arms.
    """

    try:
        return pickle.dumps(treedef, protocol=5)
    except (pickle.PickleError, TypeError, ValueError, AttributeError) as exc:
        raise BoundaryRefusal(
            f"cannot serialize exact argument PyTree definition: {exc}"
        ) from exc


def _treedef_identity(treedef: Any) -> dict[str, Any]:
    serialized = _serialized_treedef(treedef)
    return {
        "serialization": "python-pickle-protocol-5",
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "bytes": len(serialized),
        "node_count": int(getattr(treedef, "num_nodes", 0)),
        "leaf_count": int(getattr(treedef, "num_leaves", 0)),
    }


def _host_pytree_and_digest(jax, value: Any) -> tuple[Any, str]:
    """Materialize a pytree once and hash its exact host bytes.

    ``jax.device_get`` is deliberately issued once for the complete tree.  The
    previous leaf-at-a-time loop created one host-transfer dispatch per leaf and
    made the output adapter repeat those transfers.  Returning the host tree
    lets the production writer consume the same materialized exact result.
    """

    import numpy as np

    host_value = jax.device_get(value)
    leaves, treedef = jax.tree_util.tree_flatten(host_value)
    digest = hashlib.sha256(_serialized_treedef(treedef))
    for index, leaf in enumerate(leaves):
        digest.update(index.to_bytes(8, "little"))
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            host = np.asarray(leaf)
            digest.update(str(host.dtype).encode("utf-8"))
            digest.update(json.dumps(list(host.shape)).encode("utf-8"))
            contiguous = np.ascontiguousarray(host)
            digest.update(memoryview(contiguous).cast("B"))
        else:
            digest.update(
                json.dumps(
                    leaf,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
    return host_value, digest.hexdigest()


def _pytree_digest(jax, value: Any) -> str:
    """Hash pytree structure and exact host bytes outside all timed ranges."""

    _host_value, digest = _host_pytree_and_digest(jax, value)
    return digest


def _structural_argument_identity(jax, arguments: tuple[Any, Any, float]) -> dict[str, Any]:
    return {
        "semantics": _leaf_semantics(jax, arguments),
        "exact_value_sha256": _pytree_digest(jax, arguments),
        "hours": float(arguments[2]),
    }


def _lowered_text(lowered: Any) -> str:
    try:
        return lowered.as_text(dialect="stablehlo")
    except (TypeError, ValueError):
        return lowered.as_text()


def extract_exact_lowered_trip_count(text: str) -> dict[str, Any]:
    """Recover the integration trip count from the exact lowered entry graph.

    The production entry emits one top-level ``stablehlo.while`` for each
    statically sized JAX scan segment.  Each segment carries the corresponding
    ``stablehlo.iota`` through its while and compares the loop counter against
    the same static bound.  We accept only that fully mechanical shape inside
    public ``@main``; missing, extra, or ambiguously bound loops fail closed.
    """

    entry_match = re.search(r"(?m)^\s*func\.func public @main\b", text)
    if entry_match is None:
        raise BoundaryRefusal(
            "exact-lowered-trip-count: public StableHLO @main is absent"
        )
    private_match = re.search(
        r"(?m)^\s*func\.func private @", text[entry_match.end() :]
    )
    entry_end = (
        entry_match.end() + private_match.start()
        if private_match is not None
        else len(text)
    )
    entry = text[entry_match.start() : entry_end]
    iotas = list(
        re.finditer(
            r"(?m)^\s*(?P<ssa>%[A-Za-z0-9_]+)\s*=\s*"
            r"stablehlo\.iota\s+dim\s*=\s*0\s*:\s*"
            r"tensor<(?P<count>[1-9][0-9]*)xi32>\s*$",
            entry,
        )
    )
    whiles = list(re.finditer(r"\bstablehlo\.while\b", entry))
    if not iotas or len(iotas) != len(whiles):
        raise BoundaryRefusal(
            "exact-lowered-trip-count: public @main must contain one top-level "
            "stablehlo.iota for every stablehlo.while; "
            f"observed iotas={len(iotas)} whiles={len(whiles)}"
        )

    segments: list[dict[str, Any]] = []
    for index, (iota, while_match) in enumerate(zip(iotas, whiles, strict=True)):
        next_iota_start = (
            iotas[index + 1].start() if index + 1 < len(iotas) else len(entry)
        )
        if not (iota.end() <= while_match.start() < next_iota_start):
            raise BoundaryRefusal(
                "exact-lowered-trip-count: iota/while ordering is ambiguous "
                f"at segment {index}"
            )
        chain_text = entry[iota.end() : while_match.start()]
        segment_text = entry[while_match.start() : next_iota_start]
        cond_start = segment_text.find("cond {")
        cond_end = segment_text.find("} do {", cond_start + 6)
        if cond_start < 0 or cond_end < 0:
            raise BoundaryRefusal(
                "exact-lowered-trip-count: while condition is not explicit "
                f"at segment {index}"
            )
        header_and_cond = segment_text[:cond_end]
        iota_ssa = iota.group("ssa")
        count = int(iota.group("count"))
        add_matches = list(
            re.finditer(
                r"(?m)^\s*(?P<result>%[A-Za-z0-9_]+)\s*=\s*"
                r"stablehlo\.add\s+"
                r"(?P<left>%[A-Za-z0-9_]+),\s*"
                r"(?P<right>%[A-Za-z0-9_]+)\s*:\s*"
                rf"tensor<{count}xi32>\s*$",
                chain_text,
            )
        )
        index_adds = [
            match
            for match in add_matches
            if iota_ssa in {match.group("left"), match.group("right")}
        ]
        if len(index_adds) != 1:
            raise BoundaryRefusal(
                "exact-lowered-trip-count: segment iota must feed exactly one "
                f"static index-vector add at segment {index}"
            )
        index_add = index_adds[0]
        index_ssa = index_add.group("result")
        base_vector_ssa = (
            index_add.group("right")
            if index_add.group("left") == iota_ssa
            else index_add.group("left")
        )
        broadcast = re.search(
            rf"(?m)^\s*{re.escape(base_vector_ssa)}\s*=\s*"
            r"stablehlo\.broadcast_in_dim\s+"
            r"(?P<constant>%[A-Za-z0-9_]+),.*->\s*"
            rf"tensor<{count}xi32>\s*$",
            chain_text,
        )
        if broadcast is None:
            raise BoundaryRefusal(
                "exact-lowered-trip-count: scan index base is not a static "
                f"broadcast at segment {index}"
            )
        constant = re.search(
            rf"(?m)^\s*{re.escape(broadcast.group('constant'))}\s*=\s*"
            r"stablehlo\.constant\s+dense<(?P<start>[0-9]+)>\s*:\s*"
            r"tensor<i32>\s*$",
            chain_text,
        )
        if constant is None or index_ssa not in segment_text[:cond_start]:
            raise BoundaryRefusal(
                "exact-lowered-trip-count: static scan index vector is not "
                f"carried by its while at segment {index}"
            )
        start_step = int(constant.group("start"))
        bounds = [
            int(value)
            for value, _dtype in re.findall(
                r"stablehlo\.constant\s+dense<([0-9]+)>\s*:\s*"
                r"tensor<i(32|64)>",
                header_and_cond,
            )
        ]
        if count not in bounds or re.search(r"\bLT\b", header_and_cond) is None:
            raise BoundaryRefusal(
                "exact-lowered-trip-count: while condition is not bound to its "
                f"iota length at segment {index}: count={count}, bounds={bounds}"
            )
        segments.append(
            {
                "index": index,
                "iota_ssa": iota_ssa,
                "index_ssa": index_ssa,
                "trip_count": count,
                "condition_bound": count,
                "comparison": "LT",
                "start_step": start_step,
                "end_step": start_step + count - 1,
            }
        )

    expected_start = 1
    for segment in segments:
        if segment["start_step"] != expected_start:
            raise BoundaryRefusal(
                "exact-lowered-trip-count: scan index vectors do not form one "
                "contiguous integration interval beginning at step 1"
            )
        expected_start = segment["end_step"] + 1

    return {
        "status": "PASS",
        "method": "exact-lowered-trip-count",
        "entry_function": "main",
        "loop_count": len(segments),
        "segment_trip_counts": [
            segment["trip_count"] for segment in segments
        ],
        "steps": sum(segment["trip_count"] for segment in segments),
        "segments": segments,
    }


def _stablehlo_identity(lowered: Any) -> dict[str, Any]:
    """Hash UTF-8 StableHLO without materializing a second full-size byte copy."""

    text = _lowered_text(lowered)
    digest = hashlib.sha256()
    encoded_bytes = 0
    chunk_chars = 1 << 20
    for offset in range(0, len(text), chunk_chars):
        encoded = text[offset : offset + chunk_chars].encode("utf-8")
        digest.update(encoded)
        encoded_bytes += len(encoded)
    integration_trip_count = extract_exact_lowered_trip_count(text)
    stablehlo_sha256 = digest.hexdigest()
    integration_trip_count["stablehlo_sha256"] = stablehlo_sha256
    summary = {
        "sha256": stablehlo_sha256,
        "bytes": encoded_bytes,
        "encoding": "utf-8-streamed-1Mi-character-chunks",
        "full_encoded_copy_materialized": False,
        "dtype_token_counts": {
            "f32": text.count("f32"),
            "f64": text.count("f64"),
            "s32": text.count("i32"),
            "pred": text.count("i1"),
        },
        "convert_operation_count": text.count("stablehlo.convert"),
        "integration_trip_count": integration_trip_count,
    }
    del text
    return summary


def _compiled_memory_analysis(executable: Any) -> dict[str, Any]:
    """Expose compiler memory statistics without inventing unavailable fields."""

    method = getattr(executable, "memory_analysis", None)
    if not callable(method):
        return {
            "status": "MISSING",
            "reason": "compiled executable exposes no memory_analysis()",
        }
    try:
        analysis = method()
    except Exception as exc:  # noqa: BLE001 - proof metadata must fail visibly
        return {
            "status": "MISSING",
            "reason": f"memory_analysis() failed: {type(exc).__name__}: {exc}",
        }
    fields = {}
    for name in (
        "alias_size_in_bytes",
        "argument_size_in_bytes",
        "generated_code_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_generated_code_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
        "output_size_in_bytes",
        "temp_size_in_bytes",
    ):
        value = getattr(analysis, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            fields[name] = int(value)
    return {
        "status": "OK" if fields else "MISSING",
        "fields": fields,
        "peak_live_buffers": "MISSING",
        "peak_live_buffers_reason": (
            "JAX memory_analysis exposes aggregate buffer sizes, not a "
            "hash-bound live-buffer schedule"
        ),
    }


def _wrfout_finiteness_and_variables(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import netCDF4
    import numpy as np

    checked = 0
    nonfinite = 0
    per_variable: dict[str, int] = {}
    variables: list[dict[str, Any]] = []
    with netCDF4.Dataset(path, "r") as dataset:
        for name in sorted(dataset.variables):
            variable = dataset.variables[name]
            variables.append(
                {
                    "name": name,
                    "dtype": str(variable.dtype),
                    "dimensions": list(variable.dimensions),
                    "shape": [int(value) for value in variable.shape],
                }
            )
            if variable.dtype.kind != "f":
                continue
            values = np.asarray(variable[:])
            count = int((~np.isfinite(values)).sum())
            checked += 1
            nonfinite += count
            per_variable[name] = count
    return (
        {
            "status": "PASS" if nonfinite == 0 and checked > 0 else "FAIL",
            "float_variables_checked": checked,
            "non_finite_total": nonfinite,
            "per_variable": per_variable,
        },
        variables,
    )


def _publish_exact_wrfout(
    *,
    jax: Any,
    result: Any,
    run_id: str,
    run_dir: Path,
    hours: float,
    output_dir: Path,
    verified: dict[str, Any],
    case_metadata: dict[str, Any],
    integration_end_ns: int,
    exact_boundary_identity: dict[str, Any],
) -> tuple[dict[str, Any], Any, str]:
    """Publish the same synchronized compiled result with production output code."""

    from gpuwrf.integration import daily_pipeline as daily
    from gpuwrf.io import wrfout_writer
    import m0_vram_sampler as mvs

    case = verified["case"]
    config = verified["config"]
    target_dir = Path(output_dir).resolve()
    if os.path.lexists(target_dir):
        raise BoundaryRefusal(
            f"refusing stale/pre-existing wrfout output directory: {target_dir}"
        )
    target_dir.mkdir(parents=True, exist_ok=False)
    valid_time = case.run_start + timedelta(hours=float(hours))
    target = target_dir / daily._wrfout_name(valid_time, config.domain)

    materialization_start_ns = time.monotonic_ns()
    if materialization_start_ns < integration_end_ns:
        raise BoundaryRefusal("wrfout materialization began inside integration")
    surface_diagnostics = daily._surface_diagnostics_for_output(
        result,
        case.namelist,
        case.run_start,
        lead_seconds=float(hours) * 3600.0,
    )
    diagnostics = daily._merge_output_diagnostics(
        case.writer_diagnostics, surface_diagnostics
    )
    full_variable_set = daily._full_wrfout_variables_enabled(config)
    host_result, result_digest = _host_pytree_and_digest(jax, result)
    materialization_end_ns = time.monotonic_ns()

    prepare_start_ns = time.monotonic_ns()
    prepared = wrfout_writer.prepare_wrfout_payload(
        host_result,
        case.grid,
        case.namelist,
        target,
        domain=config.domain,
        domain_authority=case.writer_domain_authority,
        valid_time=valid_time,
        lead_hours=float(hours),
        run_start=case.run_start,
        diagnostics=diagnostics,
        full_variable_set=full_variable_set,
    )
    prepare_end_ns = time.monotonic_ns()
    write_start_ns = time.monotonic_ns()
    written = wrfout_writer.write_prepared_wrfout(
        prepared,
        expected_domain=config.domain,
        expected_domain_authority=case.writer_domain_authority,
    )
    write_end_ns = time.monotonic_ns()
    if written != target or not target.is_file() or target.is_symlink():
        raise BoundaryRefusal("production writer did not publish the exact target")

    inspection_start_ns = time.monotonic_ns()
    inventory = daily.build_wrfout_inventory([target])
    finiteness, variable_inventory = _wrfout_finiteness_and_variables(
        target
    )
    output_sha256 = sha256_file(target)
    inspection_end_ns = time.monotonic_ns()
    if inventory.get("status") != "PASS":
        raise BoundaryRefusal("published wrfout lacks the production minimum inventory")
    if finiteness.get("status") != "PASS":
        raise BoundaryRefusal("published wrfout contains non-finite float values")

    config_path = Path(run_dir) / "namelist.input"
    if not config_path.is_file():
        raise BoundaryRefusal(f"exact FAST namelist is missing: {config_path}")
    config_identity = json.loads(
        json.dumps(asdict(config), sort_keys=True, default=str)
    )
    input_binding = {
        "config_sha256": canonical_sha256(config_identity),
        "namelist_input_sha256": sha256_file(config_path),
        "input_manifest_sha256": mvs.input_manifest_sha256(Path(run_dir)),
        "case_sha256": canonical_sha256(case_metadata),
    }
    adapter_functions = (
        ("gpuwrf.integration.daily_pipeline._surface_diagnostics_for_output",
         daily._surface_diagnostics_for_output),
        ("gpuwrf.integration.daily_pipeline._merge_output_diagnostics",
         daily._merge_output_diagnostics),
        ("gpuwrf.integration.daily_pipeline._wrfout_name", daily._wrfout_name),
        ("gpuwrf.integration.daily_pipeline.build_wrfout_inventory",
         daily.build_wrfout_inventory),
        ("gpuwrf.io.wrfout_writer.prepare_wrfout_payload",
         wrfout_writer.prepare_wrfout_payload),
        ("gpuwrf.io.wrfout_writer.write_prepared_wrfout",
         wrfout_writer.write_prepared_wrfout),
    )
    adapter_identity = [
        {
            "fqname": name,
            "module": function.__module__,
            "qualname": function.__qualname__,
            "ast_sha256": callable_ast_sha256(function),
        }
        for name, function in adapter_functions
    ]
    adapter_source = {
        "src_gpuwrf_tree": PRODUCTION_TREE,
        "child_path": str(Path(__file__).resolve()),
        "child_sha256": sha256_file(Path(__file__).resolve()),
    }
    dimensions = inventory["files"][0]["dimensions"]
    binding_details = {
        "config_identity": config_identity,
        "variable_inventory": variable_inventory,
        "variable_inventory_sha256": canonical_sha256(variable_inventory),
        "dimensions": dimensions,
        "dimensions_sha256": canonical_sha256(dimensions),
        "inventory_sha256": canonical_sha256(inventory),
        "finiteness_sha256": canonical_sha256(finiteness),
        "adapter_identity_sha256": canonical_sha256(adapter_identity),
        "adapter_source_sha256": canonical_sha256(adapter_source),
    }
    output_record = {
        "schema": "wrf_gpu2.v025.m0.exact_result_wrfout.v1",
        "status": "PASS",
        "run_id": run_id,
        "result_exact_value_sha256": result_digest,
        "exact_boundary_identity": exact_boundary_identity,
        "exact_boundary_identity_sha256": canonical_sha256(
            exact_boundary_identity
        ),
        "input_binding": input_binding,
        "adapter_identity": adapter_identity,
        "adapter_source": adapter_source,
        "final_wrfout_path": str(target),
        "final_wrfout_bytes": target.stat().st_size,
        "final_wrfout_sha256": output_sha256,
        "domain": config.domain,
        "domain_authority_sha256":
            case.writer_domain_authority.authority_sha256,
        "run_start_utc": case.run_start.isoformat(),
        "valid_time_utc": valid_time.isoformat(),
        "lead_hours": float(hours),
        "operational_variable_set": True,
        "full_variable_set": bool(full_variable_set),
        "inventory": inventory,
        "finiteness": finiteness,
        **binding_details,
        "timing": {
            "clock": "time.monotonic_ns",
            "integration_end_monotonic_ns": integration_end_ns,
            "materialization_start_monotonic_ns": materialization_start_ns,
            "materialization_end_monotonic_ns": materialization_end_ns,
            "prepare_start_monotonic_ns": prepare_start_ns,
            "prepare_end_monotonic_ns": prepare_end_ns,
            "write_start_monotonic_ns": write_start_ns,
            "write_end_monotonic_ns": write_end_ns,
            "inspection_start_monotonic_ns": inspection_start_ns,
            "inspection_end_monotonic_ns": inspection_end_ns,
            "outside_readiness_and_integration_clocks": True,
        },
        "publication": {
            "atomic": True,
            "replacement": False,
            "same_directory_temporary": True,
        },
    }
    output_record["binding_sha256"] = canonical_sha256(
        {
            "run_id": run_id,
            "result_exact_value_sha256": result_digest,
            "exact_boundary_identity_sha256":
                output_record["exact_boundary_identity_sha256"],
            **input_binding,
            "final_wrfout_path": str(target),
            "final_wrfout_bytes": target.stat().st_size,
            "final_wrfout_sha256": output_sha256,
            "domain": config.domain,
            "domain_authority_sha256":
                case.writer_domain_authority.authority_sha256,
            "run_start_utc": case.run_start.isoformat(),
            "valid_time_utc": valid_time.isoformat(),
            "lead_hours": float(hours),
            "operational_variable_set": True,
            "full_variable_set": bool(full_variable_set),
            **{
                key: binding_details[key]
                for key in (
                    "variable_inventory_sha256",
                    "dimensions_sha256",
                    "inventory_sha256",
                    "finiteness_sha256",
                    "adapter_identity_sha256",
                    "adapter_source_sha256",
                )
            },
        }
    )
    return output_record, host_result, result_digest


def _allocator_stats(jax, result: Any) -> dict[str, Any]:
    """Read allocator peaks from the result's already-active device."""

    device = None
    for leaf in jax.tree_util.tree_leaves(result):
        candidate = getattr(leaf, "device", None)
        if candidate is None:
            continue
        device = candidate() if callable(candidate) else candidate
        break
    if device is None:
        raise BoundaryRefusal("profiled result exposes no device")
    memory_stats = getattr(device, "memory_stats", None)
    if not callable(memory_stats):
        raise BoundaryRefusal("profiled result device exposes no memory_stats()")
    stats = memory_stats()
    if not isinstance(stats, dict):
        raise BoundaryRefusal("profiled result memory_stats() returned no mapping")
    values = {}
    for key in ("peak_bytes_in_use", "peak_bytes_reserved"):
        value = stats.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BoundaryRefusal(f"allocator statistic {key} is missing/invalid")
        values[key] = int(value)
    if values["peak_bytes_reserved"] < values["peak_bytes_in_use"]:
        raise BoundaryRefusal("allocator reserved peak is below in-use peak")
    return {
        **values,
        "device_platform": str(getattr(device, "platform", "")),
        "device_local_ordinal": getattr(device, "id", None),
    }


def _invoke_and_synchronize(
    executable: Any,
    prepared_arguments: tuple[Any, Any, float],
    jax: Any,
) -> tuple[Any, int, int]:
    """Direct integration clock: exact compiled call through final readiness."""

    integration_start_ns = time.monotonic_ns()
    result = _call_compiled_with_exact_arguments(
        executable,
        prepared_arguments,
        lowered_static_hours=float(prepared_arguments[2]),
    )
    jax.block_until_ready(result)
    integration_end_ns = time.monotonic_ns()
    return result, integration_start_ns, integration_end_ns


def _call_compiled_with_exact_arguments(
    executable: Any,
    prepared_arguments: tuple[Any, Any, float],
    *,
    lowered_static_hours: float,
) -> Any:
    """Invoke a JAX ``Compiled`` while preserving the full production call.

    ``_run_forecast_operational_jit`` declares ``hours`` in
    ``static_argnames``. JAX therefore embeds it in the lowered executable and
    removes it from the runtime input pytree: the returned ``Compiled`` accepts
    exactly the dynamic ``(state, namelist)`` pair and rejects a three-tuple.
    This adapter accepts the exact production triple, fail-closes if its static
    value differs from the one lowered, and forwards its two exact dynamic
    members without reconstruction.
    """

    if len(prepared_arguments) != 3:
        raise BoundaryRefusal("exact production call must contain three arguments")
    prepared_state, namelist, hours = prepared_arguments
    if float(hours) != float(lowered_static_hours):
        raise BoundaryRefusal(
            "runtime hours differs from the static value embedded by lower()"
        )
    return executable(prepared_state, namelist)


def _atomic_json_no_replace(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise BoundaryRefusal(f"refusing to replace exact-boundary artifact: {path}")
    fd, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
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


def run_boundary(
    *,
    mode: str,
    run_id: str,
    parent_launch_monotonic_ns: int,
    run_dir: Path = FAST_RUN_DIR,
    hours: float = 1.0,
    expected_platform: str = "gpu",
    cpu_device_adapter: bool = False,
    allocator_sidecar: Path | None = None,
    wrfout_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve the exact executable and optionally invoke it once."""

    if mode not in MODES:
        raise BoundaryRefusal(f"unknown mode {mode!r}; expected {sorted(MODES)}")
    if isinstance(parent_launch_monotonic_ns, bool) or parent_launch_monotonic_ns <= 0:
        raise BoundaryRefusal("parent launch monotonic endpoint must be a positive integer")
    if mode == "profiled" and allocator_sidecar is None:
        raise BoundaryRefusal("profiled mode requires an allocator sidecar")
    if mode != "profiled" and allocator_sidecar is not None:
        raise BoundaryRefusal("clean/compile-only mode forbids allocator sidecars")
    if mode == "compile-only" and wrfout_output_dir is not None:
        raise BoundaryRefusal("compile-only mode forbids wrfout publication")

    import jax

    devices = jax.devices()
    platforms = sorted({device.platform for device in devices})
    if platforms != [expected_platform]:
        raise BoundaryRefusal(
            f"expected exactly platform {expected_platform!r}; observed {platforms}"
        )

    raw_arguments, case_metadata, verified = load_fast_call_arguments(
        run_dir=run_dir,
        hours=hours,
        cpu_device_adapter=cpu_device_adapter,
    )
    prepared_arguments, preparation_identity = prepare_exact_call(
        raw_arguments, verified
    )
    exact_jit = verified["functions"][JIT_FUNCTION]
    lowered = exact_jit.lower(*prepared_arguments)
    executable = lowered.compile()
    executable_ready_monotonic_ns = time.monotonic_ns()
    if executable_ready_monotonic_ns < parent_launch_monotonic_ns:
        raise BoundaryRefusal("executable readiness precedes parent process launch")

    # These identities are proof work, not product executable-readiness work.
    # They must remain after the compile-return readiness endpoint and before a
    # potentially donating compiled invocation.
    argument_identity_start_ns = time.monotonic_ns()
    argument_identity = _structural_argument_identity(jax, prepared_arguments)
    argument_identity_end_ns = time.monotonic_ns()
    lowered_identity_start_ns = time.monotonic_ns()
    stablehlo = _stablehlo_identity(lowered)
    lowered_sha256 = stablehlo["sha256"]
    lowered_program_bytes = stablehlo["bytes"]
    lowered_identity_end_ns = time.monotonic_ns()
    compiled_memory = _compiled_memory_analysis(executable)
    if not (
        executable_ready_monotonic_ns
        <= argument_identity_start_ns
        <= argument_identity_end_ns
        <= lowered_identity_start_ns
        <= lowered_identity_end_ns
    ):
        raise BoundaryRefusal("proof identity endpoints are not after readiness")

    result = None
    integration_start_ns = None
    integration_end_ns = None
    allocator = None
    if mode == "profiled":
        with jax.profiler.TraceAnnotation(RANGE_NAME):
            result, integration_start_ns, integration_end_ns = _invoke_and_synchronize(
                executable, prepared_arguments, jax
            )
        allocator = _allocator_stats(jax, result)
    elif mode == "clean":
        result, integration_start_ns, integration_end_ns = _invoke_and_synchronize(
            executable, prepared_arguments, jax
        )

    if integration_start_ns is not None:
        if integration_start_ns < lowered_identity_end_ns:
            raise BoundaryRefusal(
                "warm integration started before proof identities completed"
            )
        if integration_end_ns is None or integration_end_ns < integration_start_ns:
            raise BoundaryRefusal("invalid synchronized integration endpoints")

    result_semantics = _leaf_semantics(jax, result) if result is not None else None
    result_digest = None
    wrfout = None
    if result is not None:
        exact_boundary_identity = {
            "production_tree": PRODUCTION_TREE,
            "argument_identity": argument_identity,
            "lowered_program_sha256": lowered_sha256,
            "jit_identity": verified["identities"][JIT_FUNCTION],
            "case_preparation_sequence": list(CASE_PREPARATION_SEQUENCE),
            "wrapper_preparation_sequence": list(WRAPPER_PREPARATION_SEQUENCE),
            "case_sha256": canonical_sha256(case_metadata),
            "hours": float(hours),
        }
        if wrfout_output_dir is not None:
            wrfout, _host_result, result_digest = _publish_exact_wrfout(
                jax=jax,
                result=result,
                run_id=run_id,
                run_dir=run_dir,
                hours=hours,
                output_dir=wrfout_output_dir,
                verified=verified,
                case_metadata=case_metadata,
                integration_end_ns=int(integration_end_ns),
                exact_boundary_identity=exact_boundary_identity,
            )
        else:
            _host_result, result_digest = _host_pytree_and_digest(jax, result)
    payload = {
        "schema": SCHEMA,
        "status": STATUS_OK,
        "run_id": run_id,
        "pid": os.getpid(),
        "device_touched": expected_platform == "gpu",
        "backend": {
            "expected_platform": expected_platform,
            "observed_platforms": platforms,
            "device_count": len(devices),
        },
        "production_binding": {
            "src_gpuwrf_tree": PRODUCTION_TREE,
            "jit_function": JIT_FUNCTION,
            "jit_identity": verified["identities"][JIT_FUNCTION],
            "case_preparation_sequence": list(CASE_PREPARATION_SEQUENCE),
            "wrapper_preparation_sequence": list(WRAPPER_PREPARATION_SEQUENCE),
            "wrapper_preparation_identity": preparation_identity,
        },
        "case": case_metadata,
        "call": {
            "argument_identity": argument_identity,
            "lower_api": (
                "_run_forecast_operational_jit.lower(*prepared_arguments).compile()"
            ),
            "compiled_invocation": (
                "_call_compiled_with_exact_arguments("
                "executable, prepared_arguments, lowered_static_hours=hours)"
            ),
            "jax_compiled_dynamic_invocation": (
                "executable(prepared_state, namelist)"
            ),
            "static_argument_binding": {
                "name": "hours",
                "value": float(hours),
                "mechanism":
                    "_run_forecast_operational_jit static_argnames=('hours',)",
                "runtime_revalidated_before_dynamic_call": True,
            },
            "lowered_program_sha256": lowered_sha256,
            "lowered_program_bytes": lowered_program_bytes,
            "stablehlo": stablehlo,
            "compiled_memory_analysis": compiled_memory,
            "hours": float(hours),
            "compiled_invocation_count": 0 if mode == "compile-only" else 1,
        },
        "timing": {
            "clock": "time.monotonic_ns",
            "parent_process_launch_monotonic_ns": parent_launch_monotonic_ns,
            "child_executable_ready_monotonic_ns":
                executable_ready_monotonic_ns,
            "readiness_seconds": (
                executable_ready_monotonic_ns - parent_launch_monotonic_ns
            ) / 1e9,
            "argument_identity_start_monotonic_ns":
                argument_identity_start_ns,
            "argument_identity_end_monotonic_ns":
                argument_identity_end_ns,
            "argument_identity_seconds": (
                argument_identity_end_ns - argument_identity_start_ns
            ) / 1e9,
            "lowered_identity_start_monotonic_ns":
                lowered_identity_start_ns,
            "lowered_identity_end_monotonic_ns":
                lowered_identity_end_ns,
            "lowered_identity_seconds": (
                lowered_identity_end_ns - lowered_identity_start_ns
            ) / 1e9,
            "proof_identity_seconds": (
                lowered_identity_end_ns - argument_identity_start_ns
            ) / 1e9,
            "integration_start_monotonic_ns": integration_start_ns,
            "integration_end_monotonic_ns": integration_end_ns,
            "integration_seconds": (
                (integration_end_ns - integration_start_ns) / 1e9
                if integration_start_ns is not None
                and integration_end_ns is not None
                else None
            ),
            "readiness_definition": (
                "parent process launch -> exact lower().compile() return"
            ),
            "proof_identity_definition": (
                "after executable readiness -> exact argument and lowered "
                "StableHLO identities -> before compiled invocation"
            ),
            "integration_definition": (
                "immediately before compiled invocation -> after "
                "jax.block_until_ready(result)"
                if mode != "compile-only"
                else None
            ),
            "derived_by_phase_subtraction": False,
            "first_lazy_call_used_as_readiness": False,
        },
        "instrumentation": {
            "mode": mode,
            "nvtx_range": RANGE_NAME if mode == "profiled" else None,
            "allocator_sidecar": (
                str(allocator_sidecar) if mode == "profiled" else None
            ),
            "clean_has_profiler_range": False,
            "clean_has_allocator_read": False,
            "invocation_helper": (
                "m0_exact_boundary_child._invoke_and_synchronize"
            ),
        },
        "allocator": allocator,
        "result": {
            "semantics": result_semantics,
            "exact_value_sha256": result_digest,
            "synchronization": (
                "jax.block_until_ready(result)"
                if result is not None
                else None
            ),
        },
        "wrfout": wrfout,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if mode == "profiled":
        sidecar = {
            "schema": "wrf_gpu2.v025.m0.exact_allocator.v1",
            "run_id": run_id,
            "pid": os.getpid(),
            "range_name": RANGE_NAME,
            "measurement_start_ns": integration_start_ns,
            "measurement_end_ns": integration_end_ns,
            "result_sha256": result_digest,
            **(allocator or {}),
        }
        _atomic_json_no_replace(Path(allocator_sidecar), sidecar)
    return payload


def _read_json_regular_no_symlink(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink():
        raise _authority_refusal(
            "handoff_is_not_a_symlink",
            f"authorization handoff must not be a symlink: {path}",
        )
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise _authority_refusal(
                    "handoff_is_a_regular_file",
                    "authorization handoff is not a regular file",
                )
            raw = os.read(descriptor, max(info.st_size + 1, 1 << 20))
        finally:
            os.close(descriptor)
    except BoundaryRefusal:
        raise
    except OSError as exc:
        raise _authority_refusal(
            "handoff_is_openable",
            f"authorization handoff could not be opened/read: {exc}",
        ) from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _authority_refusal(
            "handoff_is_valid_json",
            f"authorization handoff is malformed: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise _authority_refusal(
            "handoff_is_a_json_object",
            "authorization handoff is not a JSON object",
        )
    return value


def _ancestor_pids(pid: int) -> set[int]:
    ancestors: set[int] = set()
    current = int(pid)
    for _ in range(128):
        status_path = Path(f"/proc/{current}/status")
        if not status_path.is_file():
            break
        parent = None
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("PPid:"):
                parent = int(line.split(":", 1)[1].strip())
                break
        if parent is None or parent <= 0 or parent in ancestors:
            break
        ancestors.add(parent)
        current = parent
    return ancestors


def consume_authorization_handoff(
    path: Path,
    *,
    expected_launch_mode: str,
    expected_session_label: str,
    expected_stage_identity: str,
    expected_stage: str,
    expected_run_id: str,
) -> dict[str, Any]:
    """Consume a parent proof before any accelerator import occurs.

    ``expected_session_label`` is the canonical lock / receipt authority the
    manager actually coordinated (for the held session:
    ``m0-core-w1-w2-w3-session``).  ``expected_stage_identity`` is which part of
    the already-authorized graph is running (``W1``/``W2``/``W3``).  They are
    two different namespaces and only the first may ever reach
    ``check_canonical_lock``: the live session recorded by ``bb1b1cd5`` died
    exactly because the stage identity was passed as the expected lock label.

    Both are validated against the frozen registry *before* the lock is
    consulted, so a swapped pair is refused by name rather than producing a
    confusing label-mismatch message.
    """

    launch_context = registry.assert_launch_context(
        launch_mode=expected_launch_mode,
        stage_identity=expected_stage_identity,
        run_id=expected_run_id,
        coordination_label=expected_session_label,
        context="authorization handoff launch context",
        exc_type=BoundaryRefusal,
    )
    launch_mode = launch_context["launch_mode"]
    session_label = launch_context["coordination_label"]
    stage_configuration = registry.SESSION_STAGE_IDENTITIES[
        launch_context["stage_identity"]
    ]

    payload = _read_json_regular_no_symlink(path)
    required = {
        "schema",
        "launch_mode",
        "session_label",
        "window",
        "stage",
        "run_id",
        "parent_pid",
        "receipt_fingerprint",
        "receipt_spent_at_utc",
        "lock_token_sha256",
        "child_nonce",
    }
    if set(payload) != required:
        raise _authority_refusal(
            "handoff_field_set_exact",
            "authorization handoff fields are incomplete or unrecognized",
        )
    if payload["schema"] != HANDOFF_SCHEMA:
        raise _authority_refusal(
            "handoff_schema_exact",
            "authorization handoff schema mismatch",
        )
    if payload["launch_mode"] != launch_mode:
        raise _authority_refusal(
            "launch_mode_matches_handoff",
            f"authorization handoff launch_mode={payload['launch_mode']!r}, "
            f"expected {launch_mode!r}",
        )
    if payload["session_label"] != session_label:
        raise _authority_refusal(
            "session_label_matches_handoff",
            f"authorization handoff session_label={payload['session_label']!r}, "
            f"expected {session_label!r}",
        )
    if payload["window"] != expected_stage_identity:
        raise _authority_refusal(
            "stage_identity_matches_handoff",
            f"authorization handoff window={payload['window']!r}, "
            f"expected {expected_stage_identity!r}",
        )
    if payload["stage"] != expected_stage:
        raise _authority_refusal(
            "stage_name_matches_handoff",
            f"authorization handoff stage={payload['stage']!r}, "
            f"expected {expected_stage!r}",
        )
    if payload["run_id"] != expected_run_id:
        raise _authority_refusal(
            "run_id_matches_handoff",
            f"authorization handoff run_id={payload['run_id']!r}, "
            f"expected {expected_run_id!r}",
        )
    parent_pid = payload["parent_pid"]
    if isinstance(parent_pid, bool) or not isinstance(parent_pid, int):
        raise _authority_refusal(
            "parent_pid_is_integer",
            "authorization parent_pid is invalid",
        )
    if parent_pid not in _ancestor_pids(os.getpid()):
        raise _authority_refusal(
            "parent_is_ancestor",
            "authorization creator is not an ancestor of this device child",
        )
    launch_text = os.environ.get("GPUWRF_M0_PARENT_LAUNCH_NS", "")
    try:
        launch_ns = int(launch_text)
    except ValueError as exc:
        raise _authority_refusal(
            "parent_launch_endpoint_is_integer",
            "parent launch monotonic endpoint is missing/invalid",
        ) from exc
    if launch_ns <= 0:
        raise _authority_refusal(
            "parent_launch_endpoint_is_positive",
            "authorization launch endpoint is invalid",
        )
    if not str(payload["receipt_fingerprint"]).strip():
        raise _authority_refusal(
            "receipt_fingerprint_nonempty",
            "authorization receipt fingerprint is empty",
        )
    try:
        datetime.fromisoformat(str(payload["receipt_spent_at_utc"]))
    except (TypeError, ValueError) as exc:
        raise _authority_refusal(
            "receipt_spent_timestamp_is_iso8601",
            "authorization receipt spent timestamp is not ISO-8601",
        ) from exc

    # ``run_gpu_arm`` is intentionally JAX-free.  This re-check proves the
    # zero-wait canonical lock remains held by this process tree.  The expected
    # label is the *session* authority; passing the stage identity here is the
    # committed live defect and is now unreachable, because
    # ``assert_session_label`` above rejects every stage identity.
    import run_gpu_arm

    try:
        lock = run_gpu_arm.check_canonical_lock(
            os.environ, expected_label=session_label
        )
    except run_gpu_arm.WindowNotAuthorised as exc:
        # Re-raised as a boundary refusal so an unauthorized lock leaves the
        # same machine-checkable pre-device payload as every other refusal.
        # The live session lost its whole window to this class of failure and
        # reported it only as an uncaught traceback in a wrapper log.
        reason = str(exc)
        matches = [
            guard
            for prefix, guard in LOCK_REFUSAL_GUARDS
            if reason.startswith(prefix)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"unclassified canonical-lock refusal (inventory drift): {reason}"
            ) from exc
        raise _authority_refusal(
            matches[0],
            f"canonical lock refused the session authority: {reason}",
        ) from exc
    token = str(os.environ.get("GPUWRF_GPU_LOCK_TOKEN", ""))
    token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if token_sha256 != payload["lock_token_sha256"]:
        raise _authority_refusal(
            "lock_token_sha256_matches_handoff",
            "authorization lock token does not match held lock",
        )

    spent_path = Path(f"{path}.spent")
    if os.path.lexists(spent_path):
        raise _authority_refusal(
            "handoff_consumed_exactly_once",
            "authorization handoff was already consumed",
        )
    try:
        os.link(path, spent_path)
        Path(path).unlink()
    except OSError as exc:
        Path(spent_path).unlink(missing_ok=True)
        raise _authority_refusal(
            "handoff_consumption_is_atomic",
            f"could not atomically consume authorization handoff: {exc}",
        ) from exc
    return {
        **payload,
        "launch_mode": launch_mode,
        "stage_identity": expected_stage_identity,
        "stage_run_id": stage_configuration["run_id"],
        "stage_window_label": stage_configuration["window_label"],
        "parent_launch_monotonic_ns": launch_ns,
        "canonical_lock": lock,
        "spent_path": str(spent_path),
    }


def _refusal_payload(args: argparse.Namespace, reason: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "REFUSED_PRE_DEVICE_IMPORT",
        "launch_mode": args.launch_mode,
        "session_label": args.session_label,
        "window": args.window,
        "stage": args.stage,
        "run_id": args.run_id,
        "handoff_consumed": False,
        "jax_imported": "jax" in sys.modules,
        "gpuwrf_imported": "gpuwrf" in sys.modules,
        "device_touched": False,
        "reason": reason,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # The session lock/receipt authority and the stage identity are separate
    # required arguments.  ``--window`` keeps its name because it names which
    # window of the frozen graph runs (W1/W2/W3); it is never a lock label.
    parser.add_argument(
        "--launch-mode",
        default=registry.HELD_LAUNCH_MODE,
    )
    parser.add_argument("--session-label", required=True)
    parser.add_argument("--window", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", required=True, choices=sorted(MODES))
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, default=FAST_RUN_DIR)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--allocator-sidecar", type=Path)
    parser.add_argument("--wrfout-output-dir", type=Path)
    args = parser.parse_args(argv)

    try:
        authorization = consume_authorization_handoff(
            args.handoff,
            expected_launch_mode=args.launch_mode,
            expected_session_label=args.session_label,
            expected_stage_identity=args.window,
            expected_stage=args.stage,
            expected_run_id=args.run_id,
        )
    except (BoundaryRefusal, FileNotFoundError, OSError, ValueError) as exc:
        print(json.dumps(_refusal_payload(args, str(exc)), indent=2, sort_keys=True))
        return 2

    # Import-time WRF resolution must be fixed before JAX/gpuwrf appears.
    try:
        import wrf_source_authority as wsa

        wsa.assert_child_binding()
    except Exception as exc:  # noqa: BLE001 - closed pre-device refusal
        payload = _refusal_payload(args, f"WRF source authority: {exc}")
        payload["handoff_consumed"] = True
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    try:
        payload = run_boundary(
            mode=args.mode,
            run_id=args.run_id,
            parent_launch_monotonic_ns=authorization[
                "parent_launch_monotonic_ns"
            ],
            run_dir=args.run_dir,
            hours=args.hours,
            expected_platform="gpu",
            cpu_device_adapter=False,
            allocator_sidecar=args.allocator_sidecar,
            wrfout_output_dir=args.wrfout_output_dir,
        )
        payload["authorization"] = {
            key: authorization[key]
            for key in (
                "launch_mode",
                "session_label",
                "window",
                "stage_identity",
                "stage_run_id",
                "stage",
                "receipt_fingerprint",
                "receipt_spent_at_utc",
                "canonical_lock",
                "spent_path",
            )
        }
        _atomic_json_no_replace(args.result, payload)
    except Exception as exc:  # noqa: BLE001 - terminal evidence must record any failure
        failure = {
            "schema": SCHEMA,
            "status": "FAILED",
            "launch_mode": args.launch_mode,
            "window": args.window,
            "stage": args.stage,
            "run_id": args.run_id,
            "handoff_consumed": True,
            "device_touched": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            _atomic_json_no_replace(args.result, failure)
        except Exception:
            pass
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
