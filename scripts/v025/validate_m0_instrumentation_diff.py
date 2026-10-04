#!/usr/bin/env python3
"""Fail-closed semantic validator for the ADR-036 production-source exception.

ADR-038 re-alignment (2026-09-18, M0 hook-realign sprint; re-bound to FINAL
main after the reviewed seam commit 689beb792): the semantic base is the seam
commit — reviewed production containing the adjudicated ADR-038 default-entry
flip (R6 G1-G5 evidence, G4 re-anchored round-off class, g4-* JSONs) — and the
immutable candidate is final main HEAD. The only allowed delta on the base is
the evidence-hook re-bind inside ``_m0_evidence_run`` (to the CURRENT product
default entry, ``run_forecast_operational_segmented`` seg=34). Everything the
validator has always enforced still holds, re-derived against that pair:
default-off remains the direct default-entry call, every HLO-producing body
(including the seam's env-gated branch) is unchanged, and every hook helper
is digest-frozen.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
# Semantic base: the reviewed production seam commit (env-gated
# fused-vertical-implicit, GPUWRF_FUSED_VERTICAL default OFF) — final-main
# production including the ADR-038 flip; the evidence hook must add nothing
# on top of it except the _m0_evidence_run re-bind to the default entry.
BASE_PRODUCTION_COMMIT = "689beb7927f3b6ac218db3369bdb6e240d8752af"
# Immutable candidate: FINAL main HEAD (post seam, post hook re-bind merge
# b699d56bc) — the working src/gpuwrf must stay byte-identical to it.
CANDIDATE_PRODUCTION_COMMIT = "062c808801a85b178c8a2aa20ec4b82552cf2035"
CANDIDATE_SRC_TREE = "d7f55214f309f61c15f7588401f01665b3d300cc"
CANDIDATE_OPERATIONAL_BLOB = "96859a4b3570dac15785a45c65f685678c9c2460"
CANDIDATE_OPERATIONAL_SHA256 = (
    "4db1b18f4252d5d676c94cc73031064488ef7c4ae4e851e2741959430247f9da"
)
ALLOWED_EXISTING_FILES = {
    "src/gpuwrf/runtime/operational_mode.py",
    "src/gpuwrf/integration/daily_pipeline.py",
}
EXPECTED_CHANGED_FILE = "src/gpuwrf/runtime/operational_mode.py"
ALLOWED_NEW_IMPORTS: set[tuple[str, str]] = set()
FORBIDDEN_ADDED_TOKENS = (
    "jax.debug.callback",
    "jax.pure_callback",
    "jax.experimental.host_callback",
    "jax.device_get",
    "jax.device_put",
    "jnp.",
    "numpy.",
    "np.",
    "lax.",
    "force_fp64",
    "DEFAULT_DTYPES",
    "STATE_FIELD_ORDER",
)
EXPECTED_PRIVATE_HELPER_AST_SHA256 = {
    "_m0_evidence_config_from_env": (
        "4fd14353ac79d0b1fcf72214052c66c8459fe7cdd8587d76425a470ac15f6ade"
    ),
    "_m0_evidence_allocator_stats": (
        "c800754a3227ca9ac2639dd4abe841aaaea85c50fe11ecb2fec739204292ccdc"
    ),
    "_m0_evidence_atomic_json": (
        "d771ebf4c6021f787c2987b00f4574a4b6c37a6473f4a7bbaf88bc3a6a1b3b05"
    ),
    "_m0_evidence_run": (
        # ADR-038 re-alignment: range now encloses the CURRENT default entry
        # (run_forecast_operational_segmented, seg=34) with the same segment
        # resolution/positivity guard as the default-off dispatch.
        "d9c785a18da55349ace9c0c75a5c4442833b39655ff58edadd7c118464a38115"
    ),
}
# ADR-038 default-entry compiled bodies: the hook must observe the entry the
# product runs WITHOUT any of these HLO-producing bodies having been touched
# relative to the adjudicated flip base. Numerics live here.
EXPECTED_UNCHANGED_HLO_BODIES = (
    "_run_forecast_operational_jit",  # monolithic escape hatch (unchanged)
    "run_forecast_operational_segmented",  # default entry host loop
    "_advance_chunk",  # default entry compiled segment dispatcher
    "_advance_chunk_fori",  # default lowering
    "_advance_chunk_static_scan",  # alternate lowering
)
EXPECTED_PRIVATE_CONSTANT_VALUES = {
    "_M0_EVIDENCE_FLAG": "GPUWRF_M0_EVIDENCE",
    "_M0_EVIDENCE_PATH": "GPUWRF_M0_EVIDENCE_PATH",
    "_M0_EVIDENCE_RUN_ID": "GPUWRF_M0_RUN_ID",
    "_M0_EVIDENCE_SOURCE_SHA256": "GPUWRF_M0_SOURCE_SHA256",
    "_M0_EVIDENCE_CONFIG_SHA256": "GPUWRF_M0_CONFIG_SHA256",
    "_M0_EVIDENCE_INPUT_SHA256": "GPUWRF_M0_INPUT_MANIFEST_SHA256",
    "_M0_EVIDENCE_DEVICE_UUID": "GPUWRF_M0_DEVICE_UUID",
    "_M0_EVIDENCE_RANGE": "GPUWRF_M0_FORECAST_INTEGRATION",
    "_M0_EVIDENCE_SCHEMA": "wrf_gpu2.v025.m0.forecast_allocator.v1",
    "_M0_EVIDENCE_RUN_ID_PATTERN": r"^[A-Za-z0-9][A-Za-z0-9._-]{7,127}$",
    "_M0_EVIDENCE_SHA256_PATTERN": r"^[0-9a-f]{64}$",
    "_M0_EVIDENCE_DEVICE_UUID_PATTERN": (
        r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
        r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
    ),
}


class DiffValidationError(RuntimeError):
    """The source exception exceeded ADR-036."""


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise DiffValidationError(
            f"command failed rc={completed.returncode}: {' '.join(command)}\n"
            f"{completed.stderr[-1000:]}"
        )
    return completed.stdout


def _git_file(revision: str, path: str) -> str:
    return _run(["git", "show", f"{revision}:{path}"])


def _ast_digest(node: ast.AST) -> str:
    material = ast.dump(node, include_attributes=False)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _definitions(tree: ast.Module) -> dict[tuple[str, str], ast.AST]:
    result: dict[tuple[str, str], ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[("function", node.name)] = node
        elif isinstance(node, ast.ClassDef):
            result[("class", node.name)] = node
    return result


def _imports(tree: ast.Module) -> set[tuple[str, str]]:
    imported: set[tuple[str, str]] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(("", alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update((node.module or "", alias.name) for alias in node.names)
    return imported


def _assigned_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    targets: list[ast.AST] = []
    if isinstance(node, ast.Assign):
        targets.extend(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets.append(node.target)
    for target in targets:
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            names.update(
                child.id for child in target.elts if isinstance(child, ast.Name)
            )
    return names


def _top_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for node in tree.body:
        for name in _assigned_names(node):
            result[name] = node
    return result


def _top_level_statement_model(tree: ast.Module) -> list[dict[str, Any]]:
    """Name and hash every executable module statement, including side effects."""

    model: list[dict[str, Any]] = []
    for index, node in enumerate(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identity = node.name
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = sorted(_assigned_names(node))
            identity = ",".join(names) if names else "<non-name-target>"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            identity = ast.dump(node, include_attributes=False)
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            identity = "<module-docstring>"
        else:
            # This is intentionally not a permissive "other" bucket. The exact
            # kind, position and AST hash are compared with the reviewed
            # candidate below, so an import-time call, subscript assignment,
            # try/del/augassign or any future statement kind is modeled and
            # rejected unless it already exists in the immutable candidate.
            identity = f"<{type(node).__name__}-at-{index}>"
        model.append(
            {
                "index": index,
                "kind": type(node).__name__,
                "identity": identity,
                "ast_sha256": _ast_digest(node),
            }
        )
    return model


def _literal_assignment_value(node: ast.AST, name: str) -> Any:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        raise DiffValidationError(f"{name} is not a module assignment")
    try:
        return ast.literal_eval(node.value)
    except (TypeError, ValueError) as exc:
        raise DiffValidationError(f"{name} is not assigned a literal value") from exc


def _added_lines(base_source: str, current_source: str) -> list[str]:
    import difflib

    return [
        line[1:]
        for line in difflib.unified_diff(
            base_source.splitlines(),
            current_source.splitlines(),
            lineterm="",
        )
        if line.startswith("+") and not line.startswith("+++")
    ]


def validate_sources(
    base_source: str,
    current_source: str,
    *,
    candidate_source: str | None = None,
) -> dict[str, Any]:
    """Validate structure and return mechanically observable identity hashes."""

    candidate_source = candidate_source or _git_file(
        CANDIDATE_PRODUCTION_COMMIT, EXPECTED_CHANGED_FILE
    )
    try:
        base_tree = ast.parse(base_source)
        current_tree = ast.parse(current_source)
        candidate_tree = ast.parse(candidate_source)
    except SyntaxError as exc:
        raise DiffValidationError(f"operational source is not parseable: {exc}") from exc
    candidate_source_sha256 = hashlib.sha256(
        candidate_source.encode("utf-8")
    ).hexdigest()
    if candidate_source_sha256 != CANDIDATE_OPERATIONAL_SHA256:
        raise DiffValidationError(
            "immutable candidate source does not match its pinned SHA-256"
        )

    base_definitions = _definitions(base_tree)
    current_definitions = _definitions(current_tree)
    # ADR-038 re-alignment: the base IS the adjudicated flip, so the wrapper
    # run_forecast_operational is frozen production; the ONLY function the
    # evidence envelope may change is the enabled helper _m0_evidence_run.
    allowed_changed = ("function", "_m0_evidence_run")
    for key, base_node in base_definitions.items():
        if key == allowed_changed:
            continue
        current_node = current_definitions.get(key)
        if current_node is None or _ast_digest(current_node) != _ast_digest(base_node):
            raise DiffValidationError(
                f"pre-existing {key[0]} {key[1]} changed outside the evidence envelope"
            )
    new_definitions = set(current_definitions) - set(base_definitions)
    expected_new_definitions: set[tuple[str, str]] = set()
    if new_definitions != expected_new_definitions:
        raise DiffValidationError(
            "new production definitions differ from the ADR-038 re-alignment "
            "(no new definitions allowed on top of the flip base): "
            f"expected={sorted(expected_new_definitions)} actual={sorted(new_definitions)}"
        )
    base_assignments = _top_assignments(base_tree)
    current_assignments = _top_assignments(current_tree)
    for name, base_node in base_assignments.items():
        current_node = current_assignments.get(name)
        if current_node is None or _ast_digest(current_node) != _ast_digest(base_node):
            raise DiffValidationError(
                f"pre-existing module assignment {name} changed"
            )
    new_assignments = set(current_assignments) - set(base_assignments)
    if new_assignments:
        raise DiffValidationError(
            "new module assignments are not allowed by the ADR-038 re-alignment: "
            f"got {sorted(new_assignments)}"
        )
    for name, expected_value in EXPECTED_PRIVATE_CONSTANT_VALUES.items():
        actual_value = _literal_assignment_value(current_assignments[name], name)
        if actual_value != expected_value:
            raise DiffValidationError(
                f"reviewed ADR constant value changed: {name}"
            )

    added_imports = _imports(current_tree) - _imports(base_tree)
    if not added_imports <= ALLOWED_NEW_IMPORTS:
        raise DiffValidationError(
            f"unapproved production imports: {sorted(added_imports - ALLOWED_NEW_IMPORTS)}"
        )
    if not _imports(base_tree) <= _imports(current_tree):
        raise DiffValidationError("a pre-existing production import was removed")

    # ADR-038 wrapper shape. The generic pre-existing-definition check above
    # already freezes the whole wrapper to the adjudicated flip base; these
    # structural pins document the reviewed dispatch and give precise
    # diagnostics when a future change trips the freeze.
    wrapper_key = ("function", "run_forecast_operational")
    base_wrapper = base_definitions.get(wrapper_key)
    current_wrapper = current_definitions.get(wrapper_key)
    if not isinstance(base_wrapper, ast.FunctionDef) or not isinstance(
        current_wrapper, ast.FunctionDef
    ):
        raise DiffValidationError("operational forecast wrapper is missing")
    if _ast_digest(base_wrapper.args) != _ast_digest(current_wrapper.args):
        raise DiffValidationError("operational forecast public signature changed")
    if ast.get_docstring(base_wrapper) != ast.get_docstring(current_wrapper):
        raise DiffValidationError("operational forecast documentation changed")

    base_statements = [
        node
        for node in base_wrapper.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    current_statements = [
        node
        for node in current_wrapper.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    if len(base_statements) != 8 or len(current_statements) != 8:
        raise DiffValidationError(
            "wrapper is not the adjudicated ADR-038 eight-statement dispatch "
            "(assert, stage, entry select, entry guard, evidence selector, "
            "default-off branch, evidence guard, enabled return)"
        )
    for index in (0, 1):
        if _ast_digest(base_statements[index]) != _ast_digest(current_statements[index]):
            raise DiffValidationError(
                "pre-integration validation/staging changed in the operational wrapper"
            )
    # ADR-038 entry dispatch (default = segmented): pinned exactly as flipped.
    expected_entry_select = ast.parse(
        'entry = os.environ.get("GPUWRF_FORECAST_ENTRY", "segmented").strip().lower()'
    ).body[0]
    expected_entry_guard = ast.parse(
        "if entry not in ('segmented', 'monolithic'):\n"
        "    raise RuntimeError(\n"
        "        \"GPUWRF_FORECAST_ENTRY must be 'segmented' (default, ADR-038) or \"\n"
        "        f\"'monolithic', got {entry!r}\"\n"
        "    )\n"
    ).body[0]
    if _ast_digest(current_statements[2]) != _ast_digest(expected_entry_select):
        raise DiffValidationError("ADR-038 entry selection changed")
    if _ast_digest(current_statements[3]) != _ast_digest(expected_entry_guard):
        raise DiffValidationError("ADR-038 entry validation guard changed")
    expected_flag = ast.parse(
        "evidence_flag = os.environ.get(_M0_EVIDENCE_FLAG)"
    ).body[0]
    if _ast_digest(current_statements[4]) != _ast_digest(expected_flag):
        raise DiffValidationError("default-off selector changed")
    default_branch = current_statements[5]
    expected_default_test = ast.parse("evidence_flag is None").body[0].value
    if (
        not isinstance(default_branch, ast.If)
        or _ast_digest(default_branch.test) != _ast_digest(expected_default_test)
        or len(default_branch.body) != 2
        or default_branch.orelse
    ):
        raise DiffValidationError(
            "default-off branch condition/body/orelse is not the reviewed "
            "ADR-038 segmented dispatch"
        )
    segmented_branch = default_branch.body[0]
    expected_segmented_test = ast.parse('entry == "segmented"').body[0].value
    expected_seg_steps = ast.parse(
        'seg = int(os.environ.get("GPUWRF_FORECAST_SEGMENT_STEPS", "34"))'
    ).body[0]
    expected_seg_guard = ast.parse(
        "if seg <= 0:\n"
        "    raise RuntimeError(\n"
        '        "GPUWRF_FORECAST_SEGMENT_STEPS must be positive"\n'
        "    )\n"
    ).body[0]
    expected_segmented_return = ast.parse(
        "return run_forecast_operational_segmented("
        "_dealias_pytree_buffers(state), namelist, hours, segment_steps=seg)"
    ).body[0]
    if (
        not isinstance(segmented_branch, ast.If)
        or _ast_digest(segmented_branch.test) != _ast_digest(expected_segmented_test)
        or len(segmented_branch.body) != 3
        or segmented_branch.orelse
        or _ast_digest(segmented_branch.body[0]) != _ast_digest(expected_seg_steps)
        or _ast_digest(segmented_branch.body[1]) != _ast_digest(expected_seg_guard)
        or _ast_digest(segmented_branch.body[2]) != _ast_digest(expected_segmented_return)
    ):
        raise DiffValidationError(
            "default-off segmented dispatch (seg=34 resolution, positivity "
            "guard, direct default-entry call) changed"
        )
    expected_monolithic_return = ast.parse(
        "return _run_forecast_operational_jit("
        "_dealias_pytree_buffers(state), namelist, hours)"
    ).body[0]
    if _ast_digest(default_branch.body[1]) != _ast_digest(expected_monolithic_return):
        raise DiffValidationError(
            "monolithic escape-hatch return changed (GPUWRF_FORECAST_ENTRY=monolithic)"
        )
    expected_guard = ast.parse(
        "if evidence_flag != '1':\n"
        "    raise RuntimeError(\n"
        "        f\"{_M0_EVIDENCE_FLAG} must be unset (default-off) or exactly '1'\"\n"
        "    )\n"
    ).body[0]
    if _ast_digest(current_statements[6]) != _ast_digest(expected_guard):
        raise DiffValidationError(
            "non-'1' evidence guard shape/body/orelse changed"
        )

    added_text = "\n".join(_added_lines(base_source, current_source))
    forbidden = [token for token in FORBIDDEN_ADDED_TOKENS if token in added_text]
    if forbidden:
        raise DiffValidationError(
            f"forbidden numerical/hot-loop tokens were added: {forbidden}"
        )
    # ADR-038: the enabled helper must invoke the CURRENT default entry. (The
    # old TraceAnnotation/block_until_ready text heuristics pinned lines that
    # pre-exist in the flip base; the exact range AST is digest-pinned below.)
    if "run_forecast_operational_segmented(" not in added_text:
        raise DiffValidationError(
            "enabled helper does not invoke the ADR-038 default entry"
        )

    enabled_key = ("function", "_m0_evidence_run")
    enabled = current_definitions.get(enabled_key)
    if not isinstance(enabled, ast.FunctionDef):
        raise DiffValidationError("enabled evidence helper is missing")
    if [arg.arg for arg in enabled.args.args] != ["state", "namelist", "hours"]:
        raise DiffValidationError("enabled evidence helper signature changed")
    stored = [
        node.id
        for node in ast.walk(enabled)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    ]
    if any(name in stored for name in ("state", "namelist", "hours")):
        raise DiffValidationError("enabled helper reassigns a numerical input")
    if stored.count("prepared_state") != 1 or stored.count("result") != 1:
        raise DiffValidationError(
            "enabled helper must prepare and assign the integration result exactly once"
        )
    integration_calls = [
        node
        for node in ast.walk(enabled)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_forecast_operational_segmented"
        )
    ]
    expected_args = ast.parse(
        "run_forecast_operational_segmented("
        "prepared_state, namelist, hours, segment_steps=seg)"
    ).body[0].value
    if (
        len(integration_calls) != 1
        or not isinstance(expected_args, ast.Call)
        or _ast_digest(integration_calls[0]) != _ast_digest(expected_args)
    ):
        raise DiffValidationError(
            "enabled helper changed the frozen integration arguments or call count"
        )
    # ADR-038: identical segment resolution to the default-off dispatch, so
    # enabled-vs-default numerical identity holds by construction.
    seg_assignments = [
        node
        for node in enabled.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "seg"
                for target in node.targets
            )
        )
    ]
    expected_seg_resolution = ast.parse(
        'seg = int(os.environ.get("GPUWRF_FORECAST_SEGMENT_STEPS", "34"))'
    ).body[0]
    if (
        len(seg_assignments) != 1
        or _ast_digest(seg_assignments[0]) != _ast_digest(expected_seg_resolution)
    ):
        raise DiffValidationError(
            "enabled helper changed the default segment-length resolution"
        )
    seg_guards = [
        node
        for node in enabled.body
        if (
            isinstance(node, ast.If)
            and _ast_digest(node.test)
            == _ast_digest(ast.parse("seg <= 0").body[0].value)
        )
    ]
    expected_seg_guard = ast.parse(
        "if seg <= 0:\n"
        "    raise RuntimeError(\n"
        '        "GPUWRF_FORECAST_SEGMENT_STEPS must be positive"\n'
        "    )\n"
    ).body[0]
    if (
        len(seg_guards) != 1
        or _ast_digest(seg_guards[0]) != _ast_digest(expected_seg_guard)
    ):
        raise DiffValidationError(
            "enabled helper changed the segment-length positivity guard"
        )
    prepared_assignments = [
        node
        for node in enabled.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "prepared_state"
                for target in node.targets
            )
        )
    ]
    expected_preparation = ast.parse(
        "prepared_state = _dealias_pytree_buffers(state)"
    ).body[0]
    if (
        len(prepared_assignments) != 1
        or _ast_digest(prepared_assignments[0]) != _ast_digest(expected_preparation)
    ):
        raise DiffValidationError("enabled helper changed donate-safety preparation")
    integration_ranges = [
        node
        for node in enabled.body
        if isinstance(node, ast.With)
    ]
    expected_range = ast.parse(
        "with jax.profiler.TraceAnnotation(_M0_EVIDENCE_RANGE):\n"
        "    result = run_forecast_operational_segmented(\n"
        "        prepared_state, namelist, hours, segment_steps=seg\n"
        "    )\n"
        "    jax.block_until_ready(result)\n"
    ).body[0]
    if (
        len(integration_ranges) != 1
        or _ast_digest(integration_ranges[0]) != _ast_digest(expected_range)
    ):
        raise DiffValidationError(
            "enabled helper range is not the exact integration call plus final sync"
        )
    if (
        not isinstance(enabled.body[-1], ast.Return)
        or not isinstance(enabled.body[-1].value, ast.Name)
        or enabled.body[-1].value.id != "result"
    ):
        raise DiffValidationError("enabled helper does not return the exact result")

    expected_enabled_return = ast.parse(
        "return _m0_evidence_run(state, namelist, hours)"
    ).body[0]
    if _ast_digest(current_statements[7]) != _ast_digest(expected_enabled_return):
        raise DiffValidationError("enabled wrapper changed numerical call arguments")

    # ADR-038: every HLO-producing body of the default entry (and the retained
    # monolithic escape hatch) must be unchanged relative to the adjudicated
    # flip base -- the evidence re-bind may not touch numerics.
    jit_hash = None
    for body_name in EXPECTED_UNCHANGED_HLO_BODIES:
        body_key = ("function", body_name)
        body_base = base_definitions.get(body_key)
        body_current = current_definitions.get(body_key)
        if body_base is None or body_current is None:
            raise DiffValidationError(f"default-entry forecast body is missing: {body_name}")
        body_hash = _ast_digest(body_current)
        if body_name == "_run_forecast_operational_jit":
            jit_hash = body_hash
        if body_hash != _ast_digest(body_base):
            raise DiffValidationError(
                f"default-entry forecast HLO source changed: {body_name}"
            )

    for name, expected_digest in EXPECTED_PRIVATE_HELPER_AST_SHA256.items():
        helper = current_definitions.get(("function", name))
        if helper is None or _ast_digest(helper) != expected_digest:
            raise DiffValidationError(f"reviewed ADR helper body changed: {name}")

    candidate_model = _top_level_statement_model(candidate_tree)
    current_model = _top_level_statement_model(current_tree)
    if current_model != candidate_model:
        raise DiffValidationError(
            "top-level statement model differs from immutable candidate; "
            "every import-time statement kind/order/body is pinned"
        )
    current_source_sha256 = hashlib.sha256(
        current_source.encode("utf-8")
    ).hexdigest()
    if current_source_sha256 != CANDIDATE_OPERATIONAL_SHA256:
        raise DiffValidationError(
            "operational source bytes differ from immutable candidate production source"
        )

    return {
        "status": "PASS",
        "existing_definitions_unchanged": len(base_definitions) - 1,
        "new_private_helpers": sorted(name for _kind, name in new_definitions),
        "new_private_constants": sorted(new_assignments),
        "added_imports": sorted(f"{module}:{name}" for module, name in added_imports),
        # ADR-038: the default direct call is the segmented dispatch return.
        "default_direct_call_ast_sha256": _ast_digest(expected_segmented_return),
        "default_off_jitted_body_ast_sha256": jit_hash,
        "top_level_statement_count": len(current_model),
        "top_level_statement_model_sha256": hashlib.sha256(
            json.dumps(current_model, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
        "candidate_operational_source_sha256": current_source_sha256,
        "reviewed_private_helper_ast_sha256": dict(
            sorted(EXPECTED_PRIVATE_HELPER_AST_SHA256.items())
        ),
        "reviewed_private_constant_values": dict(
            sorted(EXPECTED_PRIVATE_CONSTANT_VALUES.items())
        ),
        "hot_loop_callback_tokens_added": [],
    }


def validate_repository(base: str, adr: Path) -> dict[str, Any]:
    if not adr.is_file():
        raise DiffValidationError(f"ADR is missing: {adr}")
    adr_text = adr.read_text(encoding="utf-8")
    if (
        "Status: ACCEPTED" not in adr_text
        or "private, default-off profiling evidence only" not in adr_text
    ):
        raise DiffValidationError("ADR is not the accepted ADR-036 contract")

    candidate_commit = _run(
        ["git", "rev-parse", f"{CANDIDATE_PRODUCTION_COMMIT}^{{commit}}"]
    ).strip()
    if candidate_commit != CANDIDATE_PRODUCTION_COMMIT:
        raise DiffValidationError("candidate production commit did not resolve exactly")
    candidate_src_tree = _run(
        ["git", "rev-parse", f"{CANDIDATE_PRODUCTION_COMMIT}:src/gpuwrf"]
    ).strip()
    if candidate_src_tree != CANDIDATE_SRC_TREE:
        raise DiffValidationError("candidate src/gpuwrf tree does not match pinned tree")
    candidate_blob = _run(
        [
            "git",
            "rev-parse",
            f"{CANDIDATE_PRODUCTION_COMMIT}:{EXPECTED_CHANGED_FILE}",
        ]
    ).strip()
    if candidate_blob != CANDIDATE_OPERATIONAL_BLOB:
        raise DiffValidationError("candidate operational source blob does not match pin")
    production_delta = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            CANDIDATE_PRODUCTION_COMMIT,
            "--",
            "src/gpuwrf",
        ],
        cwd=REPO,
        check=False,
    )
    if production_delta.returncode != 0:
        raise DiffValidationError(
            "working production source differs from immutable candidate commit/tree"
        )

    changed = [
        line.strip()
        for line in _run(
            ["git", "diff", "--name-only", base, "--", "src/gpuwrf"]
        ).splitlines()
        if line.strip()
    ]
    changed_existing = [path for path in changed if (REPO / path).exists()]
    if len(changed_existing) > 2:
        raise DiffValidationError(
            f"ADR-036 allows at most two existing src files, got {changed_existing}"
        )
    if any(path not in ALLOWED_EXISTING_FILES for path in changed_existing):
        raise DiffValidationError(
            f"production changes escape operational-driver/runtime allowlist: {changed_existing}"
        )
    if changed != [EXPECTED_CHANGED_FILE]:
        raise DiffValidationError(
            f"expected only {EXPECTED_CHANGED_FILE}, got {changed}"
        )

    base_source = _git_file(base, EXPECTED_CHANGED_FILE)
    candidate_source = _git_file(
        CANDIDATE_PRODUCTION_COMMIT, EXPECTED_CHANGED_FILE
    )
    current_source = (REPO / EXPECTED_CHANGED_FILE).read_text(encoding="utf-8")
    semantic = validate_sources(
        base_source, current_source, candidate_source=candidate_source
    )
    validator_sha256 = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    return {
        "schema": "wrf_gpu2.v025.m0.instrumentation_diff_validation.v2",
        **semantic,
        "base": base,
        "adr": str(adr),
        "adr_sha256": hashlib.sha256(adr_text.encode("utf-8")).hexdigest(),
        "candidate_production_commit": candidate_commit,
        "candidate_src_tree": candidate_src_tree,
        "candidate_operational_blob": candidate_blob,
        "working_production_source_matches_candidate": True,
        "validator_sha256": validator_sha256,
        "binding_note": (
            "PASS is bound to immutable candidate production commit/tree and this "
            "validator's content hash; no HEAD hash is recorded, avoiding a "
            "self-referential committed proof."
        ),
        "changed_src_files": changed,
        "existing_src_files_changed": len(changed_existing),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adr", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = validate_repository(args.base, args.adr)
    except DiffValidationError as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, indent=2))
        return 1
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
