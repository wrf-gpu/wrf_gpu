#!/usr/bin/env python3
"""Mechanical inventory of the M0 child pre-device authority guards.

The rejected session-label repair compared a hand-written 16-name tuple with a
hand-written mutation table.  This module instead walks the actual enforcing
AST:

* tagged refusals in the child handoff consumer and reader;
* tagged namespace/context refusals in the frozen registry; and
* every ``WindowNotAuthorised`` raise site in the canonical lock checker,
  matched mechanically to the child's structured-refusal classifier.

It imports none of those modules, so deriving the inventory cannot import JAX,
``gpuwrf``, or touch a device.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class GuardSite:
    """One actual source refusal site that enforces a named authority guard."""

    guard: str
    provider: str
    function: str
    line: int
    mechanism: str


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one function {name!r}, found {len(matches)}")
    return matches[0]


def _call_name(call: ast.Call) -> str:
    return str(getattr(call.func, "id", getattr(call.func, "attr", "")))


def _literal_string(node: ast.AST) -> str:
    value = ast.literal_eval(node)
    if not isinstance(value, str) or not value:
        raise ValueError(f"guard identifier is not one nonempty literal: {value!r}")
    return value


def _tagged_sites(
    path: Path,
    *,
    functions: Iterable[str],
    helper: str,
    guard_argument: int,
) -> list[GuardSite]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    sites: list[GuardSite] = []
    for function_name in functions:
        function = _function(tree, function_name)
        for node in ast.walk(function):
            if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                continue
            call = node.exc
            if _call_name(call) != helper:
                continue
            if len(call.args) <= guard_argument:
                raise ValueError(
                    f"{path.name}:{node.lineno} {helper} lacks guard argument"
                )
            guard_node = call.args[guard_argument]
            if (
                helper == "_authority_refusal"
                and ast.unparse(guard_node) == "matches[0]"
            ):
                # The seven possible values are derived independently from the
                # canonical lock checker's raise sites below.
                continue
            sites.append(
                GuardSite(
                    guard=_literal_string(guard_node),
                    provider=path.name,
                    function=function_name,
                    line=node.lineno,
                    mechanism=helper,
                )
            )
    return sites


def _literal_lock_classifier(path: Path) -> tuple[tuple[str, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "LOCK_REFUSAL_GUARDS"
                    for target in node.targets
                )
            )
            or (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "LOCK_REFUSAL_GUARDS"
            )
        )
    ]
    if len(assignments) != 1:
        raise ValueError(
            f"expected one LOCK_REFUSAL_GUARDS assignment, found {len(assignments)}"
        )
    assignment = assignments[0]
    value = assignment.value
    if value is None:
        raise ValueError("LOCK_REFUSAL_GUARDS has no value")
    literal = ast.literal_eval(value)
    result = tuple((str(prefix), str(guard)) for prefix, guard in literal)
    if (
        len(result) != len(set(result))
        or len({prefix for prefix, _ in result}) != len(result)
        or len({guard for _, guard in result}) != len(result)
    ):
        raise ValueError("LOCK_REFUSAL_GUARDS entries must be unique")
    return result


def _static_prefix(node: ast.AST) -> str:
    """Return the leading runtime-static text of an exception message AST."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
                continue
            break
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_prefix(node.left)
        if not left:
            return ""
        if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
            return left + _static_prefix(node.right)
        return left
    return ""


def _lock_sites(run_gpu_arm: Path, child: Path) -> list[GuardSite]:
    source = run_gpu_arm.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(run_gpu_arm))
    function = _function(tree, "check_canonical_lock")
    raises: list[tuple[int, str]] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
            continue
        if _call_name(node.exc) != "WindowNotAuthorised":
            continue
        if len(node.exc.args) != 1:
            raise ValueError(
                f"{run_gpu_arm.name}:{node.lineno} lock refusal is not unary"
            )
        anchor = _static_prefix(node.exc.args[0])
        if not anchor:
            raise ValueError(
                f"{run_gpu_arm.name}:{node.lineno} has no static refusal anchor"
            )
        raises.append((node.lineno, anchor))

    classifier = _literal_lock_classifier(child)
    sites: list[GuardSite] = []
    used_prefixes: set[str] = set()
    for line, anchor in raises:
        matches = [
            (prefix, guard)
            for prefix, guard in classifier
            if anchor.startswith(prefix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{run_gpu_arm.name}:{line} lock anchor {anchor!r} maps to "
                f"{len(matches)} classifier entries"
            )
        prefix, guard = matches[0]
        used_prefixes.add(prefix)
        sites.append(
            GuardSite(
                guard=guard,
                provider=run_gpu_arm.name,
                function="check_canonical_lock",
                line=line,
                mechanism=f"WindowNotAuthorised:{prefix}",
            )
        )
    unused = {prefix for prefix, _ in classifier} - used_prefixes
    if unused:
        raise ValueError(f"lock classifier entries have no actual raise site: {unused}")
    return sites


def derive_pre_device_guard_sites(
    script_dir: Path | None = None,
) -> tuple[GuardSite, ...]:
    """Derive every guard in the reachable pre-device authority call graph."""

    root = Path(script_dir or Path(__file__).resolve().parent)
    child = root / "m0_exact_boundary_child.py"
    registry = root / "gpu_window_registry.py"
    run_gpu_arm = root / "run_gpu_arm.py"
    sites = [
        *_tagged_sites(
            child,
            functions=(
                "_read_json_regular_no_symlink",
                "consume_authorization_handoff",
            ),
            helper="_authority_refusal",
            guard_argument=0,
        ),
        *_tagged_sites(
            registry,
            functions=(
                "assert_registered",
                "assert_session_label",
                "assert_stage_identity",
                "assert_launch_context",
            ),
            helper="_guarded_exception",
            guard_argument=1,
        ),
        *_lock_sites(run_gpu_arm, child),
    ]
    guards = [site.guard for site in sites]
    duplicates = sorted({guard for guard in guards if guards.count(guard) > 1})
    if duplicates:
        raise ValueError(f"authority guard IDs are not one-site unique: {duplicates}")
    return tuple(sorted(sites, key=lambda site: site.guard))


def derive_pre_device_guard_inventory(
    script_dir: Path | None = None,
) -> tuple[str, ...]:
    """Return the mechanically derived, sorted guard-name inventory."""

    return tuple(site.guard for site in derive_pre_device_guard_sites(script_dir))


def guard_inventory_sha256(script_dir: Path | None = None) -> str:
    """Content address the complete guard-site inventory, including provenance."""

    encoded = json.dumps(
        [asdict(site) for site in derive_pre_device_guard_sites(script_dir)],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "GuardSite",
    "derive_pre_device_guard_inventory",
    "derive_pre_device_guard_sites",
    "guard_inventory_sha256",
]
