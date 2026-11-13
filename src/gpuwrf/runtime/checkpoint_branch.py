"""Declared, host-only changes to a verified synchronized checkpoint.

This is an explicit fork of a local trusted checkpoint, never a relaxation of
ordinary restart identity checks. The step program receives the usual carry.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
STATE_FIELDS = frozenset("u v w theta qv qc qr qi qs qg qh Nc Nn Ni Nr Ns Ng Nh qvolg qvolh nwfa nifa qke p_perturbation ph_perturbation t_skin".split())
LAND_FIELDS = frozenset("smois tslb sh2o tsno snice snliq snowh sneqv canliq canice tv tg tah lai sai t_skin".split())
PARAMETER_KEYS = frozenset({"surface_controls", "diffusion_parameters", "mynn_parameters", "thompson_parameters"})


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate branch manifest key: {key}")
        result[key] = value
    return result


def load_manifest(path):
    path = Path(path)
    if path.stat().st_size > 1024**2:
        raise ValueError("branch manifest exceeds 1 MiB")
    def invalid(value):
        raise ValueError(f"nonfinite branch manifest value: {value}")
    value = json.loads(path.read_text(), object_pairs_hook=_unique, parse_constant=invalid)
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("branch manifest requires schema_version=1")
    if set(value) - {"schema_version", "domains", "parameters", "waivers"}:
        raise ValueError("unknown branch manifest keys")
    for key in ("domains", "parameters", "waivers"):
        if not isinstance(value.get(key, {}), dict):
            raise ValueError(f"branch {key} must be an object")
    if set(value.get("parameters", {})) - PARAMETER_KEYS:
        raise ValueError("branch parameter changes require a supported traced parameter family")
    # Bind every external array by its bytes. No implicit broadcasting or pickle.
    for domain, groups in value.get("domains", {}).items():
        if not isinstance(groups, dict) or set(groups) - {"state", "land"}:
            raise ValueError(f"{domain}: only state/land overrides are supported")
        for group, edits in groups.items():
            allowed = STATE_FIELDS if group == "state" else LAND_FIELDS
            if not isinstance(edits, dict) or set(edits) - allowed:
                raise ValueError(f"{domain}/{group}: unsupported field; MU/base/static edits are refused")
            for field, operation in edits.items():
                if not isinstance(operation, dict) or len(operation) != 1 or not set(operation) <= {"set", "add", "multiply"}:
                    raise ValueError(f"{domain}/{field}: choose exactly one of set/add/multiply")
                operand = next(iter(operation.values()))
                if isinstance(operand, dict):
                    if set(operand) != {"path", "sha256"}:
                        raise ValueError("array operand needs path and sha256")
                    source = (path.parent / operand["path"]).resolve()
                    if source.suffix != ".npy":
                        raise ValueError("branch spatial operands must be .npy arrays")
                    actual = _file_digest(source)
                    if actual != operand["sha256"]:
                        raise ValueError("branch array hash mismatch")
                    operand["path"] = str(source)
                elif isinstance(operand, bool) or not isinstance(operand, (int, float)) or not np.isfinite(operand):
                    raise ValueError("branch operands must be finite numbers or hashed .npy arrays")
    digest(value)  # Reject overflow/nonfinite values anywhere, including parameter metadata.
    return value


def branch_identity(parent_manifest, current, changes, generation):
    """Compare all identity entries, allowing only explicit traced-value edits."""
    if parent_manifest.get("branch_schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint predates branch schema (E78); write a new checkpoint")
    parent = parent_manifest["identity"]
    for side, ident in (("parent", parent), ("branch", current)):
        if ident.get("feedback") or ident.get("feedback_schema") is not None:
            raise ValueError(f"{side} uses two-way feedback: checkpoint branching supports one-way nesting only")
    expected = {k: v for k, v in parent.items() if k != "branch"}
    if current["output_dir"] == parent["output_dir"]:
        raise ValueError("branch requires a new output directory")
    expected["output_dir"] = current["output_dir"]
    for family, domains in changes.get("parameters", {}).items():
        if family not in PARAMETER_KEYS or family not in expected or family not in current:
            raise ValueError(f"{family}: traced family must already be enabled in parent and child")
        if not isinstance(domains, dict) or not domains or set(domains) - set(parent_manifest["domains"]):
            raise ValueError(f"{family}: invalid parameter domains")
        allowed = dict(expected[family])
        for name, value in domains.items():
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("declared parameter value must be its SHA256 digest")
            allowed[name] = value
        expected[family] = allowed
    if expected != current:
        changed = sorted(k for k in set(expected) | set(current) if expected.get(k) != current.get(k))
        raise ValueError(f"undeclared branch identity change: {changed}")
    provenance = {"schema_version": SCHEMA_VERSION, "parent_identity": parent,
                  "parent_manifest_sha256": hashlib.sha256((Path(generation) / "manifest.json").read_bytes()).hexdigest(),
                  "parent_payload_sha256": parent_manifest["sha256"],
                  "own_steps": parent_manifest["own_steps"],
                  "manifest": changes, "manifest_sha256": digest(changes)}
    return {**current, "branch": provenance}


def resumed_identity(current, generation):
    """Retain branch provenance on subsequent ordinary resumes in that stream."""
    saved = json.loads((Path(generation) / "manifest.json").read_text())["identity"]
    if "branch" not in saved:
        return current
    branch = saved["branch"]
    if branch.get("schema_version") != SCHEMA_VERSION or digest(branch["manifest"]) != branch["manifest_sha256"]:
        raise ValueError("corrupt branch provenance")
    return {**current, "branch": branch}


def validate_paths(generation, output_dir, checkpoint_dir, scratch_dir=None):
    """``scratch_dir``: this run's own transient scratch (the CLI default ``<output-dir>/.scratch`` exists
    before the driver re-checks); that one directory is the only entry an empty branch output may hold."""
    parent = json.loads((Path(generation) / "manifest.json").read_text())["identity"]
    output, old = Path(output_dir).resolve(), Path(parent["output_dir"]).resolve()
    if output == old or output.is_relative_to(old) or old.is_relative_to(output):
        raise ValueError("branch output must be separate from the parent stream")
    own = None if scratch_dir is None else Path(scratch_dir).resolve()
    if output.exists() and any(not (entry.is_dir() and not entry.is_symlink() and entry.resolve() == own)
                               for entry in output.iterdir()):
        raise ValueError("branch output directory must be empty")
    if checkpoint_dir is not None and Path(checkpoint_dir).resolve().is_relative_to(Path(generation).resolve().parent):
        raise ValueError("branch checkpoint directory must be separate from the parent store")


def _edit(old, operation):
    old = np.asarray(old)
    if old.dtype.kind != "f" or not np.all(np.isfinite(old)):
        raise ValueError("override target must be an existing finite floating array")
    action, operand = next(iter(operation.items()))
    if isinstance(operand, dict):
        path = Path(operand["path"])
        if _file_digest(path) != operand["sha256"]:
            raise ValueError("branch array hash changed after admission")
        operand = np.load(path, allow_pickle=False, mmap_mode="r")
        if operand.shape != old.shape or operand.dtype.kind not in "fiu":
            raise ValueError("branch array shape/type must match target exactly")
    if not np.all(np.isfinite(operand)):
        raise ValueError("nonfinite branch operand")
    with np.errstate(over="ignore", invalid="ignore"):
        if action == "set":
            new = np.broadcast_to(operand, old.shape).astype(old.dtype)
        else:
            new = (old.astype(np.float64) + operand if action == "add" else old.astype(np.float64) * operand).astype(old.dtype)
        # Preserve stored bits in unchanged cells (including signed zero).
        new = np.where(new == old, old, new)
    if not np.all(np.isfinite(new)):
        raise ValueError("branch override produced NaN/Inf")
    return new


def apply_overrides(snapshot, changes, *, validate):
    """Copy only edited payload arrays; validate every domain before device puts.

    ``validate`` is the shared-consistency adapter. It is called for each domain,
    including untouched domains. A refusal leaves the parent snapshot untouched.
    """
    if snapshot["driver_state"].get("regional_streams") is not None:
        raise ValueError("branching a regional-stream checkpoint is unsupported; its output journals belong to the parent")
    names = set(snapshot["carries"])
    if set(changes.get("domains", {})) - names or set(changes.get("waivers", {})) - names:
        raise ValueError("branch manifest names a missing domain")
    carries, reports, changed = {}, {}, {}
    for name, payload in snapshot["carries"].items():
        edits = changes.get("domains", {}).get(name, {})
        result = dict(payload)
        changed[name] = {}
        for group, key in (("state", "state_fields"), ("land", "noahmp_land_fields")):
            fields = payload.get(key)
            if edits.get(group) and fields is None:
                raise ValueError(f"{name}: checkpoint has no {group} carry")
            result[key] = None if fields is None else dict(fields)
            for field, operation in edits.get(group, {}).items():
                if fields.get(field) is None:
                    raise ValueError(f"{name}: checkpoint has no {group}/{field} leaf")
                old = np.asarray(fields[field])
                new = _edit(old, operation)
                result[key][field] = new
                changed[name][f"{group}.{field}"] = int(np.count_nonzero(new != old))
                partner = {"p_perturbation": "p_total", "ph_perturbation": "ph_total"}.get(field) if group == "state" else None
                if partner:
                    # Both public representations receive the same REAL increment.
                    delta = new.astype(np.float64) - old.astype(np.float64)
                    result[key][partner] = _edit(fields[partner], {"add": delta})
        reports[name] = validate(name, result, edits, changes.get("waivers", {}).get(name, {}), payload)
        carries[name] = result
    driver = dict(snapshot["driver_state"])
    driver["outputs"] = {name: [] for name in carries}
    driver.pop("history_stream", None)
    driver["branch_start_steps"] = dict(snapshot["own_steps"])
    return {**snapshot, "carries": carries, "driver_state": driver}, {"fields": changed, "consistency": reports}
