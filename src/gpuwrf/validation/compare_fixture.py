"""Tier-1 fixture comparison CLI."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shlex
import sys
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "1"
MAX_SAMPLE_SLICE_BYTES = 100_000
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOP_LEVEL_FIELDS = {
    "fixture_id",
    "source",
    "source_commit",
    "wrf_version",
    "scenario",
    "created_utc",
    "tier",
    "precision_reference",
    "generation_command",
    "external_uri",
    "sample_slice_path",
    "git_commit",
    "license_notes",
    "variables",
    "files",
}
VARIABLE_FIELDS = {
    "name",
    "units",
    "shape",
    "staggering",
    "dtype",
    "tolerance_abs",
    "tolerance_rel",
    "tolerance_rationale",
    "tier_overrides",
}
FILE_FIELDS = {"path", "checksum_sha256", "bytes", "external"}


class ManifestError(ValueError):
    """Raised when a fixture manifest violates the pinned schema."""


class CompareError(RuntimeError):
    """Raised when comparison input cannot be loaded or compared."""


def _load_structured(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            return json.load(handle)
        return yaml.safe_load(handle)


def load_manifest(path: Path) -> dict[str, Any]:
    data = _load_structured(path)
    if not isinstance(data, dict):
        raise ManifestError("$: expected mapping")
    errors = validate_manifest(data, manifest_path=path)
    if errors:
        raise ManifestError("; ".join(errors))
    return data


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _require_string(errors: list[str], data: dict[str, Any], field: str, path: str) -> None:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        errors.append(f"{path}.{field}: expected non-empty string")


def _validate_tolerance(errors: list[str], value: Any, path: str) -> None:
    if not _is_number(value) or float(value) < 0:
        errors.append(f"{path}: expected non-negative finite number")


def _validate_tier_overrides(errors: list[str], value: Any, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        errors.append(f"{path}: expected null or mapping")
        return
    for tier, override in value.items():
        tier_key = str(tier)
        if tier_key not in {"1", "2", "3", "4"}:
            errors.append(f"{path}.{tier}: tier override key must be one of 1, 2, 3, 4")
            continue
        if not isinstance(override, dict):
            errors.append(f"{path}.{tier}: expected mapping")
            continue
        extra = set(override) - {"tolerance_abs", "tolerance_rel"}
        missing = {"tolerance_abs", "tolerance_rel"} - set(override)
        for field in sorted(missing):
            errors.append(f"{path}.{tier}.{field}: missing required field")
        for field in sorted(extra):
            errors.append(f"{path}.{tier}.{field}: unknown field")
        if "tolerance_abs" in override:
            _validate_tolerance(errors, override["tolerance_abs"], f"{path}.{tier}.tolerance_abs")
        if "tolerance_rel" in override:
            _validate_tolerance(errors, override["tolerance_rel"], f"{path}.{tier}.tolerance_rel")


def validate_manifest(data: dict[str, Any], manifest_path: Path | None = None) -> list[str]:
    """Return schema validation errors for a manifest mapping."""

    errors: list[str] = []
    missing = TOP_LEVEL_FIELDS - set(data)
    extra = set(data) - TOP_LEVEL_FIELDS
    for field in sorted(missing):
        errors.append(f"$.{field}: missing required field")
    for field in sorted(extra):
        errors.append(f"$.{field}: unknown field")

    for field in ("fixture_id", "source_commit", "scenario", "created_utc", "generation_command", "git_commit", "license_notes"):
        if field in data:
            _require_string(errors, data, field, "$")

    if data.get("source") not in {"analytic", "wrf-derived"}:
        errors.append("$.source: expected one of analytic, wrf-derived")
    if data.get("source") == "wrf-derived" and (not isinstance(data.get("wrf_version"), str) or not data.get("wrf_version")):
        errors.append("$.wrf_version: required non-empty string when source is wrf-derived")
    elif "wrf_version" in data and data.get("wrf_version") is not None and not isinstance(data.get("wrf_version"), str):
        errors.append("$.wrf_version: expected string or null")

    if data.get("tier") not in {1, 2, 3, 4}:
        errors.append("$.tier: expected integer 1, 2, 3, or 4")
    if data.get("precision_reference") not in {"fp64", "fp32", "bf16", "fp16"}:
        errors.append("$.precision_reference: expected one of fp64, fp32, bf16, fp16")
    for field in ("external_uri", "sample_slice_path"):
        if field in data and data[field] is not None and (not isinstance(data[field], str) or not data[field]):
            errors.append(f"$.{field}: expected non-empty string or null")

    sample_slice = data.get("sample_slice_path")
    if sample_slice and manifest_path is not None:
        sample_path = (ROOT / sample_slice).resolve()
        try:
            sample_path.relative_to(ROOT)
        except ValueError:
            errors.append("$.sample_slice_path: must be relative to the repository root")
        if not sample_path.exists():
            errors.append("$.sample_slice_path: referenced file does not exist")
        elif sample_path.stat().st_size > MAX_SAMPLE_SLICE_BYTES:
            errors.append("$.sample_slice_path: referenced file exceeds 100000 bytes")

    variables = data.get("variables")
    if not isinstance(variables, list) or not variables:
        errors.append("$.variables: expected non-empty list")
    elif isinstance(variables, list):
        for index, variable in enumerate(variables):
            path = f"$.variables[{index}]"
            if not isinstance(variable, dict):
                errors.append(f"{path}: expected mapping")
                continue
            missing_var = VARIABLE_FIELDS - set(variable)
            extra_var = set(variable) - VARIABLE_FIELDS
            for field in sorted(missing_var):
                errors.append(f"{path}.{field}: missing required field")
            for field in sorted(extra_var):
                errors.append(f"{path}.{field}: unknown field")
            for field in ("name", "units", "dtype", "tolerance_rationale"):
                if field in variable:
                    _require_string(errors, variable, field, path)
            rationale = variable.get("tolerance_rationale")
            if isinstance(rationale, str) and len(rationale) > 200:
                errors.append(f"{path}.tolerance_rationale: expected at most 200 characters")
            shape = variable.get("shape")
            if not isinstance(shape, list) or not shape:
                errors.append(f"{path}.shape: expected non-empty list")
            elif any(not isinstance(dim, int) or isinstance(dim, bool) or dim < 1 for dim in shape):
                errors.append(f"{path}.shape: expected positive integer dimensions")
            if variable.get("staggering") not in {"mass", "u", "v", "w", "m"}:
                errors.append(f"{path}.staggering: expected one of mass, u, v, w, m")
            if "tolerance_abs" in variable:
                _validate_tolerance(errors, variable["tolerance_abs"], f"{path}.tolerance_abs")
            if "tolerance_rel" in variable:
                _validate_tolerance(errors, variable["tolerance_rel"], f"{path}.tolerance_rel")
            if "tier_overrides" in variable:
                _validate_tier_overrides(errors, variable["tier_overrides"], f"{path}.tier_overrides")

    files = data.get("files")
    if not isinstance(files, list):
        errors.append("$.files: expected list")
    elif isinstance(files, list):
        for index, file_entry in enumerate(files):
            path = f"$.files[{index}]"
            if not isinstance(file_entry, dict):
                errors.append(f"{path}: expected mapping")
                continue
            missing_file = FILE_FIELDS - set(file_entry)
            extra_file = set(file_entry) - FILE_FIELDS
            for field in sorted(missing_file):
                errors.append(f"{path}.{field}: missing required field")
            for field in sorted(extra_file):
                errors.append(f"{path}.{field}: unknown field")
            _require_string(errors, file_entry, "path", path)
            checksum = file_entry.get("checksum_sha256")
            if not isinstance(checksum, str) or not SHA256_RE.fullmatch(checksum):
                errors.append(f"{path}.checksum_sha256: expected 64 lowercase hex characters")
            byte_count = file_entry.get("bytes")
            if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
                errors.append(f"{path}.bytes: expected non-negative integer")
            if not isinstance(file_entry.get("external"), bool):
                errors.append(f"{path}.external: expected boolean")

    return errors


def _load_array_map(path: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    if not path.exists():
        raise CompareError(f"{path}: file does not exist")
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            return {name: loaded[name] for name in loaded.files}
    variable_names = [variable["name"] for variable in manifest["variables"]]
    if len(variable_names) != 1:
        raise CompareError(f"{path}: .npy input is only valid for single-variable manifests")
    return {variable_names[0]: loaded}


def _tolerances_for_tier(variable: dict[str, Any], tier: int) -> tuple[float, float]:
    overrides = variable.get("tier_overrides")
    if isinstance(overrides, dict) and str(tier) in overrides:
        override = overrides[str(tier)]
        return float(override["tolerance_abs"]), float(override["tolerance_rel"])
    return float(variable["tolerance_abs"]), float(variable["tolerance_rel"])


def _breach_score(result: dict[str, Any]) -> tuple[float, float]:
    rel_tol = result["tolerance_rel"]
    abs_tol = result["tolerance_abs"]
    rel_score = math.inf if rel_tol == 0 and result["max_rel_diff"] > 0 else result["max_rel_diff"] / rel_tol if rel_tol else 0.0
    abs_score = math.inf if abs_tol == 0 and result["max_abs_diff"] > 0 else result["max_abs_diff"] / abs_tol if abs_tol else 0.0
    return rel_score, abs_score


def compare_arrays(manifest: dict[str, Any], candidate_path: Path, reference_path: Path, tier: int, command: str) -> dict[str, Any]:
    candidate = _load_array_map(candidate_path, manifest)
    reference = _load_array_map(reference_path, manifest)
    variable_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for variable in manifest["variables"]:
        name = variable["name"]
        expected_shape = tuple(variable["shape"])
        if name not in candidate:
            raise CompareError(f"candidate missing variable {name}")
        if name not in reference:
            raise CompareError(f"reference missing variable {name}")

        cand = np.asarray(candidate[name])
        ref = np.asarray(reference[name])
        shape_ok = tuple(cand.shape) == expected_shape and tuple(ref.shape) == expected_shape
        tolerance_abs, tolerance_rel = _tolerances_for_tier(variable, tier)
        if not shape_ok:
            result = {
                "name": name,
                "pass": False,
                "shape_ok": False,
                "max_abs_diff": None,
                "max_rel_diff": None,
                "tolerance_abs": tolerance_abs,
                "tolerance_rel": tolerance_rel,
                "violation_index": None,
            }
            variable_results.append(result)
            failures.append({"name": name, "score": (math.inf, math.inf)})
            continue

        diff = np.abs(cand.astype(np.float64) - ref.astype(np.float64))
        max_abs = float(np.max(diff)) if diff.size else 0.0
        rel = diff / (np.abs(ref.astype(np.float64)) + np.finfo(np.float64).eps)
        max_rel = float(np.max(rel)) if rel.size else 0.0
        within_tolerance = bool(np.all(diff <= (tolerance_abs + tolerance_rel * np.abs(ref.astype(np.float64)))))
        violation_index = None
        if not within_tolerance and diff.size:
            allowed = tolerance_abs + tolerance_rel * np.abs(ref.astype(np.float64))
            breach = diff - allowed
            violation_index = [int(part) for part in np.unravel_index(int(np.argmax(breach)), breach.shape)]

        result = {
            "name": name,
            "pass": within_tolerance,
            "shape_ok": True,
            "max_abs_diff": max_abs,
            "max_rel_diff": max_rel,
            "tolerance_abs": tolerance_abs,
            "tolerance_rel": tolerance_rel,
            "violation_index": violation_index,
        }
        variable_results.append(result)
        if not within_tolerance:
            failures.append({"name": name, "score": _breach_score(result)})

    first_failure = None
    if failures:
        failures.sort(key=lambda item: item["score"], reverse=True)
        first_failure = failures[0]["name"]

    return {
        "fixture_id": manifest["fixture_id"],
        "tier": tier,
        "pass": not failures,
        "variables": variable_results,
        "first_failure": first_failure,
        "command": command,
        "schema_version": SCHEMA_VERSION,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare candidate NumPy arrays against a GPUWRF fixture manifest.")
    parser.add_argument("--manifest", required=True, help="Fixture manifest YAML or JSON path.")
    parser.add_argument("--candidate", required=True, help="Candidate NumPy .npz or .npy array file.")
    parser.add_argument("--reference", help="Reference NumPy .npz or .npy array file; defaults to manifest sample_slice_path.")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3, 4], help="Validation tier selecting tolerance overrides.")
    parser.add_argument("--out", help="Output path for the comparison JSON record; defaults to stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = " ".join(shlex.quote(part) for part in [sys.executable, "-m", "gpuwrf.validation.compare_fixture", *(argv if argv is not None else sys.argv[1:])])

    try:
        manifest_path = Path(args.manifest)
        manifest = load_manifest(manifest_path)
        reference_arg = args.reference or manifest.get("sample_slice_path")
        if not reference_arg:
            raise CompareError("--reference is required when manifest sample_slice_path is null")
        record = compare_arrays(manifest, Path(args.candidate), Path(reference_arg), args.tier, command)
    except (ManifestError, CompareError, OSError, ValueError) as exc:
        print(f"compare_fixture: {exc}", file=sys.stderr)
        return 2

    output = json.dumps(record, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
