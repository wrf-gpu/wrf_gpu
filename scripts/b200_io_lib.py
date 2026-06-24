"""B200 paid-pod I/O readiness helper library.

The helpers in this module are intentionally CPU-only.  They validate staged
input manifests, WRF namelist dimensions, and block-level output drain state
before a paid B200 run is launched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from netCDF4 import Dataset


DEFAULT_DRAIN_PATTERNS = ("wrfout_d*", "*.nc")
PARTIAL_SUFFIXES = (".tmp", ".part", ".partial", ".incomplete")


@dataclass(frozen=True)
class RequiredInput:
    """One manifest-required input and the paths that may satisfy it."""

    logical_path: str
    candidates: tuple[str, ...]
    sha256: str | None = None
    size_bytes: int | None = None
    source: str = "manifest"


@dataclass(frozen=True)
class ExpectedDomain:
    """Expected WRF namelist dimensions for one domain."""

    index: int
    domain_id: str
    e_we: int
    e_sn: int
    parent_grid_ratio: int | None = None


@dataclass(frozen=True)
class DrainConfig:
    """Configuration for one drain/resume pass."""

    output_dir: Path
    target: str
    state_dir: Path
    patterns: tuple[str, ...] = DEFAULT_DRAIN_PATTERNS
    expected_vars: tuple[str, ...] = ()
    expected_dims: dict[str, int] | None = None
    min_age_seconds: float = 30.0
    local_cap_bytes: int | None = None
    target_cap_bytes: int | None = None
    target_min_free_bytes: int = 0
    delete_after_copy: bool = False
    dry_run: bool = False
    training_policy: str = "allow-partial"
    expected_block_count: int | None = None
    expected_time_steps: int | None = None
    min_time_steps: int = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level JSON value must be an object")
    return data


def _issue(severity: str, code: str, message: str, **extra: Any) -> dict[str, Any]:
    item = {"severity": severity, "code": code, "message": message}
    item.update({k: v for k, v in extra.items() if v is not None})
    return item


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _strip_namelist_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        quote: str | None = None
        out: list[str] = []
        for char in line:
            if char in {"'", '"'}:
                if quote == char:
                    quote = None
                elif quote is None:
                    quote = char
            if char == "!" and quote is None:
                break
            out.append(char)
        lines.append("".join(out))
    return "\n".join(lines)


def _split_namelist_values(raw: str) -> list[str]:
    values: list[str] = []
    quote: str | None = None
    token: list[str] = []
    for char in raw:
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
        if char == "," and quote is None:
            item = "".join(token).strip()
            if item:
                values.append(item)
            token = []
        else:
            token.append(char)
    item = "".join(token).strip()
    if item:
        values.append(item)
    return values


def parse_wrf_namelist(path: Path) -> dict[str, dict[str, list[str]]]:
    """Parse enough of a WRF namelist for dimension checks.

    This is not a full Fortran namelist parser.  It deliberately handles the
    common WRF shape of repeated ``key = comma, separated, values`` assignments
    inside named sections and preserves values as strings.
    """

    text = _strip_namelist_comments(path.read_text(encoding="utf-8"))
    sections: dict[str, dict[str, list[str]]] = {}
    section_re = re.compile(r"(?ims)&([A-Za-z0-9_]+)(.*?)(?:^\s*/\s*$|/)")
    key_re = re.compile(r"(?im)([A-Za-z][A-Za-z0-9_]*)\s*=")
    for section_match in section_re.finditer(text):
        section = section_match.group(1).lower()
        body = section_match.group(2)
        entries: dict[str, list[str]] = {}
        matches = list(key_re.finditer(body))
        for index, match in enumerate(matches):
            key = match.group(1).lower()
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            entries[key] = _split_namelist_values(body[start:end])
        sections[section] = entries
    return sections


def _to_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip().strip("'\"")
    return int(text)


def _to_int_list(values: Any, label: str) -> list[int]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [_to_int(v, label) for v in values if str(v).strip()]
    if isinstance(values, str):
        return [_to_int(v, label) for v in _split_namelist_values(values)]
    return [_to_int(values, label)]


def _entry_checksum(entry: dict[str, Any]) -> str | None:
    checksums = entry.get("checksums")
    nested_sha = checksums.get("sha256") if isinstance(checksums, dict) else None
    return _first_present(entry.get("sha256"), entry.get("checksum_sha256"), nested_sha)


def _entry_size(entry: dict[str, Any]) -> int | None:
    size = _first_present(entry.get("bytes"), entry.get("size_bytes"), entry.get("length"))
    return None if size is None else int(size)


def _coerce_file_entry(entry: Any, *, prefix: str = "", source: str = "manifest") -> RequiredInput:
    if isinstance(entry, str):
        rel = str(Path(prefix) / entry) if prefix else entry
        candidates = (rel, entry) if prefix else (entry,)
        return RequiredInput(logical_path=rel, candidates=tuple(dict.fromkeys(candidates)), source=source)
    if not isinstance(entry, dict):
        raise ValueError(f"unsupported required input entry: {entry!r}")
    raw_path = _first_present(entry.get("path"), entry.get("relative_path"), entry.get("name"), entry.get("file"))
    if raw_path is None:
        raise ValueError(f"required input entry lacks a path/name: {entry!r}")
    raw_path = str(raw_path)
    rel = str(Path(prefix) / raw_path) if prefix and not Path(raw_path).is_absolute() else raw_path
    candidates = [rel]
    if prefix:
        candidates.append(raw_path)
    if "candidates" in entry and isinstance(entry["candidates"], list):
        candidates.extend(str(item) for item in entry["candidates"])
    return RequiredInput(
        logical_path=rel,
        candidates=tuple(dict.fromkeys(candidates)),
        sha256=_entry_checksum(entry),
        size_bytes=_entry_size(entry),
        source=source,
    )


def collect_required_inputs(manifest: dict[str, Any]) -> list[RequiredInput]:
    """Collect required inputs from several manifest shapes.

    The final B200 manifest is expected to use an explicit ``required_inputs``
    list with SHA-256 values.  The existing B200 Swiss template uses
    ``inputs.staged_cases[].runtime_files`` without checksums; this collector
    reads that shape too, but the validator will fail closed when checksums are
    required and absent.
    """

    out: list[RequiredInput] = []
    containers: list[tuple[str, dict[str, Any]]] = [("$", manifest)]
    inputs = manifest.get("inputs")
    if isinstance(inputs, dict):
        containers.append(("$.inputs", inputs))
    for base, container in containers:
        for key in ("required_inputs", "required_files", "input_files", "files"):
            value = container.get(key)
            if isinstance(value, list):
                for entry in value:
                    out.append(_coerce_file_entry(entry, source=f"{base}.{key}"))
    if isinstance(inputs, dict) and isinstance(inputs.get("staged_cases"), list):
        for case in inputs["staged_cases"]:
            if not isinstance(case, dict):
                continue
            prefix = str(_first_present(case.get("s3_dir"), case.get("path"), case.get("id"), ""))
            for entry in case.get("runtime_files", []):
                spec = _coerce_file_entry(entry, prefix=prefix, source="$.inputs.staged_cases[].runtime_files")
                candidates = list(spec.candidates)
                if case.get("id"):
                    candidates.append(str(Path(str(case["id"])) / Path(spec.logical_path).name))
                candidates.append(Path(spec.logical_path).name)
                out.append(
                    RequiredInput(
                        logical_path=spec.logical_path,
                        candidates=tuple(dict.fromkeys(candidates)),
                        sha256=spec.sha256,
                        size_bytes=spec.size_bytes,
                        source=spec.source,
                    )
                )
    deduped: dict[str, RequiredInput] = {}
    for item in out:
        deduped[item.logical_path] = item
    return list(deduped.values())


def _domain_list_from_mapping(container: dict[str, Any], source: str) -> list[ExpectedDomain]:
    domains = container.get("domains")
    if isinstance(domains, list):
        out: list[ExpectedDomain] = []
        for index, item in enumerate(domains, start=1):
            if not isinstance(item, dict):
                continue
            if "e_we" not in item or "e_sn" not in item:
                continue
            domain_id = str(_first_present(item.get("id"), item.get("domain"), item.get("name"), f"d{index:02d}"))
            out.append(
                ExpectedDomain(
                    index=index,
                    domain_id=domain_id,
                    e_we=_to_int(item["e_we"], f"{source}.domains[{index - 1}].e_we"),
                    e_sn=_to_int(item["e_sn"], f"{source}.domains[{index - 1}].e_sn"),
                    parent_grid_ratio=(
                        None
                        if item.get("parent_grid_ratio") is None
                        else _to_int(item["parent_grid_ratio"], f"{source}.domains[{index - 1}].parent_grid_ratio")
                    ),
                )
            )
        if out:
            return out
    e_we = container.get("e_we")
    e_sn = container.get("e_sn")
    if e_we is not None and e_sn is not None:
        ratios = _to_int_list(container.get("parent_grid_ratio"), f"{source}.parent_grid_ratio")
        e_we_values = _to_int_list(e_we, f"{source}.e_we")
        e_sn_values = _to_int_list(e_sn, f"{source}.e_sn")
        out = []
        for index, (we, sn) in enumerate(zip(e_we_values, e_sn_values), start=1):
            ratio = ratios[index - 1] if index - 1 < len(ratios) else None
            out.append(ExpectedDomain(index=index, domain_id=f"d{index:02d}", e_we=we, e_sn=sn, parent_grid_ratio=ratio))
        return out
    return []


def extract_expected_domains(manifest: dict[str, Any]) -> list[ExpectedDomain]:
    candidates: list[tuple[str, Any]] = [
        ("$.wrf_domains", {"domains": manifest.get("wrf_domains")}),
        ("$.domains", {"domains": manifest.get("domains")}),
        ("$.wrf", manifest.get("wrf")),
        ("$.grid", manifest.get("grid")),
        ("$.intended_grid", manifest.get("intended_grid")),
    ]
    inputs = manifest.get("inputs")
    if isinstance(inputs, dict):
        candidates.extend(
            [
                ("$.inputs.domains", {"domains": inputs.get("domains")}),
                ("$.inputs.grid", inputs.get("grid")),
                ("$.inputs.intended_grid", inputs.get("intended_grid")),
            ]
        )
    for source, value in candidates:
        if isinstance(value, dict):
            domains = _domain_list_from_mapping(value, source)
            if domains:
                return domains
    return []


def _resolve_candidate(staged_dir: Path, spec: RequiredInput) -> Path | None:
    for candidate in spec.candidates:
        candidate_path = Path(candidate)
        if candidate_path.is_absolute() and candidate_path.exists():
            return candidate_path
        path = staged_dir / candidate_path
        if path.exists():
            return path
    return None


def _find_namelist(staged_dir: Path, resolved_inputs: Sequence[dict[str, Any]]) -> Path | None:
    namelists = [
        Path(item["resolved_path"])
        for item in resolved_inputs
        if item.get("resolved_path") and Path(str(item["logical_path"])).name == "namelist.input"
    ]
    if namelists:
        return sorted(namelists, key=lambda p: len(p.parts))[0]
    direct = staged_dir / "namelist.input"
    if direct.exists():
        return direct
    recursive = sorted(staged_dir.rglob("namelist.input"), key=lambda p: (len(p.relative_to(staged_dir).parts), str(p)))
    return recursive[0] if recursive else None


def _domain_payload(domain: ExpectedDomain) -> dict[str, Any]:
    return {
        "index": domain.index,
        "id": domain.domain_id,
        "e_we": domain.e_we,
        "e_sn": domain.e_sn,
        "parent_grid_ratio": domain.parent_grid_ratio,
    }


def _validate_domain_rules(domains: Sequence[ExpectedDomain], issues: list[dict[str, Any]]) -> None:
    for domain in domains:
        if domain.e_we <= 0 or domain.e_sn <= 0:
            issues.append(
                _issue(
                    "error",
                    "invalid_domain_dimension",
                    f"{domain.domain_id}: e_we/e_sn must be positive, got {domain.e_we}x{domain.e_sn}",
                )
            )
        if domain.e_we == 181 or domain.e_sn == 181:
            issues.append(
                _issue(
                    "error",
                    "old_mini_nest_rejected",
                    f"{domain.domain_id}: old 181 mini-nest is explicitly rejected",
                )
            )
        ratio = domain.parent_grid_ratio
        if domain.index > 1:
            if ratio is None:
                issues.append(
                    _issue("error", "missing_parent_grid_ratio", f"{domain.domain_id}: parent_grid_ratio is required")
                )
                continue
            if ratio <= 1:
                issues.append(
                    _issue(
                        "error",
                        "invalid_parent_grid_ratio",
                        f"{domain.domain_id}: nested parent_grid_ratio must be >1, got {ratio}",
                    )
                )
                continue
            if (domain.e_we - 1) % ratio != 0:
                issues.append(
                    _issue(
                        "error",
                        "invalid_nested_e_we",
                        f"{domain.domain_id}: (e_we-1) must be divisible by parent_grid_ratio "
                        f"({domain.e_we}-1)%{ratio} != 0",
                    )
                )
            if (domain.e_sn - 1) % ratio != 0:
                issues.append(
                    _issue(
                        "error",
                        "invalid_nested_e_sn",
                        f"{domain.domain_id}: (e_sn-1) must be divisible by parent_grid_ratio "
                        f"({domain.e_sn}-1)%{ratio} != 0",
                    )
                )


def _validate_namelist_dimensions(
    namelist_path: Path | None,
    domains: Sequence[ExpectedDomain],
    issues: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not domains:
        issues.append(_issue("error", "missing_expected_domains", "manifest does not declare expected WRF domains"))
        return None
    if namelist_path is None:
        issues.append(_issue("error", "missing_namelist", "staged inputs do not contain namelist.input"))
        return None
    try:
        parsed = parse_wrf_namelist(namelist_path)
        domain_section = parsed.get("domains", {})
        e_we_values = _to_int_list(domain_section.get("e_we"), "&domains.e_we")
        e_sn_values = _to_int_list(domain_section.get("e_sn"), "&domains.e_sn")
        ratio_values = _to_int_list(domain_section.get("parent_grid_ratio"), "&domains.parent_grid_ratio")
        max_dom_values = _to_int_list(domain_section.get("max_dom"), "&domains.max_dom")
    except Exception as exc:  # noqa: BLE001 - report as validation failure.
        issues.append(_issue("error", "namelist_parse_failed", f"failed to parse {namelist_path}: {exc}"))
        return {"path": str(namelist_path)}

    if max_dom_values and max_dom_values[0] < len(domains):
        issues.append(
            _issue(
                "error",
                "max_dom_too_small",
                f"namelist max_dom={max_dom_values[0]} but manifest declares {len(domains)} domains",
            )
        )
    for domain in domains:
        offset = domain.index - 1
        if offset >= len(e_we_values) or offset >= len(e_sn_values):
            issues.append(_issue("error", "namelist_domain_missing", f"namelist lacks dimensions for {domain.domain_id}"))
            continue
        if e_we_values[offset] != domain.e_we:
            issues.append(
                _issue(
                    "error",
                    "namelist_e_we_mismatch",
                    f"{domain.domain_id}: namelist e_we={e_we_values[offset]} but manifest expects {domain.e_we}",
                )
            )
        if e_sn_values[offset] != domain.e_sn:
            issues.append(
                _issue(
                    "error",
                    "namelist_e_sn_mismatch",
                    f"{domain.domain_id}: namelist e_sn={e_sn_values[offset]} but manifest expects {domain.e_sn}",
                )
            )
        if domain.parent_grid_ratio is not None:
            actual_ratio = ratio_values[offset] if offset < len(ratio_values) else None
            if actual_ratio != domain.parent_grid_ratio:
                issues.append(
                    _issue(
                        "error",
                        "namelist_parent_grid_ratio_mismatch",
                        f"{domain.domain_id}: namelist parent_grid_ratio={actual_ratio} "
                        f"but manifest expects {domain.parent_grid_ratio}",
                    )
                )
    return {
        "path": str(namelist_path),
        "max_dom": max_dom_values[0] if max_dom_values else None,
        "e_we": e_we_values,
        "e_sn": e_sn_values,
        "parent_grid_ratio": ratio_values,
    }


def validate_b200_manifest(
    manifest_path: Path,
    staged_input_dir: Path,
    *,
    require_checksums: bool = True,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    staged_input_dir = staged_input_dir.resolve()
    issues: list[dict[str, Any]] = []
    required = collect_required_inputs(manifest)
    if not required:
        issues.append(_issue("error", "missing_required_inputs", "manifest has no required input file list"))

    resolved_inputs: list[dict[str, Any]] = []
    for spec in required:
        resolved = _resolve_candidate(staged_input_dir, spec)
        item: dict[str, Any] = {
            "logical_path": spec.logical_path,
            "candidates": list(spec.candidates),
            "source": spec.source,
            "expected_sha256": spec.sha256,
            "expected_size_bytes": spec.size_bytes,
            "resolved_path": str(resolved) if resolved else None,
        }
        if resolved is None:
            issues.append(
                _issue(
                    "error",
                    "required_input_missing",
                    f"required input missing: {spec.logical_path}",
                    candidates=list(spec.candidates),
                )
            )
            resolved_inputs.append(item)
            continue
        item["size_bytes"] = resolved.stat().st_size
        if spec.size_bytes is not None and resolved.stat().st_size != spec.size_bytes:
            issues.append(
                _issue(
                    "error",
                    "required_input_size_mismatch",
                    f"{spec.logical_path}: size {resolved.stat().st_size} != manifest {spec.size_bytes}",
                )
            )
        if not spec.sha256:
            if require_checksums:
                issues.append(
                    _issue(
                        "error",
                        "required_input_missing_sha256",
                        f"{spec.logical_path}: required input has no sha256/checksum_sha256",
                    )
                )
        else:
            actual = sha256_file(resolved)
            item["sha256"] = actual
            if actual.lower() != spec.sha256.lower():
                issues.append(
                    _issue(
                        "error",
                        "required_input_sha256_mismatch",
                        f"{spec.logical_path}: sha256 {actual} != manifest {spec.sha256}",
                    )
                )
        resolved_inputs.append(item)

    domains = extract_expected_domains(manifest)
    _validate_domain_rules(domains, issues)
    namelist = _find_namelist(staged_input_dir, resolved_inputs)
    namelist_payload = _validate_namelist_dimensions(namelist, domains, issues)

    status = "PASS" if not any(item["severity"] == "error" for item in issues) else "FAIL"
    return {
        "schema": "gpuwrf.b200_manifest_validation.v1",
        "status": status,
        "ok": status == "PASS",
        "manifest_path": str(manifest_path),
        "staged_input_dir": str(staged_input_dir),
        "validated_utc": utc_now(),
        "require_checksums": require_checksums,
        "required_inputs": resolved_inputs,
        "domains": [_domain_payload(domain) for domain in domains],
        "namelist": namelist_payload,
        "issues": issues,
    }


def manifest_human_report(report: dict[str, Any]) -> str:
    domain_text = ", ".join(f"{d['id']}={d['e_we']}x{d['e_sn']}" for d in report["domains"]) or "NONE"
    lines = [
        f"B200 manifest validation: {report['status']}",
        f"manifest: {report['manifest_path']}",
        f"staged_input_dir: {report['staged_input_dir']}",
        f"required_inputs: {sum(1 for item in report['required_inputs'] if item.get('resolved_path'))}/"
        f"{len(report['required_inputs'])} present",
        f"domains: {domain_text}",
    ]
    if report.get("namelist"):
        lines.append(f"namelist: {report['namelist'].get('path')}")
    if report["issues"]:
        lines.append("issues:")
        lines.extend(f"- {item['severity'].upper()} {item['code']}: {item['message']}" for item in report["issues"])
    return "\n".join(lines)


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            total += child.stat().st_size
    return total


def _is_s3_target(target: str) -> bool:
    return target.startswith("s3://")


def _target_is_local(target: str) -> bool:
    return target.startswith("file://") or ("://" not in target and Path(target).is_absolute())


def _validate_target(target: str) -> dict[str, Any] | None:
    if _target_is_local(target) or _is_s3_target(target):
        return None
    if "://" not in target:
        return _issue(
            "error",
            "ambiguous_target",
            f"target {target!r} is ambiguous; use an absolute local path, file://, or s3:// URI",
        )
    return _issue("error", "unsupported_target", f"unsupported target URI: {target}")


def _target_base_path(target: str) -> Path:
    return Path(target[7:] if target.startswith("file://") else target)


def _safe_block_id(relpath: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", relpath.as_posix())


def _is_partial_path(path: Path) -> bool:
    return any(path.name.endswith(suffix) for suffix in PARTIAL_SUFFIXES)


def discover_completed_blocks(
    output_dir: Path,
    *,
    patterns: Sequence[str] = DEFAULT_DRAIN_PATTERNS,
    state_dir: Path | None = None,
    min_age_seconds: float = 30.0,
) -> list[Path]:
    output_dir = output_dir.resolve()
    state_dir_resolved = state_dir.resolve() if state_dir else None
    now = time.time()
    blocks: dict[Path, None] = {}
    for pattern in patterns:
        for path in output_dir.rglob(pattern):
            if not path.is_file():
                continue
            resolved = path.resolve()
            if state_dir_resolved and (resolved == state_dir_resolved or state_dir_resolved in resolved.parents):
                continue
            if _is_partial_path(path):
                continue
            if path.name.endswith((".manifest.json", ".done.json")):
                continue
            if (path.with_suffix(path.suffix + ".done")).exists() or min_age_seconds <= 0:
                blocks[path] = None
                continue
            if now - path.stat().st_mtime >= min_age_seconds:
                blocks[path] = None
    return sorted(blocks, key=lambda p: p.relative_to(output_dir).as_posix())


def verify_netcdf_block(
    path: Path,
    *,
    expected_vars: Sequence[str],
    expected_dims: dict[str, int],
    expected_time_steps: int | None = None,
    min_time_steps: int = 1,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "dimensions": {},
        "variables": {},
        "issues": issues,
    }
    try:
        with Dataset(path) as ds:
            dimensions = {name: int(len(dim)) for name, dim in ds.dimensions.items()}
            payload["dimensions"] = dimensions
            time_count = dimensions.get("Time")
            if time_count is None:
                issues.append(_issue("error", "time_dimension_missing", f"{path.name}: missing Time dimension"))
            else:
                payload["time_count"] = time_count
                if time_count < min_time_steps:
                    issues.append(
                        _issue(
                            "error",
                            "time_dimension_short",
                            f"{path.name}: Time dimension has {time_count} records, minimum is {min_time_steps}",
                        )
                    )
                if expected_time_steps is not None and time_count != expected_time_steps:
                    issues.append(
                        _issue(
                            "error",
                            "time_dimension_mismatch",
                            f"{path.name}: Time dimension has {time_count} records, expected {expected_time_steps}",
                        )
                    )
            for dim_name, expected in expected_dims.items():
                actual = dimensions.get(dim_name)
                if actual != expected:
                    issues.append(
                        _issue(
                            "error",
                            "dimension_mismatch",
                            f"{path.name}: dimension {dim_name}={actual} but expected {expected}",
                        )
                    )
            for var_name in expected_vars:
                if var_name not in ds.variables:
                    issues.append(_issue("error", "expected_var_missing", f"{path.name}: missing variable {var_name}"))
                    continue
                var = ds.variables[var_name]
                payload["variables"][var_name] = {
                    "dimensions": list(var.dimensions),
                    "shape": [int(v) for v in var.shape],
                    "dtype": str(var.dtype),
                }
                values = np.ma.asarray(var[:])
                if np.issubdtype(values.dtype, np.number):
                    if bool(np.ma.getmaskarray(values).any()):
                        issues.append(_issue("error", "masked_values", f"{path.name}:{var_name} contains masked values"))
                    arr = np.asarray(values.filled(np.nan) if np.ma.isMaskedArray(values) else values)
                    if arr.size == 0:
                        issues.append(_issue("error", "empty_values", f"{path.name}:{var_name} has zero values"))
                    if not bool(np.isfinite(arr).all()):
                        issues.append(
                            _issue("error", "nonfinite_values", f"{path.name}:{var_name} contains NaN or Inf")
                        )
    except Exception as exc:  # noqa: BLE001 - corrupt NetCDF is a block validation failure.
        issues.append(_issue("error", "block_open_failed", f"{path.name}: failed to read NetCDF block: {exc}"))
    payload["ok"] = not any(item["severity"] == "error" for item in issues)
    return payload


def _done_paths(state_dir: Path, block_id: str) -> tuple[Path, Path]:
    block_dir = state_dir / "blocks"
    return block_dir / f"{block_id}.manifest.json", block_dir / f"{block_id}.done.json"


def _target_uri_for(target: str, relpath: Path) -> str:
    if target.endswith("/"):
        return f"{target}{relpath.as_posix()}"
    if _target_is_local(target):
        return str(_target_base_path(target) / relpath)
    return f"{target.rstrip('/')}/{relpath.as_posix()}"


def _local_target_path(target: str, relpath: Path) -> Path:
    return _target_base_path(target) / relpath


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 URI: {uri}")
    rest = uri[5:]
    if "/" not in rest:
        raise ValueError(f"s3 URI lacks key: {uri}")
    bucket, key = rest.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"invalid s3 URI: {uri}")
    return bucket, key


def _s3_head_object(uri: str) -> dict[str, Any]:
    bucket, key = _parse_s3_uri(uri)
    proc = subprocess.run(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key, "--output", "json"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    data = json.loads(proc.stdout or "{}")
    if not isinstance(data, dict):
        raise ValueError(f"head-object returned non-object JSON for {uri}")
    return data


def _validate_s3_target_object(uri: str, expected_bytes: int | None, expected_sha256: str | None) -> tuple[bool, str | None]:
    try:
        head = _s3_head_object(uri)
    except Exception as exc:  # noqa: BLE001
        return False, f"S3 target validation failed for {uri}: {exc}"
    actual_size = head.get("ContentLength")
    if expected_bytes is not None and int(actual_size) != int(expected_bytes):
        return False, f"S3 target size mismatch for {uri}: {actual_size} != {expected_bytes}"
    metadata = head.get("Metadata") if isinstance(head.get("Metadata"), dict) else {}
    actual_sha = metadata.get("sha256") or metadata.get("x-amz-meta-sha256")
    if expected_sha256 and actual_sha != expected_sha256:
        return False, f"S3 target sha256 metadata mismatch for {uri}: {actual_sha!r} != {expected_sha256!r}"
    return True, None


def _done_marker_valid(done_path: Path, target: str, relpath: Path) -> tuple[bool, str | None]:
    try:
        done = load_json(done_path)
    except Exception as exc:  # noqa: BLE001
        return False, f"could not read done marker {done_path}: {exc}"
    if done.get("state") != "done":
        return False, f"done marker state is {done.get('state')!r}"
    if _target_is_local(target):
        target_path = _local_target_path(target, relpath)
        if not target_path.exists():
            return False, f"done marker exists but target is missing: {target_path}"
        expected_sha = done.get("sha256")
        if expected_sha and sha256_file(target_path) != expected_sha:
            return False, f"done marker sha256 does not match target: {target_path}"
    elif _is_s3_target(target):
        return _validate_s3_target_object(
            _target_uri_for(target, relpath),
            int(done["bytes"]) if done.get("bytes") is not None else None,
            str(done["sha256"]) if done.get("sha256") else None,
        )
    else:
        return False, f"unsupported target URI: {target}"
    return True, None


def _copy_to_target(
    src: Path,
    target: str,
    relpath: Path,
    *,
    dry_run: bool,
    expected_sha256: str,
    expected_bytes: int,
) -> str:
    uri = _target_uri_for(target, relpath)
    if dry_run:
        return uri
    if _target_is_local(target):
        dst = _local_target_path(target, relpath)
        dst.parent.mkdir(parents=True, exist_ok=True)
        partial = dst.with_name(f"{dst.name}.partial")
        shutil.copy2(src, partial)
        os.replace(partial, dst)
        return str(dst)
    if target.startswith("s3://"):
        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                str(src),
                uri,
                "--no-progress",
                "--metadata",
                f"sha256={expected_sha256},bytes={expected_bytes}",
            ],
            check=True,
        )
        return uri
    raise ValueError(f"unsupported target URI: {target}")


def _check_target_capacity(
    target: str,
    pending_bytes: int,
    config: DrainConfig,
    current_target_bytes: int,
) -> dict[str, Any] | None:
    if _is_s3_target(target):
        if config.target_cap_bytes is not None and current_target_bytes + pending_bytes > config.target_cap_bytes:
            return _issue(
                "error",
                "target_backpressure",
                f"S3 target byte budget would be exceeded: current {current_target_bytes} + pending "
                f"{pending_bytes} > {config.target_cap_bytes}",
            )
        return None
    if not _target_is_local(target):
        return _issue("error", "unsupported_target", f"unsupported target URI: {target}")
    base = _target_base_path(target)
    base.mkdir(parents=True, exist_ok=True)
    if config.target_cap_bytes is not None and current_target_bytes + pending_bytes > config.target_cap_bytes:
        return _issue(
            "error",
            "target_backpressure",
            f"target cap would be exceeded: current {current_target_bytes} + pending {pending_bytes} > "
            f"{config.target_cap_bytes}",
        )
    free = shutil.disk_usage(base).free
    if free - pending_bytes < config.target_min_free_bytes:
        return _issue(
            "error",
            "target_free_space_low",
            f"target free space would fall below minimum: free {free}, pending {pending_bytes}, "
            f"minimum {config.target_min_free_bytes}",
        )
    return None


def _load_valid_done_records(state_dir: Path, target: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for done_path in sorted((state_dir / "blocks").glob("*.done.json")):
        try:
            done = load_json(done_path)
            relpath_raw = done.get("relative_path")
            if not relpath_raw:
                continue
            valid, _why = _done_marker_valid(done_path, target, Path(str(relpath_raw)))
            if valid:
                records.append(done)
        except Exception:  # noqa: BLE001 - bad markers are reported during block discovery.
            continue
    return records


def _validate_existing_done_records(
    state_dir: Path,
    target: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for done_path in sorted((state_dir / "blocks").glob("*.done.json")):
        try:
            done = load_json(done_path)
            relpath_raw = done.get("relative_path")
            if not relpath_raw:
                issues.append(_issue("error", "invalid_done_marker", f"{done_path}: missing relative_path"))
                continue
            relpath = Path(str(relpath_raw))
            valid, why = _done_marker_valid(done_path, target, relpath)
            if valid:
                records.append(done)
                continue
            source = Path(str(done.get("source_path", "")))
            code = "done_marker_invalid" if source.exists() else "drained_block_lost"
            issues.append(_issue("error", code, str(why), block_id=done.get("block_id")))
        except Exception as exc:  # noqa: BLE001
            issues.append(_issue("error", "invalid_done_marker", f"{done_path}: {exc}"))
    return records, issues


def _training_ready_blocks(
    done_records: Sequence[dict[str, Any]],
    policy: str,
    expected_block_count: int | None,
) -> tuple[list[str], str]:
    ids = [str(item["block_id"]) for item in done_records if item.get("state") == "done"]
    if policy == "allow-partial":
        return ids, "allow-partial: every verified drained block is accepted as training data"
    if policy != "contiguous":
        return [], f"unknown policy {policy!r}"
    if expected_block_count is None:
        return [], "contiguous: no blocks marked training-ready without --expected-block-count"
    if len(ids) < expected_block_count:
        return [], f"contiguous: {len(ids)}/{expected_block_count} verified blocks present"
    return sorted(ids)[:expected_block_count], f"contiguous: {expected_block_count} verified blocks present"


def drain_once(config: DrainConfig) -> dict[str, Any]:
    output_dir = config.output_dir.resolve()
    state_dir = config.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    issues: list[dict[str, Any]] = []
    target_issue = _validate_target(config.target)
    if target_issue is not None:
        issues.append(target_issue)
        report = {
            "schema": "gpuwrf.b200_drain.v1",
            "status": "FAIL",
            "ok": False,
            "drained_utc": utc_now(),
            "output_dir": str(output_dir),
            "target": config.target,
            "state_dir": str(state_dir),
            "dry_run": config.dry_run,
            "delete_after_copy": config.delete_after_copy,
            "training_policy": config.training_policy,
            "expected_block_count": config.expected_block_count,
            "training_ready_blocks": [],
            "training_ready_reason": "target validation failed",
            "local_usage_bytes": dir_size_bytes(output_dir),
            "issues": issues,
            "blocks": [],
        }
        write_json_atomic(state_dir / "drain_summary.json", report)
        return report

    local_usage = dir_size_bytes(output_dir)
    existing_done_records, existing_done_issues = _validate_existing_done_records(state_dir, config.target)
    issues.extend(existing_done_issues)
    existing_done_by_block = {str(item.get("block_id")): item for item in existing_done_records}
    invalid_done_blocks = {str(item.get("block_id")) for item in existing_done_issues if item.get("block_id")}
    if _is_s3_target(config.target):
        target_current_bytes = sum(int(item.get("bytes", 0)) for item in existing_done_records)
    else:
        target_current_bytes = dir_size_bytes(_target_base_path(config.target))

    blocks = discover_completed_blocks(
        output_dir,
        patterns=config.patterns,
        state_dir=state_dir,
        min_age_seconds=config.min_age_seconds,
    )
    block_reports: list[dict[str, Any]] = []
    done_records: list[dict[str, Any]] = []
    expected_dims = config.expected_dims or {}
    for block_path in blocks:
        relpath = block_path.relative_to(output_dir)
        block_id = _safe_block_id(relpath)
        manifest_path, done_path = _done_paths(state_dir, block_id)
        if done_path.exists():
            done = existing_done_by_block.get(block_id)
            if done is not None:
                block_reports.append({"block_id": block_id, "path": str(block_path), "state": "skipped_done"})
                done_records.append(done)
                continue
            if block_id not in invalid_done_blocks:
                valid, why = _done_marker_valid(done_path, config.target, relpath)
                if valid:
                    done = load_json(done_path)
                    block_reports.append({"block_id": block_id, "path": str(block_path), "state": "skipped_done"})
                    done_records.append(done)
                    continue
                issues.append(_issue("error", "done_marker_invalid", str(why), block_id=block_id))
            block_reports.append({"block_id": block_id, "path": str(block_path), "state": "done_marker_invalid"})
            continue

        verification = verify_netcdf_block(
            block_path,
            expected_vars=config.expected_vars,
            expected_dims=expected_dims,
            expected_time_steps=config.expected_time_steps,
            min_time_steps=config.min_time_steps,
        )
        manifest_payload = {
            "schema": "gpuwrf.b200_block_manifest.v1",
            "block_id": block_id,
            "source_path": str(block_path),
            "relative_path": relpath.as_posix(),
            "verified_utc": utc_now(),
            "verification": verification,
            "state": "verified" if verification["ok"] else "failed",
        }
        write_json_atomic(manifest_path, manifest_payload)
        if not verification["ok"]:
            issues.extend(verification["issues"])
            block_reports.append({"block_id": block_id, "path": str(block_path), "state": "verify_failed"})
            continue

        block_bytes = block_path.stat().st_size
        capacity_issue = _check_target_capacity(config.target, block_bytes, config, target_current_bytes)
        if capacity_issue is not None:
            issues.append(capacity_issue)
            block_reports.append({"block_id": block_id, "path": str(block_path), "state": "target_backpressure"})
            continue

        try:
            target_uri = _copy_to_target(
                block_path,
                config.target,
                relpath,
                dry_run=config.dry_run,
                expected_sha256=str(verification["sha256"]),
                expected_bytes=int(verification["bytes"]),
            )
        except Exception as exc:  # noqa: BLE001
            issues.append(_issue("error", "copy_failed", f"{block_path}: failed to copy/upload: {exc}"))
            block_reports.append({"block_id": block_id, "path": str(block_path), "state": "copy_failed"})
            continue
        target_current_bytes += block_bytes

        done_payload = {
            "schema": "gpuwrf.b200_block_done.v1",
            "state": "done" if not config.dry_run else "dry_run",
            "block_id": block_id,
            "source_path": str(block_path),
            "relative_path": relpath.as_posix(),
            "target_uri": target_uri,
            "sha256": verification["sha256"],
            "bytes": verification["bytes"],
            "completed_utc": utc_now(),
            "training_policy": config.training_policy,
        }
        if not config.dry_run:
            write_json_atomic(done_path, done_payload)
            if config.delete_after_copy:
                target_valid, why = _done_marker_valid(done_path, config.target, relpath)
                if not target_valid:
                    issues.append(_issue("error", "delete_blocked", f"{block_path}: {why}", block_id=block_id))
                else:
                    block_path.unlink()
                    done_payload["source_deleted"] = True
                    write_json_atomic(done_path, done_payload)
        block_reports.append(
            {
                "block_id": block_id,
                "path": str(block_path),
                "target_uri": target_uri,
                "state": "dry_run" if config.dry_run else "done",
            }
        )
        if done_payload["state"] == "done":
            done_records.append(done_payload)

    status = "PASS" if not any(item["severity"] == "error" for item in issues) else "FAIL"
    final_usage = dir_size_bytes(output_dir)
    if config.local_cap_bytes is not None and final_usage > config.local_cap_bytes:
        issues.append(
            _issue(
                "error",
                "local_backpressure_remaining",
                f"output dir still uses {final_usage} bytes, above cap {config.local_cap_bytes}",
            )
        )
        status = "BACKPRESSURE"
    if config.dry_run:
        all_done_records = done_records
    else:
        all_done_records, final_done_issues = _validate_existing_done_records(state_dir, config.target)
        seen_issues = {(item.get("code"), item.get("block_id"), item.get("message")) for item in issues}
        for item in final_done_issues:
            key = (item.get("code"), item.get("block_id"), item.get("message"))
            if key not in seen_issues:
                issues.append(item)
        if any(item["severity"] == "error" for item in final_done_issues) and status == "PASS":
            status = "FAIL"
    training_ready, training_reason = _training_ready_blocks(
        all_done_records,
        config.training_policy,
        config.expected_block_count,
    )
    report = {
        "schema": "gpuwrf.b200_drain.v1",
        "status": status,
        "ok": status == "PASS",
        "drained_utc": utc_now(),
        "output_dir": str(output_dir),
        "target": config.target,
        "state_dir": str(state_dir),
        "dry_run": config.dry_run,
        "delete_after_copy": config.delete_after_copy,
        "training_policy": config.training_policy,
        "expected_block_count": config.expected_block_count,
        "training_ready_blocks": training_ready,
        "training_ready_reason": training_reason,
        "local_usage_bytes": final_usage,
        "issues": issues,
        "blocks": block_reports,
    }
    write_json_atomic(state_dir / "drain_summary.json", report)
    return report


def stop_pull(
    config: DrainConfig,
    *,
    pid_file: Path | None = None,
    stop_command: Sequence[str] | None = None,
    stop_marker: Path | None = None,
    grace_seconds: float = 10.0,
    force_kill_after_grace: bool = False,
) -> dict[str, Any]:
    output_dir = config.output_dir.resolve()
    marker = stop_marker or (output_dir / "B200_STOP_REQUESTED.json")
    stop_payload: dict[str, Any] = {
        "schema": "gpuwrf.b200_stop_request.v1",
        "requested_utc": utc_now(),
        "output_dir": str(output_dir),
        "pid_file": str(pid_file) if pid_file else None,
        "stop_command": list(stop_command) if stop_command else None,
        "grace_seconds": grace_seconds,
    }
    write_json_atomic(marker, stop_payload)

    stop_events: list[dict[str, Any]] = [{"event": "stop_marker_written", "path": str(marker)}]
    if stop_command:
        proc = subprocess.run(list(stop_command), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stop_events.append(
            {
                "event": "stop_command",
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        )
        if proc.returncode != 0:
            report = drain_once(config)
            report["stop_events"] = stop_events
            report["status"] = "FAIL"
            report["ok"] = False
            report.setdefault("issues", []).append(
                _issue("error", "stop_command_failed", f"stop command exited {proc.returncode}")
            )
            return report
    if pid_file and pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
            os.kill(pid, signal.SIGTERM)
            stop_events.append({"event": "sigterm_sent", "pid": pid})
            deadline = time.time() + grace_seconds
            while time.time() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    stop_events.append({"event": "process_exited", "pid": pid})
                    break
                time.sleep(0.1)
            else:
                stop_events.append({"event": "grace_elapsed", "pid": pid})
                if force_kill_after_grace:
                    os.kill(pid, signal.SIGKILL)
                    stop_events.append({"event": "sigkill_sent", "pid": pid})
        except Exception as exc:  # noqa: BLE001
            stop_events.append({"event": "pid_stop_failed", "error": str(exc)})

    report = drain_once(config)
    report["stop_events"] = stop_events
    report["stop_marker"] = str(marker)
    write_json_atomic(config.state_dir / "stop_pull_summary.json", report)
    return report


def drain_human_report(report: dict[str, Any]) -> str:
    states: dict[str, int] = {}
    for block in report.get("blocks", []):
        states[block.get("state", "unknown")] = states.get(block.get("state", "unknown"), 0) + 1
    lines = [
        f"B200 drain: {report['status']}",
        f"output_dir: {report['output_dir']}",
        f"target: {report['target']}",
        f"state_dir: {report['state_dir']}",
        f"blocks: {states or {}}",
        f"training_policy: {report.get('training_policy')}",
        f"training_ready_blocks: {len(report.get('training_ready_blocks', []))}",
    ]
    if report.get("issues"):
        lines.append("issues:")
        lines.extend(f"- {item['severity'].upper()} {item['code']}: {item['message']}" for item in report["issues"])
    return "\n".join(lines)


def parse_expected_dims(values: Sequence[str], expected_domain: str | None = None) -> dict[str, int]:
    dims: dict[str, int] = {}
    if expected_domain:
        match = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", expected_domain)
        if not match:
            raise argparse.ArgumentTypeError("--expected-domain must look like 898x898")
        dims["west_east"] = int(match.group(1))
        dims["south_north"] = int(match.group(2))
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected dimension must be name=size, got {item!r}")
        name, raw_size = item.split("=", 1)
        dims[name.strip()] = int(raw_size)
    return dims


def parse_bytes(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmgt]?i?b?)?", text)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid byte value: {value}")
    number = float(match.group(1))
    suffix = match.group(2) or ""
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "m": 1000**2,
        "mb": 1000**2,
        "g": 1000**3,
        "gb": 1000**3,
        "t": 1000**4,
        "tb": 1000**4,
        "ki": 1024,
        "kib": 1024,
        "mi": 1024**2,
        "mib": 1024**2,
        "gi": 1024**3,
        "gib": 1024**3,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    return int(number * multipliers[suffix])


def _make_synthetic_netcdf(path: Path, *, finite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as ds:
        ds.createDimension("Time", 1)
        ds.createDimension("south_north", 3)
        ds.createDimension("west_east", 4)
        t2 = ds.createVariable("T2", "f4", ("Time", "south_north", "west_east"))
        u10 = ds.createVariable("U10", "f4", ("Time", "south_north", "west_east"))
        values = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        if not finite:
            values[0, 1, 2] = math.nan
        t2[:] = values
        u10[:] = values + 1.0


def _write_synthetic_inputs(base: Path) -> tuple[Path, Path]:
    staged = base / "staged"
    staged.mkdir(parents=True)
    namelist = staged / "namelist.input"
    namelist.write_text(
        """&time_control
/
&domains
 max_dom = 2,
 e_we = 369, 898,
 e_sn = 369, 898,
 parent_grid_ratio = 1, 3,
/
""",
        encoding="utf-8",
    )
    for name in ("wrfinput_d01", "wrfinput_d02", "wrfbdy_d01"):
        (staged / name).write_bytes(f"synthetic {name}\n".encode("ascii"))
    required = []
    for path in sorted(staged.iterdir()):
        if path.is_file():
            required.append({"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = base / "b200_manifest.synthetic.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "gpuwrf.b200.synthetic.v1",
                "required_inputs": required,
                "wrf_domains": [
                    {"id": "d01", "e_we": 369, "e_sn": 369, "parent_grid_ratio": 1},
                    {"id": "d02", "e_we": 898, "e_sn": 898, "parent_grid_ratio": 3},
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest, staged


def synthetic_dry_run(work_dir: Path | None = None) -> dict[str, Any]:
    owns_tmp = work_dir is None
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="b200_io_dryrun_"))
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, staged_dir = _write_synthetic_inputs(work_dir)
    manifest_report = validate_b200_manifest(manifest_path, staged_dir)

    output_dir = work_dir / "running_output"
    target_dir = work_dir / "drained_target"
    state_dir = work_dir / "drain_state"
    _make_synthetic_netcdf(output_dir / "wrfout_d02_2026-06-23_01:00:00.nc")
    config = DrainConfig(
        output_dir=output_dir,
        target=str(target_dir),
        state_dir=state_dir,
        expected_vars=("T2", "U10"),
        expected_dims={"south_north": 3, "west_east": 4},
        min_age_seconds=0,
        local_cap_bytes=10 * 1024 * 1024,
        target_cap_bytes=10 * 1024 * 1024,
        delete_after_copy=True,
        training_policy="allow-partial",
    )
    first_drain = drain_once(config)
    _make_synthetic_netcdf(output_dir / "wrfout_d02_2026-06-23_02:00:00.nc")
    stop_pull_report = stop_pull(config, grace_seconds=0)
    resume_report = drain_once(config)
    report = {
        "schema": "gpuwrf.b200_synthetic_dry_run.v1",
        "status": "PASS"
        if manifest_report["ok"] and first_drain["ok"] and stop_pull_report["ok"] and resume_report["ok"]
        else "FAIL",
        "ok": manifest_report["ok"] and first_drain["ok"] and stop_pull_report["ok"] and resume_report["ok"],
        "work_dir": str(work_dir),
        "owns_tmp": owns_tmp,
        "manifest_validation": manifest_report,
        "first_drain": first_drain,
        "stop_pull": stop_pull_report,
        "resume": resume_report,
    }
    write_json_atomic(work_dir / "synthetic_dry_run_report.json", report)
    return report
