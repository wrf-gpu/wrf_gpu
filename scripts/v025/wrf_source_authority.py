#!/usr/bin/env python3
"""CPU-only WRF source authority for the exact FAST-v025 boundary.

The resolution order, checkout-shadow refusal, six-file inventory, and
mutation model are reimplemented from the useful subset of rejected candidate
``a5e8f5d8``.  Evidence strings intentionally claim only file binding and
reachability; actual loader execution is proved separately by the R1 boundary
child.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SCHEMA = "wrf_gpu2.v025.m0.wrf_source_authority.v2"
ENV_VAR = "GPUWRF_WRF_ROOT"
SRC_ENV_VAR = "GPUWRF_WRF_SRC"
ROOT_ENV_VARS = (ENV_VAR, SRC_ENV_VAR)
AUTHORITY_ENV_VAR = "GPUWRF_M0_WRF_AUTHORITY"
CANONICAL_ROOT = Path("<USER_HOME>/src/wrf_pristine/WRF")
EXTERNAL_FALLBACK_REL = "external/WRF"
CHECKOUT_DEFAULT_REL = "data/wrf_pristine/WRF"
FILE_EVIDENCE = "FILE_BINDING_AND_ACTIVE_LOADER_REACHABILITY_ONLY"


class SourceAuthorityRefusal(RuntimeError):
    """The exact WRF source tree is absent, ambiguous, or changed."""


@dataclass(frozen=True)
class SchemeSource:
    relpath: str
    namelist_key: str
    code: int
    loader: str
    evidence: str = FILE_EVIDENCE


SCHEME_SOURCES = (
    SchemeSource(
        "phys/module_ra_rrtmg_lw.F",
        "ra_lw_physics",
        4,
        "gpuwrf.physics.rrtmg_lw._native_lw_tables",
    ),
    SchemeSource(
        "phys/module_ra_rrtmg_sw.F",
        "ra_sw_physics",
        4,
        "scripts.extract_rrtmg_tables SW source declaration",
    ),
    SchemeSource(
        "run/MPTABLE.TBL",
        "sf_surface_physics",
        4,
        "gpuwrf.physics.noahmp.tables.load_noahmp_parameters",
    ),
    SchemeSource(
        "run/SOILPARM.TBL",
        "sf_surface_physics",
        4,
        "gpuwrf.physics.noahmp.tables.load_noahmp_parameters",
    ),
    SchemeSource(
        "run/GENPARM.TBL",
        "sf_surface_physics",
        4,
        "gpuwrf.physics.noahmp.tables.load_noahmp_parameters",
    ),
    SchemeSource(
        "run/CAMtr_volume_mixing_ratio",
        "ra_lw_physics",
        4,
        "gpuwrf.physics.wrf_clwrf_ghg.clwrf_ssp245_gases_for_time",
    ),
    # Kept only to prove derivation is by active code, not a hand-picked list.
    SchemeSource(
        "phys/module_ra_rrtm.F",
        "ra_lw_physics",
        1,
        "gpuwrf.physics.ra_lw_rrtm._load_tables",
    ),
    SchemeSource(
        "run/RRTM_DATA",
        "ra_lw_physics",
        1,
        "gpuwrf.physics.ra_lw_rrtm._load_tables",
    ),
)
DERIVED_KEYS = tuple(dict.fromkeys(item.namelist_key for item in SCHEME_SOURCES))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_tree_identity() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise SourceAuthorityRefusal("cannot resolve repository tree identity")
    return result.stdout.strip()


def active_scheme_codes(namelist_path: Path) -> dict[str, int]:
    import fast_case

    groups = fast_case.parse_namelist(
        Path(namelist_path).read_text(encoding="utf-8")
    )
    physics = groups.get("physics", {})
    result: dict[str, int] = {}
    for key in DERIVED_KEYS:
        if key in physics:
            try:
                result[key] = int(fast_case.first_column(str(physics[key])))
            except (TypeError, ValueError) as exc:
                raise SourceAuthorityRefusal(
                    f"namelist {key} is not an integer code"
                ) from exc
    if not result:
        raise SourceAuthorityRefusal("no authority-relevant scheme code")
    return result


def derive_inventory(namelist_path: Path) -> tuple[SchemeSource, ...]:
    codes = active_scheme_codes(namelist_path)
    selected = tuple(
        item
        for item in SCHEME_SOURCES
        if codes.get(item.namelist_key) == item.code
    )
    if len(selected) != 6:
        raise SourceAuthorityRefusal(
            f"FAST source inventory has {len(selected)} entries, expected 6"
        )
    return selected


def resolve_authority_root(
    *,
    environ: Mapping[str, str] | None = None,
    canonical_root: Path = CANONICAL_ROOT,
) -> Path:
    environ = os.environ if environ is None else environ
    raw = str(environ.get(ENV_VAR, "")).strip()
    if not raw:
        raise SourceAuthorityRefusal(
            f"{ENV_VAR} is unset; checkout-relative fallback is forbidden"
        )
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise SourceAuthorityRefusal(f"{ENV_VAR} must be absolute")
    # The checkout default may be a symlink to the frozen authority. Reject
    # that implicit checkout spelling, while accepting an explicit canonical
    # root even when both paths resolve to the same source directory.
    checkout_default = REPO_ROOT / CHECKOUT_DEFAULT_REL
    if Path(os.path.abspath(candidate)) == Path(os.path.abspath(checkout_default)):
        raise SourceAuthorityRefusal("checkout-relative WRF default is forbidden")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SourceAuthorityRefusal(f"{ENV_VAR} is unavailable: {exc}") from exc
    expected = Path(canonical_root).resolve(strict=False)
    if not resolved.is_dir() or resolved != expected:
        raise SourceAuthorityRefusal(
            f"WRF root {resolved} is not frozen authority {expected}"
        )
    src_raw = str(environ.get(SRC_ENV_VAR, "")).strip()
    if src_raw:
        try:
            src = Path(src_raw).expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise SourceAuthorityRefusal(
                f"{SRC_ENV_VAR} is unavailable: {exc}"
            ) from exc
        if src != resolved:
            raise SourceAuthorityRefusal(
                f"{SRC_ENV_VAR} conflicts with {ENV_VAR}"
            )
    return resolved


def _bind_file(root: Path, source: SchemeSource) -> dict[str, Any]:
    link = root / source.relpath
    try:
        resolved = link.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SourceAuthorityRefusal(
            f"required WRF file {source.relpath} is unavailable: {exc}"
        ) from exc
    if not resolved.is_file():
        raise SourceAuthorityRefusal(
            f"required WRF file {source.relpath} is not regular"
        )
    before = resolved.stat()
    digest = sha256_file(resolved)
    after = resolved.stat()
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if tuple(getattr(before, key) for key in stable) != tuple(
        getattr(after, key) for key in stable
    ):
        raise SourceAuthorityRefusal(f"{source.relpath} changed while hashing")
    return {
        "relpath": source.relpath,
        "is_symlink": link.is_symlink(),
        "resolved_path": str(resolved),
        "device": after.st_dev,
        "inode": after.st_ino,
        "bytes": after.st_size,
        "sha256": digest,
        "namelist_key": source.namelist_key,
        "code": source.code,
        "loader": source.loader,
        "evidence": source.evidence,
    }


def build_source_authority(
    *,
    namelist_path: Path,
    environ: Mapping[str, str] | None = None,
    canonical_root: Path = CANONICAL_ROOT,
) -> dict[str, Any]:
    root = resolve_authority_root(
        environ=environ, canonical_root=canonical_root
    )
    external_lw = (
        REPO_ROOT / EXTERNAL_FALLBACK_REL / "phys/module_ra_rrtmg_lw.F"
    )
    if external_lw.exists():
        external_root = (REPO_ROOT / EXTERNAL_FALLBACK_REL).resolve()
        if external_root != root:
            raise SourceAuthorityRefusal("checkout external/WRF shadows authority")
    inventory = derive_inventory(namelist_path)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS",
        "root": str(root),
        "root_env_vars": list(ROOT_ENV_VARS),
        "namelist_path": str(Path(namelist_path).resolve()),
        "active_scheme_codes": dict(
            sorted(active_scheme_codes(namelist_path).items())
        ),
        "required_relpaths": sorted(item.relpath for item in inventory),
        "files": sorted(
            (_bind_file(root, item) for item in inventory),
            key=lambda item: item["relpath"],
        ),
        "checkout_fallback_identity": {
            "repository_tree_identity": {
                "algorithm": "git-tree-object-id",
                "value": _repo_tree_identity(),
            },
            "relative_paths": [
                CHECKOUT_DEFAULT_REL,
                EXTERNAL_FALLBACK_REL,
            ],
            "external_lw_present": external_lw.exists(),
        },
        "portability": {
            "status": "LIMITATION",
            "text": "authority is pinned to <USER_HOME>/src/wrf_pristine/WRF",
        },
        "device_action": False,
    }
    payload["authority_sha256"] = canonical_sha256(payload)
    return payload


def validate_source_authority(
    payload: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    require_env_match: bool = True,
) -> dict[str, Any]:
    if payload.get("schema") != SCHEMA or payload.get("status") != "PASS":
        raise SourceAuthorityRefusal("source authority schema/status mismatch")
    expected_hash = canonical_sha256(
        {key: value for key, value in payload.items()
         if key != "authority_sha256"}
    )
    if payload.get("authority_sha256") != expected_hash:
        raise SourceAuthorityRefusal("source authority hash mismatch")
    bound_root = str(payload.get("root", ""))
    if require_env_match:
        check_env = os.environ if environ is None else environ
    else:
        ambient = os.environ if environ is None else environ
        for variable in ROOT_ENV_VARS:
            value = str(ambient.get(variable, "")).strip()
            if value and Path(value).resolve(strict=False) != Path(bound_root):
                raise SourceAuthorityRefusal(
                    f"ambient {variable} conflicts with bound authority"
                )
        check_env = {variable: bound_root for variable in ROOT_ENV_VARS}
    observed = build_source_authority(
        namelist_path=Path(str(payload.get("namelist_path", ""))),
        environ=check_env,
        canonical_root=Path(bound_root),
    )
    if observed["authority_sha256"] != payload["authority_sha256"]:
        raise SourceAuthorityRefusal("source authority changed after binding")
    return observed


def child_environment_binding(payload: Mapping[str, Any]) -> dict[str, str]:
    root = str(payload.get("root", "")).strip()
    if not root:
        raise SourceAuthorityRefusal("bound authority has no root")
    return {variable: root for variable in ROOT_ENV_VARS}


def write_authority(path: Path, payload: Mapping[str, Any]) -> Path:
    Path(path).write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return Path(path)


def assert_child_binding(
    *, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    environ = os.environ if environ is None else environ
    forbidden = sorted(
        name
        for name in sys.modules
        if name == "jax"
        or name.startswith("jax.")
        or name == "gpuwrf"
        or name.startswith("gpuwrf.")
    )
    if forbidden:
        raise SourceAuthorityRefusal(
            f"accelerator/product import preceded authority: {forbidden[:4]}"
        )
    authority_path = Path(str(environ.get(AUTHORITY_ENV_VAR, "")))
    if not authority_path.is_file():
        raise SourceAuthorityRefusal("child authority artifact is absent")
    payload = json.loads(authority_path.read_text(encoding="utf-8"))
    root = str(payload.get("root", ""))
    for variable in ROOT_ENV_VARS:
        if str(environ.get(variable, "")) != root:
            raise SourceAuthorityRefusal(
                f"child {variable} differs from bound authority"
            )
    return validate_source_authority(payload, environ=environ)


def apply_default_root(environ: dict[str, str]) -> str:
    """Subordinate a harness default to an explicit bound authority."""

    authority = str(environ.get(AUTHORITY_ENV_VAR, "")).strip()
    chosen = ""
    if authority:
        payload = json.loads(Path(authority).read_text(encoding="utf-8"))
        chosen = str(payload.get("root", ""))
    if not chosen:
        existing = {
            str(environ.get(variable, "")).strip()
            for variable in ROOT_ENV_VARS
            if str(environ.get(variable, "")).strip()
        }
        if len(existing) > 1:
            raise SourceAuthorityRefusal("conflicting ambient WRF roots")
        chosen = existing.pop() if existing else str(CANONICAL_ROOT)
    for variable in ROOT_ENV_VARS:
        environ[variable] = chosen
    return chosen


def resolver_snapshot(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Structural resolver identity without checkout-absolute diagnostics."""

    environ = os.environ if environ is None else environ
    return {
        "root_env": {
            variable: str(environ.get(variable, "")) or None
            for variable in ROOT_ENV_VARS
        },
        "extractor_precedence": [
            SRC_ENV_VAR,
            ENV_VAR,
            {
                "repository_tree_identity": {
                    "algorithm": "git-tree-object-id",
                    "value": _repo_tree_identity(),
                },
                "relative_path": EXTERNAL_FALLBACK_REL,
            },
        ],
        "central_default": {
            "repository_tree_identity": {
                "algorithm": "git-tree-object-id",
                "value": _repo_tree_identity(),
            },
            "relative_path": CHECKOUT_DEFAULT_REL,
        },
    }


__all__ = [
    "AUTHORITY_ENV_VAR",
    "CANONICAL_ROOT",
    "ENV_VAR",
    "FILE_EVIDENCE",
    "ROOT_ENV_VARS",
    "SCHEME_SOURCES",
    "SRC_ENV_VAR",
    "SchemeSource",
    "SourceAuthorityRefusal",
    "active_scheme_codes",
    "apply_default_root",
    "assert_child_binding",
    "build_source_authority",
    "child_environment_binding",
    "derive_inventory",
    "resolve_authority_root",
    "resolver_snapshot",
    "validate_source_authority",
    "write_authority",
]
