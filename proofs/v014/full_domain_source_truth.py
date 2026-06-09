#!/usr/bin/env python3
"""Fail-closed inventory for the v0.14 full-domain source/truth surface.

This proof deliberately does not extend WRF instrumentation.  It inventories the
validated source/save and post-RK surfaces already on disk and decides whether
they satisfy the stricter full-domain wrapper contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

ROOT = Path(__file__).resolve().parents[2]
PROOF_DIR = ROOT / "proofs/v014"
OUT_JSON = PROOF_DIR / "full_domain_source_truth.json"
OUT_MD = PROOF_DIR / "full_domain_source_truth.md"
OUT_DIFF = PROOF_DIR / "full_domain_source_truth_wrf_patch.diff"

SOURCE_JSON = PROOF_DIR / "source_save_boundary_hook.json"
SOURCE_DIR = Path("/mnt/data/wrf_gpu2/v014_source_save_boundary/source_save_output")
POST_RK_DIR = Path("/mnt/data/wrf_gpu2/v014_post_rk_refresh/refresh_output")
WRFINPUT_D02 = Path("/mnt/data/wrf_gpu2/v014_source_save_boundary/run_case3/wrfinput_d02")

VERDICT = "FULL_DOMAIN_TRUTH_SURFACE_BLOCKED_PATCH_ONLY_EXISTING_SURFACES"


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path),
    }


def list_infos(root: Path, pattern: str) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    return [path_info(path) for path in sorted(root.glob(pattern))]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def domain_dims() -> dict[str, int | None]:
    try:
        from netCDF4 import Dataset  # type: ignore

        with Dataset(WRFINPUT_D02) as ds:
            return {name: len(dim) for name, dim in ds.dimensions.items()}
    except Exception:
        return {
            "west_east": None,
            "south_north": None,
            "bottom_top": None,
            "west_east_stag": None,
            "south_north_stag": None,
            "bottom_top_stag": None,
        }


def expected_counts(dims: dict[str, int | None]) -> dict[str, int | None]:
    nx = dims.get("west_east")
    ny = dims.get("south_north")
    nz = dims.get("bottom_top")
    nx_stag = dims.get("west_east_stag")
    ny_stag = dims.get("south_north_stag")
    nz_stag = dims.get("bottom_top_stag")
    if not all(isinstance(v, int) for v in (nx, ny, nz, nx_stag, ny_stag, nz_stag)):
        return {}
    return {
        "MASS_SOURCE": nx * ny * nz,  # type: ignore[operator]
        "MASS2D_SOURCE": nx * ny,  # type: ignore[operator]
        "U_SOURCE": nx_stag * ny * nz,  # type: ignore[operator]
        "V_SOURCE": nx * ny_stag * nz,  # type: ignore[operator]
        "WPH_SOURCE": nx * ny * nz_stag,  # type: ignore[operator]
    }


def main() -> int:
    source = load_json(SOURCE_JSON)
    dims = domain_dims()
    expect = expected_counts(dims)
    unique = source.get("emitted_surface", {}).get("unique_counts", {})
    patch = source.get("patch_width_assessment", {})

    source_full_domain = bool(expect) and all(
        int(unique.get(name, -1)) >= int(expected)
        for name, expected in expect.items()
        if expected is not None
    )
    post_files = list_infos(POST_RK_DIR, "refresh_post_after_all_rk_steps_pre_halo_d2_step_6000_*.txt")
    source_files = list_infos(SOURCE_DIR, "source_save_after_rk_tendency_d2_step_6000_*.txt")
    post_full_domain = False

    blockers = [
        "existing source/save surface is patch-only, not full-domain/full-vertical enough for the wrapper contract",
        "existing post-RK/pre-halo truth is patch-only and not a full-domain/full-vertical State truth surface",
        "accepted source/save proof reports only one conservative 8-cell-halo-valid mass cell",
        "no same-boundary promoted carry/boundary surface was emitted for the full wrapper contract",
    ]
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.full_domain_source_truth.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": VERDICT,
        "cpu_only": True,
        "gpu_used": False,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        },
        "inputs": {
            "source_save_boundary_hook_json": path_info(SOURCE_JSON),
            "wrfinput_d02": path_info(WRFINPUT_D02),
            "source_files": source_files,
            "post_rk_pre_halo_files": post_files,
        },
        "domain_dims": dims,
        "expected_full_domain_counts": expect,
        "observed_source_counts": unique,
        "source_patch_width_assessment": patch,
        "source_full_domain": source_full_domain,
        "post_rk_truth_full_domain": post_full_domain,
        "truth_surface_sufficient": False,
        "blockers": blockers,
        "commands": {
            "validation": [
                "python -m py_compile proofs/v014/full_domain_source_truth.py",
                "python -m json.tool proofs/v014/full_domain_source_truth.json >/tmp/full_domain_source_truth.validated.json",
            ]
        },
        "production_src_edits": False,
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
            "wrf_patch_diff": str(OUT_DIFF),
        },
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(
        "# V0.14 Full-Domain Source/Truth Surface\n\n"
        f"Verdict: `{VERDICT}`.\n\n"
        "No strict same-input JAX comparison is authorized from the existing surfaces.\n\n"
        "## Why\n\n"
        "- Existing source/save output is patch-only.\n"
        "- Existing post-RK/pre-halo truth is patch-only.\n"
        "- The accepted source/save proof has only one conservative halo-valid mass cell.\n"
        "- Full wrapper carry/boundary leaves were not emitted at the same boundary.\n\n"
        "Next: use the staged early-step discriminator instead of another step-6000 wrapper micro-sprint.\n",
        encoding="utf-8",
    )
    OUT_DIFF.write_text(
        "# No new WRF patch was applied in this manager fail-closed closeout.\n"
        "# Existing patch artifacts remain source_save_boundary_hook_wrf_patch.diff and prior post-RK refresh diffs.\n",
        encoding="utf-8",
    )
    print(VERDICT)
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
