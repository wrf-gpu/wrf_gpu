#!/usr/bin/env python3
"""Fail-closed wrapper gate for v0.14 full-domain same-input single-RK parity."""

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
TRUTH_JSON = PROOF_DIR / "full_domain_source_truth.json"
OUT_JSON = PROOF_DIR / "same_input_single_rk_parity_wrapped.json"
OUT_MD = PROOF_DIR / "same_input_single_rk_parity_wrapped.md"

VERDICT = "FULL_DOMAIN_WRAPPER_BLOCKED_TRUTH_SURFACE_PATCH_ONLY_AND_CARRY_LEAVES"


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    truth = load_json(TRUTH_JSON)
    blockers = list(truth.get("blockers", [])) or ["full-domain truth inventory was missing"]
    payload: dict[str, Any] = {
        "schema": "wrfgpu2.v014.same_input_single_rk_parity_wrapped.v1",
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
            "full_domain_source_truth_json": path_info(TRUTH_JSON),
            "sprint_contract": path_info(
                ROOT / ".agent/sprints/2026-06-09-v014-full-domain-source-wrapper/sprint-contract.md"
            ),
        },
        "decision": {
            "strict_same_input_comparison_run": False,
            "jax_step_executed": False,
            "weak_comparison_avoided": True,
            "truth_surface_sufficient": bool(truth.get("truth_surface_sufficient", False)),
            "reason": "existing WRF source/save and post-RK truth surfaces are patch-only and missing full wrapper carry/boundary leaves",
        },
        "comparison": {
            "per_field_metrics": {},
            "ranked_residuals": [],
        },
        "blockers": blockers,
        "commands": {
            "validation": [
                "python -m py_compile proofs/v014/same_input_single_rk_parity_wrapped.py",
                "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/same_input_single_rk_parity_wrapped.py",
                "python -m json.tool proofs/v014/same_input_single_rk_parity_wrapped.json >/tmp/same_input_single_rk_parity_wrapped.validated.json",
            ]
        },
        "production_src_edits": False,
        "proof_objects": {
            "json": str(OUT_JSON),
            "markdown": str(OUT_MD),
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(
        "# V0.14 Same-Input Single-RK Wrapped Gate\n\n"
        f"Verdict: `{VERDICT}`.\n\n"
        "No JAX step was executed. A weak comparison was avoided because the available WRF surfaces do not satisfy the full-domain same-input wrapper contract.\n\n"
        "## Blockers\n\n"
        + "\n".join(f"- {item}" for item in blockers)
        + "\n\nNext: use the staged early-step discriminator and bisect from shared `wrfinput`.\n",
        encoding="utf-8",
    )
    print(VERDICT)
    print(f"json={OUT_JSON}")
    print(f"markdown={OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
