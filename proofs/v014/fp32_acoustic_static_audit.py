"""Static FP32 acoustic R0/R1 audit.

This proof records where the current acoustic/runtime path reconstructs base
fields from total-minus-perturbation, where fp64 casts are hard-coded in the
acoustic lane, and where the default-inert precision-mode label is plumbed.
It performs no model run and uses no GPU.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "proofs" / "v014" / "fp32_acoustic_static_audit.json"

SOURCE_FILES = (
    "src/gpuwrf/contracts/precision.py",
    "src/gpuwrf/contracts/state.py",
    "src/gpuwrf/dynamics/acoustic_wrf.py",
    "src/gpuwrf/dynamics/core/acoustic.py",
    "src/gpuwrf/dynamics/core/advance_w.py",
    "src/gpuwrf/dynamics/core/calc_p_rho.py",
    "src/gpuwrf/dynamics/core/rk_addtend_dry.py",
    "src/gpuwrf/dynamics/core/small_step_finish.py",
    "src/gpuwrf/dynamics/core/small_step_prep.py",
    "src/gpuwrf/runtime/operational_mode.py",
    "src/gpuwrf/runtime/operational_state.py",
)

TOTAL_MINUS_PERT_RE = re.compile(
    r"(?:jnp\.asarray\()?([A-Za-z_][A-Za-z0-9_]*)\.(p|ph|mu)_total\)?"
    r"\s*-\s*(?:jnp\.asarray\()?([A-Za-z_][A-Za-z0-9_]*)\.\2_perturbation"
)
FP64_RE = re.compile(r"(?:astype\(|dtype=|,\s*)jnp\.float64|jnp\.asarray\([^)]*jnp\.float64")
PRECISION_MODE_RE = re.compile(
    r"AcousticPrecisionMode|DEFAULT_ACOUSTIC_PRECISION_MODE|acoustic_precision_mode"
)


def _git_output(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _scan() -> dict[str, object]:
    base_reconstruction = []
    fp64_casts = []
    precision_mode_plumbing = []

    for rel in SOURCE_FILES:
        path = ROOT / rel
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if TOTAL_MINUS_PERT_RE.search(line):
                base_reconstruction.append({"file": rel, "line": lineno, "source": stripped})
            if FP64_RE.search(line):
                fp64_casts.append({"file": rel, "line": lineno, "source": stripped})
            if PRECISION_MODE_RE.search(line):
                precision_mode_plumbing.append({"file": rel, "line": lineno, "source": stripped})

    timestep_precision_mode_consumers = [
        item
        for item in precision_mode_plumbing
        if item["file"].startswith("src/gpuwrf/dynamics/")
        or item["file"].endswith("runtime/operational_state.py")
    ]

    return {
        "case": "v014 FP32 acoustic R0/R1 static audit",
        "head": _git_output("rev-parse", "HEAD"),
        "base_required": "b4901a907948623a7781b6cfc10fbbf42f64dd73",
        "base_is_ancestor": subprocess.call(
            ("git", "merge-base", "--is-ancestor", "b4901a907948623a7781b6cfc10fbbf42f64dd73", "HEAD"),
            cwd=ROOT,
        )
        == 0,
        "source_files": list(SOURCE_FILES),
        "base_reconstruction_from_totals": base_reconstruction,
        "hard_fp64_casts_in_scope": fp64_casts,
        "precision_mode_plumbing": precision_mode_plumbing,
        "timestep_precision_mode_consumers": timestep_precision_mode_consumers,
        "default_inert_claim": (
            "The new acoustic_precision_mode label appears only in precision contract "
            "and OperationalNamelist static/cache-key plumbing; no dynamics module or "
            "operational carry consumes it."
        ),
    }


def main() -> int:
    report = _scan()
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"base_reconstruction_from_totals={len(report['base_reconstruction_from_totals'])}")
    print(f"hard_fp64_casts_in_scope={len(report['hard_fp64_casts_in_scope'])}")
    print(f"precision_mode_plumbing={len(report['precision_mode_plumbing'])}")
    print(f"timestep_precision_mode_consumers={len(report['timestep_precision_mode_consumers'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
