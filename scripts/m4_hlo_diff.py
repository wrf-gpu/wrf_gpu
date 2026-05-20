#!/usr/bin/env python3
"""Produce the M4 production-vs-stripped HLO identity proof object."""

from __future__ import annotations

import difflib
import hashlib
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.dynamics.step import step  # noqa: E402
from gpuwrf.dynamics.step_debug_stripped import step_debug_stripped  # noqa: E402
from gpuwrf.profiling.budget import compiled_text, write_hlo  # noqa: E402
from gpuwrf.validation.tier2 import density_current_state, make_ideal_grid  # noqa: E402


ART = ROOT / "artifacts" / "m4" / "hlo_dump"
SCRATCH = ROOT / "data" / "scratch" / "m4"


def _normalize(text: str) -> str:
    """Normalizes HLO spelling noise so only real operations remain in the diff."""

    text = re.sub(r"metadata=\{.*?\}", "metadata={}", text, flags=re.DOTALL)
    text = re.sub(r"frontend_attributes=\{.*?\}\}", "frontend_attributes={}", text, flags=re.DOTALL)
    text = re.sub(r"fingerprint_before_lhs=\"[^\"]+\"", 'fingerprint_before_lhs=""', text)
    text = re.sub(r"FileNames.*?ENTRY", "ENTRY", text, flags=re.DOTALL)
    text = re.sub(r"step_debug_stripped", "step", text)
    text = re.sub(r"_rk3_[A-Za-z0-9_]+", "_rk3_fn", text)
    text = re.sub(r"stack_frame_id=\d+", "stack_frame_id=0", text)
    text = re.sub(r"jit_[A-Za-z0-9_]+", "jit_fn", text)
    text = re.sub(r"@[A-Za-z0-9_.$-]+", "@sym", text)
    text = re.sub(r"%[A-Za-z_][A-Za-z0-9_.-]*", "%v", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> int:
    """Writes production HLO, stripped HLO, and an empty diff on identity."""

    ART.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    grid = make_ideal_grid()
    state, tendencies = density_current_state(grid)
    prod = compiled_text(step.lower(state, tendencies, grid, 2.0, n_acoustic=4, debug=False).compile())
    stripped = compiled_text(step_debug_stripped.lower(state, tendencies, grid, 2.0, n_acoustic=4).compile())
    write_hlo(ART / "dycore_step_production.txt", prod, SCRATCH / "dycore_step_production_full.txt")
    write_hlo(ART / "dycore_step_debug_stripped.txt", stripped, SCRATCH / "dycore_step_debug_stripped_full.txt")

    diff_path = ART / "dycore_step_debug_vs_stripped.diff"
    if _normalize(prod) == _normalize(stripped):
        diff_path.write_text("", encoding="utf-8")
    else:
        diff = "\n".join(
            difflib.unified_diff(
                _normalize(prod).split(),
                _normalize(stripped).split(),
                fromfile="production",
                tofile="stripped",
                lineterm="",
            )
        )
        diff_path.write_text(diff + "\n", encoding="utf-8")

    digest = hashlib.sha256(diff_path.read_bytes()).hexdigest()
    print(f"{diff_path} bytes={diff_path.stat().st_size} sha256={digest}")
    return 0 if diff_path.stat().st_size == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
