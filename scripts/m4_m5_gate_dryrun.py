#!/usr/bin/env python3
"""Run the M5 stop/go dry-run against the compiled M4 dycore HLO."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpuwrf.dynamics.step import step  # noqa: E402
from gpuwrf.profiling.budget import compiled_text, kernel_launches_per_step  # noqa: E402
from gpuwrf.validation.tier2 import density_current_state, make_ideal_grid  # noqa: E402


ART = ROOT / "artifacts" / "m4"


def main() -> int:
    """Writes the M5 gate dry-run JSON proof object."""

    grid = make_ideal_grid()
    state, tendencies = density_current_state(grid)
    text = compiled_text(step.lower(state, tendencies, grid, 2.0, n_acoustic=4, debug=False).compile())
    launches = kernel_launches_per_step(text)
    local = None
    registers = None
    thresholds = {"kernel_launches_per_step": 10, "local_memory_bytes_per_kernel": 256, "registers_per_kernel": 128}
    tripped = []
    if launches > thresholds["kernel_launches_per_step"]:
        tripped.append("kernel_launches_per_step")
    record = {
        "kernel_launches_per_step": int(launches),
        "local_memory_bytes_per_kernel": local,
        "registers_per_kernel": registers,
        "thresholds": thresholds,
        "gate_status": "trip" if tripped else "pass",
        "tripped_thresholds": tripped,
        "rationale": "HLO-derived launch count only; local memory/register metrics unavailable from HLO and recorded as JSON null pending ncu/cuobjdump follow-up.",
    }
    path = ART / "m5_gate_dryrun.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
