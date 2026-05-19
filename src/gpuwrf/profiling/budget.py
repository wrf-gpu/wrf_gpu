"""Spacetime-budget helpers for the M3 state skeleton."""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.profiling.transfer_audit import block_until_ready


def compiled_text(compiled: Any) -> str:
    """Normalizes JAX compiled HLO text across minor API differences."""

    for kwargs in ({"dialect": "hlo"}, {}):
        try:
            return str(compiled.as_text(**kwargs))
        except TypeError:
            continue
    return str(compiled)


def write_hlo(path: Path, text: str, full_path: Path | None = None) -> None:
    """Writes manager-readable HLO, truncating tracked evidence above 100 KB."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    if len(encoded) <= 100_000:
        path.write_text(text, encoding="utf-8")
        return
    if full_path is not None:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(text, encoding="utf-8")
    head = "\n".join(text.splitlines()[:1000])
    note = f"\n\n# Truncated for git hygiene; full HLO: {full_path}\n"
    path.write_text(head + note, encoding="utf-8")


def kernel_launches_per_step(hlo_text: str) -> int:
    """Derives the raw dummy-loop launch estimate from fused HLO operations."""

    fusion_count = len(re.findall(r"\bfusion\(", hlo_text))
    custom_count = len(re.findall(r"\bcustom-call\(", hlo_text))
    while_count = len(re.findall(r"\bwhile\(", hlo_text))
    launches = fusion_count + custom_count
    if launches == 0 and while_count:
        launches = 1
    return max(1, launches)


def median_step_us(run_once, n_steps: int, samples: int = 100) -> float:
    """Measures median per-step wall time after caller-provided compilation warmup."""

    timings: list[float] = []
    for _ in range(samples):
        start = time.perf_counter()
        result = run_once()
        block_until_ready(result)
        timings.append((time.perf_counter() - start) * 1_000_000.0 / float(n_steps))
    return float(statistics.median(timings))


def write_spacetime_budget(
    path: Path,
    state: State,
    tendencies: Tendencies,
    halo_buffer_bytes: int,
    launches_per_step: int,
    wall_time_per_step_us: float,
) -> dict[str, Any]:
    """Writes the M3 budget JSON consumed by check_m3_done.py and worker-report."""

    state_bytes = state.bytes()
    tendency_bytes = tendencies.bytes()
    payload = {
        "state_bytes": int(state_bytes),
        "tendency_bytes": int(tendency_bytes),
        "temporary_bytes_per_step": 0,
        "halo_buffer_bytes": int(halo_buffer_bytes),
        "total_persistent_bytes": int(state_bytes + tendency_bytes + halo_buffer_bytes),
        "flops_per_cell_per_step": 4,
        "kernel_launches_per_step": int(launches_per_step),
        "wall_time_per_step_us": float(wall_time_per_step_us),
        "notes": {
            "temporary_bytes_per_step": "0 because the scan body has no array constructors and HLO fuses the add/sub chain.",
            "flops_per_cell_per_step": "theta no-op: multiply/add/subtract/multiply over one mass field.",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
