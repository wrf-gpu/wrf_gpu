#!/usr/bin/env python3
"""Compatibility entry point for the tracked CPU E41 gate.

Usage: gpu_smoke_e41.py CACHE OUT -> OUT/e41.json (and wave_gate.json).
Both entry points now enforce the same native/HLO gates and cache coverage.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

GATE = Path(__file__).resolve().with_name("wave_cubin_gate.py")
CUOBJDUMP = os.environ.get("CUOBJDUMP") or shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"


def _gate_module():
    spec = importlib.util.spec_from_file_location("wave_cubin_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cubin_resources(path: Path) -> list[dict]:
    return _gate_module()._scanner("cubin_scan").scan(path, CUOBJDUMP)


def kinput_transpose_over500(text: str) -> list[tuple[str, int]]:
    result = _gate_module()._scanner("hlo_scan").scan_text(text)
    rows = [row for row in result["transpose_hero_fusions"] if row["kind"] == "kInput"]
    kinput_transpose_over500.stats = {"computations": result["computations"],
        "kinput_transpose_fusions": len(rows), "max_kinput_transpose_instr": max((r["n_instr"] for r in rows), default=0)}
    return [(row["fusion"], row["n_instr"]) for row in rows if row["n_instr"] > 500]


def scan(cache: Path, out: Path) -> dict:
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise ValueError("E41 is CPU-only: set JAX_PLATFORMS=cpu")
    if Path("/tmp/wrf_gpu2_quiet").exists():
        raise ValueError("QUIET")
    (out / "e41.json").unlink(missing_ok=True)
    result = _gate_module().run_gate(cache, out)
    result.update(tool="scripts/e41 (tracked full gate)", unscanned=[],
                  max_stack_B=max((e["max_stack_B"] for r in result["entries"] for e in r["elfs"]), default=None))
    (out / "e41.json").write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    result = scan(Path(argv[0]), Path(argv[1]))
    print(json.dumps({key: result[key] for key in ("n_entries", "e41_gate", "hard_stack_gate", "max_stack_B", "conservative_flags", "unscanned")}))
    return 0 if result["e41_gate"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
