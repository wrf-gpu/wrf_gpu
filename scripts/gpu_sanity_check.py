#!/usr/bin/env python3
"""Check optional local GPU tooling without failing hard when absent."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys


MODULES = ["jax", "triton", "cupy", "numba", "torch"]


def module_status(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    return {"installed": spec is not None}


def nvidia_smi() -> dict:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"available": False}
    proc = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"available": proc.returncode == 0, "output": proc.stdout.strip(), "error": proc.stderr.strip()}


def main() -> int:
    result = {
        "ok": True,
        "cuda_related_modules": {name: module_status(name) for name in MODULES},
        "nvidia_smi": nvidia_smi(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
