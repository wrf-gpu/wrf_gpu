from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_precision_bench_self_test() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/precision_bench.py", "--self-test"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["kernels"] == ["m2_column", "m4_dycore", "m5_thompson"]
    assert "fp32" in payload["gpu_precisions"]["m5_thompson"]
