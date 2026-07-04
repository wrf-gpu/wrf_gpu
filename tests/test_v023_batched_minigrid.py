from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys


def test_f1_cpu_b2_contamination_free_tolerance_gate():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "proofs/v023/batched_minigrid/f1_cpu_b2_gate.py"
    spec = importlib.util.spec_from_file_location("f1_cpu_b2_gate", script)
    assert spec is not None and spec.loader is not None
    f1_cpu_b2_gate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = f1_cpu_b2_gate
    spec.loader.exec_module(f1_cpu_b2_gate)

    assert f1_cpu_b2_gate.main() == 0
    proof_path = Path(f1_cpu_b2_gate.JSON_OUT)
    payload = json.loads(proof_path.read_text())
    assert payload["verdict"] == "PASS"
    assert payload["gate"] == "F1_REAL_DYCORE_CPU_B2_CONTAMINATION_FREE_TOLERANCE"
    assert payload["root_cause_verdict"] == "B_XLA_ORDER_CONTAMINATION_FREE"
    assert all(payload["checks"].values())
    f1_abs = payload["global_metrics"]["f1_batched_eager_vs_standalone"]["max_abs"]["value"]
    f1_rel = payload["global_metrics"]["f1_batched_eager_vs_standalone"]["max_relative"]["value"]
    assert f1_abs <= f1_cpu_b2_gate.CPU_STANDALONE_VMAP_ABS_TOL
    assert f1_rel <= f1_cpu_b2_gate.CPU_STANDALONE_VMAP_REL_TOL
