from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_mynn_gate_result_is_go_carryforward_when_artifacts_exist():
    if not Path("artifacts/m5/mynn_gate_result.json").exists():
        subprocess.run([sys.executable, "scripts/m5_run_mynn.py"], check=True)
        subprocess.run([sys.executable, "scripts/m5_gate_mynn.py"], check=True)
    payload = json.loads(Path("artifacts/m5/mynn_gate_result.json").read_text(encoding="utf-8"))
    assert payload["gate_status"] == "GO_CARRYFORWARD"
    assert payload["kernel_launches_per_step"] <= 35
    assert payload["tier1_pass"] is True
    assert payload["tier2_pass"] is True
    assert payload["tier2_independent_budget_pass"] is True
