from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import importlib.util


_SPEC = importlib.util.spec_from_file_location("m5_gate_rrtmg", Path("scripts/m5_gate_rrtmg.py"))
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
evaluate_gate = _MODULE.evaluate_gate


def test_rrtmg_gate_reports_honest_launch_gray_zone():
    if not Path("artifacts/m5/rrtmg_profile.json").exists():
        subprocess.run([sys.executable, "scripts/m5_run_rrtmg.py"], check=True)
    record = evaluate_gate()
    assert record["gate_status"] == "GRAY-ZONE"
    assert record["tier1_sw_pass"] is True
    assert record["tier1_lw_pass"] is True
    assert record["tier2_pass"] is True
    assert record["kernel_launches_per_step"] == record["raw_hlo_launch_marker_count"]
    assert record["kernel_launches_per_step"] > 5
