"""CPU-only honesty checks for the v0.17 SAS-family cumulus lane."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "proofs" / "v017" / "sas_family_parity.json"
RUNNER = ROOT / "proofs" / "v017" / "run_sas_family_parity.py"


def _load_runner_main():
    spec = importlib.util.spec_from_file_location("run_sas_family_parity", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_sas_family_parity_report_is_honest_red(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_sas_family_parity.py"])
    assert _load_runner_main()() == 0
    report = json.loads(REPORT.read_text())
    assert report["oracle"]["self_compare"] is False
    assert report["implementation"]["operational_scan_wired"] is False
    assert report["overall_verdict"] == "RED"
    assert set(map(int, report["schemes"])) == {4, 94, 95, 96}
    for scheme in report["schemes"].values():
        assert scheme["oracle"]["nontrivial"] is True
        assert scheme["status"] == "RED"
        assert scheme["worst_max_abs"] > 0.0
