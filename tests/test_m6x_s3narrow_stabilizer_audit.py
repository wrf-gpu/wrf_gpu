from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "scripts" / "diagnostic_stabilizer_provenance_scanner.py"
DYNAMICS_DIR = ROOT / "src" / "gpuwrf" / "dynamics"
ACOUSTIC_WRF = DYNAMICS_DIR / "acoustic_wrf.py"


def _scan_counts(tmp_path: Path) -> dict[str, int]:
    output = tmp_path / "stabilizer_scan.json"
    subprocess.run(
        [
            sys.executable,
            str(SCANNER),
            "--input",
            str(DYNAMICS_DIR),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    return payload["measurements"]["classification_counts"]


def test_experiment_backed_count_below_s2_baseline(tmp_path: Path) -> None:
    counts = _scan_counts(tmp_path)

    assert counts["experiment-backed"] < 28


def test_source_backed_count_above_s2_baseline(tmp_path: Path) -> None:
    counts = _scan_counts(tmp_path)

    assert counts["source-backed"] > 8


def test_reject_count_is_zero(tmp_path: Path) -> None:
    counts = _scan_counts(tmp_path)

    assert counts["reject"] == 0


def test_mu_continuity_increment_remains() -> None:
    source = ACOUSTIC_WRF.read_text(encoding="utf-8")

    assert "def _mu_continuity_increment(" in source
    assert "DEFER to post-S2.1 sprint pending real Gen2 baseline" in source
