"""Regression tests for v0.23.4 shell resource-accounting parsers."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
VRAM_SCRIPTS = (
    "scripts/v0234_prepared_runtime_ab.sh",
    "scripts/v0234_frozen_bundle_perf_discriminator.sh",
    "scripts/v0234_flat2_fusion_discriminator.sh",
    "scripts/v0234_boundary_default_confirm.sh",
    "scripts/v0234_gpt_drift_24h.sh",
)


@pytest.mark.parametrize("relative_path", VRAM_SCRIPTS)
def test_vram_peak_comparison_is_numeric(relative_path: str) -> None:
    source = (REPO / relative_path).read_text(encoding="utf-8")
    assert 'v=$2+0; if (v>m) m=v' in source
    assert 'if ($2>m) m=$2' not in source


def test_vram_peak_parser_does_not_choose_lexicographic_max(tmp_path: Path) -> None:
    samples = tmp_path / "nvidia_smi.csv"
    samples.write_text(
        "2026/07/20 21:00:00, 9974, 22133, 90, 20\n"
        "2026/07/20 21:00:01, 12405, 19702, 80, 10\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "awk",
            "-F,",
            '{gsub(/ /,"",$2); v=$2+0; if (v>m) m=v} END {print m+0}',
            str(samples),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    assert result.stdout.strip() == "12405"
