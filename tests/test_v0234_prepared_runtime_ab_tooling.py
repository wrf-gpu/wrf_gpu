"""CPU-only tests for the prepared-runtime A/B artifact analyzer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_ANALYZER_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "v0234_prepared_runtime_ab_analyze.py"
)
_SPEC = importlib.util.spec_from_file_location("v0234_prepared_runtime_ab_analyze", _ANALYZER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ANALYZER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ANALYZER)
analyze = _ANALYZER.analyze


def _write_arm(
    root: Path,
    label: str,
    *,
    start_epoch: int,
    command_wall_s: int,
    mtimes: list[int],
    prepared: bool,
    loads: int,
) -> None:
    arm = root / f"arm_{label}"
    (arm / "proof").mkdir(parents=True)
    (arm / "proof" / "nested_pipeline_run.json").write_text(
        json.dumps(
            {
                "verdict": "PIPELINE_GREEN",
                "all_domains_finite": True,
                "all_outputs_present": True,
                "metadata": {
                    "nested_aot": {"load_count": loads},
                    "nested_runtime": {"prepared_runtime_reuse": prepared},
                },
            }
        ),
        encoding="utf-8",
    )
    (arm / "wall.json").write_text(
        json.dumps(
            {
                "start_epoch_ns": start_epoch * 1_000_000_000,
                "command_wall_s": command_wall_s,
            }
        ),
        encoding="utf-8",
    )
    valid = (
        "2026-07-06_18:20:00",
        "2026-07-06_18:40:00",
        "2026-07-06_19:00:00",
        "2026-07-06_19:20:00",
        "2026-07-06_19:40:00",
        "2026-07-06_20:00:00",
        "2026-07-06_20:20:00",
    )
    (arm / "wrfout_mtimes.tsv").write_text(
        "".join(
            f"wrfout_d09_{stamp}\t{mtime}.0\t100\n"
            for stamp, mtime in zip(valid, mtimes, strict=True)
        ),
        encoding="utf-8",
    )
    (arm / "wrfout_sha256.txt").write_text(
        "samehash  ./wrfout_d09_frame\n", encoding="utf-8"
    )
    (arm / "process_resources.tsv").write_text(
        "epoch_s\tpid\tvmrss_kib\tvmswap_kib\tmemavailable_kib\n"
        "1\t10\t1000\t20\t20000000\n",
        encoding="utf-8",
    )
    (arm / "nvidia_smi.csv").write_text(
        "2026/07/10 12:00:00, 30000, 2607, 80, 20\n", encoding="utf-8"
    )


def test_analyzer_accepts_27_to_9_exact_pair(tmp_path):
    output_root = tmp_path / "ab"
    # A post-hour gaps are 300 s; B post-hour gaps are 150 s. Ordinary gaps are
    # 100 s in both arms, and command-to-last-frame improves by one third.
    _write_arm(
        output_root,
        "a",
        start_epoch=1000,
        command_wall_s=1200,
        mtimes=[1100, 1200, 1300, 1600, 1700, 1800, 2100],
        prepared=False,
        loads=27,
    )
    _write_arm(
        output_root,
        "b",
        start_epoch=5000,
        command_wall_s=800,
        mtimes=[5100, 5200, 5300, 5450, 5550, 5650, 5800],
        prepared=True,
        loads=9,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "prepared_runtime_ab_prearm": {
                    "selection_status": "PRE_ARM_READY"
                }
            }
        ),
        encoding="utf-8",
    )
    (output_root / "wrfout_variable_identity.txt").write_text(
        "# SUMMARY: verdict=BYTE-IDENTICAL\n", encoding="utf-8"
    )

    result = analyze(output_root, manifest)

    assert result["automated_verdict"] == "GREEN_PENDING_MANUAL_PROFILE_REVIEW"
    assert result["automated_gates"]["load_count_27_to_9"] is True
    assert result["automated_gates"]["post_hour_gain_at_least_25pct"] is True
    assert result["manual_gates_remaining"]


def test_analyzer_rejects_wrong_candidate_load_count(tmp_path):
    output_root = tmp_path / "ab"
    _write_arm(
        output_root,
        "a",
        start_epoch=1000,
        command_wall_s=1200,
        mtimes=[1100, 1200, 1300, 1600, 1700, 1800, 2100],
        prepared=False,
        loads=27,
    )
    _write_arm(
        output_root,
        "b",
        start_epoch=5000,
        command_wall_s=800,
        mtimes=[5100, 5200, 5300, 5450, 5550, 5650, 5800],
        prepared=True,
        loads=18,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "prepared_runtime_ab_prearm": {
                    "selection_status": "PRE_ARM_READY"
                }
            }
        ),
        encoding="utf-8",
    )
    (output_root / "wrfout_variable_identity.txt").write_text(
        "# SUMMARY: verdict=BYTE-IDENTICAL\n", encoding="utf-8"
    )

    result = analyze(output_root, manifest)

    assert result["automated_verdict"] == "REJECT_OR_INVALID"
    assert result["automated_gates"]["load_count_27_to_9"] is False
