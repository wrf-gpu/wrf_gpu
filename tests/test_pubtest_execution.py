from __future__ import annotations

import json
from pathlib import Path

from scripts.pubtest_common import HIGH_TEST_FILES
from scripts.pubtest_execute_high_priority import execute


def test_high_priority_executor_writes_required_proof_objects(tmp_path: Path) -> None:
    result = execute(tmp_path, skip_gpu_probe=True)

    assert result["status"] == "EXECUTION_PARTIAL"
    for filename in HIGH_TEST_FILES.values():
        path = tmp_path / filename
        assert path.exists(), filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["verdict"] in {"PASS", "FAIL", "BLOCKED"}
        assert payload["test_id"]
    aggregate = tmp_path / "aggregate_report.md"
    assert aggregate.exists()
    text = aggregate.read_text(encoding="utf-8")
    assert "Total GPU hours used" in text
    assert "IDEALIZED-WARMBUBBLE" in text


def test_canary_case_manifest_is_written_by_executor(tmp_path: Path) -> None:
    execute(tmp_path, skip_gpu_probe=True)
    manifest = json.loads((tmp_path / "canary_case_manifest.json").read_text(encoding="utf-8"))

    assert "available_day_count" in manifest
    assert "cases" in manifest
