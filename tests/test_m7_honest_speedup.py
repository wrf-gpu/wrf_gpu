from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/m7_cpu_per_domain_timing.py"
spec = importlib.util.spec_from_file_location("m7_cpu_per_domain_timing", SCRIPT)
m7_cpu_per_domain_timing = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = m7_cpu_per_domain_timing
spec.loader.exec_module(m7_cpu_per_domain_timing)


def test_parse_timing_line_extracts_domain_elapsed_and_timestamp() -> None:
    record = m7_cpu_per_domain_timing.parse_timing_line(
        "Timing for main: time 2026-05-21_18:00:06 on domain   2:    5.93500 elapsed seconds"
    )

    assert record is not None
    assert record.domain == 2
    assert record.timestamp.isoformat() == "2026-05-21T18:00:06+00:00"
    assert record.elapsed_s == pytest.approx(5.935)


def test_parse_timing_files_deduplicates_mirrored_rsl_records(tmp_path: Path) -> None:
    lines = "\n".join(
        [
            "Timing for main: time 2026-05-21_18:00:06 on domain   2:    1.00000 elapsed seconds",
            "Timing for main: time 2026-05-21_18:00:12 on domain   2:    2.00000 elapsed seconds",
        ]
    )
    rsl_error = tmp_path / "rsl.error.0000"
    rsl_out = tmp_path / "rsl.out.0000"
    rsl_error.write_text(lines + "\n", encoding="utf-8")
    rsl_out.write_text(lines + "\n", encoding="utf-8")

    parsed = m7_cpu_per_domain_timing.parse_timing_files([rsl_error, rsl_out])
    domains = m7_cpu_per_domain_timing.summarize_domains(parsed["records"])

    assert parsed["raw_record_count"] == 4
    assert parsed["unique_record_count"] == 2
    assert parsed["duplicate_record_count"] == 2
    assert domains == [
        {
            "domain": "d02",
            "domain_id": 2,
            "step_count": 2,
            "total_wall_s": pytest.approx(3.0),
            "mean_per_step_s": pytest.approx(1.5),
            "median_per_step_s": pytest.approx(1.5),
            "min_per_step_s": pytest.approx(1.0),
            "max_per_step_s": pytest.approx(2.0),
            "first_model_time": "2026-05-21T18:00:06+00:00",
            "last_model_time": "2026-05-21T18:00:12+00:00",
            "coverage_s": pytest.approx(6.0),
            "median_model_step_s": pytest.approx(6.0),
            "expected_24h_steps": 14400,
            "complete_24h_timing": False,
        }
    ]


def test_build_speedup_payload_uses_required_comparison_rows() -> None:
    cpu_payload = {
        "selected_run": {
            "run_id": "synthetic",
            "run_path": "/tmp/synthetic",
            "domains": [
                {"domain_id": 1, "total_wall_s": 10.0},
                {"domain_id": 2, "total_wall_s": 20.0},
                {"domain_id": 3, "total_wall_s": 1.0},
                {"domain_id": 4, "total_wall_s": 2.0},
                {"domain_id": 5, "total_wall_s": 3.0},
            ],
        }
    }

    payload = m7_cpu_per_domain_timing.build_speedup_payload(cpu_payload, gpu_wall_s=5.0)

    rows = {row["comparison_id"]: row for row in payload["rows"]}
    assert rows["cpu_full_nest_5_domain_aggregate_24h"]["ratio"] == pytest.approx(7.2)
    assert rows["cpu_d02_only_24h"]["ratio"] == pytest.approx(4.0)
    assert rows["cpu_d01_plus_d02_minimum_physical_subset_24h"]["ratio"] == pytest.approx(6.0)
    assert rows["cpu_d01_only_24h"]["ratio"] == pytest.approx(2.0)
