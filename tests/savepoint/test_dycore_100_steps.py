from __future__ import annotations

from pathlib import Path
from typing import Callable


def test_dycore_column_coupled_step_parity_100_steps(
    wrf_fortran_reference_paths: dict[str, Path],
    wrf_reference_root: Path,
    m6b6_compare_tier: Callable[[str, int, Path], dict[str, object]],
    m6b6_compare_fields: tuple[str, ...],
) -> None:
    del wrf_fortran_reference_paths

    result = m6b6_compare_tier("column", 100, wrf_reference_root)

    assert result["oracle"]["self_compare"] is False
    assert result["oracle"]["source_type"] == "real_wrf_history_fallback"
    assert result["savepoint_count"] == 100
    assert result["operator"] == "coupled_step"
    steps = result["results"]
    assert len(steps) == 100
    assert [step["step"] for step in steps] == list(range(1, 101))
    if result["passed"]:
        assert all(step["passed"] for step in steps)
    else:
        assert result["first_divergence"] is not None
        assert result["first_divergence"]["step"] >= 1
        first = steps[result["first_divergence"]["step"] - 1]
        assert first["first_failed_field"] in m6b6_compare_fields
        assert not first["fields"][first["first_failed_field"]]["passed"]
    assert Path(steps[-1]["path"]).exists()
    assert all(set(step["fields"]) == set(m6b6_compare_fields) for step in steps)
