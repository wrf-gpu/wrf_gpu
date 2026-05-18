from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_required_top_level_files_exist() -> None:
    for name in [
        "PROJECT_CONSTITUTION.md",
        "AGENTS.md",
        "VALIDATION_STRATEGY.md",
        "PERFORMANCE_TARGETS.md",
        "ARCHITECTURE_PRINCIPLES.md",
    ]:
        assert (ROOT / name).exists(), name
