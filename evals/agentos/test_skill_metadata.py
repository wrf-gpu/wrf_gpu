from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_all_skills_have_metadata_and_evals() -> None:
    for skill in (ROOT / ".agent" / "skills").iterdir():
        if not skill.is_dir():
            continue
        text = (skill / "SKILL.md").read_text(encoding="utf-8")
        assert text.startswith("---\n"), skill.name
        assert "\nname:" in text, skill.name
        data = json.loads((skill / "evals" / "evals.json").read_text(encoding="utf-8"))
        assert len(data["cases"]) >= 3, skill.name
