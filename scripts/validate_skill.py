#!/usr/bin/env python3
"""Validate one AgentOS skill folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REQUIRED_SECTIONS = [
    "## When to use",
    "## Inputs required",
    "## Workflow",
    "## Hard rules",
    "## Deliverables",
    "## Validation",
    "## Common failure modes",
]


def validate_skill(path: Path) -> dict:
    errors: list[str] = []
    skill_md = path / "SKILL.md"
    evals_json = path / "evals" / "evals.json"

    if not skill_md.exists():
        errors.append("missing SKILL.md")
        text = ""
    else:
        text = skill_md.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            errors.append("missing YAML frontmatter")
        if "\nname:" not in text or "\ndescription:" not in text:
            errors.append("frontmatter must include name and description")
        for section in REQUIRED_SECTIONS:
            if section not in text:
                errors.append(f"missing section: {section}")

    if not evals_json.exists():
        errors.append("missing evals/evals.json")
    else:
        try:
            data = json.loads(evals_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid evals.json: {exc}")
        else:
            cases = data.get("cases", data if isinstance(data, list) else [])
            if not isinstance(cases, list) or len(cases) < 3:
                errors.append("evals.json must contain at least three cases")
            else:
                for index, case in enumerate(cases, start=1):
                    for key in ("query", "required_files", "expected_behavior", "failure_conditions"):
                        if key not in case:
                            errors.append(f"case {index} missing {key}")

    return {"skill": str(path), "ok": not errors, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("skill_path", type=Path)
    args = parser.parse_args()
    result = validate_skill(args.skill_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
