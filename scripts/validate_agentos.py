#!/usr/bin/env python3
"""Validate the repository bootstrap contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from validate_skill import validate_skill


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "PROJECT_CONSTITUTION.md",
    "PROJECT_SPEC.md",
    "PROJECT_SCOPE.md",
    "AGENTS.md",
    "CLAUDE.md",
    "PLANS.md",
    "MILESTONES.md",
    "VALIDATION_STRATEGY.md",
    "PERFORMANCE_TARGETS.md",
    "ARCHITECTURE_PRINCIPLES.md",
    "INTERFACE_CONTRACTS.md",
    "PRECISION_POLICY.md",
    "RISK_REGISTER.md",
    "LICENSE_NOTES.md",
    "CONTRIBUTING_AGENT.md",
    "pyproject.toml",
    ".gitignore",
    ".agent/README.md",
    ".agent/rules/non-negotiables.md",
    ".agent/decisions/ADR-0000-template.md",
    "scripts/create_sprint.py",
    "scripts/close_sprint.py",
    "scripts/validate_memory_patch.py",
    "scripts/validate_adr.py",
    "scripts/repo_status_snapshot.py",
    "scripts/gpu_sanity_check.py",
    "scripts/agent_call.py",
    "tests/test_agentos_smoke.py",
    "tests/test_scripts_smoke.py",
    "src/gpuwrf/__init__.py",
]

SKILLS = [
    "managing-sprints",
    "writing-execplans",
    "conducting-blind-review",
    "resolving-cross-model-disagreements",
    "maintaining-memory",
    "researching-prior-art",
    "building-wrf-oracles",
    "validating-physics",
    "designing-gpu-state",
    "writing-gpu-kernels",
    "profiling-nvidia-gpu",
    "updating-docs-minimally",
    "reporting-to-human",
]


def non_empty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def main() -> int:
    errors: list[str] = []
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).exists()]
    errors.extend(f"missing required file: {name}" for name in missing)

    for name in REQUIRED_FILES:
        path = ROOT / name
        if path.exists() and path.is_file() and path.suffix in {".md", ".py", ".toml"} and not non_empty(path):
            errors.append(f"empty critical file: {name}")

    skill_results = []
    for skill in SKILLS:
        result = validate_skill(ROOT / ".agent" / "skills" / skill)
        skill_results.append(result)
        errors.extend(f"{skill}: {err}" for err in result["errors"])

    result = {
        "ok": not errors,
        "required_files_checked": len(REQUIRED_FILES),
        "skills_checked": len(SKILLS),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
