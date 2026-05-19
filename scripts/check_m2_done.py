#!/usr/bin/env python3
"""Single-command oracle for M2 (Backend Bakeoff) completion.

Mirrors check_m1_done.py's contract: prints a JSON record with `ok` (bool)
and `errors` (list of strings). Exit 0 if `ok`, 1 otherwise.

Assertions documented in .agent/goals/M2-DONE.md.
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPRINTS_GLOB = str(ROOT / ".agent" / "sprints" / "2026-*-m2-*")
ARTIFACTS_DIR = ROOT / "artifacts" / "m2"

CANDIDATES = ["jax", "triton", "gt4py", "kokkos", "cupy_or_numba", "cuda_tile"]
PROBLEMS = ["stencil", "column"]


def _run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd,
        cwd=cwd or ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.returncode, proc.stdout, proc.stderr


def check_repo_hygiene(errors: list[str]) -> None:
    rc, out, _ = _run([sys.executable, "scripts/validate_agentos.py"])
    if rc != 0:
        errors.append(f"validate_agentos.py exited {rc}")
    else:
        try:
            j = json.loads(out)
            if not j.get("ok"):
                errors.append(f"validate_agentos.py reported errors: {j.get('errors')}")
        except json.JSONDecodeError:
            errors.append("validate_agentos.py did not emit JSON")

    rc, out, err = _run([sys.executable, "-m", "pytest", "-q"])
    if rc != 0:
        errors.append(f"pytest -q exited {rc}: {(err or out).splitlines()[-5:]}")


def check_m1_not_regressed(errors: list[str]) -> None:
    rc, out, _ = _run([sys.executable, "scripts/check_m1_done.py"])
    if rc != 0:
        try:
            j = json.loads(out)
            errors.append(f"M1 regression: {j.get('errors')}")
        except json.JSONDecodeError:
            errors.append(f"M1 regression: check_m1_done.py exited {rc}")


def check_sprints_closed(errors: list[str]) -> int:
    sprint_dirs = sorted(glob.glob(SPRINTS_GLOB))
    if not sprint_dirs:
        errors.append("no M2 sprint folders found")
        return 0
    closed = 0
    for d in sprint_dirs:
        rc, out, _ = _run([sys.executable, "scripts/close_sprint.py", d])
        if rc != 0:
            errors.append(f"sprint {Path(d).name} not closed: {out.strip()[:300]}")
        else:
            closed += 1
    return closed


def _candidate_satisfied(candidate: str, errors: list[str]) -> bool:
    cand_dir = ARTIFACTS_DIR / candidate
    if not cand_dir.exists():
        errors.append(f"candidate {candidate}: no artifacts/m2/{candidate}/ directory")
        return False
    ok = True
    for problem in PROBLEMS:
        prof = cand_dir / f"{problem}_profile.json"
        fail = cand_dir / f"{problem}_failure.json"
        if not prof.exists() and not fail.exists():
            errors.append(
                f"candidate {candidate} / problem {problem}: neither {problem}_profile.json nor {problem}_failure.json"
            )
            ok = False
    # Correctness (only required if at least one profile exists)
    has_any_profile = any((cand_dir / f"{p}_profile.json").exists() for p in PROBLEMS)
    if has_any_profile:
        corr = cand_dir / "correctness.json"
        if not corr.exists():
            errors.append(f"candidate {candidate}: missing correctness.json")
            ok = False
    # Maintainability + agent-success required for every candidate
    for fname in ("maintainability.md", "agent_success.json"):
        if not (cand_dir / fname).exists():
            errors.append(f"candidate {candidate}: missing {fname}")
            ok = False
    return ok


def check_bakeoff_coverage(errors: list[str]) -> int:
    if not ARTIFACTS_DIR.exists():
        errors.append("missing artifacts/m2/ directory")
        return 0
    satisfied = 0
    for c in CANDIDATES:
        if _candidate_satisfied(c, errors):
            satisfied += 1
    return satisfied


def check_adr_001(errors: list[str]) -> None:
    adr = ROOT / ".agent" / "decisions" / "ADR-001-backend-selection.md"
    if not adr.exists():
        errors.append("missing .agent/decisions/ADR-001-backend-selection.md")
        return
    text = adr.read_text(errors="replace")
    if len(text) < 2000:
        errors.append(f"ADR-001 too short ({len(text)} bytes; expected >=2000)")
    for token in ("Decision:", "Selected backend:", "Dissent", "Evidence summary"):
        if token not in text:
            errors.append(f"ADR-001 missing required token: {token!r}")
    crit = ROOT / ".agent" / "decisions" / "REVIEW-codex-ADR-001.md"
    if not crit.exists():
        errors.append("missing .agent/decisions/REVIEW-codex-ADR-001.md (Codex critical review of ADR-001)")


def check_milestone_closeout(errors: list[str]) -> None:
    closeout = ROOT / ".agent" / "decisions" / "MILESTONE-M2-CLOSEOUT.md"
    if not closeout.exists():
        errors.append("missing .agent/decisions/MILESTONE-M2-CLOSEOUT.md")
    plan = ROOT / ".agent" / "milestones" / "M2-backend-bakeoff.md"
    if plan.exists():
        text = plan.read_text()
        if not re.search(r"Reviewer Decision\s*[:\n]\s*Accepted", text):
            errors.append("M2-backend-bakeoff.md Reviewer Decision is not 'Accepted'")


def check_cross_ai_provenance(errors: list[str]) -> None:
    # Each M2 sprint's tester-report.md should have been produced by Claude (not codex).
    # Heuristic: tester role-prompt header writes 'via claude' OR the log file references claude.
    sprint_dirs = sorted(glob.glob(SPRINTS_GLOB))
    for d in sprint_dirs:
        tr = Path(d) / "tester-report.md"
        if not tr.exists():
            continue  # caught by close_sprint check
        # Look for any log file naming claude as the tester AI for this sprint.
        log_pattern = ROOT / "logs" / f"{Path(d).name}-tester-*.log"
        logs = sorted(glob.glob(str(log_pattern)))
        if not logs:
            errors.append(f"sprint {Path(d).name}: no tester log found to verify AI provenance")
            continue
        last_log = Path(logs[-1]).read_text(errors="replace")
        if "ai=claude" not in last_log:
            errors.append(
                f"sprint {Path(d).name}: tester log does not show ai=claude (cross-AI verification missing)"
            )


def main() -> int:
    errors: list[str] = []
    check_repo_hygiene(errors)
    check_m1_not_regressed(errors)
    closed = check_sprints_closed(errors)
    satisfied = check_bakeoff_coverage(errors)
    check_adr_001(errors)
    check_milestone_closeout(errors)
    check_cross_ai_provenance(errors)

    result = {
        "ok": not errors,
        "errors": errors,
        "sprints_closed": closed,
        "candidates_satisfied": satisfied,
        "candidates_total": len(CANDIDATES),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
