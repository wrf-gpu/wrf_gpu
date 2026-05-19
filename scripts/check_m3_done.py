#!/usr/bin/env python3
"""Single-command oracle for M3 (GPU State & Grid Skeleton) completion.

Mirrors check_m1_done.py / check_m2_done.py. Prints a JSON record with
`ok` (bool) and `errors` (list). Exit 0 if ok, 1 otherwise.

Assertions documented in .agent/goals/M3-DONE.md.
"""

from __future__ import annotations

import glob
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPRINTS_GLOB = str(ROOT / ".agent" / "sprints" / "2026-*-m3-*")
M3_ART = ROOT / "artifacts" / "m3"


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode, proc.stdout, proc.stderr


def check_repo_hygiene(errors: list[str]) -> None:
    rc, out, _ = _run([sys.executable, "scripts/validate_agentos.py"])
    if rc != 0:
        errors.append(f"validate_agentos.py exited {rc}")
    rc, out, err = _run([sys.executable, "-m", "pytest", "-q"])
    if rc != 0:
        errors.append(f"pytest -q exited {rc}: {(err or out).splitlines()[-5:]}")


def check_prior_milestones(errors: list[str]) -> None:
    for prior in ("check_m1_done.py", "check_m2_done.py"):
        rc, out, _ = _run([sys.executable, f"scripts/{prior}"])
        if rc != 0:
            try:
                j = json.loads(out)
                errors.append(f"{prior} regressed: {j.get('errors', [])[:3]}")
            except json.JSONDecodeError:
                errors.append(f"{prior} exited {rc}")


def check_sprints_closed(errors: list[str]) -> int:
    sprint_dirs = sorted(glob.glob(SPRINTS_GLOB))
    if not sprint_dirs:
        errors.append("no M3 sprint folders found")
        return 0
    closed = 0
    for d in sprint_dirs:
        rc, out, _ = _run([sys.executable, "scripts/close_sprint.py", d])
        if rc != 0:
            errors.append(f"sprint {Path(d).name} not closed: {out.strip()[:300]}")
        else:
            closed += 1
    return closed


def check_m3_artifacts(errors: list[str]) -> None:
    must_exist = [
        ROOT / "src" / "gpuwrf" / "contracts" / "grid.py",
        ROOT / "src" / "gpuwrf" / "contracts" / "state.py",
        ROOT / "src" / "gpuwrf" / "contracts" / "halo.py",
        ROOT / "src" / "gpuwrf" / "timestep" / "dummy_loop.py",
        M3_ART / "transfer_audit.json",
        M3_ART / "spacetime_budget.json",
        ROOT / ".agent" / "decisions" / "ADR-002-state-layout.md",
    ]
    for p in must_exist:
        if not p.exists():
            errors.append(f"missing required M3 artifact: {p.relative_to(ROOT)}")

    # Transfer audit must show zero post-init transfers
    audit = M3_ART / "transfer_audit.json"
    if audit.exists():
        try:
            j = json.loads(audit.read_text())
            h2d = j.get("host_to_device_bytes_post_init")
            d2h = j.get("device_to_host_bytes_post_init")
            iters = j.get("iterations")
            if h2d not in (0, "0"):
                errors.append(f"transfer_audit.json host_to_device_bytes_post_init={h2d}, must be 0")
            if d2h not in (0, "0"):
                errors.append(f"transfer_audit.json device_to_host_bytes_post_init={d2h}, must be 0")
            if iters is None or int(iters) < 1000:
                errors.append(f"transfer_audit.json iterations={iters}, must be >=1000")
        except (json.JSONDecodeError, ValueError) as e:
            errors.append(f"transfer_audit.json malformed: {e}")

    # Spacetime budget must have all 6 required keys
    sb = M3_ART / "spacetime_budget.json"
    if sb.exists():
        try:
            j = json.loads(sb.read_text())
            for key in (
                "state_bytes",
                "tendency_bytes",
                "temporary_bytes_per_step",
                "total_persistent_bytes",
                "kernel_launches_per_step",
                "wall_time_per_step_us",
            ):
                if key not in j:
                    errors.append(f"spacetime_budget.json missing required key: {key}")
        except json.JSONDecodeError as e:
            errors.append(f"spacetime_budget.json malformed: {e}")

    # ADR-002 must exist with required tokens
    adr = ROOT / ".agent" / "decisions" / "ADR-002-state-layout.md"
    if adr.exists():
        text = adr.read_text(errors="replace")
        if len(text) < 1500:
            errors.append(f"ADR-002 too short ({len(text)} bytes; expected >=1500)")
        for token in ("Decision:", "Layout:", "Staggering:", "Halo packing:"):
            if token not in text:
                errors.append(f"ADR-002 missing required token: {token!r}")


def check_milestone_closeout(errors: list[str]) -> None:
    closeout = ROOT / ".agent" / "decisions" / "MILESTONE-M3-CLOSEOUT.md"
    if not closeout.exists():
        errors.append("missing .agent/decisions/MILESTONE-M3-CLOSEOUT.md")
    plan = ROOT / ".agent" / "milestones" / "M3-gpu-state-grid.md"
    if plan.exists():
        text = plan.read_text()
        if not re.search(r"Reviewer Decision\s*[:\n]\s*Accepted", text):
            errors.append("M3-gpu-state-grid.md Reviewer Decision is not 'Accepted'")


def check_cross_ai_provenance(errors: list[str]) -> None:
    for d in sorted(glob.glob(SPRINTS_GLOB)):
        tr = Path(d) / "tester-report.md"
        if not tr.exists():
            continue
        logs = sorted(glob.glob(str(ROOT / "logs" / f"{Path(d).name}-tester-*.log")))
        if not logs:
            errors.append(f"sprint {Path(d).name}: no tester log; cannot verify cross-AI provenance")
            continue
        text = Path(logs[-1]).read_text(errors="replace")
        if "ai=claude" not in text:
            errors.append(f"sprint {Path(d).name}: tester log missing ai=claude (cross-AI verification)")


def main() -> int:
    errors: list[str] = []
    check_repo_hygiene(errors)
    check_prior_milestones(errors)
    closed = check_sprints_closed(errors)
    check_m3_artifacts(errors)
    check_milestone_closeout(errors)
    check_cross_ai_provenance(errors)

    result = {"ok": not errors, "errors": errors, "sprints_closed": closed}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
