#!/usr/bin/env python3
"""Single-command oracle for M4 (Minimal Dycore) completion.

Mirrors check_m1/m2/m3_done.py. Prints JSON `{ok, errors, sprints_closed}`.
Exit 0 if ok, 1 otherwise.

Assertions documented in .agent/goals/M4-DONE.md.
"""

from __future__ import annotations

import glob
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPRINTS_GLOB = str(ROOT / ".agent" / "sprints" / "2026-*-m4-*")
M4_ART = ROOT / "artifacts" / "m4"


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode, proc.stdout, proc.stderr


def check_repo_hygiene(errors: list[str]) -> None:
    rc, _, _ = _run([sys.executable, "scripts/validate_agentos.py"])
    if rc != 0:
        errors.append(f"validate_agentos.py exited {rc}")
    rc, out, err = _run([sys.executable, "-m", "pytest", "-q"])
    if rc != 0:
        errors.append(f"pytest -q exited {rc}: {(err or out).splitlines()[-5:]}")


def check_prior_milestones(errors: list[str]) -> None:
    for prior in ("check_m1_done.py", "check_m2_done.py", "check_m3_done.py"):
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
        errors.append("no M4 sprint folders found")
        return 0
    closed = 0
    for d in sprint_dirs:
        rc, out, _ = _run([sys.executable, "scripts/close_sprint.py", d])
        if rc != 0:
            errors.append(f"sprint {Path(d).name} not closed: {out.strip()[:300]}")
        else:
            closed += 1
    return closed


def _check_json_pass(errors: list[str], path: Path, required_keys: tuple[str, ...], pass_field: str = "pass") -> None:
    if not path.exists():
        errors.append(f"missing required M4 artifact: {path.relative_to(ROOT)}")
        return
    try:
        j = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        errors.append(f"{path.relative_to(ROOT)} malformed: {e}")
        return
    for k in required_keys:
        if k not in j:
            errors.append(f"{path.relative_to(ROOT)} missing key: {k}")
    if pass_field in required_keys and j.get(pass_field) is not True:
        errors.append(f"{path.relative_to(ROOT)} {pass_field}={j.get(pass_field)} (must be true)")


def check_m4_artifacts(errors: list[str]) -> None:
    must_exist = [
        ROOT / "src" / "gpuwrf" / "dynamics" / "__init__.py",
        ROOT / "src" / "gpuwrf" / "dynamics" / "rk3.py",
        ROOT / "src" / "gpuwrf" / "dynamics" / "advection.py",
        ROOT / "src" / "gpuwrf" / "dynamics" / "acoustic.py",
        ROOT / "src" / "gpuwrf" / "dynamics" / "step.py",
        ROOT / "src" / "gpuwrf" / "debug" / "__init__.py",
        ROOT / "src" / "gpuwrf" / "debug" / "asserts.py",
        ROOT / "src" / "gpuwrf" / "debug" / "snapshots.py",
        M4_ART / "dycore_profile.json",
        M4_ART / "m5_gate_dryrun.json",
        M4_ART / "tier1_advection_parity.json",
        M4_ART / "tier2_invariants.json",
        M4_ART / "tier3_convergence.json",
        M4_ART / "transfer_audit.json",
        M4_ART / "spacetime_budget.json",
        M4_ART / "hlo_dump" / "dycore_step_production.txt",
        M4_ART / "hlo_dump" / "dycore_step_debug_stripped.txt",
        M4_ART / "hlo_dump" / "dycore_step_debug_vs_stripped.diff",
        ROOT / ".agent" / "decisions" / "ADR-003-dycore-precision.md",
    ]
    for p in must_exist:
        if not p.exists():
            errors.append(f"missing required M4 artifact: {p.relative_to(ROOT)}")

    # Transfer audit: zero post-init bytes
    audit = M4_ART / "transfer_audit.json"
    if audit.exists():
        try:
            j = json.loads(audit.read_text())
            if j.get("host_to_device_bytes_post_init") not in (0, "0"):
                errors.append(f"transfer_audit.json host_to_device_bytes_post_init={j.get('host_to_device_bytes_post_init')}, must be 0")
            if j.get("device_to_host_bytes_post_init") not in (0, "0"):
                errors.append(f"transfer_audit.json device_to_host_bytes_post_init={j.get('device_to_host_bytes_post_init')}, must be 0")
            if int(j.get("iterations") or 0) < 100:
                errors.append(f"transfer_audit.json iterations={j.get('iterations')}, must be >=100")
        except (json.JSONDecodeError, ValueError) as e:
            errors.append(f"transfer_audit.json malformed: {e}")

    # Spacetime budget: temporary_bytes_per_step == 0
    sb = M4_ART / "spacetime_budget.json"
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
            if j.get("temporary_bytes_per_step") not in (0, "0"):
                errors.append(f"spacetime_budget.json temporary_bytes_per_step={j.get('temporary_bytes_per_step')}, must be 0")
        except json.JSONDecodeError as e:
            errors.append(f"spacetime_budget.json malformed: {e}")

    # Validation tiers must pass
    _check_json_pass(errors, M4_ART / "tier1_advection_parity.json", ("max_abs_err", "max_rel_err", "pass"))
    _check_json_pass(errors, M4_ART / "tier2_invariants.json", ("mass_residual_relative", "qv_positivity_violations", "nan_inf_violations", "pass"))
    _check_json_pass(errors, M4_ART / "tier3_convergence.json", ("observed_order", "expected_order", "pass"))

    # M5 gate dry-run: must exist, must have required keys. Trip is not a failure (per ADR-001).
    _check_json_pass(errors, M4_ART / "m5_gate_dryrun.json", ("kernel_launches_per_step", "local_memory_bytes_per_kernel", "registers_per_kernel", "gate_status"), pass_field="")

    # HLO debug-vs-stripped diff MUST be empty (debuggability contract)
    diff = M4_ART / "hlo_dump" / "dycore_step_debug_vs_stripped.diff"
    if diff.exists():
        sz = diff.stat().st_size
        if sz > 0:
            errors.append(f"HLO debug-vs-stripped diff is non-empty ({sz}B); debug branch leaking into production. Constitutional debuggability gate failed.")

    # ADR-003 must exist with required tokens
    adr = ROOT / ".agent" / "decisions" / "ADR-003-dycore-precision.md"
    if adr.exists():
        text = adr.read_text(errors="replace")
        if len(text) < 1500:
            errors.append(f"ADR-003 too short ({len(text)} bytes; expected >=1500)")
        for token in ("Decision:", "Per-field precision:", "Downcast plan:", "Validation evidence:"):
            if token not in text:
                errors.append(f"ADR-003 missing required token: {token!r}")


def check_milestone_closeout(errors: list[str]) -> None:
    closeout = ROOT / ".agent" / "decisions" / "MILESTONE-M4-CLOSEOUT.md"
    if not closeout.exists():
        errors.append("missing .agent/decisions/MILESTONE-M4-CLOSEOUT.md")
    plan = ROOT / ".agent" / "milestones" / "M4-minimal-dycore.md"
    if plan.exists():
        text = plan.read_text()
        if not re.search(r"Reviewer Decision\s*[:\n]\s*Accepted", text):
            errors.append("M4-minimal-dycore.md Reviewer Decision is not 'Accepted'")


def check_cross_ai_provenance(errors: list[str]) -> None:
    for d in sorted(glob.glob(SPRINTS_GLOB)):
        tr = Path(d) / "tester-report.md"
        if not tr.exists():
            continue
        tr_text = tr.read_text(errors="replace")
        if (
            "Tester role explicitly waived" in tr_text
            or "Decision: waived" in tr_text
            or "Decision: not applicable" in tr_text
        ):
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
    check_m4_artifacts(errors)
    check_milestone_closeout(errors)
    check_cross_ai_provenance(errors)

    result = {"ok": not errors, "errors": errors, "sprints_closed": closed}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
