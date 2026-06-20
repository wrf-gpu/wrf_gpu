#!/usr/bin/env python3
"""Single-command oracle for M1 completion.

Used by Claude Code `/goal` evaluator and by the manager loop. Prints a JSON
record with `ok` (bool) and `errors` (list of strings). Exit code 0 if `ok`,
1 otherwise.

The exact assertions are documented in `.agent/goals/M1-DONE.md`.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPRINTS_GLOB = str(ROOT / ".agent" / "sprints" / "2026-*-m1-*")
MANIFESTS_DIR = ROOT / "fixtures" / "manifests"


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


def check_sprints_closed(errors: list[str]) -> int:
    sprint_dirs = sorted(glob.glob(SPRINTS_GLOB))
    if not sprint_dirs:
        errors.append("no M1 sprint folders found")
        return 0
    closed = 0
    for d in sprint_dirs:
        rc, out, _ = _run([sys.executable, "scripts/close_sprint.py", d])
        if rc != 0:
            errors.append(f"sprint {Path(d).name} not closed: {out.strip()}")
        else:
            closed += 1
    return closed


def _manifests_with_source(source: str) -> list[Path]:
    out: list[Path] = []
    if not MANIFESTS_DIR.exists():
        return out
    for p in sorted(MANIFESTS_DIR.glob("*.yaml")):
        if p.name in {"schema.yaml", "fixture-manifest-template.yaml"}:
            continue
        try:
            text = p.read_text()
        except OSError:
            continue
        if f"source: {source}" in text or f"source: '{source}'" in text or f'source: "{source}"' in text:
            out.append(p)
    return out


def check_proof_objects(errors: list[str]) -> None:
    must_exist = [
        ROOT / "fixtures" / "manifests" / "schema.yaml",
        ROOT / "fixtures" / "manifests" / "fixture-manifest-template.yaml",
        ROOT / "docs" / "fixture-storage-policy.md",
        ROOT / "src" / "gpuwrf" / "validation" / "compare_fixture.py",
        ROOT / "scripts" / "validate_fixture_manifest.py",
    ]
    for p in must_exist:
        if not p.exists():
            errors.append(f"missing required artifact: {p.relative_to(ROOT)}")

    # JSON-Schema mirror is recommended but not blocking.

    # Manifest validation: only check that the template validates if the validator exists.
    validator = ROOT / "scripts" / "validate_fixture_manifest.py"
    template = ROOT / "fixtures" / "manifests" / "fixture-manifest-template.yaml"
    if validator.exists() and template.exists():
        rc, _, err = _run([sys.executable, str(validator), str(template)])
        if rc != 0:
            errors.append(f"template manifest fails its own validator: {err.strip()[:300]}")

    # CLI help.
    cli = ROOT / "src" / "gpuwrf" / "validation" / "compare_fixture.py"
    if cli.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-m", "gpuwrf.validation.compare_fixture", "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        if proc.returncode != 0:
            errors.append(f"compare_fixture --help exited {proc.returncode}: {(proc.stderr or proc.stdout)[:300]}")

    # Fixture coverage.
    analytic_stencil = _manifests_with_source("analytic")
    wrf_derived = _manifests_with_source("wrf-derived")
    if len(analytic_stencil) < 2:
        errors.append(
            f"need at least one analytic stencil + one analytic column fixture manifest; "
            f"found {len(analytic_stencil)} analytic manifests"
        )
    if len(wrf_derived) < 1:
        errors.append(f"need at least one wrf-derived (Canary) fixture manifest; found {len(wrf_derived)}")


def check_milestone_closeout(errors: list[str]) -> None:
    closeout = ROOT / ".agent" / "decisions" / "MILESTONE-M1-CLOSEOUT.md"
    if not closeout.exists():
        errors.append("missing .agent/decisions/MILESTONE-M1-CLOSEOUT.md")
    plan = ROOT / ".agent" / "milestones" / "M1-wrf-oracle-fixtures-plan.md"
    if plan.exists():
        text = plan.read_text()
        if "Reviewer Decision\n\nAccepted" not in text and "Reviewer Decision\nAccepted" not in text and "Reviewer Decision: Accepted" not in text:
            errors.append("M1-wrf-oracle-fixtures-plan.md Reviewer Decision is not 'Accepted'")


def check_storage_health(errors: list[str]) -> None:
    data = ROOT / "data"
    if not data.exists() or not data.is_symlink():
        errors.append("./data is not a symlink — fixture storage not bootstrapped")
    else:
        target = os.readlink(data)
        if not target.startswith("<DATA_ROOT>"):
            errors.append(f"./data symlink does not point at <DATA_ROOT>: {target}")
    # disk space
    try:
        st = os.statvfs("<DATA_ROOT>")
        free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
        if free_gb < 50:
            errors.append(f"<DATA_ROOT> has only {free_gb:.1f} GB free (<50)")
    except OSError as e:
        errors.append(f"cannot statvfs <DATA_ROOT>: {e}")


def main() -> int:
    errors: list[str] = []
    check_repo_hygiene(errors)
    check_storage_health(errors)
    closed = check_sprints_closed(errors)
    check_proof_objects(errors)
    check_milestone_closeout(errors)

    result = {
        "ok": not errors,
        "errors": errors,
        "sprints_closed": closed,
        "manifest_dir": str(MANIFESTS_DIR.relative_to(ROOT)),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
