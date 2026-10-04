"""Shared checking primitives for the M0 validators (contract §14).

Design rule taken straight from §12: **missing evidence is never PASS.** A
validator that cannot find the object it validates reports `MISSING` and exits
non-zero; it never degrades to a warning, and it never treats an absent field as
satisfied. The independent GPT critic reruns these, so every check has to be
recomputable from the committed artifact alone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
MISSING = "MISSING"
NOT_APPLICABLE = "NOT_APPLICABLE"


class Report:
    """Accumulates named checks and renders a machine-readable verdict."""

    def __init__(self, subject: str, path: Path) -> None:
        self.subject = subject
        self.path = path
        self.checks: list[dict] = []

    def add(self, name: str, status: str, detail: str = "", **extra) -> None:
        self.checks.append({"check": name, "status": status, "detail": detail, **extra})

    def require(
        self, name: str, ok: bool, detail: str = "", *, pass_detail: str | None = None, **extra
    ) -> bool:
        """Record one check.

        ``detail`` is the FAIL explanation. ``pass_detail`` is what a PASS says.

        Splitting the two is not cosmetic. The original signature attached one
        string to both outcomes, so a check whose detail was phrased as a failure
        ("no current island site was measured, so minimality is unproven")
        printed that sentence *next to a PASS* -- proof output that asserts the
        opposite of its own verdict. A critic reading the JSON has no way to tell
        which half to believe.

        Value-style details ("0.9812 vs >= 0.95", "missing=[]") read correctly in
        both outcomes and can still be passed as ``detail`` alone; ``pass_detail``
        defaults to them.
        """
        self.add(
            name,
            PASS if ok else FAIL,
            (pass_detail if (ok and pass_detail is not None) else detail),
            **extra,
        )
        return ok

    def missing(self, name: str, detail: str) -> None:
        self.add(name, MISSING, detail)

    @property
    def status(self) -> str:
        statuses = {c["status"] for c in self.checks}
        if not self.checks:
            return MISSING
        if MISSING in statuses:
            return MISSING
        if FAIL in statuses:
            return FAIL
        return PASS

    def render(self) -> dict:
        return {
            "subject": self.subject,
            "path": str(self.path),
            "status": self.status,
            "n_checks": len(self.checks),
            "checks": self.checks,
        }

    def emit(self) -> int:
        print(json.dumps(self.render(), indent=2, sort_keys=True))
        return 0 if self.status == PASS else 1


def load_json(path: Path, report: Report) -> dict | None:
    """Load a proof object, or record MISSING and return None."""
    if not path.exists():
        report.missing(
            "object_exists",
            f"{path} does not exist. Contract §12: missing evidence is never PASS. "
            "If the producing gate has not run yet, PROOF_MANIFEST must carry it as "
            "MISSING and M0 is not acceptable.",
        )
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        report.add("object_parses", FAIL, f"{path} is not valid JSON: {exc}")
        return None
    report.add("object_exists", PASS, str(path))
    return data


def check_keys(report: Report, obj: dict, keys: list[str], where: str = "") -> bool:
    prefix = f"{where}." if where else ""
    ok = True
    for key in keys:
        present = key in obj and obj[key] is not None
        report.require(f"has:{prefix}{key}", present, "" if present else "absent or null")
        ok = ok and present
    return ok


def close_enough(a: float, b: float, *, rel: float = 1e-9) -> bool:
    if a is None or b is None:
        return False
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-12)
