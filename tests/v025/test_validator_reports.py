"""Proof output must not contradict its own verdict (manager decision `main e5df2364`).

`Report.require` originally attached one detail string to both outcomes, so a
check whose detail was phrased as a failure printed that sentence next to a
PASS. `islands:minimality_was_tested` PASSed while stating "no current island
site was measured, so minimality is unproven" — the opposite of what it had just
verified. A critic reading the JSON cannot tell which half to believe.

These tests pin the fix and, more importantly, guard the whole CLASS: the last
test scans every PASS detail the real validator emits against the real proof
object and rejects failure-phrasing, so a future check with the same flaw fails
here rather than in front of a reviewer.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

from _validate_lib import PASS, Report  # noqa: E402

MAP_PATH = REPO / "proofs" / "v025" / "m0" / "cancellation_map.json"
VALIDATOR = REPO / "scripts" / "v025" / "validate_cancellation_map.py"


# --------------------------------------------------------------------------- #
# Report.require                                                               #
# --------------------------------------------------------------------------- #
def test_pass_detail_is_used_on_pass():
    rep = Report("t", Path("x"))
    rep.require("c", True, "it failed because X", pass_detail="8 sites measured")
    assert rep.checks[0]["status"] == PASS
    assert rep.checks[0]["detail"] == "8 sites measured"


def test_fail_detail_is_used_on_fail():
    rep = Report("t", Path("x"))
    rep.require("c", False, "it failed because X", pass_detail="8 sites measured")
    assert rep.checks[0]["status"] == "FAIL"
    assert rep.checks[0]["detail"] == "it failed because X"


def test_value_style_detail_still_works_without_pass_detail():
    """Details like '0.98 vs >= 0.95' read correctly either way and must survive."""
    rep = Report("t", Path("x"))
    rep.require("c", True, "0.98 vs >= 0.95")
    assert rep.checks[0]["detail"] == "0.98 vs >= 0.95"


def test_require_still_returns_the_boolean():
    rep = Report("t", Path("x"))
    assert rep.require("a", True, "d") is True
    assert rep.require("b", False, "d") is False


# --------------------------------------------------------------------------- #
# the specific check the manager flagged                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not MAP_PATH.exists(), reason="cancellation map not built")
def test_minimality_check_reports_what_it_verified():
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(MAP_PATH), "--require-complete"],
        capture_output=True, text=True, check=False, cwd=REPO,
    )
    report = json.loads(result.stdout)
    checks = {c["check"]: c for c in report["checks"]}
    entry = checks["islands:minimality_was_tested"]
    assert entry["status"] == PASS
    # It must state the measurement it made, not the failure it did not hit.
    assert "unproven" not in entry["detail"]
    assert re.search(r"\d+ current island site", entry["detail"])


# --------------------------------------------------------------------------- #
# the class guard                                                              #
# --------------------------------------------------------------------------- #
# Words that assert absence or failure. A PASS detail containing one of these as
# a standalone word is almost certainly the failure sentence leaking through.
NEGATIVE_PHRASES = (
    r"\bunproven\b",
    r"\bnot supported\b",
    r"\bfails\b",
    r"\bforbidden\b",
    r"\bno \w+ was\b",
    r"\bnever\b",
    r"\bcannot\b",
    r"\bwithout\b",
)

# Details that legitimately contain a negative word while describing a PASS.
# Keep this list short and justified; it is the escape hatch, not the rule.
ALLOWED_NEGATIVE_DETAILS = {
    # empty-list readouts are factual and read correctly in both outcomes
    "missing=[]",
    "invalid: []",
    "unlinked: []",
}


@pytest.mark.skipif(not MAP_PATH.exists(), reason="cancellation map not built")
def test_no_pass_detail_contradicts_its_verdict():
    """Every PASS the real validator emits must describe what it verified."""
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(MAP_PATH), "--require-complete"],
        capture_output=True, text=True, check=False, cwd=REPO,
    )
    assert result.returncode == 0, result.stdout
    report = json.loads(result.stdout)

    offenders = []
    for check in report["checks"]:
        if check["status"] != PASS:
            continue
        detail = (check.get("detail") or "").strip()
        if not detail or detail in ALLOWED_NEGATIVE_DETAILS:
            continue
        for pattern in NEGATIVE_PHRASES:
            if re.search(pattern, detail, flags=re.IGNORECASE):
                offenders.append((check["check"], pattern, detail))
                break
    assert not offenders, (
        "PASS details that read as failures:\n"
        + "\n".join(f"  {name}: /{pat}/ in {detail!r}" for name, pat, detail in offenders)
    )
