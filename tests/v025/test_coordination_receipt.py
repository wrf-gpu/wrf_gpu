"""The coordination receipt's `verbatim` fields must actually be verbatim (§13).

An earlier revision of `gpu_coordination_receipt.json` abridged `0:2`'s reply —
dropping the "WARUM ERST DANN" rationale and the abort-waiver paragraph — and
normalised its punctuation, while still calling the field `verbatim`. That is a
defect in a proof object rather than a formatting preference: a reviewer
reconstructing whether consent was actually given would have been reading a
summary written by the party the consent was granted to.

`GPU_COORDINATION_LOG.md` carries the replies as markdown blockquotes and is the
human-readable record; the JSON is the machine-readable one. These tests hold the
two to each other, so neither can be edited into a flattering shape alone.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPRINT = REPO / ".agent/sprints/2026-07-27-v0250-m0-setup"
RECEIPT = SPRINT / "gpu_coordination_receipt.json"
LOG = SPRINT / "GPU_COORDINATION_LOG.md"

if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import run_gpu_arm as rga  # noqa: E402

pytestmark = pytest.mark.skipif(
    not RECEIPT.exists(), reason="no coordination receipt yet (no window requested)"
)


def _normalise(text: str) -> str:
    """Collapse whitespace only. Punctuation is NOT normalised on purpose.

    Normalising dashes or quotes here would hide exactly the drift these tests
    exist to catch — an em dash silently rewritten to a hyphen is the first step
    of a reply being paraphrased.
    """
    return re.sub(r"\s+", " ", text).strip()


def _log_text() -> str:
    """The log with its markdown blockquote markers stripped."""
    lines = []
    for line in LOG.read_text().splitlines():
        stripped = line.strip()
        lines.append(stripped[1:].strip() if stripped.startswith(">") else stripped)
    return _normalise(" ".join(lines))


@pytest.mark.parametrize("manager", ["0:2", "0:3"])
def test_every_verbatim_reply_appears_in_the_log(manager):
    """The JSON's verbatim text must be present in the log, unabridged."""
    receipt = json.loads(RECEIPT.read_text())
    quoted = _normalise(receipt["replies"][manager]["verbatim"])
    assert quoted, f"{manager} has an empty verbatim field"
    assert quoted in _log_text(), (
        f"{manager}'s verbatim text is not present in GPU_COORDINATION_LOG.md. "
        "Either the receipt was abridged/paraphrased, or the log was edited away from it."
    )


def test_the_attempt_1_paragraphs_survive_in_the_permanent_log():
    """Regression: these paragraphs were abridged out of the receipt once.

    They are the substantive ones — the rationale for waiting for the seed
    boundary, and the manager's attempt to waive the abort commitment. A summary
    that keeps only the "yes" and drops the reasoning is the failure mode.

    Checked against the LOG rather than the receipt: the receipt holds only the
    CURRENT window's coordination and is legitimately replaced each time, while
    the log is append-only and is where history has to survive.
    """
    # Whitespace-normalised: the log wraps its blockquotes, so a phrase that
    # spans a line break is not a contiguous substring of the raw file. Searching
    # the raw text found "180 trainierte Epochen" but not "...wegwerfen" and
    # reported the content missing when it was present.
    log = _log_text()
    assert "WARUM ERST DANN" in log
    assert "ZU DEINER ZUSAGE" in log
    assert "180 trainierte Epochen wegwerfen" in log
    assert "EINE BITTE" in log


def test_punctuation_is_not_normalised():
    """Em dashes as sent must survive; rewriting them is paraphrase creep."""
    receipt = json.loads(RECEIPT.read_text())
    assert "—" in receipt["replies"]["0:2"]["verbatim"]


def test_receipt_is_now_spent_and_refused():
    """This assertion was inverted deliberately.

    It originally read `check("baseline-census")` with no `pytest.raises` — i.e.
    it asserted the receipt still *passes*. That was correct when written and
    became wrong the moment the window was used and invalidated. A receipt
    authorises one window; after W1 attempt 1 this one is dead, and the test now
    pins that rather than the state it had on the day it was written.

    The lifecycle itself is covered in `test_receipt_spending.py`.
    """
    with pytest.raises(rga.WindowNotAuthorised, match="SPENT"):
        rga.CoordinationReceipt.load(RECEIPT).check("baseline-census")


def test_both_managers_are_affirmative_with_conditions_recorded():
    receipt = json.loads(RECEIPT.read_text())
    for manager in rga.REQUIRED_MANAGERS:
        entry = receipt["replies"][manager]
        assert entry["affirmative"] is True
        assert entry["conditions"], f"{manager}'s attached conditions were not recorded"


def test_the_mechanism_conflict_is_recorded_not_silently_resolved():
    """0:3 said poll; 0:2 said wait for a signal. The record must say which won and why.

    Held against the append-only log: the conflict was an attempt-1 event and
    must stay on record even though attempt 2's receipt has no reason to repeat
    it.
    """
    log = _log_text().lower()
    assert "per seed" in log and "0:2" in log
    assert "free lock does not imply a safe boundary" in log or \
           "a free lock is not a safe boundary" in log


def test_the_abort_commitment_is_recorded_as_kept_despite_the_waiver():
    """0:2 tried to waive it in attempt 1; it was kept. That must stay on record."""
    log = _log_text()
    assert "waived it" in log, "the waiver attempt is not on record"
    assert "stands regardless" in log or "not theirs to waive" in log, (
        "the log does not record that the commitment was kept despite the waiver"
    )
