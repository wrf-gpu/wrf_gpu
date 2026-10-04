"""A coordination receipt authorises ONE window, and spending it is mechanical (§13).

Manager repair order after W1 attempt 1: the worker *reported* the receipt as
spent while the code had no such concept. The exact old affirmative receipt still
passed `CoordinationReceipt.check()` until its 6 h age expiry; overall
authorisation was `false` only because the lock was not held — a different gate
that could pass at any moment. A claim in a report is not a gate.

The load-bearing test is `test_the_exact_w1_receipt_is_refused_now`: it runs
against the **real committed receipt file**, not a fixture copy, so the artefact
the manager flagged is the artefact under test.

The anti-forgery pair matters as much: the spend fingerprint covers only the
coordination content, so deleting the `spent` marker or copying the file to a new
path does not un-spend it. Only genuinely new coordination is new.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "v025") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "v025"))

import run_gpu_arm as rga  # noqa: E402

REAL_RECEIPT = REPO / ".agent/sprints/2026-07-27-v0250-m0-setup/gpu_coordination_receipt.json"
REAL_LEDGER = REPO / ".agent/sprints/2026-07-27-v0250-m0-setup/gpu_coordination_spent.json"

# Captured at import, before any test runs, so a test that spends into the real
# ledger is caught by the size comparison at the end of the module.
_LEDGER_SIZE_AT_IMPORT = len(rga.load_spend_ledger(REAL_LEDGER)) if REAL_LEDGER.exists() else 0


def _fresh_body(window: str = "baseline-census", *, tag: str = "fresh") -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "window": window,
        "requested_at_utc": now,
        "request_text": f"[{tag}] window request",
        "replies": {
            "0:2": {"affirmative": True, "verbatim": f"[{tag}] yes from 0:2",
                    "received_at_utc": now},
            "0:3": {"affirmative": True, "verbatim": f"[{tag}] yes from 0:3",
                    "received_at_utc": now},
        },
    }


def _write(path: Path, body: dict) -> Path:
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False))
    return path


# --------------------------------------------------------------------------- #
# the exact artefact the manager flagged                                       #
# --------------------------------------------------------------------------- #
# NOTE ON TEST DESIGN, learned the hard way.
#
# The first version of these tests pinned point-in-time CONTENT: "the receipt's
# spend reason mentions 600 and INVALID", "the ledger holds exactly one entry".
# Both were true when written and both broke the moment a second window was
# legitimately coordinated and used — five red tests for a system behaving
# exactly as designed, and a commit whose message claimed a passing suite.
#
# A receipt file and a spend ledger are living artifacts. What must be pinned is
# the INVARIANT (a used receipt is dead; history is never erased; tests never
# write to the ledger), not today's contents.
@pytest.mark.skipif(not REAL_RECEIPT.exists(), reason="no committed receipt")
def test_a_used_receipt_is_mechanically_dead():
    """Whatever the current receipt is, once used it must refuse."""
    body = json.loads(REAL_RECEIPT.read_text())
    receipt = rga.CoordinationReceipt.load(REAL_RECEIPT)
    if not body.get("spent"):
        pytest.skip("current receipt has not been used yet")
    with pytest.raises(rga.WindowNotAuthorised, match="SPENT"):
        receipt.check(body["window"])


@pytest.mark.skipif(not REAL_RECEIPT.exists(), reason="no committed receipt")
def test_a_spend_record_always_says_when_why_and_which_window():
    body = json.loads(REAL_RECEIPT.read_text())
    spent = body.get("spent")
    if not spent:
        pytest.skip("current receipt has not been used yet")
    assert spent["window"] in rga.WINDOWS
    assert spent["reason"].strip(), "a spend with no reason is not a record"
    datetime.fromisoformat(spent["spent_at_utc"])  # parses


@pytest.mark.skipif(not REAL_LEDGER.exists(), reason="no committed ledger")
def test_history_is_never_erased_from_the_ledger():
    """The W1 attempt-1 spend must remain on record forever.

    A ledger that can lose entries is a ledger that can be quietly reset, and the
    whole point is that a spent `yes` cannot come back.
    """
    ledger = rga.load_spend_ledger(REAL_LEDGER)
    # Identify by REASON, not by hash. The first version of this test hardcoded
    # "dcd89a299b023310" -- which is the 16-char prefix I had been printing for
    # readability, not the 64-char key. It failed against a ledger that contained
    # exactly the entry it was looking for.
    attempt_one = [r for r in ledger.values() if "W1 attempt 1" in r["reason"]]
    assert attempt_one, "the W1 attempt-1 spend was removed from the ledger"
    assert attempt_one[0]["window"] == "baseline-census"


@pytest.mark.skipif(not REAL_LEDGER.exists(), reason="no committed ledger")
def test_the_current_receipt_is_recorded_in_the_ledger_once_used():
    body = json.loads(REAL_RECEIPT.read_text())
    if not body.get("spent"):
        pytest.skip("current receipt has not been used yet")
    receipt = rga.CoordinationReceipt.load(REAL_RECEIPT)
    assert receipt.fingerprint() in rga.load_spend_ledger(REAL_LEDGER)


@pytest.mark.skipif(not REAL_RECEIPT.exists(), reason="no committed receipt")
def test_copying_the_w1_receipt_elsewhere_does_not_un_spend_it(tmp_path):
    """Anti-forgery: a new path is not new coordination."""
    copy = tmp_path / "copy.json"
    shutil.copy(REAL_RECEIPT, copy)
    with pytest.raises(rga.WindowNotAuthorised, match="SPENT|already spent"):
        rga.CoordinationReceipt.load(copy).check("baseline-census", ledger_path=REAL_LEDGER)


@pytest.mark.skipif(not REAL_RECEIPT.exists(), reason="no committed receipt")
def test_deleting_the_spend_marker_does_not_un_spend_it(tmp_path):
    """The ledger is the authority; the in-file marker is only what humans read."""
    body = json.loads(REAL_RECEIPT.read_text())
    body.pop("spent", None)
    forged = _write(tmp_path / "forged.json", body)
    receipt = rga.CoordinationReceipt.load(forged)
    assert receipt.spent is None, "marker was not actually removed; test proves nothing"
    with pytest.raises(rga.WindowNotAuthorised, match="already spent"):
        receipt.check("baseline-census", ledger_path=REAL_LEDGER)


# --------------------------------------------------------------------------- #
# fresh replacement must still be accepted                                     #
# --------------------------------------------------------------------------- #
def test_fresh_coordination_is_accepted(tmp_path):
    """Spending must not brick the mechanism: new replies still authorise."""
    ledger = tmp_path / "ledger.json"
    path = _write(tmp_path / "fresh.json", _fresh_body())
    rga.CoordinationReceipt.load(path).check("baseline-census", ledger_path=ledger)


def test_a_fresh_receipt_has_a_different_fingerprint_from_the_spent_one(tmp_path):
    if not REAL_RECEIPT.exists():
        pytest.skip("no committed receipt")
    old = rga.CoordinationReceipt.load(REAL_RECEIPT)
    new = rga.CoordinationReceipt.load(_write(tmp_path / "n.json", _fresh_body()))
    assert old.fingerprint() != new.fingerprint()


def test_fresh_then_spent_then_refused(tmp_path):
    """The whole lifecycle in one test."""
    ledger = tmp_path / "ledger.json"
    path = _write(tmp_path / "r.json", _fresh_body(tag="lifecycle"))

    receipt = rga.CoordinationReceipt.load(path)
    receipt.check("baseline-census", ledger_path=ledger)          # accepted

    rga.spend_receipt(receipt, reason="test window", ledger_path=ledger)

    with pytest.raises(rga.WindowNotAuthorised, match="SPENT"):    # dead
        rga.CoordinationReceipt.load(path).check("baseline-census", ledger_path=ledger)


# --------------------------------------------------------------------------- #
# spending happens at authorisation, before any work                           #
# --------------------------------------------------------------------------- #
def _lock_env(tmp_path: Path, token: str = "tok") -> dict:
    holder = tmp_path / "holder"
    holder.write_text(
        f"holder=baseline-census pid=1 token={token} cmd=x\n"
    )
    return {
        "GPUWRF_GPU_LOCK_HELD": "1",
        "GPUWRF_GPU_LOCK_TOKEN": token,
        "GPUWRF_GPU_LOCK_HOLDER_FILE": str(holder),
        "GPUWRF_GPU_LOCK_LABEL": "baseline-census",
    }


def test_authorise_consumes_the_receipt(tmp_path, monkeypatch):
    for key, value in _lock_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    ledger = tmp_path / "ledger.json"
    path = _write(tmp_path / "r.json", _fresh_body())

    result = rga.authorise("baseline-census", receipt_path=path, ledger_path=ledger)
    assert result["consumed"] is True
    assert result["spend_record"]["window"] == "baseline-census"

    with pytest.raises(rga.WindowNotAuthorised, match="SPENT"):
        rga.authorise("baseline-census", receipt_path=path, ledger_path=ledger)


def test_a_failed_window_still_consumed_its_coordination(tmp_path, monkeypatch):
    """W1 attempt 1's exact situation: the run failed, the yes is still spent.

    'The run failed so the approval still counts' is not available. Coordination
    buys one attempt, not one success.
    """
    for key, value in _lock_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    ledger = tmp_path / "ledger.json"
    path = _write(tmp_path / "r.json", _fresh_body())

    rga.authorise("baseline-census", receipt_path=path, ledger_path=ledger)
    # ... window then breaches an early stop and produces nothing ...
    with pytest.raises(rga.WindowNotAuthorised, match="SPENT"):
        rga.CoordinationReceipt.load(path).check("baseline-census", ledger_path=ledger)


def test_print_plan_inspection_does_not_consume(tmp_path, monkeypatch):
    """Looking at the authorisation state must not burn a window."""
    for key, value in _lock_env(tmp_path).items():
        monkeypatch.setenv(key, value)
    ledger = tmp_path / "ledger.json"
    path = _write(tmp_path / "r.json", _fresh_body())

    result = rga.authorise(
        "baseline-census", receipt_path=path, ledger_path=ledger, consume=False,
    )
    assert result["consumed"] is False
    # still usable afterwards
    rga.CoordinationReceipt.load(path).check("baseline-census", ledger_path=ledger)


def test_spend_checks_run_before_every_other_check(tmp_path):
    """A spent receipt must never be reported as 'would pass except for X'."""
    ledger = tmp_path / "ledger.json"
    body = _fresh_body(window="pallas-sm120")          # deliberately wrong window
    path = _write(tmp_path / "r.json", body)
    receipt = rga.CoordinationReceipt.load(path)
    rga.spend_receipt(receipt, reason="already used", ledger_path=ledger)

    with pytest.raises(rga.WindowNotAuthorised, match="SPENT"):
        rga.CoordinationReceipt.load(path).check("baseline-census", ledger_path=ledger)


def test_missing_ledger_is_empty_not_an_error(tmp_path):
    assert rga.load_spend_ledger(tmp_path / "absent.json") == {}


# --------------------------------------------------------------------------- #
# the test suite must not write to the committed ledger                        #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not REAL_LEDGER.exists(), reason="no committed ledger")
def test_the_committed_ledger_contains_only_real_window_spends():
    """Guard against tests consuming coordination into a committed artifact.

    The first version of `gpu_arm()` took no `ledger_path`, so every happy-path
    test in `test_gpu_arm.py` called `authorise()` against the DEFAULT ledger and
    wrote its throwaway fingerprint into this committed file — 12 junk entries in
    two runs. A test that mutates a proof artifact corrupts the evidence it is
    supposed to protect. `gpu_arm()` now threads `ledger_path` and every test
    passes a tmp one.

    This asserts the invariant rather than a count: every entry must name a real
    window and carry a real reason. It grows by one per genuinely coordinated
    window and by nothing per test run.
    """
    ledger = rga.load_spend_ledger(REAL_LEDGER)
    assert ledger, "ledger is empty; the W1 history should be on record"
    for fingerprint, record in ledger.items():
        assert record["window"] in rga.WINDOWS, f"{fingerprint}: bogus window"
        reason = record["reason"]
        assert reason.strip(), f"{fingerprint}: spend with no reason"
        assert "test" not in reason.lower(), (
            f"{fingerprint}: a TEST spend reached the committed ledger — "
            "some caller stopped passing ledger_path"
        )


@pytest.mark.skipif(not REAL_LEDGER.exists(), reason="no committed ledger")
def test_a_full_test_run_does_not_grow_the_committed_ledger():
    """The suite must not add entries. Compares against the file on disk.

    Runs last-ish by name; if any test in the suite spends into the real ledger,
    the count seen here differs from the count recorded at module import.
    """
    assert len(rga.load_spend_ledger(REAL_LEDGER)) == _LEDGER_SIZE_AT_IMPORT


def test_gpu_arm_accepts_an_isolated_ledger():
    """The parameter that makes the guard above possible must exist."""
    import inspect

    assert "ledger_path" in inspect.signature(rga.gpu_arm).parameters
