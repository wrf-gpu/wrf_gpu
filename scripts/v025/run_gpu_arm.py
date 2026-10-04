#!/usr/bin/env python3
"""The FAST-v025 GPU arm — the injection `run_fast_pair` has been waiting for.

Contract §5.2 (fresh paired arm), §5.3 (every timing class reported separately),
§9 (family-attributed device time), §13 (dual-manager coordination + canonical
lock before any GPU touch).

**This module imports nothing from JAX at module scope and touches no GPU until
`authorise()` has returned.** Every refusal path below is exercised on CPU by
`tests/v025/test_gpu_arm.py`; the `--dry-run` mode runs the entire control flow
against a stub device so the logic is proven before a coordinated window is
spent on it.

## The three gates, all mechanical

1. **Coordination receipt.** A JSON file recording BOTH `0:2` and `0:3` replying
   affirmatively, verbatim, for THIS window label. Contract §13 says silence is
   not approval; this makes that a file that must exist rather than a claim in a
   report. A receipt for a different window, or one missing either manager, is
   refused.
2. **Canonical lock.** `scripts/with_gpu_lock.sh` exports `GPUWRF_GPU_LOCK_HELD`
   and a per-invocation `GPUWRF_GPU_LOCK_TOKEN`, and writes that token into the
   holder file. Checking the token against the holder file proves the lock is
   *actually held by this process tree* — an inherited or hand-set env var does
   not match a live holder file.
3. **Preemption.** SIGINT/SIGTERM mark the in-flight measurement invalid and
   exit non-zero. §13 allows no "finish this short run first" exception, so
   there is no path that returns a result after a preemption signal.

## Why the timing classes are separate objects, not one number

§5.3: "cold compile, cached load, warm integration, CPU integration, I/O,
profiler perturbation, and lock wait ... hiding one inside another invalidates
the gate." `TimingClasses` therefore has one field per class and
`as_dict()` refuses to emit while any required class is still `None`. A run that
forgot to measure one cannot silently report a total.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[2]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_fast_pair import ArmRecord  # noqa: E402
import gpu_window_registry as registry  # noqa: E402

SCHEMA = "wrf_gpu2.v025.m0.gpu_arm.v1"

WINDOWS = registry.WINDOW_LABELS
REQUIRED_MANAGERS = ("0:2", "0:3")

DEFAULT_RECEIPT = (
    REPO / ".agent/sprints/2026-07-27-v0250-m0-setup/gpu_coordination_receipt.json"
)

# A receipt older than this is stale: coordination is for a named window now, not
# a standing permission. Re-asking is cheap; running on a day-old "yes" is not.
RECEIPT_MAX_AGE = timedelta(hours=6)

# Content-addressed ledger of receipts that have already authorised a window.
#
# A receipt authorises ONE window execution. After W1 attempt 1 the worker
# *reported* the receipt as spent while the code had no such concept: the exact
# old affirmative receipt still passed `check()` until its 6 h age expiry, and
# the only thing preventing a GPU touch was not holding the lock -- a different
# gate that could pass at any moment. A claim in a report is not a gate.
#
# The fingerprint deliberately covers only the COORDINATION CONTENT (window,
# request time, replies) and not the spend marker, so deleting `spent` from the
# JSON -- or copying the file elsewhere -- does not un-spend it. Only genuinely
# new coordination produces a new fingerprint.
SPEND_LEDGER = (
    REPO / ".agent/sprints/2026-07-27-v0250-m0-setup/gpu_coordination_spent.json"
)


class WindowNotAuthorised(RuntimeError):
    """Raised when any §13 precondition for touching the GPU is unmet."""


class Preempted(RuntimeError):
    """Raised when a coordinating manager asked for the GPU mid-measurement."""


class ArmIncompleteError(RuntimeError):
    """Raised when a timing class the contract requires was never measured."""


class EarlyStopBreached(RuntimeError):
    """Raised when a frozen §5.3 economy bar is exceeded during the window."""


# Frozen §5.3 economy bars. These are the WINDOW's own early stops, enforced by
# the runner itself.
#
# The first W1 attempt had no internal enforcement at all: the 600 s bar existed
# only in the contract and in a shell-level `timeout` set ABOVE it, so the cold
# compile sailed past 600 s and the MANAGER had to stop it by hand with a signal
# at 655 s. A frozen threshold that depends on a human noticing is not a gate.
# The child is now killed by the runner at the bar, and the breach is reported as
# a measured result rather than as an interrupted run.
COLD_COMPILE_MAX_SECONDS = 600.0
CACHED_LOAD_MAX_SECONDS = 60.0
GPU_ARM_MAX_SECONDS = 300.0


# --------------------------------------------------------------------------- #
# 1. coordination receipt                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CoordinationReceipt:
    """Both managers' verbatim replies for one named window."""

    window: str
    requested_at_utc: str
    replies: dict[str, dict[str, Any]]
    request_text: str = ""
    spent: dict[str, Any] | None = None
    source_path: Path | None = None

    def fingerprint(self) -> str:
        """Content address of the COORDINATION, not of the file.

        Covers window + request time + each manager's affirmative and verbatim
        text. Excludes the spend marker on purpose, so removing `spent` from the
        JSON, or copying the receipt to a new path, yields the same fingerprint
        and is still refused. Only genuinely new coordination is new.
        """
        payload = {
            "window": self.window,
            "requested_at_utc": self.requested_at_utc,
            "replies": {
                manager: {
                    "affirmative": bool(entry.get("affirmative", False)),
                    "verbatim": str(entry.get("verbatim", "")),
                    "received_at_utc": str(entry.get("received_at_utc", "")),
                }
                for manager, entry in sorted(self.replies.items())
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, path: Path) -> "CoordinationReceipt":
        if not path.is_file():
            raise WindowNotAuthorised(
                f"no coordination receipt at {path}. Contract §13 requires affirmative "
                "replies from BOTH 0:2 and 0:3 before any GPU import, query, compile or "
                "run; silence is not approval, so the absence of this file is a NO."
            )
        raw = json.loads(path.read_text())
        return cls(
            window=raw.get("window", ""),
            requested_at_utc=raw.get("requested_at_utc", ""),
            replies=raw.get("replies", {}) or {},
            request_text=raw.get("request_text", ""),
            spent=raw.get("spent"),
            source_path=path,
        )

    def check(
        self,
        window: str,
        *,
        now: datetime | None = None,
        ledger_path: Path | None = None,
    ) -> None:
        """Raise unless this receipt authorises exactly ``window``, right now.

        Spend checks come FIRST. A receipt that has already authorised a window
        is dead regardless of how fresh or how affirmative it is, and checking
        that before anything else means a spent receipt can never be reported as
        "would have passed except for X".
        """
        now = now or datetime.now(timezone.utc)

        if self.spent:
            raise WindowNotAuthorised(
                f"receipt already SPENT at {self.spent.get('spent_at_utc')} "
                f"({self.spent.get('reason')}). A receipt authorises ONE window; "
                "re-coordinate with both managers for a new one."
            )
        ledger = load_spend_ledger(ledger_path)
        entry = ledger.get(self.fingerprint())
        if entry:
            raise WindowNotAuthorised(
                f"this coordination content was already spent at "
                f"{entry.get('spent_at_utc')} ({entry.get('reason')}). The fingerprint "
                "covers window/request-time/replies only, so removing the spend marker "
                "or copying the file does not un-spend it."
            )

        registry.assert_registered(
            window,
            context="receipt requested label",
            exc_type=WindowNotAuthorised,
        )
        registry.assert_registered(
            self.window,
            context="receipt exported label",
            exc_type=WindowNotAuthorised,
        )
        if self.window != window:
            raise WindowNotAuthorised(
                f"receipt authorises window {self.window!r}, not {window!r}. "
                "Coordination is per-window; it does not carry over."
            )
        missing = [m for m in REQUIRED_MANAGERS if m not in self.replies]
        if missing:
            raise WindowNotAuthorised(
                f"no reply recorded from {missing}. §13: silence is not approval."
            )
        refused = [
            m for m in REQUIRED_MANAGERS
            if not self.replies[m].get("affirmative", False)
        ]
        if refused:
            raise WindowNotAuthorised(
                f"{refused} did not reply affirmatively; the window is not authorised."
            )
        for manager in REQUIRED_MANAGERS:
            if not str(self.replies[manager].get("verbatim", "")).strip():
                raise WindowNotAuthorised(
                    f"reply from {manager} has no verbatim text recorded. §13 requires "
                    "every request and reply to be copied verbatim."
                )
        try:
            requested = datetime.fromisoformat(self.requested_at_utc)
        except (TypeError, ValueError) as exc:
            raise WindowNotAuthorised(
                f"receipt has no parseable requested_at_utc: {self.requested_at_utc!r}"
            ) from exc
        if requested.tzinfo is None:
            requested = requested.replace(tzinfo=timezone.utc)
        age = now - requested
        if age > RECEIPT_MAX_AGE:
            raise WindowNotAuthorised(
                f"receipt is {age} old (limit {RECEIPT_MAX_AGE}). Coordination is for a "
                "window now, not standing permission; re-request."
            )
        if age.total_seconds() < 0:
            raise WindowNotAuthorised("receipt is timestamped in the future")


def load_spend_ledger(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Fingerprint -> spend record. Missing ledger is an empty one, not an error."""
    path = path or SPEND_LEDGER
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return raw.get("spent", {}) or {}


def spend_receipt(
    receipt: CoordinationReceipt,
    *,
    reason: str,
    ledger_path: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Consume a receipt: write the ledger entry AND mark the file.

    Both, deliberately. The ledger is the authority (it survives edits to the
    receipt); the in-file marker is what a human reading the receipt sees. The
    ledger is written FIRST so a crash between the two leaves the receipt spent
    rather than usable -- failing closed on the side of refusing a window.
    """
    registry.assert_registered(
        receipt.window,
        context="receipt spend",
        exc_type=WindowNotAuthorised,
    )
    ledger_path = ledger_path or SPEND_LEDGER
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    record = {
        "spent_at_utc": stamp,
        "window": receipt.window,
        "reason": reason,
        "requested_at_utc": receipt.requested_at_utc,
        "managers": sorted(receipt.replies),
    }

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_spend_ledger(ledger_path)
    existing[receipt.fingerprint()] = record
    ledger_path.write_text(json.dumps({
        "schema": "wrf_gpu2.v025.m0.gpu_coordination_spend_ledger.v1",
        "note": (
            "A coordination receipt authorises ONE window execution. Fingerprints cover "
            "window + request time + replies only, so deleting the spend marker from a "
            "receipt, or copying it to a new path, does not un-spend it. Only genuinely "
            "new coordination produces a new fingerprint."
        ),
        "spent": existing,
    }, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    if receipt.source_path and receipt.source_path.is_file():
        raw = json.loads(receipt.source_path.read_text())
        raw["spent"] = record
        receipt.source_path.write_text(
            json.dumps(raw, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )
    return record


# --------------------------------------------------------------------------- #
# 2. canonical lock                                                            #
# --------------------------------------------------------------------------- #
def check_canonical_lock(
    env: dict[str, str] | None = None,
    *,
    expected_label: str | None = None,
) -> dict[str, Any]:
    """Prove we are running inside `scripts/with_gpu_lock.sh`, not just claiming it.

    The wrapper exports a per-invocation token AND writes it into the holder
    file. Requiring the two to match is what makes this unspoofable by an
    inherited environment: a stale `GPUWRF_GPU_LOCK_HELD=1` from an earlier shell
    has a token that no live holder file carries.
    """
    env = os.environ if env is None else env
    if env.get("GPUWRF_GPU_LOCK_HELD") != "1":
        raise WindowNotAuthorised(
            "GPUWRF_GPU_LOCK_HELD is not set: this process is not running under "
            "scripts/with_gpu_lock.sh. Contract §13 requires the canonical wrapper; "
            "unwrapped GPU use is a contract violation."
        )
    token = env.get("GPUWRF_GPU_LOCK_TOKEN", "")
    holder_path = env.get("GPUWRF_GPU_LOCK_HOLDER_FILE", "")
    if not token or not holder_path:
        raise WindowNotAuthorised(
            "the lock environment is incomplete (no token or holder file); refusing "
            "to treat an inherited variable as a held lock"
        )
    holder = Path(holder_path)
    if not holder.is_file():
        raise WindowNotAuthorised(f"lock holder file {holder} does not exist")
    text = holder.read_text()
    # Parse the wrapper's record shape, not arbitrary substrings.  ``cmd=`` is
    # intentionally last and may contain spaces (including text that looks like
    # ``token=...``); an unanchored search would therefore let a token that
    # exists only inside the command authorize a spend.
    holder_record = re.fullmatch(
        r"holder=(?P<label>\S+)\s+"
        r"pid=(?P<pid>\S+)\s+"
        r"(?:since=(?P<since>\S+)\s+)?"
        r"token=(?P<token>\S+)\s+"
        r"cmd=(?P<cmd>.*)",
        text.strip(),
    )
    if holder_record is None:
        raise WindowNotAuthorised(
            "the holder file is not one exact canonical-wrapper record: "
            f"{text.strip()!r}"
        )
    if holder_record.group("token") != token:
        raise WindowNotAuthorised(
            "the holder file does not carry this process's lock token: the lock is "
            f"held by someone else. holder={text.strip()!r}"
        )
    exported_label = env.get("GPUWRF_GPU_LOCK_LABEL", "")
    if expected_label is not None:
        if exported_label != expected_label:
            raise WindowNotAuthorised(
                "the exported canonical-lock label differs from the exact "
                f"authorized label: {exported_label!r} != {expected_label!r}"
            )
        if holder_record.group("label") != expected_label:
            raise WindowNotAuthorised(
                "the holder record does not carry the exact authorized label: "
                f"{text.strip()!r}"
            )
    return {
        "holder_file": str(holder),
        "holder_record": text.strip(),
        "token_matched": True,
        "exported_label": exported_label or None,
        "label_matched": (
            exported_label == expected_label if expected_label is not None else None
        ),
    }


def validate_spending_identity(
    *,
    window: str,
    receipt: CoordinationReceipt,
    env: dict[str, str] | None = None,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """Bind registry, receipt, exported lock, and holder to one label.

    This helper is the common, reachable P0 immediately before every receipt
    spend.  It performs no mutation.
    """

    registry.assert_registered(
        window, context="spending identity", exc_type=WindowNotAuthorised
    )
    receipt.check(window, ledger_path=ledger_path)
    lock = check_canonical_lock(env, expected_label=window)
    registry.assert_lock_label(
        lock.get("exported_label"),
        expected=window,
        context="spending identity",
        exc_type=WindowNotAuthorised,
    )
    return lock


# --------------------------------------------------------------------------- #
# 3. preemption                                                                #
# --------------------------------------------------------------------------- #
class PreemptionGuard:
    """Turns SIGINT/SIGTERM into an immediate, result-less abort.

    §13 allows no "let this short run finish" exception, so the guard does not
    set a flag for a loop to notice politely — the next `check()` raises, and
    there is no code path that returns a result object after it has fired.
    """

    def __init__(self) -> None:
        self.preempted_at: str | None = None
        self.signal: int | None = None
        self._previous: dict[int, Any] = {}

    def _handler(self, signum: int, _frame: Any) -> None:
        self.preempted_at = datetime.now(timezone.utc).isoformat()
        self.signal = signum

    def __enter__(self) -> "PreemptionGuard":
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[sig] = signal.signal(sig, self._handler)
            except ValueError:  # pragma: no cover - non-main-thread use
                pass
        return self

    def __exit__(self, *exc: Any) -> None:
        for sig, previous in self._previous.items():
            try:
                signal.signal(sig, previous)
            except ValueError:  # pragma: no cover
                pass

    def check(self) -> None:
        if self.preempted_at is not None:
            raise Preempted(
                f"preempted by signal {self.signal} at {self.preempted_at}; this "
                "measurement is INVALID and is not reported as a result"
            )


# --------------------------------------------------------------------------- #
# timing classes                                                               #
# --------------------------------------------------------------------------- #
REQUIRED_TIMING_CLASSES = (
    "lock_wait_seconds",
    "cold_compile_seconds",
    "cached_load_seconds",
    "warm_integration_seconds",
    "io_seconds",
    "profiler_perturbation_seconds",
)


@dataclass
class TimingClasses:
    """§5.3's seven classes, each measured on its own. None means NOT measured."""

    lock_wait_seconds: float | None = None
    cold_compile_seconds: float | None = None
    cached_load_seconds: float | None = None
    warm_integration_seconds: float | None = None
    io_seconds: float | None = None
    profiler_perturbation_seconds: float | None = None
    steps: int | None = None
    seconds_per_forecast_hour: float | None = None
    notes: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        unmeasured = [k for k in REQUIRED_TIMING_CLASSES if getattr(self, k) is None]
        if unmeasured:
            raise ArmIncompleteError(
                f"timing classes never measured: {unmeasured}. Contract §5.3 requires each "
                "to be reported separately; hiding one inside another invalidates the gate."
            )
        return asdict(self)


# --------------------------------------------------------------------------- #
# family attribution                                                           #
# --------------------------------------------------------------------------- #
# Maps a lowercased kernel/HLO name fragment to the census family it belongs to.
# Derived from the A6 static census's own family labels so the GPU measurement is
# compared against the same taxonomy, not a fresh one invented at trace time.
FAMILY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("rrtmg", "physics.radiation"),
    ("radiation", "physics.radiation"),
    ("thompson", "physics.microphysics"),
    ("microphys", "physics.microphysics"),
    ("mynn", "physics.pbl"),
    ("pbl", "physics.pbl"),
    ("sfclay", "physics.surface_layer"),
    ("surface_layer", "physics.surface_layer"),
    ("noahmp", "physics.land_surface"),
    ("kf_", "physics.cumulus"),
    ("cumulus", "physics.cumulus"),
    ("gwdo", "physics.gwd"),
    ("advect", "dycore.advection"),
    ("flux5", "dycore.advection"),
    ("thomas", "dycore.vertical_implicit"),
    ("advance_w", "dycore.vertical_implicit"),
    ("calc_coef_w", "dycore.vertical_implicit"),
    ("rhs_ph", "dycore.geopotential"),
    ("small_step_prep", "dycore.small_step"),
    ("calc_p_rho", "dycore.eos"),
    ("diagnose_pressure", "dycore.eos"),
    ("pressure_gradient", "dycore.pgf"),
    ("pgf", "dycore.pgf"),
    ("coriolis", "dycore.momentum"),
    ("curvature", "dycore.momentum"),
    ("mu_continuity", "dycore.mass"),
    ("hybrid", "dycore.vertical_coordinate"),
    ("diffusion", "dycore.diffusion"),
    ("smag", "dycore.diffusion"),
    ("deformation", "dycore.diffusion"),
    ("damp", "dycore.damping"),
    ("smdiv", "dycore.damping"),
)


def attribute_kernel(name: str) -> str:
    """Map one kernel name to a census family, or ``unknown``."""
    lowered = name.lower()
    for fragment, family in FAMILY_PATTERNS:
        if fragment in lowered:
            return family
    return "unknown"


def attribute_device_time(
    kernels: list[dict[str, Any]], *, min_attribution: float = 0.95
) -> dict[str, Any]:
    """Aggregate traced kernels into census families, failing closed under §9.

    ``kernels`` is a list of ``{"name": str, "device_time_ns": int, "launches": int}``
    as parsed from the nsys export. §9 caps ``unknown`` at 5% of BOTH launches and
    device time, so this returns a result whose ``meets_attribution_bar`` is False
    rather than quietly reporting a flat list — a census that cannot attribute
    itself is not a census.
    """
    by_family: dict[str, dict[str, float]] = {}
    total_time = 0.0
    total_launches = 0
    for kernel in kernels:
        family = attribute_kernel(str(kernel.get("name", "")))
        entry = by_family.setdefault(family, {"device_time_ns": 0.0, "launches": 0.0})
        device_time = float(kernel.get("device_time_ns", 0) or 0)
        launches = float(kernel.get("launches", 0) or 0)
        entry["device_time_ns"] += device_time
        entry["launches"] += launches
        total_time += device_time
        total_launches += launches

    unknown = by_family.get("unknown", {"device_time_ns": 0.0, "launches": 0.0})
    unknown_time_share = (unknown["device_time_ns"] / total_time) if total_time else 1.0
    unknown_launch_share = (
        (unknown["launches"] / total_launches) if total_launches else 1.0
    )
    dycore = sum(
        v["device_time_ns"] for k, v in by_family.items() if k.startswith("dycore")
    )
    physics = sum(
        v["device_time_ns"] for k, v in by_family.items() if k.startswith("physics")
    )
    known = dycore + physics
    return {
        "families": {
            name: {
                "device_time_ns": value["device_time_ns"],
                "launches": int(value["launches"]),
                "device_time_share": (value["device_time_ns"] / total_time) if total_time else None,
            }
            for name, value in sorted(by_family.items())
        },
        "total_device_time_ns": total_time,
        "total_launches": int(total_launches),
        "unknown_device_time_share": unknown_time_share,
        "unknown_launch_share": unknown_launch_share,
        "attributed_device_time_share": 1.0 - unknown_time_share,
        "attributed_launch_share": 1.0 - unknown_launch_share,
        "meets_attribution_bar": bool(
            (1.0 - unknown_time_share) >= min_attribution
            and (1.0 - unknown_launch_share) >= min_attribution
        ),
        # The manager's mandatory W1 discriminator: does measured device time
        # reproduce the static launch proxy's physics-heavy ordering
        # (physics 81.7% / dycore 18.3%)?
        "measured_dycore_share_of_known": (dycore / known) if known else None,
        "measured_physics_share_of_known": (physics / known) if known else None,
        "static_proxy_reference": {
            "dycore_share": 0.183,
            "physics_share": 0.817,
            "radiation_share": 0.365,
            "source": "proofs/v025/m0/hlo_dtype_transfer_census.json (A6, XLA:CPU)",
            "note": (
                "the static proxy is a CPU-backend instruction-count proxy. It is the "
                "pre-registered thing this measurement must agree or disagree with; a "
                "disagreement is a result, not an error, and per manager decision "
                "e5df2364 neither number alone authorises an M2 reorder."
            ),
        },
    }


# --------------------------------------------------------------------------- #
# the arm                                                                      #
# --------------------------------------------------------------------------- #
def authorise(
    window: str,
    *,
    receipt_path: Path,
    env: dict[str, str] | None = None,
    consume: bool = True,
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    """All three §13 gates, then CONSUME the receipt.

    ``consume=True`` is the default because the normal caller is about to run a
    window. `--print-plan` passes ``consume=False`` so merely inspecting the
    authorisation state cannot burn a window's coordination.

    The receipt is spent at authorisation time, **before** any GPU work, not
    after it. A window that then fails, is preempted, or breaches an early stop
    still consumed its coordination -- which is exactly the W1 attempt-1 case,
    and exactly why "the run failed so the yes still counts" is not available.
    """
    registry.assert_registered(
        window, context="gpu arm authorise", exc_type=WindowNotAuthorised
    )
    receipt = CoordinationReceipt.load(receipt_path)
    lock = validate_spending_identity(
        window=window,
        receipt=receipt,
        env=env,
        ledger_path=ledger_path,
    )
    spend_record = None
    if consume:
        spend_record = spend_receipt(
            receipt,
            reason=f"authorised window {window}",
            ledger_path=ledger_path,
        )
    return {
        "window": window,
        "receipt_path": str(receipt_path),
        "receipt_requested_at_utc": receipt.requested_at_utc,
        "receipt_fingerprint": receipt.fingerprint(),
        "replies": receipt.replies,
        "lock": lock,
        "consumed": bool(consume),
        "spend_record": spend_record,
        "authorised_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_forecast_command(*, run_dir: Path, out_dir: Path, hours: float = 1.0) -> list[str]:
    """The read-only production entrypoint invocation for one FAST-v025 GPU arm.

    §3 permits "read-only inspection/invocation of the production model". This
    calls the shipped CLI rather than reimplementing a driver, so the arm measures
    the real program and not a sprint-local approximation of it.

    ``--hours`` is an **int** in the shipped parser, so ``str(1.0)`` -> "1.0" is
    rejected with `invalid int value`. That cost the first W1 attempt: the
    preflight had tested this function's shape but never parsed its output with
    the real parser, so a type mismatch survived all 27 tests and only surfaced
    inside a coordinated window. `test_gpu_arm.py` now parses the built command
    with `gpuwrf.cli.build_parser()`.
    """
    integral = float(hours).is_integer()
    return [
        sys.executable, "-m", "gpuwrf", "run",
        "--namelist", str(run_dir / "namelist.input"),
        "--input-dir", str(run_dir),
        "--output-dir", str(out_dir),
        "--domain", "d01",
        "--hours", str(int(hours)) if integral else str(hours),
    ]


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    guard: PreemptionGuard,
    deadline_seconds: float | None = None,
) -> tuple[int, float, str]:
    """Run one child, killing it at ``deadline_seconds`` if given.

    The deadline is the frozen §5.3 bar for this pass, enforced here so the
    window stops itself. `subprocess.run(timeout=...)` raises after killing the
    child; the elapsed time is still returned so the breach is reported with a
    number rather than as a bare failure.
    """
    guard.check()
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command, cwd=cwd, env=env, capture_output=True, text=True,
            check=False, timeout=deadline_seconds,
        )
    except subprocess.TimeoutExpired as expired:
        elapsed = time.perf_counter() - started
        output = ""
        for stream in (expired.stdout, expired.stderr):
            if stream:
                output += stream if isinstance(stream, str) else stream.decode(errors="replace")
        raise EarlyStopBreached(
            f"frozen early stop: this pass exceeded {deadline_seconds:.0f} s "
            f"(killed at {elapsed:.1f} s). §5.3 does not permit moving the bar; the "
            f"case fails this gate.\n{output[-2000:]}"
        ) from expired
    elapsed = time.perf_counter() - started
    guard.check()
    return proc.returncode, elapsed, (proc.stdout + proc.stderr)


def gpu_arm(
    *,
    window: str = "baseline-census",
    receipt_path: Path = DEFAULT_RECEIPT,
    case_dir: Path | None = None,
    hours: float = 1.0,
    runner: Callable[..., tuple[int, float, str]] = _run,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ledger_path: Path | None = None,
) -> Callable[..., ArmRecord]:
    """Build the injectable GPU arm for `run_fast_pair`.

    Returns a callable matching the `GpuArm` protocol. Authorisation happens
    inside the returned callable, not here, so constructing the factory is always
    safe on CPU and only *invoking* it can touch the GPU.
    """

    def arm(*, run_id: str, run_root: Path) -> ArmRecord:
        auth = authorise(window, receipt_path=receipt_path, ledger_path=ledger_path)
        started = now()
        out_dir = Path(run_root) / f"gpu_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        case = case_dir or Path(run_root)

        timings = TimingClasses()
        # Lock wait is not observable from inside the wrapper (it blocks before
        # exec), so it is derived from the coordination receipt's request time
        # rather than folded into compile. §5.3 forbids hiding one class in
        # another, and an unmeasured class must stay None, not become zero.
        requested = datetime.fromisoformat(auth["receipt_requested_at_utc"])
        if requested.tzinfo is None:
            requested = requested.replace(tzinfo=timezone.utc)
        timings.lock_wait_seconds = max(0.0, (started - requested).total_seconds())
        timings.notes["lock_wait_seconds"] = (
            "receipt request -> arm start; includes coordination reply latency, which is "
            "reported here rather than silently attributed to compile"
        )

        # The A5/A6 harnesses pin JAX to CPU, and this worker's whole shell has
        # been running with that pin set. Inheriting it into a *GPU* arm would
        # produce a complete, plausible, entirely CPU measurement inside a
        # coordinated window -- the most expensive possible silent failure in
        # this sprint. Strip the pins explicitly and record that we did.
        base_env = {
            key: value for key, value in os.environ.items()
            if key not in ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "XLA_FLAGS")
        }
        stripped = sorted(
            key for key in ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES", "XLA_FLAGS")
            if key in os.environ
        )
        # "Cold" means no cached executable is available -- NOT that caching is
        # switched off. The first attempt used GPUWRF_JAX_CACHE=0, which is
        # stricter than §5.3 asks and is a configuration production never runs:
        # with the cache disabled, nothing is retained and repeated compiles of
        # the same program cannot be amortised even within the run. The honest
        # cold measurement points the cache at a fresh empty directory, so the
        # compile is genuinely from scratch AND its results are written, which is
        # exactly what "cold compile through executable readiness" describes and
        # what the following cached-load pass then has to load.
        cold_cache = Path(run_root) / f"jaxcache_{run_id}"
        cold_cache.mkdir(parents=True, exist_ok=True)
        cold_env = dict(base_env, GPUWRF_JAX_CACHE="1", GPUWRF_JAX_CACHE_DIR=str(cold_cache))
        warm_env = dict(cold_env)
        cache_note = {
            "cold": f"cache ENABLED against a fresh empty dir {cold_cache}",
            "cached_load": "same dir, now populated by the cold pass",
            "why": (
                "GPUWRF_JAX_CACHE=0 would be stricter than §5.3 and unlike production; "
                "an empty cache dir is the faithful cold condition"
            ),
        }

        with PreemptionGuard() as guard:
            command = build_forecast_command(run_dir=case, out_dir=out_dir, hours=hours)
            try:
                rc_cold, cold_seconds, cold_log = runner(
                    command, cwd=REPO, env=cold_env, guard=guard,
                    deadline_seconds=COLD_COMPILE_MAX_SECONDS,
                )
            except EarlyStopBreached as breach:
                return ArmRecord(
                    kind="gpu", run_id=run_id,
                    started_at_utc=started.isoformat(),
                    finished_at_utc=now().isoformat(),
                    status="EARLY_STOP",
                    payload={
                        "stage": "cold_compile",
                        "early_stop": "cold_compile_exceeded_600s",
                        "frozen_bar_seconds": COLD_COMPILE_MAX_SECONDS,
                        "detail": str(breach),
                        "authorisation": auth,
                        "gate_verdict": (
                            "§5.3 cold compile <= 600 s: FAIL. The bar is not moved; the case "
                            "is not adopted on this evidence and the failure is localised."
                        ),
                    },
                )
            timings.cold_compile_seconds = cold_seconds
            if rc_cold != 0:
                return ArmRecord(
                    kind="gpu", run_id=run_id,
                    started_at_utc=started.isoformat(),
                    finished_at_utc=now().isoformat(),
                    status="FAILED",
                    payload={"stage": "cold_compile", "returncode": rc_cold,
                             "log_tail": cold_log[-4000:], "authorisation": auth},
                )

            try:
                rc_warm, warm_seconds, warm_log = runner(
                    command, cwd=REPO, env=warm_env, guard=guard,
                    deadline_seconds=GPU_ARM_MAX_SECONDS,
                )
            except EarlyStopBreached as breach:
                return ArmRecord(
                    kind="gpu", run_id=run_id,
                    started_at_utc=started.isoformat(),
                    finished_at_utc=now().isoformat(),
                    status="EARLY_STOP",
                    payload={
                        "stage": "cached_load",
                        "early_stop": "gpu_arm_exceeded_300s",
                        "frozen_bar_seconds": GPU_ARM_MAX_SECONDS,
                        "detail": str(breach),
                        "authorisation": auth,
                        "cold_compile_seconds": timings.cold_compile_seconds,
                    },
                )
            timings.cached_load_seconds = warm_seconds
            if rc_warm != 0:
                return ArmRecord(
                    kind="gpu", run_id=run_id,
                    started_at_utc=started.isoformat(),
                    finished_at_utc=now().isoformat(),
                    status="FAILED",
                    payload={"stage": "cached_load", "returncode": rc_warm,
                             "log_tail": warm_log[-4000:], "authorisation": auth},
                )

        finished = now()
        wrfouts = sorted(out_dir.glob("wrfout_d01_*"))
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "window": window,
            "authorisation": auth,
            "command": command,
            "command_sha256": sha256_text(" ".join(command)),
            "out_dir": str(out_dir),
            "wrfout_files": [p.name for p in wrfouts],
            "cold_log_tail": cold_log[-4000:],
            "warm_log_tail": warm_log[-4000:],
            "cpu_pin_env_stripped": stripped,
            "cache_definition": cache_note,
            "cpu_pin_note": (
                "JAX_PLATFORMS/CUDA_VISIBLE_DEVICES/XLA_FLAGS are removed from the child "
                "environment so a GPU arm cannot silently run on CPU inheriting this "
                "worker's CPU-first pins"
            ),
        }
        if wrfouts:
            payload["final_wrfout_path"] = str(wrfouts[-1])
        # Classes this arm cannot measure without the profiler pass are left as
        # None; `TimingClasses.as_dict()` refuses to serialise until they are
        # filled by the profiled pass, so an unprofiled arm cannot masquerade as
        # a complete one.
        payload["timing_classes_partial"] = asdict(timings)
        return ArmRecord(
            kind="gpu",
            run_id=run_id,
            started_at_utc=started.isoformat(),
            finished_at_utc=finished.isoformat(),
            status="OK" if wrfouts else "FAILED",
            payload=payload,
        )

    return arm


# --------------------------------------------------------------------------- #
# dry run                                                                      #
# --------------------------------------------------------------------------- #
def stub_runner(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    guard: PreemptionGuard,
    deadline_seconds: float | None = None,
) -> tuple[int, float, str]:
    """A device-free stand-in used by `--dry-run` and the tests.

    It writes nothing and launches nothing; it exists so the authorisation,
    preemption and timing-class logic can be proven on CPU before a coordinated
    window is spent discovering a typo.

    Because it writes no `wrfout`, an arm driven by this runner always ends
    `FAILED`. That is deliberate and load-bearing: **no dry run can ever be
    mistaken for a real measurement**, so a stub result can never reach
    `run_fast_pair` as a completed GPU arm.
    """
    guard.check()
    cache_dir = env.get("GPUWRF_JAX_CACHE_DIR", "")
    marker = Path(cache_dir) / ".stub_populated" if cache_dir else None
    cached = bool(marker and marker.exists())
    if marker is not None and not cached:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("x")
    return 0, (3.0 if cached else 42.0), f"[stub] cached={cached} cmd={' '.join(command[:4])}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", default="baseline-census", choices=WINDOWS)
    parser.add_argument("--case", default="fast-v025")
    parser.add_argument("--case-dir", type=Path, default=None)
    parser.add_argument("--run-root", type=Path,
                        default=Path("<DATA_ROOT>/wrf_gpu2/v025/m0/gpu_arms"))
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--single-step", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="exercise the whole control flow with a stub device; touches no GPU",
    )
    parser.add_argument(
        "--print-plan", action="store_true",
        help="print the exact command and the authorisation state, then exit 0",
    )
    args = parser.parse_args()

    if args.case != "fast-v025":
        raise SystemExit(f"unknown case {args.case!r}; M0 freezes fast-v025")

    case_dir = args.case_dir
    if case_dir is None:
        from real_state import default_run_dir

        case_dir = default_run_dir()

    if args.print_plan:
        command = build_forecast_command(
            run_dir=case_dir, out_dir=args.run_root / "PLANNED", hours=args.hours
        )
        state: dict[str, Any] = {
            "window": args.window,
            "case_dir": str(case_dir),
            "command": command,
            "command_sha256": sha256_text(" ".join(command)),
            "receipt_path": str(args.receipt),
        }
        try:
            authorise(args.window, receipt_path=args.receipt, consume=False)
            state["authorised"] = True
            state["note"] = "inspection only; the receipt was NOT consumed"
        except WindowNotAuthorised as exc:
            state["authorised"] = False
            state["refusal"] = str(exc)
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0

    runner = stub_runner if args.dry_run else _run
    if args.dry_run:
        # Dry run must never depend on a real receipt or a real lock: its whole
        # purpose is to prove the logic without the window.
        os.environ.setdefault("GPUWRF_GPU_LOCK_HELD", "1")
        token = f"dryrun-{uuid.uuid4().hex}"
        holder = Path(args.run_root) / "dryrun.holder"
        holder.parent.mkdir(parents=True, exist_ok=True)
        holder.write_text(
            f"holder={args.window} pid={os.getpid()} token={token} cmd=dry-run\n"
        )
        os.environ["GPUWRF_GPU_LOCK_TOKEN"] = token
        os.environ["GPUWRF_GPU_LOCK_HOLDER_FILE"] = str(holder)
        os.environ["GPUWRF_GPU_LOCK_LABEL"] = args.window
        receipt = Path(args.run_root) / "dryrun_receipt.json"
        receipt.write_text(json.dumps({
            "window": args.window,
            "requested_at_utc": datetime.now(timezone.utc).isoformat(),
            "request_text": "[dry run] no request was sent to any manager",
            "replies": {
                m: {"affirmative": True, "verbatim": "[dry run] synthetic reply",
                    "received_at_utc": datetime.now(timezone.utc).isoformat()}
                for m in REQUIRED_MANAGERS
            },
        }, indent=2))
        args.receipt = receipt

    arm = gpu_arm(
        window=args.window, receipt_path=args.receipt,
        case_dir=case_dir, hours=args.hours, runner=runner,
    )
    try:
        record = arm(run_id=uuid.uuid4().hex[:12], run_root=args.run_root)
    except (WindowNotAuthorised, Preempted) as exc:
        print(json.dumps({"status": "REFUSED", "reason": str(exc)}, indent=2))
        return 2

    result = {
        "schema": SCHEMA,
        "dry_run": bool(args.dry_run),
        "gpu_touched": not args.dry_run,
        "arm": {
            "kind": record.kind, "run_id": record.run_id, "status": record.status,
            "started_at_utc": record.started_at_utc,
            "finished_at_utc": record.finished_at_utc,
            "payload": record.payload,
        },
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if record.status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
