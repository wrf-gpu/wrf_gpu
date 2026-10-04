#!/usr/bin/env python3
"""GPU window driver: later stages are SUPPRESSED once an earlier one fails (§6, §13).

This exists because of a defect that cost a coordinated window. The W1b driver
was an ad-hoc shell script in a scratch directory:

    rc1=$?; echo "STAGE1 rc=$rc1"
    [ $rc1 -ne 0 ] && { echo "STAGE1 non-zero -- see the json"; }   # <-- no exit

An earlier revision *had* `exit $rc1`; I dropped it while rewriting the script,
and nothing could catch that because the driver lived outside the repository and
outside the test suite. So when Stage 1 hit the frozen 600 s early stop and
returned rc=1, the shell logged it and launched `nsys` anyway — burning GPU on a
capture that could not be scored, and needing a separate kill.

Two things are fixed here, not one:

1. **Suppression is the default, and it is enforced in code**: any non-zero
   status, `EARLY_STOP`, or exception from a stage marks every later stage
   `SUPPRESSED` and it is never executed. There is no "continue anyway" flag,
   because the failure mode was precisely a driver that continued anyway.
2. **The driver is a committed, tested artifact** rather than a scratch file.
   `tests/v025/test_run_window.py` pins the suppression on CPU with stub stages.
   A driver that only exists inside a coordinated window is a driver whose bugs
   can only be found inside a coordinated window.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

# Terminal states of a stage. Anything not OK stops the window.
OK = "OK"
FAILED = "FAILED"
EARLY_STOP = "EARLY_STOP"
SUPPRESSED = "SUPPRESSED"
ERROR = "ERROR"


@dataclass
class Stage:
    """One step of a window. ``gpu`` marks steps that consume the device."""

    name: str
    command: Sequence[str]
    gpu: bool = True
    timeout_seconds: float | None = None
    # Optional: read a richer verdict out of an artifact the stage wrote, so a
    # stage that exits 0 while recording EARLY_STOP is still treated as a stop.
    verdict_from: Path | None = None
    verdict_key: Sequence[str] = ()


@dataclass
class StageResult:
    name: str
    status: str
    gpu: bool = True
    returncode: int | None = None
    seconds: float | None = None
    detail: str = ""
    log_tail: str = ""


@dataclass
class WindowResult:
    label: str
    started_at_utc: str
    finished_at_utc: str
    stages: list[StageResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.status == OK for s in self.stages)

    @property
    def first_failure(self) -> StageResult | None:
        for stage in self.stages:
            if stage.status != OK:
                return stage
        return None

    def as_dict(self) -> dict[str, Any]:
        failure = self.first_failure
        return {
            "schema": "wrf_gpu2.v025.m0.gpu_window.v1",
            "label": self.label,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
            "ok": self.ok,
            "first_failure": failure.name if failure else None,
            "first_failure_status": failure.status if failure else None,
            # Named precisely: only stages that would have used the device.
            # Previously this listed EVERY suppressed stage regardless of
            # Stage.gpu, which was declared and never read -- a field that does
            # nothing and a label that overstates what was prevented.
            "gpu_stages_suppressed": [
                s.name for s in self.stages if s.status == SUPPRESSED and s.gpu
            ],
            "cpu_stages_suppressed": [
                s.name for s in self.stages if s.status == SUPPRESSED and not s.gpu
            ],
            "stages": [
                {
                    "name": s.name, "status": s.status, "gpu": s.gpu,
                    "returncode": s.returncode,
                    "seconds": s.seconds, "detail": s.detail,
                    "log_tail": s.log_tail[-2000:] if s.log_tail else "",
                }
                for s in self.stages
            ],
            "suppression_rule": (
                "Any stage that is not OK suppresses every later stage. GPU stages are "
                "never executed after a failure, so a coordinated window cannot be spent "
                "on a capture that cannot be scored. There is deliberately no override."
            ),
        }


def _default_runner(
    command: Sequence[str], *, cwd: Path, env: dict[str, str], timeout: float | None
) -> tuple[int, str]:
    proc = subprocess.run(
        list(command), cwd=cwd, env=env, capture_output=True, text=True,
        check=False, timeout=timeout,
    )
    return proc.returncode, (proc.stdout + proc.stderr)


# Statuses a declared artifact is allowed to report. Anything else -- including
# a status this driver does not recognise -- is a stop.
RECOGNISED_ARTIFACT_STATUSES = frozenset({OK, FAILED, EARLY_STOP, ERROR})


def _read_verdict(stage: Stage) -> tuple[str | None, str]:
    """Read a stage's declared verdict artifact. Returns (status, why).

    **A declared artifact must produce an explicit recognised OK to pass.**

    The first version of this returned ``None`` for a missing file, malformed
    JSON, a missing key, or a non-string value, and the caller then fell back to
    the exit code — so a stage that declared an artifact and failed to write it
    was treated as OK. That is fail-OPEN, in the one function whose entire job is
    to fail closed, and I had written a test (`..._falls_back_to_the_exit_code`)
    that pinned the unsafe behaviour as if it were intended.

    Now: declaring `verdict_from` is a promise that the artifact exists, parses,
    contains the key, and names a recognised status. Breaking any part of that
    promise stops the window.
    """
    if stage.verdict_from is None:
        return None, ""  # nothing declared; the exit code governs

    if not stage.verdict_from.is_file():
        return FAILED, (
            f"declared verdict artifact {stage.verdict_from} does not exist; a stage that "
            "promises an artifact and does not write one is not OK"
        )
    try:
        node: Any = json.loads(stage.verdict_from.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return FAILED, f"declared verdict artifact is unreadable/malformed: {exc}"

    for key in stage.verdict_key:
        if not isinstance(node, dict) or key not in node:
            return FAILED, (
                f"declared verdict artifact has no {'.'.join(stage.verdict_key)!r} key"
            )
        node = node[key]

    if not isinstance(node, str):
        return FAILED, (
            f"declared verdict is {type(node).__name__}, not a status string: {node!r}"
        )
    if node not in RECOGNISED_ARTIFACT_STATUSES:
        return FAILED, (
            f"declared verdict {node!r} is not a recognised status "
            f"{sorted(RECOGNISED_ARTIFACT_STATUSES)}; an unknown status is a stop, not a pass"
        )
    return node, f"artifact recorded status={node}"


class WindowBudgetError(ValueError):
    """Raised when stage budgets cannot fit inside the window's own deadline."""


def _stage_budget(stage: Stage) -> float:
    """A stage's declared budget, refusing values that silently mean 'unbounded'.

    `None`, 0, negative and non-finite all previously counted as ZERO in the
    budget sum, so a plan full of unbounded stages validated against any
    deadline. A stage with no real budget cannot be scheduled inside a capped
    window.
    """
    value = stage.timeout_seconds
    if value is None:
        raise WindowBudgetError(
            f"stage {stage.name!r} declares no timeout_seconds; a stage with no budget "
            "cannot be scheduled inside a capped window"
        )
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise WindowBudgetError(
            f"stage {stage.name!r} has timeout_seconds={value!r}; must be finite and > 0"
        )
    return value


def validate_budget(
    stages: Sequence[Stage], *, deadline_seconds: float, overhead_seconds: float = 0.0
) -> dict[str, float]:
    """Refuse a plan whose stage budgets exceed the window it claims to fit in.

    My first Step-1 plan budgeted 700 + 700 + 300 = 1700 s of stage time inside a
    "hard cap 25 minutes" (1500 s). Nobody checked the arithmetic because nothing
    checked it -- the cap lived in prose. A plan that cannot fit its own cap is
    not a capped plan, and the point of a cap is what a coordinating manager is
    promised.
    """
    if not math.isfinite(deadline_seconds) or deadline_seconds <= 0.0:
        raise WindowBudgetError(f"deadline_seconds={deadline_seconds!r} must be finite and > 0")
    if not math.isfinite(overhead_seconds) or overhead_seconds < 0.0:
        raise WindowBudgetError(f"overhead_seconds={overhead_seconds!r} must be finite and >= 0")
    total = sum(_stage_budget(s) for s in stages)
    required = total + overhead_seconds
    if required > deadline_seconds:
        raise WindowBudgetError(
            f"stage budgets {total:.0f} s + overhead {overhead_seconds:.0f} s = "
            f"{required:.0f} s exceed the window deadline {deadline_seconds:.0f} s. "
            "Shrink the stages or raise the deadline -- but the deadline is what the "
            "coordinating managers were promised, so shrink the stages."
        )
    return {
        "stage_budget_seconds": total,
        "overhead_seconds": overhead_seconds,
        "deadline_seconds": deadline_seconds,
        "headroom_seconds": deadline_seconds - required,
    }


def run_window(
    *,
    label: str,
    stages: Sequence[Stage],
    cwd: Path,
    env: dict[str, str],
    runner: Callable[..., tuple[int, str]] = _default_runner,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    deadline_seconds: float | None = None,
    overhead_seconds: float = 0.0,
) -> WindowResult:
    """Run stages in order, suppressing everything after the first failure.

    ``deadline_seconds`` is a GLOBAL wall budget for the whole window, checked
    against the stage budgets before anything runs and re-checked before each
    stage starts. Per-stage timeouts alone do not bound a window: three stages
    that each respect a 700 s budget still take 2100 s.
    """

    if deadline_seconds is not None:
        validate_budget(stages, deadline_seconds=deadline_seconds,
                        overhead_seconds=overhead_seconds)

    started = now()
    window_began = time.perf_counter()
    results: list[StageResult] = []
    stop = False

    for stage in stages:
        # Global deadline: stop before starting a stage that cannot finish inside
        # the window we promised, rather than discovering it afterwards.
        effective_timeout = stage.timeout_seconds
        if not stop and deadline_seconds is not None:
            elapsed = time.perf_counter() - window_began
            budget = _stage_budget(stage)
            # Overhead is RESERVED at runtime, not merely checked up front: GPU
            # return has to fit inside the window too.
            remaining = deadline_seconds - overhead_seconds - elapsed
            # The child gets the SMALLER of its own budget and what is left, so a
            # stage cannot overrun the window just because its own budget fits.
            # Clamp, then decide. Stopping whenever `budget > remaining` would
            # make the clamp dead code -- those are the same condition. A stage
            # with time left runs with a CLAMPED timeout so it cannot overrun the
            # window; only a stage with no time left is skipped outright.
            effective_timeout = max(0.0, min(budget, remaining))
            if remaining <= 0.0:
                results.append(StageResult(
                    name=stage.name, status=EARLY_STOP, gpu=stage.gpu,
                    detail=(f"window deadline: {elapsed:.0f} s elapsed leaves no time "
                            f"inside {deadline_seconds:.0f} s after reserving "
                            f"{overhead_seconds:.0f} s overhead"),
                ))
                stop = True
                continue

        if stop:
            results.append(StageResult(
                name=stage.name, status=SUPPRESSED, gpu=stage.gpu,
                detail=("suppressed: an earlier stage did not return OK, so this "
                        "stage was never executed"),
            ))
            continue

        began = time.perf_counter()
        try:
            returncode, output = runner(
                stage.command, cwd=cwd, env=env, timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired:
            results.append(StageResult(
                name=stage.name, status=EARLY_STOP, gpu=stage.gpu, returncode=None,
                seconds=time.perf_counter() - began,
                detail=f"stage exceeded its {stage.timeout_seconds} s backstop",
            ))
            stop = True
            continue
        except Exception as exc:  # noqa: BLE001 - any launch failure stops the window
            results.append(StageResult(
                name=stage.name, status=ERROR, gpu=stage.gpu,
                seconds=time.perf_counter() - began,
                detail=f"{type(exc).__name__}: {exc}",
            ))
            stop = True
            continue

        elapsed = time.perf_counter() - began
        recorded, why = _read_verdict(stage)

        # Both signals must say OK. A declared artifact that says OK while the
        # process exited non-zero is not a pass, and neither is the reverse.
        if returncode != 0:
            status = FAILED
            why = why or f"exit code {returncode}"
        elif recorded is None:
            status = OK                      # nothing declared; rc0 governs
        elif recorded == OK:
            status = OK
        else:
            status = recorded if recorded in (EARLY_STOP, ERROR) else FAILED

        results.append(StageResult(
            name=stage.name, status=status, gpu=stage.gpu, returncode=returncode,
            seconds=elapsed, detail=why, log_tail=output,
        ))
        if status != OK:
            stop = True

    # A final-stage overrun previously returned ok=True: nothing re-checked the
    # deadline after the last stage, so a window could blow its cap and report
    # success.
    if deadline_seconds is not None and not stop:
        elapsed = time.perf_counter() - window_began
        if elapsed + overhead_seconds > deadline_seconds:
            results.append(StageResult(
                name="window_deadline", status=EARLY_STOP, gpu=False,
                seconds=elapsed,
                detail=(f"window overran: {elapsed:.0f} s + {overhead_seconds:.0f} s overhead "
                        f"exceeds {deadline_seconds:.0f} s"),
            ))

    return WindowResult(
        label=label,
        started_at_utc=started.isoformat(),
        finished_at_utc=now().isoformat(),
        stages=results,
    )


__all__ = [
    "EARLY_STOP", "ERROR", "FAILED", "OK", "SUPPRESSED",
    "WindowBudgetError", "validate_budget",
    "Stage", "StageResult", "WindowResult", "run_window",
]
