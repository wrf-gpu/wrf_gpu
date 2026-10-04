#!/usr/bin/env python3
"""Parse WRF ``rsl.out.0000`` timing lines into per-domain step series.

M0 (v0.25) needs one unambiguous, reusable definition of "CPU-WRF seconds per
forecast hour". WRF prints one line per integration step per domain::

    Timing for main: time 2026-07-26_00:00:54 on domain   1:   12.57225 elapsed seconds

The two competing legacy scalars (92.3 and 86.45 s/model-hour) were produced by
two different reductions of exactly these lines, so every reduction this project
uses is implemented here once, named, and recomputable.

THE NON-DIVISIBLE BOUNDARY
--------------------------
``dt=54`` does not divide 3600. A "one forecast hour" WRF run does not stop at
3600 s of model time: it takes the first step at or past the boundary, so it
advances 67*54 = **3618 s**, overshooting by 18 s (0.5%). Dividing that run's
cost by 3600 inflates the rate by 0.5%, and 0.5% is load-bearing when the
question under investigation is a 6.8% gap.

This module therefore never accepts a window length on faith. It derives the
actual advance from the parsed model timestamps, cross-checks it against
``n_steps * dt``, and reports the scheduled and actual normalisations side by
side. ``per_fc_hour`` is always the *actual*-advance rate; the scheduled-window
rate is reported next to it as ``per_fc_hour_scheduled_window`` with the
overshoot factor, so no caller can silently pick the flattering convention.

Boundaries do align periodically: gcd(54, 3600) = 18, so exactly 200 steps span
exactly 3 forecast hours. ``block_profile`` uses that to give a bias-free
profile, while ``hourly_profile`` also returns each hour's step count so the
66/67/67 sawtooth is visible rather than mistaken for physics.

Definitions produced (seconds per forecast hour of domain 1):

``step_sum``
    Sum of every domain-1 step time in the window, over the actual advance. For
    a nested run the domain-1 line already *contains* the child domains' cost,
    so this is the whole-model number, not a d01-only one.
``step_sum_warm``
    Same, after dropping the first ``warmup_steps`` domain-1 steps (the one-time
    Thompson lookup-table build and first-touch allocation land there), rescaled
    onto the full window so the two are directly comparable.
``wallclock``
    End-to-end integration wallclock over the identical model-time boundaries,
    supplied by the caller because it is not in the RSL stream.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

TIMING_RE = re.compile(
    r"Timing for main: time (?P<time>\S+) on domain\s+(?P<domain>\d+):\s+"
    r"(?P<elapsed>[0-9.]+) elapsed seconds"
)

# WRF's own per-file I/O accounting, kept separate from integration cost.
IO_RE = re.compile(
    r"Timing for Writing (?P<file>\S+) for domain\s+(?P<domain>\d+):\s+"
    r"(?P<elapsed>[0-9.]+) elapsed seconds"
)

WRF_TIME_FMT = "%Y-%m-%d_%H:%M:%S"

# A step's stamped model time must land within this many seconds of the time
# predicted by t0 + k*dt. WRF stamps whole seconds, so anything above ~1 s means
# the caller's dt is wrong or the log interleaves two runs.
TIMESTAMP_TOLERANCE_S = 1.0


class TimingConsistencyError(ValueError):
    """Raised when a step series contradicts the caller's dt/window."""


@dataclass
class DomainSeries:
    """Per-step elapsed seconds for one WRF domain, in model-time order."""

    domain: int
    times: list[str] = field(default_factory=list)
    elapsed: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.elapsed)

    def stamps(self) -> list[datetime]:
        return [datetime.strptime(t, WRF_TIME_FMT) for t in self.times]


@dataclass
class IoRecord:
    domain: int
    path: str
    elapsed: float


def parse_rsl(path: Path) -> tuple[dict[int, DomainSeries], list[IoRecord]]:
    """Return ``{domain: DomainSeries}`` and the I/O write records."""
    series: dict[int, DomainSeries] = {}
    io: list[IoRecord] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            match = TIMING_RE.search(line)
            if match:
                dom = int(match["domain"])
                entry = series.setdefault(dom, DomainSeries(domain=dom))
                entry.times.append(match["time"])
                entry.elapsed.append(float(match["elapsed"]))
                continue
            match = IO_RE.search(line)
            if match:
                io.append(
                    IoRecord(
                        domain=int(match["domain"]),
                        path=match["file"],
                        elapsed=float(match["elapsed"]),
                    )
                )
    return series, io


def window_geometry(
    series: DomainSeries,
    *,
    dt_seconds: float,
    scheduled_window_seconds: float | None = None,
    start_time: datetime | None = None,
) -> dict:
    """Derive the *actual* model advance from stamps and validate it against dt.

    Raises ``TimingConsistencyError`` on any unexplained ``n*dt`` mismatch rather
    than normalising by a number nobody checked.
    """
    stamps = series.stamps()
    n = len(stamps)
    if n == 0:
        raise TimingConsistencyError("empty step series")

    # WRF stamps the model time *after* each step, so step 1's stamp is t0 + dt.
    inferred_t0 = stamps[0] - timedelta(seconds=dt_seconds)
    t0 = start_time if start_time is not None else inferred_t0
    if start_time is not None:
        drift = abs((inferred_t0 - start_time).total_seconds())
        if drift > TIMESTAMP_TOLERANCE_S:
            raise TimingConsistencyError(
                f"first stamp {stamps[0]} implies t0={inferred_t0}, but caller gave "
                f"start_time={start_time} (drift {drift:.3f}s > {TIMESTAMP_TOLERANCE_S}s); "
                "dt or the log window is wrong"
            )

    actual_advance = (stamps[-1] - t0).total_seconds()
    predicted_advance = n * dt_seconds
    mismatch = abs(actual_advance - predicted_advance)
    if mismatch > TIMESTAMP_TOLERANCE_S:
        raise TimingConsistencyError(
            f"{n} steps at dt={dt_seconds}s predict {predicted_advance:.1f}s of model "
            f"advance, but stamps {stamps[0]}..{stamps[-1]} span {actual_advance:.1f}s "
            f"(mismatch {mismatch:.1f}s). Refusing to normalise: either dt is wrong, "
            "the domain has an adaptive timestep, or the log contains more than one run."
        )

    # Every stamp must sit on the t0 + k*dt lattice; a gap means interleaved runs.
    for k, stamp in enumerate(stamps, start=1):
        expected = t0 + timedelta(seconds=k * dt_seconds)
        if abs((stamp - expected).total_seconds()) > TIMESTAMP_TOLERANCE_S:
            raise TimingConsistencyError(
                f"step {k} stamped {stamp}, expected {expected} at dt={dt_seconds}s; "
                "step series is not a single contiguous run"
            )

    geom = {
        "start_time": t0.strftime(WRF_TIME_FMT),
        "end_time": stamps[-1].strftime(WRF_TIME_FMT),
        "dt_seconds": dt_seconds,
        "steps": n,
        "actual_model_advance_seconds": actual_advance,
        "actual_model_advance_hours": actual_advance / 3600.0,
        "predicted_advance_seconds": predicted_advance,
        "timestamp_mismatch_seconds": mismatch,
        "dt_divides_hour": (3600.0 % dt_seconds) == 0,
    }
    if scheduled_window_seconds is not None:
        overshoot = actual_advance / scheduled_window_seconds
        geom.update(
            {
                "scheduled_window_seconds": scheduled_window_seconds,
                "scheduled_window_hours": scheduled_window_seconds / 3600.0,
                "window_overshoot_factor": overshoot,
                "window_overshoot_seconds": actual_advance - scheduled_window_seconds,
                "window_overshoot_percent": (overshoot - 1.0) * 100.0,
            }
        )
        # A run may overshoot its scheduled window by at most one step; more than
        # that means the caller described the wrong window.
        if not (-TIMESTAMP_TOLERANCE_S <= actual_advance - scheduled_window_seconds < dt_seconds):
            raise TimingConsistencyError(
                f"actual advance {actual_advance:.1f}s is not within one step "
                f"(dt={dt_seconds}s) of the scheduled window "
                f"{scheduled_window_seconds:.1f}s; the described window is wrong"
            )
    return geom


def _spread(values: list[float]) -> dict[str, float]:
    """Median, MAD, CV and range for a repeat sample."""
    if not values:
        return {}
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "median": med,
        "mean": mean,
        "stdev": sd,
        "mad": mad,
        "cv": (sd / mean) if mean else 0.0,
        "min": min(values),
        "max": max(values),
    }


def reduce_series(
    series: DomainSeries,
    *,
    dt_seconds: float,
    scheduled_window_seconds: float | None = None,
    start_time: datetime | None = None,
    warmup_steps: int = 1,
) -> dict:
    """Apply every named reduction to one domain's step series.

    All ``*_per_fc_hour`` rates normalise by the **actual** model advance derived
    from the stamps. When a scheduled window is supplied, the scheduled-window
    rate is reported alongside so the convention difference is explicit.
    """
    elapsed = series.elapsed
    n = len(elapsed)
    geom = window_geometry(
        series,
        dt_seconds=dt_seconds,
        scheduled_window_seconds=scheduled_window_seconds,
        start_time=start_time,
    )
    hours = geom["actual_model_advance_hours"]

    total = sum(elapsed)
    warm = elapsed[warmup_steps:]
    # Rescale the warm mean back onto the full window so the two definitions
    # differ only by the warm-up exclusion, not by a missing step's worth of work.
    warm_total = statistics.mean(warm) * n if warm else float("nan")

    out = {
        "domain": series.domain,
        "window": geom,
        "step_sum_seconds": total,
        "step_sum_per_fc_hour": total / hours,
        "warmup_steps_excluded": warmup_steps,
        "warmup_seconds": sum(elapsed[:warmup_steps]),
        "step_sum_warm_seconds": warm_total,
        "step_sum_warm_per_fc_hour": warm_total / hours,
        "per_step": _spread(elapsed),
        "per_step_warm": _spread(warm),
        "first_step_seconds": elapsed[0],
    }
    if scheduled_window_seconds is not None:
        sched_hours = scheduled_window_seconds / 3600.0
        out["step_sum_per_fc_hour_scheduled_window"] = total / sched_hours
        out["step_sum_warm_per_fc_hour_scheduled_window"] = warm_total / sched_hours
        out["normalisation_note"] = (
            f"canonical rates use the actual {geom['actual_model_advance_seconds']:.0f}s "
            f"advance ({n} steps x {dt_seconds}s); the *_scheduled_window rates use the "
            f"requested {scheduled_window_seconds:.0f}s and are "
            f"{geom['window_overshoot_percent']:+.3f}% different"
        )
    return out


def hourly_profile(series: DomainSeries, *, dt_seconds: float) -> list[dict]:
    """Step-time sum per whole forecast hour, with each hour's step count.

    A step is credited to the hour its *end* falls in. When dt does not divide
    3600 the step count per hour alternates (66/67/67 at dt=54), which changes
    the hour's cost by ~1.5% for reasons that have nothing to do with the
    weather — so ``steps`` is returned next to every sum and ``mean_per_step``
    is the comparable quantity.
    """
    buckets: dict[int, list[float]] = {}
    for index, value in enumerate(series.elapsed):
        model_seconds = (index + 1) * dt_seconds
        # Half-open (lo, hi]: a step ending exactly on the boundary closes that hour.
        hour = int((model_seconds - 1e-9) // 3600)
        buckets.setdefault(hour, []).append(value)
    return [
        {
            "hour": h,
            "steps": len(buckets[h]),
            "sum_seconds": sum(buckets[h]),
            "mean_per_step": statistics.mean(buckets[h]),
        }
        for h in sorted(buckets)
    ]


def block_profile(series: DomainSeries, *, dt_seconds: float) -> dict:
    """Bias-free profile over the smallest exact step/hour repeat unit.

    gcd(dt, 3600) fixes the period: at dt=54, 200 steps span exactly 3 forecast
    hours, so every block holds identical model time and blocks are directly
    comparable without the sawtooth ``hourly_profile`` exposes.
    """
    from math import gcd

    dt_int = int(round(dt_seconds))
    if abs(dt_seconds - dt_int) > 1e-9:
        raise TimingConsistencyError("block profile needs an integer timestep")
    steps_per_block = 3600 // gcd(dt_int, 3600)
    hours_per_block = steps_per_block * dt_int / 3600.0

    blocks: list[dict] = []
    for start in range(0, len(series.elapsed) - steps_per_block + 1, steps_per_block):
        chunk = series.elapsed[start : start + steps_per_block]
        blocks.append(
            {
                "block": len(blocks),
                "first_step": start + 1,
                "sum_seconds": sum(chunk),
                "per_fc_hour": sum(chunk) / hours_per_block,
            }
        )
    return {
        "steps_per_block": steps_per_block,
        "hours_per_block": hours_per_block,
        "blocks": blocks,
        "per_fc_hour_spread": _spread([b["per_fc_hour"] for b in blocks]),
    }


def summarize(
    path: Path,
    *,
    dt_seconds: float,
    scheduled_window_seconds: float | None = None,
    start_time: datetime | None = None,
    warmup_steps: int = 1,
    wallclock_seconds: float | None = None,
) -> dict:
    series, io = parse_rsl(path)
    if 1 not in series:
        raise TimingConsistencyError(f"{path}: no domain-1 timing lines found")

    d1 = reduce_series(
        series[1],
        dt_seconds=dt_seconds,
        scheduled_window_seconds=scheduled_window_seconds,
        start_time=start_time,
        warmup_steps=warmup_steps,
    )
    hours = d1["window"]["actual_model_advance_hours"]
    out: dict = {
        "rsl_path": str(path),
        "domains_present": sorted(series),
        "domain1": d1,
        "io_writes": [
            {"domain": r.domain, "path": r.path, "elapsed_seconds": r.elapsed}
            for r in io
        ],
        "io_seconds_total": sum(r.elapsed for r in io),
    }
    if wallclock_seconds is not None:
        out["wallclock_seconds"] = wallclock_seconds
        out["wallclock_per_fc_hour"] = wallclock_seconds / hours
        out["wallclock_minus_step_sum_seconds"] = (
            wallclock_seconds - d1["step_sum_seconds"]
        )
    profile = hourly_profile(series[1], dt_seconds=dt_seconds)
    if len(profile) > 1:
        out["hourly_profile"] = profile
        out["hourly_sum_spread"] = _spread([h["sum_seconds"] for h in profile])
        out["hourly_step_counts"] = sorted({h["steps"] for h in profile})
        out["block_profile"] = block_profile(series[1], dt_seconds=dt_seconds)
    for dom in sorted(series):
        if dom != 1:
            out[f"domain{dom}_steps"] = len(series[dom])
            out[f"domain{dom}_step_sum_seconds"] = sum(series[dom].elapsed)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rsl", type=Path, help="path to rsl.out.0000")
    parser.add_argument("--dt", type=float, required=True, help="domain-1 time_step (s)")
    parser.add_argument(
        "--scheduled-window-seconds",
        type=float,
        default=None,
        help="model-time window the run was ASKED for (e.g. 3600); the actual "
        "advance is derived from the stamps and both rates are reported",
    )
    parser.add_argument("--start-time", type=str, default=None, help="YYYY-MM-DD_HH:MM:SS")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--wallclock-seconds", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    result = summarize(
        args.rsl,
        dt_seconds=args.dt,
        scheduled_window_seconds=args.scheduled_window_seconds,
        start_time=(
            datetime.strptime(args.start_time, WRF_TIME_FMT) if args.start_time else None
        ),
        warmup_steps=args.warmup_steps,
        wallclock_seconds=args.wallclock_seconds,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
