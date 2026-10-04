#!/usr/bin/env python3
"""Exclusive-time attribution over NESTED NVTX ranges (CPU-only, pure parsing).

Why the existing `nvtx_sum` path is not enough
----------------------------------------------
XLA's compile ranges nest: `TSL:XlaCompile` contains `TSL:XlaPassPipeline`,
which contains many `TSL:XlaPass`. `nsys stats --report nvtx_sum` reports
**inclusive** durations aggregated per name. Summing them counts a child's time
once inside every ancestor, so an "attribution" built that way can exceed 100%
of the compile and still look tidy. The manager already flagged the related
confusion in the existing capture: 107,412 summary ROWS against 157,080
INSTANCES -- different quantities.

So Step 1 works from the raw range table (`nvtx_pushpop_trace`) and computes
**exclusive (self) time**: a range's duration minus the duration of its direct
children. Exclusive times partition the timeline, so shares are real shares.

Fail-closed properties
----------------------
* Nesting is reconstructed independently of nsys, then **cross-checked** against
  nsys's own `DurNonChild` column when present. Disagreement beyond tolerance
  raises rather than silently preferring one source.
* Ranges are mapped to families by frozen patterns. Anything unmatched lands in
  `unknown` -- it is never absorbed into a named family.
* The attribution validator BLOCKS when `unknown` exceeds the pre-registered
  5% budget. It never scores a partial attribution as a result.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

# Pre-registered attribution budget (contract §11 style: a number, not a judgement).
MAX_UNKNOWN_SHARE = 0.05

# Ranges that are EXECUTION, not compilation. They are classified and reported,
# but excluded from the compile denominator: mixing thunk execution into a
# "compile attribution" would silently shrink every compile share.
NON_COMPILE_FAMILIES = frozenset({"runtime"})

# Frozen family patterns, matched IN ORDER against the NVTX range name. Order is
# load-bearing twice over: an autotuner range is also an `XlaPass`, and
# `XlaCompileGpuAsm` is codegen rather than a compile frame.
#
# Every pattern here was derived from the range names actually present in the
# W1b capture, not guessed -- the first version left 2.9% unattributed, all of it
# real XLA phases (XlaCompileBackend, XlaMemoryScheduler, XlaBufferAssignment,
# XlaCreateGpuExecutable, XlaModule, XlaCompileCudnnFusion, Thunk).
FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("autotune", re.compile(
        r"(triton[_-]?autotun|gemm[_-]algorithm[_-]picker|conv[_-]algorithm[_-]picker|"
        r"autotun)", re.I)),
    ("runtime", re.compile(r"TSL:Thunk", re.I)),
    ("codegen", re.compile(
        r"(XlaCompileGpuAsm|XlaEmitGpuAsm|XlaEmitLlvmIr|XlaOptimizeLlvmIr|"
        r"XlaCompileCudnnFusion|ptxas|llvm)", re.I)),
    ("buffer_and_schedule", re.compile(
        r"(XlaMemoryScheduler|XlaBufferAssignment)", re.I)),
    ("pass_pipeline", re.compile(r"XlaPassPipeline", re.I)),
    ("hlo_pass", re.compile(r"XlaPass\b|XlaPass:", re.I)),
    ("compile_frame", re.compile(
        r"(XlaCompileBackend|XlaCreateGpuExecutable|XlaModule|XlaCompile\b|XlaCompile:)", re.I)),
)

_NAME_FIELD = re.compile(r"#(?:name|module)=([^,#]+)")


def _to_float(value: str) -> float:
    text = (value or "").strip().replace(",", "")
    if not text or text in {"-", "N/A"}:
        return 0.0
    return float(text)


def parse_pushpop_trace(text: str) -> list[dict[str, Any]]:
    """Parse `nsys stats --report nvtx_pushpop_trace --format csv`.

    nsys emits a preamble before the CSV header on some versions, so the header
    row is located rather than assumed to be line 0.
    """
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines)
         if "Name" in line and ("Duration" in line or "Duration (ns)" in line)),
        None,
    )
    if start is None:
        raise ValueError("no nvtx_pushpop_trace header found in the CSV")
    import parse_profiler

    parse_profiler.verified_preamble(lines, start)

    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    rows: list[dict[str, Any]] = []
    for raw in reader:
        clean = {(k or "").strip(): (v or "").strip() for k, v in raw.items() if k}
        name = clean.get("Name")
        if not name:
            continue

        def pick(*candidates: str) -> str:
            for candidate in candidates:
                if candidate in clean and clean[candidate] != "":
                    return clean[candidate]
            return ""

        row: dict[str, Any] = {
            "name": name,
            "start_ns": _to_float(pick("Start (ns)", "Start")),
            "duration_ns": _to_float(pick("Duration (ns)", "Duration")),
            "pid": pick("PID"),
            "tid": pick("TID"),
        }
        for key, column in (("range_id", "RangeId"), ("parent_id", "ParentId")):
            value = pick(column)
            row[key] = value or None
        non_child = pick("DurNonChild (ns)", "DurNonChild")
        row["nsys_non_child_ns"] = _to_float(non_child) if non_child else None
        rows.append(row)
    return rows


def compute_exclusive(rows: list[dict[str, Any]], *, cross_check_tolerance: float = 0.02
                      ) -> list[dict[str, Any]]:
    """Attach `exclusive_ns` to every range.

    Direct children are found via `ParentId` when nsys supplies it, and otherwise
    by reconstructing the push/pop stack per (PID, TID) from start times and
    durations. Sorting by `(start, -duration)` puts an enclosing range before the
    children it contains when they share a start timestamp.
    """
    child_total: dict[int, float] = defaultdict(float)

    have_parents = all(row.get("range_id") for row in rows) and any(
        row.get("parent_id") for row in rows)

    if have_parents:
        index_of = {row["range_id"]: i for i, row in enumerate(rows)}
        for i, row in enumerate(rows):
            parent = row.get("parent_id")
            if parent and parent in index_of:
                child_total[index_of[parent]] += row["duration_ns"]
    else:
        per_thread: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, row in enumerate(rows):
            per_thread[(row["pid"], row["tid"])].append(i)
        for indices in per_thread.values():
            # An enclosing range must precede the children it contains, so ties on
            # start time are broken by putting the LONGER range first.
            indices.sort(key=lambda i: (rows[i]["start_ns"], -rows[i]["duration_ns"]))
            stack: list[int] = []
            for i in indices:
                start = rows[i]["start_ns"]
                while stack and (rows[stack[-1]]["start_ns"]
                                 + rows[stack[-1]]["duration_ns"]) <= start:
                    stack.pop()
                if stack:
                    child_total[stack[-1]] += rows[i]["duration_ns"]
                stack.append(i)

    enriched: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        exclusive = max(0.0, row["duration_ns"] - child_total[i])
        item = dict(row)
        item["child_ns"] = child_total[i]
        item["exclusive_ns"] = exclusive
        reported = row.get("nsys_non_child_ns")
        if reported is not None and row["duration_ns"] > 0:
            drift = abs(reported - exclusive) / row["duration_ns"]
            item["nsys_cross_check_drift"] = drift
            if drift > cross_check_tolerance:
                raise ValueError(
                    f"exclusive-time cross-check failed for {row['name']!r}: "
                    f"computed {exclusive:.0f} ns vs nsys DurNonChild {reported:.0f} ns "
                    f"({drift:.1%} of duration, tolerance {cross_check_tolerance:.0%}). "
                    "Refusing to attribute compile time from an unreconciled trace."
                )
        enriched.append(item)
    return enriched


def classify(name: str) -> str:
    for family, pattern in FAMILY_PATTERNS:
        if pattern.search(name):
            return family
    return "unknown"


def pass_label(name: str) -> str:
    """`TSL:XlaPass:#name=call-inliner,module=jit_f#` -> `call-inliner`."""
    match = _NAME_FIELD.search(name)
    return match.group(1) if match else name


_MODULE_FIELD = re.compile(r"module=([^,#]+)")


def module_of(name: str) -> str | None:
    """`…#name=call-inliner,module=jit_f,program_id=1#` -> `jit_f`."""
    match = _MODULE_FIELD.search(name)
    return match.group(1) if match else None


def dominant_module(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """The module with the most exclusive compile time, chosen mechanically.

    H2 is about "the dominant module", so which module that is must be derived
    from the trace rather than assumed. Ties are broken by name so the choice is
    deterministic across runs.
    """
    per_module: dict[str, float] = defaultdict(float)
    for row in rows:
        family = classify(row["name"])
        if family in NON_COMPILE_FAMILIES:
            continue
        module = module_of(row["name"])
        if module:
            per_module[module] += row["exclusive_ns"] / 1e9
    if not per_module:
        return {"module": None, "reason": "no module-tagged compile ranges in the trace"}
    ranked = sorted(per_module.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "module": ranked[0][0],
        "exclusive_compile_seconds": ranked[0][1],
        "ranked": [{"module": m, "seconds": s} for m, s in ranked[:10]],
        "selection": "largest exclusive compile time; ties broken by module name",
    }


def scoped_pass_seconds(rows: Iterable[dict[str, Any]], module: str) -> dict[str, Any]:
    """Per-pass exclusive seconds for ONE module, with a matching denominator.

    The denominator is built from the same scoped, exclusive rows as the
    numerators, so the shares are shares of something. Mixing a whole-process
    denominator with per-module numerators -- or an inclusive denominator with
    exclusive numerators -- produces percentages that do not add up, which is the
    failure mode H2's 15% bar would silently inherit.
    """
    per_pass: dict[str, float] = defaultdict(float)
    denominator = 0.0
    counted = 0
    for row in rows:
        if module_of(row["name"]) != module:
            continue
        family = classify(row["name"])
        if family in NON_COMPILE_FAMILIES:
            continue
        seconds = row["exclusive_ns"] / 1e9
        denominator += seconds
        counted += 1
        if family in {"hlo_pass", "autotune", "pass_pipeline", "codegen"}:
            per_pass[pass_label(row["name"])] += seconds
    return {
        "module": module,
        "pass_exclusive_seconds": dict(sorted(per_pass.items(), key=lambda kv: -kv[1])),
        "module_compile_seconds": denominator,
        "ranges_counted": counted,
        "denominator_basis": (
            "sum of EXCLUSIVE compile-family seconds for this module only -- the same rows the "
            "per-pass numerators come from, so the shares are commensurable"
        ),
    }


COMPILE_START_FAMILIES = {"compile_frame", "pass_pipeline", "hlo_pass", "autotune", "codegen",
                          "buffer_and_schedule"}


#: This is an upper-bound clock, not exact application launch-to-readiness.
#: Must match ``h1_discriminator.UPPER_BOUND_BASIS``.
SESSION_ORIGIN_UPPER_BOUND_BASIS = "nsys-session-origin-to-last-compile-range-end"


def compile_readiness_seconds(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Conservative upper bound on application-relative ``T_off``.

    ``T_on >= 601.4 s`` is process-launch-to-kill.  Nsight does not expose that
    exact origin or an executable-ready marker here: its timestamps start at
    the profiling-session origin, which precedes application launch.  The last
    compile-range end is therefore explicitly typed as an upper bound, never as
    exact product compile evidence.

    nsys timestamps are measured from the **profiling session origin**, which
    begins before the traced application starts. So the last compile range's END
    offset already contains everything before the first range -- interpreter
    start, JAX import, backend init, input load -- and that is exactly what must
    be kept.

    The earlier version returned `max(end) - min(start)`, subtracting all of that
    startup. Subtracting it shrinks `T_off`, which inflates `1 - T_off/T_on`, and
    so manufactures support for H1 out of a units error. `startup_before_first_range_seconds`
    below is the quantity that was being discarded; it is reported, never removed.

    Direction of the residual, stated because it decides whether this is safe:
    the session origin precedes the application launch by nsys's own startup, so
    this figure is an **upper bound** on the app-relative readiness time. An
    overstated `T_off` understates the removable share, so any SUPPORTED verdict
    built on it is conservative.
    """
    starts, ends, exclusive = [], [], 0.0
    for row in rows:
        if classify(row["name"]) in COMPILE_START_FAMILIES:
            starts.append(row["start_ns"])
            ends.append(row["start_ns"] + row["duration_ns"])
            exclusive += row["exclusive_ns"] / 1e9
    if not starts:
        return {"status": "BLOCKED", "reason": "no compile ranges in the trace"}

    readiness = max(ends) / 1e9          # from the session origin, startup INCLUDED
    startup = min(starts) / 1e9          # what the old span silently dropped
    span = (max(ends) - min(starts)) / 1e9
    return {
        "status": "OK",
        "measurement_kind": "upper_bound",
        "basis": SESSION_ORIGIN_UPPER_BOUND_BASIS,
        "product_compile_gate_eligible": False,
        "exact_executable_readiness_identified": False,
        "t_off_seconds": readiness,
        "t_off_upper_bound_seconds": readiness,
        "startup_before_first_range_seconds": startup,
        "first_to_last_range_span_seconds": span,
        "exclusive_compile_seconds": exclusive,
        "idle_within_span_seconds": max(0.0, span - exclusive),
        "origin": ("nsys profiling-session start, which precedes application launch; readiness "
                   "is therefore an UPPER bound on app-relative time, and an overstated T_off "
                   "understates the removable share"),
        "why_not_the_span": ("first-range-to-last-range drops process startup. T_on is "
                             "launch-to-kill, so subtracting startup from T_off compares "
                             "unlike clocks and inflates the apparent removable share."),
        "why_not_wall_time": ("subprocess wall time is NOT usable either: this arm runs "
                             "`gpuwrf run --hours 1`, so it is dominated by an hour of "
                             "forecast integration after the executable is ready."),
    }


def attribute(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Exclusive-time attribution by family, with the fail-closed 95% validator."""
    rows = list(rows)
    by_family: dict[str, float] = defaultdict(float)
    by_pass: dict[str, float] = defaultdict(float)
    for row in rows:
        seconds = row["exclusive_ns"] / 1e9
        family = classify(row["name"])
        by_family[family] += seconds
        if family in {"hlo_pass", "autotune"}:
            by_pass[pass_label(row["name"])] += seconds

    # The denominator is COMPILE time only. Thunk execution is reported beside it,
    # never inside it.
    non_compile = {f: s for f, s in by_family.items() if f in NON_COMPILE_FAMILIES}
    total = sum(s for f, s in by_family.items() if f not in NON_COMPILE_FAMILIES)
    unknown = by_family.get("unknown", 0.0)
    unknown_share = (unknown / total) if total > 0 else 1.0
    attributed_share = 1.0 - unknown_share

    if total <= 0:
        status, reason = "BLOCKED", "no exclusive compile time in the trace"
    elif unknown_share > MAX_UNKNOWN_SHARE:
        status, reason = "BLOCKED", (
            f"unknown exclusive time is {unknown_share:.1%}, above the pre-registered "
            f"{MAX_UNKNOWN_SHARE:.0%} budget; attribution is not scored"
        )
    else:
        status, reason = "OK", (
            f"{attributed_share:.1%} of exclusive compile time attributed to named families"
        )

    return {
        "status": status,
        "reason": reason,
        "total_exclusive_compile_seconds": total,
        "non_compile_seconds": non_compile,
        "attributed_share": attributed_share,
        "unknown_share": unknown_share,
        "max_unknown_share": MAX_UNKNOWN_SHARE,
        "by_family_seconds": dict(sorted(by_family.items(), key=lambda kv: -kv[1])),
        "family_share_of_compile": {
            family: (seconds / total if total > 0 else None)
            for family, seconds in sorted(by_family.items(), key=lambda kv: -kv[1])
            if family not in NON_COMPILE_FAMILIES
        },
        "top_passes_seconds": dict(
            sorted(by_pass.items(), key=lambda kv: -kv[1])[:25]),
        "ranges": len(rows),
        "method": ("exclusive (self) time: duration minus direct children. Inclusive "
                   "nvtx_sum durations are NOT used, because nesting would double-count. "
                   "Thunk execution is excluded from the compile denominator."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-csv", type=Path, required=True,
                        help="output of: nsys stats --report nvtx_pushpop_trace --format csv")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = compute_exclusive(parse_pushpop_trace(args.trace_csv.read_text(errors="replace")))
    result = attribute(rows)
    result["source_csv"] = str(args.trace_csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {result['status']} -- {result['reason']}")
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())


#: NVTX range names that delimit one ordinary integration step. §9 requires the
#: transfer audit to be scoped to the timestep loop, so this is what makes the
#: difference between "no copies in the loop" and "no copies anywhere, including
#: the initialisation that legitimately copies inputs to the device".
TIMESTEP_MARKERS = ("timestep", "time_step", "solve_em_step", "integrate_step", "wrf_step")


def timestep_window(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Find one ordinary timestep, or say it is absent.

    Returns MISSING rather than guessing. Falling back to the whole trace would
    silently turn the §7 loop-transfer gate into a whole-process gate, which
    initialisation would fail for legitimate reasons -- or, worse, which would
    pass because the loop's copies were diluted by everything else.
    """
    candidates = [r for r in rows
                  if any(marker in r["name"].lower() for marker in TIMESTEP_MARKERS)]
    if not candidates:
        return {"status": "MISSING",
                "reason": "no NVTX range matches a timestep marker",
                "markers": list(TIMESTEP_MARKERS),
                "needs": ("the forecast must emit an NVTX range per integration step, or the "
                          "loop-scoped transfer gate cannot be evaluated")}
    # the median-duration candidate: an ORDINARY step, not the first (which pays
    # one-off costs) and not the longest (a radiation or output step).
    ordered = sorted(candidates, key=lambda r: r["duration_ns"])
    chosen = ordered[len(ordered) // 2]
    return {
        "status": "OK",
        "window_ns": [chosen["start_ns"], chosen["start_ns"] + chosen["duration_ns"]],
        "name": chosen["name"],
        "candidates": len(candidates),
        "selection": "median-duration timestep range: ordinary, not the first and not the longest",
    }
