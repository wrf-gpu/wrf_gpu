#!/usr/bin/env python3
"""Parsers for the Phase-B profiler captures (contract §8, §9).

Written and tested on CPU **before** the window. A captured trace that turns out
to be unparseable afterwards is a wasted coordinated window and a second ask of
two neighbouring managers, so the reduction path is proven against realistic
fixtures first. `tests/v025/test_parse_profiler.py` holds those fixtures.

Two inputs:

* ``nsys stats --report cuda_gpu_kern_sum --format csv`` — per-kernel device
  time and launch counts, which feed `run_gpu_arm.attribute_device_time` and
  therefore §9's attribution bars and the manager's physics-vs-dycore
  discriminator.
* ``ncu --csv`` — per-kernel hardware counters, which feed §8's `F_measured`.

Both parsers fail closed. §16 makes an unavailable counter a **blocked
denominator**, never a licence to fall back on the kernel report's ±2x estimate,
so a missing metric produces an explicit `MISSING` marker in the result rather
than a zero that would silently flow into an arithmetic mean.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

# §8's op classes. fp32 and fp64 are kept apart on purpose: the entire v0.25
# thesis is that the model is paying fp64 rates for work WRF does in fp32, so a
# combined "FLOP" number would erase the finding it exists to measure.
FLOP_COUNTERS: dict[str, tuple[str, int]] = {
    # counter suffix -> (op class, flops per instruction)
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum": ("f32_add", 1),
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum": ("f32_mul", 1),
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum": ("f32_fma", 2),
    "sm__sass_thread_inst_executed_op_dadd_pred_on.sum": ("f64_add", 1),
    "sm__sass_thread_inst_executed_op_dmul_pred_on.sum": ("f64_mul", 1),
    "sm__sass_thread_inst_executed_op_dfma_pred_on.sum": ("f64_fma", 2),
}

MEMORY_COUNTERS = ("dram__bytes.sum", "lts__t_bytes.sum")
OCCUPANCY_COUNTER = "sm__warps_active.avg.pct_of_peak_sustained_active"


class ProfilerParseError(RuntimeError):
    """Raised when a capture cannot be reduced; never downgraded to a warning."""


# nsys prints a small, known set of informational lines before the CSV body.
# Tolerating *only* these keeps a real defect visible: an unrecognised preamble
# line means the export did something this reduction has never seen, and reading
# past it would risk scoring a table produced by something other than this run.
VERIFIED_NSYS_PREAMBLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\*\* [A-Za-z0-9][A-Za-z0-9 _/()+.%:-]* "
        r"\([a-z0-9_]+(?::[^)]*)?\)$"
    ),
    re.compile(
        r"^Generating SQLite file \S+\.sqlite from \S+\.nsys-rep$"
    ),
    re.compile(
        r"^Processing \[\S+\.sqlite\] with "
        r"\[\S+/[a-z0-9_]+\.py\]\.\.\.$"
    ),
    re.compile(r"^Generating NVTX Push/Pop Range Trace(?: \[STUB\])?\.\.\.$"),
)

#: The one preamble that is never tolerated: it means `nsys stats` reused an
#: existing sibling `.sqlite` instead of re-exporting the report we hashed, so
#: the table can belong to a different run entirely.
STALE_EXPORT_MARKER = "Existing SQLite export found"


def verified_preamble(lines: list[str], header_index: int) -> list[str]:
    """Return the preamble lines, refusing anything unverified or stale."""

    preamble = lines[:header_index]
    for line in preamble:
        text = line.strip()
        if not text:
            continue
        if STALE_EXPORT_MARKER in text:
            raise ProfilerParseError(
                "nsys reused an existing SQLite export: "
                f"{text!r}. A stale sibling .sqlite can carry a different run's "
                "rows, so this table is refused. Re-export with --force-export "
                "into a private output root."
            )
        if not any(pattern.fullmatch(text) for pattern in
                   VERIFIED_NSYS_PREAMBLE_PATTERNS):
            raise ProfilerParseError(
                f"unverified nsys preamble line before the CSV header: {text!r}"
            )
    return preamble


# --------------------------------------------------------------------------- #
# nsys                                                                         #
# --------------------------------------------------------------------------- #
_NUMERIC = re.compile(r"^-?[\d.,]+$")


def _number(value: str) -> float:
    """Parse a possibly thousands-separated numeric cell."""
    text = (value or "").strip().replace(",", "")
    if not text or text in {"-", "N/A", "n/a"}:
        return 0.0
    try:
        return float(text)
    except ValueError as exc:
        raise ProfilerParseError(f"non-numeric cell {value!r}") from exc


def parse_nsys_kernel_summary(text: str) -> list[dict[str, Any]]:
    """Reduce `cuda_gpu_kern_sum` CSV to the shape the attributor consumes.

    nsys prefixes its CSV with banner lines and may quote the header, so the
    header row is located by content rather than by position. Returns
    ``[{"name", "device_time_ns", "launches", "avg_ns", "time_percent"}]``.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    header_index = None
    for index, line in enumerate(lines):
        lowered = line.lower()
        if "name" in lowered and ("instances" in lowered or "count" in lowered):
            header_index = index
            break
    if header_index is None:
        raise ProfilerParseError(
            "no cuda_gpu_kern_sum header row found. Expected a CSV header containing "
            "'Name' and 'Instances'; got first line: " + (lines[0] if lines else "<empty>")
        )
    verified_preamble(lines, header_index)

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    if reader.fieldnames is None:
        raise ProfilerParseError("kernel summary has no columns")
    fields = {name.strip().lower(): name for name in reader.fieldnames}

    def column(*candidates: str) -> str:
        for candidate in candidates:
            for lowered, original in fields.items():
                if candidate in lowered:
                    return original
        raise ProfilerParseError(
            f"no column matching {candidates} in {list(fields)}"
        )

    name_col = column("name")
    time_col = column("total time", "total_time")
    count_col = column("instances", "count")
    percent_col = None
    try:
        percent_col = column("time (%)", "time(%)", "percent")
    except ProfilerParseError:
        pass

    rows: list[dict[str, Any]] = []
    for row in reader:
        name = (row.get(name_col) or "").strip()
        if not name or name.lower() == "name":
            continue
        rows.append({
            "name": name,
            "device_time_ns": _number(row.get(time_col, "")),
            "launches": int(_number(row.get(count_col, ""))),
            "time_percent": _number(row.get(percent_col, "")) if percent_col else None,
        })
    if not rows:
        raise ProfilerParseError("kernel summary parsed to zero kernels")
    return rows


def parse_cuda_gpu_trace(text: str) -> list[dict[str, Any]]:
    """Parse timestamped ``cuda_gpu_trace`` rows.

    The transfer audit deliberately rejects ``cuda_gpu_mem_time_sum`` here.  A
    whole-process aggregate can say that copies happened, but it cannot place
    them inside or outside an integration interval.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "Start (ns)" in line and "Duration (ns)" in line and "Name" in line
        ),
        None,
    )
    if header_index is None:
        raise ProfilerParseError(
            "no cuda_gpu_trace header row found; a timestamped table with "
            "'Start (ns)', 'Duration (ns)', and 'Name' is required"
        )
    verified_preamble(lines, header_index)

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    fields = {name.strip(): name for name in (reader.fieldnames or [])}
    required = ("Start (ns)", "Duration (ns)", "Name")
    missing = [name for name in required if name not in fields]
    if missing:
        raise ProfilerParseError(f"cuda_gpu_trace missing columns {missing}")

    rows: list[dict[str, Any]] = []
    for raw in reader:
        clean = {(key or "").strip(): (value or "").strip()
                 for key, value in raw.items() if key}
        name = clean.get("Name", "")
        if not name:
            continue
        start_ns = _number(clean.get("Start (ns)", ""))
        duration_ns = _number(clean.get("Duration (ns)", ""))
        if not math.isfinite(start_ns) or not math.isfinite(duration_ns):
            raise ProfilerParseError("cuda_gpu_trace contains a non-finite timestamp")
        if start_ns < 0 or duration_ns < 0:
            raise ProfilerParseError("cuda_gpu_trace contains a negative timestamp or duration")
        src = clean.get("SrcMemKd", "")
        dst = clean.get("DstMemKd", "")
        rows.append({
            "name": name,
            "start_ns": start_ns,
            "duration_ns": duration_ns,
            "end_ns": start_ns + duration_ns,
            "bytes": _number(clean.get("Bytes (MB)", "")) * 1_000_000.0,
            "source_memory_kind": src or None,
            "destination_memory_kind": dst or None,
            "device": clean.get("Device") or None,
            "context": clean.get("Ctx") or None,
            "stream": clean.get("Strm") or None,
        })
    if not rows:
        raise ProfilerParseError(
            "cuda_gpu_trace parsed to zero timestamped rows; zero transfers require an "
            "explicit profiler certificate, not an empty/aggregate table"
        )
    return rows


def parse_nsys_nvtx_summary(text: str) -> list[dict[str, Any]]:
    """Reduce `nsys stats --report nvtx_sum --format csv` to named ranges.

    XLA emits NVTX push/pop ranges for every compilation module and pass
    (``TSL:XlaCompile:#module=...#``, ``pre-spmd``, ``optimization``,
    ``call-inliner``, ...). A capture that contains **no CUDA kernel data at all**
    can still carry a complete compile-pass breakdown, which is exactly the case
    for the W1b artifact: `cuda_gpu_kern_sum` is empty while `nvtx_sum` has
    hundreds of ranges. "Unusable" had to be scoped to the kernel census rather
    than applied to the whole file.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    header_index = None
    for index, line in enumerate(lines):
        lowered = line.lower()
        if "range" in lowered and "total time" in lowered:
            header_index = index
            break
    if header_index is None:
        raise ProfilerParseError(
            "no nvtx_sum header row found (expected a CSV header with 'Range' and 'Total Time')"
        )
    verified_preamble(lines, header_index)

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    fields = {name.strip().lower(): name for name in (reader.fieldnames or [])}

    def column(*candidates: str) -> str:
        for candidate in candidates:
            for lowered, original in fields.items():
                if candidate in lowered:
                    return original
        raise ProfilerParseError(f"no column matching {candidates} in {list(fields)}")

    range_col = column("range")
    total_col = column("total time")
    count_col = column("instances", "count")

    rows: list[dict[str, Any]] = []
    for row in reader:
        name = (row.get(range_col) or "").strip()
        if not name or name.lower() == "range":
            continue
        rows.append({
            "range": name,
            "total_ns": _number(row.get(total_col, "")),
            "instances": int(_number(row.get(count_col, "")) or 0),
        })
    if not rows:
        raise ProfilerParseError("nvtx summary parsed to zero ranges")
    return rows


XLA_MODULE = re.compile(r"XlaCompile:#module=([^#]+)#")


def compile_breakdown(ranges: list[dict[str, Any]]) -> dict[str, Any]:
    """Split NVTX ranges into XLA module compiles and compiler passes."""
    modules: dict[str, dict[str, Any]] = {}
    passes: list[dict[str, Any]] = []
    for row in ranges:
        match = XLA_MODULE.search(row["range"])
        if match:
            modules[match.group(1)] = {
                "seconds": row["total_ns"] / 1e9, "instances": row["instances"],
            }
        else:
            passes.append({
                "pass": row["range"], "seconds": row["total_ns"] / 1e9,
                "instances": row["instances"],
            })
    passes.sort(key=lambda p: -p["seconds"])
    return {
        "modules": dict(sorted(modules.items(), key=lambda kv: -kv[1]["seconds"])),
        "module_compile_seconds_total": sum(m["seconds"] for m in modules.values()),
        "top_passes": passes[:25],
        "pass_count": len(passes),
    }


# --------------------------------------------------------------------------- #
# ncu                                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class CounterAvailability:
    """Which §8 counters this card/driver actually produced."""

    present: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def flop_counters_complete(self) -> bool:
        return not [c for c in FLOP_COUNTERS if c in self.missing]

    def as_dict(self) -> dict[str, Any]:
        return {
            "present": list(self.present),
            "missing": list(self.missing),
            "flop_counters_complete": self.flop_counters_complete,
            "blocked_denominator": not self.flop_counters_complete,
            "note": (
                "§16: a missing counter BLOCKS the denominator. It is never a licence to "
                "reuse the kernel report's +/-2x FLOP/cell/step estimate."
            ),
        }


def parse_ncu_csv(text: str) -> dict[str, Any]:
    """Reduce `ncu --csv` output to per-kernel counter values plus availability.

    ncu emits one row per (kernel, metric). Rows whose value is empty or `n/a`
    mean the metric was **not collected** — recorded as missing rather than
    coerced to zero, because a zero would silently reduce `F_measured` and make
    the model look more efficient than it is.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    header_index = None
    for index, line in enumerate(lines):
        lowered = line.lower()
        if "kernel name" in lowered and "metric name" in lowered:
            header_index = index
            break
    if header_index is None:
        raise ProfilerParseError(
            "no ncu CSV header found (expected 'Kernel Name' and 'Metric Name' columns)"
        )

    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    fields = {name.strip().lower(): name for name in (reader.fieldnames or [])}
    kernel_col = fields.get("kernel name")
    metric_col = fields.get("metric name")
    value_col = fields.get("metric value")
    if not (kernel_col and metric_col and value_col):
        raise ProfilerParseError(f"ncu CSV missing required columns; got {list(fields)}")

    per_kernel: dict[str, dict[str, float]] = {}
    seen: set[str] = set()
    unavailable: set[str] = set()
    for row in reader:
        kernel = (row.get(kernel_col) or "").strip()
        metric = (row.get(metric_col) or "").strip()
        raw = (row.get(value_col) or "").strip()
        if not kernel or not metric:
            continue
        seen.add(metric)
        if raw.lower() in {"", "n/a", "-", "nan"}:
            unavailable.add(metric)
            continue
        per_kernel.setdefault(kernel, {})[metric] = _number(raw)

    wanted = tuple(FLOP_COUNTERS) + MEMORY_COUNTERS + (OCCUPANCY_COUNTER,)
    missing = tuple(m for m in wanted if m not in seen or m in unavailable)
    availability = CounterAvailability(
        present=tuple(m for m in wanted if m in seen and m not in unavailable),
        missing=missing,
    )
    return {"per_kernel": per_kernel, "availability": availability.as_dict()}


def useful_flops(per_kernel: dict[str, dict[str, float]]) -> dict[str, Any]:
    """§8's `F_measured`: useful floating operations, by class, never merged.

    FMA counts as two operations. fp32 and fp64 totals are reported separately
    AND together; the separate numbers are the load-bearing ones, because the
    v0.25 thesis is precisely that fp64 work is being done where WRF uses fp32.
    """
    by_class: dict[str, float] = {}
    for metrics in per_kernel.values():
        for counter, (op_class, weight) in FLOP_COUNTERS.items():
            if counter in metrics:
                by_class[op_class] = by_class.get(op_class, 0.0) + metrics[counter] * weight
    f32 = sum(v for k, v in by_class.items() if k.startswith("f32"))
    f64 = sum(v for k, v in by_class.items() if k.startswith("f64"))
    return {
        "by_class": by_class,
        "f32_flops": f32,
        "f64_flops": f64,
        "total_flops": f32 + f64,
        "f64_fraction": (f64 / (f32 + f64)) if (f32 + f64) else None,
        "fma_weighting": "FMA counted as 2 operations; add/mul as 1",
        "note": (
            "fp32 and fp64 are reported separately on purpose: merging them would erase "
            "the very gap v0.25 exists to close."
        ),
    }


def summarise_capture(
    *, nsys_text: str | None = None, ncu_text: str | None = None
) -> dict[str, Any]:
    """Reduce one window's captures; each half is optional but never faked."""
    result: dict[str, Any] = {}
    if nsys_text is not None:
        kernels = parse_nsys_kernel_summary(nsys_text)
        result["kernels"] = kernels
        result["kernel_count"] = len(kernels)
        result["total_device_time_ns"] = sum(k["device_time_ns"] for k in kernels)
        result["total_launches"] = sum(k["launches"] for k in kernels)
    if ncu_text is not None:
        parsed = parse_ncu_csv(ncu_text)
        result["counters"] = parsed["availability"]
        result["F_measured"] = (
            useful_flops(parsed["per_kernel"])
            if parsed["availability"]["flop_counters_complete"]
            else {
                "status": "BLOCKED",
                "reason": (
                    "required FLOP counters were not collected: "
                    f"{parsed['availability']['missing']}. §16 makes this a blocked "
                    "denominator, not permission to estimate."
                ),
            }
        )
    if not result:
        raise ProfilerParseError("nothing to summarise: both captures were None")
    return result


__all__ = [
    "CounterAvailability",
    "compile_breakdown",
    "parse_nsys_nvtx_summary",
    "FLOP_COUNTERS",
    "MEMORY_COUNTERS",
    "OCCUPANCY_COUNTER",
    "ProfilerParseError",
    "STALE_EXPORT_MARKER",
    "VERIFIED_NSYS_PREAMBLE_PATTERNS",
    "verified_preamble",
    "parse_ncu_csv",
    "parse_cuda_gpu_trace",
    "parse_nsys_kernel_summary",
    "summarise_capture",
    "useful_flops",
]


# --------------------------------------------------------------------------- #
# generic nsys `*_sum` CSV readers, added for baseline_census                   #
# --------------------------------------------------------------------------- #
def parse_generic_sum(text: str) -> list[dict[str, Any]]:
    """Any nsys `--report *_sum --format csv` table with a Name and a time column.

    Kept deliberately tolerant about WHICH time column is present (nsys names it
    `Total Time (ns)`, `Total Time`, or `Duration (ns)` depending on the report)
    and strict about there being one: a summary row whose duration cannot be
    located is dropped from the numerator AND the denominator rather than being
    counted as zero, which would silently deflate every share computed from it.
    """
    import csv as _csv
    import io as _io

    # nsys names the label column `Name` in kernel/memory reports and `Range` in
    # nvtx projection reports. Accepting only `Name` silently produced an empty
    # table for nvtx_gpu_proj_sum -- and an empty table reads as "no device time"
    # rather than as "wrong column", which is the dangerous kind of parse failure.
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if ("Name" in line or "Range" in line or "Operation" in line)
                  and ("Time" in line or "Duration" in line)), None)
    if start is None:
        if any(line.strip() for line in lines):
            raise ProfilerParseError("generic nsys summary has no recognized header")
        return []
    verified_preamble(lines, start)

    rows: list[dict[str, Any]] = []
    for raw in _csv.DictReader(_io.StringIO("\n".join(lines[start:]))):
        clean = {(k or "").strip(): (v or "").strip() for k, v in raw.items() if k}
        name = clean.get("Name") or clean.get("Range") or clean.get("Operation")
        if not name:
            continue
        seconds = None
        for column, scale in (("Total Time (ns)", 1e-9), ("Duration (ns)", 1e-9),
                              ("Total Time", 1e-9), ("Total Time (s)", 1.0)):
            if clean.get(column):
                seconds = _number(clean[column]) * scale
                break
        if seconds is None:
            continue
        rows.append({"name": name, "seconds": seconds,
                     "instances": int(_number(clean.get("Instances", "0") or 0))})
    return rows


def parse_memory_usage(text: str) -> list[dict[str, Any]]:
    """`--cuda-memory-usage=true` samples: whatever carries a byte count."""
    import csv as _csv
    import io as _io

    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if "Bytes" in line or "bytes" in line), None)
    if start is None:
        if any(line.strip() for line in lines):
            raise ProfilerParseError("nsys memory usage has no recognized header")
        return []
    verified_preamble(lines, start)
    rows: list[dict[str, Any]] = []
    for raw in _csv.DictReader(_io.StringIO("\n".join(lines[start:]))):
        clean = {(k or "").strip(): (v or "").strip() for k, v in raw.items() if k}
        for column in ("Bytes", "Total Bytes", "Memory Usage (bytes)", "bytes"):
            if clean.get(column):
                rows.append({"bytes": int(_number(clean[column])),
                             "name": clean.get("Name", "")})
                break
    return rows
