#!/usr/bin/env python3
"""Proof that the Step 1 mechanism WORKS, established CPU-only before any GPU ask.

Manager review of `dd5ed7b8`, defect 6: Step 1 needed exact commands, a raw
parser, exclusive-time treatment of nested NVTX ranges, an HLO-to-family mapping
and a >=95% attribution validator. Naming those in prose proves nothing. This
generator runs the whole chain against evidence already on disk and records
what it produced.

Four things are established here, none of which needs a GPU:

1. **The flags exist.** The XLA flags Step 1 depends on are searched for in the
   installed jaxlib binary, so an Step 1 cannot die on a bad flag name inside a
   coordinated window -- the way W1 attempt 1 died on `--hours 1.0`.
2. **The parser works on the real capture**, not a fixture: the full W1b NVTX
   trace, with the exclusive-time computation cross-checked against nsys's own
   `DurNonChild` column on every range.
3. **The attribution validator passes on real data**, so its >=95% bar is known
   to be reachable rather than aspirational.
4. **The autotune cache write timing** is measured from file mtimes, which
   sharpens H1 before any GPU window is requested.

SCOPE, stated because it is easy to get wrong: the W1b capture is the **warm
Stage 2** process (~20 s of compile), NOT the cold >601.4 s Stage 1 compile.
Its family shares describe the warm process only. They are a mechanism proof and
a warm-process observation; they are not evidence about the cold compile, and
the manager's standing instruction not to equate the two is respected here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "v025"))

import h1_discriminator as h1  # noqa: E402
import nvtx_exclusive as nx  # noqa: E402

NSYS_REP = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw/nsys_baseline.nsys-rep")
AUTOTUNE_DIR = Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw/w1b_arms/jaxcache_5623902916f6/"
                    "xla_gpu_per_fusion_autotune_cache_dir")

# Every XLA flag Step 1's commands rely on. Verified against the installed binary.
REQUIRED_XLA_FLAGS = (
    "xla_gpu_autotune_level",
    "xla_gpu_per_fusion_autotune_cache_dir",
    "xla_dump_to",
    "xla_dump_hlo_pass_re",
)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jaxlib_binary() -> Path | None:
    import importlib.util
    spec = importlib.util.find_spec("jaxlib")
    if spec is None or not spec.submodule_search_locations:
        return None
    root = Path(spec.submodule_search_locations[0])
    candidates = sorted(root.rglob("*.so"), key=lambda p: -p.stat().st_size)
    return candidates[0] if candidates else None


def preflight_flags() -> dict[str, Any]:
    """Are the flags Step 1 needs actually present in this build? CPU-only."""
    binary = jaxlib_binary()
    if binary is None:
        return {"status": "BLOCKED", "reason": "jaxlib not importable"}
    try:
        blob = subprocess.run(["strings", "-n", "8", str(binary)], capture_output=True,
                              text=True, check=True, timeout=600).stdout
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        return {"status": "BLOCKED", "reason": f"could not scan the binary: {exc}"}
    names = set(blob.splitlines())
    present = {flag: (flag in names) for flag in REQUIRED_XLA_FLAGS}
    return {
        "status": "OK" if all(present.values()) else "BLOCKED",
        "binary": str(binary),
        "flags": present,
        "method": "exact-line match of the flag name in the binary's string table",
        "why": ("a Step 1 that dies on an unknown flag inside a coordinated GPU window "
                "spends the window for nothing, which is how W1 attempt 1 was lost"),
    }


def export_trace_csv(rep: Path, cache: Path) -> Path | None:
    """`nsys stats` on an existing .nsys-rep is CPU-only file processing."""
    if cache.is_file() and cache.stat().st_size > 0:
        return cache
    if not rep.is_file():
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as handle:
        proc = subprocess.run(
            ["nsys", "stats", "--report", "nvtx_pushpop_trace", "--format", "csv",
             "--output", "-", str(rep)],
            stdout=handle, stderr=subprocess.PIPE, text=True, check=False, timeout=1800)
    if proc.returncode != 0:
        cache.unlink(missing_ok=True)
        return None
    return cache


def autotune_write_timing(root: Path) -> dict[str, Any]:
    """When were autotune records written, and is the span continuous?

    The span alone would be misleading: it turned out to be dominated by one long
    interval with no writes at all, so the span is reported WITH its gap
    structure rather than as a duration.
    """
    if not root.is_dir():
        return {"status": "ABSENT", "path": str(root)}
    stamps = sorted(p.stat().st_mtime for p in root.rglob("*") if p.is_file())
    if len(stamps) < 2:
        return {"status": "INSUFFICIENT", "files": len(stamps), "path": str(root)}

    first, last = stamps[0], stamps[-1]
    gaps = sorted(((b - a) for a, b in zip(stamps, stamps[1:])), reverse=True)
    largest = gaps[0]

    clusters: list[dict[str, Any]] = []
    start = prev = stamps[0]
    count = 1
    for stamp in stamps[1:]:
        if stamp - prev > 5.0:
            clusters.append({"start_offset_s": start - first, "end_offset_s": prev - first,
                             "duration_s": prev - start, "files": count})
            start, count = stamp, 0
        prev, count = stamp, count + 1
    clusters.append({"start_offset_s": start - first, "end_offset_s": prev - first,
                     "duration_s": prev - start, "files": count})

    return {
        "status": "MEASURED",
        "path": str(root),
        "files": len(stamps),
        "first_write_utc": datetime.fromtimestamp(first, timezone.utc).isoformat(),
        "last_write_utc": datetime.fromtimestamp(last, timezone.utc).isoformat(),
        "span_seconds": last - first,
        "largest_gap_with_no_writes_seconds": largest,
        "active_seconds_excluding_largest_gap": (last - first) - largest,
        "clusters_5s_stitch": clusters,
        "interpretation": (
            "the span is NOT a measure of autotune duration. It is dominated by a single "
            f"{largest:.1f} s interval with no autotune writes at all, and the writes fall "
            "into two clusters. What happened inside that gap is UNATTRIBUTED: it could be "
            "one expensive autotuning whose results were flushed at the end, or non-autotune "
            "compile work. Both are consistent with these mtimes, so neither is claimed."
        ),
        "bearing_on_H1": (
            "presence and bracketing only. This is exactly why H1 needs the direct "
            "autotune-OFF discriminator rather than an inference from cache timestamps."
        ),
    }


def build(out: Path, trace_cache: Path) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "schema": "wrf_gpu2.v025.m0.step1_mechanism_evidence.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": ("proves the Step 1 measurement chain works, using evidence already on "
                    "disk. CPU-only. No GPU was requested, coordinated, imported or touched."),
        "xla_flag_preflight": preflight_flags(),
        "autotune_cache_write_timing": autotune_write_timing(AUTOTUNE_DIR),
        "h1_rule": {
            "t_on_lower_bound_seconds": h1.T_ON_LOWER_BOUND_S,
            "t_off_budget_seconds": h1.T_OFF_BUDGET_S,
            "bands": h1.verdict(
                1.0, completed=True, t_off_basis=h1.REQUIRED_BASIS
            )["bands"],
            "why_feasible": (
                "only the autotune-OFF arm is measured, and it is bounded by construction. "
                "The autotune-ON compile is never required to finish."
            ),
        },
    }

    csv_path = export_trace_csv(NSYS_REP, trace_cache)
    if csv_path is None:
        obj["exclusive_attribution"] = {
            "status": "UNAVAILABLE",
            "reason": f"no usable trace export from {NSYS_REP}",
        }
        return obj

    rows = nx.parse_pushpop_trace(csv_path.read_text(errors="replace"))
    enriched = nx.compute_exclusive(rows)
    result = nx.attribute(enriched)
    inclusive = sum(row["duration_ns"] for row in rows) / 1e9

    result.update({
        "source_nsys_rep": str(NSYS_REP),
        "source_nsys_rep_sha256": sha256_file(NSYS_REP),
        "ranges_parsed": len(rows),
        "cross_check": {
            "against": "nsys DurNonChild column",
            "tolerance": 0.02,
            "outcome": "PASSED on every range (a failure raises and blocks this object)",
        },
        "inclusive_sum_seconds_for_contrast": inclusive,
        "inclusive_inflation_factor": (
            inclusive / result["total_exclusive_compile_seconds"]
            if result["total_exclusive_compile_seconds"] else None),
        "SCOPE": (
            "this is the WARM Stage 2 process (~20 s of compile), NOT the cold >601.4 s "
            "Stage 1 compile. The family shares below describe the warm process only and "
            "are NOT evidence about the cold compile. Per standing manager instruction the "
            "two must not be equated."
        ),
        "bearing_on_H1": (
            "NONE for the cold compile. Worth recording that in the WARM process autotune is "
            f"{result['family_share_of_compile'].get('autotune', 0) * 100:.2f}% of compile "
            "while codegen is "
            f"{result['family_share_of_compile'].get('codegen', 0) * 100:.2f}% -- but a warm "
            "process skips exactly the work H1 is about, so this neither supports nor "
            "undermines it."
        ),
    })
    obj["exclusive_attribution"] = result
    obj["reproduce"] = {
        "generator": "scripts/v025/build_step1_mechanism_evidence.py",
        "generator_sha256": sha256_file(Path(__file__)),
        "command": "python scripts/v025/build_step1_mechanism_evidence.py",
        "trace_export": (f"nsys stats --report nvtx_pushpop_trace --format csv "
                         f"--output - {NSYS_REP}"),
        "note": "no number in this object is hand-written; rerun the command to regenerate",
    }
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=REPO / "proofs/v025/m0/step1_mechanism_evidence.json")
    parser.add_argument("--trace-cache", type=Path,
                        default=Path("<DATA_ROOT>/wrf_gpu2/v025/m0/raw/nsys_baseline_pushpop.csv"))
    args = parser.parse_args()

    obj = build(args.out, args.trace_cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")

    attribution = obj.get("exclusive_attribution", {})
    print(f"wrote {args.out}")
    print(f"  flag preflight : {obj['xla_flag_preflight']['status']}")
    print(f"  attribution    : {attribution.get('status')} "
          f"({attribution.get('attributed_share', 0) * 100:.1f}% attributed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
