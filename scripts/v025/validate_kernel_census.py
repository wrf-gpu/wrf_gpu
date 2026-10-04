#!/usr/bin/env python3
"""Validate the kernel census: attribution bars and ordinary/event exposure (§9)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _validate_lib import Report, load_json

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path)
    ap.add_argument("--min-attribution", type=float, default=0.95)
    args = ap.parse_args()
    rep = Report("kernel_census", args.path)
    obj = load_json(args.path, rep)
    if obj is None:
        return rep.emit()

    attr = obj.get("attribution") or {}
    for key in ("launches_attributed_fraction", "device_time_attributed_fraction"):
        value = attr.get(key)
        if value is None:
            rep.missing(f"attribution:{key}", f">= {args.min_attribution} required")
        else:
            rep.require(f"attribution:{key}", value >= args.min_attribution,
                        f"{value:.4f} vs >= {args.min_attribution}")
    for key in ("launches_unknown_fraction", "device_time_unknown_fraction"):
        value = attr.get(key)
        if value is None:
            rep.missing(f"attribution:{key}", "unknown is capped at 5%")
        else:
            rep.require(f"attribution:{key}", value <= 0.05, f"{value:.4f} vs <= 0.05")

    steps = obj.get("steps") or {}
    if not steps.get("ordinary"):
        rep.missing("steps:ordinary", "§9 requires one ordinary step captured")
    else:
        rep.require("steps:ordinary", True, "present")
    if steps.get("events") is None:
        rep.missing("steps:events", "§9 requires EVERY distinct event/radiation step")
    else:
        rep.require("steps:events_present", len(steps["events"]) > 0,
                    f"{len(steps['events'])} event steps captured")

    for key in ("executed_kernels_per_step", "device_time_per_step_ms",
                "inter_kernel_gap_distribution", "cold_compile_seconds",
                "cached_load_seconds", "warm_step_ms", "peak_vram_bytes",
                "peak_host_ram_bytes", "raw_artifacts"):
        if obj.get(key) is None:
            rep.missing(f"field:{key}", "required by §9")
        else:
            rep.require(f"field:{key}", True, "present")

    transfers = obj.get("host_device_transfer_audit") or {}
    n = transfers.get("timestep_loop_transfers")
    if n is None:
        rep.missing("transfer_audit", "§9 requires a transfer audit with zero loop transfers")
    else:
        rep.require("transfer:zero_in_timestep_loop", n == 0, f"{n} transfers in the loop")

    tops = obj.get("top_kernels") or []
    if not tops:
        rep.missing("top_kernels", "flat un-attributed kernel lists do not pass")
    else:
        need = {"name", "launches", "duration_ns", "registers", "achieved_occupancy"}
        complete = [k for k in tops if need.issubset(k)]
        rep.require("top_kernels:fields_complete", len(complete) == len(tops),
                    f"{len(complete)}/{len(tops)} carry {sorted(need)}")
    return rep.emit()

if __name__ == "__main__":
    raise SystemExit(main())
