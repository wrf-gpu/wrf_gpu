"""Task 3 -- XLA fusion + host/device transfer audit of the warmed coupled scan.

Confirms the warmed ``run_forecast_operational`` executes device-resident with no
host<->device transfers inside the timestep loop, counts kernels / fusions in the
compiled program, and surfaces the compile-cost structure (recompile drivers).

Method:
  * Build the real d02 case (same config as the +1h/+3h skill proof).
  * Lower+compile run_forecast_operational at a fixed hours; introspect the
    compiled executable's HLO for: total instructions, fusion count, any
    copy-start/copy-done / outfeed / infeed / send / recv (= host transfers),
    and the number of distinct ``scan``/``while`` regions (the compile-cost
    driver -- the Python radiation-cadence loop emits one scan per radiation
    interval).
  * Run a WARMED forecast under jax.profiler.trace and scan the trace for any
    post-init D2H/H2D memcpy events (gpuwrf.profiling.transfer_audit).
  * Check donate_argnums / static_argnames already declared on the public entry.

Run:
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-3 \
    python proofs/perf/fusion_transfer_audit.py --hours 0.5
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

import jax

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.profiling.transfer_audit import count_transfer_bytes
from gpuwrf.runtime.operational_mode import run_forecast_operational

PROOF = Path("proofs/perf")


def _hlo_text(state, nl, hours: float) -> str:
    lowered = jax.jit(
        run_forecast_operational, static_argnames=("hours",), donate_argnums=(0,)
    ).lower(state, nl, hours=float(hours))
    compiled = lowered.compile()
    try:
        return compiled.as_text()
    except Exception:
        # Fall back to the optimized HLO from the lowering.
        return lowered.as_text()


def _scan_hlo(hlo: str) -> dict:
    def count(pat: str) -> int:
        return len(re.findall(pat, hlo))

    return {
        "total_lines": hlo.count("\n"),
        "fusion_instructions": count(r"\bfusion\b"),
        "kind_kLoop": count(r"kind=kLoop"),
        "kind_kInput": count(r"kind=kInput"),
        "kind_kOutput": count(r"kind=kOutput"),
        "custom_call": count(r"custom-call\("),
        "while_loops": count(r"\bwhile\("),
        "scan_or_while_regions": count(r"\bwhile\("),
        # Host/device transfer ops (must be 0 inside the loop):
        "copy_start": count(r"copy-start\("),
        "copy_done": count(r"copy-done\("),
        "outfeed": count(r"\boutfeed\b"),
        "infeed": count(r"\binfeed\b"),
        "send": count(r"\bsend\("),
        "recv": count(r"\brecv\("),
        "dynamic_update_slice": count(r"dynamic-update-slice\("),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=0.5)
    args = ap.parse_args()
    hours = float(args.hours)

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    nl = case.namelist

    # --- HLO introspection (kernel/fusion/transfer counts) ---
    hlo = _hlo_text(case.state, nl, hours)
    hlo_stats = _scan_hlo(hlo)

    # --- warmed trace transfer audit ---
    state2 = _build_real_case(cfg)[0].state
    # warm compile
    out = run_forecast_operational(state2, nl, hours)
    jax.block_until_ready(out.theta)
    state3 = _build_real_case(cfg)[0].state
    trace_dir = Path(tempfile.mkdtemp(prefix="perf_trace_"))
    with jax.profiler.trace(str(trace_dir)):
        out = run_forecast_operational(state3, nl, hours)
        jax.block_until_ready(out.theta)
    h2d, d2h, matched = count_transfer_bytes(trace_dir)

    # Count distinct scan calls the Python radiation-cadence loop emits.
    dt_s = float(nl.dt_s)
    cadence = int(nl.radiation_cadence_steps)

    def count_scan_calls(h: float) -> int:
        steps = int(round(h * 3600.0 / dt_s))
        n = 0
        step = 1
        while step <= steps:
            nxt = ((step + cadence - 1) // cadence) * cadence
            if bool(nl.run_physics) and nxt <= steps:
                if nxt - step:
                    n += 1
                n += 1
                step = nxt + 1
            else:
                n += 1
                step = steps + 1
        return n

    out_json = {
        "scope": "Task 3 -- XLA fusion + host/device transfer audit (warmed coupled scan)",
        "run_dir": str(run_dir),
        "hours_profiled": hours,
        "device": str(jax.devices()[0]),
        "config": {
            "radiation_cadence_steps": cadence,
            "run_physics": bool(nl.run_physics),
            "force_fp64": bool(nl.force_fp64),
        },
        "public_entry_decorators": {
            "static_argnames": ["hours"],
            "donate_argnums": [0],
            "note": "run_forecast_operational already declares donate_argnums=(0,) and static hours.",
        },
        "hlo_stats": hlo_stats,
        "transfer_audit": {
            "method": "jax.profiler.trace scanned for post-init memcpy events",
            "host_to_device_bytes_post_init": int(h2d),
            "device_to_host_bytes_post_init": int(d2h),
            "trace_dir": str(trace_dir),
            "transfer_event_files": matched,
            "d2h_inter_kernel_verdict": "0 D2H bytes inside the warmed loop"
            if d2h == 0 else f"{d2h} D2H bytes detected -- investigate",
        },
        "compile_cost_driver": {
            "distinct_scan_calls_emitted_by_python_radiation_loop": {
                f"{h}h": count_scan_calls(h) for h in (0.1, 0.5, 1.0, 3.0, 6.0, 24.0, 72.0)
            },
            "explanation": (
                "run_forecast_operational's Python while-loop emits one jax.lax.scan "
                "per radiation interval (nonrad scan + 1-step rad scan). Each is a "
                "distinct XLA subcomputation, so compile time scales ~linearly with "
                "the number of radiation intervals -> 96 scans at 24h, 288 at 72h. "
                "This is the compile-blowup root cause (warmed throughput is unaffected; "
                "RRTMG only fires at cadence). Lossless single-scan remedy: make "
                "run_radiation a traced (step_index %% cadence == 0) predicate and "
                "gate rrtmg_adapter with jax.lax.cond -> ONE scan for the whole "
                "forecast. Numerically identical (same RRTMG cadence); recommended as "
                "a gated follow-up since it edits the compiled control flow."
            ),
        },
        "fusion_verdict": (
            f"{hlo_stats['fusion_instructions']} fusion instructions in the compiled "
            f"program; {hlo_stats['custom_call']} custom-calls; "
            f"{hlo_stats['copy_start'] + hlo_stats['outfeed'] + hlo_stats['infeed'] + hlo_stats['send'] + hlo_stats['recv']} "
            "host-transfer ops (copy-start/outfeed/infeed/send/recv)."
        ),
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    fn = PROOF / "fusion_transfer_audit.json"
    fn.write_text(json.dumps(out_json, indent=2) + "\n")
    # also save the HLO for the record
    (PROOF / "run_forecast_operational_hlo.txt").write_text(hlo)
    print(json.dumps({k: out_json[k] for k in ("hlo_stats", "transfer_audit", "compile_cost_driver", "fusion_verdict")}, indent=2))
    print(f"\nwrote {fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
