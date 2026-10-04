#!/usr/bin/env python
"""Compile ADR-031 S2 forecast shapes and summarize optimized HLO.

This is a bottleneck probe, not an acceptance oracle.  It records XLA memory
analysis plus coarse dtype/op counts from the optimized HLO text so we can see
whether MIXED_PERTURB_FP32 actually removes fp64 work from the compiled graph.
Run under the GPU lock.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import jax

from gpuwrf.runtime.operational_mode import _run_forecast_operational_jit

from proofs.perf.v016 import fp32_s2_mixed_ladder as ladder


DTYPES = ("f64", "f32", "s64", "s32", "pred")
OP_RE = re.compile(r"\b([a-z][a-z0-9_-]*)\(")


def _memory_dict(compiled) -> dict[str, int | None]:
    mem = compiled.memory_analysis()
    if mem is None:
        return {}
    names = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_temp_size_in_bytes",
    )
    return {name: getattr(mem, name, None) for name in names}


def _hlo_counts(hlo: str) -> dict[str, object]:
    lines = hlo.splitlines()
    dtype_lines = {dtype: 0 for dtype in DTYPES}
    dtype_tokens = {dtype: hlo.count(dtype + "[") for dtype in DTYPES}
    for line in lines:
        for dtype in DTYPES:
            if dtype + "[" in line:
                dtype_lines[dtype] += 1
    ops: dict[str, int] = {}
    converts: dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("HloModule", "//")):
            continue
        if " convert(" in stripped or stripped.startswith("convert"):
            match = re.search(r"=\s*([fsu][0-9]+|pred)\[", stripped)
            if match:
                dtype = match.group(1)
                converts[dtype] = converts.get(dtype, 0) + 1
        for op in OP_RE.findall(stripped):
            ops[op] = ops.get(op, 0) + 1
    return {
        "line_count": len(lines),
        "dtype_lines": dtype_lines,
        "dtype_tokens": dtype_tokens,
        "top_ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])[:40]),
        "convert_result_dtypes": dict(sorted(converts.items())),
    }


def compile_one(base_state, base_nl, ny0: int, nx0: int, fy: int, fx: int, precision: str, steps: int) -> dict:
    nl = ladder._namelist(base_nl, ny0, nx0, fy, fx, precision)
    state = ladder._fresh_state(base_state, ny0, nx0, fy, fx, precision)
    hours = float(steps) * float(base_nl.dt_s) / 3600.0
    t0 = time.perf_counter()
    lowered = _run_forecast_operational_jit.lower(state, nl, float(hours))
    compiled = lowered.compile()
    compile_s = time.perf_counter() - t0
    hlo = compiled.as_text()
    return {
        "precision": precision,
        "tile_factor": [fy, fx],
        "ny": fy * ny0,
        "nx": fx * nx0,
        "ncol": fy * fx * ny0 * nx0,
        "steps": int(steps),
        "hours": hours,
        "compile_s": compile_s,
        "memory_analysis": _memory_dict(compiled),
        "cost_analysis": compiled.cost_analysis(),
        "hlo_counts": _hlo_counts(hlo),
    }


def _parse_tiles(raw: str) -> list[tuple[int, int]]:
    return ladder._parse_tiles(raw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--tiles", default="2x2")
    ap.add_argument("--precisions", default="fp64,mixed_s2")
    ap.add_argument("--out", type=Path, default=Path("proofs/perf/v016/s2_hlo_stats.json"))
    args = ap.parse_args(argv)

    cfg = ladder.DailyPipelineConfig(
        run_id=ladder.ANCHOR_RUN_ID,
        run_root=ladder.ANCHOR_RUN_ROOT,
        domain=ladder.ANCHOR_DOMAIN,
        hours=1,
        dt_s=ladder.ANCHOR_DT_S,
        acoustic_substeps=10,
    )
    case, _ = ladder._build_real_case(cfg)
    base_state = case.state
    base_nl = case.namelist
    ny0, nx0 = int(case.grid.ny), int(case.grid.nx)
    records = []
    for fy, fx in _parse_tiles(args.tiles):
        for precision in [p.strip() for p in args.precisions.split(",") if p.strip()]:
            print(f"[compile] {precision} {fy}x{fx}", flush=True)
            records.append(compile_one(base_state, base_nl, ny0, nx0, fy, fx, precision, int(args.steps)))
            args.out.write_text(json.dumps({"device": str(jax.devices()[0]), "records": records}, indent=2) + "\n")
            rec = records[-1]
            mem = rec["memory_analysis"]
            temp = mem.get("temp_size_in_bytes") if isinstance(mem, dict) else None
            arg = mem.get("argument_size_in_bytes") if isinstance(mem, dict) else None
            print(
                f"  compile={rec['compile_s']:.1f}s arg={arg} temp={temp} "
                f"dtypes={rec['hlo_counts']['dtype_tokens']}",
                flush=True,
            )
    args.out.write_text(json.dumps({"device": str(jax.devices()[0]), "records": records}, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
