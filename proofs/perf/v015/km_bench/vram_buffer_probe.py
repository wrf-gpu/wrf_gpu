#!/usr/bin/env python
"""v0.15 Lane 2 — identify the un-chunked full-grid intermediate that sets the
VRAM ceiling.

The km-bench (grid_scaling.json) found peak VRAM is LINEAR in ncol and the run
OOMs on a SINGLE ~28 GiB allocation at ~210k cols -- some intermediate is NOT
bounded by the 16384-col radiation/MYNN tile floor.  This probe RUNS one tiny
forecast at a chosen grid with XLA HLO + buffer-assignment dumping enabled, then
parses the dumped *buffer_assignment* reports to name the largest temporary
buffers (HLO instruction + shape + bytes), pointing at the exact un-chunked op.

Run (GPU lock):
  PYTHONPATH=src JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.55 OMP_NUM_THREADS=4 taskset -c 0-3 \
    python proofs/perf/v015/km_bench/vram_buffer_probe.py --fy 2 --fx 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "grid_scaling_bench", HERE / "grid_scaling_bench.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_buffer_dumps(dump_dir: Path, top_n: int = 40):
    """Parse XLA *buffer_assignment* dumps for the largest allocated buffers.

    XLA writes lines like:
      allocation N: size 2147483648, ... instruction: %fusion.12 = f64[...]{...}
    or a 'Buffer assignment stats' / 'BufferAllocation' block.  We grep all
    'size <bytes>' occurrences with the nearest shape token and keep the top N.
    """
    _dtbytes = {"f64": 8, "s64": 8, "c64": 8, "f32": 4, "s32": 4, "u32": 4,
                "bf16": 2, "c128": 16, "pred": 1}

    def _shape_bytes(tok):
        # tok like 'f64[209880,44,140]'
        dt = re.match(r"([a-z0-9]+)\[", tok).group(1)
        dims = re.findall(r"\d+", tok[tok.index("[") + 1 :])
        n = 1
        for d in dims:
            n *= int(d)
        return n * _dtbytes.get(dt, 8)

    sizes = []
    shape_sizes = {}  # tok -> max bytes seen (from HLO shape tokens directly)
    pat_alloc = re.compile(r"size[:=]?\s*(\d{6,})")
    pat_shape = re.compile(r"((?:f64|f32|s32|s64|pred|u32|bf16|c64|c128)\[[0-9,]+\])")
    files = (
        list(dump_dir.rglob("*buffer-assignment*"))
        + list(dump_dir.rglob("*buffer_assignment*"))
        + list(dump_dir.rglob("*after_optimizations*"))
        + list(dump_dir.rglob("*.txt"))
    )
    # dedup
    files = list(dict.fromkeys(files))
    for f in files:
        try:
            txt = f.read_text(errors="ignore")
        except Exception:
            continue
        for line in txt.splitlines():
            # (1) explicit buffer-assignment "size N" lines
            m = pat_alloc.search(line)
            if m:
                nbytes = int(m.group(1))
                if nbytes >= (256 << 20):
                    sh = pat_shape.search(line)
                    sizes.append({
                        "bytes": nbytes,
                        "gib": round(nbytes / (1024.0 ** 3), 3),
                        "shape": sh.group(1) if sh else None,
                        "line": line.strip()[:240],
                        "file": f.name,
                    })
            # (2) every HLO shape token -> compute byte size directly
            for tok in pat_shape.findall(line):
                try:
                    nb = _shape_bytes(tok)
                except Exception:
                    continue
                if nb >= (256 << 20) and nb > shape_sizes.get(tok, 0):
                    shape_sizes[tok] = nb
    sizes.sort(key=lambda d: -d["bytes"])
    # dedup identical lines
    seen, out = set(), []
    for s in sizes:
        k = (s["bytes"], s["line"])
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
        if len(out) >= top_n:
            break
    # top distinct HLO shape tokens by byte size
    shape_top = sorted(
        ({"shape": t, "bytes": b, "gib": round(b / (1024.0 ** 3), 3)}
         for t, b in shape_sizes.items()),
        key=lambda d: -d["bytes"],
    )[:top_n]
    return out, [f.name for f in files], shape_top


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fy", type=int, default=2)
    ap.add_argument("--fx", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "vram_buffer_probe.json"))
    args = ap.parse_args()

    dump_dir = HERE / f"hlo_dump_{args.fy}x{args.fx}"
    dump_dir.mkdir(exist_ok=True)
    # Enable HLO + buffer-assignment dumping BEFORE importing jax.
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_dump_to={dump_dir}"
        + " --xla_dump_hlo_as_text"
    ).strip()

    import jax  # noqa: E402

    gsb = _load_bench()
    from gpuwrf.runtime.operational_mode import run_forecast_operational  # noqa: E402

    cfg = gsb.DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, _ = gsb._build_real_case(cfg)
    base_nl, base_state = case.namelist, case.state
    ny0, nx0, nz = int(case.grid.ny), int(case.grid.nx), int(case.grid.nz)

    nl = gsb._tile_namelist(base_nl, ny0, nx0, args.fy, args.fx)
    st = gsb._tile_state(base_state, ny0, nx0, args.fy, args.fx)
    st = jax.tree_util.tree_map(lambda x: (x + 0) if hasattr(x, "shape") else x, st)
    ny, nx = args.fy * ny0, args.fx * nx0
    ncol = ny * nx

    print(f"[vram-probe] grid {ny}x{nx}x{nz} ncol={ncol}; compiling 0.05h ...", flush=True)
    out = run_forecast_operational(st, nl, 0.05)
    jax.block_until_ready(jax.tree_util.tree_leaves(out)[0])
    print("[vram-probe] forecast compiled+ran; parsing buffer dumps ...", flush=True)

    biggest, files, shape_top = _parse_buffer_dumps(dump_dir)
    # Per-buffer GiB scaled to the OOM grid (209880 cols) for attribution.
    scale_to_oom = 209880.0 / float(ncol)
    for b in biggest:
        b["gib_at_209880col"] = round(b["gib"] * scale_to_oom, 2)
    for s in shape_top:
        s["gib_at_209880col"] = round(s["gib"] * scale_to_oom, 2)

    rec = {
        "schema": "V015VramBufferProbe",
        "grid": {"ny": ny, "nx": nx, "nz": nz, "ncol": ncol, "fy": args.fy, "fx": args.fx},
        "scale_to_oom_grid": round(scale_to_oom, 3),
        "dump_dir": str(dump_dir),
        "n_dump_files": len(files),
        "largest_buffers_ge_256MiB": biggest,
        "largest_hlo_shape_tokens": shape_top,
    }
    Path(args.out).write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("largest_buffers_ge_256MiB", "largest_hlo_shape_tokens")}, indent=2))
    print("--- largest HLO shape tokens (this grid -> @oom 209880col) ---")
    for s in shape_top[:15]:
        print(f"  {s['gib']:.3f} GiB  ({s['gib_at_209880col']:.1f} @oom)  {s['shape']}")
    print("--- largest buffer-assignment entries (>=256MiB) ---")
    for b in biggest[:12]:
        print(f"  {b['gib']:.3f} GiB  ({b['gib_at_209880col']:.1f} @oom)  {b['shape']}  {b['line'][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
