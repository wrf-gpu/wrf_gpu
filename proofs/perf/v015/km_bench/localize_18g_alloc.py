#!/usr/bin/env python
"""v0.15 VRAM-ceiling LOCALIZATION — name the single ~18.72 GiB non-Thompson
allocation that binds the 1km grids.

Two complementary localizers in one GPU session:

  PHASE A (compile-only, cheap, no OOM): compile the FULL operational pipeline
  at the base grid with XLA HLO + buffer-assignment dumping. Parse the
  buffer-assignment report for the largest TEMPORARY buffers (HLO op + shape +
  bytes), then scale each linearly to the 167,904-col OOM grid. Buffer
  assignment is the authoritative per-buffer allocation list, so the largest
  per-col temp names the binding op directly.

  PHASE B (the real OOM breakdown): attempt the 167,904-col forecast with the
  PLATFORM (bfc) allocator so the RESOURCE_EXHAUSTED message carries XLA's full
  peak-buffer breakdown ("Largest program allocation(s)" with shapes). Capture
  the WHOLE message (not truncated). This is ground truth for the 18.72 GiB
  attribution.

Run (GPU lock):
  scripts/with_gpu_lock.sh --label v015-vram -- env \
    PYTHONPATH=src JAX_ENABLE_X64=true OMP_NUM_THREADS=4 \
    GPUWRF_CANAIRY_ROOT=/mnt/data/canairy_meteo taskset -c 0-3 \
    python proofs/perf/v015/km_bench/localize_18g_alloc.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
OOM_GRID_NCOL = 167904


def _load_bench():
    spec = importlib.util.spec_from_file_location(
        "grid_scaling_bench", HERE / "grid_scaling_bench.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DTBYTES = {"f64": 8, "s64": 8, "c64": 8, "f32": 4, "s32": 4, "u32": 4,
            "bf16": 2, "c128": 16, "pred": 1, "u8": 1, "s8": 1}


def _shape_bytes(tok: str) -> int:
    dt = re.match(r"([a-z0-9]+)\[", tok).group(1)
    dims = re.findall(r"\d+", tok[tok.index("[") + 1:])
    n = 1
    for d in dims:
        n *= int(d)
    return n * _DTBYTES.get(dt, 8)


def _parse_buffer_assignment(dump_dir: Path, base_ncol: int, top_n: int = 50):
    """Parse XLA *-buffer-assignment.txt for the largest allocations.

    Lines look like:
      allocation 12: 0x..., size 360448000, ...
        instruction: %fusion.7 = f64[10494,44,...]{...} fusion(...)
    We pair each 'size N' with the nearest f64[...]/f32[...] shape token (same
    line OR the following indented 'value:' / 'instruction:' line).
    """
    files = sorted(dump_dir.rglob("*buffer-assignment*")) + \
        sorted(dump_dir.rglob("*buffer_assignment*"))
    files = list(dict.fromkeys(files))
    pat_size = re.compile(r"size[:\s]+(\d{6,})")
    pat_shape = re.compile(r"((?:f64|f32|s32|s64|pred|u32|bf16|c64|c128|u8|s8)\[[0-9,]+\])")
    pat_op = re.compile(r"(%[\w.\-]+)\s*=")
    entries = []
    scale = OOM_GRID_NCOL / float(base_ncol)
    for f in files:
        try:
            lines = f.read_text(errors="ignore").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines):
            m = pat_size.search(line)
            if not m:
                continue
            nbytes = int(m.group(1))
            if nbytes < (64 << 20):  # ignore < 64 MiB temps
                continue
            # search this + next 3 lines for shape + op name
            ctx = " ".join(lines[i:i + 4])
            sh = pat_shape.search(ctx)
            op = pat_op.search(ctx)
            entries.append({
                "bytes": nbytes,
                "gib": round(nbytes / 1024 ** 3, 3),
                "gib_at_167904col": round(nbytes / 1024 ** 3 * scale, 2),
                "shape": sh.group(1) if sh else None,
                "op": op.group(1) if op else None,
                "line": line.strip()[:200],
                "file": f.name,
            })
    # dedup identical (bytes, op)
    seen, out = set(), []
    for e in sorted(entries, key=lambda d: -d["bytes"]):
        k = (e["bytes"], e["op"], e["shape"])
        if k in seen:
            continue
        seen.add(k)
        out.append(e)
        if len(out) >= top_n:
            break
    # Also: every distinct HLO shape token across all dumped text (op-level).
    shape_sizes = {}
    text_files = sorted(dump_dir.rglob("*after_optimizations*txt")) + \
        sorted(dump_dir.rglob("*after_optimizations*")) + files
    for f in list(dict.fromkeys(text_files)):
        try:
            txt = f.read_text(errors="ignore")
        except Exception:
            continue
        for tok in pat_shape.findall(txt):
            try:
                nb = _shape_bytes(tok)
            except Exception:
                continue
            if nb >= (256 << 20) and nb > shape_sizes.get(tok, 0):
                shape_sizes[tok] = nb
    shape_top = sorted(
        ({"shape": t, "gib": round(b / 1024 ** 3, 3),
          "gib_at_167904col": round(b / 1024 ** 3 * scale, 2)}
         for t, b in shape_sizes.items()),
        key=lambda d: -d["gib"])[:top_n]
    return out, [f.name for f in files], shape_top


def main() -> int:
    out_path = HERE / "localize_18g_alloc.json"
    result = {"schema": "V015Localize18G", "oom_grid_ncol": OOM_GRID_NCOL}

    # ---- PHASE A: compile-only buffer-assignment dump at base grid ----
    dump_dir = HERE / "hlo_dump_localize_base"
    dump_dir.mkdir(exist_ok=True)
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "")
        + f" --xla_dump_to={dump_dir}"
        + " --xla_dump_hlo_as_text"
    ).strip()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "cuda_async")

    import jax  # noqa: E402

    gsb = _load_bench()
    from gpuwrf.runtime.operational_mode import run_forecast_operational  # noqa: E402

    cfg = gsb.DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, _ = gsb._build_real_case(cfg)
    base_nl, base_state = case.namelist, case.state
    ny0, nx0, nz = int(case.grid.ny), int(case.grid.nx), int(case.grid.nz)
    base_ncol = ny0 * nx0
    result["base_grid"] = {"ny": ny0, "nx": nx0, "nz": nz, "ncol": base_ncol}
    print(f"[A] compiling base {ny0}x{nx0} ncol={base_ncol} (radiation-on step) ...", flush=True)

    # Compile a step that EXERCISES radiation (the cadence step) so the radiation
    # graph is in the dump; 0.05h = a few steps, radiation_cadence small enough.
    try:
        fc = run_forecast_operational(base_state, base_nl, 0.05)
        jax.block_until_ready(jax.tree_util.tree_leaves(fc)[0])
        print("[A] compiled+ran base; parsing buffer-assignment ...", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[A] base run raised (still parsing dumps): {e}", flush=True)

    biggest, ba_files, shape_top = _parse_buffer_assignment(dump_dir, base_ncol)
    result["phase_a"] = {
        "dump_dir": str(dump_dir),
        "n_buffer_assignment_files": len(ba_files),
        "largest_buffers_ge_64MiB": biggest,
        "largest_hlo_shape_tokens_ge_256MiB": shape_top,
    }
    print("--- PHASE A: largest buffer-assignment temps (base -> @167904col) ---", flush=True)
    for b in biggest[:20]:
        print(f"  {b['gib']:.3f}G ({b['gib_at_167904col']:.1f}@oom)  {b['shape']}  {b['op']}", flush=True)
    print("--- PHASE A: largest distinct HLO shape tokens ---", flush=True)
    for s in shape_top[:20]:
        print(f"  {s['gib']:.3f}G ({s['gib_at_167904col']:.1f}@oom)  {s['shape']}", flush=True)

    out_path.write_text(json.dumps(result, indent=2) + "\n")

    # ---- PHASE B: real OOM breakdown at 167,904 cols ----
    # Find (fy,fx) giving 167,904 from the base grid.
    fy = fx = None
    for cfy in range(1, 12):
        for cfx in range(1, 12):
            if cfy * ny0 * cfx * nx0 == OOM_GRID_NCOL:
                fy, fx = cfy, cfx
                break
        if fy:
            break
    result["phase_b"] = {"fy": fy, "fx": fx}
    if fy is None:
        print(f"[B] no exact (fy,fx) for {OOM_GRID_NCOL}; base {ny0}x{nx0}", flush=True)
    else:
        ny, nx = fy * ny0, fx * nx0
        print(f"[B] attempting OOM grid {ny}x{nx} ncol={ny*nx} to capture full breakdown ...", flush=True)
        # Phase A donated/consumed base_state buffers; rebuild a FRESH case so
        # the tiled state leaves are live (not deleted) for the OOM attempt.
        case_b, _ = gsb._build_real_case(cfg)
        bnl_b, bst_b = case_b.namelist, case_b.state
        nl = gsb._tile_namelist(bnl_b, ny0, nx0, fy, fx)
        st = gsb._tile_state(bst_b, ny0, nx0, fy, fx)
        st = jax.tree_util.tree_map(lambda x: (x + 0) if hasattr(x, "shape") else x, st)
        try:
            o = run_forecast_operational(st, nl, 0.05)
            jax.block_until_ready(jax.tree_util.tree_leaves(o)[0])
            result["phase_b"]["ran_ok"] = True
            result["phase_b"]["oom"] = False
            print("[B] UNEXPECTED: 167904 ran OK (no OOM)", flush=True)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            is_oom = "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()
            result["phase_b"]["ran_ok"] = False
            result["phase_b"]["oom"] = bool(is_oom)
            result["phase_b"]["full_error"] = msg[:8000]
            buf_lines = [ln for ln in msg.splitlines()
                         if re.search(r"\d+ bytes|GiB|MiB|buffer|allocation|fusion|f64\[", ln)]
            result["phase_b"]["breakdown_lines"] = buf_lines[:120]
            print(f"[B] OOM={is_oom}; captured {len(buf_lines)} breakdown lines", flush=True)
            print("\n".join(buf_lines[:60]), flush=True)
            (HERE / "localize_18g_oom_fulltext.txt").write_text(msg)

    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
