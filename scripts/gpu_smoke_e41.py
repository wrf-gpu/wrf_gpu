#!/usr/bin/env python3
"""E41 static gate fallback for the v0.3.4 GPU smoke (CPU only, no GPU needed).

``scripts/wave_cubin_gate.py`` needs ``<USER_HOME>/wrf_gpu2_lanes/opus-a19/tools/{cubin_scan,hlo_scan}.py``,
which no longer exist on this host.  This module reuses the gate's executable discovery and
blob/ELF/HLO extraction and replaces only the two scanners:

* cubin resources via ``cuobjdump -res-usage`` (STACK per kernel; hard gate STACK > 256 B) and
  ``cuobjdump -sass`` (instruction and CALL counts per function; conservative flag: non-leaf
  function > 1000 SASS instructions);
* optimized HLO text (rich JIT entries only): ``kind=kInput`` fusions whose fused computation
  contains a ``transpose`` and has > 500 instructions (conservative flag).

Usage: ``gpu_smoke_e41.py CACHE OUT`` -> ``OUT/e41.json``; exit 2 on a hard STACK failure.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

GATE = Path(__file__).resolve().with_name("wave_cubin_gate.py")
CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"

_FUNC_RES = re.compile(r"Function\s+(\S+):")
_STACK = re.compile(r"\bSTACK:(\d+)")
_SASS_FUNC = re.compile(r"^\s*Function\s*:\s*(\S+)")
_SASS_INSN = re.compile(r"/\*[0-9a-f]{4,}\*/\s+[^/]")


def _gate_module():
    spec = importlib.util.spec_from_file_location("wave_cubin_gate", GATE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(mod)
    return mod


def cubin_resources(path: Path) -> list[dict]:
    res = subprocess.run([CUOBJDUMP, "-res-usage", str(path)], capture_output=True, text=True, timeout=180, check=True).stdout
    rows, current = {}, None
    for line in res.splitlines():
        m = _FUNC_RES.search(line)
        if m:
            current = m.group(1)
            rows[current] = {"kernel": current, "stack": 0, "sass": 0, "calls": 0}
        s = _STACK.search(line)
        if s and current is not None:
            rows[current]["stack"] = int(s.group(1))
    sass = subprocess.run([CUOBJDUMP, "-sass", str(path)], capture_output=True, text=True, timeout=300, check=True).stdout
    current = None
    for line in sass.splitlines():
        m = _SASS_FUNC.match(line)
        if m:
            current = m.group(1)
            rows.setdefault(current, {"kernel": current, "stack": 0, "sass": 0, "calls": 0})
            continue
        if current is not None and _SASS_INSN.search(line):
            rows[current]["sass"] += 1
            if re.search(r"\bCALL\.", line):
                rows[current]["calls"] += 1
    if not rows:
        raise ValueError(f"cuobjdump found no functions in {path}")
    return list(rows.values())


def kinput_transpose_over500(hlo_text: str) -> list[tuple[str, int]]:
    comps, name, count, has_t = {}, None, 0, False
    for line in hlo_text.splitlines():
        head = re.match(r"^\s*(?:ENTRY\s+)?(%?[\w.\-]+)\s.*\{\s*$", line)
        if head and "=" not in line.split("{")[0]:
            name, count, has_t = head.group(1).lstrip("%"), 0, False
            continue
        if line.strip() == "}" and name is not None:
            comps[name] = (count, has_t)
            name = None
            continue
        if name is not None and "=" in line:
            count += 1
            has_t = has_t or " transpose(" in line
    flags, stats = [], {"computations": len(comps), "kinput_fusions": 0, "kinput_transpose_fusions": 0,
                        "max_kinput_transpose_instr": 0}
    for line in hlo_text.splitlines():
        if "fusion(" in line and "kind=kInput" in line:
            m = re.search(r"calls=%?([\w.\-]+)", line)
            if m and m.group(1) in comps:
                stats["kinput_fusions"] += 1
                n, t = comps[m.group(1)]
                if t:
                    stats["kinput_transpose_fusions"] += 1
                    stats["max_kinput_transpose_instr"] = max(stats["max_kinput_transpose_instr"], n)
                if t and n > 500:
                    flags.append((m.group(1), n))
    kinput_transpose_over500.stats = stats
    return flags


def scan(cache: Path, out: Path) -> dict:
    g = _gate_module()
    import zstandard
    from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm

    out.mkdir(parents=True, exist_ok=True)
    entries = g.discover(cache)
    records = []
    for entry in entries:
        blob = entry.path.read_bytes()
        if entry.kind == "jax_cache":
            blob = zstandard.ZstdDecompressor().decompress(blob, max_output_size=1 << 33)[4:]
        fmt = ax._gpu_blob_format(blob)
        rec = {"entry": entry.path.name, "kind": entry.kind, "program": entry.program, "format": fmt, "elfs": []}
        if fmt not in (ax.GPU_THUNK_FORMAT, ax.GPU_LEGACY_FORMAT):
            rec["error"] = "not a supported serialized GPU executable"
            records.append(rec)
            continue
        native = g._record(blob, 2, ax, asm) if fmt == ax.GPU_THUNK_FORMAT else blob
        label = hashlib.sha256(str(entry.path).encode()).hexdigest()[:12]
        for i, cubin in enumerate(g._elfs(native)):
            cp = out / f"{label}_e{i}.cubin"
            cp.write_bytes(cubin)
            try:
                rows = cubin_resources(cp)
            finally:
                cp.unlink(missing_ok=True)
            rec["elfs"].append({
                "elf": i, "kernels": len(rows), "max_stack_B": max(r["stack"] for r in rows),
                "stack_over256": [(r["kernel"][:100], r["stack"]) for r in rows if r["stack"] > 256],
                "nonleaf_over1k": [(r["kernel"][:100], r["sass"], r["calls"]) for r in rows
                                   if r["sass"] > 1000 and r["calls"]],
            })
        if entry.kind == "jax_cache":
            try:
                rec["kinput_transpose_over500"] = kinput_transpose_over500(g._optimized_hlo(blob, ax, asm))
                rec["hlo_stats"] = dict(kinput_transpose_over500.stats)
                rec["kinput_transpose_status"] = "MEASURED_RICH_JIT_HLO"
            except Exception as exc:  # noqa: BLE001 -- reported, never a silent PASS
                rec["kinput_transpose_status"] = f"UNKNOWN_HLO_ERROR {type(exc).__name__}: {exc}"
        else:
            rec["kinput_transpose_status"] = "UNKNOWN_AOT_METADATA"
        records.append(rec)
    if not records:
        raise ValueError(f"no GPU executables discovered under {cache}")
    hard = any(e["stack_over256"] for r in records for e in r["elfs"])
    conservative = any(e["nonleaf_over1k"] for r in records for e in r["elfs"]) or any(
        r.get("kinput_transpose_over500") for r in records)
    unscanned = [r["entry"] for r in records if "error" in r or not r["elfs"]]
    gate = {"cache": str(cache), "tool": "scripts/gpu_smoke_e41.py (cuobjdump fallback; opus-a19 tools absent)",
            "n_entries": len(records), "unscanned": unscanned,
            "hard_stack_gate": "FAIL" if hard else ("INCOMPLETE" if unscanned else "PASS"),
            "max_stack_B": max((e["max_stack_B"] for r in records for e in r["elfs"]), default=None),
            "conservative_flags": bool(conservative), "entries": records}
    (out / "e41.json").write_text(json.dumps(gate, indent=1))
    return gate


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    gate = scan(Path(argv[0]), Path(argv[1]))
    print(json.dumps({k: gate[k] for k in ("n_entries", "hard_stack_gate", "max_stack_B", "conservative_flags", "unscanned")}))
    return 2 if gate["hard_stack_gate"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
