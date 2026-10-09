"""CPU E41 gate: JAX entries AND every cached domain/fused stepping AOT blob.

Usage: wave_cubin_gate.py CACHE OUT [JIT_GLOB] [--run-proof nested_pipeline_run.json]
The optional run proof also rejects an entirely absent domain-program directory.
Default discovery always includes AOT stepping blobs; JIT_GLOB filters only JIT.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sqlite3
import struct
import subprocess
import sys

DEFAULT_JIT_GLOB = "jit__advance_chunk_fori*"
TOOLS = Path("<USER_HOME>/wrf_gpu2_lanes/opus-a19/tools")
STEP_PROGRAM = re.compile(r"(?:fused_)?d\d+\Z")


@dataclass(frozen=True)
class Executable:
    path: Path
    kind: str
    program: str | None = None
    aliases: tuple[str, ...] = ()


def discover(cache: Path, pattern: str = DEFAULT_JIT_GLOB) -> list[Executable]:
    if not cache.is_dir():
        raise ValueError(f"cache directory does not exist: {cache}")
    found = [Executable(p, "jax_cache") for p in sorted({p for part in pattern.split(",") for p in cache.glob(part + "-cache")})]
    # Both HLO-addressed and cheap-key aliases are scanned; inode identity avoids
    # rereading hardlinked aliases, without conflating different domain programs.
    grouped = {}
    for path in sorted((cache / "aot").glob("*/*.xlaexec")) + sorted((cache / "aot").glob("*/*/*.xlaexec")):
        program = path.stem if path.parent.parent.name == "aot" else path.parent.name
        if not STEP_PROGRAM.fullmatch(program):
            continue  # init_radiation/other auxiliary AOT programs are not steps.
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing/empty AOT stepping blob: {path}")
        stat = path.stat()
        key = (program, stat.st_dev, stat.st_ino)
        grouped.setdefault(key, []).append(path)
    for (program, _, _), paths in grouped.items():
        found.append(Executable(paths[0], "aot", program, tuple(str(p) for p in paths[1:])))
    # Sidecars are evidence that a program was compiled. Never silently omit a
    # child whose blob was deleted while its metadata/last-key marker remains.
    declared = set()
    for tag in sorted((cache / "aot").glob("*")):
        if not tag.is_dir():
            continue
        for path in tag.iterdir():
            name = path.name if path.is_dir() else path.stem
            if STEP_PROGRAM.fullmatch(name) and (path.is_dir() or path.suffix == ".meta"):
                declared.add(name)
    for meta in list((cache / "aot").glob("*/*.meta")) + list((cache / "aot").glob("*/*/*.meta")):
        program = meta.stem if meta.parent.parent.name == "aot" else meta.parent.name
        if STEP_PROGRAM.fullmatch(program) and not meta.with_suffix(".xlaexec").is_file():
            raise ValueError(f"missing AOT stepping blob declared by metadata: {meta}")
    present = {entry.program for entry in found if entry.kind == "aot"}
    if missing := declared - present:
        raise ValueError(f"missing AOT stepping program(s): {sorted(missing)}")
    if not found:
        raise ValueError("no stepping executable entries; cannot pass gate")
    return found


def proof_programs(path: Path) -> set[str]:
    proof = json.loads(path.read_text())
    programs = proof.get("metadata", {}).get("nested_aot", {}).get("domains", {})
    if not programs:
        raise ValueError(f"run proof has no AOT stepping program inventory: {path}")
    return {name.replace("/", "_") for name in programs}


def assert_coverage(expected: list[Executable], records: list[dict], required: set[str] = frozenset()):
    paths = {str(entry.path) for entry in expected}
    measured = {record["path"] for record in records if record.get("elfs")}
    if missing := paths - measured:
        raise ValueError(f"unscanned stepping executable(s): {sorted(missing)}")
    programs = {record.get("program") for record in records if record.get("elfs")}
    if missing := required - programs:
        raise ValueError(f"run program(s) absent from scan: {sorted(missing)}")


def _elfs(blob: bytes):
    pos = 0
    while (offset := blob.find(b"\x7fELF", pos)) >= 0:
        try:
            phoff = struct.unpack_from("<Q", blob, offset + 0x20)[0]
            phe, phn = struct.unpack_from("<HH", blob, offset + 0x36)
            shoff = struct.unpack_from("<Q", blob, offset + 0x28)[0]
            she, shn = struct.unpack_from("<HH", blob, offset + 0x3A)
            end = max(phoff + phe * phn, shoff + she * shn)
            for k in range(shn):
                section = offset + shoff + k * she
                off, size = struct.unpack_from("<QQ", blob, section + 0x18)
                if struct.unpack_from("<I", blob, section + 4)[0] != 8:
                    end = max(end, off + size)
            if end <= 0 or offset + end > len(blob):
                raise ValueError("truncated ELF")
        except (struct.error, ValueError):
            pos = offset + 4
            continue
        yield blob[offset:offset + end]
        pos = offset + end


def _record(blob, index, ax, asm):
    size, pos = asm._read_varint(blob, 0)
    rf = next(v for f, w, v in ax._proto_fields(blob[pos + size:]) if f == 1)
    count = 0
    for begin, info, data_pos in asm._chunks(rf):
        if count <= index < count + info["num_records"]:
            data, _ = asm._read_logical(rf, data_pos, info["data_size"])
            return asm._decode_simple_chunk(data, info["num_records"])[index - count]
        count += info["num_records"]
    raise ValueError(f"missing Riegeli record {index}")


def _optimized_hlo(blob, ax, asm):
    import jaxlib._jax as J
    size, pos = asm._read_varint(blob, 0)
    gpu = next(v for f, w, v in ax._proto_fields(blob[pos + size:]) if f == 1)
    if ax._gpu_blob_format(blob) == ax.GPU_THUNK_FORMAT:
        begin, info, data_pos = asm._chunks(gpu)[-1]
        data, _ = asm._read_logical(gpu, data_pos, info["data_size"])
        (gpu,) = asm._decode_simple_chunk(data, info["num_records"])
    hwc = next(v for f, w, v in ax._proto_fields(gpu) if f == 1)
    proto = next(v for f, w, v in ax._proto_fields(hwc) if f == 1)
    return J.HloModule.from_serialized_hlo_module_proto(proto).to_string()


def checked_resource_rows(rows):
    if not rows:
        raise ValueError("no measured kernel resources")
    for row in rows:
        if any(type(row[k]) is not int or row[k] < 0 for k in ["stack", "sass", "calls", "callees"]):
            raise ValueError("invalid cubin resource count")
    return rows


def scan_executable(entry: Executable, out: Path, db: Path) -> dict:
    import zstandard
    from gpuwrf.runtime import aot_executable as ax, aot_slim_module as asm
    blob = entry.path.read_bytes()
    if entry.kind == "jax_cache":
        blob = zstandard.ZstdDecompressor().decompress(blob, max_output_size=1 << 33)[4:]
    label = hashlib.sha256(str(entry.path).encode()).hexdigest()[:12]
    rec = {"entry": entry.path.name, "path": str(entry.path), "source_kind": entry.kind,
           "program": entry.program, "aliases": list(entry.aliases),
           "format": ax._gpu_blob_format(blob), "blob_sha256": hashlib.sha256(blob).hexdigest(), "elfs": []}
    if rec["format"] not in (ax.GPU_THUNK_FORMAT, ax.GPU_LEGACY_FORMAT):
        raise ValueError(f"not a supported serialized GPU executable: {entry.path}")
    native = _record(blob, 2, ax, asm) if rec["format"] == ax.GPU_THUNK_FORMAT else blob
    for i, cubin in enumerate(_elfs(native)):
        cp = out / f"{label}_e{i}.cubin"
        cp.write_bytes(cubin)
        try:
            subprocess.run([sys.executable, str(TOOLS / "cubin_scan.py"), str(cp), str(db), f"{label}_e{i}"],
                           cwd=out, check=True, capture_output=True, text=True, timeout=180)
            rows = checked_resource_rows(json.loads((out / f"cubin_scan_{label}_e{i}.json").read_text()))
            rec["elfs"].append({"elf": i, "sha256": hashlib.sha256(cubin).hexdigest()[:16], "kernels": len(rows),
                "max_stack_B": max(row["stack"] for row in rows),
                "stack_over256": [(row["kernel"][:100], row["stack"]) for row in rows if row["stack"] > 256],
                "nonleaf_over1k": [(row["kernel"][:100], row["sass"], row["calls"], row["callees"])
                                   for row in rows if row["sass"] > 1000 and row["calls"]]})
        finally:
            cp.unlink(missing_ok=True)
    if entry.kind == "aot":
        # Compact AOT HLO may erase transpose/fusion-body detail. Zero flags is
        # not a measured zero. Rich JIT HLO stays a separate authoritative record.
        rec["kinput_transpose_status"] = "UNKNOWN_AOT_METADATA"
        rec["kinput_transpose_over500"] = None
    else:
        hp = out / f"{label}.optimized.hlo"
        try:
            hp.write_text(_optimized_hlo(blob, ax, asm))
            spec = importlib.util.spec_from_file_location("wave_hlo_scan", TOOLS / "hlo_scan.py")
            scanner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scanner)
            hlo = scanner.main(str(hp), label)
            rec["kinput_transpose_over500"] = [(t.get("name") or t.get("fusion"), t["n_instr"])
                for t in hlo["transpose_hero_fusions"] if t["kind"] == "kInput" and t["n_instr"] > 500]
            rec["kinput_transpose_status"] = "MEASURED_RICH_JIT_HLO"
        except Exception as exc:
            rec["hlo_error"] = f"{type(exc).__name__}: {exc}"
            rec["kinput_transpose_status"] = "UNKNOWN_HLO_ERROR"
        finally:
            hp.unlink(missing_ok=True)
    return rec


def run_gate(cache: Path, out: Path, pattern=DEFAULT_JIT_GLOB, run_proof=None, scanner=scan_executable):
    entries = discover(cache, pattern)
    required = {entry.program for entry in entries if entry.kind == "aot"}
    # Standard wave layout: WAVE/cache/jax and WAVE/runs/CASE/wrfout/proofs.
    # Use producer inventories by default when present, so deleting an entire
    # child directory cannot turn the gate into an apparent root-only PASS.
    proofs = [Path(run_proof)] if run_proof else sorted((cache.parent.parent / "runs").glob("*/wrfout/proofs/nested_pipeline_run.json"))
    for proof in proofs:
        required |= proof_programs(proof)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "empty_timing.sqlite"
    with sqlite3.connect(db) as connection:
        connection.execute("create table if not exists StringIds(id integer,value text)")
        connection.execute("create table if not exists CUPTI_ACTIVITY_KIND_KERNEL(shortName integer,start integer,end integer,gridX integer)")
    try:
        records = [scanner(entry, out, db) for entry in entries]
        assert_coverage(entries, records, required)
    finally:
        db.unlink(missing_ok=True)
    hard = any(elf["stack_over256"] for rec in records for elf in rec["elfs"])
    conservative = any(elf["nonleaf_over1k"] for rec in records for elf in rec["elfs"]) or any(rec.get("kinput_transpose_over500") for rec in records)
    gate = {"cache": str(cache), "entries": records, "n_entries": len(records),
            "hard_stack_gate": "FAIL" if hard else "PASS", "conservative_flags": bool(conservative),
            "coverage": {"status": "PASS", "aot_programs": sorted({entry.program for entry in entries if entry.kind == "aot"}),
                         "required_run_programs": sorted(required), "run_proofs": [str(p) for p in proofs],
                         "n_discovered": len(entries), "n_scanned": len(records)}}
    (out / "wave_gate.json").write_text(json.dumps(gate, indent=1, allow_nan=False))
    return gate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("module_glob", nargs="?", default=DEFAULT_JIT_GLOB)
    parser.add_argument("--run-proof", type=Path)
    args = parser.parse_args(argv)
    if Path("/tmp/wrf_gpu2_quiet").exists():
        raise SystemExit("QUIET")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise SystemExit("wave_cubin_gate is CPU-only: set JAX_PLATFORMS=cpu")
    gate = run_gate(args.cache, args.out, args.module_glob, args.run_proof)
    print("HARD (STACK>256)", gate["hard_stack_gate"], "| conservative flags", gate["conservative_flags"],
          "| entries", gate["n_entries"], "| AOT programs", gate["coverage"]["aot_programs"], flush=True)
    return 2 if gate["hard_stack_gate"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
