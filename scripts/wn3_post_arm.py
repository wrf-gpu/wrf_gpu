"""Post-arm summary for one WN3 forecast output dir (CPU only, read-only inputs).

Usage: post_arm.py OUT CASE HOURS [--cache JAXDIR]
Writes OUT/post_summary.json: receipt timing/VRAM, census totals per domain,
rain-field stats vs CPU-WRF (d02/d03, selected hours), optional E41 cubin gate.
No tolerances are decided here; D6 is scripts/wn3_score.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset

CASES = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen")
CUBIN_SCAN = "<USER_HOME>/wrf_gpu2_lanes/opus-a19/tools/cubin_scan.py"


def stats(a):
    f = a[np.isfinite(a)]
    return dict(nonfinite=int(a.size - f.size), negative=int(np.count_nonzero(f < 0)),
                min=float(f.min()) if f.size else None, max=float(f.max()) if f.size else None)


def receipt_summary(out: Path) -> dict:
    r = json.loads((out / "receipt.json").read_text())
    seg = r["derived"]["segments"]
    rates = r["derived"]["segment_rate_s_per_fc_h"]
    warm = [x["call_s"] for x in seg if not x.get("compile_s_in_call")]
    return dict(rc=r.get("rc"), end_reason=r.get("end_reason"), device=r.get("device"),
                git_head=r.get("git_head"), src_tree=r.get("src_tree"),
                wall_s=r.get("t_end"), t_first_segment_start=seg[0]["t_start"] if seg else None,
                compile_s=sum(x.get("compile_s_in_call") or 0 for x in seg),
                segment_rate_s_per_fc_h=rates,
                warm_segment_mean_s_per_fc_h=float(np.mean([rates[i] for i, x in enumerate(seg) if not x.get("compile_s_in_call")])) if warm else None,
                vram_peak_mib=r["derived"].get("vram_peak_mib"), jax_memory_stats=r.get("jax_memory_stats"),
                host_vmhwm_kb=r.get("host_vmhwm_kb"), errors=r.get("errors"))


def census_summary(out: Path) -> dict | None:
    p = out / "wrfout" / "census.json"
    if not p.exists():
        return None
    census = json.loads(p.read_text())
    res = {}
    for dom, data in census["domains"].items():
        g = data["guards"]
        res[dom] = dict(own_steps=data["own_steps"],
                        total_nonfinite=sum(v["nonfinite"] for v in g.values()),
                        total_repaired=sum(v["repaired"] for v in g.values()),
                        nonzero_guards={k: v for k, v in g.items() if any(v.values())})
    return res


def rain_rows(out: Path, case: str, hours: list[int]) -> list[dict]:
    cpu_dir = CASES / f"wg_{case}" / "run" / "run"
    gpu_files = sorted((out / "wrfout").glob("wrfout_d01_*"))
    rows = []
    for dom in ("d02", "d03"):
        files = sorted((out / "wrfout").glob(f"wrfout_{dom}_*"))
        for h in hours:
            if h >= len(files):
                continue
            name = files[h].name
            with Dataset(cpu_dir / name) as c, Dataset(files[h]) as g:
                for field in ("QRAIN", "QNRAIN", "RAINNC", "T2", "U10"):
                    a = np.ma.asarray(c[field][:], dtype=np.float64).filled(np.nan)
                    b = np.ma.asarray(g[field][:], dtype=np.float64).filled(np.nan)
                    ok = np.isfinite(a) & np.isfinite(b)
                    d = b[ok] - a[ok]
                    rows.append(dict(domain=dom, hour=h, field=field, cpu=stats(a), gpu=stats(b),
                                     rmse=float(np.sqrt(np.mean(d * d))) if d.size else None,
                                     max_abs=float(np.abs(d).max()) if d.size else None))
    del gpu_files
    return rows


def cubin_gate(cache: Path, work: Path) -> dict:
    import zstandard
    work.mkdir(exist_ok=True)
    db = work / "empty_timing.sqlite"
    c = sqlite3.connect(db)
    c.execute("create table if not exists StringIds(id integer,value text)")
    c.execute("create table if not exists CUPTI_ACTIVITY_KIND_KERNEL(shortName integer,start integer,end integer,gridX integer)")
    c.close()
    receipts = []
    for p in sorted(cache.glob("jit__advance_chunk_fori*-cache")):
        blob = zstandard.ZstdDecompressor().decompress(p.read_bytes())
        o = blob.find(b"\x7fELF")
        if o < 0:
            receipts.append(dict(cache_file=str(p), error="no ELF in blob"))
            continue
        phoff = struct.unpack_from("<Q", blob, o + 0x20)[0]
        phe, phn = struct.unpack_from("<HH", blob, o + 0x36)
        shoff = struct.unpack_from("<Q", blob, o + 0x28)[0]
        she, shn = struct.unpack_from("<HH", blob, o + 0x3A)
        end = max(phoff + phe * phn, shoff + she * shn)
        for k in range(shn):
            s = o + shoff + k * she
            off, size = struct.unpack_from("<QQ", blob, s + 0x18)
            if struct.unpack_from("<I", blob, s + 4)[0] != 8:
                end = max(end, off + size)
        label = p.name.split("-")[1][:12]
        cub = work / (label + ".cubin")
        cub.write_bytes(blob[o:o + end])
        del blob
        sha = hashlib.sha256(cub.read_bytes()).hexdigest()
        proc = subprocess.run([sys.executable, CUBIN_SCAN, str(cub), str(db), label], cwd=work,
                              capture_output=True, text=True, check=True,
                              env={"PATH": "/usr/local/cuda/bin:/usr/bin:/bin"})
        (work / (label + ".log")).write_text(proc.stdout)
        rows = json.loads((work / f"cubin_scan_{label}.json").read_text())
        receipts.append(dict(cache_file=str(p), cubin_sha256=sha, kernels=len(rows),
                             max_stack_B=max(x["stack"] for x in rows),
                             stack_over256=[x["kernel"][:120] + f" stack={x['stack']}" for x in rows if x["stack"] > 256],
                             nonleaf_over1000=[x["kernel"][:120] for x in rows if x["calls"] and x["sass"] > 1000],
                             transpose_over500=[x["kernel"][:120] for x in rows if "transpose" in x["kernel"] and x["sass"] > 500]))
        cub.unlink()
    db.unlink()
    bad = any(r.get("stack_over256") or r.get("nonleaf_over1000") or r.get("transpose_over500") or r.get("error") for r in receipts)
    return dict(gate="FAIL" if bad else ("PASS-static" if receipts else "NO-EXECUTABLES"), receipts=receipts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("case")
    ap.add_argument("hours", type=int)
    ap.add_argument("--cache", type=Path)
    a = ap.parse_args()
    assert not Path("/tmp/wrf_gpu2_quiet").exists(), "QUIET: defer"
    sel = sorted({1, a.hours} | ({3, 6, 12, 18} & set(range(a.hours + 1))))
    res = dict(out=str(a.out), case=a.case, hours=a.hours, receipt=receipt_summary(a.out),
               census=census_summary(a.out), rain=rain_rows(a.out, a.case, sel))
    if a.cache:
        res["cubin"] = cubin_gate(a.cache, a.out / "cubin_scan")
    (a.out / "post_summary.json").write_text(json.dumps(res, indent=1) + "\n")
    r = res["receipt"]
    print(json.dumps(dict(rc=r["rc"], wall_s=r["wall_s"], compile_s=r["compile_s"], warm_s_per_h=r["warm_segment_mean_s_per_fc_h"],
                          vram_peak_mib=r["vram_peak_mib"], jax_peak=(r["jax_memory_stats"] or {}).get("peak_bytes_in_use"),
                          census={k: (v["total_nonfinite"], v["total_repaired"]) for k, v in (res["census"] or {}).items()},
                          cubin=res.get("cubin", {}).get("gate"))))


if __name__ == "__main__":
    main()
