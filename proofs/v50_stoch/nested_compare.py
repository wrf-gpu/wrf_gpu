"""Nested stochastic GPU vs original CPU-WRF comparator (lane v50-stoch, critic FIX finding 8).

usage: nested_compare.py <cpu_dir> <gpu_dir> <out.json> [--domains d01 d02 d03] [--hours 24]
Exit 0 = PASS, 1 = FAIL, 2 = INCOMPLETE (missing frames/fields; never a pass). Unlike w41 pattern_compare.py:
- exactly hours+1 frames per domain on BOTH sides (more = FAIL, fewer = INCOMPLETE; never truncated), paired in order,
  identical file names and identical Times strings including seconds; history files of undeclared domains = FAIL;
- the stochastic inventory (patterns, SPFORC coefficients, six seed arrays) must exist in every CPU and GPU frame;
- per field and frame max|gpu-cpu| <= (8e-6 + 64 eps_f32) * max|cpu|; a zero CPU frame requires a zero GPU frame;
  SPFORCC_k/SPFORCS_k (real/imaginary parts of the same modes) use the joint max sqrt(C^2+S^2) of the CPU pair;
- seed arrays exact; any nonfinite value fails.
"""
import argparse
import json
from pathlib import Path
import sys

import netCDF4
import numpy as np

PATTERNS = ("RU_TENDF_STOCH", "RV_TENDF_STOCH", "RT_TENDF_STOCH", "RSTOCH",
            "PATTERN_SPP_CONV", "PATTERN_SPP_PBL", "PATTERN_SPP_LSM")
SPECTRAL = tuple(f"SPFORC{p}{i}" for i in (2, 3, 4) for p in ("C", "S"))
SEEDS = ("ISEEDARR_SKEBS", "ISEEDARR_SPPT", "ISEEDARR_RAND_PERTURB",
         "ISEEDARRAY_SPP_CONV", "ISEEDARRAY_SPP_PBL", "ISEEDARRAY_SPP_LSM")
REL = 8e-6 + 64 * float(np.finfo(np.float32).eps)


def frame(path):
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_maskandscale(False)
        names = set(ds.variables)
        stamp = b"".join(np.asarray(ds["Times"][:]).reshape(-1)).decode() if "Times" in names else None
        data = {n: np.asarray(ds[n][:]) for n in (*PATTERNS, *SPECTRAL, *SEEDS) if n in names}
    return stamp, data


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cpu", type=Path); ap.add_argument("gpu", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--domains", nargs="+", default=["d01", "d02", "d03"]); ap.add_argument("--hours", type=int, default=24)
    a = ap.parse_args(argv)
    report = dict(schema="v50-stoch-nested-compare-v1", cpu=str(a.cpu), gpu=str(a.gpu), bound_rel=REL, domains={})
    incomplete, fail = [], []
    for side, root in (("cpu", a.cpu), ("gpu", a.gpu)):
        other = sorted(p.name for p in root.glob("wrfout_*") if p.name.split("_")[1] not in a.domains)
        if other:   # undeclared domains are never silently ignored
            fail.append(f"{side}: undeclared history files {other[:3]}{'...' if len(other) > 3 else ''}")
    for d in a.domains:
        cpu = sorted(a.cpu.glob(f"wrfout_{d}_*"))   # exact declared inventory: never truncated
        gpu = sorted(a.gpu.glob(f"wrfout_{d}_*"))
        row = dict(cpu_frames=len(cpu), gpu_frames=len(gpu), fields={}, seeds={}, missing=[], stamp_mismatch=[])
        if len(cpu) > a.hours + 1 or len(gpu) > a.hours + 1:
            fail.append(f"{d}: {len(cpu)} CPU / {len(gpu)} GPU frames exceed the declared {a.hours + 1}")
        elif len(cpu) != a.hours + 1 or len(gpu) != a.hours + 1:
            incomplete.append(f"{d}: {len(cpu)} CPU / {len(gpu)} GPU frames, need {a.hours + 1} each")
        for c_path, g_path in zip(cpu, gpu):
            (cs, c), (gs, g) = frame(c_path), frame(g_path)
            if cs is None or gs is None or cs != gs or c_path.name != g_path.name:   # exact Times incl. seconds
                row["stamp_mismatch"].append([c_path.name, g_path.name, cs, gs])
            for n in (*PATTERNS, *SPECTRAL, *SEEDS):
                if n not in c or n not in g:
                    row["missing"].append(f"{c_path.name}:{n}:{'cpu' if n not in c else 'gpu'}")
                    continue
                if n in SEEDS:
                    s = row["seeds"].setdefault(n, dict(frames=0, exact=0))
                    s["frames"] += 1; s["exact"] += int(np.array_equal(c[n], g[n]))
                    continue
                r = row["fields"].setdefault(n, dict(frames=0, worst_rel=0., zero_ref_frames=0, fail_frames=[]))
                r["frames"] += 1
                cv, gv = c[n].astype(np.float64), g[n].astype(np.float64)
                if cv.shape != gv.shape or not np.isfinite(cv).all() or not np.isfinite(gv).all():
                    r["fail_frames"].append(c_path.name); continue
                scale = float(np.max(np.abs(cv)))
                if n in SPECTRAL:
                    # Real/imaginary parts of the same SPFORC modes: normalize by the joint mode amplitude
                    # max sqrt(C^2 + S^2) of the CPU pair, so a small sine (cosine) part of a large mode is not
                    # judged against its own tiny maximum (GPU arm 11.11: d01 SPFORCS3 |d| 2.8e-9 on 9.3e-5,
                    # mode-relative 1.2e-6). Shapes/finiteness of the partner are checked on its own row.
                    partner = n.replace("SPFORCC", "SPFORC_").replace("SPFORCS", "SPFORCC").replace("SPFORC_", "SPFORCS")
                    pv = c.get(partner)
                    if pv is not None and np.shape(pv) == cv.shape and np.isfinite(pv).all():
                        scale = float(np.max(np.hypot(cv, pv.astype(np.float64))))
                err = float(np.max(np.abs(gv - cv))) if cv.size else 0.
                if scale == 0:
                    r["zero_ref_frames"] += 1
                    if err != 0:
                        r["fail_frames"].append(c_path.name)
                    continue
                r["worst_rel"] = max(r["worst_rel"], err / scale)
                if err > REL * scale:
                    r["fail_frames"].append(c_path.name)
        if row["missing"]:
            incomplete.append(f"{d}: {len(row['missing'])} missing field-frames")
        if row["stamp_mismatch"]:
            fail.append(f"{d}: {len(row['stamp_mismatch'])} unpaired valid times")
        fail += [f"{d}:{n}: {len(r['fail_frames'])} frames" for n, r in row["fields"].items() if r["fail_frames"]]
        fail += [f"{d}:{n}: {s['frames'] - s['exact']} inexact seed frames" for n, s in row["seeds"].items() if s["exact"] != s["frames"]]
        report["domains"][d] = row
    verdict = "FAIL" if fail else "INCOMPLETE" if incomplete else "PASS"
    report.update(verdict=verdict, fail=fail, incomplete=incomplete)
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(verdict, *fail[:5], *incomplete[:5], sep="\n")
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2}[verdict]


if __name__ == "__main__":
    sys.exit(main())
