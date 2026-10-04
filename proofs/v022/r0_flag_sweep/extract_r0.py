#!/usr/bin/env python3
"""Extract R0 flag-sweep metrics from a capture+warm log pair into one JSONL row.

Parses the perstep_timing_driver markers + /usr/bin/time -v output:
  - cold compile wall (s): from the capture log MARKER:COLD_DONE, else the
    fixed-mode first-call wall, else /usr/bin/time wall.
  - peak host RSS (VmHWM, MiB): max of the driver's rss_peak_mib and
    /usr/bin/time "Maximum resident set size".
  - s/step: the fixed-mode FIXED_SUMMARY s_per_step (clean), else SUMMARY
    s_per_step_b.
  - AOT-blob valid: the WARM log shows source=aot_blob loads with NO jit_fused
    re-lower (JAX_LOG_COMPILES would print "Compiling ... jit_fused" on a miss).
  - bit-identity: REF_COMPARE ref_equal=True in the warm log.
PASS = materially-lower peak RSS vs baseline AND no s/step regression
       (>2% hard-fail) AND blob valid+bit-identical.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _read(p: str) -> str:
    try:
        return Path(p).read_text(errors="replace")
    except Exception:
        return ""


def _time_v_maxrss_mib(text: str) -> float | None:
    # /usr/bin/time -v: "Maximum resident set size (kbytes): N"
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if m:
        return int(m.group(1)) / 1024.0
    return None


def _time_v_wall_s(text: str) -> float | None:
    # "Elapsed (wall clock) time (h:mm:ss or m:ss): 1:23.45"
    m = re.search(r"Elapsed \(wall clock\) time[^:]*:\s*([0-9:.]+)", text)
    if not m:
        return None
    parts = m.group(1).split(":")
    try:
        parts = [float(x) for x in parts]
    except ValueError:
        return None
    s = 0.0
    for x in parts:
        s = s * 60 + x
    return s


def _f(text: str, pat: str) -> float | None:
    m = re.search(pat, text)
    return float(m.group(1)) if m else None


def _parse_phase(text: str) -> dict:
    out: dict = {}
    out["driver_rss_peak_mib"] = _f(text, r"rss_peak_mib=([0-9.]+)")
    # fixed-mode summary (clean s/step)
    out["fixed_s_per_step"] = _f(text, r"FIXED_SUMMARY[^\n]*\bs_per_step=([\-0-9.eE]+)")
    if out["fixed_s_per_step"] is None:
        out["fixed_s_per_step"] = _f(text, r"\bs_per_step=([\-0-9.eE]+)")
    out["s_per_step_b"] = _f(text, r"s_per_step_b=([\-0-9.eE]+)")
    out["cold_wall_s"] = _f(text, r"MARKER:COLD_DONE cold_wall_s=([0-9.]+)")
    # fixed-mode reports per-call walls; take the max single-call wall as a cold proxy
    walls = [float(x) for x in re.findall(r"\bwall_s=([0-9.]+)", text)]
    out["max_call_wall_s"] = max(walls) if walls else None
    out["driver_raised"] = "DRIVER RAISED" in text
    out["time_v_maxrss_mib"] = _time_v_maxrss_mib(text)
    out["time_v_wall_s"] = _time_v_wall_s(text)
    # AOT-blob load evidence
    out["aot_loaded_count"] = len(re.findall(r"loaded=true source=aot_blob", text))
    out["aot_load_fail_count"] = len(re.findall(r"loaded=false", text))
    # a fused re-lower would show in JAX_LOG_COMPILES
    out["jit_fused_recompile"] = bool(
        re.search(r"Compiling\s+\S*jit[_a-zA-Z]*fused", text)
        or re.search(r"Finished tracing\s+\S*fused", text)
    )
    # bit-identity
    m = re.search(r"REF_COMPARE[^\n]*ref_equal=(True|False)", text)
    out["ref_equal"] = (m.group(1) == "True") if m else None
    m = re.search(r"BIT_ID[^\n]*digest_sha256=([0-9a-f]+)", text)
    out["digest_sha256"] = m.group(1) if m else None
    out["all_finite"] = ("all_finite=True" in text)
    return out


def build_row(args) -> dict:
    cap = _read(args.cap_log)
    warm = _read(args.warm_log)
    cp = _parse_phase(cap)
    wp = _parse_phase(warm)

    # cold-compile peak RSS: capture phase is where the fused compile happens.
    cold_rss = max(
        x for x in [cp.get("driver_rss_peak_mib"), cp.get("time_v_maxrss_mib")] if x is not None
    ) if any(cp.get(k) is not None for k in ("driver_rss_peak_mib", "time_v_maxrss_mib")) else None
    warm_rss = max(
        x for x in [wp.get("driver_rss_peak_mib"), wp.get("time_v_maxrss_mib")] if x is not None
    ) if any(wp.get(k) is not None for k in ("driver_rss_peak_mib", "time_v_maxrss_mib")) else None

    cold_wall = cp.get("cold_wall_s") or cp.get("max_call_wall_s") or cp.get("time_v_wall_s")
    # runtime s/step: prefer the WARM phase fixed s/step (warm executable),
    # fall back to capture.
    s_step = wp.get("fixed_s_per_step")
    if s_step is None or s_step <= 0:
        s_step = wp.get("s_per_step_b")
    if s_step is None or s_step <= 0:
        s_step = cp.get("fixed_s_per_step") or cp.get("s_per_step_b")

    blob_valid = (
        (wp.get("aot_loaded_count", 0) > 0)
        and (wp.get("aot_load_fail_count", 0) == 0)
        and (not wp.get("jit_fused_recompile"))
    )

    row = {
        "setting": args.setting,
        "maxdom": args.maxdom,
        "flags": args.flags,
        "cap_rc": args.cap_rc,
        "warm_rc": args.warm_rc,
        "cache_dir": args.cache_dir,
        "cold_compile_wall_s": cold_wall,
        "cold_peak_rss_mib": cold_rss,
        "cold_peak_rss_gb": round(cold_rss / 1024.0, 3) if cold_rss else None,
        "warm_peak_rss_mib": warm_rss,
        "s_per_step": s_step,
        "aot_blob_loaded_count": wp.get("aot_loaded_count"),
        "aot_blob_load_fail_count": wp.get("aot_load_fail_count"),
        "jit_fused_recompile_in_warm": wp.get("jit_fused_recompile"),
        "aot_blob_valid": blob_valid,
        "bit_identical_ref_equal": wp.get("ref_equal"),
        "warm_digest_sha256": wp.get("digest_sha256"),
        "cap_digest_sha256": cp.get("digest_sha256"),
        "all_finite": cp.get("all_finite") and wp.get("all_finite"),
        "driver_raised": cp.get("driver_raised") or wp.get("driver_raised"),
    }
    return row


def summarize(jsonl_path: str) -> None:
    rows = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
    base = next((r for r in rows if r["setting"] == "baseline"), None)
    if base is None:
        print("NO BASELINE ROW; cannot compute deltas", file=sys.stderr)
    base_rss = base.get("cold_peak_rss_mib") if base else None
    base_sstep = base.get("s_per_step") if base else None
    print(f"\n=== R0 SWEEP SUMMARY (baseline cold_rss={base_rss} MiB, s/step={base_sstep}) ===")
    hdr = f"{'setting':14s} {'cold_rss_GB':>11s} {'rss_delta%':>10s} {'s/step':>9s} {'runtime_reg%':>12s} {'blob':>5s} {'bitid':>6s} {'PASS':>5s}"
    print(hdr)
    for r in rows:
        rss = r.get("cold_peak_rss_mib")
        rss_delta = ((rss - base_rss) / base_rss * 100.0) if (rss and base_rss) else None
        sstep = r.get("s_per_step")
        reg = ((sstep - base_sstep) / base_sstep * 100.0) if (sstep and base_sstep) else None
        blob = bool(r.get("aot_blob_valid"))
        bitid = r.get("bit_identical_ref_equal")
        # PASS: materially lower RAM (>=10% lower) AND no runtime regression
        # (<=2%) AND valid blob AND bit-identical.
        material = (rss_delta is not None and rss_delta <= -10.0)
        no_reg = (reg is not None and reg <= 2.0)
        is_pass = bool(material and no_reg and blob and bitid is True and r["setting"] != "baseline")
        print(
            f"{r['setting']:14s} "
            f"{(rss/1024.0 if rss else float('nan')):11.2f} "
            f"{(rss_delta if rss_delta is not None else float('nan')):10.1f} "
            f"{(sstep if sstep else float('nan')):9.3f} "
            f"{(reg if reg is not None else float('nan')):12.1f} "
            f"{str(blob):>5s} {str(bitid):>6s} {str(is_pass):>5s}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", nargs="?", const=None)
    ap.add_argument("--setting")
    ap.add_argument("--maxdom", type=int)
    ap.add_argument("--flags", default="")
    ap.add_argument("--cap-log")
    ap.add_argument("--warm-log")
    ap.add_argument("--cap-rc", type=int)
    ap.add_argument("--warm-rc", type=int)
    ap.add_argument("--cache-dir")
    ap.add_argument("--base-ref", default="")
    args, extra = ap.parse_known_args()
    if args.summarize:
        summarize(args.summarize)
        return
    row = build_row(args)
    print(json.dumps(row))


if __name__ == "__main__":
    main()
