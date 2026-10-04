"""bench_summary.py — turn s0_nested_harness receipts (+ optional nsys report) into bench.json.

Usage: bench_summary.py --out bench.json --tag T --cache DIR [--nsys prof.nsys-rep] [cold=r.json] warm1=r.json ...
Warm S1 value = mean s/fc-h of root segments 2..N (segment 1 carries trace/first-dispatch cost) of each warm arm;
the headline `s1_warm_s_per_fc_h` is the mean over warm arms, `aa_floor_pct` the spread between the first two.
Per-segment decomposition (all s per forecast hour of segment k, window = [t_start_k, t_start_k+1)):
  output_s      sum of output-callback wall (S0:output ranges)       [M]
  output_compile_s  backend_compile events falling inside output calls [M]
  stepping_s    period - output (stepping, radiation, force, host)   [M by subtraction]
  host_idle_est_s  period - (advance + force + output host time): time outside every instrumented call [I]
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

DT_ROOT_S = 54.0  # PROD d01 dt; matches the harness rate definition
HISTORICAL_AA_FLOOR_PCT = 6.77  # FINDINGS A5 (contended box)


def _fc_h(root_steps: int) -> float:
    return root_steps * DT_ROOT_S / 3600.0


def _overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> bool:
    """True if open intervals [a_lo,a_hi) and [b_lo,b_hi) overlap (no grace)."""
    return a_lo < b_hi and b_lo < a_hi


def segment_rows(rec: dict) -> list[dict]:
    segs = sorted(rec.get("segments", []), key=lambda s: s["k"])
    t_end = rec.get("t_end", segs[-1]["t_return"] if segs else 0.0)
    rows = []
    for i, s in enumerate(segs):
        lo = s["t_start"]
        hi = segs[i + 1]["t_start"] if i + 1 < len(segs) else max(t_end, s["t_return"])
        h = _fc_h(s.get("root_steps") or 67)
        outs = [o for o in rec.get("outputs", []) if lo <= o["t_start"] < hi]
        out_s = sum(o["s"] for o in outs)
        # compile_events record t at the event END, so the event interval is [t-s, t]; attribute by a real
        # overlap with an output interval only (no half-second grace).
        out_comp = sum(e["s"] for e in rec.get("compile_events", []) if "backend_compile" in e["event"]
                       and any(_overlap(o["t_start"], o["t_start"] + o["s"], e["t"] - e["s"], e["t"])
                               for o in outs))
        all_comp = sum(e["s"] for e in rec.get("compile_events", []) if "backend_compile" in e["event"]
                       and lo <= e["t"] < hi)
        adv = sum(a["host_s"] for a in rec.get("advance", []) if lo <= a["t"] < hi)
        frc = sum(f["host_s"] for f in rec.get("force", []) if lo <= f["t"] < hi)
        period = hi - lo
        rows.append({
            "k": s["k"], "root_steps": s.get("root_steps"), "period_s": round(period, 3),
            "s_per_fc_h": round(period / h, 3), "output_s_per_fc_h": round(out_s / h, 3),
            "output_compile_s_per_fc_h": round(out_comp / h, 3),
            "stepping_s_per_fc_h": round((period - out_s) / h, 3),
            "compile_total_s_per_fc_h": round(all_comp / h, 3),
            "host_idle_est_s_per_fc_h": round((period - adv - frc - out_s) / h, 3),
            "n_outputs": len(outs),
        })
    return rows


def arm_summary(path: Path) -> dict:
    rec = json.loads(path.read_text())
    rows = segment_rows(rec)
    warm = [r for r in rows if r["k"] >= 2]
    mean = lambda key: round(sum(r[key] for r in warm) / len(warm), 3) if warm else None  # noqa: E731
    return {
        "receipt": str(path), "rc": rec.get("rc"), "end_reason": rec.get("end_reason"),
        "t_end_s": rec.get("t_end"), "git_head": rec.get("git_head"), "src_tree": rec.get("src_tree"),
        "jax_version": rec.get("jax_version"), "segments": rows,
        "s_per_fc_h_seg2plus": mean("s_per_fc_h"), "stepping_s_per_fc_h": mean("stepping_s_per_fc_h"),
        "output_s_per_fc_h": mean("output_s_per_fc_h"),
        "output_compile_s_per_fc_h": mean("output_compile_s_per_fc_h"),
        "compile_total_s_per_fc_h": mean("compile_total_s_per_fc_h"),
        "host_idle_est_s_per_fc_h": mean("host_idle_est_s_per_fc_h"),
        "vram_peak_mib_nvidia_smi": (rec.get("derived") or {}).get("vram_peak_mib"),
        "jax_memory_stats": rec.get("jax_memory_stats"), "host_vmhwm_kb": rec.get("host_vmhwm_kb"),
        "errors": rec.get("errors"), "env": rec.get("env"),
    }


def aa_floor(warm: list[dict]) -> dict | None:
    if len(warm) < 2:
        return None
    a, b = warm[0]["s_per_fc_h_seg2plus"], warm[1]["s_per_fc_h_seg2plus"]
    pct = abs(a - b) / a * 100.0
    return {"a": a, "b": b, "aa_floor_pct": round(pct, 2), "historical_floor_pct": HISTORICAL_AA_FLOOR_PCT,
            "within_historical_floor": pct <= HISTORICAL_AA_FLOOR_PCT}


def nsys_summary(rep: Path, census_script: Path) -> dict:
    """Export the nsys report to sqlite, run the S0 node census, return per-step kernel / copy counts.

    The GPU-idle fraction uses ONE window for both numerator and denominator: `device_span_s` is the span
    of the same device-op set that `device_union_s` unions (the NVTX segment range excludes the post-call
    tail that the capture window includes, so mixing them can yield negative idle)."""
    db = rep.with_suffix(".sqlite")
    subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite=true", "-o", str(db), str(rep)],
                   check=True, capture_output=True)
    out = rep.with_suffix(".census.json")
    subprocess.run([sys.executable, str(census_script), str(db), str(out)],
                   check=True, capture_output=True)
    c = json.loads(out.read_text())
    span = c.get("device_span_s")          # span of the same device-op set as device_union_s
    busy = c.get("device_union_s")
    kinds = c["totals"]["memcpy_by_kind"]  # CUPTI copyKind: 1 HtoD, 2 DtoH, 8 DtoD
    idle = None
    if span:
        idle = round(min(max(1.0 - busy / span, 0.0), 1.0), 4)
    detail = c.get("class_op_detail", {})
    per_class = {k: {"ops": v.get("ops"), "kernels": v.get("kernels"),
                     "htod": v.get("memcpy_1"), "dtoh": v.get("memcpy_2"), "dtod": v.get("memcpy_8")}
                 for k, v in detail.items()}
    return {
        "census": str(out),
        "window": "device_span_s (span of all captured device ops) == same op set as device_union_s",
        "kernels_total_segment": c["totals"]["kernels"],
        "device_ops_total_segment": c["totals"]["device_ops"],
        "ops_per_step_class_median": c["class_medians"],
        "per_step_class_kernels_htod_dtoh": per_class,
        "htod_copies_segment": kinds.get("1", 0), "dtoh_copies_segment": kinds.get("2", 0),
        "dtod_copies_segment": kinds.get("8", 0),
        "device_span_s": span, "device_union_s": busy,
        "gpu_idle_frac_in_segment": idle,
        "nvtx_segment_span_s": (c.get("captured_segment_span_s") or [None])[0],
        "unattributed_ops": c["unattributed_ops"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache", default="")
    ap.add_argument("--src", default="", help="source tree measured (path); recorded in bench.json")
    ap.add_argument("--src-tree", default="", help="authoritative src/gpuwrf tree hash; overrides the receipt")
    ap.add_argument("--src-rev", default="", help="measured source revision (commit); recorded in bench.json")
    ap.add_argument("--xla-flags", default="", help="XLA_FLAGS exported for every arm; recorded in bench.json")
    ap.add_argument("--nsys")
    ap.add_argument("arms", nargs="+", help="name=receipt.json (cold, warm1, warm2, ...)")
    a = ap.parse_args(argv)
    arms = {n: arm_summary(Path(p)) for n, p in (x.split("=", 1) for x in a.arms)}
    warm = [v for k, v in arms.items() if k.startswith("warm")]
    first = next(iter(arms.values()))
    # Reject incomplete / incompatible arms BEFORE emitting any headline (item: no headline on bad evidence).
    trees = {v["src_tree"] for v in arms.values()}
    reasons = []
    if len(trees) > 1:
        reasons.append(f"arms span multiple src trees: {sorted(trees)}")
    if len(warm) < 2:
        reasons.append(f"need >=2 warm arms for the A/A gate, got {len(warm)}")
    if any(w["rc"] != 0 for w in warm):
        reasons.append("a warm arm has rc != 0")
    if any(w["s_per_fc_h_seg2plus"] is None for w in warm):
        reasons.append("a warm arm has no warm (seg2+) segment")
    if any(v["rc"] != 0 for v in arms.values()):
        reasons.append("a non-warm arm has rc != 0")
    headline_ok = not reasons
    res = {
        "schema": "wrf_gpu2.bench.v1", "tag": a.tag, "ok": headline_ok, "case": "PROD ALISIOS 20260725_18z d01+d02",
        "headline_valid": headline_ok, "reject_reasons": reasons,
        "tree": {"git_head": first["git_head"], "src_tree": first["src_tree"]},
        "cache_dir": a.cache, "arms": arms, "xla_flags": a.xla_flags or None,
        "src": {"path": a.src or None, "rev": a.src_rev or None,
                "tree_hash": a.src_tree or first["src_tree"]},
        "s1_warm_s_per_fc_h": (round(sum(w["s_per_fc_h_seg2plus"] for w in warm) / len(warm), 3)
                               if headline_ok else None),
        "aa": aa_floor(warm) if headline_ok else None,
        "env": {"python": platform.python_version(), "host": platform.node(), "jax": first["jax_version"]},
    }
    if a.src_tree:
        res["tree"]["src_tree"] = a.src_tree
    if "cold" in arms and arms["cold"]["segments"]:
        res["cold_s_per_fc_h_seg1"] = arms["cold"]["segments"][0]["s_per_fc_h"]
    if a.nsys:
        census_script = Path(__file__).resolve().parent / "bench" / "s0_nsys_census.py"
        res["nsys"] = nsys_summary(Path(a.nsys), census_script)
    Path(a.out).write_text(json.dumps(res, indent=1))
    for r in reasons:
        print(f"bench_summary: REJECT headline: {r}", file=sys.stderr)
    print(json.dumps({k: res[k] for k in ("tag", "ok", "s1_warm_s_per_fc_h", "aa")}, indent=1))
    return 0 if headline_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
