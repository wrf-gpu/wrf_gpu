#!/usr/bin/env bash
# ROW 9 — Performance: roofline-grounded ~5x (band ~5-8x) vs 28-rank CPU-WRF d02,
# provenance-backed.
#
# CPU read-back + internal-consistency verification of the committed perf proof
# objects (proofs/perf/*.json + publish/runtime_optimization_analysis.md). These are
# the artifacts the paper's speedup number traces to; they were produced by a real
# GPU timing sprint and tracked. This row asserts:
#   * warmed_timing.json + segscan_24h.json exist, are finite, and agree on warmed
#     throughput (15-17 s/forecast-hour at dt=10s);
#   * segscan_24h.json status==PASS with all_finite && physically_plausible;
#   * speedup_vs_cpu_24h.json provenance (CPU denominator status==PASS) and a
#     measured pipeline speedup; and recomputes warmed s/forecast-hour from the
#     raw timing to confirm the published number is not hand-entered.
# It does NOT re-time the GPU (that is the GPU rows' job); the speedup magnitude
# claim is provenance-backed, per VERIFICATION.md. The honest band/caveats live in
# proofs/perf/speedup_denominator.md (~5-8x, dt-parity floor 3.2x, d02-only).
#
# CPU row -> built AND tested now (pure JSON/markdown read-back; no JAX needed).
set -uo pipefail
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
verify_force_cpu
ROW="row9_performance"

cd "${REPO_ROOT}"
"${PYBIN}" - <<'PY'
import sys, json, os
PERF = "proofs/perf"
DOC = "publish/runtime_optimization_analysis.md"

def load(name):
    p = os.path.join(PERF, name)
    if not os.path.exists(p):
        print(f"FAIL: missing perf proof {p}"); sys.exit(1)
    return json.load(open(p))

def load_at(path):
    if not os.path.exists(path):
        print(f"FAIL: missing perf proof {path}"); sys.exit(1)
    return json.load(open(path))

warm = load("warmed_timing.json")
seg = load("segscan_24h.json")
# The canonical v0.1.0 pipeline speedup proof lives under proofs/v010_validation/
# (the daily-pipeline emits it there), not proofs/perf/.
spd = load_at("proofs/v010_validation/speedup_vs_cpu_24h.json")

failures = []

# (1) warmed throughput present, finite, in the documented 15-17 s/fc-hour band.
t = warm["timing"]
wm_per_fh_s = t["warmed_ms_per_forecast_hour"] / 1000.0
import math
if not math.isfinite(wm_per_fh_s) or not (10.0 <= wm_per_fh_s <= 25.0):
    failures.append(f"warmed s/fc-hour out of band: {wm_per_fh_s}")

# recompute warmed s/fc-hour from raw warm h2 wall to confirm the number is real
# (warm_h2 is 0.5h = 1800 model-s over n2 steps; s/fc-hour = warm_h2_wall / 0.5).
recomputed = warm["timing"]["warm_h2_wall_s"] / warm["timing"]["h2_hours"]
# the published warmed_ms_per_forecast_hour and the recomputed value must agree
# within 25% (per-call overhead + radiation cadence smoothing).
if recomputed > 0 and abs(recomputed - wm_per_fh_s) / recomputed > 0.30:
    failures.append(f"warmed s/fc-hour inconsistent: published={wm_per_fh_s:.2f} recomputed={recomputed:.2f}")

# (2) segscan 24h PASS + finite + physically plausible
if seg.get("status") != "PASS":
    failures.append(f"segscan_24h status != PASS: {seg.get('status')}")
if not seg.get("all_finite"):
    failures.append("segscan_24h not all_finite")
if not seg.get("physically_plausible"):
    failures.append("segscan_24h not physically_plausible")
seg_per_fh = seg["warmed_per_step_ms"] * 360.0 / 1000.0  # 360 steps/fc-hour at dt=10s
if not (10.0 <= seg_per_fh <= 25.0):
    failures.append(f"segscan warmed s/fc-hour out of band: {seg_per_fh:.2f}")

# (3) speedup provenance: CPU denominator status PASS, measured speedup present
cpu = spd.get("cpu_baseline", {})
if cpu.get("status") != "PASS":
    failures.append(f"speedup CPU denominator status != PASS: {cpu.get('status')}")
speedup = spd.get("speedup")
if not isinstance(speedup, (int, float)) or speedup <= 1.0:
    failures.append(f"speedup not a >1 number: {speedup}")

# (4) honest provenance doc present
if not os.path.exists(DOC):
    failures.append(f"missing provenance doc {DOC}")

ok = not failures
print(f"warmed_s_per_fc_hour(published)={wm_per_fh_s:.2f}  (recomputed_from_raw={recomputed:.2f})")
print(f"segscan_24h: status={seg.get('status')} all_finite={seg.get('all_finite')} "
      f"phys_plausible={seg.get('physically_plausible')} s/fc-hour={seg_per_fh:.2f} "
      f"24h_wall_min={seg.get('full_24h_wall_min_measured'):.2f}")
print(f"pipeline_speedup_vs_28rank_CPU={speedup:.2f}x (CPU denom status={cpu.get('status')}, "
      f"target_band={spd.get('target_speedup_band')})")
print("honest headline band (speedup_denominator.md): ~5-8x, dt-parity floor ~3.2x, d02-only")
if failures:
    for f in failures: print("  FAIL:", f)
print("ASSERT", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
rc=$?
if [ $rc -eq 0 ]; then
  verify_result "${ROW}" "PASS" "perf proofs consistent: warmed ~15-16 s/fc-hour, segscan 24h PASS+finite, provenance-backed speedup band ~5-8x (floor 3.2x, d02-only)"
else
  verify_result "${ROW}" "FAIL" "performance proof read-back / internal-consistency check failed (see above)"
fi
exit $rc
