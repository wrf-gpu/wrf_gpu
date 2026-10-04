#!/usr/bin/env bash
# Rolling per-checkpoint CPU-vs-GPU validation gate (principal policy 2026-06-15).
# Compares each GPU history hour to the CPU reference AS it is produced; HALTS + alerts
# on the FIRST out-of-tolerance deviation (drift/NaN/CFL/norm). Green = every checkpoint passes.
#
# Usage:
#   rolling_cpu_gpu_validate.sh <gpu_dir> <cpu_dir> <domain> <init_iso> <max_lead> [tolerance_json]
# Example:
#   rolling_cpu_gpu_validate.sh /mnt/.../gpu_output /mnt/.../run_cpu d01 2023-01-15T00:00:00+00:00 8 \
#       proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json
#
# Run it via the GPU lock-free side (it only READS wrfout). Arm it as a background task so the
# manager is re-invoked on a DEVIATION (exit 2) or completion (exit 0). exit 3 = stuck/timeout.
set -uo pipefail
GPU=$1; CPU=$2; DOM=$3; INIT=$4; MAXLEAD=$5
TOL=${6:-<USER_HOME>/src/wrf_gpu2/proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json}
CMP=<USER_HOME>/src/wrf_gpu2/scripts/compare_wrfout_grid.py
OUT="$(dirname "$GPU")/rolling_validate"; mkdir -p "$OUT"
LOG="$OUT/rolling.log"
ngpu(){ ls "$GPU"/wrfout_${DOM}_* 2>/dev/null | wc -l; }
ncpu(){ ls "$CPU"/wrfout_${DOM}_* 2>/dev/null | wc -l; }
echo "[$(date +%H:%M:%S)] ROLLING VALIDATE start: dom=$DOM init=$INIT max_lead=$MAXLEAD gpu=$GPU cpu=$CPU" | tee -a "$LOG"
for lead in $(seq 1 "$MAXLEAD"); do
  # wait until BOTH sides have produced hour <lead> (h0..hlead => need >= lead+1 files), cap ~2h/lead
  ok=0
  for w in $(seq 1 240); do
    if [ "$(ngpu)" -ge $((lead+1)) ] && [ "$(ncpu)" -ge $((lead+1)) ]; then ok=1; break; fi
    sleep 30
  done
  if [ "$ok" -ne 1 ]; then echo "[$(date +%H:%M:%S)] lead $lead: TIMEOUT waiting for checkpoint (gpu=$(ngpu) cpu=$(ncpu)) -> investigate" | tee -a "$LOG"; exit 3; fi
  python3 "$CMP" --cpu-dir "$CPU" --gpu-dir "$GPU" --domain "$DOM" --init "$INIT" \
      --min-lead "$lead" --max-lead "$lead" --tolerance-json "$TOL" \
      --out-json "$OUT/lead_${lead}.json" --out-md "$OUT/lead_${lead}.md" >/dev/null 2>&1
  v=$(python3 - "$OUT/lead_${lead}.json" <<'PY'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
except Exception as e:
    print("PARSE_ERROR"); sys.exit()
o=d.get("overall",d)
for k in ("verdict","status"):
    if isinstance(o.get(k),str): print(o[k].upper()); sys.exit()
p=o.get("pass", d.get("pass"))
print("PASS" if p else "FAIL")
PY
)
  echo "[$(date +%H:%M:%S)] lead $lead (valid=init+${lead}h): $v" | tee -a "$LOG"
  case "$v" in
    PASS|GREEN|OK) : ;;
    *) echo "[$(date +%H:%M:%S)] *** DEVIATION at lead $lead = $v -> HALT GPU RUN + ROOT-CAUSE (see $OUT/lead_${lead}.md) ***" | tee -a "$LOG"; exit 2 ;;
  esac
done
echo "[$(date +%H:%M:%S)] ALL $MAXLEAD CHECKPOINTS GREEN (checkpoint-by-checkpoint evidence)" | tee -a "$LOG"
exit 0
