#!/usr/bin/env bash
# Chain: wait for the Switzerland sweep to finish (all 4 result.json present),
# pick the best safe rung, then fire the 3-dom Canary validation on it.
# Runs detached; safe to leave running across agent turns.
set -uo pipefail

cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"
LADDER="$ROOT/proofs/v022/k2_dt_ladder"
LOG="$LADDER/logs/chain.log"
exec >>"$LOG" 2>&1
echo "=== chain start $(date -u +%FT%TZ) ==="

# 1) Wait until all four rung result.json files exist AND the sweep proc is gone.
while :; do
  n=0
  for tag in R1 R2 R3 R4; do
    [ -f "$LADDER/runs/$tag/result.json" ] && n=$((n+1))
  done
  sweep_alive=$(pgrep -f "k2-dt-ladder-switz" | head -1 || true)
  if [ "$n" -eq 4 ] && [ -z "$sweep_alive" ]; then
    echo "sweep complete: $n/4 rungs, no live sweep proc"
    break
  fi
  echo "waiting: $n/4 rungs done, sweep_alive=${sweep_alive:-none} $(date -u +%T)"
  sleep 60
done

# 2) Pick the best safe rung.
taskset -c 0-3 env PYTHONPATH=src python "$LADDER/pick_best_rung.py" \
  --ladder-dir "$LADDER" --out "$LADDER/best_rung.json"

BEST=$(python - "$LADDER/best_rung.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
b=d.get("best_safe_detail")
if not b:
    print("NONE 1.0 10"); raise SystemExit
# scale factor = best dt / R1 dt (10s)
scale = float(b["dt_s"])/10.0
print(f"{b['tag']} {scale:g} {b['n_sound']}")
PY
)
echo "best rung picked: $BEST"
TAG=$(echo "$BEST" | awk '{print $1}')
SCALE=$(echo "$BEST" | awk '{print $2}')
NSOUND=$(echo "$BEST" | awk '{print $3}')

if [ "$TAG" = "NONE" ]; then
  echo "NO SAFE RUNG ABOVE R1 — Canary validation skipped (R1 is the only safe config)."
  echo "=== chain done (no-safe-rung) $(date -u +%FT%TZ) ==="
  exit 0
fi

# 3) Fire the 3-dom Canary validation on the best rung (own GPU-lock acquisition).
echo "firing Canary validation: scale=$SCALE nsound=$NSOUND"
K2_SCALE_FACTOR="$SCALE" K2_BEST_NSOUND="$NSOUND" K2_CANARY_HOURS=1 \
  bash "$LADDER/run_canary_validate.sh"

echo "=== chain done $(date -u +%FT%TZ) ==="
