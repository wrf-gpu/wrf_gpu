#!/usr/bin/env bash
# K2 dt/n_sound ladder -- 3-dom CANARY VALIDATION of the best safe Switzerland rung.
#
# Cost guard: validate ONLY the best safe rung on the 3-dom Canary (dt 54/18/6
# baseline -> scaled by the same factor the Switzerland sweep chose). NOT the
# all-7 9-nest.
#
# Inputs (env):
#   K2_CANARY_INPUT_DIR  default <DATA_ROOT>/wrf_downscale/runs/20240901/cpu
#   K2_SCALE_FACTOR      dt scale factor of the best rung (e.g. 1.5 for dt 10->15)
#   K2_BEST_NSOUND       n_sound of the best rung (e.g. 8)
#   K2_CANARY_HOURS      default 1  (each 3-dom hour is expensive; 1h is the gate)
#
# It runs TWO 3-dom forecasts under ONE GPU lock:
#   R1c  : root dt = baseline (n_sound 10)
#   BESTc: root dt = baseline * K2_SCALE_FACTOR  (n_sound = K2_BEST_NSOUND)
# then computes per-domain max-Courant for each and compares BESTc to R1c.
set -euo pipefail

cd "$(dirname "$0")/../../.."
ROOT="$(pwd)"
LADDER_DIR="$ROOT/proofs/v022/k2_dt_ladder"
LOG_DIR="$LADDER_DIR/logs"
mkdir -p "$LOG_DIR"

CANARY_INPUT="${K2_CANARY_INPUT_DIR:-<DATA_ROOT>/wrf_downscale/runs/20240901/cpu}"
SCALE="${K2_SCALE_FACTOR:?set K2_SCALE_FACTOR}"
NSOUND="${K2_BEST_NSOUND:?set K2_BEST_NSOUND}"
HOURS="${K2_CANARY_HOURS:-1}"
MAXDOM="${K2_CANARY_MAXDOM:-3}"

scripts/with_gpu_lock.sh --timeout 21600 --label k2-canary-validate -- bash -lc '
set -euo pipefail
cd "'"$ROOT"'"
CANARY_INPUT="'"$CANARY_INPUT"'"
SCALE="'"$SCALE"'"
NSOUND="'"$NSOUND"'"
HOURS="'"$HOURS"'"
MAXDOM="'"$MAXDOM"'"
LADDER_DIR="'"$LADDER_DIR"'"
LOG_DIR="'"$LOG_DIR"'"

# Build a scaled-namelist input dir: symlink wrfinput/wrfbdy, edit only the root
# domains.time_step. The validation run R1c uses the unscaled root dt.
build_input () {
  local tag="$1" factor="$2"
  local dst="$LADDER_DIR/canary_runs/$tag/input"
  rm -rf "$dst"; mkdir -p "$dst"
  for f in "$CANARY_INPUT"/wrfinput_d* "$CANARY_INPUT"/wrfbdy_d*; do
    [ -e "$f" ] && ln -sf "$f" "$dst/$(basename "$f")"
  done
  taskset -c 0-3 env PYTHONPATH=src K2_FACTOR="$factor" K2_MAXDOM="$MAXDOM" \
    python proofs/v022/k2_dt_ladder/scale_namelist.py \
      "$CANARY_INPUT/namelist.input" "$dst/namelist.input"
}

run_canary () {
  local tag="$1" nsound="$2"
  local input="$LADDER_DIR/canary_runs/$tag/input"
  local out="$LADDER_DIR/canary_runs/$tag/out"
  local proof="$LADDER_DIR/canary_runs/$tag/proof"
  local scratch="$LADDER_DIR/canary_runs/$tag/scratch"
  mkdir -p "$out" "$proof" "$scratch"
  echo "=== CANARY $tag : n_sound=$nsound maxdom=$MAXDOM hours=$HOURS ==="
  taskset -c 0-7 env \
    OMP_NUM_THREADS=8 PYTHONPATH=src JAX_ENABLE_X64=true \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    GPUWRF_CANAIRY_ROOT=<DATA_ROOT>/canairy_meteo \
    GPUWRF_ACOUSTIC_SUBSTEPS="$nsound" \
    python -m gpuwrf.cli run \
      --namelist "$input/namelist.input" \
      --input-dir "$input" \
      --output-dir "$out" \
      --proof-dir "$proof" \
      --scratch-dir "$scratch" \
      --domain d01 --max-dom "$MAXDOM" --hours "$HOURS" \
      2>&1 | tee "$LOG_DIR/canary_${tag}.log"
}

build_input R1c 1.0
build_input BESTc "$SCALE"

run_canary R1c 10
run_canary BESTc "$NSOUND"

# Per-domain Courant + compare for each domain.
taskset -c 0-3 env PYTHONPATH=src JAX_ENABLE_X64=true \
  python proofs/v022/k2_dt_ladder/canary_courant_report.py \
    --r1-dir "$LADDER_DIR/canary_runs/R1c/out" \
    --best-dir "$LADDER_DIR/canary_runs/BESTc/out" \
    --scale "$SCALE" --nsound "$NSOUND" --maxdom "$MAXDOM" \
    --out "$LADDER_DIR/canary_runs/canary_validation.json" \
    2>&1 | tee "$LOG_DIR/canary_report.log"

echo "=== CANARY VALIDATION DONE ==="
'
