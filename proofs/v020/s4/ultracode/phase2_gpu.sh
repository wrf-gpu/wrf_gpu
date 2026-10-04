#!/usr/bin/env bash
# phase2_gpu.sh — runs AFTER the bigswiss arms. Two things, one GPU-lock hold:
#  (1) confirmatory back-to-back fp64 1h on bigswiss (bounds run-to-run drift in
#      the mixed speedup claim).
#  (2) synthetic DYNAMICS-ONLY VRAM-vs-size ladder (physics OFF): fp64 vs the
#      fp32-gated aggressive matrix, growing the grid until fp64 OOMs -> the
#      "does fp32 FIT what fp64 cannot" crossover for the dynamics core (the
#      fp32-addressable VRAM; the RRTMG transient is a separate tile-bounded add).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../../.." && pwd)"
INPUT="${V020_BIG_INPUT:-<DATA_ROOT>/wrf_gpu_validation/v017_bigswiss_gpu_init}"
JAX_CACHE="${V020_JAX_CACHE:-<DATA_ROOT>/gpuwrf_jax_cache}"
RR="$HERE/phase2_$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$RR"
COMMON_ENV=(PYTHONPATH="$ROOT/src" XLA_PYTHON_CLIENT_PREALLOCATE=false
  XLA_PYTHON_CLIENT_ALLOCATOR=platform JAX_ENABLE_COMPILATION_CACHE=true
  JAX_COMPILATION_CACHE_DIR="$JAX_CACHE" OMP_NUM_THREADS=4 GPUWRF_MYNN_BOULAC_ONZ=1)

echo "[p2] === (1) confirmatory fp64 1h ==="
out="$RR/fp64c/out"; mkdir -p "$out" "$RR/fp64c/proofs" "$RR/fp64c/scratch"
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null >> "$RR/fp64c.vram.csv"; sleep 1; done ) & VP=$!
env "${COMMON_ENV[@]}" taskset -c 0-3 timeout 1500 python -m gpuwrf run \
  --input-dir "$INPUT" --namelist "$INPUT/namelist.input" \
  --output-dir "$out" --proof-dir "$RR/fp64c/proofs" --scratch-dir "$RR/fp64c/scratch" \
  --max-dom 1 --domain d01 --hours 1 > "$RR/fp64c.log" 2>&1
kill $VP 2>/dev/null||true
FO=$(python3 -c "import json,glob;fs=glob.glob('$RR/fp64c/proofs/pipeline_run*.json');print(json.load(open(fs[0]))['wall_clock_forecast_only_s'] if fs else 'NA')" 2>/dev/null)
PK=$(grep -oE "[0-9]+ MiB" "$RR/fp64c.vram.csv"|grep -oE "[0-9]+"|sort -n|tail -1)
echo "[p2] fp64-confirm forecast_only_s=$FO peak=${PK}MiB"

echo "[p2] === (2) synthetic dynamics-only VRAM ladder (physics OFF) ==="
LADDER=( "50 400 400" "50 700 700" "50 1000 1000" "50 1300 1300" "50 1600 1600" "50 1900 1900" "50 2200 2200" )
SCAN="$RR/synth_scan.jsonl"; : > "$SCAN"
fp64_oomed=0
for dims in "${LADDER[@]}"; do
  read -r nz ny nx <<< "$dims"
  for regime in fp64 aggressive; do
    # skip fp64 once it has OOMed (it only gets worse); keep probing aggressive
    [[ "$regime" == "fp64" && "$fp64_oomed" == "1" ]] && continue
    of="$RR/synth_${regime}_${ny}x${nx}.json"
    env "${COMMON_ENV[@]}" taskset -c 0-3 timeout 600 python "$HERE/synth_vram_scan.py" \
      --nz "$nz" --ny "$ny" --nx "$nx" --steps 2 --regime "$regime" --out "$of" \
      > "$RR/synth_${regime}_${ny}x${nx}.log" 2>&1 || true
    if [[ -f "$of" ]]; then
      cat "$of" | python3 -c "import sys,json;d=json.load(sys.stdin);print('[p2] synth',d['regime'],f\"{d['ny']}x{d['nx']}\",'oom=',d['oom'],'peak_mib=',d.get('peak_vram_mib'),'fp32frac=',round(d.get('state_dtype_fraction_fp32',0),3))"
      cat "$of" >> "$SCAN"; echo >> "$SCAN"
      oom=$(python3 -c "import json;print(json.load(open('$of'))['oom'])")
      [[ "$regime" == "fp64" && "$oom" == "True" ]] && fp64_oomed=1
    else
      echo "[p2] synth $regime ${ny}x${nx} -> no json (timeout/crash); treat as OOM/FAIL"
      [[ "$regime" == "fp64" ]] && fp64_oomed=1
    fi
  done
  # stop once fp64 OOMs AND aggressive also OOMs (crossover captured + upper bound)
  if [[ "$fp64_oomed" == "1" ]]; then
    last_agg="$RR/synth_aggressive_${ny}x${nx}.json"
    if [[ -f "$last_agg" ]]; then
      aoom=$(python3 -c "import json;print(json.load(open('$last_agg'))['oom'])")
      [[ "$aoom" == "True" ]] && { echo "[p2] both OOM at ${ny}x${nx}; stop"; break; }
    fi
  fi
done
echo "[p2] DONE -> $RR  (scan: $SCAN)"
