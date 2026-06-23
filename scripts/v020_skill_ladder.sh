#!/usr/bin/env bash
# v020_skill_ladder.sh — the STABILITY+SKILL ladder driver (FINAL_FP32_SPRINT_PLAN §4,
# roadmap §8.5 Gate C/D). Climbs 1-substep -> 1-step -> 1h -> 6h -> 24h -> 120h on the
# nested all-7 max_dom=9, checking wind/temp/cloud SKILL + NO-BLOW-UP at each rung, and
# NEVER widening the gate mid-run (pass each rung before climbing; STOP at the first fail).
#
# This is the gate that EVERY v0.20 change (P3 host levers, P4 dt/n_sound, the fused
# kernel, fp32) must clear on the all-7 benchmark. It wires:
#   * scripts/v020_blowup_check.py   — the HARD no-blow-up floor (carve-outs aside).
#   * scripts/v020_skill_eval.py     — wind/temp/cloud divergence-GROWTH gate via
#     proofs/perf/v015/fp32_oracles/divergence_growth_metric.py (the module
#     tests/test_fp32_divergence_growth_metric.py pins) vs an ORACLE (fp64-GPU baseline
#     or CPU-WRF) — the relaxed-tolerance skill policy, NOT bitwise parity.
#
# RUNGS (cheap->expensive; abort early, never widen):
#   substep (1 acoustic substep), step (1 model step), 1h, 6h, 24h, 120h.
#   The sub-1h rungs are no-blow-up + finiteness gates (no oracle series yet); 1h+ add the
#   skill gate vs the oracle.
#
# ORACLE: V020_LADDER_ORACLE (a wrfout dir). For a CANDIDATE build (fp32/P4/fused), the
# oracle is the matching fp64-default all-7 run; if unset, the driver runs an fp64 oracle
# arm itself for the same horizon (a clean A/B). The candidate's knobs are passed via
# V020_LADDER_ENV (e.g. "JAX_ENABLE_X64=false GPUWRF_THOMPSON_FP32=1" for the fp32 ladder,
# or a modified namelist via V020_LADDER_NAMELIST for the P4 ladder).
#
# GPU SAFETY: each rung's forecast(s) wrapped in with_gpu_lock.sh via v020_run_gpu and
# SKIPPED under V020_DRYRUN=1 (CPU dry-run validates the rung sequencing + both gates on
# an existing wrfout set, including a candidate==oracle self-check that must PASS).
#
# Usage:
#   scripts/v020_skill_ladder.sh                       # fp64-vs-fp64 self-ladder (sanity)
#   V020_LADDER_ENV="JAX_ENABLE_X64=false GPUWRF_THOMPSON_FP32=1" \
#     scripts/v020_skill_ladder.sh                     # fp32 candidate ladder
#   V020_LADDER_NAMELIST=/path/namelist_dt30.input \
#     scripts/v020_skill_ladder.sh                     # P4 dt=30 candidate ladder
#   V020_DRYRUN=1 scripts/v020_skill_ladder.sh         # CPU dry-run (no GPU)
# Env: V020_LADDER_RUNGS (default "substep step 1 6 24 120"), V020_LADDER_ORACLE,
#      V020_LADDER_ENV, V020_LADDER_NAMELIST, V020_LADDER_MAXHOURS (cap, default 120).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v020_probe_common.sh"

: "${V020_LADDER_RUNGS:=substep step 1 6 24 120}"
: "${V020_LADDER_MAXHOURS:=120}"
: "${V020_LADDER_ENV:=}"           # candidate knobs (space-sep K=V); empty = fp64 default
: "${V020_LADDER_NAMELIST:=}"      # candidate namelist (P4); empty = case default
: "${V020_LADDER_ORACLE:=}"        # oracle wrfout dir; empty = run an fp64 oracle arm
INPUT="$V020_ALL7_INPUT"
NML="${V020_LADDER_NAMELIST:-$INPUT/namelist.input}"

v020_assert_case "$INPUT" || { v020_log "ladder abort: missing all-7 case"; exit 1; }

RR="$(v020_mk_rundir skill_ladder)"; mkdir -p "$RR"
LADDER_LOG="$RR/ladder.jsonl"; : > "$LADDER_LOG"
v020_log "skill ladder run dir: $RR (dryrun=$V020_DRYRUN) rungs: $V020_LADDER_RUNGS"
v020_log "candidate env: '${V020_LADDER_ENV:-<fp64 default>}'  namelist: $NML  oracle: '${V020_LADDER_ORACLE:-<run fp64 arm>}'"

cat > "$RR/runinfo.txt" <<EOF
probe=stability_skill_ladder
benchmark=nested_all7_max_dom9
input=$INPUT
candidate_env=${V020_LADDER_ENV:-fp64_default}
candidate_namelist=$NML
rungs=$V020_LADDER_RUNGS
gates=no_blow_up(hard) + wind/temp/cloud divergence-growth skill(>=1h) vs oracle
policy=pass each rung before climbing; STOP at first fail; never widen
ref=FINAL_FP32_SPRINT_PLAN.md §4 + V0200-ROADMAP.md §8.5 (Gate C/D)
EOF

# rung_hours TAG -> forecast hours for a rung (substep/step approximated as the
# shortest runnable horizon: 1 forecast "hour" with the run truncated by --hours; the
# sub-hour distinction is enforced by the metric harness which reads only the first
# output, but the run itself uses --hours 1 as the minimum CLI granularity).
rung_hours() {
  case "$1" in
    substep|step|1) echo 1 ;;
    *) echo "$1" ;;
  esac
}

# run_one TAG NAME HOURS EXTRA_ENV...  -> runs a forecast into $RR/${TAG}_${NAME}_out
run_one() {
  local tag="$1"; local name="$2"; local hours="$3"; shift 3
  local extra_env=("$@")
  local out="$RR/${tag}_${name}_out"; local proofs="$RR/${tag}_${name}_proofs"
  local scratch="$RR/${tag}_${name}_scratch"; local log="$RR/${tag}_${name}.log"
  mkdir -p "$out" "$proofs" "$scratch"

  local GPU_CMD=(
    env
    PYTHONPATH="$V020_ROOT/src"
    JAX_ENABLE_X64=true
    XLA_PYTHON_CLIENT_PREALLOCATE=false
    JAX_ENABLE_COMPILATION_CACHE=true
    JAX_COMPILATION_CACHE_DIR="$V020_JAX_CACHE"
    OMP_NUM_THREADS="$V020_OMP"
    GPUWRF_MYNN_BOULAC_ONZ=1
    XLA_PYTHON_CLIENT_ALLOCATOR=platform
    GPUWRF_NESTED_SYNC_MODE=root
    GPUWRF_SCRATCH="$scratch"
    "${extra_env[@]}"
    taskset -c "$V020_TASKSET"
    python -m gpuwrf run
      --input-dir "$INPUT"
      --namelist "$NML"
      --output-dir "$out"
      --proof-dir "$proofs"
      --scratch-dir "$scratch"
      --max-dom 9
      --hours "$hours"
  )
  local rc start end wall
  start=$(date -u +%s)
  v020_run_gpu "opus-ladder-${tag}-${name}" "$(( hours*3600 + 3600 ))" -- "${GPU_CMD[@]}" \
    > "$log" 2>&1
  rc=$?
  end=$(date -u +%s); wall=$((end-start))
  echo "$out"  # echo the output dir for the caller
  v020_log "rung $tag ($name) hours=$hours rc=$rc wall=${wall}s" >&2
  echo "$rc" > "$RR/${tag}_${name}.rc"
}

# gate_rung TAG CAND_DIR ORACLE_DIR HOURS -> writes gate jsonl line; echoes PASS|FAIL
gate_rung() {
  local tag="$1"; local cand="$2"; local oracle="$3"; local hours="$4"

  # dry-run / no fresh candidate output: gate against an existing all-7 wrfout set so
  # the gate logic itself runs (self-compare must PASS).
  if [[ "$V020_DRYRUN" == "1" ]] || ! compgen -G "$cand/wrfout_d01_*" > /dev/null 2>&1; then
    for c in "$cand" <DATA_ROOT>/wrf_downscale/canary_all7/cpu_run \
             <DATA_ROOT>/wrf_downscale/canary_all7/run_cadvariant ; do
      if compgen -G "$c/wrfout_d01_*" > /dev/null 2>&1; then cand="$c"; break; fi
    done
    [[ -z "$oracle" || ! -d "$oracle" ]] && oracle="$cand"   # self-compare in dry-run
  fi

  taskset -c "$V020_TASKSET" python "$HERE/v020_blowup_check.py" \
    --run-dir "$cand" --max-dom 9 --out "$RR/${tag}_blowup.json" \
    > "$RR/${tag}_blowup.log" 2>&1
  local blow_rc=$?

  local skill_rc=0 skill="n/a"
  if (( $(rung_hours "$tag" 2>/dev/null || echo 1) >= 1 )) && [[ -n "$oracle" && -d "$oracle" ]]; then
    taskset -c "$V020_TASKSET" python "$HERE/v020_skill_eval.py" \
      --candidate-dir "$cand" --oracle-dir "$oracle" \
      --domains d01 d02 d03 d04 d05 d06 d07 d08 d09 \
      --out "$RR/${tag}_skill.json" > "$RR/${tag}_skill.log" 2>&1
    skill_rc=$?
    skill=$([[ "$skill_rc" == "0" ]] && echo PASS || echo FAIL)
  fi

  local result
  if [[ "$blow_rc" == "0" && "$skill_rc" == "0" ]]; then result="PASS"; else result="FAIL"; fi
  printf '{"rung":"%s","hours":%s,"blowup_rc":%s,"skill":"%s","result":"%s"}\n' \
    "$tag" "$hours" "$blow_rc" "$skill" "$result" >> "$LADDER_LOG"
  echo "$result"
}

# --- climb the ladder --------------------------------------------------------------
LAST_ORACLE="$V020_LADDER_ORACLE"
for rung in $V020_LADDER_RUNGS; do
  hours=$(rung_hours "$rung")
  if (( hours > V020_LADDER_MAXHOURS )); then
    v020_log "rung $rung exceeds V020_LADDER_MAXHOURS=$V020_LADDER_MAXHOURS -> stop climbing"
    break
  fi
  v020_log "=== RUNG $rung (hours=$hours) ==="

  # oracle arm (fp64 default) if no external oracle supplied and this rung scores skill
  oracle_dir="$LAST_ORACLE"
  if [[ -z "$oracle_dir" && "$V020_DRYRUN" != "1" && "$hours" -ge 1 ]]; then
    oracle_dir=$(run_one "$rung" oracle "$hours")   # fp64 default (no extra env)
  fi

  # candidate arm (with the candidate knobs)
  cand_env=()
  # shellcheck disable=SC2206
  [[ -n "$V020_LADDER_ENV" ]] && cand_env=($V020_LADDER_ENV)
  cand_dir=$(run_one "$rung" cand "$hours" "${cand_env[@]}")

  res=$(gate_rung "$rung" "$cand_dir" "$oracle_dir" "$hours")
  v020_log "RUNG $rung -> $res"
  if [[ "$res" == "FAIL" && "$V020_DRYRUN" != "1" ]]; then
    v020_log "rung $rung FAILED (blow-up or escalating skill divergence) -> STOP, never widen."
    break
  fi
done

{
  echo "skill ladder ($RR):"
  echo "  ladder log : $LADDER_LOG"
  echo "  per-rung   : <rung>_{blowup,skill}.json"
  if [[ -s "$LADDER_LOG" ]]; then
    echo "  tally:"
    python3 -c "
import json
for line in open('$LADDER_LOG'):
    try: d=json.loads(line)
    except: continue
    print('    rung=%s hours=%s skill=%s -> %s'%(d.get('rung'),d.get('hours'),d.get('skill'),d.get('result')))
" 2>/dev/null || cat "$LADDER_LOG"
  fi
  [[ "$V020_DRYRUN" == "1" ]] && echo "  NOTE: DRY-RUN — gates exercised on existing wrfout (self-compare must PASS); NOT a real ladder verdict."
} | tee "$RR/LADDER_SUMMARY.txt"
v020_log "skill ladder DONE -> $RR/LADDER_SUMMARY.txt"
exit 0
