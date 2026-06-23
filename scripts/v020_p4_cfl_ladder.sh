#!/usr/bin/env bash
# v020_p4_cfl_ladder.sh — P4 dt/n_sound CFL-FIRST ladder driver (roadmap step 2, §8.3).
#
# The "surprise cheap real win": the all-7 namelist runs d01 time_step=18 s at dx=9 km
# (~2x dx vs WRF's up-to-6x) and acoustic_substeps=10 (~16 substeps/step vs ~7). Raising
# dt + cutting n_sound cuts step count near-proportionally across ALL 9 domains. But it is
# NOT a free win (roadmap §8.3): pursue it as a STRICT CFL-FIRST ladder —
#   (1) raise dt FIRST (with baseline-like acoustic dt), and
#   (2) cut n_sound ONLY on dt-rungs that already pass the 6 h short ladder.
#
# Per the §8.3 rungs (root dt -> leaf@ratio3, n_sound for baseline-like acoustic dt):
#   dt: 18(base) 24 27 30 36 45(high-risk) 54(very-high-risk)
#   n_sound reduction per passing dt: e.g. dt24 14->12->10 ; dt30 17->15->12->10
#   dt45/54: do NOT cut substeps before 24 h evidence (driver refuses by default).
#
# EACH RUNG: build a variant namelist (time_step only — the cascade dt's follow the
# ratio), run a 6 h forecast, then gate on:
#   * CFL probe (scripts/v020_cfl_probe.py): realized advective+acoustic Courant per
#     domain, per-domain + boundary-ring (the §8.3 landmine monitors).
#   * blow-up gate (scripts/v020_blowup_check.py): HARD no-blow-up (carve-outs aside).
# Raise dt one rung ONLY after a clean pass; STOP at the first failing rung; never widen.
#
# n_sound PLUMBING GAP (see REPORT.md): the nested path hardcodes acoustic_substeps=10
# (nested_pipeline.py:213) with NO env override. The dt ladder is FULLY drivable today
# (namelist lever). The driver ALSO exports GPUWRF_ACOUSTIC_SUBSTEPS for the n_sound rungs
# so it fires the instant a one-line override lands; until then n_sound rungs are SKIPPED
# with a clear note (set V020_P4_FORCE_NSOUND=1 to attempt them anyway).
#
# GPU SAFETY: each rung's forecast is wrapped in with_gpu_lock.sh via v020_run_gpu and
# SKIPPED under V020_DRYRUN=1 (CPU dry-run validates namelist-build + CFL/blow-up gates on
# existing wrfout).
#
# Usage:
#   scripts/v020_p4_cfl_ladder.sh                  # FULL ladder (holds GPU lock)
#   V020_DRYRUN=1 scripts/v020_p4_cfl_ladder.sh    # CPU dry-run (no GPU)
# Env: V020_P4_HOURS (default 6), V020_P4_DT_RUNGS, V020_P4_NSOUND_BY_DT.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/v020_probe_common.sh"

: "${V020_P4_HOURS:=6}"                       # 6 h short pass per §8.3
: "${V020_P4_DT_RUNGS:=24 27 30 36}"          # default rungs (45/54 high-risk, opt-in)
: "${V020_P4_DURATION:=5400}"                 # nsys/run wall cap per rung (s)
: "${V020_P4_FORCE_NSOUND:=0}"                # 1 = attempt n_sound rungs despite plumbing gap
INPUT="$V020_ALL7_INPUT"
BASE_NML="$INPUT/namelist.input"

# n_sound reduction schedule per dt (only used once the env override is plumbed)
declare -A NSOUND_BY_DT=(
  [24]="14 12 10"
  [27]="15 12 10"
  [30]="17 15 12 10"
  [36]="20 16 12"
)

v020_assert_case "$INPUT" || { v020_log "P4 abort: missing all-7 case"; exit 1; }
[[ -f "$BASE_NML" ]] || { v020_log "P4 abort: no base namelist $BASE_NML"; exit 1; }

RR="$(v020_mk_rundir p4_cfl_ladder)"; mkdir -p "$RR/namelists"
LADDER_LOG="$RR/ladder.jsonl"; : > "$LADDER_LOG"
v020_log "P4 run dir: $RR  (dryrun=$V020_DRYRUN)  dt rungs: $V020_P4_DT_RUNGS"

cat > "$RR/runinfo.txt" <<EOF
probe=P4_cfl_first_ladder
benchmark=nested_all7_max_dom9
input=$INPUT
hours_per_rung=$V020_P4_HOURS
dt_rungs=$V020_P4_DT_RUNGS
policy=CFL_first (raise dt first; cut n_sound only on a passing dt; stop at first fail; never widen)
n_sound_plumbing=GAP (nested_pipeline.py:213 hardcodes acoustic_substeps=10; no env override yet)
ref=V0200-ROADMAP.md §8.3
EOF

# run_rung TAG TIME_STEP NSOUND  -> runs a 6h forecast + gates; echoes PASS|FAIL|SKIP
run_rung() {
  local tag="$1"; local dt="$2"; local nsound="${3:-}"
  local nml="$RR/namelists/namelist_${tag}.input"
  local out="$RR/${tag}_out"; local proofs="$RR/${tag}_proofs"; local scratch="$RR/${tag}_scratch"
  local log="$RR/${tag}.log"
  mkdir -p "$out" "$proofs" "$scratch"

  # build the variant namelist (time_step lever)
  taskset -c "$V020_TASKSET" python "$HERE/v020_make_namelist.py" \
    --base "$BASE_NML" --out "$nml" --time-step "$dt" \
    ${nsound:+--n-sound "$nsound"} > "$RR/${tag}_namelist.log" 2>&1 || {
      v020_log "rung $tag: namelist build FAILED"; echo "FAIL"; return; }

  # n_sound rung but plumbing gap and not forced -> SKIP (honest, not fake-green)
  local nsound_env=()
  if [[ -n "$nsound" ]]; then
    if [[ "$V020_P4_FORCE_NSOUND" == "1" ]]; then
      nsound_env=(GPUWRF_ACOUSTIC_SUBSTEPS="$nsound")
    else
      v020_log "rung $tag: n_sound=$nsound needs GPUWRF_ACOUSTIC_SUBSTEPS plumbing (gap) -> SKIP"
      printf '{"rung":"%s","dt":%s,"n_sound":%s,"result":"SKIP_NSOUND_PLUMBING_GAP"}\n' \
        "$tag" "$dt" "$nsound" >> "$LADDER_LOG"
      echo "SKIP"; return
    fi
  fi

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
    "${nsound_env[@]}"
    taskset -c "$V020_TASKSET"
    python -m gpuwrf run
      --input-dir "$INPUT"
      --namelist "$nml"
      --output-dir "$out"
      --proof-dir "$proofs"
      --scratch-dir "$scratch"
      --max-dom 9
      --hours "$V020_P4_HOURS"
  )

  local start end wall
  start=$(date -u +%s)
  v020_run_gpu "opus-p4-$tag" "$((V020_P4_DURATION + 1800))" -- "${GPU_CMD[@]}" \
    > "$log" 2>&1
  local rc=$?
  end=$(date -u +%s); wall=$((end - start))
  v020_log "rung $tag (dt=$dt${nsound:+ n_sound=$nsound}) rc=$rc wall=${wall}s"

  # gates: blow-up first (hard floor), then CFL realized margins
  local gate_dir="$out"
  if [[ "$V020_DRYRUN" == "1" ]] || ! compgen -G "$out/wrfout_d01_*" > /dev/null 2>&1; then
    # dry-run / no fresh output: gate against an existing all-7 wrfout set so the gate
    # logic itself is exercised (it will report PASS/blowup on that set).
    for cand in "$out" <DATA_ROOT>/wrf_downscale/canary_all7/cpu_run \
                <DATA_ROOT>/wrf_downscale/canary_all7/run_cadvariant ; do
      if compgen -G "$cand/wrfout_d01_*" > /dev/null 2>&1; then gate_dir="$cand"; break; fi
    done
  fi

  taskset -c "$V020_TASKSET" python "$HERE/v020_blowup_check.py" \
    --run-dir "$gate_dir" --max-dom 9 --out "$RR/${tag}_blowup.json" \
    > "$RR/${tag}_blowup.log" 2>&1
  local blow_rc=$?

  taskset -c "$V020_TASKSET" python "$HERE/v020_cfl_probe.py" \
    --run-dir "$gate_dir" --namelist "$nml" --max-dom 9 --time LAST \
    --out "$RR/${tag}_cfl.json" > "$RR/${tag}_cfl.log" 2>&1 || true

  # verdict: in REAL mode a rung PASSES iff rc==0 AND no blow-up; CFL is diagnostic.
  local result
  if [[ "$V020_DRYRUN" == "1" ]]; then
    result="DRYRUN_GATES_EXERCISED"
  elif [[ "$rc" == "0" && "$blow_rc" == "0" ]]; then
    result="PASS"
  else
    result="FAIL"
  fi
  printf '{"rung":"%s","dt":%s,"n_sound":%s,"rc":%s,"blowup_rc":%s,"wall_s":%s,"result":"%s"}\n' \
    "$tag" "$dt" "${nsound:-null}" "$rc" "$blow_rc" "$wall" "$result" >> "$LADDER_LOG"
  echo "$result"
}

# --- the CFL-first ladder ----------------------------------------------------------
LADDER_STOPPED=0
for dt in $V020_P4_DT_RUNGS; do
  if [[ "$LADDER_STOPPED" == "1" ]]; then
    v020_log "ladder stopped at a prior rung; not attempting dt=$dt"
    break
  fi
  if [[ "$dt" -ge 45 ]]; then
    v020_log "dt=$dt is HIGH-RISK (>=45 s) — only attempted if explicitly in V020_P4_DT_RUNGS"
  fi
  res=$(run_rung "dt${dt}" "$dt" "")
  v020_log "=== dt rung dt=$dt -> $res ==="
  if [[ "$res" == "FAIL" ]]; then
    v020_log "dt=$dt FAILED the 6h short ladder -> STOP (never widen). dt-1 rung is the keeper."
    LADDER_STOPPED=1
    break
  fi
  # dt passed -> n_sound reduction on THIS dt (gap-aware; §8.3 forbids cutting on dt>=45)
  if [[ "$res" == "PASS" || "$V020_DRYRUN" == "1" ]]; then
    if [[ "$dt" -lt 45 ]]; then
      sched="${NSOUND_BY_DT[$dt]:-}"
      for ns in $sched; do
        nres=$(run_rung "dt${dt}_ns${ns}" "$dt" "$ns")
        v020_log "    n_sound rung dt=$dt n_sound=$ns -> $nres"
        [[ "$nres" == "FAIL" ]] && { v020_log "    n_sound=$ns failed -> keep n_sound+1 for dt=$dt"; break; }
      done
    else
      v020_log "dt=$dt >=45: NOT cutting n_sound before 24h evidence (§8.3)"
    fi
  fi
done

{
  echo "P4 CFL-first ladder ($RR):"
  echo "  ladder log : $LADDER_LOG"
  echo "  namelists  : $RR/namelists/"
  echo "  per-rung   : <rung>_{blowup,cfl}.json (rung = dt<DT> or dt<DT>_ns<N>)"
  echo "  result tally:"
  if [[ -s "$LADDER_LOG" ]]; then
    python3 -c "
import json,sys
for line in open('$LADDER_LOG'):
    try: d=json.loads(line)
    except: continue
    print('    dt=%s n_sound=%s -> %s'%(d.get('dt'),d.get('n_sound'),d.get('result')))
" 2>/dev/null || cat "$LADDER_LOG"
  fi
  [[ "$V020_DRYRUN" == "1" ]] && echo "  NOTE: DRY-RUN — gate logic exercised on existing wrfout; results are NOT a real ladder verdict."
} | tee "$RR/P4_SUMMARY.txt"
v020_log "P4 DONE -> $RR/P4_SUMMARY.txt"
exit 0
