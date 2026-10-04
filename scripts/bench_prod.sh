#!/usr/bin/env bash
# bench_prod.sh — one-command PROD benchmark (gate S1). See scripts/BENCH.md.
#   scripts/bench_prod.sh [--hours 3] [--cold] [--arms N] [--nsys] [--tag T] [--src PATH] [--rev REF]
# Runs the S0 nested harness (real `gpuwrf.cli run`, d01+d02) on the PROD case; EACH ARM takes the GPU lock
# itself (so short probes can slip in between arms). Do NOT pre-wrap this script in with_gpu_lock.
# pinned to BENCH_CPUS (default 10,11,26,27 — L4 GPU-holder cores; lane CPU work on 8,9,12,13,24,25,28,29; arms nice 10, CPU nice 19,
# OMP=2), output under $BENCH_ROOT/<tag>/ (disk, never <DATA_ROOT>, never /tmp).
#   --src PATH  repo to snapshot and measure (default: the checkout containing this script)
#   --rev REF   commit to measure in --src (default HEAD): a detached worktree snapshot is created/reused at
#               $BENCH_SNAP_ROOT/<rev12> and imports come ONLY from it; rev + src/gpuwrf tree are recorded.
#   --cold    first arm uses a fresh EMPTY cache (fills it); later arms are warm on that cache
#   --arms N  number of warm arms after the optional cold arm (default 2: cold + two warm = A/A gate)
#   --nsys    one extra nsys arm (segment 2, warm) -> kernel and copy counts in bench.json
#   --extra-hours H  one extra WARM arm of H hours (named extra${H}h) on the same cache; its wrfout is
#               kept for downstream scoring (not part of the A/A headline)
#   --warm1-env 'K=V ...'  extra env for warm arm 1 ONLY (word-split, added to the arm env)
#   --warm2-env 'K=V ...'  extra env for warm arm 2 ONLY (e.g. a VRAM A/B); word-split, added to the arm env
#   --arm-env N:'K=V ...'  env for warm arm N (1-based; repeatable; overrides --warm1/2-env). e.g. --arm-env 3:'A=1 B=2'
#   --single-lock  hold ONE GPU lock (+ ONE quiet window) across ALL arms (arms back-to-back, no lane
#               can slip in between); per-arm timeout starts after the lock is held (excludes queue wait)
#   --quiet-label '<label>'  the --extra-hours arm creates $BENCH_QUIET_FILE (/tmp/wrf_gpu2_quiet,
#               '<label> <UTC>') once it holds the lock and removes it on exit (trap) -- tells other lanes
#               to avoid heavy CPU work during the timed arm.
#   --census-arm  one extra WARM arm with GPUWRF_CENSUS=1 (named 'census'); its manifest/counts are attached
#               to bench.json (F2 counts). Timing arms ALWAYS run census OFF (production default).
#   --xla-flags '<flags>'  XLA_FLAGS exported for every arm AFTER s0_env.sh (which unsets it); recorded in
#               bench.json + the receipt env. e.g. '--xla_gpu_autotune_level=0'.
#   --print-src  print resolved repo/rev/snapshot/tree/cache + DIRTY and exit (no GPU/lock, no run)
# Cache: one private dir per cache identity (src tree), $BENCH_ROOT/cache/<src_tree12>; path printed.
# Guard: missing launch paths, a dirty src/gpuwrf, or a snapshot identity that moves mid-run all abort.
# GPU-arm wall caps (AGENTS.md): warm/nsys arms 1800 s, cold arm 3000 s (cold compile exempt; raise
# ARM_TIMEOUT_COLD for a first cold-setup arm on a new tree).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_WT="$(dirname "$HERE")"
SPRINT=".agent/sprints/2026-09-23-v0250-endgame-frontrunner/tools"
BENCH_CPUS="${BENCH_CPUS:-10,11,26,27}"   # L4 (alisios_mgr): GPU holder exclusive 10,11,26,27
ARM_TIMEOUT_WARM="${ARM_TIMEOUT_WARM:-1800}"; ARM_TIMEOUT_COLD="${ARM_TIMEOUT_COLD:-3000}"
HOURS=3; COLD=0; ARMS=2; NSYS=0; TAG=""; SRC_REPO="$REPO_WT"; REV="HEAD"; PRINT_SRC=0; EXTRA_HOURS=0; CENSUS_ARM=0; XLA_FLAGS_OPT=""; QUIET_LABEL=""; WARM1_ENV=""; WARM2_ENV=""; SINGLE_LOCK=0; QUEUE_BUDGET="${QUEUE_BUDGET:-3600}"
declare -A ARM_ENV=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hours) HOURS="$2"; shift 2;; --cold) COLD=1; shift;; --arms) ARMS="$2"; shift 2;;
    --nsys) NSYS=1; shift;; --tag) TAG="$2"; shift 2;; --src) SRC_REPO="$2"; shift 2;;
    --rev) REV="$2"; shift 2;; --extra-hours) EXTRA_HOURS="$2"; shift 2;; --census-arm) CENSUS_ARM=1; shift;;
    --xla-flags) XLA_FLAGS_OPT="$2"; shift 2;; --quiet-label) QUIET_LABEL="$2"; shift 2;;
    --warm2-env) WARM2_ENV="$2"; shift 2;; --warm1-env) WARM1_ENV="$2"; shift 2;; --single-lock) SINGLE_LOCK=1; shift;;
    --arm-env) ARM_ENV["${2%%:*}"]="${2#*:}"; shift 2;; --print-src) PRINT_SRC=1; shift;;
    -h|--help) sed -n 2,33p "$0"; exit 0;; *) echo "bench_prod: unknown arg $1" >&2; exit 2;;
  esac
done
[[ -n "$TAG" ]] || TAG="bench_$(date -u +%Y%m%dT%H%M%SZ)"
SRC_REPO="$(cd "$SRC_REPO" 2>/dev/null && pwd)" || { echo "bench_prod: --src '$SRC_REPO' is not a directory" >&2; exit 4; }
# A --rev snapshot measures a COMMIT, so uncommitted src/gpuwrf edits are never measured -> refuse them.
REPO_DIRTY="$(git -C "$SRC_REPO" status --porcelain --untracked-files=all -- src/gpuwrf 2>/dev/null)"
# Immutable measurement: snapshot scripts/bench/<rev> into a detached worktree and import ONLY from it.
SRC_REV="$(git -C "$SRC_REPO" rev-parse --verify "${REV}^{commit}" 2>/dev/null)" \
  || { echo "bench_prod: --rev '$REV' is not a commit in $SRC_REPO" >&2; exit 4; }
SNAP_ROOT="${BENCH_SNAP_ROOT:-<USER_HOME>/wrf_gpu2_lanes/bench/src}"
SNAP="$SNAP_ROOT/${SRC_REV:0:12}"
if [[ -e "$SNAP/.git" ]]; then
  [[ "$(git -C "$SNAP" rev-parse HEAD 2>/dev/null)" == "$SRC_REV" ]] \
    || { echo "bench_prod: snapshot $SNAP exists at a different rev; remove it or pick another --rev" >&2; exit 4; }
else
  mkdir -p "$SNAP_ROOT"
  git -C "$SRC_REPO" worktree prune
  git -C "$SRC_REPO" worktree add --detach "$SNAP" "$SRC_REV" >/dev/null 2>&1 \
    || { echo "bench_prod: git worktree add failed for $SNAP @ $SRC_REV" >&2; exit 4; }
fi
SRC="$SNAP"                     # everything below imports from the immutable snapshot, never $SRC_REPO
SRC_TREE="$(git -C "$SRC" rev-parse HEAD:src/gpuwrf 2>/dev/null)" \
  || { echo "bench_prod: snapshot $SRC has no src/gpuwrf" >&2; exit 4; }
TOOLS="$SRC/$SPRINT"                          # exported as T (points at <snapshot>/.agent); not sourced here
HARNESS="$HERE/bench/s0_nested_harness.py"   # bench-owned copy; REPO comes from BENCH_SRC_REPO below
CENSUS="$HERE/bench/s0_nsys_census.py"       # bench-owned census (per-class kernel/copy counts)
ENVFILE="$REPO_WT/$SPRINT/s0_env.sh"         # guard/run helpers ship with this script's worktree
export BENCH_ROOT="${BENCH_ROOT:-<USER_HOME>/wrf_gpu2_lanes/bench}"   # disk, never /tmp (RAM tmpfs)
CACHE="$BENCH_ROOT/cache/${SRC_TREE:0:12}"
preflight() {  # CPU-only checks; run before any GPU work and before --print-src reports success
  local f
  for f in "$HARNESS" "$ENVFILE"; do
    [[ -f "$f" ]] || { echo "bench_prod: missing launch path: $f" >&2; return 4; }
  done
  [[ $NSYS == 0 || -f "$CENSUS" ]] || { echo "bench_prod: missing census: $CENSUS" >&2; return 4; }
  [[ -z "$REPO_DIRTY" ]] || { echo "bench_prod: --src $SRC_REPO has a dirty src/gpuwrf; --rev measures a COMMIT, so commit/stash it:" >&2; echo "$REPO_DIRTY" >&2; return 5; }
  return 0
}
verify_src() {  # the snapshot must not move/change between arms
  local head now dirty
  head="$(git -C "$SRC" rev-parse HEAD 2>/dev/null)"
  now="$(git -C "$SRC" rev-parse HEAD:src/gpuwrf 2>/dev/null)"
  dirty="$(git -C "$SRC" status --porcelain --untracked-files=all -- src/gpuwrf 2>/dev/null)"
  [[ "$head" == "$SRC_REV" && "$now" == "$SRC_TREE" && -z "$dirty" ]] || {
    echo "[bench] abort: snapshot changed (head='$head' want='$SRC_REV'; tree='$now' want='$SRC_TREE'; dirty='$dirty')" >&2
    return 6; }
}
if [[ $PRINT_SRC == 1 ]]; then
  printf 'SRC_REPO=%s\nSRC_REV=%s\nSNAP=%s\nSRC_TREE=%s\nTOOLS=%s\nHARNESS=%s\nCENSUS=%s\nCACHE=%s\nPYTHONPATH=%s\nDIRTY=%s\n' \
    "$SRC_REPO" "$SRC_REV" "$SNAP" "$SRC_TREE" "$TOOLS" "$HARNESS" "$CENSUS" "$CACHE" "$SRC/src" \
    "$([[ -n "$REPO_DIRTY" ]] && echo 1 || echo 0)"
  preflight || exit $?
  exit 0
fi
preflight || exit $?
export S0_ROOT="$BENCH_ROOT/$TAG" S0_CPUS="$BENCH_CPUS" S0_MIN_FREE_MIB="${S0_MIN_FREE_MIB:-26000}"
# Each arm takes the GPU lock itself (short probes can slip in between arms). This script must NOT be
# pre-wrapped in with_gpu_lock: a nested flock on the same file would deadlock (with_gpu_lock re-flocks
# even when GPUWRF_GPU_LOCK_HELD=1). Refuse loudly instead of hanging.
if [[ "${GPUWRF_GPU_LOCK_HELD:-0}" == 1 && $SINGLE_LOCK != 1 ]]; then
  echo "bench_prod: do not wrap this script in with_gpu_lock; each ARM acquires the lock individually (or use --single-lock)" >&2; exit 2
fi
# --single-lock: this script takes the lock ONCE and holds it across all arms (they run back-to-back under it).
if [[ $SINGLE_LOCK == 1 && "${GPUWRF_GPU_LOCK_HELD:-0}" != 1 ]]; then
  _a=(--hours "$HOURS" --arms "$ARMS" --tag "$TAG" --src "$SRC_REPO" --rev "$SRC_REV" --extra-hours "$EXTRA_HOURS"
      --xla-flags "$XLA_FLAGS_OPT" --warm1-env "$WARM1_ENV" --warm2-env "$WARM2_ENV" --quiet-label "$QUIET_LABEL" --single-lock)
  [[ $COLD == 1 ]] && _a+=(--cold); [[ $NSYS == 1 ]] && _a+=(--nsys); [[ $CENSUS_ARM == 1 ]] && _a+=(--census-arm)
  exec "$HERE/with_gpu_lock.sh" --label bench -- taskset -c "$BENCH_CPUS" nice -n 10 \
    env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" bash "$0" "${_a[@]}"
fi
source "$ENVFILE"          # helpers only (s0_run/s0_disk_check); no guard here, arms guard under the lock
# Census follows the measured snapshot's API/schema; legacy snapshots stay off.
CENSUS_REPORT="$SRC/src/gpuwrf/diagnostics/census_report.py"
if [[ -f "$CENSUS_REPORT" ]]; then export GPUWRF_CENSUS="${GPUWRF_CENSUS:-1}"; fi
export GPUWRF_JAX_CACHE_DIR="$CACHE"
# single-lock: one quiet window spanning every arm (per-arm quiet would gap between arms)
ARMQ="$QUIET_LABEL"
if [[ $SINGLE_LOCK == 1 ]]; then
  ARMQ=""
  QF="${BENCH_QUIET_FILE:-/tmp/wrf_gpu2_quiet}"
  if [[ -n "$QUIET_LABEL" ]]; then printf '%s %s\n' "$QUIET_LABEL" "$(date -u +%FT%TZ)" > "$QF"; trap 'rm -f "$QF"' EXIT TERM INT; fi
fi
mkdir -p "$S/tmp"
echo "[bench] tag=$TAG repo=$SRC_REPO rev=$SRC_REV snap=$SRC src_tree=$SRC_TREE cache=$CACHE cpus=$BENCH_CPUS" | tee -a "$S/session.log"
if [[ $COLD == 1 ]]; then
  [[ -z "$(ls -A "$CACHE" 2>/dev/null)" ]] || { echo "[bench] refuse --cold: cache $CACHE not empty (use another tree or delete it by exact path)"; exit 3; }
  mkdir -p "$CACHE"
else
  [[ -n "$(ls -A "$CACHE" 2>/dev/null)" ]] || { echo "[bench] refuse warm: cache $CACHE empty (run with --cold first)"; exit 3; }
fi
run_arm() {  # $1 name, $2 arm-timeout s, $3 census(0|1), $4 quiet-label, $5 extra-env, then '--' + FULL command
  local arm=$1 tmo=$2 cen=$3 quiet=$4 extra=$5; shift 5
  [[ "${1:-}" == "--" ]] && shift
  verify_src || exit $?
  local wrap=() budget="$QUEUE_BUDGET"
  [[ $SINGLE_LOCK == 1 ]] || wrap=("$HERE/with_gpu_lock.sh" --label bench --)   # single-lock: lock already held
  [[ $SINGLE_LOCK == 1 ]] && budget=120                                       # no queue wait inside the lock
  # outer timeout = arm timeout + queue budget (safety); the REAL arm timeout is applied inside the lock (E20)
  s0_run "$S/$arm" "$((tmo + budget))" "${wrap[@]}" taskset -c "$BENCH_CPUS" nice -n 10 \
    env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" GPUWRF_CENSUS="$cen" $extra BENCH_ARM_DIR="$S/$arm" \
    bash "$HERE/bench/run_arm.sh" "$ENVFILE" "$SRC" "$TOOLS" "$XLA_FLAGS_OPT" "$quiet" "$tmo" -- "$@" \
    || { echo "[bench] arm $arm failed rc=$?"; exit 1; }
  s0_disk_check
}
ARMLIST=()
if [[ $COLD == 1 ]]; then
  run_arm cold "$ARM_TIMEOUT_COLD" 0 "" "" -- "$PY" "$HARNESS" --input-dir "$CASE" --out "$S/cold/run" --hours "$HOURS" --tag "${TAG}_cold"
  ARMLIST+=(cold)
fi
for ((i = 1; i <= ARMS; i++)); do
  _e=""; case $i in 1) _e="$WARM1_ENV";; 2) _e="$WARM2_ENV";; esac
  [[ -n "${ARM_ENV[$i]:-}" ]] && _e="${ARM_ENV[$i]}"
  run_arm "warm$i" "$ARM_TIMEOUT_WARM" 0 "$ARMQ" "$_e" -- "$PY" "$HARNESS" --input-dir "$CASE" --out "$S/warm$i/run" --hours "$HOURS" --tag "${TAG}_warm$i"
  ARMLIST+=("warm$i")
done
if (( EXTRA_HOURS > 0 )); then
  run_arm "extra${EXTRA_HOURS}h" "$ARM_TIMEOUT_WARM" 0 "$ARMQ" "" -- "$PY" "$HARNESS" --input-dir "$CASE" --out "$S/extra${EXTRA_HOURS}h/run" --hours "$EXTRA_HOURS" --tag "${TAG}_extra${EXTRA_HOURS}h"
  ARMLIST+=("extra${EXTRA_HOURS}h")
fi
if [[ $CENSUS_ARM == 1 ]]; then
  run_arm census "$ARM_TIMEOUT_WARM" 1 "" "" -- "$PY" "$HARNESS" --input-dir "$CASE" --out "$S/census/run" --hours "$HOURS" --tag "${TAG}_census"
  ARMLIST+=(census)
fi
if [[ $NSYS == 1 ]]; then
  run_arm nsys "$ARM_TIMEOUT_WARM" 0 "" "" -- nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o "$S/nsys/prof" \
    "$PY" "$HARNESS" --input-dir "$CASE" --out "$S/nsys/run" --hours 2 --tag "${TAG}_nsys" --profile-segment 2 --no-vram
  ARMLIST+=(nsys)
fi
SUMARGS=()
for a in "${ARMLIST[@]}"; do SUMARGS+=("$a=$S/$a/run/receipt.json"); done
[[ $NSYS == 1 ]] && SUMARGS+=(--nsys "$S/nsys/prof.nsys-rep")
verify_src || exit $?
nice -n 19 "$PY" "$HERE/bench_summary.py" --out "$S/bench.json" --tag "$TAG" --cache "$CACHE" \
  --src "$SRC_REPO" --src-rev "$SRC_REV" --src-tree "$SRC_TREE" "--xla-flags=$XLA_FLAGS_OPT" "${SUMARGS[@]}"
SUMRC=$?
if [[ $SUMRC == 0 && -f "$CENSUS_REPORT" ]]; then
  nice -n 19 "$PY" "$CENSUS_REPORT" --attach-bench "$S/bench.json" || SUMRC=$?
fi
if [[ $SUMRC == 0 ]]; then
  echo "[bench] done -> $S/bench.json"
else
  echo "[bench] summary FAILED rc=$SUMRC (no valid headline in $S/bench.json)" >&2
fi
exit "$SUMRC"
