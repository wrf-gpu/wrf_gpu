#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# equivalence_switzerland.sh — ONE-COMMAND user test for the Switzerland
# (Gotthard / Central Alps) GPU-vs-CPU-WRF equivalence case (v0.12.0).
#
# A user with python + (conda or venv) + an NVIDIA driver runs ONE command:
#
#     PYTHONPATH=src bash scripts/equivalence_switzerland.sh
#
# and gets the per-field RMSE/bias/max-diff table, the EQUIVALENCE verdict
# against predeclared tolerances, and the GPU-vs-CPU speedup — on a region the
# port was NOT tuned on (proving generalization beyond the Canary Islands).
#
# The user does NOT build WPS or CPU-WRF: the CPU reference is a published,
# checksummed compact wrfout set. The user only runs the GPU port (their GPU)
# and the comparison.
#
# WHAT THIS COMMAND DOES
#   1. Locates/obtains the Switzerland case inputs (wrfinput_d01 + wrfbdy_d01,
#      produced offline by scripts/build_switzerland_case.sh from GFS via WPS +
#      real.exe) and the CPU-WRF reference wrfout set.
#   2. Runs the GPU port forecast STANDALONE from wrfinput/wrfbdy (native-init;
#      NO CPU wrfout consumed) for $HOURS hours -> GPU wrfout.
#   3. Compares GPU vs CPU-WRF field-by-field, all grid points, all hours.
#   4. Prints the verdict + speedup and writes a proof JSON.
#
# Honest framing: numerical/operational equivalence within predeclared tol, NOT
# bitwise-vs-Fortran, NOT a self-compare. See docs/equivalence-switzerland.md.
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

# ── Repo root (this script lives in <repo>/scripts) ────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-${REPO}/src}"
# Operational defaults the GPU port expects (matches the rest of the suite).
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-true}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# ── Configuration (override via env) ───────────────────────────────────────
CASE_ROOT="${CASE_ROOT:-/mnt/data/wrf_gpu_switzerland}"
CASE_INPUTS="${CASE_INPUTS:-${CASE_ROOT}/run_cpu}"          # holds wrfinput/wrfbdy/namelist
GPU_OUT="${GPU_OUT:-${CASE_ROOT}/run_gpu}"                  # GPU wrfout (this run)
GPU_INPUT="${GPU_INPUT:-${CASE_ROOT}/run_gpu_input}"        # clean standalone input dir
SCRATCH="${SCRATCH:-${CASE_ROOT}/scratch}"
DOMAIN="${DOMAIN:-d01}"
HOURS="${HOURS:-24}"
PROOF="${PROOF:-${REPO}/proofs/v0120/equivalence_switzerland.json}"

# CPU reference resolution (precedence): explicit env -> compact set in repo ->
# compact set in CASE_ROOT -> full CPU run dir -> download a published tarball.
CPU_REF="${CPU_REF:-}"
CPU_REF_URL="${CPU_REF_URL:-}"             # optional: published compact tarball URL
CPU_REF_SHA256="${CPU_REF_SHA256:-}"       # optional: its checksum

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*"; }
die(){ echo "[$(ts)] ERROR: $*" >&2; exit 1; }

PY="${PYTHON:-python3}"

log "═══ Switzerland GPU-vs-CPU-WRF equivalence (Gotthard, ${DOMAIN}, ${HOURS} h) ═══"
log "repo=${REPO}  case_root=${CASE_ROOT}"

# ── Resolve the CPU reference directory ────────────────────────────────────
resolve_cpu_ref() {
  local c
  if [[ -n "$CPU_REF" ]]; then
    [[ -d "$CPU_REF" ]] && { echo "$CPU_REF"; return 0; }
    die "CPU_REF=$CPU_REF not a directory"
  fi
  for c in \
      "${REPO}/tests/fixtures/switzerland/cpu_reference_compact" \
      "${CASE_ROOT}/cpu_reference_compact" \
      "${CASE_ROOT}/run_cpu" ; do
    if compgen -G "${c}/wrfout_${DOMAIN}_*" > /dev/null 2>&1; then
      echo "$c"; return 0
    fi
  done
  # Last resort: download a published compact tarball if a URL was given.
  if [[ -n "$CPU_REF_URL" ]]; then
    local dl="${CASE_ROOT}/cpu_reference_compact"
    mkdir -p "$dl"
    log "downloading published CPU reference: $CPU_REF_URL"
    curl -fSL -o "${dl}/cpu_reference_compact.tar.gz" "$CPU_REF_URL" \
      || die "CPU reference download failed"
    if [[ -n "$CPU_REF_SHA256" ]]; then
      echo "${CPU_REF_SHA256}  ${dl}/cpu_reference_compact.tar.gz" | sha256sum -c - \
        || die "CPU reference checksum mismatch"
    fi
    tar -xzf "${dl}/cpu_reference_compact.tar.gz" -C "$dl"
    compgen -G "${dl}/wrfout_${DOMAIN}_*" > /dev/null 2>&1 && { echo "$dl"; return 0; }
    # tarball may extract into a subdir
    local sub; sub="$(find "$dl" -maxdepth 2 -name "wrfout_${DOMAIN}_*" -printf '%h\n' 2>/dev/null | head -1)"
    [[ -n "$sub" ]] && { echo "$sub"; return 0; }
    die "downloaded reference contains no wrfout_${DOMAIN}_*"
  fi
  return 1
}

CPU_REF_DIR="$(resolve_cpu_ref)" || die "No CPU reference found. Provide one of:
  - CPU_REF=<dir with wrfout_${DOMAIN}_*>            (a local CPU-WRF run or compact set)
  - tests/fixtures/switzerland/cpu_reference_compact (shipped compact set)
  - CPU_REF_URL=<published tarball> [CPU_REF_SHA256=<sum>]
  - run scripts/build_switzerland_case.sh + CPU-WRF to produce ${CASE_ROOT}/run_cpu
See docs/equivalence-switzerland.md."
log "CPU reference: ${CPU_REF_DIR}"

# ── Verify case inputs exist (wrfinput + wrfbdy + namelist) ─────────────────
[[ -s "${CASE_INPUTS}/wrfinput_${DOMAIN}" ]] || die "missing ${CASE_INPUTS}/wrfinput_${DOMAIN} — run scripts/build_switzerland_case.sh first"
[[ -s "${CASE_INPUTS}/wrfbdy_d01"        ]] || die "missing ${CASE_INPUTS}/wrfbdy_d01"
[[ -s "${CASE_INPUTS}/namelist.input"    ]] || die "missing ${CASE_INPUTS}/namelist.input"

# ── Build a CLEAN standalone GPU input dir (NO CPU wrfout -> native-init) ───
# detect_init_mode() flips to CPU-WRF replay if it sees >=2 wrfout_<domain>_* in
# the input dir; so the GPU input dir gets ONLY wrfinput/wrfbdy/namelist.
rm -rf "$GPU_INPUT"; mkdir -p "$GPU_INPUT"
ln -sf "${CASE_INPUTS}/wrfinput_${DOMAIN}" "${GPU_INPUT}/wrfinput_${DOMAIN}"
ln -sf "${CASE_INPUTS}/wrfbdy_d01"         "${GPU_INPUT}/wrfbdy_d01"
ln -sf "${CASE_INPUTS}/namelist.input"     "${GPU_INPUT}/namelist.input"

# ── Run the GPU port forecast (standalone native-init) ─────────────────────
rm -rf "$GPU_OUT" "$SCRATCH"; mkdir -p "$GPU_OUT" "$SCRATCH"
log "── GPU forecast (standalone native-init; reads wrfinput/wrfbdy, no CPU wrfout) ──"
GPU_T0=$(date +%s.%N)
"$PY" -m gpuwrf.cli run \
    --input-dir  "$GPU_INPUT" \
    --output-dir "$GPU_OUT" \
    --scratch-dir "$SCRATCH" \
    --domain "$DOMAIN" \
    --hours  "$HOURS" \
    > "${CASE_ROOT}/gpu_run.json" 2> "${CASE_ROOT}/gpu_run.stderr"
GPU_RC=$?
GPU_T1=$(date +%s.%N)
GPU_WALL=$(python3 -c "print(${GPU_T1}-${GPU_T0})")
if [[ $GPU_RC -ne 0 ]]; then
  echo "---- gpu_run.stderr (tail) ----" >&2; tail -25 "${CASE_ROOT}/gpu_run.stderr" >&2
  die "GPU forecast failed (exit ${GPU_RC}); see ${CASE_ROOT}/gpu_run.stderr"
fi
N_GPU=$(compgen -G "${GPU_OUT}/wrfout_${DOMAIN}_*" | wc -l)
log "  GPU wrfout files: ${N_GPU}  wall=${GPU_WALL}s"
[[ $N_GPU -ge 2 ]] || die "GPU produced <2 wrfout files"

# ── A CPU forecast wall, if a sibling timing file exists, else leave to JSON ─
CPU_WALL_ARG=()
if [[ -f "${CPU_REF_DIR}/cpu_wall_seconds.txt" ]]; then
  CPU_WALL=$(cat "${CPU_REF_DIR}/cpu_wall_seconds.txt")
  CPU_WALL_ARG=(--cpu-wall-s "$CPU_WALL")
elif [[ -f "${CASE_ROOT}/run_cpu/cpu_wall_seconds.txt" ]]; then
  CPU_WALL=$(cat "${CASE_ROOT}/run_cpu/cpu_wall_seconds.txt")
  CPU_WALL_ARG=(--cpu-wall-s "$CPU_WALL")
fi

# ── Compare GPU vs CPU reference ───────────────────────────────────────────
log "── Comparison (GPU vs CPU-WRF, predeclared tolerances) ──"
"$PY" "${SCRIPT_DIR}/equivalence_switzerland_compare.py" \
    --gpu-dir "$GPU_OUT" \
    --cpu-dir "$CPU_REF_DIR" \
    --domain  "$DOMAIN" \
    --hours   "$HOURS" \
    --gpu-wall-s "$GPU_WALL" \
    "${CPU_WALL_ARG[@]}" \
    --out "$PROOF"
CMP_RC=$?
log "═══ done (proof: ${PROOF}) ═══"
exit $CMP_RC
