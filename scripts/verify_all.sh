#!/usr/bin/env bash
# v0.1.0 master verification suite.
#
# Runs every scripts/verify/<row>.sh in order and writes proofs/PROOF_TABLE.md with
# PASS/FAIL + actual numbers + the git commit SHA. CPU rows run first; GPU rows run
# STRICTLY SERIAL (one at a time, NEVER in parallel -- they share one GPU). A row
# that FAILS records FAIL and the suite continues (it does not abort).
#
# GPU rows are NOT executed unless VERIFY_RUN_GPU=1 is exported. By default they are
# recorded as "[GPU: manager-sequenced]" (GPU_DEFERRED) so this suite is fully
# runnable on a CPU-only box and the manager sequences the real GPU runs.
#
# Usage:
#   bash scripts/verify_all.sh                 # CPU rows + GPU-deferred placeholders
#   VERIFY_RUN_GPU=1 bash scripts/verify_all.sh   # also execute GPU rows (idle GPU!)
#   bash scripts/verify_all.sh 20260531T120000Z   # optional timestamp tag for outputs
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
VERIFY_DIR="${HERE}/verify"
cd "${REPO_ROOT}"

TS="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
SHA="$(git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
SHORT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo UNKNOWN)"
BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo UNKNOWN)"

OUT_DIR="${REPO_ROOT}/proofs"
mkdir -p "${OUT_DIR}"
TABLE="${OUT_DIR}/PROOF_TABLE.md"
LOG_DIR="${OUT_DIR}/verify_logs/${TS}"
mkdir -p "${LOG_DIR}"

# Row registry: "id|script|kind|claim"
# kind = CPU (executed here) | GPU (serial; deferred unless VERIFY_RUN_GPU=1)
CPU_ROWS=(
  "6|tost.sh|CPU|Equivalence (paired TOST) machinery, T2/U10/V10 (CPU self-test; GPU achievable-N + seasonal limitation manager-sequenced)"
  "9|performance.sh|CPU|Performance: provenance-backed ~5-8x vs 28-rank CPU-WRF d02 (floor 3.2x, d02-only)"
  "10|precip.sh|CPU|Precipitation: physically correct & functional (honest characterization, not parity)"
)
GPU_ROWS=(
  "1|idealized_warmbubble.sh|GPU|Dycore: Skamarock warm bubble matches the benchmark reference"
  "2|idealized_straka.sh|GPU|Dycore: Straka density current matches the benchmark reference"
  "3|savepoint_parity.sh|GPU|Operator parity vs pristine WRF v4 savepoints"
  "4|d02_validation.sh|GPU|Canary 3 km (d02): finite/stable, no blow-up, near-CPU-WRF, beats persistence on winds"
  "5|d03_validation.sh|GPU|Canary 1 km (d03): bounded gate (BLOCKED pending HFX fix #56)"
  "7|conservation.sh|GPU|Conservation: dry-mass/water/energy bounded; guards not load-bearing"
  "8|repeatability.sh|GPU|Reproducibility: deterministic re-run + restart-continuity"
  "11|device_residency.sh|GPU|Device residency: zero host<->device transfer inside the timestep loop"
)

declare -A RESULT_STATUS
declare -A RESULT_NOTE
declare -A RESULT_KIND
declare -A RESULT_CLAIM

run_row() {
  local id="$1" script="$2" kind="$3" claim="$4"
  local path="${VERIFY_DIR}/${script}"
  local log="${LOG_DIR}/row${id}_${script%.sh}.log"
  RESULT_KIND["$id"]="$kind"
  RESULT_CLAIM["$id"]="$claim"
  echo "============================================================"
  echo "ROW ${id} [${kind}] :: ${script}"
  echo "------------------------------------------------------------"
  if [ ! -f "${path}" ]; then
    RESULT_STATUS["$id"]="MISSING"
    RESULT_NOTE["$id"]="verify script not found: ${path}"
    echo "MISSING: ${path}"
    return
  fi
  # Run the row; capture combined output to a log and tee to the console.
  bash "${path}" >"${log}" 2>&1
  local rc=$?
  cat "${log}"
  # Parse the canonical RESULT line the row emits.
  local result_line
  result_line="$(grep -E '^RESULT: ' "${log}" | tail -1 || true)"
  local status note
  if [ -n "${result_line}" ]; then
    status="$(echo "${result_line}" | awk '{print $3}')"
    note="$(echo "${result_line}" | cut -d' ' -f4-)"
  else
    status=""
    note="(no RESULT line emitted)"
  fi
  # Reconcile status with exit code: a non-zero exit is always a FAIL unless the
  # row explicitly declared GPU_DEFERRED (which exits 0 by design).
  if [ "${status}" = "GPU_DEFERRED" ]; then
    RESULT_STATUS["$id"]="GPU_DEFERRED"
  elif [ ${rc} -eq 0 ] && [ "${status}" = "PASS" ]; then
    RESULT_STATUS["$id"]="PASS"
  else
    RESULT_STATUS["$id"]="FAIL"
    if [ -z "${status}" ]; then note="row exited ${rc} with no RESULT line"; fi
  fi
  RESULT_NOTE["$id"]="${note}"
  echo "-> row ${id}: ${RESULT_STATUS[$id]}"
}

echo "############################################################"
echo "# v0.1.0 verification suite"
echo "# commit ${SHA} (branch ${BRANCH})  ts=${TS}"
echo "# VERIFY_RUN_GPU=${VERIFY_RUN_GPU:-0}  (GPU rows ${VERIFY_RUN_GPU:+execute}${VERIFY_RUN_GPU:-deferred})"
echo "############################################################"

echo
echo "===== CPU ROWS (executed here) ====="
for entry in "${CPU_ROWS[@]}"; do
  IFS='|' read -r id script kind claim <<<"${entry}"
  run_row "${id}" "${script}" "${kind}" "${claim}"
done

echo
echo "===== GPU ROWS (STRICTLY SERIAL -- one at a time, never parallel) ====="
echo "NOTE: GPU rows share a single GPU; they are run sequentially. When"
echo "      VERIFY_RUN_GPU is not set they are recorded as GPU-deferred (manager-sequenced)."
for entry in "${GPU_ROWS[@]}"; do
  IFS='|' read -r id script kind claim <<<"${entry}"
  run_row "${id}" "${script}" "${kind}" "${claim}"
  # Strict serialization barrier: ensure the previous GPU process has fully exited
  # (run_row is synchronous, so this is already serial; the wait is a belt-and-braces
  # guard against any backgrounding inside a row script).
  wait 2>/dev/null || true
done

# ---- write proofs/PROOF_TABLE.md ----
ALL_IDS=(1 2 3 4 5 6 7 8 9 10 11)
declare -A SCRIPT_FOR
for entry in "${CPU_ROWS[@]}" "${GPU_ROWS[@]}"; do
  IFS='|' read -r id script kind claim <<<"${entry}"
  SCRIPT_FOR["$id"]="$script"
done

pass=0; fail=0; deferred=0; missing=0
for id in "${ALL_IDS[@]}"; do
  case "${RESULT_STATUS[$id]:-MISSING}" in
    PASS) pass=$((pass+1));;
    FAIL) fail=$((fail+1));;
    GPU_DEFERRED) deferred=$((deferred+1));;
    *) missing=$((missing+1));;
  esac
done

{
  echo "# v0.1.0 Proof Table (generated)"
  echo
  echo "- **Commit:** \`${SHA}\` (\`${SHORT_SHA}\`)"
  echo "- **Branch:** \`${BRANCH}\`"
  echo "- **Generated (UTC):** ${TS}"
  echo "- **GPU rows executed:** $([ "${VERIFY_RUN_GPU:-0}" = "1" ] && echo yes || echo "no (manager-sequenced)")"
  echo "- **Tally:** ${pass} PASS / ${fail} FAIL / ${deferred} GPU-deferred / ${missing} missing (of 11)"
  echo "- **Logs:** \`${LOG_DIR#${REPO_ROOT}/}\`"
  echo
  echo "Each row is re-derived from source by \`scripts/verify/<row>.sh\` (it re-runs the"
  echo "real validation and asserts the gate); this table records the outcome + key numbers."
  echo
  echo "| # | Status | Verify script | Claim / key numbers |"
  echo "|---|--------|---------------|----------------------|"
  for id in "${ALL_IDS[@]}"; do
    st="${RESULT_STATUS[$id]:-MISSING}"
    case "${st}" in
      PASS) badge="PASS";;
      FAIL) badge="**FAIL**";;
      GPU_DEFERRED) badge="GPU: manager-sequenced";;
      *) badge="MISSING";;
    esac
    note="${RESULT_NOTE[$id]:-}"
    # escape pipes in notes
    note="${note//|/\\|}"
    echo "| ${id} | ${badge} | \`scripts/verify/${SCRIPT_FOR[$id]:-?}\` | ${note} |"
  done
  echo
  echo "## Notes"
  echo
  echo "- **GPU rows (1-5, 7, 8, 11)** run on a single GPU **strictly serially**, never in"
  echo "  parallel. Re-run this suite on an idle GPU with \`VERIFY_RUN_GPU=1\` to execute"
  echo "  and assert them; until then they read \`GPU: manager-sequenced\`."
  echo "- **CPU rows (6, 9, 10)** are executed here and asserted from source."
  echo "- **Row 5 (d03)** asserts the REAL bounded gate honestly; it records FAIL until the"
  echo "  HFX fix (#56) lands -- the gate is NOT relaxed to manufacture a PASS."
  echo "- **Row 6 (TOST)** here verifies the scorer machinery (CPU-vs-CPU paired delta = 0);"
  echo "  the achievable-N GPU equivalence pass + seasonal-breadth limitation are documented"
  echo "  and manager-sequenced."
  echo "- Per VERIFICATION.md, v0.1.0 tags only when rows 1-11 are all PASS on the release"
  echo "  commit (row 6 = pass on achievable N; row 10 = honest characterization)."
} > "${TABLE}"

echo
echo "############################################################"
echo "# wrote ${TABLE#${REPO_ROOT}/}"
echo "# tally: ${pass} PASS / ${fail} FAIL / ${deferred} GPU-deferred / ${missing} missing"
echo "############################################################"
cat "${TABLE}"

# Exit non-zero if any row actually FAILED (deferred/missing do not fail the suite,
# so a CPU-only reviewer run is green when all CPU rows pass and GPU rows are deferred).
if [ "${fail}" -gt 0 ]; then
  exit 1
fi
exit 0
