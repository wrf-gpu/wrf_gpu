# shellcheck shell=bash
# Common helpers + canonical environment for every v0.1.0 verify/<row>.sh script.
#
# Each row script sources this, then either:
#   * (CPU rows) re-runs the real validation on CPU and asserts the gate, or
#   * (GPU rows) sets up the exact documented runner command and either runs it
#     (when the manager invokes with VERIFY_RUN_GPU=1) or prints it and exits
#     with the GPU-deferred marker so the suite can record "[GPU: manager-sequenced]".
#
# A row PASSES iff it prints "RESULT: <row> PASS ..." and exits 0.
# A row FAILS  iff it exits non-zero (it must also print "RESULT: <row> FAIL ...").
# A GPU row that is NOT run prints "RESULT: <row> GPU_DEFERRED ..." and exits 0
# with $VERIFY_GPU_DEFERRED_EXIT semantics handled by verify_all.sh.

set -u  # individual scripts decide on -e; helpers tolerate command failure.

# --- canonical reproducibility environment (VERIFICATION.md documented) ---
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-true}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.7}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Repo root = two levels up from this file (scripts/verify/_common.sh).
VERIFY_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${VERIFY_COMMON_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Interpreter: honour VERIFY_PYBIN if set (point it at the env where JAX+CUDA is
# installed), otherwise use python3 from PATH. No hard-coded personal path.
PYBIN="${VERIFY_PYBIN:-$(command -v python3)}"
if [ ! -x "${PYBIN}" ]; then PYBIN="$(command -v python3)"; fi
export PYBIN

# taskset prefix for CPU pinning (cores 0-3). Skipped if taskset is absent.
if command -v taskset >/dev/null 2>&1; then
  TASKSET="taskset -c 0-3"
else
  TASKSET=""
fi
export TASKSET

# Force JAX onto CPU for the rows we are allowed to execute here (the GPU is busy
# with a physics fix; the manager sequences true GPU rows). Set VERIFY_RUN_GPU=1 to
# let a GPU row actually run on the GPU.
verify_force_cpu() { export JAX_PLATFORMS=cpu; }

# The current git commit SHA (release-commit provenance for every proof).
verify_commit_sha() { git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo "UNKNOWN"; }

# Emit the machine-parseable result line that verify_all.sh greps for.
# usage: verify_result <ROWID> <PASS|FAIL|GPU_DEFERRED> "free-form numbers/notes"
verify_result() {
  printf 'RESULT: %s %s %s\n' "$1" "$2" "${3:-}"
}

# GPU rows call this to register the documented runner and bail out cleanly when
# not actually running on the GPU. Honors VERIFY_RUN_GPU=1 to proceed instead.
# usage: verify_gpu_guard <ROWID> "<one-line description of the runner command>"
verify_gpu_guard() {
  local rowid="$1" cmd="$2"
  if [ "${VERIFY_RUN_GPU:-0}" = "1" ]; then
    return 0  # caller proceeds to actually run on GPU
  fi
  echo "[GPU: manager-sequenced] row ${rowid} is a GPU proof and is NOT executed here."
  echo "  Documented runner (run on the GPU box, serial, taskset -c 0-3):"
  echo "    ${cmd}"
  echo "  Re-invoke this script with VERIFY_RUN_GPU=1 (on an idle GPU) to execute + assert."
  verify_result "${rowid}" "GPU_DEFERRED" "runner=[${cmd}]"
  exit 0
}
