#!/bin/bash
# Real Swiss d01 legacy CPU step, time-boxed (lane o1-tridiag, E105 deadlock fix).
#
#   run_repro.sh <tree> <outdir> [steps]
#
# `tree` selects the code under test via PYTHONPATH=<tree>/src. Run it with the
# tree's CPU solve routing to the LAPACK gtsv FFI (e.g. the mutant
# mutants/M1_cpu_routing_removed, or a pre-fix checkout) to exercise the program
# that deadlocked (E105; o1-restart eu-stack, deterministic at 2 pinned cores in
# ~6.5 CPU-min). With the fix it must finish and report non_finite_scalars == 0.
#
# Operator notes: the deadlock is scheduling-dependent -- XLA:CPU sizes its
# intra-op pool from the CPU affinity, so it is *more* likely on fewer pinned
# cores; on a single pinned core it may or may not re-trigger. The fix removes
# the FFI from the CPU lowering entirely, so the hazard cannot occur at any core
# count (proofs/o1_tridiag/gpu_identity.py asserts the CPU lowering is FFI-free
# and the CUDA lowering is byte-identical).
set -u
tree=$1; out=$2; steps=${3:-1}
if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi
mkdir -p "$out"
JAX_PLATFORMS=cpu GPUWRF_JAX_CACHE=0 PYTHONPATH="$tree/src" \
  GPUWRF_WRF_ROOT=<USER_HOME>/src/wrf_pristine/WRF \
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" \
  CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false GPUWRF_FAST_DEFAULTS=0 \
  timeout 1600 taskset -c "${GPUWRF_REPRO_CORE:-9}" nice -n 19 \
  python "$(dirname "$0")/coupled_smoke.py" --steps "$steps" --out "$out"
echo "rc=$?"
