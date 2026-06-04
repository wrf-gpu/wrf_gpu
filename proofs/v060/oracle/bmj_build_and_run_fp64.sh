#!/usr/bin/env bash
# Build a SECOND single-column BMJ oracle against UNMODIFIED WRF
# module_cu_bmj.F, but in fp64 (-fdefault-real-8), and emit JSON savepoints to
# proofs/v060/savepoints_fp64_oracle.
#
# Rationale: the primary oracle (bmj_build_and_run.sh) compiles default REAL
# (fp32). The predeclared BMJ parity tolerances (RTHCUTEN abs 5e-8, RAINCV abs
# 5e-7, rel 1e-6) are tighter than fp32 accumulated round-off on the
# iteratively-corrected DEEP branch, so a faithful fp64 JAX port cannot match
# the fp32 savepoints to 1e-6 even though it IS WRF-faithful. This fp64 oracle
# is a precision-matched cross-check: the JAX port matches it within the SAME
# predeclared tolerances (see run_bmj_parity_fp64.py).
#
# CPU-only, pinned to cores 0-3. Requires conda env `wrfbuild`.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF_BMJ="/home/enric/src/wrf_pristine/WRF/phys/module_cu_bmj.F"
OUT_SAVE="${HERE}/../savepoints_fp64_oracle"

set +u
source /home/enric/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -euo pipefail
export OMP_NUM_THREADS=2

WORK="$(mktemp -d)"
cp "${WRF_BMJ}" "${WORK}/module_cu_bmj.F"
cp "${HERE}/bmj_module_model_constants.f90" "${WORK}/"
cp "${HERE}/bmj_oracle_driver.f90" "${WORK}/"
cd "${WORK}"

FLAGS="-O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none -ffpe-summary=none"
taskset -c 0-3 gfortran $FLAGS -c bmj_module_model_constants.f90
taskset -c 0-3 gfortran $FLAGS -ffree-form -cpp -c module_cu_bmj.F
taskset -c 0-3 gfortran $FLAGS -c bmj_oracle_driver.f90
taskset -c 0-3 gfortran $FLAGS -o bmj_oracle_fp64 \
  bmj_module_model_constants.o module_cu_bmj.o bmj_oracle_driver.o

mkdir -p "${OUT_SAVE}"
for c in 1 2 3 4 5; do
  taskset -c 0-3 ./bmj_oracle_fp64 "$c" > "bmj_fp64_case_${c}.txt"
  python3 "${HERE}/bmj_dump_to_json.py" "bmj_fp64_case_${c}.txt" "${OUT_SAVE}/bmj_case_${c}.json"
done
sha256sum "${WRF_BMJ}" > "${OUT_SAVE}/bmj_wrf_source_checksums.txt"
echo "OK: fp64 BMJ oracle built and 5 savepoints written to ${OUT_SAVE}"
