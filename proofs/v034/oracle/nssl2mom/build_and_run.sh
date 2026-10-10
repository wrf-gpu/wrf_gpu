#!/usr/bin/env bash
# v0.3.4 lane o1-nssl: extended NSSL 2-moment (mp_physics=18) single-column oracle.
#   * pristine build   : UNMODIFIED WRF module_mp_nssl_2mom.F + v2 column driver
#   * instrumented build: print-only copy (make_instrumented.py) dumping module constants and
#                         the state after every driver stage; proven bit-identical (E26).
# Usage: build_and_run.sh <workdir> [cores]   (fp32 and fp64; cases 1..14)
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${1:?workdir}"; CORES="${2:-25}"
WRF_F=<USER_HOME>/src/wrf_pristine/WRF/phys/module_mp_nssl_2mom.F
OUT="${NSSL_O1_OUT:-${HERE}/../../../v034/f2_oracles/nssl_2mom}"
NCASES="${NSSL_O1_NCASES:-15}"
source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild 2>/dev/null
mkdir -p "${WORK}" "${OUT}"
python3 "${HERE}/list_module_vars.py" "${WRF_F}" 1226 > "${WORK}/module_vars.txt" || exit 2
python3 "${HERE}/list_module_vars.py" "${WRF_F}" 13714 12638 > "${WORK}/gs_vars.txt" || exit 2
python3 "${HERE}/list_module_vars.py" --dims "${WRF_F}" 13714 12638 > "${WORK}/gs_vars_dims.txt" || exit 2
python3 "${HERE}/make_instrumented.py" "${WRF_F}" "${WORK}/module_mp_nssl_2mom_instr.F" "${WORK}/module_vars.txt" "${WORK}/gs_vars.txt" || exit 3
for MODE in fp32 fp64; do
  PROMOTE=""; [ "${MODE}" = fp64 ] && PROMOTE="-fdefault-real-8 -fdefault-double-8"
  FFLAGS="-O2 -ffree-form -ffree-line-length-none -ffpe-summary=none -cpp ${PROMOTE}"
  for VAR in pristine instr; do
    B="${WORK}/build_${MODE}_${VAR}"; rm -rf "$B"; mkdir -p "$B"; cd "$B" || exit 4
    if [ "$VAR" = pristine ]; then cp "${WRF_F}" module_mp_nssl_2mom.F; DEF="";
    else cp "${WORK}/module_mp_nssl_2mom_instr.F" module_mp_nssl_2mom.F; DEF="-DO1_INSTR"; fi
    cp "${HERE}/nssl2mom_stub_modules.f90" "${HERE}/nssl2mom_oracle_driver_v2.F90" .
    taskset -c "$CORES" nice -n 19 gfortran ${FFLAGS} -c nssl2mom_stub_modules.f90 || exit 11
    taskset -c "$CORES" nice -n 19 gfortran ${FFLAGS} -c module_mp_nssl_2mom.F || exit 12
    taskset -c "$CORES" nice -n 19 gfortran ${FFLAGS} ${DEF} -c nssl2mom_oracle_driver_v2.F90 || exit 13
    taskset -c "$CORES" nice -n 19 gfortran ${FFLAGS} -o oracle nssl2mom_stub_modules.o module_mp_nssl_2mom.o nssl2mom_oracle_driver_v2.o || exit 14
    : > namelist.input
    for c in $(seq 1 ${NCASES}); do
      if [ "$VAR" = instr ] && [ "$c" = 1 ]; then A2=dumpmv; else A2=""; fi
      taskset -c "$CORES" nice -n 19 ./oracle "$c" $A2 > "case_${c}.txt" || { echo "RUN FAIL $MODE $VAR $c"; exit 20; }
    done
    sha256sum module_mp_nssl_2mom.F nssl2mom_oracle_driver_v2.F90 nssl2mom_stub_modules.f90 > checksums.txt
  done
  # E26: instrumented final outputs must equal pristine bit-for-bit
  for c in $(seq 1 ${NCASES}); do
    if ! cmp -s <(grep -v '^S[0-9]:\|^MV:\|^MVS:\|^G[0-9]:' "${WORK}/build_${MODE}_instr/case_${c}.txt") "${WORK}/build_${MODE}_pristine/case_${c}.txt"; then
      echo "E26 FAIL: instrumented != pristine ($MODE case $c)"; exit 30; fi
  done
  echo "E26 PASS ${MODE}: instrumented final outputs byte-identical to pristine (${NCASES} cases)"
  PREFIX=nssl; [ "$MODE" = fp64 ] && PREFIX=nssl_fp64
  for c in $(seq 1 ${NCASES}); do
    python3 "${HERE}/dump_to_json_v2.py" "${WORK}/build_${MODE}_instr/case_${c}.txt" "${OUT}/${PREFIX}_case_${c}.json" "${MODE}" || exit 31
  done
  cp "${WORK}/build_${MODE}_pristine/checksums.txt" "${OUT}/checksums_${MODE}_pristine.txt"
  cp "${WORK}/build_${MODE}_instr/checksums.txt" "${OUT}/checksums_${MODE}_instr.txt"
  { echo "mode: ${MODE}"; echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "gfortran: $(gfortran --version | head -1)";
    echo "fflags: ${FFLAGS}" | sed 's/[ ]*$//'; echo "pinning: taskset -c ${CORES}"; } > "${OUT}/build_manifest_${MODE}.txt"
done
echo OK
