#!/usr/bin/env bash
# v0.3.4 O1: aerosol-aware Morrison (WRF mp_physics=40) single-column oracle at
# the WRF-REACHABLE stand-alone operating point aercu_opt=0 (Registry default).
#
# WRF share/module_check_a_mundo.F:1407 makes aercu_opt>0 FATAL unless
# cu_physics=11 (MSKF) AND mp_physics=40, and aercu_opt>0 also needs the
# CESM_RCP4.5_Aerosol_Data.dat climatology (module_physics_init.F:5545). With
# aercu_opt=0 the mp=40 wrapper takes the ELSE branch (aero.F l.1071-1074):
# nc1d=0 placeholder, iinum=INUM=1 (constant droplets NDCNST=250 cm-3),
# INUC stays 0 (init l.410), no activation, no CCN/EFCG diagnostics -- but the
# aero module's own constants (DCS=350e-6 cascade) and cloud-free Reff
# defaults still apply, so mp=40/aercu_opt=0 != mp=10.
#
# Reuses the v0.23 driver/stub/converter (proofs/v023/oracle/morraero); the
# committed driver is never edited -- the BUILD COPY gets two documented
# patches (sha256 of the original and the patched copy are both recorded):
#   (a) PARAMETER AERCU_OPT 2 -> ${AERCU} (env AERCU, default 0);
#   (b) an added sounding CASE (7): cold (tsfc 258 K), ice-supersaturated,
#       hydrometeor-FREE column -> ice nucleation actually fires (cases 1-6
#       all seed ice wherever nucleation would trigger, so INUC=0 Cooper vs
#       INUC=2 Liu-Penner was invisible in them; E154 census).
# AERCU=2 with cases 1-6 must reproduce proofs/v022/f2_oracles/morrison_aero
# byte-for-byte: those rebuilt JSONs stay in the build dir (no duplicates) and
# their sha256 go to repro_opt2_vs_v022_${MODE}.txt, which
# tests/savepoint/test_morrison_aero_opt0_parity.py checks against the
# committed v022 files.
# Pristine WRF sources copied verbatim.
# CPU only, pinned to lane cores 24,25 (o1-morraero contract).
set -o pipefail   # NOT -e/-u: conda activate hooks reference unset vars.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${HERE}/../../../v023/oracle/morraero"
WRF="<USER_HOME>/src/wrf_pristine/WRF"
OUT_SAVE="${HERE}/../../f2_oracles/morrison_aero_opt0"
CORES="${MORRAERO_CORES:-24,25}"

source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
export OMP_NUM_THREADS=1
BLDDIR="${MORRAERO_BLDDIR:-<USER_HOME>/wrf_gpu2_lanes/o1-morraero/oracle_opt0_build}"

MODE="${1:-fp32}"
AERCU="${AERCU:-0}"
case "${AERCU}" in 0|2) ;; *) echo "AERCU must be 0 or 2"; exit 9;; esac
PROMOTE=""
PREFIX="morr_aero_opt${AERCU}_case"
if [ "${MODE}" = "fp64" ]; then
  PREFIX="morr_aero_opt${AERCU}_fp64_case"
  PROMOTE="-fdefault-real-8 -fdefault-double-8"
fi

rm -rf "${BLDDIR}"; mkdir -p "${BLDDIR}" "${OUT_SAVE}"
cp "${WRF}/share/module_model_constants.F"         "${BLDDIR}/"
cp "${WRF}/phys/module_mp_radar.F"                 "${BLDDIR}/"
cp "${WRF}/phys/module_mp_morr_two_moment_aero.F"  "${BLDDIR}/"
cp "${SRC}/morraero_stub_modules.f90"              "${BLDDIR}/"
cp "${SRC}/morraero_oracle_driver.f90"             "${BLDDIR}/morraero_oracle_driver_v023.f90"
sed -e "s/INTEGER, PARAMETER :: AERCU_OPT = 2/INTEGER, PARAMETER :: AERCU_OPT = ${AERCU}/" \
    -e '0,/^    CASE DEFAULT$/s//    CASE (7)   ! v034 O1: cold, ice-supersaturated, hydrometeor-free -> ice nucleation\
      psfc=850.0E2;  tsfc=258.0; zml=400.0;  lapse=5.5E-3; rh_ml=1.05; rh_trop=0.90; wmax=0.5\
    CASE DEFAULT/' \
    "${BLDDIR}/morraero_oracle_driver_v023.f90" > "${BLDDIR}/morraero_oracle_driver.f90"
grep -q "AERCU_OPT = ${AERCU}\$" "${BLDDIR}/morraero_oracle_driver.f90" || { echo "PATCH FAIL"; exit 10; }
NDIFF="$(diff "${BLDDIR}/morraero_oracle_driver_v023.f90" "${BLDDIR}/morraero_oracle_driver.f90" | grep -c '^[<>]')"
# diff marker lines: opt2 = 2 inserted (>); opt0 = 1 changed (<,>) + 2 inserted.
EXPECT=2; [ "${AERCU}" = 0 ] && EXPECT=4
[ "${NDIFF}" = "${EXPECT}" ] || { echo "PATCH SIZE ${NDIFF} != ${EXPECT}"; exit 10; }

( cd "${BLDDIR}" && sha256sum module_model_constants.F module_mp_radar.F \
    module_mp_morr_two_moment_aero.F morraero_stub_modules.f90 \
    morraero_oracle_driver_v023.f90 morraero_oracle_driver.f90 \
    > "${OUT_SAVE}/wrf_source_checksums_opt${AERCU}_${MODE}.txt" )

cd "${BLDDIR}"
FFLAGS="-O2 -ffree-form -ffree-line-length-none -ffpe-summary=none -cpp ${PROMOTE}"
for f in morraero_stub_modules.f90 module_model_constants.F module_mp_radar.F \
         module_mp_morr_two_moment_aero.F morraero_oracle_driver.f90; do
  taskset -c "${CORES}" nice -n 19 gfortran ${FFLAGS} -c "$f" || exit 12
done
taskset -c "${CORES}" gfortran ${FFLAGS} -o morraero_oracle \
    morraero_stub_modules.o module_model_constants.o module_mp_radar.o \
    module_mp_morr_two_moment_aero.o morraero_oracle_driver.o || exit 16

REPRO="${OUT_SAVE}/repro_opt2_vs_v022_${MODE}.txt"
[ "${AERCU}" = 2 ] && : > "${REPRO}"
for c in 1 2 3 4 5 6 7; do
  taskset -c "${CORES}" nice -n 19 ./morraero_oracle "$c" > "case_${c}.txt" || { echo "RUN FAIL case $c"; exit 20; }
  if [ "${AERCU}" = 2 ] && [ "$c" != 7 ]; then
    # cases 1-6 duplicate v022: keep only their hash (reproducibility receipt)
    python3 -I "${SRC}/dump_to_json.py" "case_${c}.txt" "${BLDDIR}/${PREFIX}_${c}.json" || exit 21
    echo "$(sha256sum "${BLDDIR}/${PREFIX}_${c}.json" | cut -d' ' -f1)  ${PREFIX}_${c}.json" >> "${REPRO}"
  else
    python3 -I "${SRC}/dump_to_json.py" "case_${c}.txt" "${OUT_SAVE}/${PREFIX}_${c}.json" || exit 21
  fi
done

MANIFEST="${OUT_SAVE}/build_manifest.txt"
if [ "${MODE}" = "fp32" ] && [ "${AERCU}" = 0 ]; then : > "${MANIFEST}"; fi
{
  echo "== morraero aercu_opt=${AERCU} oracle build: mode=${MODE} $(date -Is) host=$(hostname) =="
  echo "compiler : $(gfortran --version | head -1)"
  echo "FFLAGS   : ${FFLAGS}"
  echo "WRF tree : ${WRF} (sources copied verbatim; sha256 in wrf_source_checksums.txt)"
  echo "driver   : proofs/v023/oracle/morraero/morraero_oracle_driver.f90 + build-copy patches (AERCU_OPT=${AERCU}, sounding CASE 7)"
  echo "entry    : MORR_TWO_MOMENT_INIT_AERO(0); MP_MORR_TWO_MOMENT_AERO (aercu_opt=${AERCU})"
  echo "cases    : 1-7 -> ${PREFIX}_{1..7}.json"
  echo ""
} >> "${MANIFEST}"
echo "OK: Morrison-aero aercu_opt=0 oracle (mode=${MODE}) -> ${OUT_SAVE}"
