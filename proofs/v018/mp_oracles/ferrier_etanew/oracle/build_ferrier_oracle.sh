#!/usr/bin/env bash
# Build the Ferrier "new Eta" (WRF mp_physics=95, etampnew/EGCP01)
# single-column savepoint oracle against the UNMODIFIED pristine WRF source
# phys/module_mp_etanew.F + the REAL binary lookup-table file
# (run/ETAMPNEW_DATA.expanded_rain).
#
# Usage:
#   bash proofs/v017/oracle/build_ferrier_oracle.sh fp32
#   bash proofs/v017/oracle/build_ferrier_oracle.sh fp64
#
# fp32 = canonical WRF single precision (binding PROGNOSTIC reference); reads
#        the fp32 lookup table ETAMPNEW_DATA.expanded_rain.
# fp64 = -fdefault-real-8 override (scheme source otherwise UNMODIFIED); reads
#        the fp64 lookup table ETAMPNEW_DATA.expanded_rain_DBL (renamed in CWD).
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="<USER_HOME>/src/wrf_pristine/WRF"
WRF_PHYS="${WRF}/phys"
WRF_RUN="${WRF}/run"

source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -e
export OMP_NUM_THREADS=2

MODE="${1:-fp32}"
OUT_SAVE="${HERE}/../savepoints_ferrier"
# IWORDSIZE/RWORDSIZE: WRF default single-precision build (4/4). For -fdefault-
# real-8 the default REAL becomes 8 bytes; the unformatted lookup table is read
# with the matching record length, hence the *_DBL data file.
# -fallow-argument-mismatch: the real WRF build uses this (gfortran 10+) because
# module_mp_etanew.F passes both INTEGER (unit) and REAL arrays to the same
# implicit-interface external wrf_dm_bcast_bytes. It demotes the cross-call type
# check to a warning -- it does NOT modify the scheme source.
FFLAGS_FREE="-O2 -cpp -ffree-form -ffree-line-length-none -fallow-argument-mismatch -ffpe-summary=none -DIWORDSIZE=4 -DRWORDSIZE=4 -I${WRF_PHYS}"
FFLAGS_FIXED="-O2 -cpp -ffixed-form -ffixed-line-length-none -fallow-argument-mismatch -ffpe-summary=none -DIWORDSIZE=4 -DRWORDSIZE=4 -I${WRF_PHYS}"
DATA_FILE="${WRF_RUN}/ETAMPNEW_DATA.expanded_rain"
if [ "${MODE}" = "fp64" ]; then
  OUT_SAVE="${HERE}/../savepoints_ferrier_fp64"
  FFLAGS_FREE="${FFLAGS_FREE} -DDOUBLE_PRECISION -fdefault-real-8 -fdefault-double-8"
  FFLAGS_FIXED="${FFLAGS_FIXED} -DDOUBLE_PRECISION -fdefault-real-8 -fdefault-double-8"
  DATA_FILE="${WRF_RUN}/ETAMPNEW_DATA.expanded_rain_DBL"
fi

BLDDIR="${HERE}/build_ferrier_${MODE}"
rm -rf "${BLDDIR}"
mkdir -p "${BLDDIR}" "${OUT_SAVE}"

cp "${WRF_PHYS}/module_mp_etanew.F" "${BLDDIR}/module_mp_etanew.F"
cp "${HERE}/etanew_stubs.f90"       "${BLDDIR}/etanew_stubs.f90"
cp "${HERE}/ferrier_oracle_driver.f90" "${BLDDIR}/ferrier_oracle_driver.f90"
# the scheme OPENs "ETAMPNEW_DATA.expanded_rain" from CWD; for fp64 we feed the
# _DBL table under that same name.
cp "${DATA_FILE}" "${BLDDIR}/ETAMPNEW_DATA.expanded_rain"

( cd "${BLDDIR}" && sha256sum \
    module_mp_etanew.F etanew_stubs.f90 ferrier_oracle_driver.f90 \
    ETAMPNEW_DATA.expanded_rain \
    > "${OUT_SAVE}/ferrier_source_checksums.txt" )

cd "${BLDDIR}"
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c etanew_stubs.f90
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c module_mp_etanew.F
taskset -c 0-3 gfortran ${FFLAGS_FREE} -c ferrier_oracle_driver.f90

taskset -c 0-3 gfortran ${FFLAGS_FREE} -o ferrier_oracle \
  etanew_stubs.o module_mp_etanew.o ferrier_oracle_driver.o

# The ETAMPNEW_DATA.expanded_rain lookup table is a BIG-ENDIAN unformatted
# sequential file (record markers confirmed: 951 fp32 elements = 3804-byte
# big-endian record length). WRF on this platform built its data tables
# big-endian; tell gfortran's runtime to byte-swap when reading.
export GFORTRAN_CONVERT_UNIT=big_endian
for c in 1 2 3 4 5 6; do
  taskset -c 0-3 ./ferrier_oracle "$c" > "ferrier_case_${c}.txt"
  taskset -c 0-3 python3 "${HERE}/ferrier_dump_to_json.py" "ferrier_case_${c}.txt" \
      "${OUT_SAVE}/ferrier_case_${c}.json" "Ferrier new-Eta (mp_physics=95)"
done

echo "OK: Ferrier oracle (mode=${MODE}) wrote savepoints to ${OUT_SAVE}"
