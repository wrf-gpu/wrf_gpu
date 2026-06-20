#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}"
WRF_ROOT="${WRF_ROOT:-<USER_HOME>/src/wrf_pristine/WRF}"
BUILD_DIR="$ROOT/proofs/v018/mp_oracles/ferrier_hires/oracle/build"
OUT_DIR="$ROOT/proofs/v018/mp_oracles/ferrier_hires/savepoints"
SRC_DIR="$ROOT/proofs/v018/mp_oracles/ferrier_hires/oracle"

mkdir -p "$BUILD_DIR" "$OUT_DIR"

if [ -f <USER_HOME>/miniconda3/etc/profile.d/conda.sh ]; then
  set +u
  source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
  conda activate wrfbuild
  set -u
fi

if ! command -v gfortran >/dev/null 2>&1; then
  echo "gfortran not found" >&2
  exit 2
fi

cp "$WRF_ROOT/run/ETAMPNEW_DATA.expanded_rain" "$BUILD_DIR/ETAMPNEW_DATA.expanded_rain"

sha256sum \
  "$WRF_ROOT/phys/module_mp_fer_hires.F" \
  "$WRF_ROOT/phys/module_mp_etanew.F" \
  "$WRF_ROOT/run/ETAMPNEW_DATA.expanded_rain" \
  "$SRC_DIR/ferrier_oracle_driver.f90" \
  "$SRC_DIR/ferrier_wrf_stubs.f90" \
  > "$OUT_DIR/ferrier_hires_source_checksums.txt"

FCFLAGS=(-O2 -cpp -ffree-form -ffree-line-length-none -ffpe-summary=none -fallow-argument-mismatch -fconvert=big-endian -frecord-marker=4 -DIWORDSIZE=4 -DRWORDSIZE=4)

(
  cd "$BUILD_DIR"
  gfortran "${FCFLAGS[@]}" -c "$SRC_DIR/ferrier_wrf_stubs.f90"
  gfortran "${FCFLAGS[@]}" -c "$WRF_ROOT/phys/module_mp_etanew.F"
  gfortran "${FCFLAGS[@]}" -c "$WRF_ROOT/phys/module_mp_fer_hires.F"
  gfortran "${FCFLAGS[@]}" -c "$SRC_DIR/ferrier_oracle_driver.f90"
  gfortran -o ferrier_oracle_driver ferrier_wrf_stubs.o module_mp_etanew.o module_mp_fer_hires.o ferrier_oracle_driver.o
  for scheme in mp5; do
    for cid in 1 2 3 4; do
      ./ferrier_oracle_driver "$scheme" "$cid" > "$OUT_DIR/ferrier_${scheme}_case_${cid}.dump"
      python3 "$SRC_DIR/ferrier_dump_to_json.py" \
        "$OUT_DIR/ferrier_${scheme}_case_${cid}.dump" \
        "$OUT_DIR/ferrier_${scheme}_case_${cid}.json"
    done
  done
)

echo "Ferrier oracle savepoints written to $OUT_DIR"
