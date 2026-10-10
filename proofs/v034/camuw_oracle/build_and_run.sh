#!/usr/bin/env bash
# Build + run the v0.3.4 CAM-UW pristine-WRF oracle (CPU only). Usage: build_and_run.sh [BUILD_DIR]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="<USER_HOME>/src/wrf_pristine/WRF"; PHYS="${WRF}/phys"; SHARE="${WRF}/share"; FRAME="${WRF}/frame"
B="${1:-<USER_HOME>/wrf_gpu2_lanes/o1-camuw/oracle/build}"
PY="$(command -v python3)"
set +u; source <USER_HOME>/miniconda3/etc/profile.d/conda.sh; conda activate wrfbuild; set -u
mkdir -p "$B"; cd "$B"
FFLAGS="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2 -I${PHYS} -I${SHARE} -I${FRAME}"
gfortran ${FFLAGS} -c "${HERE}/wrf_runtime_stubs.f90" -o wrf_runtime_stubs.o
gfortran ${FFLAGS} -c "${HERE}/camuw_oracle_driver.f90" -o camuw_oracle_driver.o
gfortran camuw_oracle_driver.o wrf_runtime_stubs.o \
  "${PHYS}"/module_bl_camuwpbl_driver.o "${PHYS}"/module_cam_bl_eddy_diff.o "${PHYS}"/module_cam_bl_diffusion_solver.o \
  "${PHYS}"/module_cam_constituents.o "${PHYS}"/module_cam_support.o "${PHYS}"/module_cam_molec_diff.o \
  "${PHYS}"/module_cam_trb_mtn_stress.o "${PHYS}"/module_cam_wv_saturation.o "${PHYS}"/module_cam_physconst.o \
  "${PHYS}"/module_cam_shr_const_mod.o "${PHYS}"/module_cam_shr_kind_mod.o "${PHYS}"/module_cam_esinti.o \
  "${PHYS}"/module_cam_gffgch.o "${PHYS}"/module_cam_upper_bc.o "${SHARE}"/module_model_constants.o \
  "${FRAME}"/module_state_description.o -lmvec -lm -o camuw_oracle.exe
"$PY" "${HERE}/make_columns.py" "$B/columns.txt"
"$PY" "${HERE}/make_real_columns.py" "$B/columns_swiss.txt"
for set in columns columns_swiss; do
  ./camuw_oracle.exe "$B/${set}.txt" > "$B/${set}_out.txt"
done
"$PY" "${HERE}/parse_oracle.py" "$B/columns_out.txt" "$B/columns.txt.names" "${HERE}/camuw_oracle_v034.json"
"$PY" "${HERE}/parse_oracle.py" "$B/columns_swiss_out.txt" "$B/columns_swiss.txt.names" "${HERE}/camuw_oracle_v034_swiss.json"
{ echo "compiler=$(gfortran --version | head -n 1)"; echo "fflags=${FFLAGS}"; echo "wrf_root=${WRF}";
  echo "objects=prebuilt pristine-WRF objects (WRF REAL ABI, WRF FCOPTIM -O2 -ftree-vectorize -funroll-loops)";
  sha256sum "${PHYS}"/module_bl_camuwpbl_driver.F "${PHYS}"/module_cam_bl_eddy_diff.F "${PHYS}"/module_cam_bl_diffusion_solver.F \
    "${PHYS}"/module_cam_wv_saturation.F "${PHYS}"/module_cam_esinti.F "${PHYS}"/module_cam_gffgch.F \
    "${PHYS}"/module_bl_camuwpbl_driver.o "${PHYS}"/module_cam_bl_eddy_diff.o "${PHYS}"/module_cam_bl_diffusion_solver.o
  sha256sum "${HERE}/camuw_oracle_v034.json" "${HERE}/camuw_oracle_v034_swiss.json"; } > "${HERE}/camuw_build_manifest.txt"
