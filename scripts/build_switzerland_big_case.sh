#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# build_switzerland_big_case.sh — BIG-GRID variant of the Gotthard /
# Central-Switzerland case, minted for the v0.12.0 GPU-vs-28-rank-CPU BENCHMARK.
#
# WHY BIG: at small grids the RTX 5090 is launch-bound (under-utilized). A large
# grid saturates the GPU -> GPU per-cell time drops while 28-rank CPU per-cell
# time stays ~flat -> the HONEST speedup factor GROWS, up to the GPU memory wall.
# So we mint the biggest grid that comfortably fits single-GPU fp64 (~32 GB).
#
# Target: same center/projection/levels/date as F's 43x43 case, but scaled UP to
#   ~150x150 (~450 km square). 45 levels, 24 h, single domain, GFS-forced.
# Falls back to 129x129 (128 mass pts) automatically if 151 trips real.exe.
#
# DIFFERENCES vs scripts/build_switzerland_case.sh (F's robust 43x43 default):
#   * Parameterized large grid (E_WE/E_SN), separate RUNROOT (does NOT clobber
#     F's /mnt/data/wrf_gpu_switzerland 43x43 case).
#   * Reuses F's already-downloaded GFS (symlinks) — no re-download.
#   * Runs real.exe via the **dmpar MPI build** (mpirun -np ${REAL_NP}) so the
#     wrfinput_d01/wrfbdy_d01 are produced by the same WRF build family that the
#     28-rank wrf.exe reference consumes. The GPU port and CPU-WRF consume the
#     SAME wrfinput/wrfbdy.
#
# Produces in $RUNROOT/run_cpu: wrfinput_d01 + wrfbdy_d01 + namelist.input.
# Next: scripts/run_switzerland_cpu_reference_mpi.sh (28-rank reference) and the
# GPU command in scripts/equivalence_switzerland.sh (CASE_ROOT=$RUNROOT).
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

# ── Configuration (override via env) ──────────────────────────────────────
# Maintainer/benchmark step (28-rank big grid). Output roots default to a
# repo-relative writable dir (no /mnt required); the WPS/geog/WRF tool paths are
# this workstation's documented defaults — override them for your machine.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNROOT="${RUNROOT:-${REPO}/runs/switzerland_big}"
# Reuse the 43x43 case assets (cached GFS) by default.
SRC_CASE="${SRC_CASE:-${REPO}/runs/switzerland}"

WPS_SRC="${WPS_SRC:-/path/to/your/WPS}"
WPS_BIN="${WPS_BIN:-${WPS_SRC}/install_gen2_dmpar/bin}"
GEOG="${GEOG:-/path/to/your/WPS_GEOG}"

# dmpar MPI WRF build (28-rank reference build family) — used for real.exe here.
WRF_DMPAR="${WRF_DMPAR:-/path/to/your/WRF/install_dmpar/run}"
WRF_BUILD_ENV="${WRF_BUILD_ENV:-/path/to/your/wrf-build-env}"
MPIRUN="${MPIRUN:-${WRF_BUILD_ENV}/bin/mpirun}"
# real.exe ranks: keep modest; real is cheap and oversubscribe-safe.
REAL_NP="${REAL_NP:-4}"

VTABLE="${VTABLE:-${WPS_SRC}/ungrib/Variable_Tables/Vtable.GFS}"

# Case definition (identical to F's case so physics/forcing match)
INIT_DATE="${INIT_DATE:-2023-01-15}"
INIT_CYCLE="${INIT_CYCLE:-00}"
FCST_HOURS="${FCST_HOURS:-24}"
BDY_INTERVAL_H="${BDY_INTERVAL_H:-3}"
GFS_RES="${GFS_RES:-0p50}"

# Domain geometry (Gotthard / Central Switzerland) — BIG grid
REF_LAT="${REF_LAT:-46.65}"
REF_LON="${REF_LON:-8.55}"
DX="${DX:-3000}"
E_WE="${E_WE:-151}"        # 150 mass points; fallback 129 (=128) on real.exe failure
E_SN="${E_SN:-151}"
E_VERT="${E_VERT:-45}"
FALLBACK_E="${FALLBACK_E:-129}"   # 128 mass points

# ── Derived ────────────────────────────────────────────────────────────────
GFS_DIR="${RUNROOT}/gfs"
WPS_DIR="${RUNROOT}/wps"
CPU_DIR="${RUNROOT}/run_cpu"
INTERVAL_S=$(( BDY_INTERVAL_H * 3600 ))
YEAR="${INIT_DATE:0:4}"; MON="${INIT_DATE:5:2}"; DAY="${INIT_DATE:8:2}"
GFSDATE="${YEAR}${MON}${DAY}"
INIT_STR="${INIT_DATE}_${INIT_CYCLE}:00:00"
END_STR=$(python3 -c "
from datetime import datetime, timedelta
d=datetime(${YEAR},${MON#0},${DAY#0},${INIT_CYCLE#0})
print((d+timedelta(hours=${FCST_HOURS})).strftime('%Y-%m-%d_%H:%M:%S'))")

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
log(){ echo "[$(ts)] $*"; }
die(){ log "FATAL: $*"; exit 1; }

# Run dmpar exes under the wrf-build conda runtime (RPATH already points there,
# but export LD_LIBRARY_PATH defensively + put mpirun on PATH).
export LD_LIBRARY_PATH="${WRF_BUILD_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${WRF_BUILD_ENV}/bin:${PATH}"

mkdir -p "$GFS_DIR" "$WPS_DIR" "$CPU_DIR"
log "═══ build_switzerland_BIG_case ═══"
log "case: ${INIT_STR} -> ${END_STR}  (${FCST_HOURS}h, bdy=${BDY_INTERVAL_H}h, GFS ${GFS_RES})"
log "domain: center ${REF_LAT}N ${REF_LON}E  ${E_WE}x${E_SN} @ ${DX}m  ${E_VERT} levels  (${E_WE} -> $((E_WE-1)) mass pts)"
log "runroot: ${RUNROOT}   (real.exe via dmpar MPI build, -np ${REAL_NP})"

# ── Preflight ────────────────────────────────────────────────────────────
[[ -x "${WPS_BIN}/geogrid.exe" ]] || die "geogrid.exe missing at ${WPS_BIN}"
[[ -x "${WPS_BIN}/ungrib.exe"  ]] || die "ungrib.exe missing"
[[ -x "${WPS_BIN}/metgrid.exe" ]] || die "metgrid.exe missing"
[[ -x "${WRF_DMPAR}/real.exe"  ]] || die "dmpar real.exe missing at ${WRF_DMPAR}"
[[ -x "${MPIRUN}" ]] || die "mpirun missing at ${MPIRUN}"
[[ -f "$VTABLE" ]] || die "Vtable.GFS missing at ${VTABLE}"
[[ -d "$GEOG"   ]] || die "geog dataset missing at ${GEOG}"

# ── Reuse F's cached GFS (symlink) so we do NOT re-download ─────────────────
log "── reuse cached GFS from ${SRC_CASE}/gfs ──"
N_LINKED=0
for f in "${SRC_CASE}"/gfs/gfs.t${INIT_CYCLE}z.pgrb2.${GFS_RES}.f*; do
  [[ -s "$f" ]] || continue
  ln -sf "$f" "${GFS_DIR}/$(basename "$f")"
  N_LINKED=$((N_LINKED+1))
done
log "  linked ${N_LINKED} cached GFS files"
[[ $N_LINKED -ge $(( FCST_HOURS / BDY_INTERVAL_H + 1 )) ]] \
  || die "too few cached GFS files (${N_LINKED}); run F's build_switzerland_case.sh first to download GFS"

# ════════════════════════════════════════════════════════════════════════
# Build the WPS + real pipeline for a given e_we/e_sn. Returns 0 on success.
# ════════════════════════════════════════════════════════════════════════
build_for_grid() {
  local ewe="$1" esn="$2"
  local time_step
  time_step=$(python3 -c "print(int(6*${DX}/1000))")   # 6*dx_km rule -> 18s @3km
  log "── build_for_grid ${ewe}x${esn} (time_step=${time_step}s) ──"

  # STEP 1 — geogrid
  cat > "${WPS_DIR}/namelist.wps" << EOF
&share
 wrf_core = 'ARW',
 max_dom = 1,
 start_date = '${INIT_STR}',
 end_date   = '${END_STR}',
 interval_seconds = ${INTERVAL_S},
/
&geogrid
 parent_id         = 1,
 parent_grid_ratio = 1,
 i_parent_start    = 1,
 j_parent_start    = 1,
 e_we              = ${ewe},
 e_sn              = ${esn},
 geog_data_res     = 'lowres',
 dx = ${DX},
 dy = ${DX},
 map_proj = 'lambert',
 ref_lat   = ${REF_LAT},
 ref_lon   = ${REF_LON},
 truelat1  = 30.0,
 truelat2  = 60.0,
 stand_lon = ${REF_LON},
 geog_data_path = '${GEOG}',
/
&ungrib
 out_format = 'WPS',
 prefix = 'GFS',
/
&metgrid
 fg_name = 'GFS',
 io_form_metgrid = 2,
/
EOF
  mkdir -p "${WPS_DIR}/geogrid"
  ln -sf "${WPS_SRC}/geogrid/GEOGRID.TBL.ARW" "${WPS_DIR}/geogrid/GEOGRID.TBL"
  rm -f "${WPS_DIR}/geo_em.d01.nc"
  ( cd "$WPS_DIR" && "${WPS_BIN}/geogrid.exe" > geogrid.log 2>&1 )
  if [[ ! -f "${WPS_DIR}/geo_em.d01.nc" ]]; then
    tail -25 "${WPS_DIR}/geogrid.log"; log "  geogrid FAILED for ${ewe}x${esn}"; return 1
  fi
  log "  geo_em.d01.nc: $(ls -la ${WPS_DIR}/geo_em.d01.nc | awk '{print $5}') bytes"

  # STEP 3 — ungrib (GFS GRIB2 -> intermediate). geog is grid-independent so
  # ungrib slices could be reused, but re-run is cheap and keeps it self-contained.
  rm -f "${WPS_DIR}"/GRIBFILE.* "${WPS_DIR}"/GFS:* "${WPS_DIR}/Vtable"
  ln -sf "$VTABLE" "${WPS_DIR}/Vtable"
  local idx=0
  for GRIB in $(ls "${GFS_DIR}"/gfs.t${INIT_CYCLE}z.pgrb2.${GFS_RES}.f* | sort); do
    local letter
    letter=$(python3 -c "i=${idx};print(chr(65+i//676)+chr(65+(i//26)%26)+chr(65+i%26))")
    ln -sf "$GRIB" "${WPS_DIR}/GRIBFILE.${letter}"
    idx=$((idx+1))
  done
  ( cd "$WPS_DIR" && "${WPS_BIN}/ungrib.exe" > ungrib.log 2>&1 )
  local n_slices expect_slices
  n_slices=$(find "${WPS_DIR}" -maxdepth 1 -name 'GFS:*' | wc -l)
  expect_slices=$(( FCST_HOURS / BDY_INTERVAL_H + 1 ))
  log "  ungrib: ${n_slices}/${expect_slices} slices"
  [[ $n_slices -ge $expect_slices ]] || { tail -25 "${WPS_DIR}/ungrib.log"; log "  ungrib too few slices"; return 1; }

  # STEP 4 — metgrid
  mkdir -p "${WPS_DIR}/metgrid"
  ln -sf "${WPS_SRC}/metgrid/METGRID.TBL.ARW" "${WPS_DIR}/metgrid/METGRID.TBL"
  rm -f "${WPS_DIR}"/met_em.d01.*.nc
  ( cd "$WPS_DIR" && "${WPS_BIN}/metgrid.exe" > metgrid.log 2>&1 )
  local n_metem
  n_metem=$(find "${WPS_DIR}" -maxdepth 1 -name 'met_em.d01.*.nc' | wc -l)
  log "  metgrid: ${n_metem}/${expect_slices} met_em files"
  [[ $n_metem -ge $expect_slices ]] || { tail -25 "${WPS_DIR}/metgrid.log"; log "  metgrid too few files"; return 1; }

  local num_metgrid_levels num_soil_levels
  num_metgrid_levels=$(python3 -c "
from netCDF4 import Dataset
ds=Dataset(sorted(__import__('glob').glob('${WPS_DIR}/met_em.d01.*.nc'))[0])
print(ds.dimensions['num_metgrid_levels'].size)")
  num_soil_levels=$(python3 -c "
from netCDF4 import Dataset
ds=Dataset(sorted(__import__('glob').glob('${WPS_DIR}/met_em.d01.*.nc'))[0])
print(ds.getncattr('NUM_METGRID_SOIL_LEVELS'))" 2>/dev/null || echo 4)
  log "  num_metgrid_levels=${num_metgrid_levels}  num_metgrid_soil_levels=${num_soil_levels}"

  # STEP 5 — namelist.input (identical physics to F's case; only grid scaled)
  local run_days run_hrs
  run_days=$(( FCST_HOURS / 24 ))
  run_hrs=$(( FCST_HOURS % 24 ))
  cat > "${CPU_DIR}/namelist.input" << EOF
&time_control
 run_days                            = ${run_days},
 run_hours                           = ${run_hrs},
 run_minutes                         = 0,
 run_seconds                         = 0,
 start_year                          = ${YEAR},
 start_month                         = ${MON#0},
 start_day                           = ${DAY#0},
 start_hour                          = ${INIT_CYCLE#0},
 end_year                            = $(echo $END_STR | cut -c1-4),
 end_month                           = $(echo $END_STR | cut -c6-7 | sed 's/^0//'),
 end_day                             = $(echo $END_STR | cut -c9-10 | sed 's/^0//'),
 end_hour                            = $(echo $END_STR | cut -c12-13 | sed 's/^0//'),
 end_minute                          = 0,
 end_second                          = 0,
 interval_seconds                    = ${INTERVAL_S},
 input_from_file                     = .true.,
 history_interval                    = 60,
 frames_per_outfile                  = 1,
 restart                             = .false.,
 restart_interval                    = 100000,
 io_form_history                     = 2,
 io_form_restart                     = 2,
 io_form_input                       = 2,
 io_form_boundary                    = 2,
/

&domains
 time_step                           = ${time_step},
 time_step_fract_num                 = 0,
 time_step_fract_den                 = 1,
 max_dom                             = 1,
 e_we                                = ${ewe},
 e_sn                                = ${esn},
 e_vert                              = ${E_VERT},
 p_top_requested                     = 5000,
 num_metgrid_levels                  = ${num_metgrid_levels},
 num_metgrid_soil_levels             = ${num_soil_levels},
 dx                                  = ${DX},
 dy                                  = ${DX},
 grid_id                             = 1,
 parent_id                           = 1,
 i_parent_start                      = 1,
 j_parent_start                      = 1,
 parent_grid_ratio                   = 1,
 parent_time_step_ratio              = 1,
 feedback                            = 0,
 smooth_option                       = 0,
/

&physics
 mp_physics                          = 8,
 ra_lw_physics                       = 4,
 ra_sw_physics                       = 4,
 radt                                = 10,
 sf_sfclay_physics                   = 5,
 sf_surface_physics                  = 4,
 bl_pbl_physics                      = 5,
 bldt                                = 0,
 cu_physics                          = 0,
 cudt                                = 0,
 isfflx                              = 1,
 ifsnow                              = 1,
 icloud                              = 1,
 surface_input_source                = 1,
 num_soil_layers                     = 4,
 num_land_cat                        = 21,
 sf_urban_physics                    = 0,
/

&dynamics
 w_damping                           = 1,
 diff_opt                            = 1,
 km_opt                              = 4,
 diff_6th_opt                        = 2,
 diff_6th_factor                     = 0.12,
 base_temp                           = 290.,
 damp_opt                            = 3,
 zdamp                               = 5000.,
 dampcoef                            = 0.2,
 khdif                               = 0,
 kvdif                               = 0,
 non_hydrostatic                     = .true.,
 moist_adv_opt                       = 1,
 scalar_adv_opt                      = 1,
 epssm                               = 0.5,
 gwd_opt                             = 0,
/

&bdy_control
 spec_bdy_width                      = 5,
 spec_zone                           = 1,
 relax_zone                          = 4,
 specified                           = .true.,
 nested                              = .false.,
/

&grib2
/

&namelist_quilt
 nio_tasks_per_group                 = 0,
 nio_groups                          = 1,
/
EOF

  # Stage run-dir tables (from the dmpar build's run dir) + met_em into CPU_DIR.
  local f
  for f in LANDUSE.TBL VEGPARM.TBL SOILPARM.TBL GENPARM.TBL MPTABLE.TBL \
           RRTMG_LW_DATA RRTMG_SW_DATA RRTM_DATA \
           ozone.formatted ozone_lat.formatted ozone_plev.formatted \
           aerosol.formatted aerosol_lat.formatted aerosol_lon.formatted aerosol_plev.formatted \
           CAMtr_volume_mixing_ratio CAMtr_volume_mixing_ratio.A1B CAMtr_volume_mixing_ratio.A2 \
           CAMtr_volume_mixing_ratio.RCP4.5 CAMtr_volume_mixing_ratio.RCP6 CAMtr_volume_mixing_ratio.RCP8.5 \
           grib2map.tbl tr49t67 tr49t85 tr67t85 ; do
    [[ -e "${WRF_DMPAR}/${f}" ]] && ln -sf "${WRF_DMPAR}/${f}" "${CPU_DIR}/${f}"
  done
  rm -f "${CPU_DIR}"/met_em.d01.*.nc
  ln -sf "${WPS_DIR}"/met_em.d01.*.nc "${CPU_DIR}/"

  # real.exe via dmpar MPI build.
  log "  running real.exe (dmpar, -np ${REAL_NP}) ..."
  rm -f "${CPU_DIR}"/wrfinput_d01 "${CPU_DIR}"/wrfbdy_d01 "${CPU_DIR}"/rsl.*
  ( cd "$CPU_DIR" && "${MPIRUN}" --oversubscribe --bind-to none -np "${REAL_NP}" \
        "${WRF_DMPAR}/real.exe" > real.log 2>&1 )
  if [[ ! -s "${CPU_DIR}/wrfinput_d01" || ! -s "${CPU_DIR}/wrfbdy_d01" ]]; then
    log "  real.exe did not produce both outputs; tails:"
    tail -20 "${CPU_DIR}/real.log" 2>/dev/null
    tail -15 "${CPU_DIR}"/rsl.error.0000 2>/dev/null
    return 1
  fi
  log "  wrfinput_d01: $(ls -la ${CPU_DIR}/wrfinput_d01 | awk '{print $5}') bytes"
  log "  wrfbdy_d01:   $(ls -la ${CPU_DIR}/wrfbdy_d01   | awk '{print $5}') bytes"
  return 0
}

# ── Try the big grid; fall back to FALLBACK_E on failure ────────────────────
USED_E=$E_WE
if ! build_for_grid "$E_WE" "$E_SN"; then
  log "── ${E_WE}x${E_SN} failed; FALLING BACK to ${FALLBACK_E}x${FALLBACK_E} ──"
  USED_E=$FALLBACK_E
  build_for_grid "$FALLBACK_E" "$FALLBACK_E" || die "fallback grid ${FALLBACK_E}x${FALLBACK_E} also failed"
fi
log "── built grid: ${USED_E}x${USED_E} ($((USED_E-1)) mass pts each axis) ──"

# ════════════════════════════════════════════════════════════════════════
# VALIDATE — dims + finiteness of the produced IC/BC
# ════════════════════════════════════════════════════════════════════════
log "── VALIDATE wrfinput/wrfbdy ──"
python3 - "$CPU_DIR" << 'PYEOF'
import sys, numpy as np
from netCDF4 import Dataset
d=sys.argv[1]
ok=True
ic=Dataset(f"{d}/wrfinput_d01")
print("  wrfinput dims:", {k:v.size for k,v in ic.dimensions.items() if k in
      ("west_east","south_north","bottom_top","Time","soil_layers_stag")})
for v in ("T2","U10","V10","PSFC","HGT","T","U","V","QVAPOR"):
    if v in ic.variables:
        a=np.asarray(ic.variables[v][:],dtype=np.float64)
        fin=np.isfinite(a).all()
        print(f"    {v:8s} shape={a.shape} min={a.min():.3g} max={a.max():.3g} finite={fin}")
        ok = ok and fin
    else:
        print(f"    {v:8s} MISSING"); ok=False
bd=Dataset(f"{d}/wrfbdy_d01")
print("  wrfbdy dims:", {k:v.size for k,v in bd.dimensions.items() if k in
      ("Time","bdy_width","west_east","south_north","bottom_top")})
bvars=[v for v in bd.variables if v.endswith("_BXS") or v.endswith("_BTXS")][:6]
for v in bvars:
    a=np.asarray(bd.variables[v][:],dtype=np.float64)
    print(f"    {v:14s} shape={a.shape} finite={np.isfinite(a).all()}")
    ok = ok and np.isfinite(a).all()
print("  VALIDATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 3)
PYEOF
RC=$?
[[ $RC -eq 0 ]] || die "wrfinput/wrfbdy validation FAILED (rc=$RC)"

log "═══ BIG build complete ═══"
log "  CPU/GPU case dir: ${CPU_DIR}   grid=${USED_E}x${USED_E} @ ${DX}m  ${E_VERT} levels  ${FCST_HOURS}h"
log "  -> wrfinput_d01, wrfbdy_d01, namelist.input ready."
log "  Next (28-rank CPU reference):"
log "    RUNROOT=${RUNROOT} bash scripts/run_switzerland_cpu_reference_mpi.sh"
log "  Then (GPU):"
log "    CASE_ROOT=${RUNROOT} PYTHONPATH=src bash scripts/equivalence_switzerland.sh"
