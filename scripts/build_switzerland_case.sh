#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# build_switzerland_case.sh — native real-init pipeline for the Gotthard /
# Central-Switzerland equivalence case (v0.12.0, NON-Canary generalization).
#
# Produces, in $RUNROOT/run_cpu:  wrfinput_d01 + wrfbdy_d01  (real.exe output)
#   plus the namelist.input the GPU port and CPU-WRF both consume.
#
# Pipeline:  geogrid  ->  download GFS (GCP public archive)  ->  ungrib  ->
#            metgrid  ->  real.exe.   All offline assets are local; only the
#            GFS GRIB2 download needs internet (one cycle, ~1.3 GB at 0.5deg).
#
# This is the SETUP step (the hard part). It is run ONCE to mint the case.
# The GPU forecast + CPU reference + comparison are driven separately by
# scripts/equivalence_switzerland.sh.
#
# Domain (robust "it works" default; see docs/equivalence-switzerland.md):
#   center 46.65N 8.55E (Gotthard), Lambert, dx=dy=3 km, 43x43 (~126 km sq),
#   45 levels, p_top=5000 Pa, single domain forced by GFS lateral boundaries.
#
# Env assumptions (all confirmed present on this workstation):
#   WPS  : /mnt/data/canairy_meteo/artifacts/wrf_src/WPS/install_gen2_dmpar/bin
#   geog : /mnt/data/canairy_meteo/artifacts/wps_geog/WPS_GEOG_LOW_RES (global)
#   WRF  : /home/enric/src/wrf_pristine/WRF (serial gfortran build; real.exe)
#   conda: env that holds the WPS runtime libs (wrf-build) is sourced via
#          LD_LIBRARY_PATH already baked into the WPS exes' RPATH.
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

# ── Configuration (override via env) ──────────────────────────────────────
RUNROOT="${RUNROOT:-/mnt/data/wrf_gpu_switzerland}"
WPS_SRC="${WPS_SRC:-/mnt/data/canairy_meteo/artifacts/wrf_src/WPS}"
WPS_BIN="${WPS_BIN:-${WPS_SRC}/install_gen2_dmpar/bin}"
GEOG="${GEOG:-/mnt/data/canairy_meteo/artifacts/wps_geog/WPS_GEOG_LOW_RES}"
WRF="${WRF:-/home/enric/src/wrf_pristine/WRF}"
VTABLE="${VTABLE:-${WPS_SRC}/ungrib/Variable_Tables/Vtable.GFS}"

# Case definition
INIT_DATE="${INIT_DATE:-2023-01-15}"   # YYYY-MM-DD
INIT_CYCLE="${INIT_CYCLE:-00}"         # GFS cycle hour (00/06/12/18)
FCST_HOURS="${FCST_HOURS:-24}"         # forecast length (drop to 12 for short)
BDY_INTERVAL_H="${BDY_INTERVAL_H:-3}"  # boundary interval in hours
GFS_RES="${GFS_RES:-0p50}"             # 0p50 (148 MB/file) or 0p25 (499 MB/file)

# Domain geometry (Gotthard / Central Switzerland)
REF_LAT="${REF_LAT:-46.65}"
REF_LON="${REF_LON:-8.55}"
DX="${DX:-3000}"
E_WE="${E_WE:-43}"
E_SN="${E_SN:-43}"
E_VERT="${E_VERT:-45}"

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

mkdir -p "$GFS_DIR" "$WPS_DIR" "$CPU_DIR"
log "═══ build_switzerland_case ═══"
log "case: ${INIT_STR} -> ${END_STR}  (${FCST_HOURS}h, bdy=${BDY_INTERVAL_H}h, GFS ${GFS_RES})"
log "domain: center ${REF_LAT}N ${REF_LON}E  ${E_WE}x${E_SN} @ ${DX}m  ${E_VERT} levels"
log "runroot: ${RUNROOT}"

# ── Preflight ────────────────────────────────────────────────────────────
[[ -x "${WPS_BIN}/geogrid.exe" ]] || die "geogrid.exe missing at ${WPS_BIN}"
[[ -x "${WPS_BIN}/ungrib.exe"  ]] || die "ungrib.exe missing"
[[ -x "${WPS_BIN}/metgrid.exe" ]] || die "metgrid.exe missing"
[[ -x "${WRF}/main/real.exe"   ]] || die "real.exe missing at ${WRF}/main"
[[ -f "$VTABLE" ]] || die "Vtable.GFS missing at ${VTABLE}"
[[ -d "$GEOG"   ]] || die "geog dataset missing at ${GEOG}"

# ════════════════════════════════════════════════════════════════════════
# STEP 1 — geogrid (terrain/landuse on the Gotthard domain)
# ════════════════════════════════════════════════════════════════════════
log "── STEP 1: geogrid ──"
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
 e_we              = ${E_WE},
 e_sn              = ${E_SN},
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
N_GEO=$(find "${WPS_DIR}" -maxdepth 1 -name 'geo_em.d01.nc' | wc -l)
if [[ $N_GEO -lt 1 ]]; then
  ( cd "$WPS_DIR" && "${WPS_BIN}/geogrid.exe" > geogrid.log 2>&1 )
  [[ -f "${WPS_DIR}/geo_em.d01.nc" ]] || { tail -25 "${WPS_DIR}/geogrid.log"; die "geogrid failed"; }
fi
log "  geo_em.d01.nc: $(ls -la ${WPS_DIR}/geo_em.d01.nc 2>/dev/null | awk '{print $5}') bytes"

# ════════════════════════════════════════════════════════════════════════
# STEP 2 — download GFS (GCP public archive; only step needing internet)
# ════════════════════════════════════════════════════════════════════════
log "── STEP 2: GFS download (${GFS_RES}, f000..f$(printf '%03d' $FCST_HOURS) every ${BDY_INTERVAL_H}h) ──"
GCP_BASE="https://storage.googleapis.com/global-forecast-system/gfs.${GFSDATE}/${INIT_CYCLE}/atmos"
for ((fh=0; fh<=FCST_HOURS; fh+=BDY_INTERVAL_H)); do
  fff=$(printf '%03d' "$fh")
  url="${GCP_BASE}/gfs.t${INIT_CYCLE}z.pgrb2.${GFS_RES}.f${fff}"
  out="${GFS_DIR}/gfs.t${INIT_CYCLE}z.pgrb2.${GFS_RES}.f${fff}"
  if [[ -s "$out" ]] && [[ "$(head -c4 "$out")" == "GRIB" ]]; then
    log "  f${fff}: cached ($(du -h "$out" | cut -f1))"; continue
  fi
  log "  f${fff}: downloading ..."
  curl -fS -o "$out" "$url" 2>>"${GFS_DIR}/download.log" \
    || die "GFS download failed for f${fff} (${url}) — see ${GFS_DIR}/download.log"
  [[ "$(head -c4 "$out")" == "GRIB" ]] || die "f${fff} not a GRIB file"
done
log "  GFS files: $(ls ${GFS_DIR}/gfs.t${INIT_CYCLE}z.pgrb2.${GFS_RES}.f* 2>/dev/null | wc -l)  total $(du -sh ${GFS_DIR} | cut -f1)"

# ════════════════════════════════════════════════════════════════════════
# STEP 3 — ungrib (GFS GRIB2 -> intermediate WPS format)
# ════════════════════════════════════════════════════════════════════════
log "── STEP 3: ungrib ──"
rm -f "${WPS_DIR}"/GRIBFILE.* "${WPS_DIR}"/GFS:* "${WPS_DIR}/Vtable"
ln -sf "$VTABLE" "${WPS_DIR}/Vtable"
IDX=0
for GRIB in $(ls "${GFS_DIR}"/gfs.t${INIT_CYCLE}z.pgrb2.${GFS_RES}.f* | sort); do
  LETTER=$(python3 -c "i=${IDX};print(chr(65+i//676)+chr(65+(i//26)%26)+chr(65+i%26))")
  ln -sf "$GRIB" "${WPS_DIR}/GRIBFILE.${LETTER}"
  IDX=$((IDX+1))
done
log "  linked ${IDX} GRIBFILE.* "
( cd "$WPS_DIR" && "${WPS_BIN}/ungrib.exe" > ungrib.log 2>&1 )
N_SLICES=$(find "${WPS_DIR}" -maxdepth 1 -name 'GFS:*' | wc -l)
EXPECT_SLICES=$(( FCST_HOURS / BDY_INTERVAL_H + 1 ))
log "  ungrib produced ${N_SLICES}/${EXPECT_SLICES} intermediate slices"
[[ $N_SLICES -ge $EXPECT_SLICES ]] || { tail -25 "${WPS_DIR}/ungrib.log"; die "ungrib produced too few slices"; }

# ════════════════════════════════════════════════════════════════════════
# STEP 4 — metgrid (horizontal interpolation onto the model grid)
# ════════════════════════════════════════════════════════════════════════
log "── STEP 4: metgrid ──"
mkdir -p "${WPS_DIR}/metgrid"
ln -sf "${WPS_SRC}/metgrid/METGRID.TBL.ARW" "${WPS_DIR}/metgrid/METGRID.TBL"
rm -f "${WPS_DIR}"/met_em.d01.*.nc
( cd "$WPS_DIR" && "${WPS_BIN}/metgrid.exe" > metgrid.log 2>&1 )
N_METEM=$(find "${WPS_DIR}" -maxdepth 1 -name 'met_em.d01.*.nc' | wc -l)
log "  metgrid produced ${N_METEM}/${EXPECT_SLICES} met_em files"
[[ $N_METEM -ge $EXPECT_SLICES ]] || { tail -25 "${WPS_DIR}/metgrid.log"; die "metgrid produced too few met_em files"; }

# Discover the number of metgrid pressure + soil levels for the namelist.
NUM_METGRID_LEVELS=$(python3 -c "
from netCDF4 import Dataset
ds=Dataset(sorted(__import__('glob').glob('${WPS_DIR}/met_em.d01.*.nc'))[0])
print(ds.dimensions['num_metgrid_levels'].size)")
NUM_SOIL_LEVELS=$(python3 -c "
from netCDF4 import Dataset
ds=Dataset(sorted(__import__('glob').glob('${WPS_DIR}/met_em.d01.*.nc'))[0])
print(ds.getncattr('NUM_METGRID_SOIL_LEVELS'))" 2>/dev/null || echo 4)
log "  num_metgrid_levels=${NUM_METGRID_LEVELS}  num_metgrid_soil_levels=${NUM_SOIL_LEVELS}"

# ════════════════════════════════════════════════════════════════════════
# STEP 5 — real.exe (vertical interp + IC/BC -> wrfinput_d01 + wrfbdy_d01)
# ════════════════════════════════════════════════════════════════════════
log "── STEP 5: real.exe ──"
RUN_DAYS=$(( FCST_HOURS / 24 ))
RUN_HRS=$(( FCST_HOURS % 24 ))
# time_step: 6*dx_km is the WRF rule of thumb; 3km -> 18 s (CFL-safe).
TIME_STEP=$(python3 -c "print(int(6*${DX}/1000))")

cat > "${CPU_DIR}/namelist.input" << EOF
&time_control
 run_days                            = ${RUN_DAYS},
 run_hours                           = ${RUN_HRS},
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
 time_step                           = ${TIME_STEP},
 time_step_fract_num                 = 0,
 time_step_fract_den                 = 1,
 max_dom                             = 1,
 e_we                                = ${E_WE},
 e_sn                                = ${E_SN},
 e_vert                              = ${E_VERT},
 p_top_requested                     = 5000,
 num_metgrid_levels                  = ${NUM_METGRID_LEVELS},
 num_metgrid_soil_levels             = ${NUM_SOIL_LEVELS},
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

# Stage WRF run-dir tables + met_em into the CPU run dir.
for f in LANDUSE.TBL VEGPARM.TBL SOILPARM.TBL GENPARM.TBL MPTABLE.TBL \
         RRTMG_LW_DATA RRTMG_SW_DATA RRTM_DATA \
         ozone.formatted ozone_lat.formatted ozone_plev.formatted \
         aerosol.formatted aerosol_lat.formatted aerosol_lon.formatted aerosol_plev.formatted \
         CAMtr_volume_mixing_ratio CAMtr_volume_mixing_ratio.A1B CAMtr_volume_mixing_ratio.A2 \
         CAMtr_volume_mixing_ratio.RCP4.5 CAMtr_volume_mixing_ratio.RCP6 CAMtr_volume_mixing_ratio.RCP8.5 \
         grib2map.tbl tr49t67 tr49t85 tr67t85 ; do
  [[ -e "${WRF}/run/${f}" ]] && ln -sf "${WRF}/run/${f}" "${CPU_DIR}/${f}"
done
ln -sf "${WPS_DIR}"/met_em.d01.*.nc "${CPU_DIR}/"

# real.exe (serial) — uses the case namelist + met_em to write wrfinput/wrfbdy.
log "  running real.exe (serial) ..."
rm -f "${CPU_DIR}"/wrfinput_d01 "${CPU_DIR}"/wrfbdy_d01 "${CPU_DIR}"/rsl.*
( cd "$CPU_DIR" && "${WRF}/main/real.exe" > real.log 2>&1 )
if [[ ! -s "${CPU_DIR}/wrfinput_d01" || ! -s "${CPU_DIR}/wrfbdy_d01" ]]; then
  log "  real.exe did not produce both outputs; tail of log:"
  tail -30 "${CPU_DIR}/real.log" 2>/dev/null
  tail -10 "${CPU_DIR}"/rsl.error.0000 2>/dev/null
  die "real.exe failed"
fi
log "  wrfinput_d01: $(ls -la ${CPU_DIR}/wrfinput_d01 | awk '{print $5}') bytes"
log "  wrfbdy_d01:   $(ls -la ${CPU_DIR}/wrfbdy_d01   | awk '{print $5}') bytes"

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
print("  wrfinput dims:", {k:v.size for k,v in ic.dimensions.items()})
for v in ("T2","U10","V10","PSFC","HGT","T","U","V","QVAPOR"):
    if v in ic.variables:
        a=np.asarray(ic.variables[v][:],dtype=np.float64)
        fin=np.isfinite(a).all()
        print(f"    {v:8s} shape={a.shape} min={a.min():.3g} max={a.max():.3g} finite={fin}")
        ok = ok and fin
    else:
        print(f"    {v:8s} MISSING"); ok=False
bd=Dataset(f"{d}/wrfbdy_d01")
print("  wrfbdy dims:", {k:v.size for k,v in bd.dimensions.items()})
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

log "═══ build complete ═══"
log "  CPU/GPU case dir: ${CPU_DIR}"
log "  -> wrfinput_d01, wrfbdy_d01, namelist.input ready."
log "  Next: scripts/equivalence_switzerland.sh"
