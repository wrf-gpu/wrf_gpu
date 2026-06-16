#!/usr/bin/env bash
# v0.9.0 Thompson + MYNN-PBL ACTIVE-PRECIP oracle run.
#
# Re-runs the SAME instrumented pristine WRF v4.7.1 (oracle hooks already compiled
# into main/wrf.exe; sources module_mp_thompson.F + module_bl_mynnedmf.F) from the
# SAME real IC/BC (20260428_18z_l3_24h), but for several model hours so warm-rain +
# ice/snow/graupel develop, then captures ONE late, hydrometeor-ACTIVE timestep.
#
# The existing /mnt/data/.../microphysics oracle was captured at itimestep=1 which
# is a near-inactive step (only trace qc). This run captures itimestep=WRFGPU2_ORACLE_STEP
# (default 1000 ~= 5h) where qr/qi/qs are active, into a SEPARATE oracle subdir so the
# original is preserved.
#
# Resource rule: WRF pinned to cores 0-3 (Claude/JAX budget); cores 4-31 are the live
# 28-rank CPU-WRF backfill and MUST NOT be touched. (Backfill is currently spread over
# 0-29 unpinned; pinning here keeps US within 0-3 regardless.)
set -u

# Resolve the directory holding this script + the namelist template BEFORE any cd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

PRISTINE=/home/user/src/wrf_pristine/WRF
SRC=/mnt/data/canairy_meteo/runs/wrf_l3/20260428_18z_l3_24h_20260525T221139Z
RUN="$PRISTINE/test/em_real/oracle_run_v090"
ORACLE_ROOT=/mnt/data/wrf_gpu2/physics_oracle_v090
STEP="${WRFGPU2_ORACLE_STEP:-1000}"
RUN_HOURS="${RUN_HOURS:-6}"

mkdir -p "$RUN"
cd "$RUN" || exit 2

# Symlink everything em_real needs (tables + exe) from the canonical em_real dir.
for f in "$PRISTINE"/test/em_real/*; do
  bn=$(basename "$f")
  case "$bn" in
    namelist.input|oracle_run|oracle_run_v090|rsl.*|wrfout_*|run.log|wrf.pid|wrfinput_d01|wrfbdy_d01) continue;;
  esac
  ln -sf "$f" "$bn" 2>/dev/null
done
ln -sf "$PRISTINE/main/wrf.exe" wrf.exe
ln -sf "$SRC/wrfinput_d01" wrfinput_d01
ln -sf "$SRC/wrfbdy_d01" wrfbdy_d01

# Namelist: SAME physics/dynamics as the original oracle factory, longer run.
# Start 2026-04-28 18z; run RUN_HOURS hours. Compute end date with proper rollover.
END_EPOCH=$(date -u -d "2026-04-28 18:00:00 UTC + ${RUN_HOURS} hours" +"%Y %m %d %H")
read -r EY EM ED EH <<< "$END_EPOCH"
sed "s/^ run_minutes .*/ run_minutes                         = 0,/; \
     s/^ run_hours .*/ run_hours                           = ${RUN_HOURS},/; \
     s/^ end_year .*/ end_year                            = ${EY},/; \
     s/^ end_month .*/ end_month                           = ${EM#0},/; \
     s/^ end_day .*/ end_day                             = ${ED#0},/; \
     s/^ end_hour .*/ end_hour                            = ${EH#0},/; \
     s/^ end_minute .*/ end_minute                          = 0,/; \
     s/^ history_interval .*/ history_interval                    = 360,/" \
     "$SCRIPT_DIR/namelist.oracle.input" > namelist.input

# Oracle activation: grid 1, target step STEP (active precip).
export WRFGPU2_ORACLE=1
export WRFGPU2_ORACLE_GRID=1
export WRFGPU2_ORACLE_STEP="$STEP"
export WRFGPU2_ORACLE_ROOT="$ORACLE_ROOT"

export LD_LIBRARY_PATH=/home/user/miniconda3/envs/wrfbuild/lib:${LD_LIBRARY_PATH:-}
ulimit -s unlimited 2>/dev/null || true
export OMP_NUM_THREADS=1

rm -f rsl.out.0000 rsl.error.0000 run.log
echo "STEP=$STEP RUN_HOURS=$RUN_HOURS ORACLE_ROOT=$ORACLE_ROOT"
# Serial WRF pinned to cores 0-3 ONLY.
taskset -c 0-3 ./wrf.exe > run.log 2>&1
rc=$?
echo "wrf.exe exit=$rc"
tail -5 run.log
exit $rc
