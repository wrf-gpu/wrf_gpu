#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# run_switzerland_cpu_reference_mpi.sh — 28-RANK MPI CPU-WRF reference for the
# BIG Switzerland (Gotthard) benchmark.  *** MANAGER / maintainer step. ***
#
# This is the HONEST DENOMINATOR for the v0.12.0 speedup: 28-rank dmpar WRF on
# the SAME wrfinput_d01 + wrfbdy_d01 + namelist.input the GPU port consumes, on
# the SAME big grid. NOT 1-core serial.
#
# Captures TWO timings into the run dir:
#   cpu_wall_seconds.txt       — total end-to-end wall of wrf.exe (subprocess).
#   cpu_mainloop_seconds.txt   — sum of "Timing for main" over all steps from
#                                rsl.error.0000 (the pure integration time, the
#                                fairest per-forecast-hour denominator; excludes
#                                MPI init + final I/O).
#   cpu_timing.json            — both, plus ranks/grid/per-fcst-hour numbers.
#
# Launch it DETACHED (it is ~1-1.5 h); hand the manager the PID + log:
#     RUNROOT=/mnt/data/wrf_gpu_switzerland_big \
#       setsid nohup taskset -c 0-27 bash scripts/run_switzerland_cpu_reference_mpi.sh \
#       > $RUNROOT/run_cpu/cpu_reference.out 2>&1 &
#     echo "PID=$!"
# (28 ranks pinned to cores 0-27; leave 28-31 for the OS / Claude orchestration.)
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

RUNROOT="${RUNROOT:-/mnt/data/wrf_gpu_switzerland_big}"
CPU_DIR="${CPU_DIR:-${RUNROOT}/run_cpu}"
NRANKS="${NRANKS:-28}"

WRF_DMPAR="${WRF_DMPAR:-/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_src/WRF/install_gen2_dmpar/run}"
WRF_BUILD_ENV="${WRF_BUILD_ENV:-/home/enric/src/canairy_meteo/Gen2/artifacts/envs/wrf-build}"
MPIRUN="${MPIRUN:-${WRF_BUILD_ENV}/bin/mpirun}"
# OpenMPI flags. We pin the whole job to cores 0-27 with `taskset -c 0-27` at
# launch; under that cgroup mask OpenMPI's own `--bind-to core` policy conflicts
# (it counts fewer bindable cores than ranks and aborts), so we use
# `--bind-to none` and let the taskset mask + the kernel scheduler place the 28
# ranks across cores 0-27. WRF is MPI-only (no OpenMP) in this build.
# --oversubscribe lets us request 28 even if OpenMPI thinks fewer slots are free.
MPI_FLAGS="${MPI_FLAGS:---oversubscribe --bind-to none}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

export LD_LIBRARY_PATH="${WRF_BUILD_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${WRF_BUILD_ENV}/bin:${PATH}"

ts(){ date '+%H:%M:%S'; }
log(){ echo "[$(ts)] $*"; }
die(){ echo "[$(ts)] ERROR: $*" >&2; exit 1; }

[[ -s "${CPU_DIR}/wrfinput_d01"   ]] || die "missing ${CPU_DIR}/wrfinput_d01 — run scripts/build_switzerland_big_case.sh first"
[[ -s "${CPU_DIR}/wrfbdy_d01"     ]] || die "missing ${CPU_DIR}/wrfbdy_d01"
[[ -s "${CPU_DIR}/namelist.input" ]] || die "missing ${CPU_DIR}/namelist.input"
[[ -x "${WRF_DMPAR}/wrf.exe" ]] || die "missing dmpar wrf.exe at ${WRF_DMPAR}"
[[ -x "${MPIRUN}" ]] || die "missing mpirun at ${MPIRUN}"

log "── CPU-WRF reference (dmpar MPI, -np ${NRANKS}) ──"
log "  run dir: ${CPU_DIR}"
log "  binary : ${WRF_DMPAR}/wrf.exe"

# Fresh run: clear prior wrfout/rsl so the comparator + timing see only this run.
rm -f "${CPU_DIR}"/wrfout_d01_* "${CPU_DIR}"/rsl.* "${CPU_DIR}"/wrf.log

T0=$(date +%s.%N)
( cd "$CPU_DIR" && "${MPIRUN}" ${MPI_FLAGS} -np "${NRANKS}" "${WRF_DMPAR}/wrf.exe" > wrf.log 2>&1 )
RC=$?
T1=$(date +%s.%N)
WALL=$(python3 -c "print(${T1}-${T0})")
echo "$WALL" > "${CPU_DIR}/cpu_wall_seconds.txt"

N=$(find "${CPU_DIR}" -maxdepth 1 -name 'wrfout_d01_*' | wc -l)
log "  wrf.exe rc=${RC}  wrfout files=${N}  total wall=${WALL}s"
if [[ $N -lt 2 ]]; then
  log "  tail wrf.log:"; tail -25 "${CPU_DIR}/wrf.log" 2>/dev/null
  tail -15 "${CPU_DIR}"/rsl.error.0000 2>/dev/null
  die "CPU-WRF produced <2 wrfout files (did it crash?)"
fi
grep -q "SUCCESS COMPLETE WRF" "${CPU_DIR}/rsl.error.0000" 2>/dev/null \
  && log "  WRF reported SUCCESS COMPLETE" \
  || log "  NOTE: 'SUCCESS COMPLETE WRF' not in rsl.error.0000 (check tail)"

# ── Honest main-loop integration time: sum "Timing for main" from rank 0 ────
# WRF prints "Timing for main: time <date> on domain 1: <sec> elapsed seconds"
# every step. Their sum is the pure integration wall (no MPI init / final flush).
python3 - "$CPU_DIR" "$NRANKS" << 'PY'
import sys, glob, json, re, os
d, nranks = sys.argv[1], int(sys.argv[2])
rsl = os.path.join(d, "rsl.error.0000")
main_s = 0.0; nsteps = 0
pat = re.compile(r"Timing for main.*?:\s+([0-9.]+)\s+elapsed")
try:
    with open(rsl, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                main_s += float(m.group(1)); nsteps += 1
except FileNotFoundError:
    pass

wall = float(open(os.path.join(d, "cpu_wall_seconds.txt")).read().strip())

# forecast hours from namelist
nm = open(os.path.join(d, "namelist.input")).read()
def gi(key, default=0):
    m = re.search(rf"{key}\s*=\s*([0-9]+)", nm)
    return int(m.group(1)) if m else default
fhours = gi("run_days")*24 + gi("run_hours")
ewe, esn, evert = gi("e_we"), gi("e_sn"), gi("e_vert")
ts = gi("time_step")

out = {
    "cpu_build": f"dmpar MPI gfortran, {nranks} ranks (HONEST denominator, NOT 1-core)",
    "ranks": nranks,
    "grid": {"e_we": ewe, "e_sn": esn, "e_vert": evert,
             "mass_pts": f"{ewe-1}x{esn-1}", "dx_m": 3000, "time_step_s": ts},
    "forecast_hours": fhours,
    "total_wall_s": round(wall, 1),
    "mainloop_sum_s": round(main_s, 1),
    "mainloop_steps": nsteps,
    "total_wall_per_fcst_hour_s": round(wall / fhours, 2) if fhours else None,
    "mainloop_per_fcst_hour_s": round(main_s / fhours, 2) if (fhours and main_s) else None,
    "note": ("mainloop_sum_s = sum of rank-0 'Timing for main' (pure integration); "
             "total_wall_s includes MPI init + I/O. Use the SAME basis for GPU."),
}
with open(os.path.join(d, "cpu_mainloop_seconds.txt"), "w") as f:
    f.write(f"{main_s}\n")
with open(os.path.join(d, "cpu_timing.json"), "w") as f:
    json.dump(out, f, indent=2)
print("  CPU timing:", json.dumps(out, indent=2))
PY

# ── Finiteness check on the last frame ──────────────────────────────────────
python3 - "$CPU_DIR" << 'PY'
import sys, glob, numpy as np
from netCDF4 import Dataset
d=sys.argv[1]
f=sorted(glob.glob(f"{d}/wrfout_d01_*"))[-1]
ds=Dataset(f); ok=True
for v in ("T2","U10","V10","PSFC","T","U","V","QVAPOR"):
    if v in ds.variables:
        a=np.asarray(ds.variables[v][:],dtype=np.float64); fin=np.isfinite(a).all()
        print(f"  {v:8s} {f.split('/')[-1]} finite={fin} min={a.min():.3g} max={a.max():.3g}")
        ok = ok and fin
print("  LAST-FRAME FINITE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 3)
PY

log "═══ CPU reference complete: ${CPU_DIR} ═══"
log "  total wall  -> cpu_wall_seconds.txt"
log "  mainloop    -> cpu_mainloop_seconds.txt  (honest per-fcst-hour basis)"
log "  summary     -> cpu_timing.json"
