#!/usr/bin/env bash
# v0.34 EXTENDED RUC LSM oracle: build + run against UNMODIFIED WRF
# phys/module_sf_ruclsm.F (LSMRUC -> SOILVEGIN/SFCTMP/..., RUCLSM_SOILVEGPARM,
# RUCLSMINIT) and emit per-step JSON savepoints in two precisions:
#   fp64 : -fdefault-real-8 -fdefault-double-8 (as v017)
#   fp32 : WRF default REAL=4 build (same flags without -fdefault-*)
# CPU only, every compile/run pinned `taskset -c 9 nice -n 19` (o1-ruc core).
# Build scratch: <USER_HOME>/wrf_gpu2_lanes/o1-ruc/oracle_build/v2
# Savepoints  : proofs/v034/savepoints/ruclsm/{fp64,fp32}/
#
# Usage: ruclsm_build_and_run_v2.sh [--with-checks]
#   --with-checks additionally builds diagnostic fp64/fp32 variants
#   (-finit-local-zero, -finit-real=inf, -O0, -fcheck=bounds) and writes
#   ruclsm_v2_checks.json next to the fp64 savepoint.
set -o pipefail
set -u

quiet_guard() { if [ -e /tmp/wrf_gpu2_quiet ]; then echo QUIET; exit 3; fi; }
quiet_guard

WITH_CHECKS=0
[ "${1:-}" = "--with-checks" ] && WITH_CHECKS=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRF="<USER_HOME>/src/wrf_pristine/WRF"
SRC_RUC="${WRF}/phys/module_sf_ruclsm.F"
SRC_CONST="${WRF}/share/module_model_constants.F"
SRC_ERR="${WRF}/frame/module_wrf_error.F"
TBL_DIR="${WRF}/run"
BUILD_ROOT="<USER_HOME>/wrf_gpu2_lanes/o1-ruc/oracle_build/v2"
SAVE_ROOT="$(cd "${HERE}/../.." && pwd)/savepoints/ruclsm"
PIN="taskset -c 9 nice -n 19"  # manager re-pin 2026-10-10 13:30Z (E105)
EXE=ruclsm_oracle_v2
NREG=51

set +u
source <USER_HOME>/miniconda3/etc/profile.d/conda.sh
conda activate wrfbuild
set -u
set -e
export OMP_NUM_THREADS=1
export LC_ALL=C

BASEFLAGS="-O2 -ffree-form -ffree-line-length-none -cpp -DEM_CORE=1 -ffpe-summary=none -fallow-argument-mismatch"
# DETERMINISM FLAG (no source change): WRF sfctmp's local INTEGER ilnb is
# only assigned in snowtemp/snowseaice when snhei >= snth, but both routines
# branch on it (module_sf_ruclsm.F:5716, :4410) for THIN snow too -> TSNAV
# (and, via compaction bsn, the snowpack) depends on stack garbage in WRF
# itself.  -finit-integer=0 pins ilnb<=1 (one-layer formula).  The
# -finit-integer=2 twin below marks every step that depends on it.  Checks
# show no uninitialized REAL/LOGICAL read affects any output.
DETFLAG="-finit-integer=0"
FFLAGS64="${BASEFLAGS} -fdefault-real-8 -fdefault-double-8 ${DETFLAG}"
FFLAGS32="${BASEFLAGS} ${DETFLAG}"
SENSFLAG="-finit-integer=2"

# ---------------------------------------------------------------- staging
STAGE="${BUILD_ROOT}/stage"
rm -rf "${STAGE}"
mkdir -p "${STAGE}"
cp "${SRC_CONST}" "${STAGE}/module_model_constants.F"
cp "${SRC_ERR}"   "${STAGE}/module_wrf_error.F"
cp "${SRC_RUC}"   "${STAGE}/module_sf_ruclsm.F"
cp "${TBL_DIR}/VEGPARM.TBL" "${TBL_DIR}/SOILPARM.TBL" "${TBL_DIR}/GENPARM.TBL" "${STAGE}/"
cp "${HERE}/ruclsm_oracle_v2.f90" "${STAGE}/"

# Serial single-rank WRF stubs (wrf_dm_on_monitor -> .TRUE. so the table
# reader runs; bcasts are no-ops; wrf_debug / wrf_abort as in v017).
cat > "${STAGE}/_wrfstubs.f90" <<'EOF'
logical function wrf_dm_on_monitor()
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor
subroutine wrf_dm_bcast_string(buf, n)
  character(len=*), intent(inout) :: buf
  integer, intent(in) :: n
end subroutine wrf_dm_bcast_string
subroutine wrf_dm_bcast_integer(buf, n)
  integer, intent(in) :: n
  integer, intent(inout) :: buf(*)
end subroutine wrf_dm_bcast_integer
subroutine wrf_dm_bcast_real(buf, n)
  integer, intent(in) :: n
  real, intent(inout) :: buf(*)
end subroutine wrf_dm_bcast_real
subroutine wrf_debug(level, msg)
  integer, intent(in) :: level
  character(len=*), intent(in) :: msg
end subroutine wrf_debug
subroutine wrf_abort()
  stop 1
end subroutine wrf_abort
EOF

# Pristine provenance: originals and staged copies must hash identically.
for pair in "module_sf_ruclsm.F:${SRC_RUC}" "module_model_constants.F:${SRC_CONST}" \
            "module_wrf_error.F:${SRC_ERR}" "VEGPARM.TBL:${TBL_DIR}/VEGPARM.TBL" \
            "SOILPARM.TBL:${TBL_DIR}/SOILPARM.TBL" "GENPARM.TBL:${TBL_DIR}/GENPARM.TBL"; do
  a=$(sha256sum "${STAGE}/${pair%%:*}" | cut -d' ' -f1)
  b=$(sha256sum "${pair#*:}" | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "FATAL: staged ${pair%%:*} differs from pristine"; exit 4; }
done

# ---------------------------------------------------------------- helpers
build_variant() {  # name flags...
  local name="$1"; shift
  local flags="$*"
  local dir="${BUILD_ROOT}/build_${name}"
  quiet_guard
  rm -rf "${dir}"; mkdir -p "${dir}"
  cp "${STAGE}"/* "${dir}/"
  ( cd "${dir}"
    for f in module_model_constants.F module_wrf_error.F module_sf_ruclsm.F _wrfstubs.f90 ruclsm_oracle_v2.f90; do
      quiet_guard
      ${PIN} gfortran ${flags} -c "$f"
    done
    ${PIN} gfortran ${flags} -o "${EXE}" ruclsm_oracle_v2.o module_sf_ruclsm.o \
        module_wrf_error.o module_model_constants.o _wrfstubs.o
    echo "${flags}" > fflags.txt )
}

run_variant() {  # name label [excluded regimes...] ; writes dump_<label>.txt
  # One process per regime ('only:N'): a WRF fatal (STOP 1) in one regime is
  # recorded as RUN_RC[N]=rc (+ its stderr) and cannot truncate the others.
  local name="$1" label="$2"; shift 2
  local excl=" $* "
  local dir="${BUILD_ROOT}/build_${name}"
  local all="${dir}/dump_${label}.txt"
  : > "${all}"
  local r rc
  for r in $(seq 1 "${NREG}"); do
    case "${excl}" in *" ${r} "*) continue ;; esac
    quiet_guard
    rc=0
    ( cd "${dir}" && ${PIN} "./${EXE}" "${label}" "only:${r}" > "dump_${label}_r${r}.txt" \
        2> "stderr_${label}_r${r}.txt" ) || rc=$?
    cat "${dir}/dump_${label}_r${r}.txt" >> "${all}"
    if [ "${rc}" != "0" ]; then
      sed -e '/^[[:space:]]*$/d' -e 's/^/stderr: /' "${dir}/stderr_${label}_r${r}.txt" >> "${all}"
    fi
    echo "RUN_RC[${r}]=${rc}" >> "${all}"
  done
  grep -c '^RUN_RC\[[0-9]*\]=[1-9]' "${all}" > "${dir}/nfatal_${label}.txt" || true
}

emit_savepoint() {  # name prec
  local name="$1" prec="$2"
  local dir="${BUILD_ROOT}/build_${name}"
  local out="${SAVE_ROOT}/${prec}"
  mkdir -p "${out}"
  grep -q "^NREG=${NREG}$" "${dir}/dump_${prec}.txt" || { echo "FATAL: ${prec} dump lacks NREG=${NREG}"; exit 5; }
  ${PIN} python3 -I "${HERE}/ruclsm_dump_to_json_v2.py" "${dir}/dump_${prec}.txt" \
      "${out}/ruclsm_v2_${prec}.json" "${BUILD_ROOT}/build_${name}_ilnb2/dump_${prec}.txt"
  ( cd "${STAGE}" && sha256sum module_sf_ruclsm.F module_wrf_error.F module_model_constants.F \
        VEGPARM.TBL SOILPARM.TBL GENPARM.TBL ) > "${out}/ruclsm_v2_wrf_source_checksums.txt"
  ( cd "${HERE}" && sha256sum ruclsm_oracle_v2.f90 ruclsm_dump_to_json_v2.py ruclsm_build_and_run_v2.sh ) \
      > "${out}/ruclsm_v2_driver_checksums.txt"
  {
    echo "mode=${prec}"
    echo "full_wrf_exe=false"
    echo "nreg=${NREG} nsteps=12 (per-step full-state dump; one process per regime)"
    echo "regimes_wrf_fatal=$(grep '^RUN_RC\[[0-9]*\]=[1-9]' "${dir}/dump_${prec}.txt" | tr '\n' ' ' | sed 's/ *$//')"
    echo "wrf_sources=${SRC_RUC} ${SRC_ERR} ${SRC_CONST} (unmodified copies)"
    echo "wrf_tables=${TBL_DIR}/VEGPARM.TBL ${TBL_DIR}/SOILPARM.TBL ${TBL_DIR}/GENPARM.TBL"
    echo "luse=USGS-RUC soil=STAS-RUC nlcat=24 nscat=19 iswater=16 isice=24"
    echo "compiler=$(gfortran --version | head -n 1)"
    echo "fflags=$(cat "${dir}/fflags.txt")"
    echo "determinism_flag=${DETFLAG} (uninitialized sfctmp ilnb, thin snow; see script header)"
    echo "ilnb_sensitivity_twin_fflags=$(cat "${BUILD_ROOT}/build_${name}_ilnb2/fflags.txt")"
    echo "libm=$(ldd "${dir}/${EXE}" | awk '/libm\.so/{print $3}') $(ldd --version | head -n 1)"
    echo "libgfortran=$(ldd "${dir}/${EXE}" | awk '/libgfortran/{print $3}')"
    echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
    echo "pin=${PIN}"
    echo "dump_sha256=$(sha256sum "${dir}/dump_${prec}.txt" | cut -d' ' -f1)"
    echo "json_sha256=$(sha256sum "${out}/ruclsm_v2_${prec}.json" | cut -d' ' -f1)"
    echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/ruclsm_v2_build_manifest.txt"
}

# ---------------------------------------------------------------- deliverables
build_variant fp64 ${FFLAGS64}
build_variant fp32 ${FFLAGS32}
build_variant fp64_ilnb2 ${FFLAGS64/${DETFLAG}/${SENSFLAG}}
build_variant fp32_ilnb2 ${FFLAGS32/${DETFLAG}/${SENSFLAG}}
run_variant fp64 fp64
run_variant fp32 fp32
run_variant fp64_ilnb2 fp64
run_variant fp32_ilnb2 fp32
emit_savepoint fp64 fp64
emit_savepoint fp32 fp32

# ---------------------------------------------------------------- diagnostics
if [ "${WITH_CHECKS}" = "1" ]; then
  build_variant fp64_noinit  ${FFLAGS64/${DETFLAG}/}
  build_variant fp64_initinf ${FFLAGS64} -finit-real=inf -finit-logical=true
  build_variant fp64_O0      ${FFLAGS64/-O2/-O0}
  build_variant fp64_bounds  ${FFLAGS64} -fcheck=bounds,do,mem,pointer
  build_variant fp32_noinit  ${FFLAGS32/${DETFLAG}/}
  build_variant fp32_initinf ${FFLAGS32} -finit-real=inf -finit-logical=true
  build_variant fp32_bounds  ${FFLAGS32} -fcheck=bounds,do,mem,pointer
  for v in fp64_noinit fp64_initinf fp64_O0 fp64_bounds; do run_variant "$v" fp64; done
  for v in fp32_noinit fp32_initinf fp32_bounds; do run_variant "$v" fp32; done
  quiet_guard
  ${PIN} python3 -I - "${BUILD_ROOT}" "${SAVE_ROOT}/fp64/ruclsm_v2_checks.json" <<'PY'
import json, re, sys
from pathlib import Path
root = Path(sys.argv[1]); out = Path(sys.argv[2])
reg = re.compile(r"^(?:IN_)?[A-Z][A-Z0-9_]*\[(\d+)\]")
rcre = re.compile(r"^RUN_RC\[(\d+)\]=(-?\d+)$")

def load(p):
    vals, rcs, errs = {}, {}, {}
    cur = 0
    for line in p.read_text().splitlines():
        m = rcre.match(line)
        if m:
            rcs[int(m.group(1))] = int(m.group(2)); continue
        if line.startswith("stderr: "):
            errs.setdefault(cur, []).append(line[8:]); continue
        m = reg.match(line)
        if m and "=" in line:
            cur = int(m.group(1))
            k, v = line.split("=", 1)
            vals[k] = v.strip()
    return vals, rcs, errs

res = {"note": "values compared as exact dump text (ES24.16E3); regimes listed have >=1 differing value"}
for prec in ("fp64", "fp32"):
    ref, ref_rc, _ = load(root / f"build_{prec}" / f"dump_{prec}.txt")
    res[f"{prec}_reference_fatal_regimes"] = sorted(r for r, c in ref_rc.items() if c)
    variants = [f"{prec}_ilnb2", f"{prec}_noinit", f"{prec}_initinf", f"{prec}_bounds"] + (["fp64_O0"] if prec == "fp64" else [])
    for var in variants:
        vals, rcs, errs = load(root / f"build_{var}" / f"dump_{prec}.txt")
        keys = set(ref) | set(vals)
        diff = sorted({int(reg.match(k).group(1)) for k in keys if vals.get(k) != ref.get(k)})
        res[var] = {
            "fflags": (root / f"build_{var}" / "fflags.txt").read_text().strip(),
            "fatal_regimes": sorted(r for r, c in rcs.items() if c),
            "regimes_with_text_differences": diff,
            "n_values_different": sum(1 for k in keys if vals.get(k) != ref.get(k)),
            "fatal_stderr_head": {str(r): e[:3] for r, e in errs.items()},
        }
out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
print(json.dumps(res, indent=1, sort_keys=True))
PY
fi

echo "OK: RUC LSM v2 oracle savepoints under ${SAVE_ROOT}/{fp64,fp32}"
