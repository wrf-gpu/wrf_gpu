"""v0.18 Thompson process oracle for RAINNC attribution.

Builds a temporary, instrumented copy of pristine WRF's Thompson module and
drives the v0.15 cold mixed-phase savepoint.  The generated WRF module dumps
selected internal process-rate arrays after WRF conservation scaling and before
the tendency apply, so the comparison is process-by-process rather than a
post-step field symptom.

Run:
  JAX_PLATFORMS=cpu PYTHONPATH=src python proofs/v018/thompson_process_oracle.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

PROOF_DIR = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
WRF = Path("<USER_HOME>/src/wrf_pristine/WRF")
WRFBUILD_PREFIX = Path("<USER_HOME>/miniconda3/envs/wrfbuild")
BUILD = Path("<DATA_ROOT>/wrf_gpu2/v018_thompson_process_oracle/build")
STATE_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v018_thompson_process_oracle/state")
PROC_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v018_thompson_process_oracle/process")
REPORT = PROOF_DIR / "thompson_process_oracle.json"
DT = 18.0

Q3D = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "pii", "p", "dz8w")

DP_TERMS = (
    "prr_wau",
    "prr_rcw",
    "pnr_wau",
    "pnr_rcr",
    "prr_rcs",
    "prs_rcs",
    "prg_rcs",
    "prr_rcg",
    "prg_rcg",
    "pnr_rcs",
    "png_rcs",
    "pnr_rcg",
    "png_rcg",
    "prs_sci",
    "pni_sci",
    "pri_rci",
    "pni_rci",
    "prr_rci",
    "pnr_rci",
    "prg_rci",
    "prg_rfz",
    "pri_rfz",
    "pnr_rfz",
    "pni_rfz",
    "pri_wfz",
    "pni_wfz",
    "pri_inu",
    "pni_inu",
    "pri_iha",
    "pni_iha",
    "pri_ide",
    "pni_ide",
    "prs_ide",
    "prs_iau",
    "pni_iau",
    "prs_scw",
    "prg_scw",
    "prg_gcw",
    "prs_ihm",
    "prg_ihm",
    "pri_ihm",
    "pni_ihm",
    "prs_sde",
    "prg_gde",
    "prr_sml",
    "prr_gml",
)

RAW_DP_TERMS = (
    "prr_wau",
    "prr_rcw",
    "prr_rcs",
    "prg_rcs",
    "prr_rcg",
    "prg_rcg",
    "prr_rci",
    "prg_rfz",
    "pri_rfz",
    "pri_wfz",
    "pri_inu",
    "pri_iha",
)

DIAG_DP_TERMS = (
    "diag_idx_r",
    "diag_idx_r1",
    "diag_idx_g",
    "diag_idx_g1",
    "diag_idx_bg",
    "diag_rr",
    "diag_rg",
    "diag_nr",
    "diag_ng",
    "diag_lamg",
    "diag_n0g_exp",
    "diag_tmr_racg",
    "diag_tcr_gacr",
    "diag_tnr_racg",
    "diag_tnr_gacr",
)

R_TERMS = (
    "qvten_stage",
    "qcten_stage",
    "qiten_stage",
    "qrten_stage",
    "qsten_stage",
    "qgten_stage",
    "niten_stage",
    "nrten_stage",
    "ngten_stage",
    "tten_stage",
    "vtrk",
    "vtnrk",
    "vtsk",
    "vtgk",
    "vtngk",
    "onstep_rain",
    "onstep_snow",
    "onstep_graupel",
    "onstep_ice",
)

SCALAR_TERMS = (
    "pptrain",
    "pptsnow",
    "pptgraul",
    "pptice",
    "pptsnow_plus_pptice",
)

PRESENT_COLD_TERMS = ("prr_rcs", "prs_rcs", "prg_rcs", "prr_rcg", "prg_rcg")
MISSING_ICE_COLLECTION_TERMS = ("prs_sci", "pri_rci", "prr_rci", "prg_rci")
PRODUCTION_TERMS = (
    "prr_wau",
    "prr_rcw",
    "pnr_wau",
    "pnr_rcr",
    "prg_rfz",
    "pri_rfz",
    "pnr_rfz",
    "pni_rfz",
    "pri_wfz",
    "pni_wfz",
    "pri_inu",
    "pni_inu",
    "pri_iha",
    "pni_iha",
    "prs_ihm",
    "prg_ihm",
    "pri_ihm",
    "pni_ihm",
)
STAGE_TENDENCY_TERMS = (
    "qvten_stage",
    "qcten_stage",
    "qiten_stage",
    "qrten_stage",
    "qsten_stage",
    "qgten_stage",
    "niten_stage",
    "nrten_stage",
    "ngten_stage",
    "tten_stage",
)
PRECIP_PARTITION_TERMS = SCALAR_TERMS


def _run(cmd: str, *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(["bash", "-lc", cmd], cwd=cwd, env=env, check=True)


def _patch_module(src: str) -> str:
    decl = "      REAL, DIMENSION(kts:kte):: sed_r,sed_s,sed_g,sed_i,sed_n,sed_c,sed_b\n"
    if decl not in src:
        raise RuntimeError("could not find mp_thompson sediment declaration")
    src = src.replace(
        decl,
        decl
        + "      REAL, DIMENSION(kts:kte):: wrfgpu2_diag\n"
        + "      DOUBLE PRECISION, DIMENSION(kts:kte):: wrfgpu2_idx_r,wrfgpu2_idx_r1\n"
        + "      DOUBLE PRECISION, DIMENSION(kts:kte):: wrfgpu2_idx_g,wrfgpu2_idx_g1,wrfgpu2_idx_bg\n"
        + "      DOUBLE PRECISION, DIMENSION(kts:kte):: wrfgpu2_rr,wrfgpu2_rg,wrfgpu2_nr,wrfgpu2_ng\n"
        + "      DOUBLE PRECISION, DIMENSION(kts:kte):: wrfgpu2_lamg,wrfgpu2_n0g_exp\n"
        + "      DOUBLE PRECISION, DIMENSION(kts:kte):: wrfgpu2_tmr_racg,wrfgpu2_tcr_gacr\n"
        + "      DOUBLE PRECISION, DIMENSION(kts:kte):: wrfgpu2_tnr_racg,wrfgpu2_tnr_gacr\n",
        1,
    )

    init_marker = (
        "!+---+-----------------------------------------------------------------+\n"
        "!..Calculate y-intercept, slope, and useful moments for snow.\n"
        "!+---+-----------------------------------------------------------------+\n"
        "      if (.not. iiwarm) then\n"
    )
    diag_init = (
        "      wrfgpu2_idx_r(:) = 0.0d0\n"
        "      wrfgpu2_idx_r1(:) = 0.0d0\n"
        "      wrfgpu2_idx_g(:) = 0.0d0\n"
        "      wrfgpu2_idx_g1(:) = 0.0d0\n"
        "      wrfgpu2_idx_bg(:) = 0.0d0\n"
        "      wrfgpu2_rr(:) = 0.0d0\n"
        "      wrfgpu2_rg(:) = 0.0d0\n"
        "      wrfgpu2_nr(:) = 0.0d0\n"
        "      wrfgpu2_ng(:) = 0.0d0\n"
        "      wrfgpu2_lamg(:) = 0.0d0\n"
        "      wrfgpu2_n0g_exp(:) = 0.0d0\n"
        "      wrfgpu2_tmr_racg(:) = 0.0d0\n"
        "      wrfgpu2_tcr_gacr(:) = 0.0d0\n"
        "      wrfgpu2_tnr_racg(:) = 0.0d0\n"
        "      wrfgpu2_tnr_gacr(:) = 0.0d0\n"
    )
    if init_marker not in src:
        raise RuntimeError("could not find process-loop diagnostic init point")
    src = src.replace(init_marker, init_marker[:-len("      if (.not. iiwarm) then\n")] + diag_init + "      if (.not. iiwarm) then\n", 1)

    diag_marker = (
        "\n!..Ice multiplication from rime-splinters (Hallet & Mossop 1974).\n"
    )
    diag_assign = (
        "          wrfgpu2_idx_r(k) = DBLE(idx_r)\n"
        "          wrfgpu2_idx_r1(k) = DBLE(idx_r1)\n"
        "          wrfgpu2_idx_g(k) = DBLE(idx_g)\n"
        "          wrfgpu2_idx_g1(k) = DBLE(idx_g1)\n"
        "          wrfgpu2_idx_bg(k) = DBLE(idx_bg(k))\n"
        "          wrfgpu2_rr(k) = DBLE(rr(k))\n"
        "          wrfgpu2_rg(k) = DBLE(rg(k))\n"
        "          wrfgpu2_nr(k) = DBLE(nr(k))\n"
        "          wrfgpu2_ng(k) = DBLE(ng(k))\n"
        "          wrfgpu2_lamg(k) = DBLE(1.0/ilamg(k))\n"
        "          if (rg(k).gt.r_g(1)) then\n"
        "             wrfgpu2_n0g_exp(k) = ogg1*DBLE(rg(k))/DBLE(am_g(idx_bg(k))) &\n"
        "                               * DBLE((1.0/ilamg(k)) * (cgg(3,1)*ogg2*ogg1)**bm_g)**DBLE(cge(1,1))\n"
        "          endif\n"
        "          wrfgpu2_tmr_racg(k) = tmr_racg(idx_g1,idx_g,idx_bg(k),idx_r1,idx_r)\n"
        "          wrfgpu2_tcr_gacr(k) = tcr_gacr(idx_g1,idx_g,idx_bg(k),idx_r1,idx_r)\n"
        "          wrfgpu2_tnr_racg(k) = tnr_racg(idx_g1,idx_g,idx_bg(k),idx_r1,idx_r)\n"
        "          wrfgpu2_tnr_gacr(k) = tnr_gacr(idx_g1,idx_g,idx_bg(k),idx_r1,idx_r)\n"
        "\n"
    )
    if diag_marker not in src:
        raise RuntimeError("could not find process-loop index diagnostic insertion point")
    src = src.replace(diag_marker, "\n" + diag_assign + diag_marker.lstrip("\n"), 1)

    raw_marker = (
        "!+---+-----------------------------------------------------------------+\n"
        "!..Ensure we do not deplete more hydrometeor species than exists.\n"
    )
    raw_calls = "".join(
        f"      call wrfgpu2_dump_process_dp('raw_{name}', {name}, kts, kte, ii, jj)\n"
        for name in RAW_DP_TERMS
    )
    raw_calls += "".join(
        f"      call wrfgpu2_dump_process_dp('{name}', wrfgpu2_{name[5:]}, kts, kte, ii, jj)\n"
        for name in DIAG_DP_TERMS
    )
    if raw_marker not in src:
        raise RuntimeError("could not find raw-process insertion point")
    src = src.replace(raw_marker, raw_calls + "\n" + raw_marker, 1)

    process_marker = (
        "      enddo\n\n"
        "!+---+-----------------------------------------------------------------+\n"
        "!..Calculate tendencies of all species but constrain the number of ice\n"
    )
    process_calls = "".join(
        f"      call wrfgpu2_dump_process_dp('{name}', {name}, kts, kte, ii, jj)\n"
        for name in DP_TERMS
    )
    if process_marker not in src:
        raise RuntimeError("could not find process-dump insertion point")
    src = src.replace(process_marker, "      enddo\n\n" + process_calls + "\n" + process_marker[13:], 1)

    tendency_marker = (
        "!+---+-----------------------------------------------------------------+\n"
        "!..Update variables for TAU+1 before condensation & sedimention.\n"
    )
    tendency_calls = (
        "      call wrfgpu2_dump_process_r('qvten_stage', qvten, kts, kte, ii, jj)\n"
        "      call wrfgpu2_dump_process_r('qcten_stage', qcten, kts, kte, ii, jj)\n"
        "      call wrfgpu2_dump_process_r('qiten_stage', qiten, kts, kte, ii, jj)\n"
        "      call wrfgpu2_dump_process_r('qrten_stage', qrten, kts, kte, ii, jj)\n"
        "      call wrfgpu2_dump_process_r('qsten_stage', qsten, kts, kte, ii, jj)\n"
        "      call wrfgpu2_dump_process_r('qgten_stage', qgten, kts, kte, ii, jj)\n"
        "      call wrfgpu2_dump_process_r('niten_stage', niten, kts, kte, ii, jj)\n"
        "      call wrfgpu2_dump_process_r('nrten_stage', nrten, kts, kte, ii, jj)\n"
        "      call wrfgpu2_dump_process_r('ngten_stage', ngten, kts, kte, ii, jj)\n"
        "      call wrfgpu2_dump_process_r('tten_stage', tten, kts, kte, ii, jj)\n\n"
    )
    if tendency_marker not in src:
        raise RuntimeError("could not find tendency-dump insertion point")
    src = src.replace(tendency_marker, tendency_calls + tendency_marker, 1)

    fall_marker = (
        "!+---+-----------------------------------------------------------------+\n"
        "!..Sedimentation of mixing ratio is the integral of v(D)*m(D)*N(D)*dD,\n"
    )
    fall_calls = (
        "      call wrfgpu2_dump_process_r('vtrk', vtrk(kts:kte), kts, kte, ii, jj)\n"
        "      call wrfgpu2_dump_process_r('vtnrk', vtnrk(kts:kte), kts, kte, ii, jj)\n"
        "      call wrfgpu2_dump_process_r('vtsk', vtsk(kts:kte), kts, kte, ii, jj)\n"
        "      call wrfgpu2_dump_process_r('vtgk', vtgk(kts:kte), kts, kte, ii, jj)\n"
        "      call wrfgpu2_dump_process_r('vtngk', vtngk(kts:kte), kts, kte, ii, jj)\n"
        "      wrfgpu2_diag(:) = onstep(1)\n"
        "      call wrfgpu2_dump_process_r('onstep_rain', wrfgpu2_diag, kts, kte, ii, jj)\n"
        "      wrfgpu2_diag(:) = onstep(2)\n"
        "      call wrfgpu2_dump_process_r('onstep_ice', wrfgpu2_diag, kts, kte, ii, jj)\n"
        "      wrfgpu2_diag(:) = onstep(3)\n"
        "      call wrfgpu2_dump_process_r('onstep_snow', wrfgpu2_diag, kts, kte, ii, jj)\n"
        "      wrfgpu2_diag(:) = onstep(4)\n"
        "      call wrfgpu2_dump_process_r('onstep_graupel', wrfgpu2_diag, kts, kte, ii, jj)\n\n"
    )
    if fall_marker not in src:
        raise RuntimeError("could not find fall-speed dump insertion point")
    src = src.replace(fall_marker, fall_calls + fall_marker, 1)

    partition_marker = (
        "!+---+-----------------------------------------------------------------+\n"
        "!.. Instantly melt any cloud ice into cloud water if above 0C and\n"
    )
    partition_calls = (
        "      call wrfgpu2_dump_process_scalar('pptrain', dble(pptrain), ii, jj)\n"
        "      call wrfgpu2_dump_process_scalar('pptsnow', dble(pptsnow), ii, jj)\n"
        "      call wrfgpu2_dump_process_scalar('pptgraul', dble(pptgraul), ii, jj)\n"
        "      call wrfgpu2_dump_process_scalar('pptice', dble(pptice), ii, jj)\n"
        "      call wrfgpu2_dump_process_scalar('pptsnow_plus_pptice', dble(pptsnow+pptice), ii, jj)\n\n"
    )
    if partition_marker not in src:
        raise RuntimeError("could not find precip-partition dump insertion point")
    src = src.replace(partition_marker, partition_calls + partition_marker, 1)

    helper = r"""

      subroutine wrfgpu2_dump_process_dp(name, arr, kts, kte, ii, jj)
      implicit none
      character(len=*), intent(in) :: name
      integer, intent(in) :: kts, kte, ii, jj
      double precision, dimension(kts:kte), intent(in) :: arr
      character(len=512) :: root, path
      integer :: u, k, nlev
      double precision, allocatable :: buf(:)
      call get_environment_variable('WRFGPU2_PROCESS_ORACLE_ROOT', root)
      if (len_trim(root) .eq. 0) return
      call execute_command_line('mkdir -p '//trim(root), wait=.true.)
      write(path,'(A,"/",A,".f64")') trim(root), trim(name)
      nlev = kte - kts + 1
      allocate(buf(nlev))
      do k = kts, kte
         buf(k-kts+1) = arr(k)
      enddo
      open(newunit=u, file=trim(path), status='unknown', position='append', &
           action='write', form='unformatted', access='stream')
      write(u) buf
      close(u)
      deallocate(buf)
      end subroutine wrfgpu2_dump_process_dp

      subroutine wrfgpu2_dump_process_r(name, arr, kts, kte, ii, jj)
      implicit none
      character(len=*), intent(in) :: name
      integer, intent(in) :: kts, kte, ii, jj
      real, dimension(kts:kte), intent(in) :: arr
      character(len=512) :: root, path
      integer :: u, k, nlev
      double precision, allocatable :: buf(:)
      call get_environment_variable('WRFGPU2_PROCESS_ORACLE_ROOT', root)
      if (len_trim(root) .eq. 0) return
      call execute_command_line('mkdir -p '//trim(root), wait=.true.)
      write(path,'(A,"/",A,".f64")') trim(root), trim(name)
      nlev = kte - kts + 1
      allocate(buf(nlev))
      do k = kts, kte
         buf(k-kts+1) = dble(arr(k))
      enddo
      open(newunit=u, file=trim(path), status='unknown', position='append', &
           action='write', form='unformatted', access='stream')
      write(u) buf
      close(u)
      deallocate(buf)
      end subroutine wrfgpu2_dump_process_r

      subroutine wrfgpu2_dump_process_scalar(name, val, ii, jj)
      implicit none
      character(len=*), intent(in) :: name
      double precision, intent(in) :: val
      integer, intent(in) :: ii, jj
      character(len=512) :: root, path
      integer :: u
      call get_environment_variable('WRFGPU2_PROCESS_ORACLE_ROOT', root)
      if (len_trim(root) .eq. 0) return
      call execute_command_line('mkdir -p '//trim(root), wait=.true.)
      write(path,'(A,"/",A,".f64")') trim(root), trim(name)
      open(newunit=u, file=trim(path), status='unknown', position='append', &
           action='write', form='unformatted', access='stream')
      write(u) val
      close(u)
      end subroutine wrfgpu2_dump_process_scalar
"""
    end_module = "END MODULE module_mp_thompson"
    if end_module not in src:
        raise RuntimeError("could not find module end")
    return src.replace(end_module, helper + "\n" + end_module, 1)


def build_and_run_wrf_oracle() -> None:
    shutil.rmtree(BUILD, ignore_errors=True)
    shutil.rmtree(STATE_ROOT, ignore_errors=True)
    shutil.rmtree(PROC_ROOT, ignore_errors=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    PROC_ROOT.mkdir(parents=True, exist_ok=True)

    module_src = (WRF / "phys" / "module_mp_thompson.F").read_text()
    (BUILD / "module_mp_thompson.F").write_text(_patch_module(module_src))
    harness_src = (REPO / "proofs" / "v015" / "cold_collection_oracle" / "coldmix_column_oracle.F").read_text()
    harness_src = harness_src.replace(
        "USE module_wrfgpu2_oracle, ONLY : oracle_open_scheme, oracle_close_scheme, &",
        "USE module_wrfgpu2_oracle, ONLY : oracle_enabled, oracle_open_scheme, oracle_close_scheme, &",
        1,
    )
    harness_src = harness_src.replace(
        "  CALL oracle_open_scheme('microphysics_coldmix','thompson','in', 1, itimestep, NCOL, NLEV, NROW)",
        "  IF (oracle_enabled()) CONTINUE\n"
        "  CALL oracle_open_scheme('microphysics_coldmix','thompson','in', 1, itimestep, NCOL, NLEV, NROW)",
        1,
    )
    (BUILD / "coldmix_column_oracle.F").write_text(harness_src)

    for name in ("qr_acr_qg_V4.dat", "qr_acr_qsV2.dat", "freezeH2O.dat", "CCN_ACTIVATE.BIN"):
        candidates = [
            WRF / "test" / "em_real" / "oracle_run" / name,
            WRF / "run" / name,
        ]
        candidates.extend(Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l2").glob(f"*/{name}"))
        for candidate in candidates:
            if candidate.exists():
                target = BUILD / name
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(candidate)
                break

    cmd = f"""
set -euo pipefail
export CONDA_PREFIX={WRFBUILD_PREFIX}
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${{LD_LIBRARY_PATH:-}}"
FC="$CONDA_PREFIX/bin/gfortran"
FLAGS="-w -ffree-form -ffree-line-length-none -fconvert=big-endian -frecord-marker=4 -O2"
INC="-I. -I{WRF}/phys -I{WRF}/frame -I{WRF}/share -I{WRF}/main -I{WRF}/inc"
taskset -c 0-3 "$FC" -c $FLAGS $INC module_mp_thompson.F -o module_mp_thompson.o
taskset -c 0-3 "$FC" -c $FLAGS $INC coldmix_column_oracle.F -o coldmix_column_oracle.o
taskset -c 0-3 "$FC" $FLAGS module_mp_thompson.o coldmix_column_oracle.o \\
  "{WRF}/main/libwrflib.a" \\
  "{WRF}/external/fftpack/fftpack5/libfftpack.a" \\
  "{WRF}/external/io_grib1/libio_grib1.a" \\
  "{WRF}/external/io_grib_share/libio_grib_share.a" \\
  "{WRF}/external/io_int/libwrfio_int.a" \\
  -L"{WRF}/external/esmf_time_f90" -lesmf_time \\
  "{WRF}/frame/module_internal_header_util.o" \\
  "{WRF}/frame/pack_utils.o" \\
  -L"{WRF}/external/io_netcdf" -lwrfio_nf \\
  -L"$CONDA_PREFIX/lib" -lnetcdff -lnetcdf \\
  -o coldmix_process_oracle.exe
WRFGPU2_ORACLE=1 WRFGPU2_ORACLE_ROOT={STATE_ROOT} WRFGPU2_PROCESS_ORACLE_ROOT={PROC_ROOT} \\
  taskset -c 0-3 ./coldmix_process_oracle.exe
"""
    _run(cmd, cwd=BUILD)


def _meta(oracle: Path) -> tuple[int, int, int]:
    side = (oracle / "thompson_in.sidecar.txt").read_text().splitlines()
    for line in side:
        if line.startswith("dims_ni_nk_nj"):
            return tuple(int(x) for x in line.split()[1:4])  # type: ignore[return-value]
    raise RuntimeError("no dims in sidecar")


def _rd_state(tag: str, name: str, ni: int, nk: int, nj: int) -> np.ndarray:
    path = STATE_ROOT / "microphysics_coldmix" / f"thompson_{tag}__{name}.f64"
    arr = np.fromfile(path, dtype=">f8")
    if name in ("rainnc", "rainncv", "snownc", "graupelnc", "sr"):
        return arr.reshape(nj, ni)
    return arr.reshape(nj, nk, ni)


def _to_cols(arr_jki: np.ndarray):
    import jax.numpy as jnp

    arr = np.moveaxis(np.asarray(arr_jki, dtype=np.float64), 1, -1)
    return jnp.asarray(np.ascontiguousarray(arr.reshape(-1, arr.shape[-1])))


def build_state(ni: int, nk: int, nj: int):
    sys.path.insert(0, str(REPO / "src"))
    import jax.numpy as jnp
    from gpuwrf.physics.thompson_column import ThompsonColumnState, density_from_pressure_temperature

    arr = {name: _rd_state("in", name, ni, nk, nj) for name in Q3D}
    temperature = _to_cols(arr["th"] * arr["pii"])
    pressure = _to_cols(arr["p"])
    qv = _to_cols(arr["qv"])
    rho = density_from_pressure_temperature(pressure, temperature, qv)
    return ThompsonColumnState(
        qv=qv,
        qc=_to_cols(arr["qc"]),
        qr=_to_cols(arr["qr"]),
        qi=_to_cols(arr["qi"]),
        qs=_to_cols(arr["qs"]),
        qg=_to_cols(arr["qg"]),
        Ni=_to_cols(arr["ni"]),
        Nr=_to_cols(arr["nr"]),
        T=temperature,
        p=pressure,
        rho=rho,
        dz=_to_cols(arr["dz8w"]),
        w=jnp.zeros_like(qv),
    )


def read_process_terms(ncols: int, nk: int) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for term in (*DP_TERMS, *(f"raw_{name}" for name in RAW_DP_TERMS), *DIAG_DP_TERMS, *R_TERMS, *SCALAR_TERMS):
        path = PROC_ROOT / f"{term}.f64"
        if not path.exists():
            continue
        arr = np.fromfile(path, dtype=">f8")
        out[term] = arr.reshape(ncols) if term in SCALAR_TERMS else arr.reshape(ncols, nk)
    return out


def jax_current_cold_collection_rates(state, dt: float) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(REPO / "src"))
    import jax.numpy as jnp
    import gpuwrf.physics.thompson_column as tc
    from gpuwrf.physics.thompson_constants import AM_R, EPS, HGFR, ORG1, R1
    from gpuwrf.physics.thompson_tables import N_R1_TABLE, N_R_TABLE, R_R_FIRST

    cold_tables = tc.COLD_COLLECTION_TABLES
    if cold_tables is None:
        raise RuntimeError("cold collection tables unavailable in JAX port")

    base = tc._reset_mp8_graupel_number(tc._clip_species(state))
    prs_sci, _pni_sci, pri_rci, _pni_rci, prr_rci, _pnr_rci, prg_rci = tc._ice_collection_rates(base, dt)

    rho = base.rho
    rr = jnp.maximum(base.qr * rho, R1)
    _, _nr_c, lamr, _ilamr, _mvd_r, _n0r, active_rain = tc._rain_distribution(base.qr, base.Nr, rho)
    rr_idx = jnp.maximum(rr, R_R_FIRST)
    n0r_exp = ORG1 * rr_idx / AM_R * lamr**tc.CRE1
    idx_r = tc._lookup_digit_index(rr_idx, -6, N_R_TABLE)
    idx_r1 = tc._lookup_digit_index(n0r_exp, 6, N_R1_TABLE)

    idx_tc = jnp.clip(jnp.floor(-(base.T - 273.15) + 0.5).astype(jnp.int32) - 1, 0, tc.N_TC_TABLE - 1)
    qrfz = tc._take_qrfz(tc.THOMPSON_TABLES.qrfz, idx_r, idx_r1, idx_tc)
    table_active = (base.T < tc.T_0) & (rr > R_R_FIRST) & (base.qr > R1)
    table_ice_mass = jnp.where(table_active, qrfz[..., 0], 0.0)
    table_graupel_mass = jnp.where(table_active, qrfz[..., 1], 0.0)
    fallback_ice_mass = jnp.where((base.T < HGFR) & (base.qr > R1) & ~table_active, rr, 0.0)
    frozen_total = (table_ice_mass + table_graupel_mass + fallback_ice_mass) / rho
    freeze_ratio = jnp.where(frozen_total > base.qr, base.qr / jnp.maximum(frozen_total, R1), 1.0)
    pri_rfz = (table_ice_mass + fallback_ice_mass) * freeze_ratio / dt
    prg_rfz = table_graupel_mass * freeze_ratio / dt

    cold_rates = tc._cold_collection_rates(base, dt, cold_tables)
    prr_rcs, prs_rcs, prg_rcs, _pnr_rcs, _png_rcs, prr_rcg, prg_rcg, _pnr_rcg = cold_rates
    rain_sump = -prg_rfz - pri_rfz - prr_rci + prr_rcs + prr_rcg
    rain_rate_max = -rr / dt
    rain_ratio = jnp.where(
        (rain_sump < rain_rate_max) & active_rain,
        rain_rate_max / jnp.minimum(rain_sump, -EPS),
        1.0,
    )
    prr_rci = prr_rci * rain_ratio
    prr_rcs, prs_rcs, prg_rcs, _pnr_rcs, _png_rcs, prr_rcg, prg_rcg, _pnr_rcg = (
        tc._scale_cold_collection_rain_rates(cold_rates, rain_ratio)
    )

    return {
        "prr_rcs": np.asarray(prr_rcs, dtype=np.float64),
        "prs_rcs": np.asarray(prs_rcs, dtype=np.float64),
        "prg_rcs": np.asarray(prg_rcs, dtype=np.float64),
        "prr_rcg": np.asarray(prr_rcg, dtype=np.float64),
        "prg_rcg": np.asarray(prg_rcg, dtype=np.float64),
        "prs_sci": np.asarray(prs_sci, dtype=np.float64),
        "pri_rci": np.asarray(pri_rci, dtype=np.float64),
        "prr_rci": np.asarray(prr_rci, dtype=np.float64),
        "prg_rci": np.asarray(prg_rci, dtype=np.float64),
    }


def jax_current_production_partition_rates(state, dt: float) -> dict[str, np.ndarray]:
    sys.path.insert(0, str(REPO / "src"))
    import jax.numpy as jnp
    import gpuwrf.physics.thompson_column as tc

    base = tc._reset_mp8_graupel_number(tc._clip_species(state))

    # Current-port warm-rain rate equivalents from _warm_rain_collection().
    _tempc, _diffu, _visco, _tcond, _lvap, _ocp, rhof, _rhof2, _vsc2 = tc._air_properties(base)
    rc, lamc, xdc, mvd_c, active_cloud = tc._cloud_distribution(base.qc, base.rho)
    _rr, nr, lamr, _ilamr, mvd_r, n0_r, active_rain = tc._rain_distribution(base.qr, base.Nr, base.rho)

    dc_g = ((tc.CCG3_NU12 * tc.OCG2_NU12) ** tc.OBMR / lamc) * 1.0e6
    dc_b = jnp.maximum(xdc**3 * dc_g**3 - xdc**6, 0.0) ** (1.0 / 6.0)
    zeta1_raw = 6.25e-6 * xdc * dc_b**3 - 0.4
    zeta1 = 0.5 * (zeta1_raw + jnp.abs(zeta1_raw))
    zeta = 0.027 * rc * zeta1
    taud_raw = 0.5 * dc_b - 7.5
    taud = 0.5 * (taud_raw + jnp.abs(taud_raw)) + tc.R1
    tau = 3.72 / jnp.maximum(rc * taud, tc.R1)
    prr_wau = jnp.where((rc > 0.01e-3) & active_cloud, jnp.minimum(rc / float(dt), zeta / tau), 0.0)
    pnr_wau = prr_wau / (tc.AM_R * tc.NU_C_MP8 * 10.0 * tc.D0R**3)

    idx_r_eff = jnp.clip(
        jnp.floor(
            tc.N_EFRW_R
            * jnp.log(jnp.maximum(mvd_r, tc.DR_FIRST) / tc.DR_FIRST)
            / jnp.log(tc.DR_LAST / tc.DR_FIRST)
        ),
        0,
        tc.N_EFRW_R - 1,
    ).astype(jnp.int32)
    idx_c_eff = jnp.clip(jnp.floor(mvd_c * 1.0e6).astype(jnp.int32) - 1, 0, tc.N_EFRW_C - 1)
    ef_rw = tc._take2(tc.THOMPSON_TABLES.t_Efrw, idx_r_eff, idx_c_eff)
    prr_rcw_raw = rhof * tc.T1_QR_QC * ef_rw * rc * n0_r * ((lamr + tc.FV_R) ** (-tc.CRE9))
    prr_rcw = jnp.where(active_rain & (mvd_r > tc.D0R) & (mvd_c > tc.D0C), prr_rcw_raw, 0.0)
    prr_rcw = jnp.minimum(jnp.maximum(rc - prr_wau * float(dt), 0.0) / float(dt), prr_rcw)
    ef_rr = 1.0 - jnp.exp(2300.0 * (mvd_r - 1950.0e-6))
    pnr_rcr = jnp.where(active_rain & (mvd_r > tc.D0R), ef_rr * 2.0 * nr * _rr, 0.0)

    state_warm = tc._warm_rain_collection(base, dt)

    # Current-port rain freezing + deposition nucleation source equivalents from
    # _ice_sources_with_process_flags().  Source-stage cloud-water freezing,
    # homogeneous/aerosol freezing, and Hallet-Mossop ice multiplication are not
    # implemented as named source terms in the current JAX body, so they stay zero
    # here and will appear as WRF-active/JAX-zero if pristine WRF uses them.
    rr0 = jnp.maximum(state_warm.qr * state_warm.rho, tc.R1)
    nr0 = jnp.maximum(state_warm.Nr * state_warm.rho, tc.R2)
    rr_for_index = jnp.maximum(rr0, tc.R_R_FIRST)
    _, _nr_rf, lamr0, _ilamr0, _mvd_r0, _n0_r0, _active_rain0 = tc._rain_distribution(
        state_warm.qr, state_warm.Nr, state_warm.rho
    )
    n0_exp = tc.ORG1 * rr_for_index / tc.AM_R * lamr0**tc.CRE1
    idx_r = tc._lookup_digit_index(rr_for_index, -6, tc.N_R_TABLE)
    idx_r1 = tc._lookup_digit_index(n0_exp, 6, tc.N_R1_TABLE)
    idx_tc = jnp.clip(jnp.floor(-(state_warm.T - 273.15) + 0.5).astype(jnp.int32) - 1, 0, tc.N_TC_TABLE - 1)
    qrfz = tc._take_qrfz(tc.THOMPSON_TABLES.qrfz, idx_r, idx_r1, idx_tc)
    table_active = (state_warm.T < tc.T_0) & (rr0 > tc.R_R_FIRST) & (state_warm.qr > tc.R1)
    table_ice = jnp.where(table_active, qrfz[..., 0] / state_warm.rho, 0.0)
    table_graupel = jnp.where(table_active, qrfz[..., 1] / state_warm.rho, 0.0)
    table_ni = jnp.where(table_active, qrfz[..., 2] / state_warm.rho, 0.0)
    table_nr = jnp.where(table_active, qrfz[..., 3] / state_warm.rho, 0.0)
    fallback_active = (state_warm.T < tc.HGFR) & (state_warm.qr > tc.R1)
    fallback_ice = jnp.where(fallback_active & ~table_active, state_warm.qr, 0.0)
    fallback_ni = jnp.where(fallback_active & ~table_active, nr0 / state_warm.rho, 0.0)
    ice_freeze = table_ice + fallback_ice
    graupel_freeze = table_graupel
    frozen_total = ice_freeze + graupel_freeze
    freeze_ratio = jnp.where(
        frozen_total > state_warm.qr,
        state_warm.qr / jnp.maximum(frozen_total, tc.R1),
        1.0,
    )
    ice_freeze = ice_freeze * freeze_ratio
    graupel_freeze = graupel_freeze * freeze_ratio
    table_ni = table_ni * freeze_ratio
    table_nr = table_nr * freeze_ratio
    pri_rfz = ice_freeze * state_warm.rho / float(dt)
    prg_rfz = graupel_freeze * state_warm.rho / float(dt)
    pni_rfz = (table_ni + fallback_ni) * state_warm.rho / float(dt)
    nr_loss = jnp.minimum(state_warm.Nr, table_nr + table_ni + fallback_ni)
    pnr_rfz = nr_loss * state_warm.rho / float(dt)
    cloud_freeze, cloud_ni, pri_wfz, pni_wfz = tc._cloud_water_freezing_rates(
        state_warm,
        dt,
        tc.COLD_COLLECTION_TABLES if tc._cold_collection_enabled() else None,
    )

    ocp = tc.cp_inverse(state_warm.qv)
    lvap = tc.latent_heat_vaporization(state_warm.T)
    lfus2 = tc.LSUB - lvap
    state_rf = state_warm.replace(
        qr=state_warm.qr - ice_freeze - graupel_freeze,
        qc=state_warm.qc - cloud_freeze,
        qi=state_warm.qi + ice_freeze + cloud_freeze,
        qg=state_warm.qg + graupel_freeze,
        Ni=state_warm.Ni + table_ni + fallback_ni + cloud_ni,
        Nr=jnp.maximum(0.0, state_warm.Nr - nr_loss),
        T=state_warm.T + lfus2 * ocp * (ice_freeze + graupel_freeze + cloud_freeze),
    )
    state_rf = state_rf.replace(rho=tc.density_from_pressure_temperature(state_rf.p, state_rf.T, state_rf.qv))
    qvsi_freeze = tc.saturation_mixing_ratio_ice(state_rf.p, state_rf.T)
    qvsw_freeze = tc.saturation_mixing_ratio_liquid(state_rf.p, state_rf.T)
    ssati_freeze = state_rf.qv / qvsi_freeze - 1.0
    ssatw_freeze = state_rf.qv / qvsw_freeze - 1.0
    deposition_nucleation_active = (state_rf.T < tc.T_0) & (
        (ssati_freeze >= 0.25) | ((ssatw_freeze > tc.EPS) & (state_rf.T < 253.15))
    )
    xnc = jnp.minimum(250.0e3, tc.TNO * jnp.exp(tc.ATO * (tc.T_0 - state_rf.T)))
    xni = state_rf.Ni * state_rf.rho
    pni_inu = jnp.maximum(xnc - xni, 0.0) / float(dt)
    vapor_rate_max = jnp.maximum(0.0, (state_rf.qv - qvsi_freeze) * state_rf.rho / float(dt) * 0.999)
    pri_inu = jnp.where(deposition_nucleation_active, jnp.minimum(vapor_rate_max, tc.XM0I * pni_inu), 0.0)
    pni_inu = jnp.where(deposition_nucleation_active, pri_inu / tc.XM0I, 0.0)

    cold_rates = (
        tc._cold_collection_rates(state_warm, dt, tc.COLD_COLLECTION_TABLES)
        if tc._cold_collection_enabled()
        else tc._zero_cold_collection_rates(state_warm)
    )
    state_stage, _graupel_melt, _vts_boost, cold_rates = tc._ice_sources_with_process_flags(
        state_warm, dt, cold_collection_rates=cold_rates
    )
    if tc._cold_collection_enabled():
        state_stage = tc._apply_cold_collection_rates(state_stage, dt, cold_rates)

    tendencies = {
        "qvten_stage": (state_stage.qv - base.qv) / float(dt),
        "qcten_stage": (state_stage.qc - base.qc) / float(dt),
        "qiten_stage": (state_stage.qi - base.qi) / float(dt),
        "qrten_stage": (state_stage.qr - base.qr) / float(dt),
        "qsten_stage": (state_stage.qs - base.qs) / float(dt),
        "qgten_stage": (state_stage.qg - base.qg) / float(dt),
        "niten_stage": (state_stage.Ni - base.Ni) / float(dt),
        "nrten_stage": (state_stage.Nr - base.Nr) / float(dt),
        "ngten_stage": (state_stage.Ng - base.Ng) / float(dt),
        "tten_stage": (state_stage.T - base.T) / float(dt),
    }

    _out, precip = tc.step_thompson_column_with_precip(base, dt, debug=False)
    zero = jnp.zeros_like(prr_wau)
    jax_rates = {
        "prr_wau": prr_wau,
        "prr_rcw": prr_rcw,
        "pnr_wau": pnr_wau,
        "pnr_rcr": pnr_rcr,
        "prg_rfz": prg_rfz,
        "pri_rfz": pri_rfz,
        "pnr_rfz": pnr_rfz,
        "pni_rfz": pni_rfz,
        "pri_wfz": pri_wfz,
        "pni_wfz": pni_wfz,
        "pri_inu": pri_inu,
        "pni_inu": pni_inu,
        "pri_iha": zero,
        "pni_iha": zero,
        "prs_ihm": zero,
        "prg_ihm": zero,
        "pri_ihm": zero,
        "pni_ihm": zero,
        **tendencies,
        "pptrain": precip["rain"],
        "pptsnow": precip["snow"],
        "pptgraul": precip["graupel"],
        "pptice": precip["ice"],
        "pptsnow_plus_pptice": precip["snow"] + precip["ice"],
    }
    return {name: np.asarray(value, dtype=np.float64) for name, value in jax_rates.items()}


def _summary(arr: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(arr, dtype=np.float64)
    active = np.abs(arr) > 1.0e-20
    return {
        "active_cells": int(active.sum()),
        "sum": float(arr.sum()),
        "abs_sum": float(np.abs(arr).sum()),
        "max_abs": float(np.abs(arr).max()) if arr.size else 0.0,
    }


def _diff_summary(wrf: np.ndarray, jax_arr: np.ndarray) -> dict[str, float | int]:
    diff = np.asarray(jax_arr, dtype=np.float64) - np.asarray(wrf, dtype=np.float64)
    denom = float(np.abs(wrf).sum())
    return {
        "wrf_abs_sum": denom,
        "jax_abs_sum": float(np.abs(jax_arr).sum()),
        "diff_abs_sum": float(np.abs(diff).sum()),
        "l1_rel_to_wrf": float(np.abs(diff).sum() / denom) if denom > 0.0 else None,
        "wrf_sum": float(np.asarray(wrf).sum()),
        "jax_sum": float(np.asarray(jax_arr).sum()),
        "active_wrf_cells": int((np.abs(wrf) > 1.0e-20).sum()),
        "active_jax_cells": int((np.abs(jax_arr) > 1.0e-20).sum()),
    }


def main() -> None:
    build_and_run_wrf_oracle()
    oracle_dir = STATE_ROOT / "microphysics_coldmix"
    ni, nk, nj = _meta(oracle_dir)
    ncols = ni * nj
    terms = read_process_terms(ncols, nk)
    state = build_state(ni, nk, nj)
    jax_rates = {
        **jax_current_cold_collection_rates(state, DT),
        **jax_current_production_partition_rates(state, DT),
    }

    wrf_summaries = {name: _summary(arr) for name, arr in terms.items()}
    production_set = set(PRODUCTION_TERMS) | set(STAGE_TENDENCY_TERMS) | set(PRECIP_PARTITION_TERMS)
    compare_terms = sorted(set(PRESENT_COLD_TERMS) | set(MISSING_ICE_COLLECTION_TERMS) | production_set)
    comparisons = {
        name: _diff_summary(terms[name], jax_rates[name])
        for name in compare_terms
        if name in terms and name in jax_rates
    }
    zero_gpu_active = []
    for name, item in comparisons.items():
        if float(item["wrf_abs_sum"]) > 1.0e-20 and float(item["jax_abs_sum"]) <= 1.0e-30:
            zero_gpu_active.append({"term": name, **_summary(terms[name])})
    zero_gpu_active.sort(key=lambda item: float(item["abs_sum"]), reverse=True)

    rci_sci_terms = ("prs_sci", "pri_rci", "prr_rci", "prg_rci")
    rci_sci_match = all(
        name in comparisons
        and comparisons[name]["l1_rel_to_wrf"] is not None
        and float(comparisons[name]["l1_rel_to_wrf"]) < 1.0e-5
        for name in rci_sci_terms
    )
    rcg_match = all(
        name in comparisons
        and comparisons[name]["l1_rel_to_wrf"] is not None
        and float(comparisons[name]["l1_rel_to_wrf"]) < 1.0e-5
        for name in ("prr_rcg", "prg_rcg")
    )
    divergent_terms = [
        {
            "term": name,
            "wrf_abs_sum": item["wrf_abs_sum"],
            "jax_abs_sum": item["jax_abs_sum"],
            "l1_rel_to_wrf": item["l1_rel_to_wrf"],
            "diff_abs_sum": item["diff_abs_sum"],
        }
        for name, item in comparisons.items()
        if (
            (item["l1_rel_to_wrf"] is not None and float(item["l1_rel_to_wrf"]) >= 1.0e-5)
            or (float(item["wrf_abs_sum"]) <= 1.0e-20 and float(item["diff_abs_sum"]) > 1.0e-20)
        )
    ]
    divergent_terms.sort(key=lambda item: float(item["wrf_abs_sum"]), reverse=True)
    production_divergent_terms = [
        item for item in divergent_terms if str(item["term"]) in production_set
    ]

    rain_process_abs = {
        name: wrf_summaries[name]["abs_sum"]
        for name in (
            "prr_wau",
            "prr_rcw",
            "pri_wfz",
            "pri_inu",
            "pri_iha",
            "prr_rcs",
            "prr_rcg",
            "prr_rci",
            "pri_rci",
            "prg_rci",
            "prs_sci",
            "pptrain",
            "pptsnow_plus_pptice",
            "pptgraul",
        )
        if name in wrf_summaries
    }
    out = {
        "state_oracle": str(oracle_dir),
        "process_oracle": str(PROC_ROOT),
        "build_dir": str(BUILD),
        "n_columns": ncols,
        "n_levels": nk,
        "dt_s": DT,
        "wrf_process_summaries": wrf_summaries,
        "jax_current_vs_wrf_process": comparisons,
        "wrf_active_jax_zero_terms": zero_gpu_active,
        "rci_sci_terms_match_wrf": rci_sci_match,
        "rcg_terms_match_wrf": rcg_match,
        "remaining_divergent_compared_terms": divergent_terms,
        "production_partition_compared_terms": sorted(name for name in production_set if name in comparisons),
        "production_partition_divergent_terms": production_divergent_terms,
        "rainnc_relevant_wrf_abs_process_mass_rates": rain_process_abs,
        "verdict": (
            "rci/sci/rcg process family matches pristine WRF within 1e-5 L1-relative; "
            "bounded production/partition pass is summarized in production_partition_divergent_terms"
            if rci_sci_match and rcg_match
            else "cold collision family still diverges; inspect remaining_divergent_compared_terms"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(out, indent=2))

    print("=== v018 Thompson process oracle ===")
    print(f"state: {oracle_dir}")
    print(f"process: {PROC_ROOT}")
    print(f"wrote: {REPORT}")
    for name in compare_terms:
        if name in comparisons:
            c = comparisons[name]
            print(
                f"{name:22s} WRF_abs={c['wrf_abs_sum']:.6e} "
                f"JAX_abs={c['jax_abs_sum']:.6e} L1rel={c['l1_rel_to_wrf']}"
            )
    if zero_gpu_active:
        print("WRF-active JAX-zero terms:")
        for item in zero_gpu_active:
            print(f"  {item['term']}: abs_sum={item['abs_sum']:.6e}, active={item['active_cells']}")
    else:
        print("WRF-active JAX-zero terms: none among compared terms")
    print(f"rci/sci match: {rci_sci_match}")
    print(f"rcg match: {rcg_match}")


if __name__ == "__main__":
    main()
