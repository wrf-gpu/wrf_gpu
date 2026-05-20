#!/usr/bin/env python3
"""Extract WRF Thompson lookup tables into a deterministic NumPy asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WRF_ROOT = Path("/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF")
ENV_SCRIPT = Path("/home/enric/src/canairy_meteo/Gen2/artifacts/wrf_gpu_src/env_wrf_gpu.sh")
WRF_SOURCE_CANDIDATES = (
    ROOT.parent
    / "wrf_gpu"
    / "sidecar_reports"
    / "post13_thompson_first_divergence_20260508T224837Z"
    / "source_snapshots_pre"
    / "module_mp_thompson.F.pre",
    Path("/home/enric/src/wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre"),
)
SCRATCH = ROOT / "data" / "scratch" / "thompson_tables"
DEFAULT_OUTPUT = ROOT / "data" / "fixtures" / "thompson-tables-v1.npz"
TABLE_RAW = SCRATCH / "thompson_tables.raw"


TABLE_SPECS: tuple[tuple[str, tuple[int, ...], str], ...] = (
    ("r_c", (37,), "r_c"),
    ("r_i", (64,), "r_i"),
    ("r_r", (37,), "r_r"),
    ("r_s", (37,), "r_s"),
    ("r_g", (37,), "r_g"),
    ("n0r_exp", (37,), "N0r_exp"),
    ("n0g_exp", (37,), "N0g_exp"),
    ("nt_i", (55,), "Nt_i"),
    ("nt_in", (55,), "Nt_IN"),
    ("dr", (100,), "Dr"),
    ("dc", (100,), "Dc"),
    ("t_nc", (100,), "t_Nc"),
    ("t_Efrw", (100, 100), "t_Efrw"),
    ("t_Efsw", (100, 100), "t_Efsw"),
    ("tps_iaus", (64, 55), "tps_iaus"),
    ("tni_iaus", (64, 55), "tni_iaus"),
    ("tpi_ide", (64, 55), "tpi_ide"),
    ("tpi_qrfz", (37, 37, 45, 55), "tpi_qrfz"),
    ("tpg_qrfz", (37, 37, 45, 55), "tpg_qrfz"),
    ("tni_qrfz", (37, 37, 45, 55), "tni_qrfz"),
    ("tnr_qrfz", (37, 37, 45, 55), "tnr_qrfz"),
    ("snow_sa", (10,), "sa"),
    ("snow_sb", (10,), "sb"),
    ("cse", (17,), "cse"),
    ("csg", (17,), "csg"),
    ("graupel_cge", (12, 9), "cge"),
    ("graupel_cgg", (12, 9), "cgg"),
    ("am_g", (9,), "am_g"),
    ("av_g", (9,), "av_g"),
    ("bv_g", (9,), "bv_g"),
    ("rho_g", (9,), "rho_g"),
)


def _sha256(path: Path) -> str:
    """Returns the SHA-256 digest for a generated proof object."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wrf_source() -> Path:
    """Finds the WRF source snapshot named by the sprint contract."""

    for path in WRF_SOURCE_CANDIDATES:
        if path.exists():
            return path
    candidates = "\n".join(str(path) for path in WRF_SOURCE_CANDIDATES)
    raise FileNotFoundError(f"module_mp_thompson.F.pre not found; tried:\n{candidates}")


def _nvfortran() -> str:
    """Finds the NVHPC compiler needed for ABI-compatible WRF modules."""

    compiler = os.environ.get("FC") or shutil.which("nvfortran")
    if compiler:
        return compiler
    if ENV_SCRIPT.exists():
        command = (
            f"source {shlex.quote(str(ENV_SCRIPT))} >/dev/null 2>&1 || true; "
            "python - <<'PY'\n"
            "import json, os, shutil\n"
            "keys = ('PATH', 'LD_LIBRARY_PATH', 'FC', 'F77', 'CC', 'CXX', 'NETCDF', 'HDF5', 'ZLIB', 'LDFLAGS', 'CPPFLAGS', 'LIBS')\n"
            "print(json.dumps({key: os.environ.get(key, '') for key in keys} | {'NVFORTRAN': shutil.which('nvfortran') or ''}))\n"
            "PY"
        )
        proc = subprocess.run(["bash", "-lc", command], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        env = json.loads(proc.stdout)
        for key, value in env.items():
            if key != "NVFORTRAN" and value:
                os.environ[key] = value
        compiler = env.get("NVFORTRAN") or os.environ.get("FC") or shutil.which("nvfortran")
        if compiler:
            return compiler
    raise RuntimeError("nvfortran not found; cannot compile the WRF Thompson table extractor")


def _mod_dir() -> Path:
    """Returns the WRF module directory produced by the existing WRF build."""

    for path in (WRF_ROOT / "main", WRF_ROOT / "install_gen2_dmpar" / "modules"):
        if (path / "module_mp_thompson.mod").exists():
            return path
    raise FileNotFoundError("module_mp_thompson.mod not found in the WRF build tree")


def _table_writer_subroutine() -> str:
    """Fortran subroutine injected into the WRF module to expose private arrays."""

    writes = "\n".join(f"         write(unit) real({fortran_name}, kind=8)" for _name, _shape, fortran_name in TABLE_SPECS)
    return f"""

      subroutine m5_dump_thompson_tables(path)
      implicit none
      character(len=*), intent(in) :: path
      integer :: unit

      open(newunit=unit, file=trim(path), access='stream', form='unformatted', &
           status='replace', action='write')
{writes}
      close(unit)
      end subroutine m5_dump_thompson_tables
"""


def _patch_source(source: Path, target: Path) -> None:
    """Writes a build-local WRF source copy with a table-dump subroutine."""

    text = source.read_text(encoding="utf-8")
    marker = "END MODULE module_mp_thompson"
    if marker not in text:
        raise RuntimeError(f"{marker!r} not found in {source}")
    text = text.replace(marker, _table_writer_subroutine() + "\n" + marker, 1)
    target.write_text(text, encoding="utf-8")


def _write_harness(path: Path) -> None:
    """Writes the tiny program that initializes WRF tables and dumps them."""

    path.write_text(
        """program m5_extract_thompson_tables
  use module_mp_thompson, only: thompson_init, m5_dump_thompson_tables
  implicit none

  integer, parameter :: nx = 2, ny = 2, nz = 12
  integer, parameter :: ids = 1, ide = 2, jds = 1, jde = 2, kds = 1, kde = nz
  integer, parameter :: ims = 1, ime = nx, jms = 1, jme = ny, kms = 1, kme = nz
  integer, parameter :: its = 1, ite = 1, jts = 1, jte = 1, kts = 1, kte = nz
  integer :: k, narg
  character(len=512) :: output_path
  real, dimension(ims:ime,kms:kme,jms:jme) :: hgt

  narg = command_argument_count()
  if (narg /= 1) then
    write(*,'(A)') 'usage: m5_extract_thompson_tables <output.raw>'
    stop 2
  endif
  call get_command_argument(1, output_path)

  hgt = 0.0
  do k = kms, kme
    hgt(:,k,:) = real(k - 1) * 1000.0
  enddo

  call thompson_init( &
    hgt=hgt, dx=3000.0, dy=3000.0, is_start=.true., &
    ids=ids, ide=ide, jds=jds, jde=jde, kds=kds, kde=kde, &
    ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme, &
    its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte)

  call m5_dump_thompson_tables(trim(output_path))
end program m5_extract_thompson_tables

subroutine nl_get_force_read_thompson(id, value)
  implicit none
  integer, intent(in) :: id
  logical, intent(out) :: value
  value = .false.
end subroutine nl_get_force_read_thompson

subroutine nl_get_write_thompson_tables(id, value)
  implicit none
  integer, intent(in) :: id
  logical, intent(out) :: value
  value = .false.
end subroutine nl_get_write_thompson_tables

subroutine nl_get_write_thompson_mp38table(id, value)
  implicit none
  integer, intent(in) :: id
  logical, intent(out) :: value
  value = .false.
end subroutine nl_get_write_thompson_mp38table

logical function wrf_dm_on_monitor()
  implicit none
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

subroutine wrf_dm_decomp1d(nitems, first_item, last_item)
  implicit none
  integer, intent(in) :: nitems
  integer, intent(out) :: first_item, last_item
  first_item = 0
  last_item = nitems - 1
end subroutine wrf_dm_decomp1d

subroutine wrf_dm_bcast_integer(values, nitems)
  implicit none
  integer, intent(inout) :: values(*)
  integer, intent(in) :: nitems
end subroutine wrf_dm_bcast_integer

subroutine wrf_dm_bcast_double(values, nitems)
  implicit none
  double precision, intent(inout) :: values(*)
  integer, intent(in) :: nitems
end subroutine wrf_dm_bcast_double

subroutine wrf_dm_bcast_bytes(values, nitems)
  implicit none
  character(len=1), intent(inout) :: values(*)
  integer, intent(in) :: nitems
end subroutine wrf_dm_bcast_bytes

subroutine wrf_dm_gatherv(values, nitems, first_item, last_item, item_size)
  implicit none
  double precision, intent(inout) :: values(*)
  integer, intent(in) :: nitems, first_item, last_item, item_size
end subroutine wrf_dm_gatherv

real function module_dm_wrf_dm_max_real(value)
  implicit none
  real, intent(in) :: value
  module_dm_wrf_dm_max_real = value
end function module_dm_wrf_dm_max_real

subroutine module_timing_start_timing(label)
  implicit none
  character(len=*), intent(in) :: label
end subroutine module_timing_start_timing

subroutine module_timing_end_timing(label)
  implicit none
  character(len=*), intent(in) :: label
end subroutine module_timing_end_timing

subroutine wrf_abort()
  implicit none
  stop 99
end subroutine wrf_abort

subroutine wrf_debug(level, message)
  implicit none
  integer, intent(in) :: level
  character(len=*), intent(in) :: message
end subroutine wrf_debug
""",
        encoding="utf-8",
    )


def _run(command: list[str], *, cwd: Path, log: Path) -> None:
    """Runs one compiler or extractor command and appends output to the log."""

    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def _build_and_dump() -> Path:
    """Compiles the patched WRF module and writes the raw table dump."""

    SCRATCH.mkdir(parents=True, exist_ok=True)
    log = SCRATCH / "extract_thompson_tables.log"
    log.write_text("", encoding="utf-8")

    source = _wrf_source()
    patched = SCRATCH / "module_mp_thompson_tables.F90"
    patched_obj = SCRATCH / "module_mp_thompson_tables.o"
    harness_src = SCRATCH / "m5_extract_thompson_tables.f90"
    harness_obj = SCRATCH / "m5_extract_thompson_tables.o"
    harness_bin = SCRATCH / "m5_extract_thompson_tables"

    _patch_source(source, patched)
    _write_harness(harness_src)

    compiler = _nvfortran()
    mod_dir = _mod_dir()
    includes = [
        f"-I{mod_dir}",
        f"-I{WRF_ROOT / 'main'}",
        f"-I{WRF_ROOT / 'install_gen2_dmpar' / 'modules'}",
        f"-I{WRF_ROOT / 'external' / 'esmf_time_f90'}",
        f"-I{WRF_ROOT / 'install_gen2_dmpar' / 'esmf_time_f90'}",
    ]

    _run(
        [
            compiler,
            "-c",
            "-Mpreprocess",
            "-module",
            str(SCRATCH),
            *includes,
            "-o",
            str(patched_obj),
            str(patched),
        ],
        cwd=ROOT,
        log=log,
    )
    _run(
        [
            compiler,
            "-c",
            f"-I{SCRATCH}",
            *includes,
            "-o",
            str(harness_obj),
            str(harness_src),
        ],
        cwd=ROOT,
        log=log,
    )
    _run(
        [
            compiler,
            "-o",
            str(harness_bin),
            str(harness_obj),
            str(patched_obj),
            str(WRF_ROOT / "phys" / "module_mp_radar.o"),
            str(WRF_ROOT / "share" / "module_model_constants.o"),
            str(WRF_ROOT / "frame" / "module_wrf_error.o"),
        ],
        cwd=ROOT,
        log=log,
    )
    _run([str(harness_bin), str(TABLE_RAW)], cwd=ROOT, log=log)
    if not TABLE_RAW.exists():
        raise RuntimeError(f"extractor did not produce {TABLE_RAW}; see {log}")
    return TABLE_RAW


def _read_raw(path: Path) -> dict[str, np.ndarray]:
    """Reads the Fortran stream dump using the known source table shapes."""

    payload: dict[str, np.ndarray] = {}
    offset = 0
    raw = path.read_bytes()
    for name, shape, _fortran_name in TABLE_SPECS:
        count = int(np.prod(shape))
        nbytes = count * np.dtype("<f8").itemsize
        if offset + nbytes > len(raw):
            raise RuntimeError(f"raw dump ended while reading {name}")
        values = np.frombuffer(raw, dtype="<f8", count=count, offset=offset)
        payload[name] = values.reshape(shape, order="F").copy()
        offset += nbytes
    if offset != len(raw):
        raise RuntimeError(f"raw dump has {len(raw) - offset} trailing bytes")
    return payload


def extract(output: Path) -> dict[str, object]:
    """Builds WRF, extracts tables, and writes the compressed table asset."""

    raw = _build_and_dump()
    payload = _read_raw(raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    return {
        "output": str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output),
        "sha256": _sha256(output),
        "bytes": output.stat().st_size,
        "raw_bytes": raw.stat().st_size,
        "tables": {name: list(shape) for name, shape, _fortran_name in TABLE_SPECS},
        "wrf_source": str(_wrf_source()),
        "scratch_log": str((SCRATCH / "extract_thompson_tables.log").relative_to(ROOT)),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the M5-S1.x validation command."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output .npz path")
    args = parser.parse_args(argv)
    record = extract(args.output.resolve())
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
