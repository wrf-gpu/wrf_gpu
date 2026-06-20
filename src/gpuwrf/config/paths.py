"""Environment-overridable filesystem locations for gpuwrf.

Historically the runnable path hardcoded a single workstation layout rooted at
``<DATA_ROOT>/canairy_meteo`` (the Canary Gen2 / CPU-WRF backfill corpus) and a few
``<USER_HOME>`` build paths. That makes a clean clone un-runnable: a naive agent
following ``README.md`` only has no such directories.

This module centralizes every such location behind a small set of environment
variables with sane, checkout-relative defaults. CLI arguments remain the
authoritative source (``gpuwrf run --input-dir ...`` passes an absolute path that
bypasses these defaults entirely); these helpers are the *fallback* used by
library code, scripts, and tests.

Precedence everywhere is: explicit CLI argument > environment variable > default.

Environment variables
----------------------
``GPUWRF_CANAIRY_ROOT``
    Root of the Canary Gen2 / CPU-WRF corpus. Default: ``<repo>/data/canairy_meteo``
    in an editable checkout. On the user's workstation set it to
    ``<DATA_ROOT>/canairy_meteo`` to keep the original layout working.
``GPUWRF_RUN_ROOT``
    Override the L3 run root directly. Default: ``<canairy_root>/runs/wrf_l3``.
``GPUWRF_L2_RUN_ROOT``
    Override the L2 run root directly. Default: ``<canairy_root>/runs/wrf_l2``.
``GPUWRF_AEMET_ROOT``
    Station-observation data (only needed for optional ``--score``).
    Default: ``<canairy_root>/artifacts/datasets/aemet_stations``.
``GPUWRF_MPIRUN``
    Path to the CPU-WRF ``mpirun`` (only needed by the CPU baseline helper).
``GPUWRF_WRF_EXE``
    Path to the CPU-WRF ``wrf.exe`` (only needed by the CPU baseline helper).
``GPUWRF_WRF_ROOT``
    Root of a pristine WRF v4 source/run tree. Used to read the unmodified WRF
    runtime tables that gpuwrf parses at startup: Noah-MP ``run/MPTABLE.TBL`` /
    ``SOILPARM.TBL`` / ``GENPARM.TBL`` (``sf_surface_physics=4``) and the classic
    RRTM longwave ``run/RRTM_DATA[_DBL]`` + ``phys/module_ra_rrtm.F``
    (``ra_lw_physics=1``). Default: ``<repo>/data/wrf_pristine/WRF``. On the
    workstation set it to e.g. ``<USER_HOME>/<you>/src/wrf_pristine/WRF``.
``GPUWRF_AIFS_VTABLE``
    Path to the ``Vtable.AIFS_PURE`` GRIB2 Vtable (only needed by the AIFS GRIB
    ingest). Default: ``<canairy_root>/Gen2/configs/Vtable.AIFS_PURE``.
``GPUWRF_TMPDIR``
    Scratch root for replay traces / cached outputs.
    Default: ``~/.cache/gpuwrf``.
``GPUWRF_JAX_CACHE_DIR`` / ``JAX_COMPILATION_CACHE_DIR``
    XLA compilation cache directory. Default: ``<repo>/.gpuwrf-cache/jax``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "repo_root",
    "data_root",
    "canairy_root",
    "wrf_l3_root",
    "wrf_l2_root",
    "aemet_root",
    "jax_cache_dir",
    "mpirun_path",
    "wrf_exe_path",
    "wrf_root",
    "wrf_run_dir",
    "aifs_vtable_path",
    "tmp_root",
]


def _env_path(name: str) -> Path | None:
    """Return ``os.environ[name]`` as an expanded ``Path``, or ``None`` if unset/empty."""
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return Path(value).expanduser()


def repo_root() -> Path:
    """Repository root of the editable checkout (``src/gpuwrf/config/paths.py`` -> repo)."""
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """Checkout-relative ``data/`` directory used for clean-clone sample cases."""
    return repo_root() / "data"


def canairy_root() -> Path:
    """Root of the Canary Gen2 / CPU-WRF corpus.

    ``GPUWRF_CANAIRY_ROOT`` overrides; default is ``<repo>/data/canairy_meteo``.
    """
    return _env_path("GPUWRF_CANAIRY_ROOT") or (data_root() / "canairy_meteo")


def wrf_l3_root() -> Path:
    """L3 (1 km) run root. ``GPUWRF_RUN_ROOT`` overrides."""
    return _env_path("GPUWRF_RUN_ROOT") or (canairy_root() / "runs" / "wrf_l3")


def wrf_l2_root() -> Path:
    """L2 (3 km) run root. ``GPUWRF_L2_RUN_ROOT`` overrides."""
    return _env_path("GPUWRF_L2_RUN_ROOT") or (canairy_root() / "runs" / "wrf_l2")


def aemet_root() -> Path:
    """AEMET station-observation root (optional ``--score`` only)."""
    return _env_path("GPUWRF_AEMET_ROOT") or (
        canairy_root() / "artifacts" / "datasets" / "aemet_stations"
    )


def jax_cache_dir() -> Path:
    """XLA compilation cache directory.

    Prefers ``GPUWRF_JAX_CACHE_DIR``, then ``JAX_COMPILATION_CACHE_DIR``, else a
    checkout-relative ``<repo>/.gpuwrf-cache/jax`` so a fresh clone never writes
    its cache into a private ``/mnt`` path.
    """
    return (
        _env_path("GPUWRF_JAX_CACHE_DIR")
        or _env_path("JAX_COMPILATION_CACHE_DIR")
        or (repo_root() / ".gpuwrf-cache" / "jax")
    )


def mpirun_path() -> Path | None:
    """CPU-WRF ``mpirun`` path (``GPUWRF_MPIRUN``), or ``None`` if unset."""
    return _env_path("GPUWRF_MPIRUN")


def wrf_exe_path() -> Path | None:
    """CPU-WRF ``wrf.exe`` path (``GPUWRF_WRF_EXE``), or ``None`` if unset."""
    return _env_path("GPUWRF_WRF_EXE")


def wrf_root() -> Path:
    """Root of a pristine WRF v4 source/run tree.

    gpuwrf parses a few *unmodified* WRF runtime tables at startup (Noah-MP
    ``run/MPTABLE.TBL`` etc. for ``sf_surface_physics=4`` and the classic RRTM
    longwave ``run/RRTM_DATA[_DBL]`` for ``ra_lw_physics=1``). ``GPUWRF_WRF_ROOT``
    overrides; default is ``<repo>/data/wrf_pristine/WRF`` for a clean checkout.
    """
    return _env_path("GPUWRF_WRF_ROOT") or (data_root() / "wrf_pristine" / "WRF")


def wrf_run_dir() -> Path:
    """Pristine WRF ``run/`` directory holding the runtime tables."""
    return wrf_root() / "run"


def aifs_vtable_path() -> Path:
    """``Vtable.AIFS_PURE`` GRIB2 Vtable used by the AIFS ingest.

    ``GPUWRF_AIFS_VTABLE`` overrides; default is
    ``<canairy_root>/Gen2/configs/Vtable.AIFS_PURE``.
    """
    return _env_path("GPUWRF_AIFS_VTABLE") or (
        canairy_root() / "Gen2" / "configs" / "Vtable.AIFS_PURE"
    )


def tmp_root() -> Path:
    """Scratch root for replay traces / cached outputs.

    ``GPUWRF_TMPDIR`` overrides; default is ``~/.cache/gpuwrf`` so a fresh clone
    never writes scratch into a private hardcoded ``<USER_HOME>/<name>`` path.
    """
    return _env_path("GPUWRF_TMPDIR") or (Path.home() / ".cache" / "gpuwrf")
