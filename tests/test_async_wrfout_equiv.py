"""Win #3 numerics-identical proof: async double-buffered writer == sync writer.

Writes the SAME synthetic case via the synchronous ``write_wrfout_netcdf`` and via
``prepare_wrfout_payload`` + ``AsyncWrfoutWriter`` (background thread), then asserts
every wrfout variable is byte-for-byte identical. Confirms double-buffering changes
only the wall-clock timing of the NetCDF write, not its content.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from gpuwrf.io.async_wrfout import AsyncWrfoutWriter
from gpuwrf.io.wrfout_writer import (
    MINIMUM_WRFOUT_VARIABLES,
    prepare_wrfout_payload,
    write_wrfout_netcdf,
)

from test_m7_netcdf_writer import synthetic_case


def _write_sync(tmp_path: Path, name: str):
    state, grid, namelist = synthetic_case()
    path = tmp_path / name
    write_wrfout_netcdf(
        state, grid, namelist, path,
        valid_time=datetime(2026, 5, 25, 21), lead_hours=3.0,
        run_start=datetime(2026, 5, 25, 18),
    )
    return path


def _write_async(tmp_path: Path, name: str):
    state, grid, namelist = synthetic_case()
    path = tmp_path / name
    with AsyncWrfoutWriter(max_pending=2) as writer:
        prepared = prepare_wrfout_payload(
            state, grid, namelist, path,
            valid_time=datetime(2026, 5, 25, 21), lead_hours=3.0,
            run_start=datetime(2026, 5, 25, 18),
        )
        writer.submit(prepared)
        # leaving the context joins/flushes the background write
    return path


def test_async_writer_byte_identical_to_sync(tmp_path: Path):
    p_sync = _write_sync(tmp_path, "sync.nc")
    p_async = _write_async(tmp_path, "async.nc")

    with Dataset(p_sync) as ds_a, Dataset(p_async) as ds_b:
        assert sorted(ds_a.variables) == sorted(ds_b.variables)
        for name in ds_a.variables:
            a = np.asarray(ds_a.variables[name][:])
            b = np.asarray(ds_b.variables[name][:])
            assert a.shape == b.shape, f"{name} shape differs"
            if np.issubdtype(a.dtype, np.floating):
                # bit-identical floats
                assert np.array_equal(a, b, equal_nan=True), f"{name} bytes differ"
            else:
                assert np.array_equal(a, b), f"{name} bytes differ"
        # all minimum variables present in both
        for name in MINIMUM_WRFOUT_VARIABLES:
            assert name in ds_b.variables


def test_async_writer_multiple_hours_ordering(tmp_path: Path):
    """Several hours submitted in order all land on disk after join()."""
    state, grid, namelist = synthetic_case()
    paths = []
    with AsyncWrfoutWriter(max_pending=2) as writer:
        for hour in range(1, 6):
            path = tmp_path / f"wrfout_h{hour}.nc"
            prepared = prepare_wrfout_payload(
                state, grid, namelist, path,
                valid_time=datetime(2026, 5, 25, 18 + hour), lead_hours=float(hour),
                run_start=datetime(2026, 5, 25, 18),
            )
            writer.submit(prepared)
            paths.append(path)
    for path in paths:
        assert path.exists(), f"{path} not written after join"
        with Dataset(path) as ds:
            assert "T2" in ds.variables


def test_async_writer_surfaces_write_error(tmp_path: Path):
    """A failing write is re-raised at join (fail-closed)."""
    state, grid, namelist = synthetic_case()
    # Target a path whose parent cannot be created (a file used as a directory).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    bad_path = blocker / "subdir" / "wrfout.nc"

    raised = False
    try:
        with AsyncWrfoutWriter(max_pending=2) as writer:
            prepared = prepare_wrfout_payload(
                state, grid, namelist, bad_path,
                valid_time=datetime(2026, 5, 25, 21), lead_hours=3.0,
                run_start=datetime(2026, 5, 25, 18),
            )
            writer.submit(prepared)
    except (OSError, RuntimeError, Exception):  # noqa: BLE001
        raised = True
    assert raised, "writer error was not surfaced at join"
