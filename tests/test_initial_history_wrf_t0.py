"""SI38: lead-zero history holds WRF's pre-solve values; no surface/radiation re-solve at step 0."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.integration.daily_pipeline import _M9_OUTPUT_FIELDS, _merge_output_diagnostics, finite_summary

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_m7_netcdf_writer import synthetic_case, writer_authority  # type: ignore  # noqa: E402

WN3_0227 = Path("<DATA_ROOT>/server/work/src/alisios/wrf_gen/wg_20260227_18z_a1/run/run")
INPUT = {"PSFC": 95000.0, "T2": 290.5, "TSK": 291.25, "U10": 3.5, "V10": -1.25, "Q2": 0.0075, "PBLH": 123.0}


def _wrfinput(path: Path, ny: int, nx: int) -> None:
    with Dataset(path, "w") as ds:
        ds.createDimension("Time", None)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        for name, value in INPUT.items():
            ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))[0] = np.full((ny, nx), value, np.float32)


def _writer(tmp_path: Path, resolve) -> tuple[object, object]:
    state, grid, namelist = synthetic_case()
    writer = object.__new__(pipeline._PerDomainWrfoutWriter)
    writer._surface_diagnostics_for_output = resolve
    writer._merge_output_diagnostics = _merge_output_diagnostics
    writer._finite_summary = finite_summary
    writer.writer_diagnostics = {}
    writer.writer_static_latlon_metadata = {}
    writer._async_writer = None
    writer._output_pipeline = None
    writer._variable_subset = None
    writer._full_variable_set = False
    writer.written = {"d01": []}
    writer.census_io_ledger = None
    writer._census_persist_info = {}
    writer.domain_authorities = {"d01": writer_authority(grid, "d01")}
    writer.output_dir = tmp_path / "out"
    writer.output_dir.mkdir()
    writer.input_dir = tmp_path
    writer.run_start = datetime(2026, 2, 28, 0, tzinfo=timezone.utc)
    writer.bundles = {"d01": SimpleNamespace(namelist=namelist, grid=grid)}
    writer.dt_by_domain = {"d01": 18.0}
    _wrfinput(tmp_path / "wrfinput_d01", grid.ny, grid.nx)
    return writer, state


def test_lead_zero_history_writes_pre_solve_values_without_resolve(tmp_path, monkeypatch):
    def _no_resolve(*args, **kwargs):
        raise AssertionError("surface/radiation re-solve at lead zero")

    monkeypatch.setattr(pipeline, "_noahmp_surface_diagnostics_for_output", _no_resolve)
    writer, state = _writer(tmp_path, _no_resolve)
    result = writer("d01", 0, state)
    with Dataset(result["wrfout"]) as ds:
        for name, value in INPUT.items():
            assert np.all(np.asarray(ds.variables[name][0]) == np.float32(value)), name
        # absent from wrfinput -> WRF's initialized zero, overriding the state/fallback values
        for name in ("GLW", "SWDOWN", "HFX", "LH", "LWDNB", "LWUPB", "LWUPT", "OLR", "SWNORM"):
            assert name in ds.variables, name
            assert np.all(np.asarray(ds.variables[name][0]) == 0.0), name


def test_later_history_frames_keep_the_surface_path(tmp_path):
    calls = []

    def _recording(*args, **kwargs):
        calls.append(kwargs.get("lead_seconds"))
        return None

    writer, state = _writer(tmp_path, _recording)
    writer("d01", 200, state)
    assert calls == [3600.0]


@pytest.mark.skipif(not WN3_0227.is_dir(), reason="WN3 CPU-WRF case not mounted")
@pytest.mark.parametrize("domain", ["d01", "d02", "d03"])
def test_initial_fields_equal_cpu_wrf_t0_history(domain):
    """Oracle: CPU-WRF's own lead-zero history (WN3 gate case 0227), bitwise."""
    with Dataset(WN3_0227 / f"wrfinput_{domain}") as ds:
        ny, nx = len(ds.dimensions["south_north"]), len(ds.dimensions["west_east"])
    writer = object.__new__(pipeline._PerDomainWrfoutWriter)
    writer.input_dir = WN3_0227
    writer.bundles = {domain: SimpleNamespace(grid=SimpleNamespace(ny=ny, nx=nx))}
    fields = writer._initial_surface_fields(domain)
    assert set(fields) == {wrf_name for wrf_name, _attr in _M9_OUTPUT_FIELDS} | {
        "HFX", "LH", "SWDNBC", "SWUPBC", "SWDNTC", "SWUPTC",
        "LWDNBC", "LWUPBC", "LWDNTC", "LWUPTC",
        "QKE", "CLDFRA", "QC_BL", "CLDFRA_BL", "DTAUX3D", "DTAUY3D",
        "DUSFCG", "DVSFCG",
    }
    cpu_t0 = sorted(WN3_0227.glob(f"wrfout_{domain}_*"))[0]
    with Dataset(cpu_t0) as cpu:
        for name, value in fields.items():
            if name not in cpu.variables:
                assert name in {"QC_BL", "CLDFRA_BL"} and np.all(value == 0)
                continue
            expect = np.asarray(cpu.variables[name][0])
            assert np.array_equal(np.asarray(value, dtype=expect.dtype), expect), f"{cpu_t0.name}:{name}"
