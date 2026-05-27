from __future__ import annotations

import importlib.util
from pathlib import Path

from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/m7_wrfout_io_compat_audit.py"


def load_audit_module():
    spec = importlib.util.spec_from_file_location("m7_wrfout_io_compat_audit", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_writer_inventory_parses_payload_without_importing_driver():
    audit = load_audit_module()

    keys, expressions, line_span = audit.extract_payload_keys()

    assert "U10" in keys
    assert "container_note" in keys
    assert "jax.device_get(surface.u10)" in expressions["U10"]
    assert line_span[0] < line_span[1]


def test_cpu_inventory_schema_from_minimal_netcdf(tmp_path: Path):
    audit = load_audit_module()
    path = tmp_path / "wrfout_d02_2026-05-25_18:00:00"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("south_north", 2)
        dataset.createDimension("west_east", 3)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[:] = list(b"2026-05-25_18:00:00")
        u10 = dataset.createVariable("U10", "f4", ("Time", "south_north", "west_east"))
        u10.units = "m s-1"
        u10.description = "U at 10 M"

    inventory = audit.inventory_cpu_wrfout(path)

    assert inventory["schema"] == "m7_cpu_wrfout_reference_inventory_v1"
    assert inventory["variable_count"] == 2
    assert inventory["variables"]["U10"]["dimensions"] == ["Time", "south_north", "west_east"]
    assert inventory["variables"]["U10"]["units"] == "m s-1"


def test_compat_rows_document_time_dimension_deviation():
    audit = load_audit_module()
    cpu = {
        "variables": {
            "U10": {
                "dimensions": ["Time", "south_north", "west_east"],
                "dtype": "float32",
                "units": "m s-1",
            },
            "PSFC": {
                "dimensions": ["Time", "south_north", "west_east"],
                "dtype": "float32",
                "units": "Pa",
            },
        }
    }
    gpu = {
        "variables": {
            "U10": {
                "dimensions": ["south_north", "west_east"],
                "dtype": "float32",
                "units": "m s-1",
                "semantic_agreement": True,
                "container_format": "npz",
                "note": "test note",
            },
            "lead_hours": {
                "dimensions": [],
                "dtype": "float32",
                "units": "h",
                "semantic_agreement": False,
                "container_format": "npz",
                "note": "metadata",
            },
        }
    }

    rows = {row["variable"]: row for row in audit.build_compat_rows(cpu, gpu)}

    assert rows["U10"]["classification"] == "DEVIATION_DOCUMENTED"
    assert rows["U10"]["dim_agreement"] == "NO: GPU omits singleton Time dimension"
    assert rows["PSFC"]["classification"] == "MISSING_GPU"
    assert rows["lead_hours"]["classification"] == "EXTRA_GPU"
