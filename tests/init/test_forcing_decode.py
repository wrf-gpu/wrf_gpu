from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpuwrf.init.aifs_grib import ISOBARIC_LEVELS_PA, WPS_SURFACE_LEVEL
from gpuwrf.init.forcing_decode import (
    METEM_SOIL_LAYER_ORDER_CM,
    SURFACE_SPECHUMD_WPS_FILL,
    build_forcing_decode_report,
    decode_forcing,
    physical_range_checks,
    specific_humidity_from_dewpoint,
)


CASE = "20260428_18z_72h"
UNGRIB_DIR = Path("/mnt/data/canairy_meteo/runs/wps_cases") / CASE / "ungrib"
GRIB = UNGRIB_DIR / "step_000.grib2"
INTERMEDIATE = UNGRIB_DIR / "AIFS:2026-04-28_18"


def _require_fixture(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"required local fixture is missing: {path}")


def test_decode_forcing_assembles_14_level_native_source_arrays():
    _require_fixture(GRIB)

    decoded = decode_forcing(GRIB)
    ny, nx = decoded.source_grid.shape

    assert decoded.valid_time == "2026-04-28_18:00:00"
    assert decoded.isobaric_levels_pa == tuple(ISOBARIC_LEVELS_PA)
    assert decoded.soil_layer_order_cm == METEM_SOIL_LAYER_ORDER_CM
    for name in ("TT", "UU", "VV", "GHT", "SPECHUMD", "PRES"):
        assert decoded.arrays[name].shape == (14, ny, nx)
        assert decoded.arrays[name].dtype == np.float32
    assert decoded.arrays["PRES"][0].shape == (ny, nx)
    assert np.array_equal(decoded.arrays["PRES"][0], decoded.arrays["PSFC"])
    for idx, level in enumerate(ISOBARIC_LEVELS_PA, start=1):
        assert np.all(decoded.arrays["PRES"][idx] == np.float32(level))

    assert float(np.nanmax(decoded.arrays["GHT"][13])) > 15000.0
    assert float(np.nanmax(decoded.arrays["GHT"][13])) < 25000.0
    assert float(np.nanmax(decoded.arrays["SOILHGT"])) < 9000.0


def test_decode_forcing_uses_oracle_surface_spechumd_fill_by_default():
    _require_fixture(GRIB)

    decoded = decode_forcing(GRIB)

    assert decoded.surface_spechumd_policy == "wps_fill"
    assert np.all(decoded.arrays["SPECHUMD"][0] == SURFACE_SPECHUMD_WPS_FILL)
    assert float(np.nanmin(decoded.arrays["SPECHUMD"][1:])) >= 0.0
    assert float(np.nanmax(decoded.arrays["SPECHUMD"][1:])) < 0.03

    dewpoint = specific_humidity_from_dewpoint(decoded.arrays["DEWPT"], decoded.arrays["PSFC"])
    assert float(np.nanmin(dewpoint)) > 0.0
    assert float(np.nanmax(dewpoint)) < 0.03
    assert not np.array_equal(decoded.arrays["SPECHUMD"][0], dewpoint)


def test_decode_forcing_can_emit_contract_dewpoint_surface_spechumd_policy():
    _require_fixture(GRIB)

    decoded = decode_forcing(GRIB, surface_spechumd_policy="dewpoint")

    assert decoded.surface_spechumd_policy == "dewpoint"
    assert float(np.nanmin(decoded.arrays["SPECHUMD"])) >= 0.0
    assert float(np.nanmax(decoded.arrays["SPECHUMD"])) < 0.03
    assert np.array_equal(
        decoded.arrays["SPECHUMD"][0],
        specific_humidity_from_dewpoint(decoded.arrays["DEWPT"], decoded.arrays["PSFC"]),
    )


def test_soil_named_layers_stack_in_met_em_oracle_order():
    _require_fixture(GRIB)

    decoded = decode_forcing(GRIB)

    assert np.array_equal(decoded.arrays["ST"][0], decoded.arrays["ST010040"])
    assert np.array_equal(decoded.arrays["ST"][1], decoded.arrays["ST000010"])
    assert np.array_equal(decoded.arrays["SM"][0], decoded.arrays["SM010040"])
    assert np.array_equal(decoded.arrays["SM"][1], decoded.arrays["SM000010"])
    assert np.all(decoded.arrays["SOIL_LAYERS"][0] == np.float32(40.0))
    assert np.all(decoded.arrays["SOIL_LAYERS"][1] == np.float32(10.0))


def test_specific_humidity_from_dewpoint_matches_declared_formula():
    dewpoint = np.array([[293.15]], dtype=np.float32)
    pressure = np.array([[100000.0]], dtype=np.float32)

    got = specific_humidity_from_dewpoint(dewpoint, pressure)

    e = 611.2 * np.exp(17.67 * (293.15 - 273.15) / (293.15 - 29.65))
    expected = 0.622 * e / (100000.0 - (1.0 - 0.622) * e)
    assert abs(float(got[0, 0]) - float(expected)) <= 1.0e-8


def test_physical_ranges_and_wps_intermediate_oracle_report_pass():
    _require_fixture(GRIB)
    _require_fixture(INTERMEDIATE)

    decoded = decode_forcing(GRIB)
    checks = physical_range_checks(decoded)
    assert checks["TT_all_180_330_K"]["pass"] is True
    assert checks["UU_abs_le_120_mps"]["pass"] is True
    assert checks["VV_abs_le_120_mps"]["pass"] is True
    assert checks["SPECHUMD_surface_wps_sentinel"]["pass"] is True
    assert checks["SPECHUMD_isobaric_0_0p03_kgkg"]["pass"] is True
    assert checks["PSFC_50000_105000_Pa"]["pass"] is False
    assert checks["ST_220_330_K"]["pass"] is False

    report = build_forcing_decode_report(GRIB, intermediate_path=INTERMEDIATE)
    compare = report["wps_intermediate_compare"]
    assert report["overall_pass"] is True
    assert report["oracle_fidelity_pass"] is True
    assert report["strict_contract_physical_ranges_pass"] is False
    assert compare["all_pass"] is True
    assert compare["all_bit_equal"] is True
    assert compare["record_count"] == 78
    assert len(compare["comparisons"]) == 78
    assert any(item["field"] == "GHT" and item["wps_field"] == "HGT" for item in compare["comparisons"])
    assert report["array_ranges"]["SPECHUMD"]["per_level_minmax"][0] == [-1.0, -1.0]
