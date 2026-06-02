from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpuwrf.init.aifs_grib import (
    DEFAULT_AIFS_VTABLE,
    ISOBARIC_LEVELS_PA,
    WPS_MEAN_SEA_LEVEL,
    WPS_SURFACE_LEVEL,
    parse_aifs_vtable,
    read_aifs_grib,
    read_wps_intermediate,
)


CASE = "20260428_18z_72h"
UNGRIB_DIR = Path("/mnt/data/canairy_meteo/runs/wps_cases") / CASE / "ungrib"
GRIB = UNGRIB_DIR / "step_000.grib2"
INTERMEDIATE = UNGRIB_DIR / "AIFS:2026-04-28_18"


def _require_fixture(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"required local fixture is missing: {path}")


def test_vtable_parser_preserves_grib2_triplets_and_soil_depths():
    _require_fixture(DEFAULT_AIFS_VTABLE)

    entries = parse_aifs_vtable(DEFAULT_AIFS_VTABLE)

    assert len(entries) == 18
    assert any(
        e.metgrid_name == "SPECHUMD"
        and (e.discipline, e.category, e.parameter, e.grib2_level_code) == (0, 1, 0, 100)
        for e in entries
    )
    soil_moisture = [e for e in entries if e.metgrid_name.startswith("SM")]
    assert [(e.metgrid_name, e.soil_depth_mm) for e in soil_moisture] == [
        ("SM000010", (0, 1000)),
        ("SM010040", (1000, 4000)),
    ]
    assert all((e.discipline, e.category, e.parameter, e.grib2_level_code) == (2, 0, 192, 106) for e in soil_moisture)


def test_read_aifs_grib_decodes_all_real_messages_by_vtable_key():
    _require_fixture(GRIB)

    decoded = read_aifs_grib(GRIB)

    assert len(decoded.messages) == 78
    assert decoded.valid_time == "2026-04-28_18:00:00"
    assert decoded.grid.shape == (721, 1440)
    assert decoded.grid.grid_type == "regular_ll"
    assert decoded.grid.latitude_first == 90.0
    assert decoded.grid.latitude_last == -90.0
    assert decoded.grid.longitude_first == 180.0
    assert decoded.grid.longitude_last == 179.75
    assert decoded.levels_pa_for("TT") == tuple(ISOBARIC_LEVELS_PA)
    assert decoded.levels_pa_for("UU") == tuple(ISOBARIC_LEVELS_PA)
    assert decoded.levels_pa_for("VV") == tuple(ISOBARIC_LEVELS_PA)
    assert decoded.levels_pa_for("SPECHUMD") == tuple(ISOBARIC_LEVELS_PA)
    assert decoded.levels_pa_for("GHT") == tuple(ISOBARIC_LEVELS_PA)

    sm000010 = decoded.get("SM000010", soil_depth_mm=(0, 1000))
    sm010040 = decoded.get("SM010040", soil_depth_mm=(1000, 4000))
    assert sm000010.short_name == "unknown"
    assert sm010040.short_name == "unknown"
    assert sm000010.entry.line_number == 18
    assert sm010040.entry.line_number == 19


def test_decoded_grib_values_match_ungrib_intermediate_bit_exact_for_direct_fields():
    _require_fixture(GRIB)
    _require_fixture(INTERMEDIATE)

    decoded = read_aifs_grib(GRIB)
    records = read_wps_intermediate(INTERMEDIATE, fields={"TT", "HGT", "SM010040", "PMSL"})
    idx = {(rec.field_name, rec.level): rec for rec in records}

    assert np.array_equal(decoded.get("TT", wps_level=WPS_SURFACE_LEVEL, height_m=2.0).data, idx[("TT", WPS_SURFACE_LEVEL)].data)
    assert np.array_equal(decoded.get("GHT", level_pa=5000.0).data, idx[("HGT", 5000.0)].data)
    assert np.array_equal(decoded.get("PMSL", wps_level=WPS_MEAN_SEA_LEVEL).data, idx[("PMSL", WPS_MEAN_SEA_LEVEL)].data)
    assert np.array_equal(decoded.get("SM010040", soil_depth_mm=(1000, 4000)).data, idx[("SM010040", WPS_SURFACE_LEVEL)].data)


def test_wps_intermediate_reader_inventory_matches_recon():
    _require_fixture(INTERMEDIATE)

    records = read_wps_intermediate(INTERMEDIATE)
    by_field: dict[str, list[float]] = {}
    for record in records:
        by_field.setdefault(record.field_name, []).append(record.level)

    assert len(records) == 78
    assert by_field["TT"] == [WPS_SURFACE_LEVEL, *ISOBARIC_LEVELS_PA]
    assert by_field["UU"] == [WPS_SURFACE_LEVEL, *ISOBARIC_LEVELS_PA]
    assert by_field["VV"] == [WPS_SURFACE_LEVEL, *ISOBARIC_LEVELS_PA]
    assert by_field["HGT"] == list(ISOBARIC_LEVELS_PA)
    assert by_field["SPECHUMD"] == list(ISOBARIC_LEVELS_PA)
    assert by_field["PMSL"] == [WPS_MEAN_SEA_LEVEL]
    assert by_field["ST000010"] == [WPS_SURFACE_LEVEL]
    assert by_field["SM010040"] == [WPS_SURFACE_LEVEL]
