"""#122: compact training-ready wrfout via ``variable_subset``.

Proves the opt-in compact training output path:

* a ``variable_subset`` restricts the emitted variables to the named set (plus the
  always-written mandatory coordinates);
* the mandatory dimension / coordinate variables are always present;
* a name absent from the prepared payload is silently skipped -- the file never
  fabricates a field;
* the DEFAULT (``variable_subset=None``) output is value-byte-identical to the
  pre-#122 behaviour (full, uncompressed);
* lossless NetCDF4 compression is applied to a subset stream but NOT to the
  default full stream, and round-trips bit-identical values;
* the nest callback env opt-in (``GPUWRF_TRAINING_OUTPUT_SUBSET``) resolves the
  subset only when explicitly enabled.

CPU-only (no GPU). 0:2's GPU smoketest is the LATER joint acceptance gate.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.io.wrfout_writer import (
    MANDATORY_WRFOUT_COORDINATES,
    MINIMAL_TRAINING_SET,
    WRFOUT_VARIABLE_SPECS,
    prepare_wrfout_payload,
    write_prepared_wrfout,
    write_wrfout_netcdf,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_m7_netcdf_writer import synthetic_case  # type: ignore  # noqa: E402


_VALID = datetime(2026, 5, 25, 21)
_START = datetime(2026, 5, 25, 18)


def _prepared(tmp_path: Path, name: str = "wrfout.nc"):
    state, grid, namelist = synthetic_case()
    return prepare_wrfout_payload(
        state, grid, namelist, tmp_path / name,
        valid_time=_VALID, lead_hours=3.0, run_start=_START,
    )


def _var_values(path: Path) -> dict[str, np.ndarray]:
    with Dataset(path) as ds:
        return {k: np.asarray(ds.variables[k][:]) for k in ds.variables}


# --- the 36-name training set is well-formed --------------------------------

def test_minimal_training_set_is_36_known_specs():
    assert len(MINIMAL_TRAINING_SET) == 36
    assert len(set(MINIMAL_TRAINING_SET)) == 36  # no duplicates
    for name in MINIMAL_TRAINING_SET:
        assert name in WRFOUT_VARIABLE_SPECS, f"{name} is not a real writer spec"


def test_minimal_training_set_exact_membership():
    expected = {
        # 3D (16)
        "U", "V", "W", "T", "P", "PB", "PH", "PHB",
        "QVAPOR", "QCLOUD", "QICE", "QRAIN", "QSNOW", "QGRAUP", "CLDFRA", "QKE",
        # 2D (12)
        "T2", "Q2", "U10", "V10", "PSFC", "RAINNC",
        "SWDOWN", "GLW", "HFX", "LH", "PBLH", "TSK",
        # static (5)
        "HGT", "XLAT", "XLONG", "LANDMASK", "LU_INDEX",
        # wind-rot / map (3)
        "SINALPHA", "COSALPHA", "MAPFAC_M",
    }
    assert set(MINIMAL_TRAINING_SET) == expected


# --- subset restriction + mandatory coords + skip-absent --------------------

def test_subset_restricts_to_named_set_plus_mandatory_coords(tmp_path):
    prepared = _prepared(tmp_path)
    out = tmp_path / "subset.nc"
    write_prepared_wrfout(
        prepared, variable_subset=MINIMAL_TRAINING_SET, target_override=out,
        include_mandatory_coords=True, compress=True,
    )

    written = set(_var_values(out))
    payload = set(prepared.fields)

    # Every emitted variable is EITHER a requested training var OR a mandatory
    # coordinate -- nothing outside that union leaks into the subset file.
    allowed = set(MINIMAL_TRAINING_SET) | set(MANDATORY_WRFOUT_COORDINATES)
    assert written <= allowed, f"subset emitted unexpected vars: {written - allowed}"

    # Times is always present (written unconditionally by _write_times).
    assert "Times" in written

    # Every requested training var that HAS a source in the payload is present...
    for name in MINIMAL_TRAINING_SET:
        if name in payload:
            assert name in written, f"requested+available {name} missing from subset file"
        else:
            # ...and one with no source is silently skipped (never fabricated).
            assert name not in written, f"{name} fabricated despite no payload source"

    # The non-training fields that ARE in the payload must be excluded.
    excluded = payload - allowed
    assert excluded, "test precondition: synthetic payload has non-training vars to drop"
    for name in excluded:
        assert name not in written, f"non-training {name} leaked into subset file"


def test_subset_force_includes_available_mandatory_coords(tmp_path):
    prepared = _prepared(tmp_path)
    out = tmp_path / "coords.nc"
    # A subset of ONLY 2D surface fields -- mandatory coords must still appear.
    write_prepared_wrfout(
        prepared, variable_subset=("T2", "U10"), target_override=out,
        include_mandatory_coords=True,
    )
    written = set(_var_values(out))
    assert {"T2", "U10"} <= written
    # Mandatory coords present in the payload are force-emitted even though they
    # were not in the requested subset.
    for coord in MANDATORY_WRFOUT_COORDINATES:
        if coord in prepared.fields or coord == "Times":
            assert coord in written, f"mandatory coord {coord} dropped from subset file"


def test_subset_skips_unknown_requested_name(tmp_path):
    prepared = _prepared(tmp_path)
    out = tmp_path / "unknown.nc"
    # A name not present in the payload (and not even a real var) must be skipped,
    # never fabricated, and must not raise.
    write_prepared_wrfout(
        prepared, variable_subset=("T2", "NOT_A_REAL_VARIABLE"), target_override=out
    )
    written = set(_var_values(out))
    assert "T2" in written
    assert "NOT_A_REAL_VARIABLE" not in written


# --- default-None is byte-identical (value-level) to before -----------------

def test_default_none_output_is_value_identical(tmp_path):
    """write_prepared_wrfout(prepared) (no subset) == write_wrfout_netcdf legacy."""
    state, grid, namelist = synthetic_case()
    legacy = tmp_path / "legacy.nc"
    write_wrfout_netcdf(
        state, grid, namelist, legacy,
        valid_time=_VALID, lead_hours=3.0, run_start=_START,
    )
    prepared = _prepared(tmp_path, "prepared.nc")
    default = tmp_path / "default.nc"
    write_prepared_wrfout(prepared, variable_subset=None, target_override=default)

    a, b = _var_values(legacy), _var_values(default)
    assert sorted(a) == sorted(b)
    for name in a:
        eq_nan = np.issubdtype(np.asarray(a[name]).dtype, np.floating)
        assert np.array_equal(a[name], b[name], equal_nan=eq_nan), f"{name} differs"


def test_default_full_output_is_uncompressed(tmp_path):
    """The default path leaves filters off so its on-disk bytes are unchanged."""
    prepared = _prepared(tmp_path)
    out = tmp_path / "full.nc"
    write_prepared_wrfout(prepared, variable_subset=None, target_override=out)
    with Dataset(out) as ds:
        for name, var in ds.variables.items():
            filt = var.filters() or {}
            assert not filt.get("zlib", False), f"{name} unexpectedly compressed in default output"


# --- compression on subset stream, lossless ---------------------------------

def test_subset_stream_is_compressed_and_lossless(tmp_path):
    prepared = _prepared(tmp_path)
    out = tmp_path / "compressed.nc"
    write_prepared_wrfout(
        prepared, variable_subset=MINIMAL_TRAINING_SET, target_override=out,
        include_mandatory_coords=True, compress=True,
    )

    with Dataset(out) as ds:
        # The float data variables carry zlib (Times is an S1 char coordinate).
        compressed_any = False
        for name, var in ds.variables.items():
            if name in {"Times", "XTIME"}:
                continue
            filt = var.filters() or {}
            if filt.get("zlib", False):
                compressed_any = True
                assert filt.get("complevel", 0) >= 1
        assert compressed_any, "subset stream was not compressed"

    # Lossless: values read back match the prepared payload exactly.
    written = _var_values(out)
    for name in MINIMAL_TRAINING_SET:
        if name in prepared.fields and name in written:
            expected = np.asarray(prepared.fields[name])
            got = np.asarray(written[name]).reshape(expected.shape)
            assert np.array_equal(got, expected, equal_nan=True), f"{name} not lossless"


# --- write_wrfout_netcdf threads variable_subset ----------------------------

def test_write_wrfout_netcdf_threads_subset(tmp_path):
    state, grid, namelist = synthetic_case()
    out = tmp_path / "direct_subset.nc"
    write_wrfout_netcdf(
        state, grid, namelist, out,
        valid_time=_VALID, lead_hours=3.0, run_start=_START,
        variable_subset=MINIMAL_TRAINING_SET,
        include_mandatory_coords=True, compress=True,
    )
    written = set(_var_values(out))
    allowed = set(MINIMAL_TRAINING_SET) | set(MANDATORY_WRFOUT_COORDINATES)
    assert written <= allowed
    assert "T2" in written


# --- env opt-in resolver (nest callback) ------------------------------------

def test_training_output_subset_env_opt_in(monkeypatch):
    from gpuwrf.integration.nested_pipeline import _resolve_training_output_subset

    monkeypatch.delenv("GPUWRF_TRAINING_OUTPUT_SUBSET", raising=False)
    assert _resolve_training_output_subset() is None

    for falsy in ("", "0", "false", "no", "off", "garbage"):
        monkeypatch.setenv("GPUWRF_TRAINING_OUTPUT_SUBSET", falsy)
        assert _resolve_training_output_subset() is None, f"{falsy!r} should not opt in"

    for truthy in ("1", "true", "TRUE", "Yes", "on"):
        monkeypatch.setenv("GPUWRF_TRAINING_OUTPUT_SUBSET", truthy)
        assert _resolve_training_output_subset() == MINIMAL_TRAINING_SET, f"{truthy!r} should opt in"
