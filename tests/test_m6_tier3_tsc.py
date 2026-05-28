from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

from gpuwrf.coupling.physics_couplers import thompson_adapter_with_tendencies
from gpuwrf.io.boundary_replay import compare_boundary_tendency_to_wrfbdy, decode_wrfbdy
from gpuwrf.io.proof_schemas import Tier3DriftEnvelope
from gpuwrf.validation.tier2_coupled import water_budget_residual
from gpuwrf.validation.tier3_coupled import (
    classify_drift,
    compute_tsc_envelope,
    idealized_coupled_state,
)


def test_tsc_envelope_uses_raw_pairwise_max_without_cap():
    outputs = {
        18.0: {6.0: {"U10": np.ones((2, 2)) * 7.0}},
        9.0: {6.0: {"U10": np.ones((2, 2)) * 3.0}},
        4.5: {6.0: {"U10": np.ones((2, 2)) * -4.0}},
    }

    envelope = compute_tsc_envelope(outputs, variables=("U10",), leads_h=(6.0,), dts=(18.0, 9.0, 4.5))

    assert envelope["U10"]["+6h"]["base_vs_refine"]["max_abs"] == 4.0
    assert envelope["U10"]["+6h"]["refine_vs_further"]["max_abs"] == 7.0
    assert envelope["U10"]["+6h"]["envelope"] == 7.0


def test_per_variable_status_is_per_lead_not_aggregate_only():
    envelope = {
        "U10": {
            "+6h": {"envelope": 1.0},
            "+12h": {"envelope": 1.0},
        }
    }
    drift = {
        "U10": {
            "+6h": {"max_abs": 0.5},
            "+12h": {"max_abs": 2.0},
        }
    }

    statuses, overall = classify_drift(envelope, drift)

    assert overall == "PARTIAL"
    assert statuses["U10"]["status"] == "PARTIAL"
    assert statuses["U10"]["leads"]["+6h"]["status"] == "GREEN"
    assert statuses["U10"]["leads"]["+12h"]["status"] == "FAIL"


def test_thompson_water_budget_side_channel_is_load_bearing():
    state, _tendencies, _grid = idealized_coupled_state()
    next_state, oracle = thompson_adapter_with_tendencies(state, 18.0)

    closed = water_budget_residual(state, next_state, 18.0, oracle)
    corrupted = water_budget_residual(state, next_state.replace(qv=next_state.qv + 1.0e-6), 18.0, oracle)

    assert closed["oracle_source"] == "ThompsonTendencySideChannel"
    assert closed["max_abs"] <= 1.0e-9
    assert corrupted["max_abs"] > 1.0e-7


def _write_var(dataset: Dataset, name: str, dims: tuple[str, ...], value: float) -> None:
    variable = dataset.createVariable(name, "f4", dims)
    variable.units = "test"
    variable[:] = np.ones(tuple(len(dataset.dimensions[dim]) for dim in dims), dtype=np.float32) * value


def _write_minimal_wrfbdy(path: Path) -> None:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("Time", 1)
        dataset.createDimension("DateStrLen", 19)
        dataset.createDimension("bdy_width", 1)
        dataset.createDimension("bottom_top", 2)
        dataset.createDimension("south_north", 3)
        dataset.createDimension("west_east_stag", 5)
        times = dataset.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0, :] = np.asarray(list("2026-05-20_18:00:00"), dtype="S1")
        _write_var(dataset, "U_BXS", ("Time", "bdy_width", "bottom_top", "south_north"), 0.0)
        _write_var(dataset, "U_BXE", ("Time", "bdy_width", "bottom_top", "south_north"), 0.0)
        _write_var(dataset, "U_BYS", ("Time", "bdy_width", "bottom_top", "west_east_stag"), 0.0)
        _write_var(dataset, "U_BYE", ("Time", "bdy_width", "bottom_top", "west_east_stag"), 0.0)
        _write_var(dataset, "U_BTXS", ("Time", "bdy_width", "bottom_top", "south_north"), 2.0)
        _write_var(dataset, "U_BTXE", ("Time", "bdy_width", "bottom_top", "south_north"), 2.0)
        _write_var(dataset, "U_BTYS", ("Time", "bdy_width", "bottom_top", "west_east_stag"), 2.0)
        _write_var(dataset, "U_BTYE", ("Time", "bdy_width", "bottom_top", "west_east_stag"), 2.0)


def test_wrfbdy_decoder_compares_gpu_boundary_tendency(tmp_path: Path):
    path = tmp_path / "wrfbdy_d01"
    _write_minimal_wrfbdy(path)
    decoded = decode_wrfbdy(path, variables=("U",), time_index=0)
    gpu_u_tendency = np.ones((2, 3, 5), dtype=np.float64) * 2.0

    comparison = compare_boundary_tendency_to_wrfbdy({"u": gpu_u_tendency}, decoded, variables=("U",), width=1)

    assert decoded["schema"] == "wrfbdy_decoder_v1"
    assert decoded["times"] == ["2026-05-20_18:00:00"]
    assert comparison["variables"]["U"]["aggregate"]["max_abs_max"] == 0.0


def test_tier3_schema_requires_per_variable_tables():
    payload = {
        "run_id": "test",
        "domain": "d02",
        "status": "GREEN",
        "base_dt_s": 18.0,
        "refined_dt_s": 9.0,
        "further_refined_dt_s": 4.5,
        "lead_hours": [6.0],
        "variables": ["U10"],
        "boundary_mode": {},
        "forcing_mode": {},
        "regridding": {},
        "norm_definitions": {},
        "envelope_derivation": {},
        "envelope": {"U10": {"+6h": {"envelope": 1.0}}},
        "gpu_drift": {"U10": {"+6h": {"max_abs": 0.5}}},
        "per_variable_status": {"U10": {"status": "GREEN", "leads": {"+6h": {"status": "GREEN"}}}},
        "artifact_paths": ["artifacts/m6/tier3/tsc_envelope.json"],
    }

    assert Tier3DriftEnvelope.validate_dict(payload) is payload
