"""fid-q2 W01: Noah-MP glacier-point history = WRF's undefined sentinel, byte-exact vs original CPU-WRF.

module_sf_noahmpdrv.F (V4.7.1) :1065-1128 sets undefined_value (-1.E36) for the vegetation/canopy
outputs of a land-ice point (IVGTYP == ISICE) after the first solve, resets CANICE/CANLIQ/QTLDRN to 0
(:1150-1152) and writes Q2MVXY = Q2MV/(1-Q2MV) = -1 (:1283).  Oracle: CPU-WRF Swiss nest SN02
(2025-02-18_18z, d01 24 / d02 215 ice cells); operands: the measured GPU frames of the same case.
"""
from datetime import datetime
from pathlib import Path

import jax
import numpy as np
import pytest
from netCDF4 import Dataset

from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.land_history import GLACIER_UNDEFINED_FIELDS, LAND_LEAVES, land_history_diagnostics, load_land_history_inputs
from gpuwrf.io.noahmp_land_init import build_noahmp_land_state
from gpuwrf.io.wrfout_writer import bind_wrfout_domain_authority, prepare_wrfout_payload, write_prepared_wrfout
from gpuwrf.runtime.history_accumulators import LAND_FLUX_FIELDS

TABLE = Path("<USER_HOME>/src/wrf_pristine/WRF/run")
CASE = Path("<DATA_ROOT>/alisios/registry/wrf_cases/payload/swiss_nest/2025-02-18")
CPU = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/SWISS_NEST_REF/SN02/2025-02-18_18z/cpu")
GPU = Path("<USER_HOME>/wrf_gpu2_lanes/integrate/VAL31/cases/swiss_2025-02-18_18z/2025-02-18_18z/wrfout")
ICE_CELLS = {"d01": 24, "d02": 215}
HOURS = (0, 1, 2, 12, 24)


def _bytes(tree):
    return [(x.dtype.str, x.shape, x.tobytes()) for x in jax.tree.leaves(jax.device_get(tree))]


def _frame(path):
    ds = Dataset(path)
    ds.set_auto_mask(False)
    return ds


@pytest.fixture(scope="module", params=["d01", "d02"])
def swiss(request):
    domain = request.param
    cpu = sorted(CPU.glob(f"wrfout_{domain}_*"))
    gpu = sorted(GPU.glob(f"wrfout_{domain}_*"))
    if len(cpu) < 25 or len(gpu) < 25 or not (CASE / f"wrfinput_{domain}").is_file():
        pytest.skip("Swiss SN02 CPU-WRF history, its inputs or the measured GPU frames not mounted")
    initial, static, _ = build_noahmp_land_state(CASE, domain, table_dir=TABLE)
    inputs = load_land_history_inputs(CASE / f"wrfinput_{domain}", table_dir=TABLE, parameters=static.parameters)
    return domain, cpu, gpu, initial, inputs


def _operands(initial, path):
    with _frame(path) as ds:
        updates = {attr: np.asarray(ds[name][0]).astype(np.asarray(getattr(initial, attr)).dtype)
                   for name, attr in LAND_LEAVES.items()}
        history = {name: np.asarray(ds[name][0]) for name in LAND_FLUX_FIELDS if name in ds.variables}
    return initial.replace(**updates), history


def test_glacier_list_is_wrf_source_list():
    # 32 undefined outputs + Q2V (-1) — the source-derived set the CPU-WRF files carry; no canopy water.
    assert len(set(GLACIER_UNDEFINED_FIELDS)) == len(GLACIER_UNDEFINED_FIELDS) == 32
    assert not {"CANLIQ", "CANICE", "CANWAT", "ZWT", "QTDRAIN", "Q2V"} & set(GLACIER_UNDEFINED_FIELDS)


def test_glacier_history_equals_cpu_wrf_bytes(swiss, tmp_path):
    domain, cpu, gpu, initial, inputs = swiss
    ice = inputs["_GLACIER_MASK"]
    assert int(ice.sum()) == ICE_CELLS[domain]
    # Oracle-derived variable set (independent of the module constant): every CPU-WRF history field
    # holding the sentinel on the ice cells after the first solve, plus Q2V and the reset canopy water.
    with _frame(cpu[1]) as first:
        undefined = sorted(name for name, var in first.variables.items()
                           if var.dtype.kind == "f" and var.shape[1:] == ice.shape
                           and (np.asarray(var[0])[ice] <= -1e35).any())
    assert len(undefined) == 32
    names = (*undefined, "Q2V", "CANLIQ", "CANICE", "CANWAT")
    grid = Gen2Run(CASE).grid(domain)
    authority = bind_wrfout_domain_authority(domain, grid, grid)
    for hour in HOURS:
        carry, history = _operands(initial, gpu[hour])
        before = _bytes((carry, initial, history))
        fields, host = land_history_diagnostics(carry, initial, inputs, own_step=hour * 200, history=history)
        with _frame(cpu[hour]) as oracle:
            valid = oracle.variables["Times"][0].tobytes().decode()
        valid = datetime.strptime(valid, "%Y-%m-%d_%H:%M:%S")
        target = tmp_path / f"{domain}_{hour:02d}.nc"
        prepared = prepare_wrfout_payload({}, grid, None, target, domain=domain, domain_authority=authority,
            valid_time=valid, run_start=datetime(2025, 2, 18, 18), lead_hours=hour, diagnostics=fields,
            land_state=host, variable_subset=names, full_variable_set=True)
        write_prepared_wrfout(prepared, expected_domain=domain, expected_domain_authority=authority)
        with _frame(target) as actual_ds, _frame(cpu[hour]) as oracle:
            sentinel_seen = 0
            for name in names:
                expected = np.asarray(oracle[name][0])
                actual = np.asarray(actual_ds[name][0])
                assert actual.dtype == expected.dtype, name
                assert actual[ice].tobytes() == expected[ice].tobytes(), f"{domain} h{hour} {name}"
                sentinel_seen += int((expected[ice] <= -1e35).any())
            assert sentinel_seen == (0 if hour == 0 else len(undefined))
        # Non-glacier history is the model's own value, untouched by the fill.
        for name in GLACIER_UNDEFINED_FIELDS:
            source = history.get(name)
            if source is not None and hour:
                np.testing.assert_array_equal(np.asarray(fields[name])[~ice], source[~ice])
        assert before == _bytes((carry, initial, history))
        assert _bytes(host) == _bytes(carry)


def test_without_in_step_packet_history_fields_are_left_to_their_writer_source(swiss):
    domain, _cpu, gpu, initial, inputs = swiss
    carry, _history = _operands(initial, gpu[1])
    fields, _ = land_history_diagnostics(carry, initial, inputs, own_step=200, history=None)
    for name in GLACIER_UNDEFINED_FIELDS:
        if name in LAND_FLUX_FIELDS:
            assert name not in fields, name  # no all-land zero overwrite of another source's field
        else:
            assert (np.asarray(fields[name])[inputs["_GLACIER_MASK"]] == np.float32(-1e36)).all(), name
    assert "Q2V" not in fields
