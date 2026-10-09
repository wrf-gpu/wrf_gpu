"""Static WRF base lifetime and output routing; CPU regression gates."""
from datetime import datetime
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from netCDF4 import Dataset
import numpy as np
import pytest

from gpuwrf.contracts import state as state_contract
from gpuwrf.io.wrfout_writer import prepare_wrfout_payload
from gpuwrf.runtime import operational_mode as op
from gpuwrf.dynamics.metrics import load_wrfinput_metrics
from gpuwrf.validation.tier2 import make_ideal_grid


def writer_fixture():
    path = Path(__file__).resolve().parents[2] / "test_m7_netcdf_writer.py"
    spec = importlib.util.spec_from_file_location("base_writer_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("native", [False, True])
def test_initial_carry_retains_wrf_real_base_only_with_native_rk(monkeypatch, native):
    path = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725/wrfinput_d01")
    if not path.exists():
        pytest.skip("PROD wrfinput fixture unavailable")
    assert jax.devices()[0].platform == "cpu"
    monkeypatch.setenv("GPUWRF_DYN_RK_FP32", "1" if native else "0")
    if not native:  # legacy RK arm: the native-RK dependents the release defaults set are off too
        for name in ("GPUWRF_DYN_CARRY_FP32", "GPUWRF_DYN_REAL_ALL", "GPUWRF_CARRY_REAL_ALL"):
            monkeypatch.setenv(name, "0")
    monkeypatch.setattr(state_contract, "_gpu_device", lambda: jax.devices()[0])
    grid = make_ideal_grid(44, 3, 3)
    with Dataset(path) as ds:
        fields = {n: np.asarray(ds[n][0, ..., 40:43, 63:66])
                  for n in ("PB", "PHB", "MUB", "P", "PH", "MU")}
    state = state_contract.State.zeros(grid).replace(
        _cast=False,
        p_total=jnp.asarray(fields["PB"], jnp.float64) + jnp.asarray(fields["P"], jnp.float64),
        p_perturbation=jnp.asarray(fields["P"], jnp.float64),
        ph_total=jnp.asarray(fields["PHB"], jnp.float64) + jnp.asarray(fields["PH"], jnp.float64),
        ph_perturbation=jnp.asarray(fields["PH"], jnp.float64),
        mu_total=jnp.asarray(fields["MUB"], jnp.float64) + jnp.asarray(fields["MU"], jnp.float64),
        mu_perturbation=jnp.asarray(fields["MU"], jnp.float64),
    )
    nml = replace(op.OperationalNamelist.from_grid(grid, force_fp64=True),
        mp_physics=0, cu_physics=0,
        bl_pbl_physics=0, sf_sfclay_physics=0,
        sf_surface_physics=0, ra_sw_physics=0, ra_lw_physics=0,
    )
    carry = op._initial_carry_for_run(state, nml)
    if not native:
        assert carry.base_state is None
    else:
        assert carry.base_state is not None, "native RK lost its resolved initial WRF base"
        for name, wrf in (("pb", "PB"), ("phb", "PHB"), ("mub", "MUB")):
            value = getattr(carry.base_state, name)
            assert value.dtype == jnp.float32
            np.testing.assert_array_equal(np.asarray(value), fields[wrf])


@pytest.mark.parametrize("subset", [None, ("P", "PB", "PH", "PHB", "MU", "MUB", "PSFC")])
def test_native_writer_uses_retained_base_without_total_subtraction(monkeypatch, tmp_path, subset):
    fixture = writer_fixture()
    state, grid, nml = fixture.synthetic_case()
    base = {"PB": np.full_like(state.p_total, np.float32(65523.55859375)),
            "PHB": np.full_like(state.ph_total, np.float32(130845.0078125)),
            "MUB": np.full_like(state.mu_total, np.float32(90001.125))}
    # A rounded total no longer contains the exact WRF base. Supplying the
    # retained base must work even if that redundant total has already drifted.
    state.p_total = np.float32(base["PB"] + state.p_perturbation + np.float32(.2890625))
    state.ph_total = np.float32(base["PHB"] + state.ph_perturbation + np.float32(1.0625))
    state.mu_total = np.float32(base["MUB"] + state.mu_perturbation + np.float32(.25))
    monkeypatch.setenv("GPUWRF_DYN_RK_FP32", "1")
    prepared = prepare_wrfout_payload(
        state, grid, nml, tmp_path / "native.nc", domain="d02",
        domain_authority=fixture.writer_authority(grid),
        valid_time=datetime(2026, 7, 26, 6), lead_hours=6,
        run_start=datetime(2026, 7, 26), diagnostics=base,
        variable_subset=subset,
    )
    for name, expected in base.items():
        np.testing.assert_array_equal(prepared.fields[name], expected)
    for name, attr in (("P", "p_perturbation"), ("PH", "ph_perturbation"), ("MU", "mu_perturbation")):
        np.testing.assert_array_equal(prepared.fields[name], getattr(state, attr))
    monkeypatch.setenv("GPUWRF_DYN_RK_FP32", "0")
    legacy = prepare_wrfout_payload(
        state, grid, nml, tmp_path / "off.nc", domain="d02",
        domain_authority=fixture.writer_authority(grid),
        valid_time=datetime(2026, 7, 26, 6), lead_hours=6,
        run_start=datetime(2026, 7, 26), diagnostics=base,
        variable_subset=subset,
    )
    for name in base:
        assert not np.array_equal(legacy.fields[name], base[name])


def test_nested_writer_passes_immutable_base_at_two_output_boundaries(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[2] / "test_v0222_output_pipeline.py"
    spec = importlib.util.spec_from_file_location("base_pipeline_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    writer = fixture._make_writer(async_writer=None)
    # The existing fixture predates the census writer sink.
    writer.census_io_ledger = None
    state = fixture._bind_case(writer, output_dir=tmp_path, dt_s=fixture.DT_S,
                               run_start=fixture.RUN_START)
    base = SimpleNamespace(
        pb=jnp.full_like(state.p_total, 65523.55859375),
        phb=jnp.full_like(state.ph_total, 130845.0078125),
        mub=jnp.full_like(state.mu_total, 90001.125),
    )
    monkeypatch.setenv("GPUWRF_DYN_RK_FP32", "1")
    for step, delta in ((120, .25), (240, 1.0)):
        changed = SimpleNamespace(**vars(state))
        changed.p_total = base.pb + changed.p_perturbation + delta
        changed.ph_total = base.phb + changed.ph_perturbation + delta
        changed.mu_total = base.mub + changed.mu_perturbation + delta
        result = writer("d01", own_step=step,
                        carry=SimpleNamespace(state=changed, base_state=base))
        with Dataset(result["wrfout"]) as ds:
            for name, value in (("PB", base.pb), ("PHB", base.phb), ("MUB", base.mub)):
                np.testing.assert_array_equal(np.asarray(ds[name][0]), np.asarray(value))


def test_native_base_capture_preserves_terrain_theta_diffusion_reference(monkeypatch, cpu_pallas_interpret):
    """Consumer regression on real terrain; six-hour WRF fidelity is separate."""
    path = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725/wrfinput_d01")
    if not path.exists():
        pytest.skip("PROD wrfinput fixture unavailable")
    assert jax.devices()[0].platform == "cpu"
    monkeypatch.setenv("GPUWRF_DYN_RK_FP32", "1")
    monkeypatch.setattr(state_contract, "_gpu_device", lambda: jax.devices()[0])
    ny = nx = 5
    y, x = 29, 48
    with Dataset(path) as ds:
        full_ny = len(ds.dimensions["south_north"])
        full_nx = len(ds.dimensions["west_east"])
        nz = len(ds.dimensions["bottom_top"])
        def read(name):
            var = ds[name]
            ey = int(var.dimensions[-2] == "south_north_stag")
            ex = int(var.dimensions[-1] == "west_east_stag")
            return jnp.asarray(np.asarray(var[0, ..., y:y+ny+ey, x:x+nx+ex]), jnp.float64)
        a = {n: read(n) for n in ("PB", "PHB", "MUB", "P", "PH", "MU", "U", "V", "W", "T", "QVAPOR", "HGT")}
        dx, dy = float(ds.DX), float(ds.DY)
    metrics = load_wrfinput_metrics(path)
    updates = {}
    for name in metrics._array_names():
        value = getattr(metrics, name)
        if value.ndim == 2:
            ey = int(value.shape[0] == full_ny + 1)
            ex = int(value.shape[1] == full_nx + 1)
            updates[name] = value[y:y+ny+ey, x:x+nx+ex]
    metrics = replace(metrics, **updates)
    grid = make_ideal_grid(nz, ny, nx, dx_m=dx, dy_m=dy)
    grid = replace(grid, metrics=metrics, terrain_height=a["HGT"],
                   bc=replace(grid.bc, source="wrfbdy"))
    assert float(jnp.max(a["HGT"]) - jnp.min(a["HGT"])) > 1
    state = state_contract.State.zeros(grid).replace(
        _cast=False, u=a["U"], v=a["V"], w=a["W"], qv=a["QVAPOR"],
        theta=(a["T"] + 300) * (1 + (461.6 / 287) * a["QVAPOR"]),
        p_total=a["PB"] + a["P"], p_perturbation=a["P"],
        ph_total=a["PHB"] + a["PH"], ph_perturbation=a["PH"],
        mu_total=a["MUB"] + a["MU"], mu_perturbation=a["MU"],
    )
    nml = op.OperationalNamelist.from_grid(
        grid, metrics=metrics, force_fp64=True, diff_opt=1, km_opt=4,
        hypsometric_opt=2,
    )
    assert op._acoustic_lateral_bc_flags(nml) == (False, True, False)
    reference = op._diffopt1_dry_forward_tendencies(state, nml)
    captured = op._base_state_from_totals(state)
    candidate = op._diffopt1_dry_forward_tendencies(state, nml, base_state=captured)
    assert float(jnp.max(jnp.abs(reference[3]))) > 0
    for before, after in zip(reference, candidate):
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))
