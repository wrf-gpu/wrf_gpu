from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset, chartostring

from gpuwrf.io.gen2_accessor import Gen2GridSpec
from gpuwrf.io.wrfout_writer import (
    DERIVED_DIAGNOSTIC_VARIABLES,
    FULL_WRFOUT_VARIABLES,
    GRID_METRIC_EXTRA_VARIABLES,
    MINIMUM_WRFOUT_VARIABLES,
    RADIATION_FLUX_DIAGNOSTIC_VARIABLES,
    WRFOUT_VARIABLE_SPECS,
    WrfoutDomainAuthority,
    _domain_authority_payload,
    _domain_authority_sha256,
    _set_wrfout_grid_id_attr,
    _write_global_attrs,
    bind_wrfout_domain_authority,
    prepare_wrfout_payload,
    write_prepared_wrfout,
    write_wrfout_netcdf,
)


REFERENCE = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l3/"
    "20260525_18z_l3_24h_20260526T221207Z/"
    "wrfout_d02_2026-05-25_18:00:00"
)
FULL_REFERENCE = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l3/"
    "20260428_18z_l3_24h_20260525T221139Z/"
    "wrfout_d02_2026-04-28_19:00:00"
)


def synthetic_case(nx: int = 5, ny: int = 4, nz: int = 3):
    y2, x2 = np.indices((ny, nx), dtype=np.float32)
    z3 = np.arange(nz, dtype=np.float32)[:, None, None]
    zf = np.arange(nz + 1, dtype=np.float32)[:, None, None]
    terrain = 100.0 + y2 + 0.5 * x2
    pb = (90_000.0 - 800.0 * z3 + terrain[None, :, :]).astype(np.float32)
    p_pert = (100.0 + 0.5 * z3 + 0.1 * x2[None, :, :]).astype(np.float32)
    phb = (9.81 * (terrain[None, :, :] + 600.0 * zf)).astype(np.float32)
    ph_pert = (3.0 + 0.2 * zf + 0.01 * y2[None, :, :]).astype(np.float32)
    mub = (85_000.0 + terrain).astype(np.float32)
    mu_pert = (40.0 + 0.2 * x2 + 0.1 * y2).astype(np.float32)
    landmask = np.where((x2 + y2) % 3 == 0, 0.0, 1.0).astype(np.float32)

    state = SimpleNamespace(
        u=(4.0 + np.zeros((nz, ny, nx + 1), dtype=np.float32)),
        v=(1.5 + np.zeros((nz, ny + 1, nx), dtype=np.float32)),
        w=np.zeros((nz + 1, ny, nx), dtype=np.float32),
        theta=(300.0 + z3 + 0.1 * y2[None, :, :]).astype(np.float32),
        qv=(0.009 + np.zeros((nz, ny, nx), dtype=np.float32)),
        qc=(1.0e-5 + np.zeros((nz, ny, nx), dtype=np.float32)),
        qi=(2.0e-6 + np.zeros((nz, ny, nx), dtype=np.float32)),
        qr=(3.0e-6 + np.zeros((nz, ny, nx), dtype=np.float32)),
        p_total=pb + p_pert,
        p_perturbation=p_pert,
        ph_total=phb + ph_pert,
        ph_perturbation=ph_pert,
        mu_total=mub + mu_pert,
        mu_perturbation=mu_pert,
        u10=(4.2 + 0.01 * x2).astype(np.float32),
        v10=(1.1 + 0.01 * y2).astype(np.float32),
        t2=(289.0 + 0.1 * y2).astype(np.float32),
        q2=(0.008 + np.zeros((ny, nx), dtype=np.float32)),
        psfc=(pb[0] + p_pert[0]).astype(np.float32),
        rainc=np.zeros((ny, nx), dtype=np.float32),
        rain_acc=(0.2 + 0.01 * x2).astype(np.float32),
        rainsh=np.zeros((ny, nx), dtype=np.float32),
        swdown=(500.0 + x2).astype(np.float32),
        glw=(300.0 + y2).astype(np.float32),
        pblh=(800.0 + y2).astype(np.float32),
        ustar=(0.3 + np.zeros((ny, nx), dtype=np.float32)),
        hfx=(20.0 + y2).astype(np.float32),
        lh=(70.0 + x2).astype(np.float32),
        t_skin=(290.0 + y2).astype(np.float32),
        cldfra=(0.25 + np.zeros((nz, ny, nx), dtype=np.float32)),
        landmask=landmask,
        lu_index=np.where(landmask > 0.5, 2.0, 17.0).astype(np.float32),
    )
    grid = SimpleNamespace(
        nx=nx,
        ny=ny,
        nz=nz,
        projection=SimpleNamespace(kind="lambert", lat_0=28.34, lon_0=-16.12, dx_m=3000.0, dy_m=3000.0),
        vertical=SimpleNamespace(nz=nz, top_pressure_pa=5_000.0),
        terrain_height=terrain.astype(np.float32),
    )
    namelist = {
        "title": " OUTPUT FROM GPUWRF WRF-COMPATIBLE NETCDF WRITER",
        "truelat1": 25.0,
        "truelat2": 30.0,
        "stand_lon": -16.4,
        "moad_cen_lat": 28.3,
        "cen_lat": 28.34,
        "cen_lon": -16.12,
        "soil_layers_stag": 4,
    }
    return state, grid, namelist


def authenticated_source_grid(grid, domain: str = "d02") -> Gen2GridSpec:
    projection = getattr(grid, "projection", SimpleNamespace(dx_m=3000.0, dy_m=3000.0))
    return Gen2GridSpec(
        id=domain,
        dx_m=float(getattr(projection, "dx_m", 3000.0)),
        dy_m=float(getattr(projection, "dy_m", 3000.0)),
        e_we=int(grid.nx) + 1,
        e_sn=int(grid.ny) + 1,
        e_vert=int(grid.nz) + 1,
        mass_nx=int(grid.nx),
        mass_ny=int(grid.ny),
        mass_nz=int(grid.nz),
        grid_proj="lambert",
        map_proj_id=1,
        cen_lat=28.3,
        cen_lon=-16.1,
        truelat1=25.0,
        truelat2=30.0,
        stand_lon=-16.4,
        parent_id=1,
        parent_grid_ratio=1 if domain == "d01" else 3,
        i_parent_start=1,
        j_parent_start=1,
        znu=tuple(float(i) for i in range(int(grid.nz))),
        znw=tuple(float(i) for i in range(int(grid.nz) + 1)),
        top_pressure_pa=5000.0,
        source_wrfout="authenticated-test-source",
        source_namelist="authenticated-test-namelist",
    )


def writer_authority(grid, domain: str = "d02") -> WrfoutDomainAuthority:
    source_grid = authenticated_source_grid(grid, domain)
    return bind_wrfout_domain_authority(domain, source_grid, grid)


def write_small(tmp_path: Path) -> tuple[Path, SimpleNamespace]:
    state, grid, namelist = synthetic_case()
    path = tmp_path / "wrfout_d02_2026-05-25_21:00:00"
    write_wrfout_netcdf(
        state,
        grid,
        namelist,
        path,
        domain="d02",
        domain_authority=writer_authority(grid),
        valid_time=datetime(2026, 5, 25, 21),
        lead_hours=3.0,
        run_start=datetime(2026, 5, 25, 18),
    )
    return path, state


def test_roundtrip_writes_minimum_variables_and_time_coordinates(tmp_path: Path):
    path, _ = write_small(tmp_path)

    with Dataset(path) as dataset:
        missing = [name for name in MINIMUM_WRFOUT_VARIABLES if name not in dataset.variables]
        assert missing == []
        assert dataset.file_format == "NETCDF4"
        assert len(dataset.dimensions["Time"]) == 1
        assert len(dataset.dimensions["DateStrLen"]) == 19
        assert len(dataset.dimensions["west_east"]) == 5
        assert len(dataset.dimensions["west_east_stag"]) == 6
        assert len(dataset.dimensions["south_north"]) == 4
        assert len(dataset.dimensions["south_north_stag"]) == 5
        assert len(dataset.dimensions["bottom_top"]) == 3
        assert len(dataset.dimensions["bottom_top_stag"]) == 4
        assert chartostring(dataset["Times"][:])[0] == "2026-05-25_21:00:00"
        assert float(dataset["XTIME"][0]) == 180.0
        assert dataset.getncattr("START_DATE") == "2026-05-25_18:00:00"
        assert dataset.getncattr("WEST-EAST_GRID_DIMENSION") == 6


@pytest.mark.skipif(not REFERENCE.is_file(), reason="Gen2 reference wrfout unavailable")
def test_dim_and_attr_conformance_against_reference_schema(tmp_path: Path):
    path, _ = write_small(tmp_path)

    with Dataset(REFERENCE) as reference, Dataset(path) as candidate:
        for name in MINIMUM_WRFOUT_VARIABLES:
            assert name in reference.variables
            ref_var = reference.variables[name]
            out_var = candidate.variables[name]
            assert tuple(out_var.dimensions) == tuple(ref_var.dimensions)
            assert str(np.dtype(out_var.dtype)) == str(np.dtype(ref_var.dtype))
            for attr in ("units", "description", "MemoryOrder", "stagger"):
                if attr not in ref_var.ncattrs() and attr not in out_var.ncattrs():
                    continue
                assert str(out_var.getncattr(attr)) == str(ref_var.getncattr(attr))


def test_total_state_split_writes_base_and_perturbation_pairs(tmp_path: Path):
    path, state = write_small(tmp_path)

    with Dataset(path) as dataset:
        np.testing.assert_allclose(dataset["P"][0], state.p_perturbation)
        np.testing.assert_allclose(dataset["PB"][0], state.p_total - state.p_perturbation)
        np.testing.assert_allclose(dataset["P"][0] + dataset["PB"][0], state.p_total)
        np.testing.assert_allclose(dataset["PH"][0], state.ph_perturbation)
        np.testing.assert_allclose(dataset["PHB"][0], state.ph_total - state.ph_perturbation)
        np.testing.assert_allclose(dataset["PH"][0] + dataset["PHB"][0], state.ph_total)
        np.testing.assert_allclose(dataset["MU"][0], state.mu_perturbation)
        np.testing.assert_allclose(dataset["MUB"][0], state.mu_total - state.mu_perturbation)
        np.testing.assert_allclose(dataset["MU"][0] + dataset["MUB"][0], state.mu_total)


def test_variable_specs_cover_minimum_set():
    # 42 = the historical 41 + THM (v0.14: State.theta is moist theta_m, the
    # writer decouples dry T and emits THM like WRF use_theta_m=1 wrfout).
    assert len(MINIMUM_WRFOUT_VARIABLES) == 42
    # Every minimum-set field (except the special-cased Times string var) must
    # have a spec. P0-5a ADDS operational fields, so the spec dict is a SUPERSET
    # of the minimum set rather than exactly equal.
    assert set(MINIMUM_WRFOUT_VARIABLES) - {"Times"} <= set(WRFOUT_VARIABLE_SPECS)


@pytest.mark.skipif(not FULL_REFERENCE.is_file(), reason="Gen2 full-reference wrfout unavailable")
def test_full_wrfout_mode_matches_reference_variable_list_and_schema(tmp_path: Path):
    state, grid, namelist = synthetic_case()
    path = tmp_path / "wrfout_d02_2026-05-25_21:00:00"
    write_wrfout_netcdf(
        state,
        grid,
        namelist,
        path,
        domain="d02",
        domain_authority=writer_authority(grid),
        valid_time=datetime(2026, 5, 25, 21),
        lead_hours=3.0,
        run_start=datetime(2026, 5, 25, 18),
        full_variable_set=True,
    )

    with Dataset(FULL_REFERENCE) as reference, Dataset(path) as candidate:
        ref_names = list(reference.variables)
        assert len(FULL_WRFOUT_VARIABLES) == 375
        assert list(FULL_WRFOUT_VARIABLES) == ref_names
        assert list(candidate.variables) == ref_names
        for name in ref_names:
            if name == "Times":
                continue
            ref_var = reference.variables[name]
            out_var = candidate.variables[name]
            assert tuple(out_var.dimensions) == tuple(ref_var.dimensions), name
            assert np.dtype(out_var.dtype) == np.dtype(ref_var.dtype), name
            assert np.isfinite(np.asarray(out_var[:])).all(), name


def test_full_wrfout_mode_is_opt_in(tmp_path: Path):
    path, _ = write_small(tmp_path)
    with Dataset(path) as dataset:
        assert "VAR_SSO" not in dataset.variables
        assert list(dataset.variables) != list(FULL_WRFOUT_VARIABLES)


def test_full_prepared_payload_can_write_auxhist_full_subset(tmp_path: Path):
    state, grid, namelist = synthetic_case()
    prepared = prepare_wrfout_payload(
        state,
        grid,
        namelist,
        tmp_path / "wrfout_d02_2026-05-25_21:00:00",
        domain="d02",
        domain_authority=writer_authority(grid),
        valid_time=datetime(2026, 5, 25, 21),
        lead_hours=3.0,
        run_start=datetime(2026, 5, 25, 18),
        full_variable_set=True,
    )
    aux_path = tmp_path / "auxhist1_d02_2026-05-25_21:00:00"
    write_prepared_wrfout(
        prepared,
        expected_domain="d02",
        expected_domain_authority=prepared.domain_authority,
        variable_subset=FULL_WRFOUT_VARIABLES,
        target_override=aux_path,
    )
    with Dataset(aux_path) as dataset:
        assert list(dataset.variables) == list(FULL_WRFOUT_VARIABLES)
        assert len(dataset.variables) == 375


def _full_operational_case():
    """A real ``GridSpec`` (with resident ``DycoreMetrics``) + a State surrogate
    carrying every leaf the writer reads, plus a minimal Noah-MP land carry.

    Exercises the v0.12.0 A1 grid-metric routing + A2 derived diagnostics on
    CPU without a GPU ``State.zeros`` allocation. The State surrogate uses the
    exact operational leaf names so the writer routing path is identical.
    """

    from gpuwrf.contracts.grid import GridSpec

    grid = GridSpec.canary_3km_template()
    nx, ny, nz = grid.nx, grid.ny, grid.nz

    def f3(*shape, val=0.0):
        return val + np.zeros(shape, dtype=np.float32)

    z3 = np.arange(nz, dtype=np.float32)[:, None, None]
    zf = np.arange(nz + 1, dtype=np.float32)[:, None, None]
    _, x2 = np.indices((ny, nx), dtype=np.float32)
    terrain = np.asarray(grid.terrain_height)
    pb = (90_000.0 - 800.0 * z3 + terrain[None]).astype(np.float32)
    p_pert = (100.0 + 0.5 * z3).astype(np.float32)
    phb = (9.81 * (terrain[None] + 600.0 * zf)).astype(np.float32)
    ph_pert = (3.0 + 0.2 * zf).astype(np.float32)
    mub = (85_000.0 + terrain).astype(np.float32)
    mu_pert = (40.0 + 0.2 * x2).astype(np.float32)
    state = SimpleNamespace(
        u=f3(nz, ny, nx + 1, val=4.0), v=f3(nz, ny + 1, nx, val=1.5), w=f3(nz + 1, ny, nx),
        theta=(300.0 + z3).astype(np.float32), qv=f3(nz, ny, nx, val=0.009),
        qc=f3(nz, ny, nx, val=1e-5), qi=f3(nz, ny, nx, val=2e-6), qr=f3(nz, ny, nx, val=3e-6),
        qs=f3(nz, ny, nx, val=1e-6), qg=f3(nz, ny, nx, val=5e-7),
        Ni=f3(nz, ny, nx, val=1e3), Nr=f3(nz, ny, nx, val=1e2),
        Ns=f3(nz, ny, nx, val=50.0), Ng=f3(nz, ny, nx, val=10.0),
        Nc=f3(nz, ny, nx, val=1e8), Nn=f3(nz, ny, nx, val=1e8), qke=f3(nz, ny, nx, val=0.5),
        p_total=pb + p_pert, p_perturbation=p_pert,
        ph_total=phb + ph_pert, ph_perturbation=ph_pert,
        mu_total=mub + mu_pert, mu_perturbation=mu_pert,
        u10=f3(ny, nx, val=4.2), v10=f3(ny, nx, val=1.1), t2=f3(ny, nx, val=289.0),
        q2=f3(ny, nx, val=0.008), psfc=(pb[0] + p_pert[0]).astype(np.float32),
        rain_acc=f3(ny, nx, val=2.0), snow_acc=f3(ny, nx, val=0.5),
        graupel_acc=f3(ny, nx, val=0.1), ice_acc=f3(ny, nx, val=0.05),
        swdown=f3(ny, nx, val=500.0), glw=f3(ny, nx, val=300.0), pblh=f3(ny, nx, val=800.0),
        ustar=f3(ny, nx, val=0.3), hfx=f3(ny, nx, val=20.0), lh=f3(ny, nx, val=70.0),
        t_skin=f3(ny, nx, val=290.0), cldfra=f3(nz, ny, nx, val=0.25), landmask=f3(ny, nx, val=1.0),
    )
    land = SimpleNamespace(
        tslb=f3(4, ny, nx, val=285.0), smois=f3(4, ny, nx, val=0.3), sh2o=f3(4, ny, nx, val=0.28),
        sneqv=f3(ny, nx, val=10.0), snowh=f3(ny, nx, val=0.05),
        canliq=f3(ny, nx, val=0.1), canice=f3(ny, nx, val=0.02),
        sfcrunoff=f3(ny, nx, val=0.0), udrunoff=f3(ny, nx, val=0.0),
        albedo=f3(ny, nx, val=0.2), emiss=f3(ny, nx, val=0.98),
        # B3: Noah-MP snow-layer (NSNOW=3) + snow+soil (NSNOW+NSOIL=7) columns
        # plus the scalar snow/canopy diagnostics (KI-3). ISNOW is the int32
        # active-layer count; here a single active layer (-1).
        tsno=f3(3, ny, nx, val=270.0), snice=f3(3, ny, nx, val=3.0),
        snliq=f3(3, ny, nx, val=0.5), zsnso=f3(7, ny, nx, val=-0.1),
        isnow=np.full((ny, nx), -1, dtype=np.int32),
        sneqvo=f3(ny, nx, val=9.5),
    )
    namelist = {"cen_lat": 28.3, "cen_lon": -15.6, "soil_layers_stag": 4}
    return state, grid, namelist, land


def test_grid_metric_and_derived_diagnostics_present_and_finite(tmp_path: Path):
    state, grid, namelist, land = _full_operational_case()
    path = tmp_path / "wrfout_d02_2026-05-25_21:00:00"
    write_wrfout_netcdf(
        state, grid, namelist, path,
        domain="d02", domain_authority=writer_authority(grid),
        valid_time=datetime(2026, 5, 25, 21), lead_hours=3.0,
        run_start=datetime(2026, 5, 25, 18), land_state=land,
    )

    with Dataset(path) as dataset:
        # v0.12.0 raised the coverage well past the old ~74-variable operational
        # subset; the full state must now emit 100+ variables.
        assert len(dataset.variables) >= 100

        # Every A1 grid-metric extra + A2 diagnostic is present and finite.
        for name in (*GRID_METRIC_EXTRA_VARIABLES, *DERIVED_DIAGNOSTIC_VARIABLES):
            assert name in dataset.variables, f"missing new variable {name}"
            arr = np.asarray(dataset.variables[name][:])
            assert np.isfinite(arr).all(), f"{name} has non-finite values"

        # Physical sanity on the cheap derived diagnostics.
        sr = np.asarray(dataset["SR"][:])
        assert np.all((sr >= 0.0) & (sr <= 1.0))
        coszen = np.asarray(dataset["COSZEN"][:])
        assert np.all((coszen >= -1.0) & (coszen <= 1.0))
        snowc = np.asarray(dataset["SNOWC"][:])
        assert set(np.unique(snowc)).issubset({0.0, 1.0})
        # SNOWC == 1 here because the land carry has SWE > 0 everywhere.
        assert np.all(snowc == 1.0)
        # CLAT mirrors XLAT exactly.
        np.testing.assert_allclose(np.asarray(dataset["CLAT"][:]), np.asarray(dataset["XLAT"][:]))
        # RDX/RDY are the inverse grid lengths.
        np.testing.assert_allclose(float(dataset["RDX"][0]), 1.0 / 3000.0, rtol=1e-5)
        np.testing.assert_allclose(float(dataset["RDY"][0]), 1.0 / 3000.0, rtol=1e-5)


def test_rainnc_is_wrf_all_phase_total(tmp_path: Path):
    """RAINNC must follow WRF's wrfout convention (module_mp_thompson.F:1298-1306).

    WRF accumulates RAINNC = rain + snow + graupel + ice (the ALL-PHASE total),
    with SNOWNC = snow + ice and GRAUPELNC = graupel as overlapping subsets. The
    GPU State keeps DISJOINT accumulators (rain_acc=liquid only), so the writer
    must fold them into the total. A rain-only RAINNC under-counts frozen Alpine
    precip (the v0.14 Switzerland 72h RAINNC gate miss) and lets SNOWNC > RAINNC,
    which is impossible in WRF.
    """

    state, grid, namelist, land = _full_operational_case()
    # _full_operational_case sets rain_acc=2.0, snow_acc=0.5, graupel_acc=0.1,
    # ice_acc=0.05 -> WRF total RAINNC = 2.65, SNOWNC = 0.55, GRAUPELNC = 0.1.
    path = tmp_path / "wrfout_d02_2026-05-25_21:00:00"
    write_wrfout_netcdf(
        state, grid, namelist, path,
        domain="d02", domain_authority=writer_authority(grid),
        valid_time=datetime(2026, 5, 25, 21), lead_hours=3.0,
        run_start=datetime(2026, 5, 25, 18), land_state=land,
    )
    with Dataset(path) as dataset:
        rainnc = np.asarray(dataset["RAINNC"][0])
        snownc = np.asarray(dataset["SNOWNC"][0])
        graupelnc = np.asarray(dataset["GRAUPELNC"][0])
        # All-phase total, not rain-only.
        np.testing.assert_allclose(rainnc, 2.65, rtol=0, atol=1e-4)
        np.testing.assert_allclose(snownc, 0.55, rtol=0, atol=1e-4)
        np.testing.assert_allclose(graupelnc, 0.10, rtol=0, atol=1e-4)
        # WRF invariants: RAINNC is the superset of every frozen channel.
        assert np.all(rainnc >= snownc - 1e-6), "RAINNC must include SNOWNC"
        assert np.all(rainnc >= graupelnc - 1e-6), "RAINNC must include GRAUPELNC"


def test_existing_variables_unchanged_when_metrics_added(tmp_path: Path):
    """The A1/A2 additions must not perturb any pre-existing variable's values."""

    state, grid, namelist, land = _full_operational_case()
    path = tmp_path / "wrfout_d02_2026-05-25_21:00:00"
    write_wrfout_netcdf(
        state, grid, namelist, path,
        domain="d02", domain_authority=writer_authority(grid),
        valid_time=datetime(2026, 5, 25, 21), lead_hours=3.0,
        run_start=datetime(2026, 5, 25, 18), land_state=land,
    )
    with Dataset(path) as dataset:
        # The state/perturbation split and surface fields are untouched.
        np.testing.assert_allclose(dataset["P"][0] + dataset["PB"][0], state.p_total)
        np.testing.assert_allclose(dataset["MU"][0] + dataset["MUB"][0], state.mu_total)
        np.testing.assert_allclose(dataset["U10"][0], state.u10)
        np.testing.assert_allclose(dataset["T2"][0], state.t2)
        # MAPFAC_M (primary) still aliases the x-direction mass map factor.
        np.testing.assert_allclose(dataset["MAPFAC_M"][0], dataset["MAPFAC_MX"][0])


# B3 (KI-3): Noah-MP snow-layer + canopy diagnostics. The expected schema is the
# authoritative reference wrfout (ncdump -h). Each tuple is
# (name, dimensions, MemoryOrder, stagger, units, description, dtype).
SNOW_CANOPY_REFERENCE_SCHEMA = (
    ("TSNO", ("Time", "snow_layers_stag", "south_north", "west_east"),
     "XYZ", "Z", "K", "snow temperature", "f4"),
    ("SNICE", ("Time", "snow_layers_stag", "south_north", "west_east"),
     "XYZ", "Z", "mm", "snow layer ice", "f4"),
    ("SNLIQ", ("Time", "snow_layers_stag", "south_north", "west_east"),
     "XYZ", "Z", "mm", "snow layer liquid", "f4"),
    ("ZSNSO", ("Time", "snso_layers_stag", "south_north", "west_east"),
     "XYZ", "Z", "m", "layer-bottom depth from snow surf", "f4"),
    ("ISNOW", ("Time", "south_north", "west_east"),
     "XY ", "", "m3 m-3", "no. of snow layer", "i4"),
    ("SNEQVO", ("Time", "south_north", "west_east"),
     "XY ", "", "mm", "snow mass at last time step", "f4"),
    ("CANLIQ", ("Time", "south_north", "west_east"),
     "XY ", "", "mm", "intercepted liquid water", "f4"),
    ("CANICE", ("Time", "south_north", "west_east"),
     "XY ", "", "mm", "intercepted ice mass", "f4"),
)

_B3_REFERENCE = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l3/"
    "20260428_18z_l3_24h_20260525T221139Z/"
    "wrfout_d02_2026-04-28_19:00:00"
)


def test_noahmp_snow_canopy_diagnostics_match_reference_schema(tmp_path: Path):
    """B3/KI-3: TSNO/SNICE/SNLIQ/ZSNSO + ISNOW/SNEQVO/CANLIQ/CANICE are emitted
    with the EXACT shape/staggering/dtype/attrs of the reference WRF wrfout, are
    finite, and round-trip the values handed in by the Noah-MP land carry."""

    state, grid, namelist, land = _full_operational_case()
    nx, ny = grid.nx, grid.ny
    path = tmp_path / "wrfout_d02_2026-05-25_21:00:00"
    write_wrfout_netcdf(
        state, grid, namelist, path,
        domain="d02", domain_authority=writer_authority(grid),
        valid_time=datetime(2026, 5, 25, 21), lead_hours=3.0,
        run_start=datetime(2026, 5, 25, 18), land_state=land,
    )

    # Cross-check the hardcoded expected schema against the live reference wrfout
    # when it is present on disk (defensive: skip the cross-check if purged).
    ref_attrs = {}
    if _B3_REFERENCE.exists():
        with Dataset(_B3_REFERENCE) as ref:
            for name, *_ in SNOW_CANOPY_REFERENCE_SCHEMA:
                assert name in ref.variables, f"{name} missing from reference"
                rv = ref.variables[name]
                ref_attrs[name] = {
                    "dimensions": tuple(rv.dimensions),
                    "MemoryOrder": str(rv.getncattr("MemoryOrder")),
                    "stagger": str(rv.getncattr("stagger")),
                    "units": str(rv.getncattr("units")),
                    "description": str(rv.getncattr("description")),
                    "kind": np.dtype(rv.dtype).kind,
                }

    with Dataset(path) as dataset:
        assert int(dataset.dimensions["snow_layers_stag"].size) == 3
        assert int(dataset.dimensions["snso_layers_stag"].size) == 7

        for name, dims, mem, stag, units, desc, dtype in SNOW_CANOPY_REFERENCE_SCHEMA:
            assert name in dataset.variables, f"missing snow/canopy var {name}"
            var = dataset.variables[name]
            assert tuple(var.dimensions) == dims, f"{name} dims {var.dimensions} != {dims}"
            assert str(var.getncattr("MemoryOrder")) == mem
            assert str(var.getncattr("stagger")) == stag
            assert str(var.getncattr("units")) == units
            assert str(var.getncattr("description")) == desc
            expected_kind = "i" if dtype.startswith("i") else "f"
            assert np.dtype(var.dtype).kind == expected_kind, f"{name} dtype kind"
            arr = np.asarray(var[:])
            assert np.isfinite(arr).all(), f"{name} non-finite"

            # Hardcoded schema must equal the live reference when available.
            if name in ref_attrs:
                r = ref_attrs[name]
                assert r["dimensions"] == dims, f"{name} ref dims drift"
                assert r["MemoryOrder"] == mem, f"{name} ref MemoryOrder drift"
                assert r["stagger"] == stag, f"{name} ref stagger drift"
                assert r["units"] == units, f"{name} ref units drift"
                assert r["description"] == desc, f"{name} ref description drift"
                assert r["kind"] == expected_kind, f"{name} ref dtype-kind drift"

        # FieldType: WRF tags integer fields 106, real fields 104.
        assert int(dataset.variables["ISNOW"].getncattr("FieldType")) == 106
        assert int(dataset.variables["TSNO"].getncattr("FieldType")) == 104

        # Values round-trip from the carry (no fabrication / no rescale).
        np.testing.assert_allclose(dataset["TSNO"][0], land.tsno)
        np.testing.assert_allclose(dataset["SNICE"][0], land.snice)
        np.testing.assert_allclose(dataset["SNLIQ"][0], land.snliq)
        np.testing.assert_allclose(dataset["ZSNSO"][0], land.zsnso)
        np.testing.assert_array_equal(np.asarray(dataset["ISNOW"][0]), land.isnow)
        np.testing.assert_allclose(dataset["SNEQVO"][0], land.sneqvo)
        np.testing.assert_allclose(dataset["CANLIQ"][0], land.canliq)
        np.testing.assert_allclose(dataset["CANICE"][0], land.canice)
        # CANWAT is still the canliq+canice bulk (unchanged behaviour).
        np.testing.assert_allclose(dataset["CANWAT"][0], land.canliq + land.canice)
        assert dataset["TSNO"][0].shape == (3, ny, nx)
        assert dataset["ZSNSO"][0].shape == (7, ny, nx)


def test_snow_canopy_diagnostics_self_gate_without_carry(tmp_path: Path):
    """When NO land carry is supplied the snow/canopy diagnostics are simply
    absent — the writer never fabricates a snow profile (honesty rule)."""

    state, grid, namelist, _land = _full_operational_case()
    path = tmp_path / "wrfout_d02_2026-05-25_21:00:00"
    write_wrfout_netcdf(
        state, grid, namelist, path,
        domain="d02", domain_authority=writer_authority(grid),
        valid_time=datetime(2026, 5, 25, 21), lead_hours=3.0,
        run_start=datetime(2026, 5, 25, 18), land_state=None,
    )
    with Dataset(path) as dataset:
        for name, *_ in SNOW_CANOPY_REFERENCE_SCHEMA:
            assert name not in dataset.variables, f"{name} fabricated without carry"


# --------------------------------------------------------------------------
# B1 (v0.12.0): RRTMG up/down all-sky radiation flux diagnostics into wrfout.
# --------------------------------------------------------------------------

# Expected reference schema (dims/MemoryOrder/stagger/units/description/dtype),
# copied verbatim from the Gen2 reference wrfout_d02 (ncdump -h). All-sky only;
# the clear-sky ``...C`` vars are intentionally NOT emitted (see SKIPPED below).
RADIATION_FLUX_REFERENCE_SCHEMA = (
    ("SWDNB", ("Time", "south_north", "west_east"), "XY ", "",
     "W m-2", "INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT BOTTOM", "f4"),
    ("SWUPB", ("Time", "south_north", "west_east"), "XY ", "",
     "W m-2", "INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT BOTTOM", "f4"),
    ("LWDNB", ("Time", "south_north", "west_east"), "XY ", "",
     "W m-2", "INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT BOTTOM", "f4"),
    ("LWUPB", ("Time", "south_north", "west_east"), "XY ", "",
     "W m-2", "INSTANTANEOUS UPWELLING LONGWAVE FLUX AT BOTTOM", "f4"),
    ("SWDNT", ("Time", "south_north", "west_east"), "XY ", "",
     "W m-2", "INSTANTANEOUS DOWNWELLING SHORTWAVE FLUX AT TOP", "f4"),
    ("SWUPT", ("Time", "south_north", "west_east"), "XY ", "",
     "W m-2", "INSTANTANEOUS UPWELLING SHORTWAVE FLUX AT TOP", "f4"),
    ("LWDNT", ("Time", "south_north", "west_east"), "XY ", "",
     "W m-2", "INSTANTANEOUS DOWNWELLING LONGWAVE FLUX AT TOP", "f4"),
    ("LWUPT", ("Time", "south_north", "west_east"), "XY ", "",
     "W m-2", "INSTANTANEOUS UPWELLING LONGWAVE FLUX AT TOP", "f4"),
    ("OLR", ("Time", "south_north", "west_east"), "XY ", "",
     "W m-2", "TOA OUTGOING LONG WAVE", "f4"),
    ("SWNORM", ("Time", "south_north", "west_east"), "XY ", "",
     "W m-2", "NORMAL SHORT WAVE FLUX AT GROUND SURFACE (SLOPE-DEPENDENT)", "f4"),
)

# Clear-sky flux vars the RRTMG port does NOT produce (no separate clear-sky
# radiative-transfer pass) and therefore deliberately does not emit (no
# fabrication). The B1 self-gate test asserts none of these ever appear.
RADIATION_FLUX_SKIPPED_CLEARSKY = (
    "SWDNBC", "SWUPBC", "LWDNBC", "LWUPBC",
    "SWDNTC", "SWUPTC", "LWDNTC", "LWUPTC",
)

_B1_REFERENCE = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l3/"
    "20260428_18z_l3_24h_20260525T221139Z/"
    "wrfout_d02_2026-04-28_19:00:00"
)


def _radiation_flux_diagnostics(ny: int, nx: int) -> dict[str, np.ndarray]:
    """A synthetic operational radiation-diagnostics map (mass-point, W m-2).

    Mirrors the ``M9Diagnostics -> _surface_diagnostics_for_output`` mapping that
    the live pipeline hands the writer: physically plausible all-sky up/down
    surface + TOA SW/LW fluxes plus SWNORM, and the SWDOWN/GLW the consistency
    asserts pin against. Values chosen distinct per field so a mis-wire is caught.
    """
    y, x = np.indices((ny, nx), dtype=np.float64)
    swdown = 500.0 + 3.0 * x + y          # bottom-of-atmosphere downwelling SW
    glw = 320.0 + 2.0 * y                 # bottom-of-atmosphere downwelling LW
    lwupt = 240.0 + 1.5 * x               # TOA upwelling LW (== OLR)
    return {
        "SWDOWN": swdown,
        "GLW": glw,
        "SWDNB": swdown,                  # WRF SWDNB == SWDOWN (no-slope config)
        "SWUPB": 0.18 * swdown,           # surface reflected SW
        "LWDNB": glw,                     # WRF LWDNB == GLW
        "LWUPB": 410.0 + 1.0 * y,         # surface emitted LW
        "SWDNT": 800.0 + 2.0 * x,         # TOA incoming SW
        "SWUPT": 120.0 + 0.5 * x,         # TOA reflected SW
        "LWDNT": np.zeros((ny, nx)),      # TOA downwelling LW ~ 0 (no source above)
        "LWUPT": lwupt,
        "SWNORM": swdown / 0.92,          # slope-normal SW (> swdown for tilt)
    }


def test_radiation_flux_diagnostics_match_reference_schema(tmp_path: Path):
    """B1: SWDNB/SWUPB/LWDNB/LWUPB (surface) + SWDNT/SWUPT/LWDNT/LWUPT (TOA) +
    OLR + SWNORM are emitted with the EXACT reference WRF schema, are finite,
    round-trip the diagnostics handed in, satisfy the radiative consistency
    relations (SWDNB==SWDOWN, LWDNB==GLW, OLR==LWUPT), and the clear-sky ``...C``
    vars are never fabricated."""

    state, grid, namelist = synthetic_case()
    nx, ny = grid.nx, grid.ny
    diagnostics = _radiation_flux_diagnostics(ny, nx)
    path = tmp_path / "wrfout_d02_2026-04-28_19:00:00"
    write_wrfout_netcdf(
        state, grid, namelist, path,
        domain="d02", domain_authority=writer_authority(grid),
        valid_time=datetime(2026, 4, 28, 19), lead_hours=1.0,
        run_start=datetime(2026, 4, 28, 18), diagnostics=diagnostics,
    )

    # Defensive cross-check against the live reference wrfout when present.
    ref_attrs: dict[str, dict] = {}
    if _B1_REFERENCE.exists():
        with Dataset(_B1_REFERENCE) as ref:
            for name, *_ in RADIATION_FLUX_REFERENCE_SCHEMA:
                assert name in ref.variables, f"{name} missing from reference"
                rv = ref.variables[name]
                ref_attrs[name] = {
                    "dimensions": tuple(rv.dimensions),
                    "MemoryOrder": str(rv.getncattr("MemoryOrder")),
                    "stagger": str(rv.getncattr("stagger")),
                    "units": str(rv.getncattr("units")),
                    "description": str(rv.getncattr("description")),
                    "kind": np.dtype(rv.dtype).kind,
                }

    with Dataset(path) as dataset:
        for name, dims, mem, stag, units, desc, dtype in RADIATION_FLUX_REFERENCE_SCHEMA:
            assert name in dataset.variables, f"missing radiation flux var {name}"
            var = dataset.variables[name]
            assert tuple(var.dimensions) == dims, f"{name} dims {var.dimensions} != {dims}"
            assert str(var.getncattr("MemoryOrder")) == mem, f"{name} MemoryOrder"
            assert str(var.getncattr("stagger")) == stag, f"{name} stagger"
            assert str(var.getncattr("units")) == units, f"{name} units"
            assert str(var.getncattr("description")) == desc, f"{name} description"
            assert np.dtype(var.dtype).kind == "f", f"{name} dtype kind"
            assert int(var.getncattr("FieldType")) == 104, f"{name} FieldType"
            arr = np.asarray(var[:])
            assert np.isfinite(arr).all(), f"{name} non-finite"

            if name in ref_attrs:
                r = ref_attrs[name]
                assert r["dimensions"] == dims, f"{name} ref dims drift"
                assert r["MemoryOrder"] == mem, f"{name} ref MemoryOrder drift"
                assert r["stagger"] == stag, f"{name} ref stagger drift"
                assert r["units"] == units, f"{name} ref units drift"
                assert r["description"] == desc, f"{name} ref description drift"
                assert r["kind"] == "f", f"{name} ref dtype-kind drift"

        # Values round-trip from the diagnostics map (no rescale / no fabrication).
        np.testing.assert_allclose(dataset["SWDNB"][0], diagnostics["SWDNB"], rtol=1e-6)
        np.testing.assert_allclose(dataset["SWUPB"][0], diagnostics["SWUPB"], rtol=1e-6)
        np.testing.assert_allclose(dataset["LWDNB"][0], diagnostics["LWDNB"], rtol=1e-6)
        np.testing.assert_allclose(dataset["LWUPB"][0], diagnostics["LWUPB"], rtol=1e-6)
        np.testing.assert_allclose(dataset["SWDNT"][0], diagnostics["SWDNT"], rtol=1e-6)
        np.testing.assert_allclose(dataset["SWUPT"][0], diagnostics["SWUPT"], rtol=1e-6)
        np.testing.assert_allclose(dataset["LWDNT"][0], diagnostics["LWDNT"], atol=1e-6)
        np.testing.assert_allclose(dataset["LWUPT"][0], diagnostics["LWUPT"], rtol=1e-6)
        np.testing.assert_allclose(dataset["SWNORM"][0], diagnostics["SWNORM"], rtol=1e-6)

        # Radiative consistency relations (the B1 acceptance asserts):
        #   surface SWDNB == SWDOWN, surface LWDNB == GLW, OLR == LWUPT.
        np.testing.assert_allclose(dataset["SWDNB"][0], dataset["SWDOWN"][0], rtol=1e-6)
        np.testing.assert_allclose(dataset["LWDNB"][0], dataset["GLW"][0], rtol=1e-6)
        np.testing.assert_allclose(dataset["OLR"][0], dataset["LWUPT"][0], rtol=1e-6)

        # No clear-sky flux var is ever fabricated (no clear-sky pass exists).
        for name in RADIATION_FLUX_SKIPPED_CLEARSKY:
            assert name not in dataset.variables, f"{name} clear-sky fabricated"


def test_radiation_flux_diagnostics_self_gate_without_diagnostics(tmp_path: Path):
    """When NO radiation diagnostics are supplied the B1 flux vars are simply
    absent — the writer never fabricates a radiation flux (honesty rule)."""

    state, grid, namelist = synthetic_case()
    path = tmp_path / "wrfout_d02_2026-04-28_19:00:00"
    write_wrfout_netcdf(
        state, grid, namelist, path,
        domain="d02", domain_authority=writer_authority(grid),
        valid_time=datetime(2026, 4, 28, 19), lead_hours=1.0,
        run_start=datetime(2026, 4, 28, 18), diagnostics=None,
    )
    with Dataset(path) as dataset:
        for name in RADIATION_FLUX_DIAGNOSTIC_VARIABLES:
            assert name not in dataset.variables, f"{name} fabricated without diagnostics"
        for name in RADIATION_FLUX_SKIPPED_CLEARSKY:
            assert name not in dataset.variables, f"{name} clear-sky fabricated"


# --------------------------------------------------------------------------
# v0.23.4 authenticated global GRID_ID metadata repair
# --------------------------------------------------------------------------


def _prepare_grid_id_case(tmp_path: Path, domain: str = "d02"):
    state, grid, namelist = synthetic_case()
    authority = writer_authority(grid, domain)
    prepared = prepare_wrfout_payload(
        state, grid, namelist, tmp_path / "shape_and_filename_are_not_authority.nc",
        domain=domain, domain_authority=authority,
        valid_time=datetime(2026, 5, 25, 21), lead_hours=3.0,
        run_start=datetime(2026, 5, 25, 18),
    )
    return prepared, authority


def _value_bytes(value):
    array = np.ma.asarray(value)
    return (
        str(array.dtype), tuple(array.shape),
        np.ascontiguousarray(np.ma.getdata(array)).tobytes(),
        np.ascontiguousarray(np.ma.getmaskarray(array)).tobytes(),
    )


def _attr_bytes(value):
    return ("str", value) if isinstance(value, str) else _value_bytes(value)


def _assert_netcdf_equal_except_grid_id(before: Path, after: Path) -> None:
    with Dataset(before) as left, Dataset(after) as right:
        assert list(left.dimensions) == list(right.dimensions)
        for name in left.dimensions:
            assert len(left.dimensions[name]) == len(right.dimensions[name])
            assert left.dimensions[name].isunlimited() == right.dimensions[name].isunlimited()
        assert list(left.variables) == list(right.variables)
        for name in left.variables:
            lvar, rvar = left.variables[name], right.variables[name]
            assert np.dtype(lvar.dtype) == np.dtype(rvar.dtype), name
            assert lvar.dimensions == rvar.dimensions, name
            assert lvar.shape == rvar.shape, name
            assert lvar.ncattrs() == rvar.ncattrs(), name
            for attr in lvar.ncattrs():
                assert _attr_bytes(lvar.getncattr(attr)) == _attr_bytes(rvar.getncattr(attr)), (name, attr)
            assert _value_bytes(lvar[:]) == _value_bytes(rvar[:]), name
        assert "GRID_ID" not in left.ncattrs()
        assert int(right.getncattr("GRID_ID")) == 2
        assert [name for name in right.ncattrs() if name != "GRID_ID"] == left.ncattrs()
        for attr in left.ncattrs():
            assert _attr_bytes(left.getncattr(attr)) == _attr_bytes(right.getncattr(attr)), attr


def test_grid_id_is_only_semantic_change_for_every_written_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import gpuwrf.io.wrfout_writer as writer_module

    prepared, _ = _prepare_grid_id_case(tmp_path)
    before, after = tmp_path / "before.nc", tmp_path / "after.nc"
    original = writer_module._write_global_attrs

    def legacy_global_attrs(dataset, grid, namelist, dimensions, run_start, valid_time, _authority):
        return original(dataset, grid, namelist, dimensions, run_start, valid_time, None)

    monkeypatch.setattr(writer_module, "_write_global_attrs", legacy_global_attrs)
    write_prepared_wrfout(
        prepared, expected_domain="d02",
        expected_domain_authority=prepared.domain_authority, target_override=before
    )
    monkeypatch.setattr(writer_module, "_write_global_attrs", original)
    write_prepared_wrfout(
        prepared, expected_domain="d02",
        expected_domain_authority=prepared.domain_authority, target_override=after
    )
    _assert_netcdf_equal_except_grid_id(before, after)


@pytest.mark.parametrize(("domain", "expected"), (("d01", 1), ("d02", 2), ("d03", 3)))
def test_exact_domain_authority_writes_exact_grid_id_despite_path_spoof(
    tmp_path: Path, domain: str, expected: int
):
    prepared, _ = _prepare_grid_id_case(tmp_path, domain)
    target = tmp_path / f"wrfout_d{(expected % 3) + 1:02d}_spoofed_shape_name"
    write_prepared_wrfout(
        prepared, expected_domain=domain,
        expected_domain_authority=prepared.domain_authority, target_override=target
    )
    with Dataset(target) as dataset:
        assert type(dataset.getncattr("GRID_ID")) is np.int32
        assert int(dataset.getncattr("GRID_ID")) == expected


def test_missing_wrong_rehashed_and_extent_substituted_authority_fail_closed(tmp_path: Path):
    state, grid, namelist = synthetic_case()
    authority = writer_authority(grid, "d02")
    common = dict(valid_time=datetime(2026, 5, 25, 21), lead_hours=3.0,
                  run_start=datetime(2026, 5, 25, 18))
    with pytest.raises(ValueError, match="WRFOUT_DOMAIN_AUTHORITY_MISSING_OR_TYPE"):
        prepare_wrfout_payload(state, grid, namelist, tmp_path / "missing",
                               domain="d02", domain_authority=None, **common)
    with pytest.raises(ValueError, match="WRFOUT_DOMAIN_AUTHORITY_GRID_ID_MISMATCH"):
        prepare_wrfout_payload(state, grid, namelist, tmp_path / "wrong",
                               domain="d02", domain_authority=replace(authority, grid_id=3), **common)

    substituted_payload = _domain_authority_payload(
        domain="d03", grid_id=3, mass_nx=grid.nx, mass_ny=grid.ny, mass_nz=grid.nz,
        e_we=grid.nx + 1, e_sn=grid.ny + 1, e_vert=grid.nz + 1,
    )
    rehashed = WrfoutDomainAuthority(
        **substituted_payload, authority_sha256=_domain_authority_sha256(substituted_payload)
    )
    with pytest.raises(ValueError, match="WRFOUT_DOMAIN_AUTHORITY_DOMAIN_MISMATCH"):
        prepare_wrfout_payload(state, grid, namelist, tmp_path / "wrfout_d03_spoof",
                               domain="d02", domain_authority=rehashed, **common)

    rehashed_grid_id_payload = _domain_authority_payload(
        domain="d02", grid_id=3, mass_nx=grid.nx, mass_ny=grid.ny, mass_nz=grid.nz,
        e_we=grid.nx + 1, e_sn=grid.ny + 1, e_vert=grid.nz + 1,
    )
    rehashed_grid_id = WrfoutDomainAuthority(
        **rehashed_grid_id_payload,
        authority_sha256=_domain_authority_sha256(rehashed_grid_id_payload),
    )
    with pytest.raises(ValueError, match="WRFOUT_DOMAIN_AUTHORITY_GRID_ID_MISMATCH"):
        prepare_wrfout_payload(state, grid, namelist, tmp_path / "rehashed_grid_id",
                               domain="d02", domain_authority=rehashed_grid_id, **common)

    extent_payload = _domain_authority_payload(
        domain="d02", grid_id=2, mass_nx=grid.nx + 1, mass_ny=grid.ny, mass_nz=grid.nz,
        e_we=grid.nx + 2, e_sn=grid.ny + 1, e_vert=grid.nz + 1,
    )
    extent_rehashed = WrfoutDomainAuthority(
        **extent_payload, authority_sha256=_domain_authority_sha256(extent_payload)
    )
    with pytest.raises(ValueError, match="WRFOUT_DOMAIN_AUTHORITY_OUTPUT_GRID_MISMATCH"):
        prepare_wrfout_payload(state, grid, namelist, tmp_path / "shape_spoof",
                               domain="d02", domain_authority=extent_rehashed, **common)

    stagger_payload = _domain_authority_payload(
        domain="d02", grid_id=2, mass_nx=grid.nx, mass_ny=grid.ny, mass_nz=grid.nz,
        e_we=grid.nx + 2, e_sn=grid.ny + 1, e_vert=grid.nz + 1,
    )
    stagger_rehashed = WrfoutDomainAuthority(
        **stagger_payload, authority_sha256=_domain_authority_sha256(stagger_payload)
    )
    with pytest.raises(ValueError, match="WRFOUT_DOMAIN_AUTHORITY_EXTENT_RELATION"):
        prepare_wrfout_payload(
            state, grid, namelist, tmp_path / "stagger_spoof",
            domain="d02", domain_authority=stagger_rehashed, **common,
        )


def test_public_sync_rejects_fully_rehashed_domain_authority_cosubstitution(tmp_path: Path):
    prepared, original = _prepare_grid_id_case(tmp_path, "d02")
    d03 = writer_authority(prepared.grid, "d03")
    substituted = replace(prepared, domain="d03", domain_authority=d03)
    target = tmp_path / "cosubstitution-must-not-publish.nc"
    with pytest.raises(ValueError, match="WRFOUT_PREPARED_AUTHORITY_SUBSTITUTION"):
        write_prepared_wrfout(
            substituted,
            expected_domain="d02",
            expected_domain_authority=original,
            target_override=target,
        )
    assert not target.exists()


@pytest.mark.parametrize("existing", (2, 3))
def test_public_sync_rejects_existing_equal_or_conflicting_target_without_truncation(
    tmp_path: Path, existing: int
):
    prepared, authority = _prepare_grid_id_case(tmp_path, "d02")
    target = tmp_path / f"public-existing-{existing}.nc"
    with Dataset(target, "w") as dataset:
        dataset.setncattr("GRID_ID", np.int32(existing))
        dataset.setncattr("SENTINEL", "preserve-exactly")
    before = target.read_bytes()
    with pytest.raises(FileExistsError, match="WRFOUT_TARGET_EXISTS"):
        write_prepared_wrfout(
            prepared,
            expected_domain="d02",
            expected_domain_authority=authority,
            target_override=target,
        )
    assert target.read_bytes() == before
    with Dataset(target) as dataset:
        assert int(dataset.getncattr("GRID_ID")) == existing
        assert dataset.getncattr("SENTINEL") == "preserve-exactly"


def test_publication_race_is_noreplace_and_cleans_private_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import gpuwrf.io.wrfout_writer as writer_module

    prepared, authority = _prepare_grid_id_case(tmp_path, "d02")
    target = tmp_path / "publication-race.nc"
    real_publish = writer_module._publish_wrfout_noreplace

    def racing_publish(temporary, final_target):
        assert final_target == target
        with Dataset(final_target, "w") as dataset:
            dataset.setncattr("GRID_ID", np.int32(3))
            dataset.setncattr("SENTINEL", "racer-won")
        return real_publish(temporary, final_target)

    monkeypatch.setattr(writer_module, "_publish_wrfout_noreplace", racing_publish)
    with pytest.raises(FileExistsError, match="WRFOUT_TARGET_EXISTS"):
        write_prepared_wrfout(
            prepared,
            expected_domain="d02",
            expected_domain_authority=authority,
            target_override=target,
        )
    with Dataset(target) as dataset:
        assert int(dataset.getncattr("GRID_ID")) == 3
        assert dataset.getncattr("SENTINEL") == "racer-won"
    assert list(tmp_path.glob(f".{target.name}.grid-id-*.tmp")) == []


def test_source_grid_substitution_and_missing_identity_fail_before_writer():
    _, grid, _ = synthetic_case()
    with pytest.raises(ValueError, match="SOURCE_TYPE"):
        bind_wrfout_domain_authority(
            "d03", SimpleNamespace(mass_nx=grid.nx, mass_ny=grid.ny, mass_nz=grid.nz), grid
        )
    with pytest.raises(ValueError, match="SOURCE_ID_MISMATCH"):
        bind_wrfout_domain_authority(
            "d03", replace(authenticated_source_grid(grid, "d03"), id="d02"), grid
        )
    with pytest.raises(ValueError, match="OUTPUT_GRID_MISMATCH"):
        bind_wrfout_domain_authority(
            "d03", replace(authenticated_source_grid(grid, "d03"),
                           mass_nx=grid.nx + 1, e_we=grid.nx + 2), grid
        )
    with pytest.raises(ValueError, match="SOURCE_EXTENT_RELATION"):
        bind_wrfout_domain_authority(
            "d03", replace(authenticated_source_grid(grid, "d03"), e_we=999,
                           e_sn=888, e_vert=777), grid
        )


@pytest.mark.parametrize("existing", (2, 3))
def test_duplicate_or_conflicting_grid_id_is_never_overwritten(tmp_path: Path, existing: int):
    _, grid, namelist = synthetic_case()
    authority = writer_authority(grid, "d02")
    path = tmp_path / f"duplicate-{existing}.nc"
    with Dataset(path, "w") as dataset:
        dataset.setncattr("GRID_ID", np.int32(existing))
        with pytest.raises(ValueError, match="WRFOUT_GRID_ID_DUPLICATE_OR_CONFLICT"):
            _write_global_attrs(
                dataset, grid, namelist,
                {"west_east_stag": 6, "south_north_stag": 5, "bottom_top_stag": 4},
                datetime(2026, 5, 25, 18), datetime(2026, 5, 25, 21), authority,
            )
        assert int(dataset.getncattr("GRID_ID")) == existing
        assert dataset.ncattrs() == ["GRID_ID"]


def _load_corrected_gate_module():
    path = Path(__file__).parents[1] / "scripts" / "v0234_corrected_fullbuffer_gate.py"
    spec = importlib.util.spec_from_file_location("v0234_corrected_gate_grid_id_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retained_retry3_frame_reproduces_frame_grid_then_temp_copy_passes(tmp_path: Path):
    retained = Path(
        "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
        "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/"
        "gpu_validation_8d82220d_retry3/gpu-output/"
        "wrfout_d03_2025-03-01_00:20:00"
    )
    if not retained.is_file():
        pytest.skip("immutable retained retry3 frame is unavailable")
    expected_sha = "0f01fd8dde586572318b6fa87654f9893e35ba590fdc029520a252a904f4fe54"
    assert hashlib.sha256(retained.read_bytes()).hexdigest() == expected_sha
    gate = _load_corrected_gate_module()
    valid_time = datetime(2025, 3, 1, 0, 20, tzinfo=timezone.utc)
    with pytest.raises(gate.GateError, match="FRAME_GRID"):
        gate.qa_d03_frame(retained, valid_time)

    repaired = tmp_path / retained.name
    shutil.copyfile(retained, repaired)
    authority = writer_authority(SimpleNamespace(nx=111, ny=93, nz=44), "d03")
    with Dataset(repaired, "r+") as dataset:
        _set_wrfout_grid_id_attr(dataset, authority)
    result = gate.qa_d03_frame(repaired, valid_time)
    assert result["valid_time"] == valid_time.isoformat()
    with Dataset(repaired) as dataset:
        assert int(dataset.getncattr("GRID_ID")) == 3
    assert hashlib.sha256(retained.read_bytes()).hexdigest() == expected_sha
