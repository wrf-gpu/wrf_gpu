"""S3 assembly + met_em-writer structural tests.

Two gates:
1. **Assembly**: a synthetic ForcingFields + TargetGrid drive
   ``assemble_met_em`` and the result passes ``MetEmArtifact.validate`` (incl
   shapes/dims/stagger) for d01/d02/d03 geometry; PRES/GHT surface assembly,
   soil packing, and U/V re-staggering land in the right schema shapes.
2. **Writer structural parity**: the NetCDF written by ``write_met_em`` re-opens
   with the same dims/var-dims/var-attrs/FLAG_* as a real WPS met_em file
   (structural diff = 0; field VALUES are the S4 gate, not here).

The real met_em oracle is read read-only from the validation corpus when present.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from gpuwrf.init import interp_metgrid as im  # noqa: E402
from gpuwrf.init.metgrid_assemble import (  # noqa: E402
    ForcingFields,
    TargetGrid,
    assemble_met_em,
)
from gpuwrf.init.metem_writer import FIELD_TYPE, write_met_em  # noqa: E402
from gpuwrf.init.metgrid_schema import (  # noqa: E402
    ISOBARIC_LEVELS_PA,
    NUM_METGRID_LEVELS,
    MetgridProjection,
    metem_field_specs,
)

netCDF4 = pytest.importorskip("netCDF4")

_ORACLE_GLOB = "<DATA_ROOT>/canairy_meteo/runs/wps_cases/*/l3/met_em.d01.*.nc"


# --- synthetic source grid + forcing -----------------------------------------
def _source_grid():
    # small regional patch of a 0.25-deg grid around the Canaries; lon -22..-10,
    # lat 24..32. Stored as i=lon (first), j=lat (second) on the kernel side, but
    # ForcingFields are (ny_src, nx_src). nx_src=lon count, ny_src=lat count.
    return im.LatLonSourceGrid(
        lon0_deg=-22.0, dlon_deg=0.25, lat0_deg=32.0, dlat_deg=-0.25, nx=49, ny=33,
        global_wrap=False,
    )


def _src_latlon(sg):
    lon = sg.lon0_deg + sg.dlon_deg * np.arange(sg.nx)  # (nx,)
    lat = sg.lat0_deg + sg.dlat_deg * np.arange(sg.ny)  # (ny,)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")  # (ny, nx)
    return LAT, LON


def _make_forcing(sg):
    LAT, LON = _src_latlon(sg)  # (ny, nx)
    ny, nx = LAT.shape

    def smooth2d(amp, base):
        return base + amp * (np.sin(np.radians(LON)) + np.cos(np.radians(LAT)))

    t2 = smooth2d(5.0, 290.0)
    t_iso = np.stack([smooth2d(3.0, 280.0 - 8.0 * k) for k in range(13)])  # (13,ny,nx)
    u_iso = np.stack([smooth2d(2.0, 5.0 + k) for k in range(13)])
    v_iso = np.stack([smooth2d(2.0, -3.0 + 0.5 * k) for k in range(13)])
    gh_iso = np.stack([smooth2d(20.0, 100.0 + 1500.0 * k) for k in range(13)])
    q_iso = np.stack([np.clip(smooth2d(0.001, 0.005 - 3e-4 * k), 0, None) for k in range(13)])
    u10 = smooth2d(1.0, 4.0)
    v10 = smooth2d(1.0, -2.0)
    q2 = np.clip(smooth2d(0.001, 0.006), 0, None)
    psfc = smooth2d(500.0, 100500.0)
    pmsl = smooth2d(300.0, 101300.0)
    orog = np.clip(smooth2d(200.0, 300.0), 0, None)
    skt = smooth2d(4.0, 291.0)
    # land where lon > -18 (rough), water otherwise
    landsea = (LON > -18.0).astype(float)
    st0010 = smooth2d(2.0, 292.0)
    st1040 = smooth2d(2.0, 290.0)
    sm0010 = np.clip(smooth2d(0.05, 0.18), 0, 1)
    sm1040 = np.clip(smooth2d(0.05, 0.16), 0, 1)
    return ForcingFields(
        t_iso=t_iso, u_iso=u_iso, v_iso=v_iso, gh_iso=gh_iso, q_iso=q_iso,
        t2=t2, u10=u10, v10=v10, q2=q2, psfc=psfc, pmsl=pmsl, soilhgt=orog,
        skintemp=skt, landsea=landsea, dewpt=t2 - 5.0,
        st000010=st0010, st010040=st1040, sm000010=sm0010, sm010040=sm1040,
    )


def _make_target_grid(nx, ny):
    # a small target grid inside the source patch (lon -19..-15, lat 27..29)
    lon = np.linspace(-19.0, -15.0, nx)
    lat = np.linspace(27.0, 29.0, ny)
    lat_m, lon_m = np.meshgrid(lat, lon, indexing="ij")  # (ny, nx)
    # U stag: nx+1 in west_east_stag
    lonu = np.linspace(-19.0, -15.0, nx + 1)
    lat_u, lon_u = np.meshgrid(lat, lonu, indexing="ij")  # (ny, nx+1)
    latv = np.linspace(27.0, 29.0, ny + 1)
    lat_v, lon_v = np.meshgrid(latv, lon, indexing="ij")  # (ny+1, nx)
    return TargetGrid(lat_m, lon_m, lat_u, lon_u, lat_v, lon_v)


def _make_static(nx, ny, proj):
    """Minimal static-geog block on the target grid (S2's job). Only the
    mandatory geo_em fields, with plausible shapes; values are placeholders (S4
    grades values, not this test)."""

    sn, we = ny, nx
    static = {}
    rng = np.random.default_rng(0)

    def m(shape):
        return rng.uniform(0, 1, shape).astype(np.float64)

    lon = np.linspace(-19.0, -15.0, nx)
    landmask = (lon[None, :] > -17.0).astype(float) * np.ones((sn, we))
    static["XLAT_M"] = m((sn, we))
    static["XLONG_M"] = m((sn, we))
    static["HGT_M"] = m((sn, we)) * 500.0
    static["LANDMASK"] = landmask
    static["SOILTEMP"] = 285.0 + m((sn, we))
    static["LU_INDEX"] = np.round(m((sn, we)) * 20).astype(float)
    static["MAPFAC_M"] = 1.0 + 0.01 * m((sn, we))
    static["MAPFAC_U"] = 1.0 + 0.01 * m((sn, we + 1))
    static["MAPFAC_V"] = 1.0 + 0.01 * m((sn + 1, we))
    static["MAPFAC_MX"] = static["MAPFAC_M"].copy()
    static["MAPFAC_MY"] = static["MAPFAC_M"].copy()
    static["MAPFAC_UX"] = static["MAPFAC_U"].copy()
    static["MAPFAC_UY"] = static["MAPFAC_U"].copy()
    static["MAPFAC_VX"] = static["MAPFAC_V"].copy()
    static["MAPFAC_VY"] = static["MAPFAC_V"].copy()
    static["F"] = 7e-5 + 1e-6 * m((sn, we))
    static["LANDUSEF"] = m((proj.num_land_cat, sn, we))
    return static


def _projection(nx, ny, grid_id=1):
    return MetgridProjection(
        map_proj=1, truelat1=25.0, truelat2=30.0, stand_lon=-16.4,
        moad_cen_lat=28.3, pole_lat=90.0, pole_lon=0.0, dx_m=9000.0, dy_m=9000.0,
        nx=nx, ny=ny, grid_id=grid_id, parent_id=1, parent_grid_ratio=1,
        i_parent_start=1, j_parent_start=1,
    )


# --- assembly gate -----------------------------------------------------------
@pytest.mark.parametrize(
    "domain,nx,ny",
    [("d01", 12, 9), ("d02", 16, 11), ("d03", 14, 13)],
)
def test_assemble_validates(domain, nx, ny):
    sg = _source_grid()
    forcing = _make_forcing(sg)
    tg = _make_target_grid(nx, ny)
    proj = _projection(nx, ny)
    static = _make_static(nx, ny, proj)
    art = assemble_met_em(domain, "2026-04-28_18:00:00", proj, forcing, static, tg, sg)
    # mandatory-field validation (the contract's assemble->validate gate)
    art.validate(require_optional=False)

    # shapes / staggering
    assert art.arrays["TT"].shape == (NUM_METGRID_LEVELS, ny, nx)
    assert art.arrays["UU"].shape == (NUM_METGRID_LEVELS, ny, nx + 1)  # U stag
    assert art.arrays["VV"].shape == (NUM_METGRID_LEVELS, ny + 1, nx)  # V stag
    assert art.arrays["PRES"].shape == (NUM_METGRID_LEVELS, ny, nx)
    assert art.arrays["PSFC"].shape == (ny, nx)
    assert art.arrays["ST"].shape == (2, ny, nx)
    assert art.arrays["SM"].shape == (2, ny, nx)

    # PRES build: level 0 == PSFC ; levels 1..13 == isobaric constants
    assert np.allclose(art.arrays["PRES"][0], art.arrays["PSFC"], atol=1e-2)
    for lev, p in enumerate(ISOBARIC_LEVELS_PA):
        assert np.allclose(art.arrays["PRES"][lev + 1], p, atol=1e-2)

    # GHT surface level == SOILHGT
    assert np.allclose(art.arrays["GHT"][0], art.arrays["SOILHGT"], atol=1e-3)

    # SOIL_LAYERS thickness [40, 10]
    assert np.allclose(art.arrays["SOIL_LAYERS"][0], 40.0)
    assert np.allclose(art.arrays["SOIL_LAYERS"][1], 10.0)
    # ST 3D stack: index0 == ST010040, index1 == ST000010
    assert np.allclose(art.arrays["ST"][0], art.arrays["ST010040"], atol=1e-4)
    assert np.allclose(art.arrays["ST"][1], art.arrays["ST000010"], atol=1e-4)

    # soil water masking: target water points (LANDMASK==0) filled with 1.0
    lm = static["LANDMASK"]
    water = lm == 0
    if water.any():
        assert np.allclose(art.arrays["ST000010"][water], 1.0)


def test_soil_water_masking_fill():
    sg = _source_grid()
    forcing = _make_forcing(sg)
    nx, ny = 14, 11
    tg = _make_target_grid(nx, ny)
    proj = _projection(nx, ny)
    static = _make_static(nx, ny, proj)
    art = assemble_met_em("d02", "2026-04-28_18:00:00", proj, forcing, static, tg, sg)
    lm = static["LANDMASK"]
    for f in ("ST000010", "ST010040", "SM000010", "SM010040"):
        assert np.allclose(art.arrays[f][lm == 0], 1.0), f"{f} water fill"
        # land points must NOT be the fill value everywhere (real interp ran)
        assert not np.allclose(art.arrays[f][lm == 1], 1.0)


# --- writer round-trip + structural gate -------------------------------------
def test_writer_roundtrip(tmp_path):
    sg = _source_grid()
    forcing = _make_forcing(sg)
    nx, ny = 12, 9
    tg = _make_target_grid(nx, ny)
    proj = _projection(nx, ny)
    static = _make_static(nx, ny, proj)
    art = assemble_met_em("d01", "2026-04-28_18:00:00", proj, forcing, static, tg, sg)

    out = str(tmp_path / "met_em.d01.test.nc")
    write_met_em(art, out)
    ds = netCDF4.Dataset(out)
    try:
        assert ds.dimensions["west_east"].size == nx
        assert ds.dimensions["south_north"].size == ny
        assert ds.dimensions["west_east_stag"].size == nx + 1
        assert ds.dimensions["south_north_stag"].size == ny + 1
        assert ds.dimensions["num_metgrid_levels"].size == NUM_METGRID_LEVELS
        assert ds.dimensions["Time"].isunlimited()
        assert ds.dimensions["DateStrLen"].size == 19
        # Times char
        t = ds.variables["Times"][0].tobytes().decode().strip("\x00").strip()
        assert t == "2026-04-28_18:00:00"
        # TT attrs
        tt = ds.variables["TT"]
        assert int(tt.FieldType) == FIELD_TYPE
        assert tt.MemoryOrder == "XYZ"
        assert tt.stagger == "M"
        assert tt.dimensions == ("Time", "num_metgrid_levels", "south_north", "west_east")
        # UU stagger dims
        assert ds.variables["UU"].dimensions == (
            "Time", "num_metgrid_levels", "south_north", "west_east_stag",
        )
        assert ds.variables["VV"].dimensions == (
            "Time", "num_metgrid_levels", "south_north_stag", "west_east",
        )
        # PSFC 2D MemoryOrder trailing space
        assert ds.variables["PSFC"].MemoryOrder == "XY "
        # FLAG_* present
        for fl in ("FLAG_METGRID", "FLAG_PSFC", "FLAG_SH", "FLAG_SOIL_LAYERS", "FLAG_MF_XY"):
            assert int(ds.getncattr(fl)) == 1
        assert int(ds.getncattr("MAP_PROJ")) == 1
        assert int(ds.getncattr("WEST-EAST_GRID_DIMENSION")) == nx + 1
    finally:
        ds.close()


def _find_oracle():
    import glob

    hits = sorted(glob.glob(_ORACLE_GLOB))
    return hits[0] if hits else None


def test_writer_structural_parity_vs_real_met_em(tmp_path):
    """Structural diff vs a REAL met_em.d01: every variable we write must carry
    the SAME dims + the SAME attribute NAME set + FieldType/MemoryOrder/stagger
    as the oracle, and our global FLAG_* / projection attrs must be a subset of
    the oracle's with matching values. Values are NOT compared (S4 gate)."""

    oracle_path = _find_oracle()
    if oracle_path is None:
        pytest.skip("no real met_em oracle on disk")

    ref = netCDF4.Dataset(oracle_path)
    try:
        ref_nx = ref.dimensions["west_east"].size
        ref_ny = ref.dimensions["south_north"].size
        # Build an artifact matching the oracle's grid dims so dims line up.
        sg = _source_grid()
        forcing = _make_forcing(sg)
        tg = _make_target_grid(ref_nx, ref_ny)
        proj = _projection(ref_nx, ref_ny)
        static = _make_static(ref_nx, ref_ny, proj)
        art = assemble_met_em(
            "d01", ref.variables["Times"][0].tobytes().decode().strip(),
            proj, forcing, static, tg, sg,
        )
        out = str(tmp_path / "met_em.d01.parity.nc")
        write_met_em(art, out)
        ours = netCDF4.Dataset(out)
        try:
            problems = []
            # dimension sizes for the dims we both have
            for dname in ours.dimensions:
                if dname in ref.dimensions:
                    if ours.dimensions[dname].size != ref.dimensions[dname].size and dname != "Time":
                        problems.append(
                            f"dim {dname}: ours={ours.dimensions[dname].size} ref={ref.dimensions[dname].size}"
                        )
                else:
                    problems.append(f"dim {dname} not in oracle")

            # per-variable: dims + key attrs must match the oracle's
            for vname in ours.variables:
                if vname == "Times":
                    continue
                if vname not in ref.variables:
                    problems.append(f"var {vname} not in oracle")
                    continue
                ov = ours.variables[vname]
                rv = ref.variables[vname]
                if ov.dimensions != rv.dimensions:
                    problems.append(f"var {vname} dims ours={ov.dimensions} ref={rv.dimensions}")
                for a in ("FieldType", "MemoryOrder", "stagger"):
                    if ov.getncattr(a) != rv.getncattr(a):
                        problems.append(
                            f"var {vname}.{a} ours={ov.getncattr(a)!r} ref={rv.getncattr(a)!r}"
                        )

            # global FLAG_* + projection attrs must match the oracle where we set them
            for a in (
                "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "GRIDTYPE",
                "NUM_LAND_CAT", "ISWATER", "NUM_METGRID_SOIL_LEVELS",
                "FLAG_METGRID", "FLAG_PSFC", "FLAG_SH", "FLAG_SOIL_LAYERS",
                "FLAG_ST000010", "FLAG_SM000010", "FLAG_MF_XY",
                "WEST-EAST_GRID_DIMENSION", "SOUTH-NORTH_GRID_DIMENSION",
            ):
                if a in ref.ncattrs():
                    ov = ours.getncattr(a)
                    rv = ref.getncattr(a)
                    if isinstance(rv, str):
                        ok = str(ov) == str(rv)
                    elif isinstance(rv, (np.floating, float)):
                        ok = np.isclose(float(ov), float(rv), atol=1e-3)
                    else:
                        ok = int(ov) == int(rv)
                    if not ok:
                        problems.append(f"global {a}: ours={ov!r} ref={rv!r}")

            assert not problems, "structural parity problems:\n" + "\n".join(problems)
        finally:
            ours.close()
    finally:
        ref.close()
