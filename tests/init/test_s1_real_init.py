"""S1 real-init dynamics checks against real.exe wrfinput fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from gpuwrf.init.metgrid_schema import MetEmArtifact, MetgridProjection, metem_field_specs
from gpuwrf.init.real_init.base_state import compute_base_state
from gpuwrf.init.real_init.hydrostatic import balance
from gpuwrf.init.real_init.types import RealInitConfig, WRFINPUT_TOLS
from gpuwrf.init.real_init.vertical_coord import compute_vertical_coord
from gpuwrf.init.real_init.vinterp import vertical_interpolate


WRFINPUT = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wrf_l3/"
    "20260523_18z_l3_24h_20260524T004313Z/wrfinput_d01"
)
METEM = Path(
    "<DATA_ROOT>/canairy_meteo/runs/wps_cases/20260523_18z_72h/l3/"
    "met_em.d01.2026-05-23_18:00:00.nc"
)


def _read_metem(path: Path, domain: str) -> MetEmArtifact:
    with netCDF4.Dataset(str(path)) as ds:
        attrs = ds.__dict__
        projection = MetgridProjection(
            map_proj=int(attrs["MAP_PROJ"]),
            truelat1=float(attrs["TRUELAT1"]),
            truelat2=float(attrs["TRUELAT2"]),
            stand_lon=float(attrs["STAND_LON"]),
            moad_cen_lat=float(attrs["MOAD_CEN_LAT"]),
            pole_lat=float(attrs["POLE_LAT"]),
            pole_lon=float(attrs["POLE_LON"]),
            dx_m=float(attrs["DX"]),
            dy_m=float(attrs["DY"]),
            nx=len(ds.dimensions["west_east"]),
            ny=len(ds.dimensions["south_north"]),
            grid_id=int(attrs.get("grid_id", domain[-1])),
            parent_id=int(attrs.get("parent_id", 1)),
            parent_grid_ratio=int(attrs.get("parent_grid_ratio", 1)),
            i_parent_start=int(attrs.get("i_parent_start", 1)),
            j_parent_start=int(attrs.get("j_parent_start", 1)),
        )
        specs = {spec.name for spec in metem_field_specs()}
        arrays = {
            name: np.asarray(ds.variables[name][0])
            for name in specs
            if name in ds.variables
        }
        times = "".join(
            ch.decode() if isinstance(ch, bytes) else str(ch)
            for ch in np.asarray(ds.variables["Times"][0])
        ).strip()
    return MetEmArtifact(domain=domain, valid_time=times, projection=projection, arrays=arrays)


def _config_from_wrfinput(ds) -> RealInitConfig:
    return RealInitConfig(
        nz=len(ds.dimensions["bottom_top"]),
        p_top_pa=float(ds.variables["P_TOP"][0]),
        hybrid_opt=int(getattr(ds, "HYBRID_OPT")),
        etac=float(getattr(ds, "ETAC")),
        base_pres=float(ds.variables["P00"][0]),
        base_temp=float(ds.variables["T00"][0]),
        base_lapse=float(ds.variables["TLP"][0]),
        iso_temp=float(ds.variables["TISO"][0]),
        base_pres_strat=float(ds.variables["P_STRAT"][0]),
        base_lapse_strat=float(ds.variables["TLP_STRAT"][0]),
        grid_id=int(getattr(ds, "GRID_ID", 1)),
    )


def _rmse_max(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff))), float(np.max(np.abs(diff)))


@pytest.mark.skipif(not WRFINPUT.exists() or not METEM.exists(), reason="real S1 fixtures unavailable")
def test_s1_d01_main_dynamics_match_real_exe_fixture() -> None:
    metem = _read_metem(METEM, "d01")
    with netCDF4.Dataset(str(WRFINPUT)) as ds:
        config = _config_from_wrfinput(ds)
        vcoord = compute_vertical_coord(config)
        base = compute_base_state(config, vcoord, metem.arrays["HGT_M"])
        seed = vertical_interpolate(config, vcoord, metem)
        dynamics = balance(config, vcoord, base, seed)

        fields = {
            "MU": dynamics.mu,
            "MUB": base.mub,
            "PB": base.pb,
            "P": dynamics.p,
            "PH": dynamics.ph,
            "PHB": base.phb,
            "T": dynamics.theta,
            "U": dynamics.u,
            "V": dynamics.v,
            "QVAPOR": dynamics.qv,
            "ZNW": vcoord.znw,
            "ZNU": vcoord.znu,
            "C3H": vcoord.c3h,
            "C4H": vcoord.c4h,
        }
        for name, actual in fields.items():
            rmse, maxabs = _rmse_max(actual, np.asarray(ds.variables[name][0]))
            rmse_tol, max_tol = WRFINPUT_TOLS[name]
            assert rmse <= rmse_tol, (name, rmse, rmse_tol)
            assert maxabs <= max_tol, (name, maxabs, max_tol)
