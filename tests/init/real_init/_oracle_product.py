"""S4 test helper — build a :class:`RealInitProduct` directly FROM an oracle.

This is the *self-test scaffolding* for the comparator. It reads a real.exe
wrfinput (+ optional wrfbdy) file and reconstructs the frozen handoff dataclasses
that the S1/S2/S3 lanes will eventually produce. It is used two ways:

* **sanity candidate** (``perturb=None``): a verbatim copy of the oracle. Scored
  against the SAME oracle file it was built from, every field must be ~0 error.
  This is the ungameable proof that the comparator actually compares (a broken
  comparator that always passes would also pass a deliberately-perturbed
  candidate; this harness ALSO builds a perturbed candidate that must FAIL).
* **stub candidate** (``perturb={field: delta}``): the oracle with a controlled
  offset added to selected fields, used to demonstrate the PASS/FAIL mechanics
  (a small delta within tol -> PASS; a delta above tol -> FAIL).

NOT a production lane — this lives under ``tests/`` and is owned by S4. It does
NOT exercise any S1/S2/S3 lane code (those are NotImplementedError stubs at the
S0 base); it exists purely to validate the comparator harness itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from netCDF4 import Dataset

from gpuwrf.init.real_init.types import (
    BaseStateColumns,
    DynamicsInit,
    LateralBC,
    RealInitConfig,
    RealInitProduct,
    SoilInit,
    SurfaceInit,
    VerticalCoord1D,
)


def _rd(ds: Dataset, name: str, dtype=np.float64) -> np.ndarray | None:
    if name not in ds.variables:
        return None
    raw = ds.variables[name][:]
    a = np.asarray(np.ma.filled(raw, np.nan), dtype=dtype)
    if a.ndim >= 1 and a.shape[0] == 1:
        a = a[0]
    return a


def _scalar(ds: Dataset, name: str, default: float) -> float:
    a = _rd(ds, name)
    if a is None:
        return default
    return float(np.asarray(a).ravel()[0])


def build_product_from_oracle(
    wrfinput_path: str | Path,
    *,
    domain: str = "d01",
    wrfbdy_path: str | Path | None = None,
    perturb: Mapping[str, float] | None = None,
) -> RealInitProduct:
    """Reconstruct a RealInitProduct from an oracle wrfinput (+ optional wrfbdy).

    ``perturb`` maps a wrfinput variable name (e.g. "T", "MU") to an additive
    offset applied to that field in the candidate — used to exercise PASS/FAIL.
    """
    perturb = dict(perturb or {})
    wrfinput_path = Path(wrfinput_path)

    def pv(name: str, arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        if name in perturb:
            return arr + perturb[name]
        return arr

    with Dataset(wrfinput_path, "r") as ds:
        nz = int(ds.dimensions["bottom_top"].size)
        p_top = _scalar(ds, "P_TOP", 5000.0)
        etac = float(getattr(ds, "ETAC", 0.2))
        hybrid_opt = int(getattr(ds, "HYBRID_OPT", 2))
        nsoil = int(ds.dimensions["soil_layers_stag"].size)

        config = RealInitConfig(
            nz=nz, p_top_pa=p_top, hybrid_opt=hybrid_opt, etac=etac,
            num_soil_layers=nsoil,
            grid_id=int(getattr(ds, "GRID_ID", 1)),
        )

        vcoord = VerticalCoord1D(
            znw=pv("ZNW", _rd(ds, "ZNW")),
            znu=pv("ZNU", _rd(ds, "ZNU")),
            dnw=np.diff(_rd(ds, "ZNW")),
            rdnw=1.0 / np.diff(_rd(ds, "ZNW")),
            dn=np.zeros(nz),
            rdn=np.zeros(nz),
            fnp=np.zeros(nz),
            fnm=np.zeros(nz),
            c1f=pv("C1F", _rd(ds, "C1F")),
            c2f=pv("C2F", _rd(ds, "C2F")),
            c3f=pv("C3F", _rd(ds, "C3F")),
            c4f=pv("C4F", _rd(ds, "C4F")),
            c1h=pv("C1H", _rd(ds, "C1H")),
            c2h=pv("C2H", _rd(ds, "C2H")),
            c3h=pv("C3H", _rd(ds, "C3H")),
            c4h=pv("C4H", _rd(ds, "C4H")),
            cf1=0.0, cf2=0.0, cf3=0.0, cfn=0.0, cfn1=0.0,
            p_top_pa=p_top + perturb.get("P_TOP", 0.0),
        )

        base = BaseStateColumns(
            pb=pv("PB", _rd(ds, "PB")),
            alb=_rd(ds, "ALB") if "ALB" in ds.variables else np.zeros((nz,) + _rd(ds, "MUB").shape),
            t_init=_rd(ds, "T_INIT") if "T_INIT" in ds.variables else np.zeros((nz,) + _rd(ds, "MUB").shape),
            mub=pv("MUB", _rd(ds, "MUB")),
            phb=pv("PHB", _rd(ds, "PHB")),
        )

        dynamics = DynamicsInit(
            u=pv("U", _rd(ds, "U")),
            v=pv("V", _rd(ds, "V")),
            w=pv("W", _rd(ds, "W")),
            theta=pv("T", _rd(ds, "T")),
            qv=pv("QVAPOR", _rd(ds, "QVAPOR")),
            mu=pv("MU", _rd(ds, "MU")),
            mu0=_rd(ds, "MU") + _rd(ds, "MUB"),
            p=pv("P", _rd(ds, "P")),
            ph=pv("PH", _rd(ds, "PH")),
            al=pv("AL", _rd(ds, "AL")),
            alt=pv("ALT", _rd(ds, "ALT")) if "ALT" in ds.variables else _rd(ds, "AL"),
            p_hyd=_rd(ds, "P_HYD") if "P_HYD" in ds.variables else _rd(ds, "P"),
        )

        surface = SurfaceInit(
            xlat=pv("XLAT", _rd(ds, "XLAT")),
            xlong=pv("XLONG", _rd(ds, "XLONG")),
            xlat_u=_rd(ds, "XLAT_U"),
            xlong_u=_rd(ds, "XLONG_U"),
            xlat_v=_rd(ds, "XLAT_V"),
            xlong_v=_rd(ds, "XLONG_V"),
            mapfac_m=pv("MAPFAC_M", _rd(ds, "MAPFAC_M")),
            mapfac_u=pv("MAPFAC_U", _rd(ds, "MAPFAC_U")),
            mapfac_v=pv("MAPFAC_V", _rd(ds, "MAPFAC_V")),
            mapfac_mx=_rd(ds, "MAPFAC_MX"),
            mapfac_my=_rd(ds, "MAPFAC_MY"),
            mapfac_ux=_rd(ds, "MAPFAC_UX"),
            mapfac_uy=_rd(ds, "MAPFAC_UY"),
            mapfac_vx=_rd(ds, "MAPFAC_VX"),
            mapfac_vy=_rd(ds, "MAPFAC_VY"),
            f=pv("F", _rd(ds, "F")),
            e=pv("E", _rd(ds, "E")),
            sinalpha=pv("SINALPHA", _rd(ds, "SINALPHA")),
            cosalpha=pv("COSALPHA", _rd(ds, "COSALPHA")),
            hgt=pv("HGT", _rd(ds, "HGT")),
            tsk=pv("TSK", _rd(ds, "TSK")),
            sst=pv("SST", _rd(ds, "SST")),
            tmn=pv("TMN", _rd(ds, "TMN")),
            xland=pv("XLAND", _rd(ds, "XLAND")),
            landmask=_rd(ds, "LANDMASK"),
            snowh=_rd(ds, "SNOWH"),
            seaice=_rd(ds, "SEAICE"),
        )

        soil = SoilInit(
            tslb=pv("TSLB", _rd(ds, "TSLB")),
            smois=pv("SMOIS", _rd(ds, "SMOIS")),
            sh2o=_rd(ds, "SH2O"),
            zs=pv("ZS", _rd(ds, "ZS")),
            dzs=pv("DZS", _rd(ds, "DZS")),
            isltyp=pv("ISLTYP", _rd(ds, "ISLTYP")),
            ivgtyp=pv("IVGTYP", _rd(ds, "IVGTYP")),
            lu_index=_rd(ds, "LU_INDEX"),
            vegfra=_rd(ds, "VEGFRA"),
            canwat=_rd(ds, "CANWAT"),
        )

        init_time = wrfinput_path.parent.name

    lbc = None
    if wrfbdy_path is not None:
        lbc = _build_lbc_from_oracle(Path(wrfbdy_path), config, perturb)

    return RealInitProduct(
        domain=domain,
        init_time=init_time,
        config=config,
        vcoord=vcoord,
        base=base,
        dynamics=dynamics,
        surface=surface,
        soil=soil,
        lateral_bc=lbc,
        provenance={"built_from_oracle": str(wrfinput_path)},
    )


_BDY_NC_TO_NAME = {"U": "u", "V": "v", "T": "t", "PH": "ph", "QVAPOR": "qv", "MU": "mu"}
_VALUE_SUFFIX = {"W": "BXS", "E": "BXE", "S": "BYS", "N": "BYE"}
_TEND_SUFFIX = {"W": "BTXS", "E": "BTXE", "S": "BTYS", "N": "BTYE"}
_VALUE_KEY = {"W": "bxs", "E": "bxe", "S": "bys", "N": "bye"}
_TEND_KEY = {"W": "btxs", "E": "btxe", "S": "btys", "N": "btye"}


def _build_lbc_from_oracle(
    wrfbdy_path: Path,
    config: RealInitConfig,
    perturb: Mapping[str, float],
) -> LateralBC:
    """Reconstruct the LateralBC dataclass from an oracle wrfbdy (first frame)."""
    values: dict[str, dict[str, np.ndarray]] = {}
    tendencies: dict[str, dict[str, np.ndarray]] = {}
    with Dataset(wrfbdy_path, "r") as ds:
        for nc_prefix, name in _BDY_NC_TO_NAME.items():
            vmap: dict[str, np.ndarray] = {}
            tmap: dict[str, np.ndarray] = {}
            for side in ("W", "E", "S", "N"):
                vraw = ds.variables.get(f"{nc_prefix}_{_VALUE_SUFFIX[side]}")
                traw = ds.variables.get(f"{nc_prefix}_{_TEND_SUFFIX[side]}")
                if vraw is not None:
                    v = np.asarray(np.ma.filled(vraw[:], np.nan), dtype=np.float64)
                    if v.ndim >= 1 and v.shape[0] >= 1:
                        v = v[0]
                    # perturb the bdy field by the same key (uppercase NC name)
                    if nc_prefix in perturb:
                        v = v + perturb[nc_prefix]
                    vmap[_VALUE_KEY[side]] = v
                if traw is not None:
                    t = np.asarray(np.ma.filled(traw[:], np.nan), dtype=np.float64)
                    if t.ndim >= 1 and t.shape[0] >= 1:
                        t = t[0]
                    tmap[_TEND_KEY[side]] = t
            values[name] = vmap
            tendencies[name] = tmap
    return LateralBC(
        values=values,
        tendencies=tendencies,
        spec_bdy_width=config.spec_bdy_width,
        bdyfrq_seconds=float(config.interval_seconds),
        valid_times=(),
    )
