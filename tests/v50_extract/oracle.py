"""Independent offline recomputation of the V1 extract from a full wrfout frame.

Shares no code with gpuwrf.extract. Uses ALISIOS code read-only and SHA-pinned:
cloud9.frame_cloud (cloud_max/cloud_low_max/LWP), teacher_extract_v2.liquid_ice_mass
(IWP) and TM-S6 inversion_base (both copied verbatim with SHA in alisios_pins.py); wind as extract9.py lines
76-79; cloud base / inversion layer details by explicit per-column loops.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import alisios_pins

ALISIOS = Path("<USER_HOME>/src/alisios")
CLOUD9 = ALISIOS / "ops/nightly_9nest/cloud9.py"
CLOUD9_SHA = "26424dc414515f16b4094c3637b60982f07925c7a92b0581abc7e8392d1d9de9"
EXTRACT9 = ALISIOS / "ops/nightly_9nest/extract9.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def available() -> bool:
    return CLOUD9.is_file()


def _load(name: str, path: Path, sha: str):
    assert _sha(path) == sha, f"pinned ALISIOS file changed: {path}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # never write into ALISIOS directories
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def cloud9():
    return _load("cloud9", CLOUD9, CLOUD9_SHA)


def frame_time_unix(path: Path) -> int:
    stamp = re.search(r"_(\d{4}-\d{2}-\d{2}_\d{2}[:_]\d{2}[:_]\d{2})", path.name).group(1).replace("_", " ", 1)
    stamp = stamp.replace("_", ":")
    return int(datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def recompute(path: Path, sst_available: bool = True) -> dict:
    import netCDF4

    c9 = cloud9()
    out = {}
    with netCDF4.Dataset(path) as ds:
        raw = lambda n: c9.raw_array(ds, n)  # teacher raw_array: f4, no fill, finite
        out.update(c9.frame_cloud(ds))
        for n in ("U10", "V10", "T2", "Q2", "PSFC", "PBLH", "HFX", "LH", "UST", "RAINC", "RAINNC",
                  "SWDOWN", "GLW", "SST", "TSK", "XLAT", "XLONG", "HGT", "LANDMASK", "COSALPHA", "SINALPHA"):
            out[n] = raw(n)
        # A gpuwrf run without an SST producer (declared by the caller from the case namelist: sst_update) has
        # no real SST; its history SST is the writer's TSK copy -> V1 SST NOT_AVAILABLE (all NaN).
        if not sst_available:
            out["SST"] = np.full(out["SST"].shape, np.nan, np.float32)
        # extract9.py lines 76-79 (f4 arithmetic), calm -> NaN per spec.
        c, s = out["COSALPHA"], out["SINALPHA"]
        ue = out["U10"] * c - out["V10"] * s
        vn = out["V10"] * c + out["U10"] * s
        spd = np.hypot(ue, vn)
        wdir = (np.degrees(np.arctan2(-ue, -vn)) + 360.0) % 360.0
        out.update(u10_east=ue, v10_north=vn, wind_speed=spd,
                   wind_direction_from_deg=np.where(spd == 0, np.float32(np.nan), wdir).astype(np.float32))
        mu = raw("MU").astype("f8") + raw("MUB").astype("f8")
        mass = -(raw("C1H").astype("f8")[:, None, None] * mu + raw("C2H").astype("f8")[:, None, None]) \
            * raw("DNW").astype("f8")[:, None, None] / 9.81
        qs = {k: raw(k) for k in ("QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")}
        out["IWP"] = alisios_pins.liquid_ice_mass(qs, mass)["IWP"]
        zw = (raw("PH").astype("f8") + raw("PHB").astype("f8")) / 9.81
        zmsl = ((zw[:-1] + zw[1:]) / 2).astype("f4")
        zagl = zmsl - out["HGT"]
        theta = (raw("T").astype("f8") + 300).astype("f4")  # WT1 theta_K
        qsum = qs["QCLOUD"] + qs["QICE"]
        nz, ny, nx = zmsl.shape
        names = ("cloud_base_msl", "cloud_base_agl", "inv_base_msl", "inv_top_msl", "inv_strength_K",
                 "inv_gradient_K_m")
        for n in names:
            out[n] = np.full((ny, nx), np.nan, np.float32)
        out["cloud_base_status"] = np.ones((ny, nx), np.int8)
        out["inv_status"] = np.ones((ny, nx), np.int8)
        for j in range(ny):
            for i in range(nx):
                for k in range(nz):
                    if qsum[k, j, i] >= np.float32(1e-5):
                        out["cloud_base_msl"][j, i] = zmsl[k, j, i]
                        out["cloud_base_agl"][j, i] = zagl[k, j, i]
                        out["cloud_base_status"][j, i] = 0
                        break
                th, zz = theta[:, j, i], zmsl[:, j, i]
                base = alisios_pins.inversion_base(th, zz)
                if not any(float(zz[k]) >= 300.0 and float(zz[k + 1]) <= 3500.0 for k in range(nz - 1)):
                    out["inv_status"][j, i] = 3
                    continue
                if math.isnan(base):
                    continue
                k = int(np.flatnonzero(zz[:-1].astype(float) == base)[0])
                g = (float(th[k + 1]) - float(th[k])) / (float(zz[k + 1]) - float(zz[k]))
                out["inv_status"][j, i] = 0
                out["inv_base_msl"][j, i] = zz[k]
                out["inv_top_msl"][j, i] = zz[k + 1]
                out["inv_strength_K"][j, i] = np.float32(float(th[k + 1]) - float(th[k]))
                out["inv_gradient_K_m"][j, i] = np.float32(g)
    out["actual_time_unix_s"] = frame_time_unix(Path(path))
    return out


def sst_update(case_dir) -> bool:
    """Case declaration (independent of gpuwrf.extract): &physics sst_update = 1 in the case namelist."""
    text = (Path(case_dir) / "namelist.input").read_text().lower()
    m = re.search(r"^\s*sst_update\s*=\s*(\d+)", text, re.M)
    return bool(m) and int(m.group(1)) == 1
