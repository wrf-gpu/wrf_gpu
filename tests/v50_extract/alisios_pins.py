"""Read-only copies of ALISIOS reference functions used as independent test oracles.

inversion_base: verbatim from ALISIOS Git 62d80f6a0:src/alisios/truthmap/reretro.py
(blob 4f205a99f51dec0e50f00b393cc5e7d3ae578915, file sha256
33fb6fc41313faedfaecf51447d8cf3b4e8c7d8873d2d208c1add44f1f4a8f62), lines 93-111 (TM-S6).
Never edit; test_pins_match_alisios re-reads the blob when the repo is present.
"""
import math

import numpy as np

RERETRO_COMMIT = "62d80f6a0"
RERETRO_PATH = "src/alisios/truthmap/reretro.py"
RERETRO_LINES = (93, 111)
INV_ZMIN, INV_ZMAX, INV_MIN_GRAD = 300.0, 3500.0, 0.002     # m, m, K/m


def inversion_base(theta, z, *, zmin=INV_ZMIN, zmax=INV_ZMAX, min_grad=INV_MIN_GRAD) -> float:
    """Bottom (MSL m) of the layer with the largest d(theta)/dz inside [zmin, zmax]; NaN if the
    largest gradient there is below ``min_grad`` (no inversion) or the profile is not ascending."""
    th, zz = np.asarray(theta, float), np.asarray(z, float)
    ok = np.isfinite(th) & np.isfinite(zz)
    th, zz = th[ok], zz[ok]
    if len(zz) < 3:
        return math.nan
    if (np.diff(zz) <= 0).any():
        raise ValueError("profile heights must ascend")
    g = np.diff(th) / np.diff(zz)
    zb = zz[:-1]
    sel = (zb >= zmin) & (zz[1:] <= zmax)
    if not sel.any():
        return math.nan
    k = np.flatnonzero(sel)[np.argmax(g[sel])]
    return float(zb[k]) if g[k] >= min_grad else math.nan




# liquid_ice_mass: verbatim teacher_extract_v2.py lines 339-342 (sha256 75f02a61b5830cce9a2f624a067a5214bdde8b3011a0411439ef2837a14c3a5d,
# <DATA_ROOT>/alisios/state/manager/runs/v32_sprints_20261002/R32-13/; copied because the test suite guards <DATA_ROOT>/alisios reads).
TEACHER_V2_LINES = (339, 342)


def liquid_ice_mass(qs, mass):
    # Native hybrid-coordinate dry mass: -(C1H*(MU+MUB)+C2H)*DNW/g.
    return {name: sum(qs[k].astype('f8') * mass for k in fields).sum(axis=0).astype('f4')
            for name, fields in [('LWP', ('QCLOUD', 'QRAIN')), ('IWP', ('QICE', 'QSNOW', 'QGRAUP'))]}
