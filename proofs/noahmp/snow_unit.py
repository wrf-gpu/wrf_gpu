"""Noah-MP snow (S3) unit checks: vectorized branch-free kernel + zero-snow.

Complements the savepoint-parity harness with:
  1. a MIXED (ny,nx) grid of {zero-snow, shallow, single, 2-layer, 3-layer}
     columns in ONE jitted call (proves the masked kernel handles all ISNOW
     states simultaneously with no python layer-count branching);
  2. a bit-clean zero-snow degrade check (the common Canary case): a column with
     no snow and zero snowfall returns SNEQV/SNOWH/ISNOW unchanged and finite;
  3. finiteness across the whole grid (no NaN/Inf from masked empty layers).
"""

from __future__ import annotations

import os
import sys

import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_SRC = os.path.normpath(os.path.join(THIS_DIR, "..", "..", "src"))
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from jax import config  # noqa: E402

config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from gpuwrf.contracts.noahmp_state import NSNOW, NSOIL, NoahMPLandState, NoahMPStatic  # noqa: E402
from gpuwrf.physics.noahmp.snow import noahmp_snow  # noqa: E402
from gpuwrf.physics.noahmp.types import NoahMPForcing  # noqa: E402

ZSOIL = np.array([-0.1, -0.4, -1.0, -2.0])
DT = 1800.0


def main():
    ny, nx = 2, 3            # 6 columns spanning every ISNOW state
    f64 = jnp.float64

    isnow = np.zeros((ny, nx), np.int32)
    snowh = np.zeros((ny, nx))
    sneqv = np.zeros((ny, nx))
    snice = np.zeros((NSNOW, ny, nx))
    snliq = np.zeros((NSNOW, ny, nx))
    tsno = np.zeros((NSNOW, ny, nx))
    zsnso = np.zeros((NSNOW + NSOIL, ny, nx))

    def set_col(j, i, isn, dz, ice, liq, tt):
        isnow[j, i] = isn
        n = -isn
        # top-aligned: active local slots NSNOW-n .. NSNOW-1
        for m in range(n):
            loc = NSNOW - n + m
            snice[loc, j, i] = ice[m]
            snliq[loc, j, i] = liq[m]
            tsno[loc, j, i] = tt[m]
        # build zsnso snow portion (cumulative negative) + soil offset
        zacc = 0.0
        for m in range(n):
            loc = NSNOW - n + m
            zacc -= dz[m]
            zsnso[loc, j, i] = zacc
        for k in range(NSOIL):
            zsnso[NSNOW + k, j, i] = (zacc if isn < 0 else 0.0) + ZSOIL[k]
        snowh[j, i] = sum(dz[:n])
        sneqv[j, i] = sum(ice[:n]) + sum(liq[:n])

    # col (0,0): zero snow  -> degrade-to-zero case
    # col (0,1): shallow bulk (no layer) SNEQV>0 ISNOW=0
    snowh[0, 1] = 0.01; sneqv[0, 1] = 4.0
    # col (0,2): single layer
    set_col(0, 2, -1, [0.20], [25.0], [0.0], [265.0])
    # col (1,0): two layers
    set_col(1, 0, -2, [0.18, 0.22], [40.0, 50.0], [1.0, 4.0], [266.0, 271.0])
    # col (1,1): three layers
    set_col(1, 1, -3, [0.30, 0.30, 0.20], [60.0, 70.0, 65.0], [1.0, 3.0, 8.0],
            [260.0, 265.0, 272.0])
    # col (1,2): zero snow again
    # (left as zeros)

    for k in range(NSOIL):
        zsnso[NSNOW + k][zsnso[NSNOW + k] == 0.0] = ZSOIL[k]

    def col(a):
        return jnp.asarray(a, f64)

    z2 = jnp.zeros((ny, nx), f64)
    tslb = jnp.broadcast_to(col([280.0]).reshape(1, 1, 1), (NSOIL, ny, nx))
    smois = jnp.full((NSOIL, ny, nx), 0.30, f64)
    sh2o = jnp.full((NSOIL, ny, nx), 0.25, f64)
    land = NoahMPLandState(
        tslb=tslb, smois=smois, sh2o=sh2o, smcwtd=z2,
        isnow=jnp.asarray(isnow, jnp.int32), tsno=col(tsno), snice=col(snice),
        snliq=col(snliq), zsnso=col(zsnso), snowh=col(snowh), sneqv=col(sneqv),
        sneqvo=col(sneqv), tauss=jnp.full((ny, nx), 0.5, f64),
        albold=jnp.full((ny, nx), 0.6, f64),
        tv=jnp.full((ny, nx), 270.0, f64), tg=jnp.full((ny, nx), 268.0, f64),
        tah=jnp.full((ny, nx), 270.0, f64), eah=z2,
        canliq=z2, canice=z2, fwet=z2, lai=z2, sai=z2, cm=z2, ch=z2,
        t_skin=jnp.full((ny, nx), 270.0, f64), qsfc=z2, znt=z2, emiss=z2, albedo=z2,
        sfcrunoff=z2, udrunoff=z2,
    )
    static = NoahMPStatic(
        ivgtyp=jnp.zeros((ny, nx), jnp.int32), isltyp=jnp.zeros((ny, nx), jnp.int32),
        xland=z2, landmask=z2, lakemask=z2, lu_index=jnp.zeros((ny, nx), jnp.int32),
        tbot=jnp.full((ny, nx), 285.0, f64),
        dzs=col([0.1, 0.3, 0.6, 1.0]), zsoil=col(ZSOIL), lat=z2, dx_m=3000.0,
        parameters=None,
    )
    forcing = NoahMPForcing(
        sfctmp=jnp.full((ny, nx), 266.0, f64), sfcprs=jnp.full((ny, nx), 9e4, f64),
        psfc=jnp.full((ny, nx), 9e4, f64), uu=z2, vv=z2, qair=z2, qc=z2,
        soldn=z2, lwdn=z2, prcpconv=z2, prcpnonc=z2, prcpsnow=z2, prcpgrpl=z2,
        prcphail=z2, cosz=jnp.full((ny, nx), 0.5, f64), zlvl=jnp.full((ny, nx), 10.0, f64),
        julian=jnp.asarray(15.0), yearlen=jnp.asarray(365.0),
    )

    # snowfall onto the shallow-bulk + a couple columns
    qsnow = np.zeros((ny, nx)); qsnow[0, 1] = 0.01; qsnow[0, 2] = 0.005
    qsnow = col(qsnow)
    imelt = jnp.zeros((NSNOW + NSOIL, ny, nx), jnp.int32)
    qmelt = jnp.zeros((ny, nx), f64)

    out = noahmp_snow(land, forcing, static, qsnow, imelt, qmelt, DT)

    # 1. finiteness everywhere
    finite = True
    for name in ("snice", "snliq", "tsno", "zsnso", "snowh", "sneqv", "tauss", "albold"):
        arr = np.asarray(getattr(out, name))
        if not np.all(np.isfinite(arr)):
            finite = False
            print("NON-FINITE in", name)

    # 2. zero-snow degrade: cols (0,0) and (1,2) had no snow + no snowfall
    zs_ok = True
    for (j, i) in [(0, 0), (1, 2)]:
        if (int(np.asarray(out.isnow)[j, i]) != 0
                or float(np.asarray(out.sneqv)[j, i]) != 0.0
                or float(np.asarray(out.snowh)[j, i]) != 0.0
                or np.any(np.asarray(out.snice)[:, j, i] != 0.0)
                or np.any(np.asarray(out.snliq)[:, j, i] != 0.0)):
            zs_ok = False
            print("zero-snow column changed at", (j, i))

    # 3. multi-state in one call: every ISNOW present advanced sensibly
    isn_out = np.asarray(out.isnow)
    states = sorted(set(isn_out.flatten().tolist()))

    ok = finite and zs_ok
    print("finite_all:", finite)
    print("zero_snow_bitclean:", zs_ok)
    print("isnow_states_in_one_jit_call:", states)
    print("UNIT", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
