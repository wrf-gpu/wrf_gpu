"""setvtz branch checks not reachable through sediment1d (which always uses infdo = 2).

gs calls setvtz with infdo = 1: Fortran lines 7069-7078 then set the rain Z-weighted speed to the
mass-weighted one where qr > qxmin (ELSE branch) and leave it untouched elsewhere.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gpuwrf.physics.nssl2mom import fallspeed  # noqa: E402
from gpuwrf.physics.nssl2mom.constants import get_constants  # noqa: E402
from gpuwrf.physics.nssl2mom.indices import FP64, LC, LH, LHL, LI, LR, LS  # noqa: E402


def _inputs():
    sp = (LC, LR, LI, LS, LH, LHL)
    z = jnp.zeros(3)
    qx = {il: z for il in sp}
    qx[LR] = jnp.array([1e-3, 1e-15, 2e-4])
    cx = {il: z for il in sp}
    cx[LR] = jnp.array([1e4, 1.0, 3e3])
    keep = jnp.array([7.0, 8.0, 9.0])
    vt = {(il, m): z for il in sp for m in (1, 2, 3)}
    vt[(LR, 3)] = keep
    alpha = {il: z for il in sp}
    return dict(qx=qx, cx=cx, rho0=jnp.ones(3), rhovt=jnp.ones(3),
                xdia={(il, m): z for il in sp for m in (1, 2, 3)}, xmas={il: z for il in sp},
                vtxbar=vt, xdn={il: jnp.full(3, 1000.0) for il in sp}, xv={il: z for il in sp},
                cdxgs={il: z for il in sp}, fadvisc=jnp.full(3, 1.8e-5), temcg=z, alpha=alpha,
                axx={LH: z, LHL: z}, bxx={LH: z, LHL: z}), keep


def test_rain_z_speed_copies_mass_speed_for_infdo1():
    args, keep = _inputs()
    out = fallspeed.setvtz(**args, C=get_constants("fp64"), prec=FP64, infdo=1)
    v1 = np.asarray(out["vtxbar"][(LR, 1)])
    v3 = np.asarray(out["vtxbar"][(LR, 3)])
    has = np.array([True, False, True])
    assert np.all(v1[has] > 0)
    np.testing.assert_array_equal(v3[has], v1[has])
    np.testing.assert_array_equal(v3[~has], np.asarray(keep)[~has])


def test_rain_z_speed_infdo2_differs_from_mass_speed():
    args, _keep = _inputs()
    out = fallspeed.setvtz(**args, C=get_constants("fp64"), prec=FP64, infdo=2)
    v1 = np.asarray(out["vtxbar"][(LR, 1)])
    v3 = np.asarray(out["vtxbar"][(LR, 3)])
    assert v3[0] > v1[0] and v3[2] > v1[2]
