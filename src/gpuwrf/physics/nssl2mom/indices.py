"""Species indices and precision helper for the NSSL 2-moment port (WRF default mp=18).

Values are the Fortran indices produced by ``nssl_2mom_init`` for ipconc=5, hail on,
CCN on with activated CCN (irenuc=5), graupel+hail volume (module lines 1518-1554,
1762-1802, 1876-1894): lhab=8, na=18.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

LT, LV, LC, LR, LI, LS, LH, LHL = 1, 2, 3, 4, 5, 6, 7, 8
LCCN = 9
LNC, LNR, LNI, LNS, LNH, LNHL = 10, 11, 12, 13, 14, 15
LVH, LVHL = 16, 17
LCCNA = 18
NA = 18
LHAB = 8
LQMX = 30

# Map from hydrometeor mass index to its number-concentration index (Fortran ln(il)).
LN = {LC: LNC, LR: LNR, LI: LNI, LS: LNS, LH: LNH, LHL: LNHL}
# Volume index (Fortran lvol(il)); only graupel and hail carry a volume.
LVOL = {LH: LVH, LHL: LVHL}


@dataclass(frozen=True)
class Prec:
    """REAL / DOUBLE PRECISION dtypes of one build (fp32 = WRF default, fp64 = -fdefault-real-8)."""

    name: str

    @property
    def R(self):  # default REAL
        return jnp.float32 if self.name == "fp32" else jnp.float64

    @property
    def D(self):  # DOUBLE PRECISION
        return jnp.float64

    @property
    def nR(self):
        return np.float32 if self.name == "fp32" else np.float64


FP32 = Prec("fp32")
FP64 = Prec("fp64")
