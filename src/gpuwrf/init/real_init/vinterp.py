"""S1 (Opus) — vertical interpolation of met_em columns onto model eta.

FROZEN ENTRY SIGNATURE. Implements the real.exe vertical-interpolation part
(module_initialize_real.F:1450-2809): build dry pressure column, find p_top,
integrate moisture, compute MU0, then vert_interp every atmos field from the
metgrid isobaric levels (grid%pd_gc) to the model dry-pressure target (grid%pb).

Key faithful steps (the implementer reproduces vert_interp / integ_moist /
p_dts / p_dry / lagrange logic — module_initialize_real.F:5590 vert_interp,
:6967 integ_moist, :6764 p_dts, :6710 p_dry, :7163 rh_to_mxrat):
  * find_p_top from the metgrid PRES (:1279) -> grid%p_top (capped at config).
  * integ_moist: column-integrate qv to get dry-pressure column pd_gc + intq (:1450).
  * p_dts -> MU0 (full dry column mass) from intq/psfc/p_top (:1475).
  * target dry pressure pb on half levels: p_dry(mu0,znw,p_top,...) (:1701).
  * vert_interp each of {ght->ph0, qv, t/theta, p, u, v, + hydrometeors}
    from pd_gc to pb with the configured interp_type/lagrange_order/extrap.
  * The vertical index order FLIPS metgrid (sfc=0..top) to model order here.

Output: a DynamicsInit with u/v/w/theta/qv/mu/mu0 and the *interpolated* p/ph
seeds; the FINAL hydrostatically-balanced p/ph/al/alt/p_hyd are produced by
hydrostatic.py (which takes this DynamicsInit + the BaseStateColumns). This lane
delivers the pre-balance interpolated state; hydrostatic.py finishes it.

Oracle: indirect — its product feeds hydrostatic.py, whose output (T/U/V/QVAPOR/
P/PH/MU) is compared to wrfinput. The lane's own unit test should check a single
column vert_interp against a hand-computed Lagrange result.

FILE OWNERSHIP: S1 exclusive.
"""

from __future__ import annotations

from gpuwrf.init.real_init.types import (
    DynamicsInit,
    RealInitConfig,
    VerticalCoord1D,
)
from gpuwrf.init.metgrid_schema import MetEmArtifact


def vertical_interpolate(
    config: RealInitConfig,
    vcoord: VerticalCoord1D,
    metem: MetEmArtifact,
) -> DynamicsInit:
    """Interpolates met_em atmospheric columns onto the model eta column.

    Consumes the FROZEN v0.3.0 ``MetEmArtifact`` (the metgrid-equivalent input).
    Returns a DynamicsInit whose p/ph are interpolation seeds; the final
    balanced p/ph/al/alt/p_hyd come from :func:`hydrostatic.balance`.
    """

    raise NotImplementedError("v0.4.0 S1 (Opus): vertical_interpolate — frozen stub")
