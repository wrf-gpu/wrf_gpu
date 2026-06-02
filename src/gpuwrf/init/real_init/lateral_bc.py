"""S3 (Opus) — lateral boundary value + tendency generation (wrfbdy-equiv).

FROZEN ENTRY SIGNATURE. Reproduces real_em.F::assemble_output (main/real_em.F:680
-1240) over the forcing intervals. For each interval the forcing met_em column is
run through the SAME S1+S2 init (vinterp + hydrostatic + surface) to get a full
state at that valid time, then:

  1. COUPLE each prognostic field by total mu and the stagger-specific map scale
     factor (real_em.F:866-878, share/module_bc.F couple):
         coupled = field * (mu_2 + mub) / msf
     Stagger map (real.exe): u -> msfuy ; v -> msfvx ; t/qv/ph -> msfty ;
     mu (2D) -> msfty. ph uses the 'W' stuff_bdy stagger.
  2. STUFF the coupled value into the per-side boundary frames (stuff_bdy):
     XS = first ``spec_bdy_width`` i-columns, XE = last, YS/YE the j-rows; for
     U-stagger ide->ide, V-stagger jde->jde (share/module_bc.F:2934-3100).
  3. TENDENCY between consecutive intervals (stuff_bdytend_new,
     share/module_bc.F:2893, used at real_em.F:1123-1163):
         tend = (coupled_{n+1} - coupled_{n}) / interval_seconds      (:2938)
     The first interval's coupled value is the specified VALUE (the ``_bxs`` etc
     in LateralBC.values); the tendency is stored in ``_btxs`` etc.

NOTE the coupling needs MU/MUB/MSF from S1+S2 at EACH forcing time, so S3 calls
the S1/S2 entry points (a true data dependency on their *interfaces*, which are
frozen here). S3 develops against the frozen stubs / the real.exe wrfbdy oracle;
it does NOT need S1/S2 *implementations* merged to start (it can mock the per-
time state from the oracle wrfinput at each interval during development).

Oracle: wrfbdy U_BXS/.../U_BTXS/... etc for d01 (the parent carries LBC) across
the ≥10 cases; tols ``types.WRFBDY_TOLS`` on the coupled values + tendencies.

FILE OWNERSHIP: this file is S3's exclusive file. Do not edit types.py, driver.py,
or any S1/S2 file.
"""

from __future__ import annotations

from collections.abc import Sequence

from gpuwrf.init.real_init.types import LateralBC, RealInitConfig
from gpuwrf.init.metgrid_schema import MetEmArtifact


def generate_lateral_bc(
    config: RealInitConfig,
    forcing_sequence: Sequence[MetEmArtifact],
) -> LateralBC:
    """Builds wrfbdy-equivalent specified values + tendencies.

    ``forcing_sequence`` is the time-ordered list of met_em artifacts at the
    forcing intervals (e.g. AIFS 6-hourly). Each is initialized to a full state
    via the S1/S2 entry points, coupled, stuffed, and differenced. Returns the
    per-side value + tendency frames in :class:`LateralBC`.
    """

    raise NotImplementedError("v0.4.0 S3 (Opus): generate_lateral_bc — frozen stub")
