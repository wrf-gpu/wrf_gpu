"""WRF NSSL 2-moment microphysics (``mp_physics=18``) -- public entry points.

v0.3.4 (lane o1-nssl): JAX port of the UNMODIFIED ``phys/module_mp_nssl_2mom.F`` (sha256 29f42e76...)
default mp=18 configuration (2-moment + hail + predicted/activated CCN + graupel/hail volume, MM2013
fall speeds) in :mod:`gpuwrf.physics.nssl2mom`:

* column kernel ``nssl2mom.driver.nssl2mom_column`` = WRF ``nssl_2mom_driver`` (calcnfromq at
  itimestep==1, sediment1d, nssl_2mom_gs, NUCOND, smallvalues, radardd02/calc_eff_radius);
* operational State adapter ``nssl2mom.adapter.nssl2mom_adapter`` (scan-wired for single/root
  domains in ``runtime.operational_mode``; nested domains fail closed).

Qualification: CPU-oracle-qualified -- stage-by-stage and end-to-end parity against the pristine-WRF
oracle ``proofs/v034/f2_oracles/nssl_2mom`` (17 cases, fp64 <= 1e-12 relative, fp32 dual-reference
band); GPU / coupled-forecast qualification pending.
"""

from __future__ import annotations

NSSL2MOM_ORACLE_DIR = "proofs/v034/oracle/nssl2mom"
NSSL2MOM_SAVEPOINT_DIR = "proofs/v034/f2_oracles/nssl_2mom"

# WRF Registry members of the default mp=18 configuration (packages nssl_2mom + nssl2mconc + nssl_hail +
# nssl_ccn_opt + nssl_hailvol) and their State leaves: qg = NSSL graupel, qh = NSSL hail,
# qndrop -> Nc, qnn (activated CCN) -> Nn, qvolg/qvolh = graupel/hail particle volume.
NSSL2MOM_MOIST_MEMBERS = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
NSSL2MOM_NUMBER_MEMBERS = ("Nn", "Nc", "Nr", "Ni", "Ns", "Ng", "Nh")
NSSL2MOM_VOLUME_MEMBERS = ("qvolg", "qvolh")


def nssl2mom_run(fields, dt, *, itimestep: int = 2, precision: str = "fp32", diag: bool = False):
    """One WRF ``nssl_2mom_driver`` call on WRF-named column fields ``(..., nz)`` (k=0 surface).

    ``fields``: th qv qc qr qi qs qg qh qndrop qnr qni qns qng qnh qnn qvolg qvolh (mixing ratios,
    #/kg, m3/kg) + pii p w dz rho.  Returns ``(out, precip, diagnostics)`` (see
    :func:`gpuwrf.physics.nssl2mom.driver.nssl2mom_column`).
    """
    from gpuwrf.physics.nssl2mom.constants import get_constants
    from gpuwrf.physics.nssl2mom.driver import nssl2mom_column
    from gpuwrf.physics.nssl2mom.indices import FP32, FP64

    prec = FP32 if precision == "fp32" else FP64
    return nssl2mom_column(fields, dt, get_constants(prec.name), prec, itimestep=itimestep, diag=diag)


__all__ = [
    "NSSL2MOM_ORACLE_DIR",
    "NSSL2MOM_SAVEPOINT_DIR",
    "NSSL2MOM_MOIST_MEMBERS",
    "NSSL2MOM_NUMBER_MEMBERS",
    "NSSL2MOM_VOLUME_MEMBERS",
    "nssl2mom_run",
]
