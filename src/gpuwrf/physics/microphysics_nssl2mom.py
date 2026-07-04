"""WRF NSSL 2-moment microphysics (``mp_physics=18``).

REFERENCE-ONLY (v0.23 F2).  This module owns the JAX endpoint name the
dispatcher records (``coupling.physics_dispatch._MP_ENTRIES[18]``).

A real single-column **oracle** built from the UNMODIFIED WRF source is staged:
``proofs/v023/oracle/nssl2mom`` drives the pristine
``phys/module_mp_nssl_2mom.F`` (``nssl_2mom_init`` -> ``nssl_2mom_driver``) in
the WRF-default mp=18 configuration (2-moment + hail + predicted CCN + variable
graupel/hail density, MM2013 fall speeds; every init choice cited to
``module_check_a_mundo.F``/``module_physics_init.F`` in the oracle README), both
canonical fp32 and ``-fdefault-real-8`` fp64, 6 regimes ->
``proofs/v022/f2_oracles/nssl_2mom/nssl{,_fp64}_case_{1..6}.json`` with source
sha256 provenance.  Known documented gap: the hail (QHL) process rates are
unexercised by the current Morrison-mirroring seeds (a hail-seeded supplementary
case is required before any hail-parity claim).

The **traceable JAX column kernel** is a documented carry-over: the driver-level
path exercised by the oracle (``sediment1d``/``ziegfall1d`` fall-speed
sedimentation with hybrid number fallout, ``nssl_2mom_gs`` process rates,
``nucond`` saturation adjustment/nucleation, ``calcnfromq``/``smallvalues``) is
~10k+ LOC of coupled multi-moment microphysics inside the 25k-LOC module, plus
graupel/hail volume scalars (``qvolg``/``qvolh``) that have no operational
State substrate yet.  Shipping a partial kernel would risk silently-wrong
hydrometeors, so NO operational kernel is provided: mp=18 is namelist-accepted
(REFERENCE_ONLY, selectable for single-column oracle comparison) and
fail-closes in the operational scan (not in
``runtime.operational_mode._SCAN_WIRED_OPTIONS``; dispatch entry has
``gpu_runnable=False``).

Cited to ``/home/user/src/wrf_pristine/WRF/phys/module_mp_nssl_2mom.F``
(``nssl_2mom_init`` line ~1248; ``nssl_2mom_driver`` line ~2361).
"""

from __future__ import annotations

NSSL2MOM_ORACLE_DIR = "proofs/v023/oracle/nssl2mom"
NSSL2MOM_SAVEPOINT_DIR = "proofs/v022/f2_oracles/nssl_2mom"

# Frozen mp=18 state contract (WRF Registry packages nssl_2mom + nssl2mconc):
# moist qv/qc/qr/qi/qs/qg(=NSSL graupel QH)/qh(=NSSL hail QHL) and numbers
# Nn(CCN)/Nc/Nr/Ni/Ns/Ng/Nh; the qvolg/qvolh volume scalars are NOT yet State
# leaves (documented blocker for scan wiring).
NSSL2MOM_MOIST_MEMBERS = ("qv", "qc", "qr", "qi", "qs", "qg", "qh")
NSSL2MOM_NUMBER_MEMBERS = ("Nn", "Nc", "Nr", "Ni", "Ns", "Ng", "Nh")
NSSL2MOM_MISSING_STATE_SCALARS = ("qvolg", "qvolh")


def nssl2mom_run(*args, **kwargs):
    """NSSL 2-moment column endpoint -- REFERENCE-ONLY carry-over.

    Raises instead of silently returning a wrong hydrometeor state.  The
    non-self-compare evidence for a future faithful port is the pristine-WRF
    oracle at :data:`NSSL2MOM_ORACLE_DIR` with savepoints at
    :data:`NSSL2MOM_SAVEPOINT_DIR`.
    """

    raise NotImplementedError(
        "mp_physics=18 (NSSL 2-moment) is REFERENCE-ONLY: the faithful traceable "
        f"JAX kernel is a documented carry-over. A real single-column oracle is "
        f"staged at {NSSL2MOM_ORACLE_DIR} ({NSSL2MOM_SAVEPOINT_DIR}) for a future "
        "faithful port. mp=18 fail-closes in the operational scan."
    )


__all__ = [
    "NSSL2MOM_ORACLE_DIR",
    "NSSL2MOM_SAVEPOINT_DIR",
    "NSSL2MOM_MOIST_MEMBERS",
    "NSSL2MOM_NUMBER_MEMBERS",
    "NSSL2MOM_MISSING_STATE_SCALARS",
    "nssl2mom_run",
]
