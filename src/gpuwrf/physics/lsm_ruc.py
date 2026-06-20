"""WRF RUC multi-layer soil/snow land-surface model (``sf_surface_physics=3``).

REFERENCE-ONLY (v0.17). This module is the JAX column endpoint for the RUC LSM.
A faithful fp64 single-column **oracle** built from the UNMODIFIED WRF source is
staged (``proofs/v017/oracle/ruclsm`` -> ``proofs/v017/savepoints/ruclsm/fp64``),
driving the real ``LSMRUC`` -> ``SOILVEGIN`` -> ``SFCTMP`` call path with the
unmodified ``VEGPARM.TBL``/``SOILPARM.TBL``/``GENPARM.TBL`` parameter tables, so a
future faithful port has a ready non-self-compare reference.

The **traceable JAX column kernel** is a documented carry-over: RUC's column path
(``SFCTMP`` + ``SOIL``/``SNOWSOIL`` + ``SOILTEMP``/``SNOWTEMP``/``SOILMOIST`` +
``SOILPROP``/``TRANSF``/``VILKA``, ~7.5k LOC of coupled multi-layer soil/snow
thermodynamics + hydrology with an implicit tridiagonal soil solve and a
multi-layer snow model) is far too large to port faithfully in a single session
without becoming a self-compare/happy-path. Shipping a partial kernel would risk
a silently-wrong land surface, so NO operational kernel is provided: RUC is
namelist-accepted (selectable for a single-column reference comparison) and
fail-closes in the operational scan (it is NOT in
``runtime.operational_mode._SCAN_WIRED_OPTIONS`` and its dispatch entry has
``gpu_runnable=False``).

This module exists to (1) own the JAX column entrypoint name the dispatcher
records (``coupling.physics_dispatch._SURFACE_ENTRIES[3]``), and (2) freeze the
RUC land-carry state shape (:class:`RucLandState`) the future port must thread,
so the interface/registry/dispatch/catalog freeze stays consistent. Calling the
column kernel raises :class:`NotImplementedError` with a pointer to the oracle.

Cited to ``<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_ruclsm.F``
(``LSMRUC`` line 84; ``SFCTMP`` line 1180; ``SOIL`` line 2229; ``SNOWSOIL`` line
3120; ``SOILTEMP`` line 4530; ``SNOWTEMP`` line 4836; ``SOILMOIST`` line 5732;
``SOILPROP`` line 6092; ``TRANSF`` line 6305; ``VILKA`` line 6486; ``SOILVEGIN``
line 6526; ``RUCLSM_SOILVEGPARM`` line 7149).
"""

from __future__ import annotations

from typing import NamedTuple

# fp64 path: the RUC oracle is compiled at -fdefault-real-8; any future JAX port
# must enable x64 to reproduce it. We do not import jax here (no kernel yet) to
# keep this reference-only stub import-light, matching the fail-closed contract.

# Pointer to the staged non-self-compare oracle.
RUC_ORACLE_DIR = "proofs/v017/oracle/ruclsm"
RUC_SAVEPOINT = "proofs/v017/savepoints/ruclsm/fp64/ruclsm_fp64.json"

# WRF RUC default operational soil configuration (the oracle uses nsl=6 RUC soil
# levels zs = [0, 0.05, 0.20, 0.40, 1.00, 2.00] m). Documented for the port.
RUC_NUM_SOIL_LAYERS = 6


class RucLandState(NamedTuple):
    """RUC multi-layer soil/snow land carry (shapes the future port must thread).

    Mirrors the INOUT land state of WRF ``LSMRUC``: a multi-layer soil column
    (``num_soil_layers = RUC_NUM_SOIL_LAYERS``) of temperature / total+liquid
    moisture / frozen fraction, plus the skin temperature, the snow water+depth,
    and the diagnostic surface moisture. The leaf names match
    ``physics_registry.LAND_CARRY_MEMBERS[3]``.

    Per-column leading axis is the flattened grid ``(ncol,)``; soil-profile leaves
    carry an extra trailing soil-layer axis ``(ncol, RUC_NUM_SOIL_LAYERS)``.
    """

    soilt: "object"          # (ncol,) skin / surface soil temperature SOILT (K)
    tso: "object"            # (ncol, nsl) soil-layer temperatures TSO (K)
    soilmois: "object"       # (ncol, nsl) total volumetric soil moisture SOILMOIS
    sh2o: "object"           # (ncol, nsl) liquid (unfrozen) soil moisture SH2O
    smfr3d: "object"         # (ncol, nsl) frozen soil-moisture fraction SMFR3D
    keepfr3dflag: "object"   # (ncol, nsl) KEEPFR3DFLAG hysteresis flag
    snow: "object"           # (ncol,) snow water equivalent (mm)
    snowh: "object"          # (ncol,) snow depth (m)
    qsfc: "object"           # (ncol,) diagnostic surface specific humidity

    def replace(self, **updates) -> "RucLandState":
        return self._replace(**updates)


def ruc_column(*args, **kwargs):
    """RUC LSM column kernel -- REFERENCE-ONLY carry-over (not yet ported).

    A faithful traceable JAX port of the RUC multi-layer soil/snow column solver
    is a documented v0.17 carry-over. A fp64 pristine-WRF single-column oracle is
    staged for the future port; see :data:`RUC_ORACLE_DIR` / :data:`RUC_SAVEPOINT`.
    RUC fail-closes in the operational scan, so this entrypoint is never reached
    operationally -- it raises rather than silently returning a wrong land state.
    """

    raise NotImplementedError(
        "RUC LSM (sf_surface_physics=3) is REFERENCE-ONLY in v0.17: the faithful "
        "JAX column kernel (SFCTMP + SOIL/SNOWSOIL + SOILTEMP/SNOWTEMP/SOILMOIST + "
        "SOILPROP/TRANSF/VILKA) is a documented carry-over. A fp64 pristine-WRF "
        f"single-column oracle is staged at {RUC_ORACLE_DIR} ({RUC_SAVEPOINT}) for "
        "a future faithful port. RUC fail-closes in the operational scan."
    )


# vmap entry name parity with the other LSM kernels (slab_columns / pxlsm_columns).
def ruc_columns(*args, **kwargs):
    """Batched RUC column entry -- REFERENCE-ONLY carry-over (see :func:`ruc_column`)."""

    return ruc_column(*args, **kwargs)


__all__ = [
    "RUC_ORACLE_DIR",
    "RUC_SAVEPOINT",
    "RUC_NUM_SOIL_LAYERS",
    "RucLandState",
    "ruc_column",
    "ruc_columns",
]
