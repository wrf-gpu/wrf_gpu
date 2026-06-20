"""WRF SSiB SiB biophysical land-surface model (``sf_surface_physics=8``).

REFERENCE-ONLY (v0.17). This module is the JAX column endpoint for the SSiB LSM.
A faithful fp64 single-column **oracle** built from the UNMODIFIED WRF source is
staged (``proofs/v017/oracle/ssib`` -> ``proofs/v017/savepoints/ssib/fp64``),
driving the real ``SSIB`` subroutine and its ~30 internal subroutines. SSiB reads
ALL its vegetation/soil parameters from module-level ``DATA`` arrays (no external
table), so the oracle is fully self-contained against the unmodified module; it is
a genuine non-self-compare reference for a future port.

The **traceable JAX column kernel** is a documented carry-over: SSiB is the most
coupled LSM in WRF (~6.6k LOC) -- a two-stream-radiation (``RADAB``) /
stomatal-resistance (``STOMA1``/``STRES1``) / canopy-interception (``INTERC``) /
implicit canopy-soil energy solve (``TEMRS1``/``TEMRS2`` + ``UPDAT1`` +
``NEWTON``) biophysical column with a 4-level prognostic snow model
(``SNOW_1ST``/``LAYERN``/``NEWSNOW``). Porting it faithfully in a single session
is infeasible without becoming a self-compare/happy-path, and a partial kernel
would risk a silently-wrong land surface, so NO operational kernel is provided:
SSiB is namelist-accepted (selectable for a single-column reference comparison)
and fail-closes in the operational scan (it is NOT in
``runtime.operational_mode._SCAN_WIRED_OPTIONS`` and its dispatch entry has
``gpu_runnable=False``).

This module exists to (1) own the JAX column entrypoint name the dispatcher
records (``coupling.physics_dispatch._SURFACE_ENTRIES[8]``), and (2) freeze the
SSiB land-carry state shape (:class:`SsibLandState`) the future port must thread,
so the interface/registry/dispatch/catalog freeze stays consistent. Calling the
column kernel raises :class:`NotImplementedError` with a pointer to the oracle.

Cited to ``<USER_HOME>/src/wrf_pristine/WRF/phys/module_sf_ssib.F``
(``SSIB`` line 885; ``VEGOUT`` parameter tables line 1995; ``RADAB`` line 3047;
``STOMA1`` line 1910; ``STRES1`` line 4419; ``INTERC`` line 2457; ``TEMRS1``
line 4544; ``TEMRS2`` line 4955; ``UPDAT1`` line 5668; ``NEWTON`` line 2906;
4-level snow: ``SNOW_1ST`` line 4115, ``LAYERN`` line 2734, ``NEWSNOW`` line 2860).
"""

from __future__ import annotations

from typing import NamedTuple

# SSiB carries its own fp64 physical constants; the oracle is compiled at
# -fdefault-real-8. A future JAX port must enable x64. No jax import here (no
# kernel yet) -- this reference-only stub stays import-light, matching the
# fail-closed contract.

# Pointer to the staged non-self-compare oracle.
SSIB_ORACLE_DIR = "proofs/v017/oracle/ssib"
SSIB_SAVEPOINT_GLOB = "proofs/v017/savepoints/ssib/fp64/ssib_case_*.json"

# SSiB soil moisture is a fixed 3-layer column (WWW1/WWW2/WWW3). Documented for
# the port.
SSIB_NUM_SOIL_LAYERS = 3


class SsibLandState(NamedTuple):
    """SSiB SiB biophysical land carry (shapes the future port must thread).

    Mirrors the INOUT land state of WRF ``SSIB``: canopy / soil-surface / deep-soil
    temperatures, the 3-layer soil moisture, and the canopy interception store. The
    leaf names match ``physics_registry.LAND_CARRY_MEMBERS[8]``.

    Per-column leading axis is the flattened grid ``(ncol,)``. (The optional
    4-level prognostic snow sub-state -- ``DZO``/``WO``/``TSSN``/``BWO``/... -- is
    NOT yet threaded; the staged oracle exercises the dry / single-layer-snow path
    ``ISNOW=1``. Threading the multi-layer-snow carry is part of the carry-over.)
    """

    tc: "object"             # (ncol,) canopy temperature TC (K)
    tgs: "object"            # (ncol,) soil-surface temperature TGS (K)
    td: "object"             # (ncol,) deep-soil temperature TD (K)
    www1: "object"           # (ncol,) soil moisture, layer 1 WWW1 (fraction of field cap.)
    www2: "object"           # (ncol,) soil moisture, layer 2 WWW2
    www3: "object"           # (ncol,) soil moisture, layer 3 WWW3
    capac: "object"          # (ncol,) canopy interception / snow store CAPAC

    def replace(self, **updates) -> "SsibLandState":
        return self._replace(**updates)


def ssib_column(*args, **kwargs):
    """SSiB LSM column kernel -- REFERENCE-ONLY carry-over (not yet ported).

    A faithful traceable JAX port of the SSiB SiB biophysical canopy/soil/snow
    column solver is a documented v0.17 carry-over. A fp64 pristine-WRF
    single-column oracle is staged for the future port; see
    :data:`SSIB_ORACLE_DIR` / :data:`SSIB_SAVEPOINT_GLOB`. SSiB fail-closes in the
    operational scan, so this entrypoint is never reached operationally -- it
    raises rather than silently returning a wrong land state.
    """

    raise NotImplementedError(
        "SSiB LSM (sf_surface_physics=8) is REFERENCE-ONLY in v0.17: the faithful "
        "JAX column kernel (TEMRS1/TEMRS2 + UPDAT1 + RADAB + STOMA1 + INTERC + "
        "STRES1 + NEWTON, plus the 4-level snow model) is a documented carry-over. "
        f"A fp64 pristine-WRF single-column oracle is staged at {SSIB_ORACLE_DIR} "
        f"({SSIB_SAVEPOINT_GLOB}) for a future faithful port. SSiB fail-closes in "
        "the operational scan."
    )


# vmap entry name parity with the other LSM kernels (slab_columns / pxlsm_columns).
def ssib_columns(*args, **kwargs):
    """Batched SSiB column entry -- REFERENCE-ONLY carry-over (see :func:`ssib_column`)."""

    return ssib_column(*args, **kwargs)


__all__ = [
    "SSIB_ORACLE_DIR",
    "SSIB_SAVEPOINT_GLOB",
    "SSIB_NUM_SOIL_LAYERS",
    "SsibLandState",
    "ssib_column",
    "ssib_columns",
]
