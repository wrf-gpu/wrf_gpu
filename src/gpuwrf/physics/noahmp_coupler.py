"""Noah-MP <-> surface/PBL/dycore coupling adapter (Sprint S6) — FREEZE STUB.

The handshake that plugs prognostic Noah-MP into the existing surface->PBL->dycore
chain. LAND-ONLY MASKED: ocean/lake columns keep the current prescribed-SST bulk
``surface_layer_with_diagnostics`` path VERBATIM (byte-for-byte unchanged).

This is a NEW module. ``runtime.operational_mode.surface_adapter`` will call it
once implemented; operational_mode.py is NOT edited here (perf-sidecar worktree
owns it). See ADR-NOAHMP-INTERFACES.md §4.

Per physics step (frozen sequence):
  1. sfclay first (UNCHANGED): ``surface_layer_with_diagnostics(state)`` over ALL
     columns -> CH/CM/ustar/tau + water/lake HFX/LH/QFX/T2/Q2. opt_sfc=1: sfclay
     OWNS CH/CM and feeds them INTO Noah-MP.
  2. Noah-MP over land: ``noah_mp_step(land_state, forcing, static, dt)`` -> land
     HFX/LH/QFX/TSK/albedo/emiss/ZNT.
  3. Masked blend (the ONLY land/water flux switch):
       hfx = where(is_land, noahmp.hfx, sfclay.hfx); likewise lh/qfx/tsk/znt.
     Rebuild kinematic handles (theta_flux/qv_flux/fltv) from the BLENDED flux,
     using the identical formulae as surface_layer.py:710-715.
  4. PBL bottom BC: the blended SurfaceFluxes is passed as ``surface=`` into the
     MYNN column (mynn_pbl._surface_terms, the FROZEN Gate-1 hand-off). mynn_pbl
     is NOT changed.
  5. Write-back: return (state', land_state') with state' carrying blended
     t_skin/roughness_m/qsfc via State.replace; land_state' threaded to next step.

Invariants: no in-loop host transfer; ocean path unchanged; CH/CM from sfclay;
dycore + microphysics + State.__slots__ untouched.
"""

from __future__ import annotations

from typing import Any

from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic
from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes
from gpuwrf.physics.noahmp.types import NoahMPForcing


def assemble_noahmp_forcing(
    state: Any,
    static: NoahMPStatic,
    radiation: Any,
    clock: Any,
    dt: float,
) -> NoahMPForcing:
    """Build the Noah-MP forcing pytree from device state — STUB.

    Pulls the atmosphere lowest level (sfctmp/sfcprs/uu/vv/qair/qc), radiation
    (soldn/lwdn/cosz), the microphysics precip partition, and the clock
    (julian/yearlen) into a ``NoahMPForcing``. No host transfer.
    """

    raise NotImplementedError("assemble_noahmp_forcing: Sprint S6 coupler")


def noahmp_surface_adapter(
    state: Any,
    land_state: NoahMPLandState,
    static: NoahMPStatic,
    radiation: Any,
    clock: Any,
    dt: float,
) -> tuple[Any, NoahMPLandState, SurfaceFluxes]:
    """Run the land-masked Noah-MP / sfclay blend for one physics step — STUB.

    Returns ``(state', land_state', blended_surface_fluxes)``:
      - ``state'``  : ``State`` with blended t_skin/roughness_m/qsfc written back.
      - ``land_state'`` : advanced prognostic Noah-MP land carry.
      - ``blended_surface_fluxes`` : the ``SurfaceFluxes`` passed as ``surface=``
        into the MYNN PBL column (the frozen Gate-1 hand-off).
    Ocean/lake columns take the sfclay branch unchanged.
    """

    raise NotImplementedError("noahmp_surface_adapter: Sprint S6 coupling adapter")


__all__ = ["assemble_noahmp_forcing", "noahmp_surface_adapter"]
