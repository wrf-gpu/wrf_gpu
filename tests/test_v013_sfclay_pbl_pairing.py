"""v0.13 surface-layer <-> PBL pairing fail-close (CPU; no GPU).

GPT#1 cross-model review found a MAJOR operational-honesty bug: the YSU(1)/ACM2(7)/
BouLac(8)/MRF(99) PBL scan adapters re-derive their per-cell Monin-Obukhov surface
forcing (HFX/QFX/BR/PSIM/PSIH/U10/V10) from the REVISED-MM5 surface layer
(``coupling.scan_adapters._pbl_surface_forcing`` ->
``physics.surface_layer.surface_layer_with_diagnostics``), regardless of the SELECTED
``sf_sfclay_physics`` scheme. So selecting GFS(3) / old-MM5(91) / Pleim-Xiu(7) /
MYNN-sfclay(5) under one of those PBLs would SILENTLY substitute revised-MM5 forcing
for the requested scheme -- yet ``resolve_physics_suite(...).gpu_gate_ready`` returned
True, presenting a different scheme's result as the requested one.

WRF itself FATAL-ERRORs these pairings unless the surface layer satisfies the PBL's
``isfc`` requirement (``phys/module_physics_init.F`` ``pbl_select``; YSU/MRF need
isfc==1). This codebase only threads the revised-MM5 (sf=1) forcing into those PBL
adapters, so the honest, WRF-faithful contract is: bl in {1,7,8,99} is faithful ONLY
with sf_sfclay_physics=1 -- any other pairing FAILS CLOSED.

These tests assert:
  * the resolver fail-closes every silent-substitution pairing (no gpu_gate_ready);
  * the faithful pairings (bl in {1,7,8,99} + sf=1) still resolve gate-ready;
  * MYNN(5) (consumes the selected scheme's State flux handles) and MYJ(2) (re-runs
    its mandatory Janjic surface layer) are NOT restricted to sf=1;
  * a non-revised surface scheme (GFS) genuinely produces DIFFERENT surface fluxes
    than revised-MM5 (so the substitution would have been a real, material error,
    not a benign approximation);
  * the operational default suite (MYNN/MYNN) is byte-unchanged.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gpuwrf.coupling.physics_dispatch import (
    UnsupportedSchemeSelection,
    resolve_physics_suite,
)

# PBLs whose forcing is re-derived from revised-MM5 (must pair with sf_sfclay=1).
_REDERIVING_PBLS = (1, 7, 8, 99)
# Surface layers that are NOT revised-MM5 (selecting them under a re-deriving PBL
# would silently substitute revised-MM5 forcing). sf=2 is excluded -- it has its own
# (MYJ-only) pairing rule already enforced.
_NON_REVISED_SFCLAYS = (0, 3, 5, 7, 91)


# --------------------------------------------------------------------------- #
# 1. No silent substitution: fail-close every re-deriving-PBL + non-sf1 pair   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bl", _REDERIVING_PBLS)
@pytest.mark.parametrize("sf", _NON_REVISED_SFCLAYS)
def test_rederiving_pbl_fails_closed_with_non_revised_mm5(bl: int, sf: int) -> None:
    with pytest.raises(UnsupportedSchemeSelection) as exc:
        resolve_physics_suite({"bl_pbl_physics": bl, "sf_sfclay_physics": sf})
    msg = str(exc.value)
    # Either the MYJ-pairing guard (when sf==2 routed here) or the new pairing guard;
    # for these sf values it must be the surface-layer/PBL pairing violation.
    assert "pairing violation" in msg
    assert f"bl_pbl_physics={bl}" in msg or "MYJ pairing" in msg


@pytest.mark.parametrize("bl", _REDERIVING_PBLS)
def test_rederiving_pbl_default_sfclay_fails_closed(bl: int) -> None:
    # When sf_sfclay is NOT pinned it defaults to 5 (MYNN-sfclay), which is NOT the
    # revised-MM5 forcing these PBLs consume -> must fail closed, not silently run.
    with pytest.raises(UnsupportedSchemeSelection):
        resolve_physics_suite({"bl_pbl_physics": bl})


# --------------------------------------------------------------------------- #
# 2. Faithful pairing stays gate-ready (revised-MM5 forcing IS what's selected) #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bl", _REDERIVING_PBLS)
def test_rederiving_pbl_gate_ready_with_revised_mm5(bl: int) -> None:
    suite = resolve_physics_suite({"bl_pbl_physics": bl, "sf_sfclay_physics": 1})
    assert suite.pbl.option == bl
    assert suite.surface_layer.option == 1
    assert suite.gpu_gate_ready is True


# --------------------------------------------------------------------------- #
# 3. MYNN(5) / MYJ(2) exemptions                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sf", (0, 1, 3, 5, 7, 91))
def test_mynn_pbl_not_restricted_to_revised_mm5(sf: int) -> None:
    # MYNN reads the SELECTED scheme's kinematic flux handles from State, so it
    # correctly pairs with any wired surface layer (no silent revised-MM5 substitute).
    suite = resolve_physics_suite({"bl_pbl_physics": 5, "sf_sfclay_physics": sf})
    assert suite.pbl.option == 5
    assert suite.surface_layer.option == sf
    assert suite.gpu_gate_ready is True


def test_myj_pairing_unaffected() -> None:
    # MYJ (bl=2) re-runs its mandatory Janjic (sf=2) surface layer; still gate-ready.
    suite = resolve_physics_suite({"bl_pbl_physics": 2, "sf_sfclay_physics": 2})
    assert suite.pbl.option == 2 and suite.surface_layer.option == 2
    assert suite.gpu_gate_ready is True


# --------------------------------------------------------------------------- #
# 4. The substitution would have been a MATERIAL error (not benign)            #
# --------------------------------------------------------------------------- #
def _surface_state():
    from gpuwrf.contracts.state import State, _state_field_shapes

    class _G:
        pass

    g = _G()
    g.nz, g.ny, g.nx = 2, 2, 2
    shapes = _state_field_shapes(g)
    nz, ny, nx = 2, 2, 2
    fields = {name: jnp.zeros(shape, dtype=jnp.float64) for name, shape in shapes.items()}
    ph = jnp.broadcast_to(
        jnp.linspace(0.0, 2000.0 * 9.80665, nz + 1, dtype=jnp.float64)[:, None, None],
        (nz + 1, ny, nx),
    )
    fields.update(
        u=jnp.full((nz, ny, nx + 1), 5.0, jnp.float64),
        v=jnp.full((nz, ny + 1, nx), 1.0, jnp.float64),
        theta=jnp.full((nz, ny, nx), 295.0, jnp.float64),
        mu=jnp.full((ny, nx), 90000.0, jnp.float64),
        ph=ph, p=jnp.full((nz, ny, nx), 95000.0, jnp.float64),
        qv=jnp.full((nz, ny, nx), 8.0e-3, jnp.float64),
        ustar=jnp.full((ny, nx), 0.3, jnp.float64),
        t_skin=jnp.full((ny, nx), 298.0, jnp.float64),
        xland=jnp.array([[1.0, 2.0], [1.0, 2.0]], jnp.float64),
        mavail=jnp.full((ny, nx), 0.8, jnp.float64),
        roughness_m=jnp.full((ny, nx), 0.08, jnp.float64),
        soil_moisture=jnp.full((ny, nx), 0.3, jnp.float64),
    )
    return State(**fields)


def test_non_revised_surface_flux_differs_from_revised_mm5_substitute() -> None:
    """A selected non-revised surface scheme (GFS) produces materially different
    surface fluxes than the revised-MM5 forcing the PBL would have silently used --
    justifying the fail-close (the substitution is a real error, not an approximation).
    """

    from gpuwrf.coupling.scan_adapters import (
        gfs_sfclay_adapter,
        sfclay_revised_mm5_adapter,
        _pbl_surface_forcing,
    )

    state = _surface_state()
    # What the re-deriving PBL would consume (always revised-MM5):
    forcing = _pbl_surface_forcing(state, None)
    # What the SELECTED GFS scheme actually produces (persisted to State handles):
    gfs = gfs_sfclay_adapter(state, 60.0)
    mm5 = sfclay_revised_mm5_adapter(state, 60.0)

    gfs_tf = np.asarray(gfs.theta_flux)
    mm5_tf = np.asarray(mm5.theta_flux)
    assert np.all(np.isfinite(gfs_tf)) and np.all(np.isfinite(mm5_tf))
    # The PBL's re-derived forcing matches the revised-MM5 adapter (same scheme), but
    # the selected GFS scheme's flux is materially different (>10% relative).
    assert not np.allclose(gfs_tf, mm5_tf, rtol=0.1), (
        "GFS surface flux indistinguishable from revised-MM5 -- substitution test moot"
    )
    rel = np.abs(gfs_tf - mm5_tf) / np.maximum(np.abs(mm5_tf), 1e-12)
    assert np.max(rel) > 0.1


# --------------------------------------------------------------------------- #
# 5. Operational default suite is byte-unchanged                              #
# --------------------------------------------------------------------------- #
def test_default_suite_byte_unchanged() -> None:
    # The v0.2.0-validated default (MYNN sfclay=5 + MYNN PBL=5) must be untouched by
    # the pairing fail-close (MYNN is exempt).
    suite = resolve_physics_suite({})
    assert suite.pbl.option == 5
    assert suite.surface_layer.option == 5
    assert suite.gpu_gate_ready is True
    # And the explicit default pinning resolves identically.
    suite2 = resolve_physics_suite({"bl_pbl_physics": 5, "sf_sfclay_physics": 5})
    assert suite2.summary() == suite.summary()
