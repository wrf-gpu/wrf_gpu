"""v0.18 LSM family — CLM4 (sf=5) / CTSM (sf=6) ARCHITECTURE-BOUNDARY proof.

This is the documented-boundary proof object the v0.18 ship-gate requires for the
two land-surface schemes carried OPEN to the v1.0 CAM/CLM/CTSM-family ADR:

* CLM4 (``sf_surface_physics=5``) — ``phys/module_sf_clm.F`` is ~61.5k LOC built
  around a single global ``clmtype`` PFT/column/landunit/gridcell subgrid
  hierarchy initialized from an EXTERNAL CLM surface dataset, so a faithful
  single-column pristine-WRF oracle is itself architecture-scale (multi-session).
* CTSM (``sf_surface_physics=6``) — ``phys/module_sf_ctsm.F`` is compiled only
  under ``-DWRF_USE_CTSM`` and runs the external CESM/CTSM land model via LILAC
  (empty WRF-side Registry state); there is no in-core WRF physics to oracle.

Per the manager's ruling these are NOT silently gapped and NOT happy-path stubs:
each is catalog-RECOGNIZED yet FAILS CLOSED with a clear named architecture-
boundary reason at the namelist gate AND at the operational dispatcher. This test
asserts exactly that contract — it is the boundary's proof object, so the
boundary is closed (not a silent gap) only while this test is green.

Contrast with the reference-only LSMs (RUC sf=3, SSiB sf=8) which have a real
fp64 pristine-WRF single-column oracle staged and are a DIFFERENT support class
(``REFERENCE_ONLY``); this test pins that distinction so the two are never
conflated.
"""

from __future__ import annotations

import pytest

from gpuwrf.contracts.physics_registry import ACCEPTED_SF_SURFACE_PHYSICS
from gpuwrf.coupling.physics_dispatch import UnsupportedSchemeSelection, scheme_entry
from gpuwrf.io.namelist_check import UnsupportedSchemeError, validate_namelist
from gpuwrf.io.scheme_catalog import (
    SupportStatus,
    assert_catalog_consistent,
    classify_scheme,
)

# (code, WRF name substring, reason key-facts that must appear)
_BOUNDARY = (
    (5, "CLM4", ("61.5k", "clmtype", "surface dataset")),
    (6, "CTSM", ("WRF_USE_CTSM", "LILAC", "external")),
)


@pytest.mark.parametrize("code, name, facts", _BOUNDARY)
def test_clm_ctsm_catalog_recognized_but_fail_closed(code, name, facts) -> None:
    """Catalog RECOGNIZES sf=5/6 as the named WRF scheme yet fails them closed."""
    support = classify_scheme("sf_surface_physics", code)
    # recognized: the catalog knows the real WRF scheme name (not "unknown option")
    assert support.wrf_name is not None and name in support.wrf_name
    # fail-closed class — NOT implemented, NOT reference-only (no oracle), NOT a stub
    assert support.status is SupportStatus.RECOGNIZED_FAIL_CLOSED
    assert not support.status.passes_namelist_check
    # the reason is the SPECIFIC architecture-boundary message, not the generic
    # "NOT YET IMPLEMENTED" placeholder
    assert "ARCHITECTURE BOUNDARY" in support.reason
    assert "v0.18->v1.0" in support.reason
    for fact in facts:
        assert fact in support.reason, f"missing key fact {fact!r} for {name}"
    # the named operational alternative points at a real GPU-operational LSM
    assert "Noah-MP" in support.alternative or "sf_surface_physics=4" in support.alternative
    assert "CAM/CLM/CTSM" in support.alternative


@pytest.mark.parametrize("code, name, facts", _BOUNDARY)
def test_clm_ctsm_namelist_selection_errors_cleanly(code, name, facts) -> None:
    """Selecting sf=5/6 in a namelist raises a clear, named architecture error."""
    with pytest.raises(UnsupportedSchemeError) as excinfo:
        validate_namelist({"physics": {"sf_surface_physics": [code]}})
    message = str(excinfo.value)
    assert name in message, f"rejection must name {name}"
    assert "ARCHITECTURE-BOUNDARY" in message or "architecture-boundary" in message
    assert "CAM/CLM/CTSM" in message
    # never silently substituted by another LSM
    sel = [s for s in excinfo.value.selections if s.key == "sf_surface_physics"]
    assert sel, "sf_surface_physics must be the rejected key"


@pytest.mark.parametrize("code, name, facts", _BOUNDARY)
def test_clm_ctsm_operational_dispatcher_fail_closed(code, name, facts) -> None:
    """Defense-in-depth: even bypassing the namelist gate, the operational
    dispatcher refuses sf=5/6 as out-of-matrix — never a happy-path route."""
    with pytest.raises(UnsupportedSchemeSelection):
        scheme_entry("land_surface", code)


def test_clm_ctsm_not_in_accept_matrix() -> None:
    """The frozen accept-matrix must NOT list sf=5/6 (they are boundary, not run)."""
    assert 5 not in ACCEPTED_SF_SURFACE_PHYSICS
    assert 6 not in ACCEPTED_SF_SURFACE_PHYSICS


def test_boundary_distinct_from_reference_only() -> None:
    """CLM4/CTSM (no oracle) must be a DIFFERENT class from the reference-only
    RUC/SSiB (real fp64 oracle staged) — never conflated."""
    for ref in (3, 8):
        assert classify_scheme("sf_surface_physics", ref).status is SupportStatus.REFERENCE_ONLY
    for boundary in (5, 6):
        assert classify_scheme("sf_surface_physics", boundary).status is SupportStatus.RECOGNIZED_FAIL_CLOSED


def test_catalog_still_internally_consistent() -> None:
    """Adding the CLM4/CTSM boundary reasons must not break catalog invariants."""
    assert_catalog_consistent()
