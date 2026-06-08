"""v0.6.0 operational physics-suite dispatcher tests (CPU; no GPU, no forecast).

Locks the integration behavior: option->scheme routing, fail-closed rejection on
out-of-matrix options, GPU-gate readiness flagging (KF/BMJ GPU-runnable; GF/Tiedtke
CPU-reference), the v0.2.0 baseline default, and that every routed entrypoint
actually exists on its module (no dangling reference after the 12-lane merge).
"""

from __future__ import annotations

import importlib

import pytest

from gpuwrf.contracts.physics_registry import (
    ACCEPTED_BL_PBL_PHYSICS,
    ACCEPTED_CU_PHYSICS,
    ACCEPTED_MP_PHYSICS,
    ACCEPTED_SF_SFCLAY_PHYSICS,
    ACCEPTED_SF_SURFACE_PHYSICS,
    CU_SCHEMES,
)
from gpuwrf.coupling.scan_adapters import CU_SCAN_ADAPTERS
from gpuwrf.coupling.physics_dispatch import (
    DEFAULT_BL_PBL_PHYSICS,
    DEFAULT_CU_PHYSICS,
    DEFAULT_MP_PHYSICS,
    DEFAULT_SF_SFCLAY_PHYSICS,
    DEFAULT_SF_SURFACE_PHYSICS,
    UnsupportedSchemeSelection,
    dispatch_matrix,
    resolve_physics_suite,
    scheme_entry,
)


def test_default_suite_is_v020_baseline_and_gpu_ready() -> None:
    suite = resolve_physics_suite({})
    assert suite.microphysics.option == DEFAULT_MP_PHYSICS == 8
    assert suite.pbl.option == DEFAULT_BL_PBL_PHYSICS == 5
    assert suite.surface_layer.option == DEFAULT_SF_SFCLAY_PHYSICS == 5
    assert suite.cumulus.option == DEFAULT_CU_PHYSICS == 0
    assert suite.land_surface.option == DEFAULT_SF_SURFACE_PHYSICS == 4
    assert suite.gpu_gate_ready is True
    assert suite.non_gpu_schemes == ()


def test_every_accepted_option_routes() -> None:
    for fam, accepted in (
        ("microphysics", ACCEPTED_MP_PHYSICS),
        ("pbl", ACCEPTED_BL_PBL_PHYSICS),
        ("surface_layer", ACCEPTED_SF_SFCLAY_PHYSICS),
        ("cumulus", ACCEPTED_CU_PHYSICS),
        ("land_surface", ACCEPTED_SF_SURFACE_PHYSICS),
    ):
        for opt in accepted:
            entry = scheme_entry(fam, opt)
            assert entry.family == fam and entry.option == opt


@pytest.mark.parametrize(
    "fam,opt",
    # surface_layer=3 (GFS) / land_surface=1 (slab) became accepted in v0.13
    # Tier-3; remaining out-of-matrix: surface_layer=4 (QNSE), land_surface=3 (RUC).
    [("microphysics", 5), ("pbl", 3), ("surface_layer", 4), ("cumulus", 5), ("land_surface", 3)],
)
def test_fail_closed_on_out_of_matrix(fam: str, opt: int) -> None:
    with pytest.raises(UnsupportedSchemeSelection):
        scheme_entry(fam, opt)


def test_myj_pairing_enforced_by_dispatcher_resolution() -> None:
    suite = resolve_physics_suite({"bl_pbl_physics": 2, "sf_sfclay_physics": 2})
    assert suite.pbl.option == 2
    assert suite.surface_layer.option == 2
    for config in ({"bl_pbl_physics": 2}, {"sf_sfclay_physics": 2}, {"bl_pbl_physics": 2, "sf_sfclay_physics": 5}):
        with pytest.raises(UnsupportedSchemeSelection, match="MYJ pairing violation"):
            resolve_physics_suite(config)


def test_cumulus_gpu_readiness_flags() -> None:
    # KF (cu=1), BMJ (cu=2), Grell-Freitas (cu=3), and Tiedtke (cu=6) are
    # operational GPU cumulus options (scan-wired) -> gate-ready. GF (cu=3) is the
    # v0.9.0 GPU-batched jit/vmap scale-aware adapter (CU_SCAN_ADAPTERS[3]).
    for cu in (1, 2, 3, 6):
        suite = resolve_physics_suite({"cu_physics": cu})
        assert suite.gpu_gate_ready is True
        assert suite.cumulus.gpu_runnable is True
    # New Tiedtke (cu=16, not separately source-gated) is fail-closed -> excluded.
    suite16 = resolve_physics_suite({"cu_physics": 16})
    assert suite16.gpu_gate_ready is False
    assert suite16.cumulus.gpu_runnable is False


def test_kf_is_implemented_and_scan_wired() -> None:
    assert CU_SCHEMES[1].status == "implemented"
    assert CU_SCAN_ADAPTERS[1].__name__ == "kf_adapter"

    suite = resolve_physics_suite({"cu_physics": 1})
    assert suite.cumulus.owner_module == "gpuwrf.physics.cumulus_kf"
    assert suite.cumulus.entrypoint == "step_kf_column"
    assert suite.cumulus.gpu_runnable is True
    assert suite.gpu_gate_ready is True


def test_use_noahmp_toggle_maps_land_surface() -> None:
    assert resolve_physics_suite({"use_noahmp": True}).land_surface.option == 4
    assert resolve_physics_suite({"use_noahmp": False}).land_surface.option == 2


def test_all_routed_entrypoints_exist_on_module() -> None:
    matrix = dispatch_matrix()
    for row in matrix["rows"]:
        if row["convention"] == "disabled":
            continue
        module = importlib.import_module(row["module"])
        assert hasattr(module, row["entrypoint"]), (row["module"], row["entrypoint"])


def test_nested_wrf_style_mapping_resolves() -> None:
    suite = resolve_physics_suite(
        {"physics": {"mp_physics": [6, 6], "bl_pbl_physics": 1, "cu_physics": 3}}
    )
    assert suite.microphysics.option == 6
    assert suite.pbl.option == 1
    assert suite.cumulus.option == 3
    # GF cu=3 is now the v0.9.0 GPU-batched scan-wired adapter -> gate-ready
    # (WSM6 + YSU + GF are all GPU-runnable).
    assert suite.gpu_gate_ready is True
