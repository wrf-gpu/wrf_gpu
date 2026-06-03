"""v0.6.0 operational physics-suite dispatcher tests (CPU; no GPU, no forecast).

Locks the integration behavior: option->scheme routing, fail-closed rejection on
out-of-matrix options, GPU-gate readiness flagging (KF GPU-runnable; GF/Tiedtke
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
)
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
    [("microphysics", 2), ("pbl", 3), ("surface_layer", 3), ("cumulus", 2), ("land_surface", 1)],
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


def test_grell_freitas_and_tiedtke_flagged_not_gpu_ready() -> None:
    # KF (cu=1) is the operational GPU cumulus -> gate-ready.
    assert resolve_physics_suite({"cu_physics": 1}).gpu_gate_ready is True
    # Grell-Freitas (cu=3) and Tiedtke (cu=6/16) are CPU-reference -> excluded.
    for cu in (3, 6, 16):
        suite = resolve_physics_suite({"cu_physics": cu})
        assert suite.gpu_gate_ready is False
        assert suite.cumulus.gpu_runnable is False


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
    assert suite.gpu_gate_ready is False  # GF cu=3 excludes the GPU gate
