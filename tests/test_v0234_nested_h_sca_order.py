"""Source and loader gates for WRF scalar-order threading in live nesting."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from gpuwrf.integration.nested_pipeline import _domain_int, _make_namelist
from tests.dynamics.test_diffopt1_smagorinsky_integration import (
    _build_grid,
    _namelist,
)


REGISTRY = Path("<USER_HOME>/src/wrf_pristine/WRF/Registry/Registry.EM_COMMON")
WRF_RHS_PH = Path(
    "<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F"
)
CANONICAL_NAMELIST = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/config/namelist.input"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build(*, order: int = 5, moist_adv_opt: int = 0, scalar_adv_opt: int = 0):
    grid = _build_grid(ny=8, nx=10, nz=5, dx=4000.0)
    return _make_namelist(
        grid=grid,
        tendencies=_namelist(grid).tendencies,
        metrics=grid.metrics,
        dt_s=6.0,
        parent_dt_s=18.0,
        run_start=datetime(2025, 3, 1, tzinfo=timezone.utc),
        radiation_static=None,
        cu_physics=0,
        diff_opt=1,
        km_opt=4,
        h_sca_adv_order=order,
        moist_adv_opt=moist_adv_opt,
        scalar_adv_opt=scalar_adv_opt,
    )


def test_pristine_registry_default_and_canonical_omission_are_authenticated() -> None:
    assert _sha256(REGISTRY) == (
        "6f3ee02175b76487c5c6c046ff2fb4c5d41980b86c0f06eb2e09e34dacc9623a"
    )
    assert _sha256(WRF_RHS_PH) == (
        "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
    )
    assert _sha256(CANONICAL_NAMELIST) == (
        "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838"
    )
    registry_line = next(
        line for line in REGISTRY.read_text().splitlines()
        if "rconfig   integer     h_sca_adv_order" in line
    )
    assert registry_line.split()[5] == "5"
    assert "h_sca_adv_order" not in CANONICAL_NAMELIST.read_text().lower()


def test_live_nested_loader_defaults_to_wrf_order_five() -> None:
    assert _build().h_sca_adv_order == 5


def test_live_nested_loader_preserves_explicit_order_override() -> None:
    assert _build(order=6).h_sca_adv_order == 6


def test_live_nested_loader_preserves_actual_scalar_options() -> None:
    text = CANONICAL_NAMELIST.read_text().lower()
    assert "moist_adv_opt = 1, 1, 1" in text
    assert "scalar_adv_opt = 1, 1, 1" in text
    namelist = _build(moist_adv_opt=1, scalar_adv_opt=1)
    assert (namelist.moist_adv_opt, namelist.scalar_adv_opt) == (1, 1)


def test_per_domain_scalar_order_resolution_matches_wrf_namelist_semantics() -> None:
    class Explicit:
        namelist = {"dynamics": {"h_sca_adv_order": [5, 6, 4]}}

    class Scalar:
        namelist = {"dynamics": {"h_sca_adv_order": 6}}

    class Omitted:
        namelist = {"dynamics": {}}

    assert [
        _domain_int(Explicit(), "dynamics", "h_sca_adv_order", f"d0{i}", 5)
        for i in (1, 2, 3)
    ] == [5, 6, 4]
    assert [
        _domain_int(Scalar(), "dynamics", "h_sca_adv_order", f"d0{i}", 5)
        for i in (1, 2, 3)
    ] == [6, 6, 6]
    assert [
        _domain_int(Omitted(), "dynamics", "h_sca_adv_order", f"d0{i}", 5)
        for i in (1, 2, 3)
    ] == [5, 5, 5]


def test_per_domain_scalar_option_resolution_matches_actual_wrf_namelist() -> None:
    class Explicit:
        namelist = {
            "dynamics": {
                "moist_adv_opt": [1, 2, 0],
                "scalar_adv_opt": [1, 1, 2],
            }
        }

    class Omitted:
        namelist = {"dynamics": {}}

    assert [
        _domain_int(Explicit(), "dynamics", "moist_adv_opt", f"d0{i}", 0)
        for i in (1, 2, 3)
    ] == [1, 2, 0]
    assert [
        _domain_int(Explicit(), "dynamics", "scalar_adv_opt", f"d0{i}", 0)
        for i in (1, 2, 3)
    ] == [1, 1, 2]
    assert [
        _domain_int(Omitted(), "dynamics", key, "d03", 0)
        for key in ("moist_adv_opt", "scalar_adv_opt")
    ] == [0, 0]
