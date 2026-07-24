"""Source and loader gates for the nested WRF theta-tendency cadence."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

from gpuwrf.integration.nested_pipeline import _make_namelist
from tests.dynamics.test_diffopt1_smagorinsky_integration import (
    _build_grid,
    _namelist,
)


WRF_FIRST_RK = Path(
    "<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_first_rk_step_part2.F"
)
WRF_BIG_STEP = Path(
    "<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F"
)
WRF_MODULE_EM = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_em.F")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested_namelist(monkeypatch, value: str | None):
    if value is None:
        monkeypatch.delenv("GPUWRF_PHYS_RK_TENDF", raising=False)
    else:
        monkeypatch.setenv("GPUWRF_PHYS_RK_TENDF", value)
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
    )


def test_pristine_wrf_orders_update_conversion_and_rk_reuse() -> None:
    # Bind the exact pristine source files used by the causal claim.
    assert _sha256(WRF_FIRST_RK) == "9c8c06b246c1cb1bb1632c25e5683a0cfd8545e6832df6eeaa1c744524325738"
    assert _sha256(WRF_BIG_STEP) == "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
    assert _sha256(WRF_MODULE_EM) == "11105cbf8255f30ca6a44cd7429a92cedce1fb91db6ce90fd7217002a72fb7fa"

    first = WRF_FIRST_RK.read_text()
    update = first.index("CALL update_phy_ten")
    convert = first.index("CALL conv_t_tendf_to_moist", update)
    diffusion = first.index("! calculate vertical diffusion first", convert)
    assert update < convert < diffusion

    big = WRF_BIG_STEP.read_text()
    conv = big.index("SUBROUTINE conv_t_tendf_to_moist")
    formula = big.index("t_tendf(i,k,j) = (1. + (R_v/R_d)", conv)
    assert conv < formula

    module = WRF_MODULE_EM.read_text()
    merge = module.index("SUBROUTINE rk_addtend_dry")
    reuse = module.index("t_tend(i,k,j) =  t_tend(i,k,j) +  t_tendf", merge)
    assert merge < reuse


def test_nested_loader_matches_real_case_default_and_exact_zero_rollback(
    monkeypatch,
) -> None:
    assert _nested_namelist(monkeypatch, None).rad_rk_tendf == 1
    assert _nested_namelist(monkeypatch, "1").rad_rk_tendf == 1
    assert _nested_namelist(monkeypatch, "0").rad_rk_tendf == 0


def test_nested_loader_does_not_treat_ambiguous_text_as_rollback(monkeypatch) -> None:
    # Match daily_pipeline exactly: only the explicit value "0" disables the
    # WRF source path.  Unknown text cannot silently turn model physics off.
    for value in ("false", "off", " 0 ", "00"):
        assert _nested_namelist(monkeypatch, value).rad_rk_tendf == 1
