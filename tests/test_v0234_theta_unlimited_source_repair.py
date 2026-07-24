"""Source and wiring gates for the WRF-faithful unlimited-theta repair."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import gpuwrf.runtime.operational_mode as operational


PRISTINE_MODULE_EM = Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_em.F")
PRISTINE_ADVECT_SHA256 = "58253bdbeb188dd47ed0579fcd2891086be1889b75c0c7d3696c9ad1d213559d"
CANONICAL_NAMELIST = Path(
    "<DATA_ROOT>/wrf_downscale/artifacts/cpu_oracles/"
    "tenerife_operational_v2_fullbuffer_111x93/20250228_18z/run/wrf/namelist.input"
)
CANONICAL_NAMELIST_SHA256 = "7f8f6099cacafdb1a4e6f0ad562e63081f8110d86bd8bf2bd8f5b4980716a838"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _theta_branch(source: str) -> str:
    start = source.index("!  theta flux divergence")
    stop = source.index("   IF( ieva ) THEN", start)
    return source[start:stop]


def _rk_scalar_branch(source: str) -> str:
    start = source.index("   scalar_loop : DO im = scs, sce")
    stop = source.index("   END DO scalar_loop", start)
    return source[start:stop]


def test_pristine_wrf_separates_theta_from_pd_and_monotonic_scalar_loops():
    source = PRISTINE_MODULE_EM.read_text()
    theta = _theta_branch(source)
    scalar = _rk_scalar_branch(source)

    assert "CALL advect_scalar ( t, t, t_tend" in theta
    assert "advect_scalar_pd" not in theta
    assert "advect_scalar_mono" not in theta
    assert "CALL advect_scalar_pd" in scalar
    assert "CALL advect_scalar_mono" in scalar
    assert "rk_step == rk_order" in scalar


def test_pristine_and_operational_input_authorities_are_immutable():
    assert _sha256(Path("<USER_HOME>/src/wrf_pristine/WRF/dyn_em/module_advect_em.F")) == PRISTINE_ADVECT_SHA256
    assert _sha256(CANONICAL_NAMELIST) == CANONICAL_NAMELIST_SHA256
    compact = "".join(CANONICAL_NAMELIST.read_text().lower().split())
    assert "scalar_adv_opt=1,1,1," in compact


def test_operational_theta_helper_has_no_tracer_limiter_call():
    source = inspect.getsource(operational._augment_large_step_tendencies)
    assert "advect_scalar_flux_limited" not in source
    assert source.count("coupled_tend = advect_scalar_flux(") == 1
    assert "haloed.theta - theta_offset" in source


def test_moisture_and_other_scalar_limiter_wiring_remains_present():
    source = inspect.getsource(operational._scalar_transport_coupled_tendencies)
    assert "advect_moisture_scalars(" in source
    assert "int(advection_opt) in (1, 2)" in source
    assert "int(rk_step) == int(namelist.rk_order)" in source
    number_source = inspect.getsource(operational._nested_number_scalar_coupled_tendencies)
    assert 'species=("Ni", "Nr")' in number_source
    assert "advection_opt=int(namelist.scalar_adv_opt)" in number_source


def test_partial_wind_discriminator_gate_is_unchanged():
    source = inspect.getsource(operational._specified_adv_degrade_active)
    assert "_nested_frozen_wrf_boundary_active(namelist)" in source
    assert "return True" in source

