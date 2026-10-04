"""CPU regression gates for single-domain scalar-advection namelist binding."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.state import Tendencies
from gpuwrf.integration import daily_pipeline
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.runtime.operational_mode import OperationalNamelist


class _MinimalState:
    """Only the aliases touched by ``_build_real_case`` are needed here."""

    p_total = object()
    ph_total = object()
    mu_total = object()

    def replace(self, **_kwargs):
        return self


def _cpu_tendencies(grid: GridSpec) -> Tendencies:
    nz, ny, nx = grid.nz, grid.ny, grid.nx

    def zeros(shape):
        return jnp.zeros(shape, dtype=jnp.float64)

    return Tendencies(
        zeros((nz, ny, nx + 1)),
        zeros((nz, ny + 1, nx)),
        zeros((nz + 1, ny, nx)),
        zeros((nz, ny, nx)),
        zeros((nz, ny, nx)),
        zeros((nz, ny, nx)),
        zeros((nz + 1, ny, nx)),
        zeros((ny, nx)),
    )


def _build_daily_case(monkeypatch, tmp_path, dynamics_lines: str):
    (tmp_path / "namelist.input").write_text(
        (
            "&domains\n"
            " max_dom = 1,\n"
            "/\n"
            "&dynamics\n"
            f"{dynamics_lines}"
            "/\n"
        ),
        encoding="utf-8",
    )
    run = Gen2Run(tmp_path)
    run.grid = lambda _domain: SimpleNamespace()

    grid = GridSpec.canary_3km_template()
    replay = SimpleNamespace(
        run=run,
        state=_MinimalState(),
        tendencies=_cpu_tendencies(grid),
        grid=grid,
        metrics=grid.metrics,
        metadata={
            "run_id": "daily-c2-binding",
            "run_start_label": "2026-07-24_00:00:00",
            "grid": {},
            "boundary": {},
            "standalone_native_init": True,
        },
    )

    monkeypatch.setattr(
        daily_pipeline, "build_replay_case", lambda _run_dir, domain: replay
    )
    monkeypatch.setattr(daily_pipeline, "dealias_state_buffers", lambda state: state)
    monkeypatch.setattr(
        daily_pipeline,
        "load_radiation_static",
        lambda *_args, **_kwargs: (None, {"status": "test"}),
    )
    monkeypatch.setattr(
        daily_pipeline,
        "_load_static_latlon_writer_diagnostics",
        lambda *_args, **_kwargs: (None, {"status": "test"}),
    )
    monkeypatch.setattr(
        daily_pipeline,
        "bind_wrfout_domain_authority",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    case, run_dir = daily_pipeline._build_real_case(
        daily_pipeline.DailyPipelineConfig(run_id=str(tmp_path), domain="d01")
    )
    assert run_dir == tmp_path
    assert isinstance(case.namelist, OperationalNamelist)
    return case


def test_daily_pipeline_binds_explicit_scalar_advection_options(
    monkeypatch, tmp_path
) -> None:
    case = _build_daily_case(
        monkeypatch,
        tmp_path,
        " moist_adv_opt = 1,\n scalar_adv_opt = 1,\n",
    )

    assert (case.namelist.moist_adv_opt, case.namelist.scalar_adv_opt) == (1, 1)
    assert case.metadata["namelist"]["moist_adv_opt"] == 1
    assert case.metadata["namelist"]["scalar_adv_opt"] == 1


def test_daily_pipeline_keeps_wrf_defaults_when_scalar_options_are_omitted(
    monkeypatch, tmp_path
) -> None:
    case = _build_daily_case(
        monkeypatch,
        tmp_path,
        " h_sca_adv_order = 5,\n",
    )

    assert (case.namelist.moist_adv_opt, case.namelist.scalar_adv_opt) == (0, 0)
    assert case.metadata["namelist"]["moist_adv_opt"] == 0
    assert case.metadata["namelist"]["scalar_adv_opt"] == 0
