"""Frozen ALISIOS nested SW terrain-control binding, with input mutations."""

from __future__ import annotations

import ast
import hashlib
import inspect
import os
import re
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np
import pytest

from gpuwrf.contracts.grid import (
    BCMetadata,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.coupling.physics_couplers import (
    _compute_solar_geometry,
    _rrtmg_topography_state,
    build_radiation_static_from_wrf_fields,
)
from gpuwrf.integration import nested_pipeline
from gpuwrf.integration.nested_pipeline import (
    NestedPipelineConfig,
    _domain_physics_int,
    _load_domains,
    _make_namelist,
)
from gpuwrf.io.gen2_accessor import Gen2Run
from gpuwrf.io.radiation_static import load_radiation_static
from gpuwrf.runtime.operational_state import OperationalCarry

jax.config.update("jax_enable_x64", True)


CASE = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")
NAMELIST_SHA256 = "5ef3a96b"  # frozen full hash checked against SHA256SUMS below


def _tiny_grid() -> GridSpec:
    eta = jnp.linspace(1.0, 0.0, 4, dtype=jnp.float64)
    return GridSpec(
        Projection("lambert", 28.3, -15.6, 3000.0, 3000.0, 2, 2),
        TerrainProvenance("analytic://s2", "s2", (2, 2), "m", "flat", 0.0, True),
        VerticalCoord("hybrid_eta", 3, 5000.0, eta),
        BCMetadata("ideal", (), 1, "linear", True),
        eta,
        jnp.zeros((2, 2), dtype=jnp.float64),
    )


def _frozen_case() -> Path:
    if not CASE.is_dir():
        pytest.skip("frozen ALISIOS S0 case unavailable")
    expected = dict(
        line.split(maxsplit=1)[::-1]
        for line in (CASE / "SHA256SUMS").read_text().splitlines()
    )
    actual = hashlib.sha256((CASE / "namelist.input").read_bytes()).hexdigest()
    assert actual == expected["namelist.input"]
    assert actual.startswith(NAMELIST_SHA256)
    return CASE


def _bound_values(run: Gen2Run, domain: str) -> tuple[int, int]:
    """Exercise the existing per-domain parser and the operational namelist seam."""
    grid = _tiny_grid()
    nml = _make_namelist(
        grid=grid,
        tendencies=object(),
        metrics=grid.metrics,
        dt_s=54.0 if domain == "d01" else 18.0,
        parent_dt_s=None,
        run_start=datetime(2026, 7, 26, tzinfo=timezone.utc),
        radiation_static=None,
        cu_physics=0,
        topo_shading=_domain_physics_int(run, "topo_shading", domain, 0),
        slope_rad=_domain_physics_int(run, "slope_rad", domain, 0),
    )
    return int(nml.topo_shading), int(nml.slope_rad)


def test_real_loader_call_site_reads_both_physics_controls() -> None:
    """Guard the actual call site that the small input-mutation test cannot mock."""
    tree = ast.parse(inspect.getsource(nested_pipeline._load_domains))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_make_namelist"
    ]
    assert len(calls) == 1
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    assert kwargs["topo_shading"] == "_domain_physics_int(run, 'topo_shading', name, 0)"
    assert kwargs["slope_rad"] == "slope_rad"
    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "slope_rad" for target in node.targets)
        and ast.unparse(node.value) == "_domain_physics_int(run, 'slope_rad', name, 0)"
        for node in ast.walk(tree)
    )


def test_frozen_case_and_each_input_mutation_bind_per_domain(tmp_path: Path) -> None:
    source = (_frozen_case() / "namelist.input").read_text()
    assert [_bound_values(Gen2Run(CASE), domain) for domain in ("d01", "d02")] == [
        (1, 1),
        (1, 1),
    ]
    for control in ("topo_shading", "slope_rad"):
        for domain, replacement in (("d01", "0, 1"), ("d02", "1, 0")):
            changed, count = re.subn(
                rf"(?m)^(\s*{control}\s*=\s*)1,\s*1,",
                rf"\g<1>{replacement},",
                source,
            )
            assert count == 1
            (tmp_path / "namelist.input").write_text(changed)
            run = Gen2Run(tmp_path)
            values = {name: _bound_values(run, name) for name in ("d01", "d02")}
            assert values[domain][0 if control == "topo_shading" else 1] == 0
            assert values["d02" if domain == "d01" else "d01"] == (1, 1)
            assert values[domain][1 if control == "topo_shading" else 0] == 1


def _load_with_inert_state(
    source: Path, tmp_path: Path, monkeypatch, static_loader, reached_states=None
):
    """Keep the product loader and namelist seam; substitute only heavy initializers."""
    if reached_states is None:
        reached_states = []

    class FakeState:
        p_total = ph_total = mu_total = object()
        t_skin = jnp.zeros((2, 2), dtype=jnp.float32)

        def __init__(self, domain):
            self.domain = domain

        def replace(self, **_kwargs):
            reached_states.append(self.domain)
            return self

    class FakeRun:
        def __init__(self, source: Path):
            self.namelist = Gen2Run(source).namelist

        def grid(self, domain: str):
            return SimpleNamespace(
                parent_id=1,
                parent_grid_ratio=1 if domain == "d01" else 3,
                i_parent_start=1,
                j_parent_start=1,
            )

    grid = _tiny_grid()
    monkeypatch.setattr(nested_pipeline, "load_radiation_static", static_loader)
    monkeypatch.setattr(nested_pipeline, "_domain_gwd_opt", lambda *_a: 0)
    monkeypatch.setattr(nested_pipeline, "_domain_sf_surface_physics", lambda *_a: 0)
    def inert_carry(state, _namelist):
        # Keep the real carry interface for post-seed storage adapters while
        # substituting only the heavy initialization this loader gate excludes.
        return OperationalCarry(**{
            field.name: state if field.name == "state" else None
            for field in fields(OperationalCarry)
        })

    monkeypatch.setattr(nested_pipeline, "_initial_carry_for_run", inert_carry)
    monkeypatch.setattr(nested_pipeline, "_commit_to_operational_device", lambda carry: carry)
    monkeypatch.setattr(nested_pipeline, "_root_boundary_cadence_override", lambda nml, *_a: nml)
    # Force-down seeding is outside this radiation-static loader gate.
    monkeypatch.setattr(nested_pipeline, "initialize_child_scalar_boundaries", lambda state, *_a, **_k: state)

    run = FakeRun(source)

    def fake_case(_source, *, domain, **_kwargs):
        return SimpleNamespace(
            run=run,
            grid=grid,
            metrics=grid.metrics,
            tendencies=object(),
            state=FakeState(domain),
            metadata={"run_start_label": "2026-07-26_00:00:00"},
        )
    monkeypatch.setattr(nested_pipeline, "build_replay_case", fake_case)
    config = NestedPipelineConfig(
        input_dir=source,
        output_dir=tmp_path / "out",
        proof_dir=tmp_path / "proof",
        hours=1,
        max_dom=2,
    )
    return _load_domains(config, ("d01", "d02")), reached_states


def test_real_nested_loader_retains_frozen_controls_and_mutations(
    tmp_path: Path, monkeypatch
) -> None:
    """Exercise the real loader/namelist seam with only heavy initializers inert."""
    static = build_radiation_static_from_wrf_fields(
        np.full((2, 2), 28.3),
        np.full((2, 2), -16.5),
        np.array([[0.0, 800.0], [0.0, 800.0]]),
        dx_m=3000.0,
        dy_m=3000.0,
    )

    def load(source: Path) -> dict[str, tuple[int, int]]:
        result, _ = _load_with_inert_state(
            source, tmp_path, monkeypatch, lambda *_a, **_k: (static, {})
        )
        _, bundles, meta, _, _, _ = result
        assert all(bundles[name].namelist.radiation_static is static for name in bundles)
        controls = {
            name: (int(bundle.namelist.topo_shading), int(bundle.namelist.slope_rad))
            for name, bundle in bundles.items()
        }
        recorded = {name: (meta["domains"][name]["namelist"]["topo_shading"],
                           meta["domains"][name]["namelist"]["slope_rad"]) for name in bundles}
        assert recorded == controls  # run-proof metadata carries the effective per-domain controls
        return controls

    source = (_frozen_case() / "namelist.input").read_text()
    assert load(CASE) == {"d01": (1, 1), "d02": (1, 1)}
    for control in ("topo_shading", "slope_rad"):
        for domain, replacement in (("d01", "0, 1"), ("d02", "1, 0")):
            changed, count = re.subn(
                rf"(?m)^(\s*{control}\s*=\s*)1,\s*1,",
                rf"\g<1>{replacement},",
                source,
            )
            assert count == 1
            (tmp_path / "namelist.input").write_text(changed)
            expected = {"d01": (1, 1), "d02": (1, 1)}
            expected[domain] = (0, 1) if control == "topo_shading" else (1, 0)
            assert load(tmp_path) == expected


def test_frozen_radiation_static_inputs_exist_for_both_domains() -> None:
    run = Gen2Run(_frozen_case())
    for domain in ("d01", "d02"):
        terrain = jnp.asarray(run.load_wrfinput(domain, "HGT", lazy=False))
        spec = run.grid(domain)
        grid = SimpleNamespace(
            terrain_height=terrain,
            terrain=SimpleNamespace(source_path=str(run.wrfinput_file(domain))),
            projection=SimpleNamespace(dx_m=spec.dx_m, dy_m=spec.dy_m),
            ny=terrain.shape[0],
            nx=terrain.shape[1],
        )
        static, metadata = load_radiation_static(run, domain, grid=grid)
        assert static is not None
        assert metadata["domain"] == domain
        assert metadata["uses_real_xlat_xlong"] is True
        assert np.max(np.asarray(static.slope_rad)) > 0


@pytest.mark.parametrize("failed_domain", ["d01", "d02"])
@pytest.mark.parametrize("failure", ["raise", "none", "empty_static"])
def test_requested_slope_radiation_fails_closed_before_state_init(
    tmp_path: Path, monkeypatch, failed_domain: str, failure: str
) -> None:
    reached_states = []
    attempted = []

    def static_loader(_run, domain, **_kwargs):
        attempted.append(domain)
        if domain == failed_domain:
            if failure == "raise":
                raise ValueError("missing terrain")
            if failure == "none":
                return None
            return None, {}
        return object(), {}

    with pytest.raises(
        RuntimeError, match=rf"^{failed_domain}: slope_rad=1 requires radiation static"
    ) as err:
        _load_with_inert_state(
            _frozen_case(), tmp_path, monkeypatch, static_loader, reached_states
        )
    assert failed_domain not in reached_states
    assert attempted == (["d01"] if failed_domain == "d01" else ["d01", "d02"])
    if failure == "raise":
        assert isinstance(err.value.__cause__, ValueError)
        assert str(err.value.__cause__) == "missing terrain"
    elif failure == "none":
        assert isinstance(err.value.__cause__, TypeError)
    else:
        assert err.value.__cause__ is None


@pytest.mark.parametrize("controls", [(0, 0), (1, 0)])
@pytest.mark.parametrize("failure", ["raise", "none"])
def test_slope_off_static_load_remains_best_effort(
    tmp_path: Path, monkeypatch, controls: tuple[int, int], failure: str
) -> None:
    source = (_frozen_case() / "namelist.input").read_text()
    for control, value in zip(("topo_shading", "slope_rad"), controls):
        source, count = re.subn(
            rf"(?m)^(\s*{control}\s*=\s*)1,\s*1,",
            rf"\g<1>{value}, {value},",
            source,
        )
        assert count == 1
    (tmp_path / "namelist.input").write_text(source)

    def static_loader(*_args, **_kwargs):
        if failure == "raise":
            raise ValueError("optional terrain absent")
        return None

    result, reached_states = _load_with_inert_state(
        tmp_path, tmp_path, monkeypatch, static_loader
    )
    _, bundles, meta, _, _, _ = result
    assert reached_states == ["d01", "d02"]
    for domain in ("d01", "d02"):
        nml = bundles[domain].namelist
        assert (int(nml.topo_shading), int(nml.slope_rad)) == controls
        assert nml.radiation_static is None
        assert meta["domains"][domain]["namelist"]["radiation_static_loaded"] is False
        assert (meta["domains"][domain]["namelist"]["topo_shading"],
                meta["domains"][domain]["namelist"]["slope_rad"]) == controls


def test_explicit_zero_zero_defaults_remain_valid() -> None:
    grid = _tiny_grid()
    nml = _make_namelist(
        grid=grid,
        tendencies=object(),
        metrics=grid.metrics,
        dt_s=18.0,
        parent_dt_s=None,
        run_start=datetime(2026, 7, 26, tzinfo=timezone.utc),
        radiation_static=None,
        cu_physics=0,
    )
    assert (int(nml.topo_shading), int(nml.slope_rad)) == (0, 0)


def test_daytime_ridge_exercises_slope_aspect_and_shadow_with_night_control() -> None:
    terrain = np.zeros((7, 7), dtype=np.float64)
    terrain[3, 3] = 3000.0
    lat = np.full_like(terrain, 28.3)
    lon = np.full_like(terrain, -16.4)
    static = build_radiation_static_from_wrf_fields(
        lat, lon, terrain, dx_m=1000.0, dy_m=1000.0
    )
    grid = SimpleNamespace(projection=SimpleNamespace(dx_m=1000.0, dy_m=1000.0))
    day = _compute_solar_geometry(
        lat, lon, datetime(2026, 7, 26, 7, tzinfo=timezone.utc)
    )
    night = _compute_solar_geometry(
        lat, lon, datetime(2026, 7, 26, 0, tzinfo=timezone.utc)
    )
    active = _rrtmg_topography_state(static, grid, day, topo_shading=1, slope_rad=1)
    unshaded = _rrtmg_topography_state(static, grid, day, topo_shading=0, slope_rad=1)
    nocturnal = _rrtmg_topography_state(static, grid, night, topo_shading=1, slope_rad=1)
    assert active is not None and unshaded is not None and nocturnal is not None
    assert np.max(np.asarray(active.slope_rad)) > 0.0
    assert np.ptp(np.asarray(active.slope_azimuth_rad)) > 0.0
    assert np.sum(np.asarray(active.shadow_mask)) > 0
    assert np.sum(np.asarray(unshaded.shadow_mask)) == 0
    assert np.sum(np.asarray(nocturnal.shadow_mask)) == 0
    assert _rrtmg_topography_state(static, grid, day, topo_shading=1, slope_rad=0) is None
