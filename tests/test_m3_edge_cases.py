"""Adversarial edge-case tests for the M3 GPU state/grid/halo skeleton.

These checks are owned by the sprint tester (Claude Opus 4.7) and exercise the
allocation discipline, contract invariants, and proof-object schema that the
worker-owned tests sketch only at the happy-path level.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from gpuwrf.contracts.grid import (
    BCMetadata,
    GridSpec,
    Projection,
    TerrainProvenance,
    VerticalCoord,
)
from gpuwrf.contracts.halo import HaloSpec, apply_halo
from gpuwrf.contracts.precision import DEFAULT_DTYPES, DTypeRegistry
from gpuwrf.contracts.state import State, Tendencies
from gpuwrf.profiling.budget import kernel_launches_per_step
from gpuwrf.timestep.dummy_loop import dummy_step, run_dummy_loop


ROOT = Path(__file__).resolve().parents[1]
_HOT_PATH_FILES = (
    ROOT / "src" / "gpuwrf" / "timestep" / "dummy_loop.py",
)
_ALLOCATOR_TOKENS = (
    "jnp.array(",
    "jnp.asarray(",
    "jnp.zeros(",
    "jnp.empty(",
    "jnp.ones(",
    "jnp.full(",
    "jnp.linspace(",
    "jnp.arange(",
    "jax.device_put(",
)


# ---------- Allocation discipline -------------------------------------------------


def test_hot_path_has_no_allocator_tokens():
    """Static guard: every allocator call would be a regression of AC §4.3 / §8.3."""
    offenders = []
    for path in _HOT_PATH_FILES:
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.split("#", 1)[0]
            for tok in _ALLOCATOR_TOKENS:
                if tok in stripped:
                    offenders.append(f"{path}:{lineno}: {tok}")
    assert not offenders, f"hot path contains forbidden allocators: {offenders}"


def test_init_path_allocations_are_bounded_and_listed():
    """The audited set of init-time allocators must not silently expand."""
    expected = {
        (ROOT / "src" / "gpuwrf" / "contracts" / "grid.py").as_posix(): {"jnp.linspace(", "jnp.zeros("},
        (ROOT / "src" / "gpuwrf" / "contracts" / "state.py").as_posix(): {"jnp.zeros(", "jax.device_put("},
    }
    seen: dict[str, set[str]] = {}
    for path in [
        ROOT / "src" / "gpuwrf" / "contracts" / "grid.py",
        ROOT / "src" / "gpuwrf" / "contracts" / "state.py",
        ROOT / "src" / "gpuwrf" / "contracts" / "halo.py",
        ROOT / "src" / "gpuwrf" / "contracts" / "precision.py",
    ]:
        toks: set[str] = set()
        for line in path.read_text().splitlines():
            stripped = line.split("#", 1)[0]
            for tok in _ALLOCATOR_TOKENS:
                if tok in stripped:
                    toks.add(tok)
        if toks:
            seen[path.as_posix()] = toks
    assert seen == expected, (
        "new init-time allocator surfaced without contract update. "
        f"expected={expected!r} got={seen!r}"
    )


# ---------- GridSpec invariants --------------------------------------------------


def _base_components():
    """Builds a Canary-shaped GridSpec and returns its plain components for mutation."""
    grid = GridSpec.canary_3km_template()
    return grid.projection, grid.terrain, grid.vertical, grid.bc, grid.eta_levels, grid.terrain_height


def test_gridspec_rejects_invalid_projection_kind():
    proj, terrain, vertical, bc, eta, th = _base_components()
    bad = replace(proj, kind="oblique-mercator")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="projection"):
        GridSpec(bad, terrain, vertical, bc, eta, th)


def test_gridspec_rejects_non_hybrid_eta_vertical():
    proj, terrain, vertical, bc, eta, th = _base_components()
    bad = replace(vertical, kind="sigma")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vertical"):
        GridSpec(proj, terrain, bad, bc, eta, th)


def test_gridspec_rejects_halo_width_below_one():
    proj, terrain, vertical, bc, eta, th = _base_components()
    with pytest.raises(ValueError, match="halo_width"):
        GridSpec(proj, terrain, vertical, bc, eta, th, halo_width=0)


def test_gridspec_rejects_halo_width_above_four():
    proj, terrain, vertical, bc, eta, th = _base_components()
    with pytest.raises(ValueError, match="halo_width"):
        GridSpec(proj, terrain, vertical, bc, eta, th, halo_width=5)


def test_gridspec_rejects_terrain_provenance_shape_mismatch():
    proj, terrain, vertical, bc, eta, th = _base_components()
    skewed = replace(terrain, shape=(proj.ny + 1, proj.nx))
    with pytest.raises(ValueError, match="terrain"):
        GridSpec(proj, skewed, vertical, bc, eta, th)


def test_gridspec_rejects_terrain_height_shape_mismatch():
    proj, terrain, vertical, bc, eta, th = _base_components()
    skewed = jnp.zeros((proj.ny + 1, proj.nx), dtype=jnp.float64)
    with pytest.raises(ValueError, match="terrain_height"):
        GridSpec(proj, terrain, vertical, bc, eta, skewed)


def test_gridspec_rejects_eta_levels_wrong_length():
    proj, terrain, vertical, bc, _eta, th = _base_components()
    bad = jnp.zeros((vertical.nz,), dtype=jnp.float64)
    with pytest.raises(ValueError, match="eta_levels"):
        GridSpec(proj, terrain, vertical, bc, bad, th)


def test_gridspec_rejects_fp32_arrays():
    proj, terrain, vertical, bc, eta, th = _base_components()
    bad_eta = eta.astype(jnp.float32)
    with pytest.raises(TypeError, match="fp64"):
        GridSpec(proj, terrain, vertical, bc, bad_eta, th)


def test_gridspec_rejects_non_c_grid_staggering():
    proj, terrain, vertical, bc, eta, th = _base_components()
    with pytest.raises(ValueError, match="C-grid"):
        GridSpec(proj, terrain, vertical, bc, eta, th, staggering="b-grid")  # type: ignore[arg-type]


def test_gridspec_template_eta_is_monotone_descending():
    grid = GridSpec.canary_3km_template()
    diffs = jnp.diff(grid.eta_levels)
    assert bool(jnp.all(diffs <= 0.0))
    assert float(grid.eta_levels[0]) == pytest.approx(1.0)
    assert float(grid.eta_levels[-1]) == pytest.approx(0.0)


def test_gridspec_canary_template_has_expected_dimensions():
    grid = GridSpec.canary_3km_template()
    assert grid.nx == 8
    assert grid.ny == 8
    assert grid.nz == 10
    assert grid.vertical.top_pressure_pa == 5000.0
    assert grid.bc.source == "AIFS"
    assert grid.bc.restart_compatible is True


def test_gridspec_hash_is_stable_across_independent_rebuilds():
    a = GridSpec.canary_3km_template()
    b = GridSpec.canary_3km_template()
    assert hash(a) == hash(b)


# ---------- State / Tendencies invariants ---------------------------------------


def test_state_replace_is_immutable_and_returns_new_object():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    original_theta = state.theta
    bumped = state.replace(theta=state.theta + 1.0)
    assert bumped is not state
    assert state.theta is original_theta
    assert float(state.theta.sum()) == 0.0
    assert float(bumped.theta.sum()) == float(grid.nz * grid.ny * grid.nx)


def test_tendencies_replace_preserves_other_fields():
    grid = GridSpec.canary_3km_template()
    tendencies = Tendencies.zeros(grid)
    new = tendencies.replace(theta=tendencies.theta + 0.5)
    assert new.theta is not tendencies.theta
    for name in ("u", "v", "w", "qv", "p", "ph", "mu"):
        assert getattr(new, name) is getattr(tendencies, name)


def test_state_and_tendencies_pytree_round_trip():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)
    leaves_s, def_s = jax.tree_util.tree_flatten(state)
    leaves_t, def_t = jax.tree_util.tree_flatten(tendencies)
    assert len(leaves_s) == 8
    assert len(leaves_t) == 8
    rebuilt_s = jax.tree_util.tree_unflatten(def_s, leaves_s)
    rebuilt_t = jax.tree_util.tree_unflatten(def_t, leaves_t)
    assert isinstance(rebuilt_s, State)
    assert isinstance(rebuilt_t, Tendencies)
    for name in ("u", "v", "w", "theta", "qv", "p", "ph", "mu"):
        assert getattr(rebuilt_s, name).shape == getattr(state, name).shape
        assert getattr(rebuilt_t, name).shape == getattr(tendencies, name).shape


def test_state_from_init_is_path_independent_for_m3():
    """M3 has no IC loader; the call shape must accept a Path placeholder."""
    grid = GridSpec.canary_3km_template()
    a = State.from_init(grid, Path("/dev/null"))
    b = State.zeros(grid)
    leaves_a = jax.tree_util.tree_leaves(a)
    leaves_b = jax.tree_util.tree_leaves(b)
    assert len(leaves_a) == len(leaves_b)
    for x, y in zip(leaves_a, leaves_b, strict=True):
        assert x.shape == y.shape
        assert x.dtype == y.dtype


def test_state_and_tendencies_share_shape_per_field():
    """SoA pytrees must agree per-field so the scan carry is shape-compatible."""
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)
    for name in ("u", "v", "w", "theta", "qv", "p", "ph", "mu"):
        assert getattr(state, name).shape == getattr(tendencies, name).shape


# ---------- Precision registry --------------------------------------------------


def test_precision_registry_blocks_typo_field():
    with pytest.raises(KeyError, match="not-a-field"):
        DEFAULT_DTYPES.dtype_for("not-a-field")


def test_precision_registry_returns_fp64_for_every_state_field():
    for field in ("u", "v", "w", "theta", "qv", "p", "ph", "mu"):
        assert DEFAULT_DTYPES.dtype_for(field) == jnp.float64


def test_precision_registry_factory_is_pure():
    a = DTypeRegistry.fp64_defaults()
    b = DTypeRegistry.fp64_defaults()
    assert a == b


# ---------- HaloSpec invariants -------------------------------------------------


def test_halospec_rejects_width_zero():
    with pytest.raises(ValueError, match="halo width"):
        HaloSpec(width=0, fields_to_exchange=("u",), edge_type="open")


def test_halospec_rejects_width_above_four():
    with pytest.raises(ValueError, match="halo width"):
        HaloSpec(width=5, fields_to_exchange=("u",), edge_type="open")


@pytest.mark.parametrize("edge", ["periodic", "open", "nest_boundary"])
def test_halospec_accepts_all_documented_edge_types(edge):
    spec = HaloSpec(width=2, fields_to_exchange=("theta",), edge_type=edge)
    assert spec.edge_type == edge


def test_apply_halo_is_identity_for_empty_fields():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    spec = HaloSpec(width=grid.halo_width, fields_to_exchange=(), edge_type="periodic")
    assert apply_halo(state, spec) is state


# ---------- Dummy timestep loop -------------------------------------------------


def test_dummy_step_with_zero_tendencies_is_bitwise_identity():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid).replace(theta=jnp.full((grid.nz, grid.ny, grid.nx), 7.5))
    tendencies = Tendencies.zeros(grid)
    new_state, new_tendencies = dummy_step(state, tendencies, dt=3.0)
    assert bool(jnp.all(new_state.theta == state.theta))
    assert new_tendencies is tendencies


def test_run_dummy_loop_is_deterministic():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)
    a, _ = run_dummy_loop(state, tendencies, 3.0, 1000)
    b, _ = run_dummy_loop(state, tendencies, 3.0, 1000)
    for name in ("u", "v", "w", "theta", "qv", "p", "ph", "mu"):
        assert bool(jnp.all(getattr(a, name) == getattr(b, name)))


def test_run_dummy_loop_zero_steps_is_state_passthrough():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid).replace(theta=jnp.full((grid.nz, grid.ny, grid.nx), 2.0))
    tendencies = Tendencies.zeros(grid)
    out_state, out_tendencies = run_dummy_loop(state, tendencies, 3.0, 0)
    assert bool(jnp.all(out_state.theta == state.theta))
    assert all(
        bool(jnp.all(getattr(out_tendencies, n) == getattr(tendencies, n)))
        for n in ("u", "v", "w", "theta", "qv", "p", "ph", "mu")
    )


def test_run_dummy_loop_keeps_outputs_on_gpu():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)
    out_state, out_tendencies = run_dummy_loop(state, tendencies, 3.0, 100)
    out_state.theta.block_until_ready()
    for leaf in jax.tree_util.tree_leaves((out_state, out_tendencies)):
        assert leaf.devices().copy().pop().platform == "gpu"


def test_run_dummy_loop_signature_uses_static_n_steps():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)
    a = run_dummy_loop.lower(state, tendencies, 3.0, 10).compile()
    b = run_dummy_loop.lower(state, tendencies, 3.0, 1000).compile()
    text_a = a.as_text() if hasattr(a, "as_text") else str(a)
    text_b = b.as_text() if hasattr(b, "as_text") else str(b)
    assert text_a != text_b, "n_steps must be static; different trip counts must produce distinct HLO"


def test_run_dummy_loop_hlo_has_exactly_one_while_loop():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)
    compiled = run_dummy_loop.lower(state, tendencies, 3.0, 1000).compile()
    text = compiled.as_text() if hasattr(compiled, "as_text") else str(compiled)
    while_calls = re.findall(r"\bwhile\(", text)
    assert len(while_calls) == 1, f"expected single scan->while; got {len(while_calls)}"


# ---------- Spacetime budget + transfer audit schema ---------------------------


def _budget():
    return json.loads((ROOT / "artifacts" / "m3" / "spacetime_budget.json").read_text())


def _audit():
    return json.loads((ROOT / "artifacts" / "m3" / "transfer_audit.json").read_text())


def test_spacetime_budget_totals_are_self_consistent():
    b = _budget()
    assert b["total_persistent_bytes"] == b["state_bytes"] + b["tendency_bytes"] + b["halo_buffer_bytes"]


def test_spacetime_budget_state_bytes_matches_live_state():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    tendencies = Tendencies.zeros(grid)
    b = _budget()
    assert b["state_bytes"] == state.bytes()
    assert b["tendency_bytes"] == tendencies.bytes()


def test_spacetime_budget_obeys_soft_perf_bound():
    b = _budget()
    assert b["wall_time_per_step_us"] < 100.0, "AC §4.4 soft sanity bound violated"


def test_spacetime_budget_persistent_bytes_under_one_gb():
    b = _budget()
    assert b["total_persistent_bytes"] < 1 * 1024 * 1024 * 1024


def test_transfer_audit_records_jax_version_and_method():
    a = _audit()
    assert isinstance(a.get("method"), str) and a["method"]
    assert a.get("jax_version") == jax.__version__


def test_transfer_audit_records_gpu_device_name():
    a = _audit()
    assert isinstance(a.get("gpu_name"), str) and "cuda" in a["gpu_name"].lower()


def test_hlo_dump_artifact_exists_and_is_nonempty():
    p = ROOT / "artifacts" / "m3" / "hlo_dump" / "dummy_loop.txt"
    assert p.exists()
    size = p.stat().st_size
    assert 100 < size < 100_000, f"hlo dump size {size} outside [100B, 100KB]"


def test_hlo_dump_contains_jit_run_dummy_loop_module():
    p = ROOT / "artifacts" / "m3" / "hlo_dump" / "dummy_loop.txt"
    text = p.read_text()
    assert "HloModule jit_run_dummy_loop" in text
    assert "while(" in text


def test_kernel_launches_helper_reports_raw_hlo_count():
    assert kernel_launches_per_step("") == 1
    assert kernel_launches_per_step("while(") == 1
    assert kernel_launches_per_step("fusion(\n" * 20) == 20


def test_agent_success_records_jax_pin():
    payload = json.loads((ROOT / "artifacts" / "m3" / "agent_success.json").read_text())
    assert payload.get("toolchain", {}).get("jax_version") == "0.10.0"


# ---------- ADR-002 cross-reference --------------------------------------------


def test_adr002_contains_required_tokens_and_size():
    path = ROOT / ".agent" / "decisions" / "ADR-002-state-layout.md"
    assert path.exists()
    text = path.read_text()
    assert path.stat().st_size >= 1500
    for token in ("Decision:", "Layout:", "Staggering:", "Halo packing:"):
        assert token in text, f"ADR-002 missing required token {token!r}"
