"""CPU oracle for the default-off B2 event-aware K=1 cascade scheduler."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.integration.nested_pipeline import (
    _nested_event_aware_fusion_k_from_env,
)
from gpuwrf.runtime.domain_tree import DomainTreeResult, run_domain_tree_callbacks


@dataclass(frozen=True)
class _Carry:
    value: int


def _hierarchy(*, leaves: int = 1) -> DomainHierarchy:
    names = ("d01", "d02") + tuple(f"d{idx:02d}" for idx in range(3, 3 + leaves))
    edges = [DomainNest("d01", "d02", 3, 30, 20)]
    edges.extend(DomainNest("d02", name, 3, 10, 10) for name in names[2:])
    return DomainHierarchy.from_edges(names, tuple(edges), max_dom=len(names))


def _fused_lookup(hierarchy: DomainHierarchy, calls: list[tuple[int, tuple[int, ...]]]):
    child_specs = hierarchy.children("d02")

    def lookup(parent_name: str):
        if parent_name != "d02":
            return None

        def fused(parent_carry, child_carries, parent_start, child_starts):
            calls.append((int(parent_start), tuple(int(step) for step in child_starts)))
            parent_new = _Carry(parent_carry.value + 1)
            children_new = []
            for spec, child_carry in zip(child_specs, child_carries, strict=True):
                forced = _Carry(child_carry.value + parent_new.value * 1000)
                children_new.append(
                    _Carry(forced.value + int(spec.parent_grid_ratio))
                )
            return parent_new, tuple(children_new)

        return fused

    return lookup


def _run(
    *,
    leaves: int = 1,
    root_steps: int = 2,
    fusion_k: int = 0,
    use_fused_lookup: bool = False,
    leaf_cadences: tuple[int, ...] | None = None,
    leaf_alarms: dict[str, tuple[int, ...]] | None = None,
    carries: dict[str, _Carry] | None = None,
    initial_own_steps: dict[str, int] | None = None,
    with_output: bool = True,
    block_between: bool = False,
    root_sync_cadence: int | None = None,
) -> tuple[DomainTreeResult, list[tuple[int, tuple[int, ...]]]]:
    hierarchy = _hierarchy(leaves=leaves)
    calls: list[tuple[int, tuple[int, ...]]] = []

    def advance(_name, carry, _start_step, n_steps):
        return _Carry(carry.value + int(n_steps))

    def force(_edge, parent, child):
        return _Carry(child.value + parent.value * 1000)

    if carries is None:
        carries = {name: _Carry(0) for name in hierarchy.order}
    if leaf_cadences is None:
        leaf_cadences = tuple(5 for _ in hierarchy.order[2:])
    cadence = {"d01": 1, "d02": 3}
    cadence.update(
        {
            name: int(value)
            for name, value in zip(
                hierarchy.order[2:], leaf_cadences, strict=True
            )
        }
    )
    result = run_domain_tree_callbacks(
        hierarchy,
        carries,
        root_steps=int(root_steps),
        advance=advance,
        force=force,
        output=(
            (lambda name, step, state: (name, step, state.value))
            if with_output
            else None
        ),
        output_cadence_steps=cadence,
        output_alarm_steps=leaf_alarms,
        block_between=block_between,
        root_sync_cadence=root_sync_cadence,
        fused_cascade=(
            _fused_lookup(hierarchy, calls) if use_fused_lookup else None
        ),
        initial_own_steps=initial_own_steps,
        event_aware_fusion_k=fusion_k,
    )
    return result, calls


def _semantic_projection(result: DomainTreeResult):
    return (
        result.events,
        result.own_steps,
        {name: carry.value for name, carry in result.carries.items()},
        {name: state.value for name, state in result.states.items()},
        result.outputs,
    )


def test_k1_nondivisible_history_is_exact_and_only_event_groups_fall_back():
    eager, _ = _run()
    candidate, calls = _run(fusion_k=1, use_fused_lookup=True)

    assert _semantic_projection(candidate) == _semantic_projection(eager)
    assert len(calls) == 4
    assert candidate.cascade_counts == {
        "fused:d02": 4,
        "event_fallback:d02": 2,
    }
    assert [step for name, step, _value in candidate.outputs if name == "d03"] == [
        5,
        10,
        15,
    ]


def test_k1_explicit_absolute_alarms_preserve_exact_snapshots():
    alarms = {"d03": (4, 9, 14)}
    eager, _ = _run(leaf_alarms=alarms)
    candidate, calls = _run(
        fusion_k=1,
        use_fused_lookup=True,
        leaf_alarms=alarms,
    )

    assert _semantic_projection(candidate) == _semantic_projection(eager)
    assert len(calls) == 4
    assert candidate.cascade_counts == {
        "fused:d02": 4,
        "event_fallback:d02": 2,
    }
    assert [step for name, step, _value in candidate.outputs if name == "d03"] == [
        4,
        9,
        14,
    ]


def test_k1_mixed_leaf_alarms_fall_back_atomically_and_remain_exact():
    eager, _ = _run(leaves=2, leaf_cadences=(5, 7))
    candidate, calls = _run(
        leaves=2,
        leaf_cadences=(5, 7),
        fusion_k=1,
        use_fused_lookup=True,
    )

    assert _semantic_projection(candidate) == _semantic_projection(eager)
    assert len(calls) == candidate.cascade_counts["fused:d02"]
    assert candidate.cascade_counts["fused:d02"] > 0
    assert candidate.cascade_counts["event_fallback:d02"] > 0
    assert sum(candidate.cascade_counts.values()) == 6


def test_k1_real_all7_twenty_min_shape_preserves_every_leaf_snapshot():
    cadences = (200,) * 7
    eager, _ = _run(leaves=7, root_steps=23, leaf_cadences=cadences)
    candidate, calls = _run(
        leaves=7,
        root_steps=23,
        leaf_cadences=cadences,
        fusion_k=1,
        use_fused_lookup=True,
    )

    assert _semantic_projection(candidate) == _semantic_projection(eager)
    assert len(calls) == 68
    assert candidate.cascade_counts == {
        "fused:d02": 68,
        "event_fallback:d02": 1,
    }
    for name in candidate.own_steps:
        if name in {"d01", "d02"}:
            continue
        assert [
            step
            for domain, step, _value in candidate.outputs
            if domain == name
        ] == [200]


def test_k1_segment_resume_matches_single_call_and_eager_oracle():
    eager, _ = _run()
    full, _ = _run(fusion_k=1, use_fused_lookup=True)
    first, first_calls = _run(
        root_steps=1,
        fusion_k=1,
        use_fused_lookup=True,
    )
    second, second_calls = _run(
        root_steps=1,
        fusion_k=1,
        use_fused_lookup=True,
        carries=first.carries,
        initial_own_steps=first.own_steps,
    )

    assert _semantic_projection(full) == _semantic_projection(eager)
    assert first.events + second.events == full.events
    assert first.outputs + second.outputs == full.outputs
    assert second.own_steps == full.own_steps
    assert second.carries == full.carries
    assert len(first_calls) + len(second_calls) == full.cascade_counts["fused:d02"]
    assert {
        key: first.cascade_counts.get(key, 0) + second.cascade_counts.get(key, 0)
        for key in full.cascade_counts
    } == full.cascade_counts


@pytest.mark.parametrize("with_output", [False, True])
def test_k1_aligned_or_output_free_groups_all_use_existing_fused_program(with_output):
    eager, _ = _run(leaf_cadences=(9,), with_output=with_output)
    candidate, calls = _run(
        leaf_cadences=(9,),
        with_output=with_output,
        fusion_k=1,
        use_fused_lookup=True,
    )

    assert _semantic_projection(candidate) == _semantic_projection(eager)
    assert len(calls) == 6
    assert candidate.cascade_counts == {"fused:d02": 6}


def test_k1_per_advance_sync_mode_fails_closed_to_released_eager_path():
    eager, _ = _run(block_between=True)
    candidate, calls = _run(
        fusion_k=1,
        use_fused_lookup=True,
        block_between=True,
    )

    assert _semantic_projection(candidate) == _semantic_projection(eager)
    assert calls == []
    assert candidate.cascade_counts == {}


@pytest.mark.parametrize("value", [-1, 2, 0.5, True, "1", "invalid", None])
def test_callback_rejects_unsupported_k_values(value):
    if value is None:
        result, _ = _run()
        assert result.cascade_counts == {}
        return
    with pytest.raises(ValueError, match="event_aware_fusion_k must be 0 or 1"):
        _run(**{"fusion_k": value})


@pytest.mark.parametrize("value", ["", "2", "-1", "true", "k1"])
def test_pipeline_env_rejects_unknown_event_aware_k(monkeypatch, value):
    monkeypatch.setenv("GPUWRF_NESTED_EVENT_AWARE_FUSION_K", value)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _nested_event_aware_fusion_k_from_env()


def test_pipeline_env_is_default_off_and_accepts_only_zero_or_one(monkeypatch):
    monkeypatch.delenv("GPUWRF_NESTED_EVENT_AWARE_FUSION_K", raising=False)
    assert _nested_event_aware_fusion_k_from_env() == 0
    monkeypatch.setenv("GPUWRF_NESTED_EVENT_AWARE_FUSION_K", "0")
    assert _nested_event_aware_fusion_k_from_env() == 0
    monkeypatch.setenv("GPUWRF_NESTED_EVENT_AWARE_FUSION_K", "1")
    assert _nested_event_aware_fusion_k_from_env() == 1


def test_short_probe_does_not_freeze_placeholder_testbed_boundaries():
    from scripts.v0234_event_aware_fusion_probe import (
        SYNTHETIC_LEAF_ALARM_STEP,
        _build_all7_minigrid,
    )

    tree = _build_all7_minigrid()

    assert SYNTHETIC_LEAF_ALARM_STEP == 2
    assert tuple(tree.hierarchy.order) == tuple(
        f"d{index:02d}" for index in range(1, 10)
    )
    for name in tuple(tree.hierarchy.order)[1:]:
        boundary = tree.domains[name].namelist.boundary_config
        assert not bool(boundary.nested_frozen_wrf_boundary_bundle)
    for nest in tree.hierarchy.nests:
        assert int(nest.i_parent_start) >= 5
        assert int(nest.j_parent_start) >= 5
