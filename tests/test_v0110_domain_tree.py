from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.runtime.domain_tree import (
    DomainBundle,
    DomainTree,
    _FUSED_PROGRAM_CACHE,
    _fusable_parent,
    _operational_fused_cascade_factory,
    build_live_nested_boundary_config,
    run_domain_tree_callbacks,
)


def _canary_hierarchy() -> DomainHierarchy:
    return DomainHierarchy.from_edges(
        ("d01", "d02", "d03", "d04", "d05"),
        (
            DomainNest("d01", "d02", 3, 30, 20),
            DomainNest("d02", "d03", 3, 52, 20),
            DomainNest("d02", "d04", 3, 42, 32),
            DomainNest("d02", "d05", 3, 64, 34),
        ),
    )


@dataclass(frozen=True)
class _Carry:
    value: int


def test_domain_hierarchy_counts_canary_5_domains():
    hierarchy = _canary_hierarchy()
    assert hierarchy.roots() == ("d01",)
    assert hierarchy.parent("d03") == "d02"
    assert [edge.child for edge in hierarchy.children("d02")] == ["d03", "d04", "d05"]
    assert hierarchy.expected_step_counts(root_steps=2) == {
        "d01": 2,
        "d02": 6,
        "d03": 18,
        "d04": 18,
        "d05": 18,
    }


def test_domain_tree_callbacks_force_recurse_and_output_counts():
    hierarchy = DomainHierarchy.from_edges(
        ("d01", "d02", "d03"),
        (
            DomainNest("d01", "d02", 3, 30, 20),
            DomainNest("d02", "d03", 3, 52, 20),
        ),
    )
    force_seen: list[tuple[str, str, int, int]] = []

    def advance(name, carry, start_step, n_steps):
        return _Carry(carry.value + int(n_steps))

    def force(edge, parent, child):
        force_seen.append((edge.parent, edge.child, parent.value, child.value))
        return child

    result = run_domain_tree_callbacks(
        hierarchy,
        {"d01": _Carry(0), "d02": _Carry(0), "d03": _Carry(0)},
        root_steps=2,
        advance=advance,
        force=force,
        output=lambda name, step, state: (name, step, state.value),
        output_cadence_steps={"d01": 1, "d02": 3, "d03": 9},
        block_between=False,
    )

    assert result.own_steps == {"d01": 2, "d02": 6, "d03": 18}
    assert {name: carry.value for name, carry in result.carries.items()} == {
        "d01": 2,
        "d02": 6,
        "d03": 18,
    }
    assert force_seen[0][:2] == ("d01", "d02")
    assert force_seen[1][:2] == ("d02", "d03")
    assert result.outputs == (
        ("d03", 9, 9),
        ("d02", 3, 3),
        ("d01", 1, 1),
        ("d03", 18, 18),
        ("d02", 6, 6),
        ("d01", 2, 2),
    )


def test_feedback_callback_is_behind_gate():
    hierarchy = DomainHierarchy.from_edges(
        ("d01", "d02"),
        (DomainNest("d01", "d02", 3, 2, 2),),
    )
    calls = {"feedback": 0}

    def advance(name, carry, start_step, n_steps):
        return _Carry(carry.value + int(n_steps))

    def feedback(edge, parent, child):
        calls["feedback"] += 1
        return parent

    run_domain_tree_callbacks(
        hierarchy,
        {"d01": _Carry(0), "d02": _Carry(0)},
        root_steps=1,
        advance=advance,
        feedback=feedback,
        feedback_enabled=False,
        block_between=False,
    )
    assert calls["feedback"] == 0

    run_domain_tree_callbacks(
        hierarchy,
        {"d01": _Carry(0), "d02": _Carry(0)},
        root_steps=1,
        advance=advance,
        feedback=feedback,
        feedback_enabled=True,
        block_between=False,
    )
    assert calls["feedback"] == 1


def test_child_start_steps_are_domain_global_not_subcycle_local():
    hierarchy = DomainHierarchy.from_edges(
        ("d01", "d02"),
        (DomainNest("d01", "d02", 3, 2, 2),),
    )
    calls: list[tuple[str, int, int]] = []

    def advance(name, carry, start_step, n_steps):
        calls.append((name, int(start_step), int(n_steps)))
        return _Carry(carry.value + int(n_steps))

    run_domain_tree_callbacks(
        hierarchy,
        {"d01": _Carry(0), "d02": _Carry(0)},
        root_steps=2,
        advance=advance,
        block_between=False,
    )

    assert calls == [
        ("d01", 1, 1),
        ("d02", 1, 3),
        ("d01", 2, 1),
        ("d02", 4, 3),
    ]


def _run_canary_tree_recorded(**runner_kwargs):
    """Run the 5-domain canary tree with deterministic recording callbacks.

    Returns ``(events, own_steps, carry_values, outputs, advance_calls,
    force_calls)`` so two sync configurations can be compared for byte-identical
    WRF nesting cadence.
    """
    hierarchy = _canary_hierarchy()
    advance_calls: list[tuple[str, int, int]] = []
    force_calls: list[tuple[str, str, int]] = []

    def advance(name, carry, start_step, n_steps):
        advance_calls.append((name, int(start_step), int(n_steps)))
        return _Carry(carry.value + int(n_steps))

    def force(edge, parent, child):
        force_calls.append((edge.parent, edge.child, parent.value))
        return child

    result = run_domain_tree_callbacks(
        hierarchy,
        {name: _Carry(0) for name in hierarchy.order},
        root_steps=2,
        advance=advance,
        force=force,
        output=lambda name, step, state: (name, step, state.value),
        output_cadence_steps={"d01": 1, "d02": 3, "d03": 9, "d04": 9, "d05": 9},
        **runner_kwargs,
    )
    return (
        result.events,
        result.own_steps,
        {name: carry.value for name, carry in result.carries.items()},
        result.outputs,
        advance_calls,
        force_calls,
    )


def test_root_sync_cadence_orchestration_is_identical_to_legacy_block_between():
    """root_sync_cadence is a HOST-WAIT policy only: the recursion cadence,
    advance/force call sequence, step clocks, carries and outputs must be
    byte-identical to the legacy per-advance ``block_between`` path (the v0.17
    GPU-idle fix only moves where ``block_until_ready`` is called)."""
    legacy = _run_canary_tree_recorded(block_between=True)
    modes = {
        "block_between_false": dict(block_between=False),
        "root_sync_1": dict(block_between=False, root_sync_cadence=1),
        "root_sync_2": dict(block_between=False, root_sync_cadence=2),
        "root_sync_8": dict(block_between=False, root_sync_cadence=8),
        # root_sync_cadence overrides block_between even if left True:
        "root_sync_1_blkTrue": dict(block_between=True, root_sync_cadence=1),
    }
    for label, kwargs in modes.items():
        got = _run_canary_tree_recorded(**kwargs)
        assert got == legacy, f"sync mode {label} diverged from legacy orchestration"


class _Grid:
    def __init__(self, ny: int, nx: int) -> None:
        self.ny = ny
        self.nx = nx


class _State:
    def __init__(self, bdy_width: int = 5) -> None:
        import numpy as _np

        self.u_bdy = _np.zeros((2, 4, int(bdy_width), 1, 1))

    def bytes(self) -> int:
        return 0


class _Namelist:
    radiation_cadence_steps = 4


def test_feedback_weights_are_available_when_runtime_gate_flips_on():
    hierarchy = DomainHierarchy.from_edges(
        ("d01", "d02"),
        (DomainNest("d01", "d02", 3, 2, 2),),
    )
    domains = {
        "d01": DomainBundle("d01", _State(), None, grid=_Grid(12, 12), metrics=object()),
        "d02": DomainBundle("d02", _State(), None, grid=_Grid(24, 24), metrics=object()),
    }

    tree = DomainTree.from_domains(hierarchy, domains, feedback_enabled=False)
    edge = tree.children("d01")[0]

    assert tree.feedback_enabled is False
    assert edge.feedback_weights is not None


def test_live_nested_boundary_config_sets_parent_dt_and_wrf_toggles():
    cfg = build_live_nested_boundary_config(18.0, nested_w_relax=True)
    assert cfg.update_cadence_s == 18.0
    assert cfg.force_geopotential is False
    assert cfg.nested_ph_relax is True
    assert cfg.nested_w_relax is True
    assert cfg.nested_ph_spec is True


def _all7_shaped_hierarchy() -> DomainHierarchy:
    order = ("d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09")
    edges = [DomainNest("d01", "d02", 3, 30, 20)]
    edges += [DomainNest("d02", child, 3, 10, 10) for child in order[2:]]
    return DomainHierarchy.from_edges(order, tuple(edges), max_dom=9)


def _python_d02_fused_lookup(hierarchy: DomainHierarchy):
    child_specs = hierarchy.children("d02")

    def lookup(parent_name: str):
        if parent_name != "d02":
            return None

        def fused(parent_carry, child_carries, parent_start, child_starts):
            del parent_start, child_starts
            parent_new = _Carry(parent_carry.value + 1)
            new_children = []
            for spec, child_carry in zip(child_specs, child_carries):
                forced = _Carry(child_carry.value + parent_new.value * 1000)
                new_children.append(_Carry(forced.value + int(spec.parent_grid_ratio)))
            return parent_new, tuple(new_children)

        return fused

    return lookup


def _run_all7_recorded(*, fused_cascade=None):
    hierarchy = _all7_shaped_hierarchy()
    names = hierarchy.order

    def advance(name, carry, start_step, n_steps):
        del name, start_step
        return _Carry(carry.value + int(n_steps))

    def force(edge, parent, child):
        return _Carry(child.value + parent.value * 1000)

    result = run_domain_tree_callbacks(
        hierarchy,
        {name: _Carry(0) for name in names},
        root_steps=2,
        advance=advance,
        force=force,
        output=lambda name, step, state: (name, step, state.value),
        output_cadence_steps={"d01": 1, "d02": 3, **{child: 9 for child in names[2:]}},
        block_between=False,
        fused_cascade=fused_cascade,
    )
    return (
        result.events,
        result.own_steps,
        {name: carry.value for name, carry in result.carries.items()},
        result.outputs,
    )


def test_fused_cascade_is_scheduler_and_value_identical_to_eager_all7():
    hierarchy = _all7_shaped_hierarchy()
    eager = _run_all7_recorded(fused_cascade=None)
    fused = _run_all7_recorded(fused_cascade=_python_d02_fused_lookup(hierarchy))
    assert fused == eager
    events, own_steps, _carries, _outputs = eager
    assert own_steps == {
        "d01": 2,
        "d02": 6,
        "d03": 18,
        "d04": 18,
        "d05": 18,
        "d06": 18,
        "d07": 18,
        "d08": 18,
        "d09": 18,
    }
    assert sum(1 for event in events if event[0] == "force") == 44
    assert sum(1 for event in events if event[0] == "force" and event[1] == "d02") == 42
    assert sum(1 for event in events if event[0] == "force" and event[1] == "d01") == 2


def test_fused_cascade_none_lookup_is_eager_byte_identical():
    eager = _run_all7_recorded(fused_cascade=None)
    always_none = _run_all7_recorded(fused_cascade=lambda name: None)
    assert always_none == eager


def _stub_all7_tree(*, feedback_enabled: bool = False) -> DomainTree:
    hierarchy = _all7_shaped_hierarchy()
    namelist = _Namelist()
    domains = {
        "d01": DomainBundle("d01", _State(), namelist, grid=_Grid(94, 60), metrics=object()),
        "d02": DomainBundle("d02", _State(), namelist, grid=_Grid(196, 94), metrics=object()),
    }
    for child in hierarchy.order[2:]:
        domains[child] = DomainBundle(child, _State(), namelist, grid=_Grid(40, 40), metrics=object())
    return DomainTree.from_domains(hierarchy, domains, feedback_enabled=feedback_enabled)


def test_fusable_parent_accepts_d02_and_rejects_root_and_leaves():
    tree = _stub_all7_tree(feedback_enabled=False)
    d02_edges = _fusable_parent(tree, "d02")
    assert d02_edges is not None
    assert [edge.child for edge in d02_edges] == list(tree.hierarchy.order[2:])
    assert _fusable_parent(tree, "d01") is None
    assert _fusable_parent(tree, "d03") is None


def test_fusable_parent_rejects_when_feedback_enabled():
    tree = _stub_all7_tree(feedback_enabled=True)
    assert _fusable_parent(tree, "d02") is None


def test_fused_factory_default_on_and_gates_d02_only(monkeypatch):
    monkeypatch.delenv("GPUWRF_BITWISE", raising=False)
    monkeypatch.delenv("GPUWRF_NESTED_FUSE", raising=False)
    _FUSED_PROGRAM_CACHE.clear()
    tree = _stub_all7_tree()
    lookup = _operational_fused_cascade_factory(tree)
    program = lookup("d02")
    assert program is not None
    assert callable(program)
    assert lookup("d01") is None
    assert lookup("d03") is None
    assert lookup("d02") is program
    lookup2 = _operational_fused_cascade_factory(tree)
    assert lookup2("d02") is program
    _FUSED_PROGRAM_CACHE.clear()


def test_fused_factory_eager_optouts(monkeypatch):
    tree = _stub_all7_tree()
    for value in ("0", "false", "off", "no"):
        monkeypatch.delenv("GPUWRF_BITWISE", raising=False)
        monkeypatch.setenv("GPUWRF_NESTED_FUSE", value)
        assert _operational_fused_cascade_factory(tree)("d02") is None
    monkeypatch.delenv("GPUWRF_NESTED_FUSE", raising=False)
    monkeypatch.setenv("GPUWRF_BITWISE", "1")
    assert _operational_fused_cascade_factory(tree)("d02") is None
