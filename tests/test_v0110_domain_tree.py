from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.runtime.domain_tree import (
    DomainBundle,
    DomainTree,
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
        # Only the boundary-leaf SHAPE is read at fused-program BUILD time
        # (``state.u_bdy.shape[2]`` == bdy_width, exactly as eager force does); the
        # contents are never executed in the CPU gate, so a shaped stub suffices.
        import numpy as _np

        self.u_bdy = _np.zeros((2, 4, int(bdy_width), 1, 1))

    def bytes(self) -> int:
        return 0


class _Namelist:
    """Minimal namelist stub: the fused factory reads only the radiation cadence
    (a positive int) at program-build time; the program is not traced in the gate."""

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


# ===========================================================================
# v0.17 GPUWRF_NESTED_FUSE: fused d02-substep cascade scheduler-identity gates.
# ===========================================================================
#
# These prove the FUSED path (one device program per d02 substep) reproduces the
# EAGER recursion's host orchestration byte-for-byte: identical events, own_steps,
# carry structure+values, and outputs.  This is the CPU "scheduler identity" gate
# required by the spec; the GPU numerical (FMA) bit-identity is the parent's
# bitcompare.  We drive the GENERIC ``run_domain_tree_callbacks`` with a Python
# fused closure that mimics the operational cascade arithmetic, so no real
# ``State``/GPU is needed -- the host bookkeeping under test is platform-agnostic.

from gpuwrf.runtime.domain_tree import (  # noqa: E402
    DomainEdge,
    _FUSED_PROGRAM_CACHE,
    _fusable_parent,
    _operational_fused_cascade_factory,
)


def _all7_shaped_hierarchy() -> DomainHierarchy:
    """d01 -> d02 -> {d03..d09}: the all-7 canary shape (one non-leaf parent d02
    whose seven children are all leaves, ratio 3)."""
    order = ("d01", "d02", "d03", "d04", "d05", "d06", "d07", "d08", "d09")
    edges = [DomainNest("d01", "d02", 3, 30, 20)]
    edges += [DomainNest("d02", child, 3, 10, 10) for child in order[2:]]
    return DomainHierarchy.from_edges(order, edges, max_dom=9)


def _python_d02_fused_lookup(hierarchy: DomainHierarchy):
    """A pure-Python fused-cascade closure for d02 that performs EXACTLY the same
    arithmetic the eager advance/force callbacks below do, but as one combined
    call -- the analogue of the operational jitted cascade."""

    child_specs = hierarchy.children("d02")

    def lookup(parent_name: str):
        if parent_name != "d02":
            return None  # fail-closed to eager for the non-fusable root

        def fused(parent_carry, child_carries, parent_start, child_starts):
            parent_new = _Carry(parent_carry.value + 1)  # advance parent 1 step
            new_children = []
            for spec, child_carry in zip(child_specs, child_carries):
                forced = _Carry(child_carry.value + parent_new.value * 1000)  # force
                advanced = _Carry(forced.value + int(spec.parent_grid_ratio))  # subcycle
                new_children.append(advanced)
            return parent_new, tuple(new_children)

        return fused

    return lookup


def _run_all7_recorded(*, fused_cascade=None):
    hierarchy = _all7_shaped_hierarchy()
    names = hierarchy.order
    advance_calls: list[tuple[str, int, int]] = []
    force_calls: list[tuple[str, str, int]] = []

    def advance(name, carry, start_step, n_steps):
        advance_calls.append((name, int(start_step), int(n_steps)))
        return _Carry(carry.value + int(n_steps))

    def force(edge, parent, child):
        force_calls.append((edge.parent, edge.child, parent.value))
        # boundary coupling: child boundary becomes a function of the parent state
        return _Carry(child.value + parent.value * 1000)

    result = run_domain_tree_callbacks(
        hierarchy,
        {name: _Carry(0) for name in names},
        root_steps=2,
        advance=advance,
        force=force,
        output=lambda name, step, state: (name, step, state.value),
        output_cadence_steps={"d01": 1, "d02": 3, **{c: 9 for c in names[2:]}},
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
    """The fused d02-substep path must yield byte-identical events / own_steps /
    carry values / outputs vs the eager recursion for the all-7 shape (one fused
    device program replaces 1 parent advance + 7 forces + 7 child subcycles per
    substep, but the host bookkeeping and arithmetic composition are identical)."""
    hierarchy = _all7_shaped_hierarchy()
    eager = _run_all7_recorded(fused_cascade=None)
    fused = _run_all7_recorded(fused_cascade=_python_d02_fused_lookup(hierarchy))
    assert fused == eager
    # sanity: the all-7 cadence actually exercised the seven-leaf fan-out.
    events, own_steps, _carries, _outputs = eager
    assert own_steps == {
        "d01": 2, "d02": 6,
        "d03": 18, "d04": 18, "d05": 18, "d06": 18, "d07": 18, "d08": 18, "d09": 18,
    }
    # Force events: 2 root steps * 1 (d01->d02) + 2 root steps * 3 d02 substeps *
    # 7 children (d02->d03..d09) = 2 + 42 = 44.  The d02->child forces are the ones
    # the fused program absorbs; the d01->d02 force stays eager.
    assert sum(1 for e in events if e[0] == "force") == 44
    assert sum(1 for e in events if e[0] == "force" and e[1] == "d02") == 42
    assert sum(1 for e in events if e[0] == "force" and e[1] == "d01") == 2


def test_fused_cascade_unset_flag_is_eager_byte_identical():
    """GPUWRF_NESTED_FUSE unset -> the fused lookup is bypassed (None) and results
    are byte-identical to the legacy eager path (default-OFF safety gate)."""
    no_fuse = _run_all7_recorded(fused_cascade=None)
    # passing a lookup that always returns None == flag-unset behaviour:
    always_none = _run_all7_recorded(fused_cascade=lambda name: None)
    assert always_none == no_fuse


def _stub_all7_tree(*, feedback_enabled: bool = False) -> DomainTree:
    hierarchy = _all7_shaped_hierarchy()
    nl = _Namelist()
    domains = {
        "d01": DomainBundle("d01", _State(), nl, grid=_Grid(94, 60), metrics=object()),
        "d02": DomainBundle("d02", _State(), nl, grid=_Grid(196, 94), metrics=object()),
    }
    for child in hierarchy.order[2:]:
        domains[child] = DomainBundle(child, _State(), nl, grid=_Grid(40, 40), metrics=object())
    return DomainTree.from_domains(hierarchy, domains, feedback_enabled=feedback_enabled)


def test_fusable_parent_accepts_d02_and_rejects_root_and_leaves():
    """``_fusable_parent`` fail-closes the fusable d02 subtree IN (flat parent ->
    {leaf children}) and everything else OUT (root has a non-leaf child; leaves
    have no children)."""
    tree = _stub_all7_tree(feedback_enabled=False)
    d02_edges = _fusable_parent(tree, "d02")
    assert d02_edges is not None
    assert [e.child for e in d02_edges] == list(tree.hierarchy.order[2:])
    # d01's only child (d02) is NOT a leaf -> not a flat substep -> reject.
    assert _fusable_parent(tree, "d01") is None
    # leaves have no children.
    assert _fusable_parent(tree, "d03") is None


def test_fusable_parent_rejects_when_feedback_enabled():
    """Two-way feedback would add a parent-write the fused host mirror does not
    model, so the fused path must fail closed to eager when feedback is on."""
    tree = _stub_all7_tree(feedback_enabled=True)
    assert _fusable_parent(tree, "d02") is None


def test_fused_factory_returns_none_when_flag_unset(monkeypatch):
    """The operational factory returns ``None`` for every parent when the env flag
    is unset (default OFF) -> the runner takes the eager path."""
    monkeypatch.delenv("GPUWRF_NESTED_FUSE", raising=False)
    tree = _stub_all7_tree()
    lookup = _operational_fused_cascade_factory(tree)
    assert lookup("d02") is None
    assert lookup("d01") is None


def test_fused_factory_gates_d02_only_when_flag_set(monkeypatch):
    """With the flag set, the factory builds a fused program for the fusable d02
    subtree and still returns ``None`` for the (non-fusable) root and leaves.  The
    program is cached per (tree, parent) so a second lookup returns the SAME object
    (JAX compiles it once across forecast segments)."""
    monkeypatch.setenv("GPUWRF_NESTED_FUSE", "1")
    _FUSED_PROGRAM_CACHE.clear()
    tree = _stub_all7_tree()
    lookup = _operational_fused_cascade_factory(tree)
    prog = lookup("d02")
    assert prog is not None
    assert callable(prog)
    assert lookup("d01") is None
    assert lookup("d03") is None
    # cache identity within one factory:
    assert lookup("d02") is prog
    # cache identity across factories for the SAME tree (segment-to-segment reuse):
    lookup2 = _operational_fused_cascade_factory(tree)
    assert lookup2("d02") is prog
    _FUSED_PROGRAM_CACHE.clear()
