from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.runtime.domain_tree import (
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


def test_live_nested_boundary_config_sets_parent_dt_and_wrf_toggles():
    cfg = build_live_nested_boundary_config(18.0, nested_w_relax=True)
    assert cfg.update_cadence_s == 18.0
    assert cfg.force_geopotential is False
    assert cfg.nested_ph_relax is True
    assert cfg.nested_w_relax is True
    assert cfg.nested_ph_spec is True
