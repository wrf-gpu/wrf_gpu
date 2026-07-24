"""CPU-only gates for prepared B=1 operational-domain-tree runtime reuse."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

from gpuwrf.contracts.grid import DomainHierarchy, DomainNest
from gpuwrf.integration import nested_pipeline as pipeline
from gpuwrf.runtime import domain_tree as dt


@dataclass(frozen=True)
class _Carry:
    value: int


def _bundle() -> SimpleNamespace:
    return SimpleNamespace(
        state=_Carry(0),
        namelist=SimpleNamespace(rk_order=3),
    )


def _single_domain_tree() -> dt.DomainTree:
    hierarchy = DomainHierarchy.from_edges(("d01",), ())
    return dt.DomainTree(hierarchy=hierarchy, domains={"d01": _bundle()})


def _patch_deterministic_runtime(monkeypatch):
    counts = {"advance_factory": 0, "fused_factory": 0}

    monkeypatch.setattr(dt, "_resolve_operational_suite", lambda _namelist: None)

    def advance_factory(_tree):
        counts["advance_factory"] += 1

        def advance(_name, carry, _start_step, n_steps):
            return _Carry(carry.value + int(n_steps))

        return advance

    def fused_factory(_tree):
        counts["fused_factory"] += 1
        return lambda _parent: None

    monkeypatch.setattr(dt, "_operational_advance_factory", advance_factory)
    monkeypatch.setattr(dt, "_operational_fused_cascade_factory", fused_factory)
    return counts


def _run_resumed_segments(tree, *, prepared_runtime=None):
    carries = {"d01": _Carry(0)}
    own_steps = {"d01": 0}
    events = []
    outputs = []
    for _segment in range(2):
        result = dt.run_operational_domain_tree(
            tree,
            root_steps=2,
            carries=carries,
            initial_own_steps=own_steps,
            prepared_runtime=prepared_runtime,
            output=lambda name, step, state: (name, step, state.value),
            output_cadence_steps={"d01": 1},
            block_between=False,
        )
        carries = result.carries
        own_steps = result.own_steps
        events.extend(result.events)
        outputs.extend(result.outputs)
    return carries, own_steps, tuple(events), tuple(outputs)


def test_prepared_runtime_reuses_one_factory_and_is_exact(monkeypatch):
    """Two resumed segments reuse one callable without changing any host result."""

    tree = _single_domain_tree()
    counts = _patch_deterministic_runtime(monkeypatch)

    legacy = _run_resumed_segments(tree)
    assert counts == {"advance_factory": 2, "fused_factory": 2}

    counts.update(advance_factory=0, fused_factory=0)
    runtime = dt._prepare_operational_domain_tree_runtime(tree)
    prepared = _run_resumed_segments(tree, prepared_runtime=runtime)

    assert counts == {"advance_factory": 1, "fused_factory": 1}
    assert prepared == legacy
    assert prepared[0]["d01"] == _Carry(4)
    assert prepared[1] == {"d01": 4}


def test_prepared_runtime_reuses_with_move_and_adaptive_hooks(monkeypatch):
    """Dynamic hooks disable fusion as before but safely reuse the advance memo."""

    hierarchy = DomainHierarchy.from_edges(
        ("d01", "d02"),
        (DomainNest("d01", "d02", 2, 2, 2),),
    )
    edge = dt.DomainEdge(hierarchy.nests[0], weights=object())
    tree = dt.DomainTree(
        hierarchy=hierarchy,
        domains={"d01": _bundle(), "d02": _bundle()},
        edges={"d01": (edge,)},
    )
    counts = _patch_deterministic_runtime(monkeypatch)
    monkeypatch.setattr(dt, "_operational_force", lambda _edge, _parent, child: child)

    runtime = dt._prepare_operational_domain_tree_runtime(tree)
    first = dt.run_operational_domain_tree(
        tree,
        root_steps=1,
        carries={"d01": _Carry(0), "d02": _Carry(0)},
        prepared_runtime=runtime,
        move=lambda _edge, _parent, _child, _step: None,
        adaptive_dt=lambda _name, carry, _step: carry,
        block_between=False,
    )
    second = dt.run_operational_domain_tree(
        tree,
        root_steps=1,
        carries=first.carries,
        initial_own_steps=first.own_steps,
        prepared_runtime=runtime,
        move=lambda _edge, _parent, _child, _step: None,
        adaptive_dt=lambda _name, carry, _step: carry,
        block_between=False,
    )

    assert counts == {"advance_factory": 1, "fused_factory": 1}
    assert second.own_steps == {"d01": 2, "d02": 4}


def test_two_way_runtime_reuses_and_mode_mismatch_falls_back(monkeypatch):
    """Two-way preparation is reusable; a one-way runtime cannot cross the gate."""

    tree = _single_domain_tree()
    counts = _patch_deterministic_runtime(monkeypatch)

    two_way = dt._prepare_operational_domain_tree_runtime(tree, feedback_enabled=True)
    assert two_way.fused_cascade is None
    for _ in range(2):
        dt.run_operational_domain_tree(
            tree,
            root_steps=1,
            carries={"d01": _Carry(0)},
            feedback_enabled=True,
            prepared_runtime=two_way,
            block_between=False,
        )
    assert counts == {"advance_factory": 1, "fused_factory": 0}

    counts.update(advance_factory=0, fused_factory=0)
    one_way = dt._prepare_operational_domain_tree_runtime(tree, feedback_enabled=False)
    dt.run_operational_domain_tree(
        tree,
        root_steps=1,
        carries={"d01": _Carry(0)},
        feedback_enabled=True,
        prepared_runtime=one_way,
        block_between=False,
    )
    # Preparing one-way constructs once; the feedback-mode mismatch explicitly
    # rebuilds the safe two-way runtime rather than silently reusing it.
    assert counts == {"advance_factory": 2, "fused_factory": 1}


def test_wrong_tree_runtime_uses_one_safe_fresh_construction(monkeypatch):
    """A runtime prepared for tree A is never executed against distinct tree B."""

    tree_a = _single_domain_tree()
    tree_b = _single_domain_tree()
    factory_trees = []

    monkeypatch.setattr(dt, "_resolve_operational_suite", lambda _namelist: None)

    def advance_factory(tree):
        factory_trees.append(tree)
        increment = 1 if tree is tree_a else 10

        def advance(_name, carry, _start_step, n_steps):
            return _Carry(carry.value + increment * int(n_steps))

        return advance

    monkeypatch.setattr(dt, "_operational_advance_factory", advance_factory)
    monkeypatch.setattr(
        dt, "_operational_fused_cascade_factory", lambda _tree: lambda _parent: None
    )

    runtime_a = dt._prepare_operational_domain_tree_runtime(tree_a)
    result_b = dt.run_operational_domain_tree(
        tree_b,
        root_steps=1,
        carries={"d01": _Carry(0)},
        prepared_runtime=runtime_a,
        block_between=False,
    )

    assert factory_trees == [tree_a, tree_b]
    assert result_b.carries["d01"] == _Carry(10)


class _PipelineState:
    def __init__(self) -> None:
        self.theta = np.asarray([300.0], dtype=np.float64)

    def bytes(self) -> int:
        return int(self.theta.nbytes)


@pytest.mark.parametrize(
    ("reuse_env", "expect_prepared", "fusion_env", "expect_fusion_k"),
    [(None, True, None, 0), ("0", False, None, 0), (None, True, "1", 1)],
    ids=("default-prepared", "ab-released-lifetime", "event-aware-k1"),
)
def test_execute_pipeline_prepared_runtime_ab_wiring(
    monkeypatch,
    tmp_path,
    reuse_env,
    expect_prepared,
    fusion_env,
    expect_fusion_k,
):
    """Default reuses once; the proof arm reconstructs the released lifetime."""

    state = _PipelineState()
    hierarchy = DomainHierarchy.from_edges(("d01",), ())
    bundle = SimpleNamespace(
        state=state,
        namelist=SimpleNamespace(rk_order=3),
    )
    sentinel = object()
    prepare_calls = []
    segment_runtimes = []
    segment_fusion_ks = []

    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    if reuse_env is None:
        monkeypatch.delenv("GPUWRF_PREPARED_RUNTIME_REUSE", raising=False)
    else:
        monkeypatch.setenv("GPUWRF_PREPARED_RUNTIME_REUSE", reuse_env)
    if fusion_env is None:
        monkeypatch.delenv("GPUWRF_NESTED_EVENT_AWARE_FUSION_K", raising=False)
    else:
        monkeypatch.setenv("GPUWRF_NESTED_EVENT_AWARE_FUSION_K", fusion_env)
    monkeypatch.setattr(pipeline, "_batch_ensemble_size_from_env", lambda: 1)
    monkeypatch.setattr(
        pipeline,
        "_load_domains",
        lambda _config, _names: (
            hierarchy,
            {"d01": bundle},
            {"domains": {"d01": {}}},
            datetime(2025, 2, 28, 18, tzinfo=timezone.utc),
            {"d01": 900.0},
            {"d01": _Carry(0)},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_output_cadence_steps_by_domain",
        lambda _run, _names, _dt: ({"d01": 2}, {"d01": 30.0}),
    )
    monkeypatch.setattr(
        pipeline,
        "maybe_prewarm_defused_nest",
        lambda _tree, *, carries: {
            "active": False,
            "source": "test",
            "workers": 0,
            "report": {},
            "error": None,
        },
    )
    monkeypatch.setattr(pipeline, "nested_precompile_report", lambda: {})
    monkeypatch.setattr(pipeline, "_nested_async_output_from_env", lambda: False)
    monkeypatch.setattr(
        pipeline, "assert_state_finite_at_boundary", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        pipeline, "_finite_stats_host", lambda _state: {"all_finite": True}
    )
    aot_reports = iter(
        ([{"load_count": 9, "domains": {"d01": {"load_count": 9}}}]
         if expect_prepared
         else [
             {"load_count": 9, "domains": {"d01": {"load_count": 9}}},
             {"load_count": 9, "domains": {"d01": {"load_count": 9}}},
         ])
    )
    monkeypatch.setattr(pipeline, "nested_aot_report", lambda: next(aot_reports))

    def fake_prepare(tree, *, feedback_enabled):
        prepare_calls.append((tree, feedback_enabled))
        return sentinel

    monkeypatch.setattr(
        pipeline, "_prepare_operational_domain_tree_runtime", fake_prepare
    )

    def fake_run(
        tree,
        *,
        root_steps,
        carries,
        initial_own_steps,
        prepared_runtime,
        **_kwargs,
    ):
        segment_runtimes.append(prepared_runtime)
        segment_fusion_ks.append(_kwargs["event_aware_fusion_k"])
        own_step = int(initial_own_steps["d01"]) + int(root_steps)
        return dt.DomainTreeResult(
            carries={"d01": _Carry(own_step)},
            states={"d01": state},
            own_steps={"d01": own_step},
            events=(),
            outputs=(),
            cascade_counts=(
                {"fused:d02": 1} if expect_fusion_k == 1 else {}
            ),
        )

    monkeypatch.setattr(pipeline, "run_operational_domain_tree", fake_run)

    class FakeWriter:
        writer_static_latlon_metadata = {}

        def __init__(self, **_kwargs) -> None:
            self.written = {"d01": ["frame-1", "frame-2"]}

    monkeypatch.setattr(pipeline, "_PerDomainWrfoutWriter", FakeWriter)

    from gpuwrf.io import gen2_accessor
    from gpuwrf.profiling import transfer_audit

    monkeypatch.setattr(gen2_accessor, "Gen2Run", lambda _path: object())
    monkeypatch.setattr(transfer_audit, "visible_gpu_name", lambda: "CPU-test")

    import jax

    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)

    payload = pipeline.execute_nested_pipeline(
        pipeline.NestedPipelineConfig(
            input_dir=tmp_path / "input",
            output_dir=tmp_path / "output",
            proof_dir=tmp_path / "proof",
            hours=1,
            max_dom=1,
        )
    )

    assert len(prepare_calls) == int(expect_prepared)
    if expect_prepared:
        assert prepare_calls[0][1] is False
    assert len(segment_runtimes) == 2
    if expect_prepared:
        assert segment_runtimes[0] is sentinel
        assert segment_runtimes[1] is sentinel
    else:
        assert segment_runtimes == [None, None]
    assert segment_fusion_ks == [expect_fusion_k, expect_fusion_k]
    assert payload["root_steps"] == 4
    assert payload["metadata"]["nested_runtime"] == {
        "prepared_runtime_reuse": expect_prepared,
        "selection_env": "GPUWRF_PREPARED_RUNTIME_REUSE",
    }
    assert payload["metadata"]["nested_aot"]["load_count"] == (
        9 if expect_prepared else 18
    )
    assert payload["metadata"]["nested_aot"]["domains"]["d01"]["load_count"] == (
        9 if expect_prepared else 18
    )
    if expect_fusion_k:
        assert payload["metadata"]["nested_event_aware_fusion"] == {
            "k": 1,
            "selection_env": "GPUWRF_NESTED_EVENT_AWARE_FUSION_K",
            "cascade_counts": {"fused:d02": 2},
        }
    else:
        assert "nested_event_aware_fusion" not in payload["metadata"]


@pytest.mark.parametrize("value", ["maybe", "2", "", "enabled"])
def test_prepared_runtime_ab_control_rejects_unknown_values(monkeypatch, value):
    monkeypatch.setenv("GPUWRF_PREPARED_RUNTIME_REUSE", value)
    with pytest.raises(ValueError, match="GPUWRF_PREPARED_RUNTIME_REUSE"):
        pipeline._prepared_runtime_reuse_from_env()


def test_released_lifetime_telemetry_aggregates_three_nine_domain_factories():
    reports = []
    for _segment in range(3):
        reports.append(
            {
                "enabled": True,
                "load_count": 9,
                "load_wall_seconds": 9.0,
                "cache_hit_count": 0,
                "domains": {
                    f"d{index:02d}": {
                        "load_count": 1,
                        "load_wall_seconds": 1.0,
                        "load_attempt_count": 1,
                        "load_attempt_wall_seconds": 1.0,
                        "cache_hit_count": 0,
                    }
                    for index in range(1, 10)
                },
            }
        )

    aggregate = pipeline._aggregate_nested_aot_reports(reports)

    assert aggregate["load_count"] == 27
    assert aggregate["load_wall_seconds"] == 27.0
    assert aggregate["factory_report_count"] == 3
    assert all(domain["load_count"] == 3 for domain in aggregate["domains"].values())
