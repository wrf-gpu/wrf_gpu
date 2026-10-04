"""SI36: live-nest child *_bdy placeholders get the force-down's 2-level shape at init.

The first force-down's ``two_time`` reads only the placeholder's last-level shape/dtype
(boundary_construction._package_updates), so widening 1-level placeholders is exact and
lets force-down call 1 share the steady jit signature (one warm load instead of two).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np

from gpuwrf.integration.nested_pipeline import _two_time_live_nest_bdy


@dataclass(frozen=True)
class _State:
    __slots__ = ("u", "u_bdy", "mu_bdy", "qc_bdy", "Nh_bdy")
    u: object
    u_bdy: object
    mu_bdy: object
    qc_bdy: object
    Nh_bdy: object

    def replace(self, **kw):
        return replace(self, **kw)


def test_widens_only_single_level_bdy_placeholders():
    u = jnp.ones((3, 4, 5))
    one = jnp.arange(2 * 3 * 4, dtype=jnp.float32).reshape(1, 2, 3, 4)
    two = jnp.zeros((2, 2, 3, 4), jnp.float32)
    state = _State(u=u, u_bdy=one, mu_bdy=one[..., :1, :], qc_bdy=two, Nh_bdy=None)
    out = _two_time_live_nest_bdy(state)
    assert out.u_bdy.shape == (2, 2, 3, 4) and out.u_bdy.dtype == one.dtype
    assert np.array_equal(np.asarray(out.u_bdy[0]), np.asarray(one[0]))
    assert np.array_equal(np.asarray(out.u_bdy[1]), np.asarray(one[0]))
    assert out.mu_bdy.shape == (2, 2, 1, 4)
    assert out.qc_bdy is two and out.Nh_bdy is None and out.u is u


def test_root_like_state_without_single_level_leaves_is_unchanged():
    state = _State(u=jnp.ones(2), u_bdy=jnp.zeros((5, 2)), mu_bdy=None, qc_bdy=None, Nh_bdy=None)
    assert _two_time_live_nest_bdy(state) is state


class _Edge:
    def __init__(self, parent, child, coupled):
        self.parent, self.child, self.coupled_forcedown = parent, child, coupled


class _Tree:
    def __init__(self, edges):
        self.edges = edges


def test_forcedown_warmup_runs_each_coupled_edge_once_and_discards(monkeypatch):
    from types import SimpleNamespace

    import gpuwrf.runtime.domain_tree as dt
    from gpuwrf.integration.nested_pipeline import _start_forcedown_warmup

    calls = []

    def _force(edge, parent, child):
        calls.append((edge.parent, edge.child, parent, child))
        return SimpleNamespace(state=SimpleNamespace(u_bdy=jnp.zeros(2)))

    monkeypatch.setattr(dt, "_operational_force", _force)
    monkeypatch.delenv("GPUWRF_FORCEDOWN_WARMUP", raising=False)
    carries = {"d01": "c1", "d02": "c2", "d03": "c3"}
    tree = _Tree({"d01": (_Edge("d01", "d02", True),), "d02": (_Edge("d02", "d03", False),)})
    report = _start_forcedown_warmup(tree, carries)
    report["thread"].join()
    assert calls == [("d01", "d02", "c1", "c2")]
    assert report["edges"] == ["d01->d02"] and report["error"] is None


def test_forcedown_warmup_fails_open_and_honours_opt_out(monkeypatch):
    import gpuwrf.runtime.domain_tree as dt
    from gpuwrf.integration.nested_pipeline import _start_forcedown_warmup

    def _boom(*_a):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(dt, "_operational_force", _boom)
    tree = _Tree({"d01": (_Edge("d01", "d02", True),)})
    report = _start_forcedown_warmup(tree, {"d01": 1, "d02": 2})
    report["thread"].join()
    assert report["error"] == "RuntimeError: compile failed"
    monkeypatch.setenv("GPUWRF_FORCEDOWN_WARMUP", "0")
    assert _start_forcedown_warmup(tree, {"d01": 1, "d02": 2}) is None


def test_forcedown_warmup_setup_fails_open_on_unexpected_tree_shapes():
    """Trees without (or with broken) edges -- e.g. restart/test stand-ins -- only skip the warm-up."""
    from types import SimpleNamespace

    from gpuwrf.integration import nested_pipeline as pipeline

    class _BrokenEdges:
        @property
        def edges(self):
            raise RuntimeError("no edges here")

    carries = {"d01": object(), "d02": object()}
    assert pipeline._start_forcedown_warmup(SimpleNamespace(), carries) is None
    assert pipeline._start_forcedown_warmup(SimpleNamespace(edges=None), carries) is None
    assert pipeline._start_forcedown_warmup(_BrokenEdges(), carries) is None
    assert pipeline._start_forcedown_warmup(SimpleNamespace(edges={"d01": [object()]}), carries) is None


def test_forcedown_warmup_join_is_bounded():
    """A stuck warm-up is abandoned after the timeout instead of stalling the run."""
    import threading

    from gpuwrf.integration import nested_pipeline as pipeline

    release = threading.Event()
    thread = threading.Thread(target=release.wait, daemon=True)
    thread.start()
    report = {"edges": ["d01->d02"], "error": None, "thread": thread}
    meta = pipeline._join_forcedown_warmup(report, timeout_s=0.05)
    assert meta["joined"] is False and meta["remaining_wait_s"] < 5.0 and meta["edges"] == ["d01->d02"]
    release.set()
    assert pipeline._join_forcedown_warmup(report, timeout_s=5.0)["joined"] is True
