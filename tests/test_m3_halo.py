from __future__ import annotations

import inspect

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.halo import HaloSpec, apply_halo
from gpuwrf.contracts.state import State


def test_apply_halo_single_gpu_noop_identity():
    grid = GridSpec.canary_3km_template()
    state = State.zeros(grid)
    halo = HaloSpec(width=grid.halo_width, fields_to_exchange=("u", "v", "theta"), edge_type="open")

    assert apply_halo(state, halo) is state


def test_apply_halo_signature_is_future_drop_in():
    signature = inspect.signature(apply_halo)

    assert list(signature.parameters) == ["state", "halo"]
    assert signature.return_annotation in {"State", State}


def test_halospec_rejects_unknown_edge_type():
    try:
        HaloSpec(width=2, fields_to_exchange=("theta",), edge_type="closed")
    except ValueError as exc:
        assert "edge_type" in str(exc)
    else:
        raise AssertionError("HaloSpec accepted invalid edge_type")
