"""Pickle checkpoints for M7 restart-continuity probes."""

from __future__ import annotations

from dataclasses import is_dataclass, replace
from pathlib import Path
import pickle
from typing import Any

import jax
from jax import config
import numpy as np

from gpuwrf.contracts.state import State


config.update("jax_enable_x64", True)

FORMAT_VERSION = 1


def _hostify_tree(tree: Any) -> Any:
    """Copy pytree array leaves to NumPy arrays for process-independent pickle."""

    return jax.tree_util.tree_map(lambda leaf: np.asarray(leaf), tree)


def _device_tree(tree: Any) -> Any:
    """Place checkpoint pytree leaves on the default JAX device after load."""

    return jax.tree_util.tree_map(lambda leaf: jax.device_put(leaf), tree)


def _hostify_namelist(namelist: Any, grid: Any) -> Any:
    host_namelist = _hostify_tree(namelist)
    if is_dataclass(host_namelist) and hasattr(host_namelist, "grid"):
        return replace(host_namelist, grid=grid)
    return host_namelist


def _restore_namelist(namelist: Any, grid: Any) -> Any:
    device_namelist = _device_tree(namelist)
    if is_dataclass(device_namelist) and hasattr(device_namelist, "grid"):
        return replace(device_namelist, grid=grid)
    return device_namelist


def _state_to_host_fields(state: State) -> dict[str, np.ndarray]:
    return {field: np.asarray(getattr(state, field)) for field in State.__slots__}


def write_checkpoint(
    state: State,
    namelist: Any,
    grid: Any,
    step_index: int,
    path: str | Path,
    *,
    runtime_state: Any | None = None,
) -> Path:
    """Write a restart checkpoint containing State, namelist, grid, and step index.

    The State payload is stored as an explicit field dictionary so schema drift
    fails closed instead of silently relying on pickle internals.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    host_grid = _hostify_tree(grid)
    payload = {
        "format": "gpuwrf-runtime-checkpoint",
        "format_version": FORMAT_VERSION,
        "state_type": "gpuwrf.contracts.state.State",
        "state_field_order": list(State.__slots__),
        "state_field_count": len(State.__slots__),
        "state_fields": _state_to_host_fields(state),
        "namelist": _hostify_namelist(namelist, host_grid),
        "grid": host_grid,
        "step_index": int(step_index),
        "runtime_state": None if runtime_state is None else _hostify_tree(runtime_state),
    }
    tmp = target.with_name(f"{target.name}.tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(target)
    return target


def _read_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("rb") as handle:
        payload = pickle.load(handle)

    if payload.get("format") != "gpuwrf-runtime-checkpoint":
        raise ValueError(f"unsupported checkpoint format in {source}")
    if int(payload.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {payload.get('format_version')} in {source}; "
            f"expected {FORMAT_VERSION}"
        )

    expected = tuple(State.__slots__)
    recorded = tuple(payload.get("state_field_order", ()))
    if recorded != expected:
        raise ValueError("checkpoint State field order does not match current State schema")
    fields = payload.get("state_fields")
    if not isinstance(fields, dict) or set(fields) != set(expected):
        raise ValueError("checkpoint State fields do not match current State schema")
    return payload


def read_checkpoint(path: str | Path) -> tuple[State, Any, Any, int]:
    """Read a checkpoint and reconstruct JAX pytrees on the default device."""

    payload = _read_payload(path)
    return _restore_checkpoint_payload(payload)


def _restore_checkpoint_payload(payload: dict[str, Any]) -> tuple[State, Any, Any, int]:
    expected = tuple(State.__slots__)
    fields = payload["state_fields"]
    state = State(**{field: jax.device_put(fields[field]) for field in expected})
    grid = _device_tree(payload["grid"])
    namelist = _restore_namelist(payload["namelist"], grid)
    return state, namelist, grid, int(payload["step_index"])


def read_checkpoint_with_runtime_state(path: str | Path) -> tuple[State, Any, Any, int, Any | None]:
    """Read a checkpoint including optional operational reproducibility state."""

    payload = _read_payload(path)
    state, namelist, grid, step_index = _restore_checkpoint_payload(payload)
    runtime_state = payload.get("runtime_state")
    if runtime_state is not None:
        runtime_state = _device_tree(runtime_state)
    return state, namelist, grid, step_index, runtime_state


__all__ = ["read_checkpoint", "read_checkpoint_with_runtime_state", "write_checkpoint"]
