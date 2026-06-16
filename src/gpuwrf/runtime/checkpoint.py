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

try:  # NoahMPLandState lives in the v0.2.0 land package; import is optional so a
    # pure-dycore checkout without the noahmp package can still read/write v1.
    from gpuwrf.contracts.noahmp_state import NoahMPLandState
except Exception:  # pragma: no cover
    NoahMPLandState = None  # type: ignore


config.update("jax_enable_x64", True)

# v2 (ADR-NOAHMP-INTERFACES.md §5): adds the optional prognostic Noah-MP land
# carry + a scope-options guard. v1 checkpoints (no land state) stay READABLE and
# load with ``noahmp_land_state = None`` -> driver cold-inits land from wrfinput.
# v3 (v0.6.0 S0 State-leaf materialization): appends the additive physics State
# leaves Nc/Nn/rainc_acc. A v1/v2 checkpoint recorded the pre-v0.6.0 leaf order
# (no Nc/Nn/rainc_acc); it stays READABLE because the new leaves are APPEND-ONLY
# -- the reader backfills any missing additive leaf with zeros (cold-start), and
# requires the recorded order to be a prefix of the current schema (fail-closed
# on any non-prefix divergence). A v3 writer always records the full order.
FORMAT_VERSION = 3
SUPPORTED_FORMAT_VERSIONS = (1, 2, 3)

# State leaves added after the original (v1/v2) checkpoint schema. The reader
# backfills any of these absent from an older checkpoint with zeros, so old
# restarts cold-start the new physics fields rather than failing closed.
# v0.6.0 added Nc/Nn/rainc_acc; v0.15 the MYNN SGS-cloud leaves; v0.17 ADR-032
# the graupel/hail substrate qh/Nh/qvolg/qvolh. v0.16 Thompson-aero added
# nwfa/nifa. All are append-only physics tail leaves that legitimately
# cold-start at zero from an older checkpoint.
ADDITIVE_STATE_LEAVES_SINCE_V2 = (
    "Nc",
    "Nn",
    "rainc_acc",
    "qsq",
    "qc_bl",
    "qi_bl",
    "cldfra_bl",
    "nwfa",
    "nifa",
    "qh",
    "Nh",
    "qvolg",
    "qvolh",
)

# The frozen Noah-MP scope-options the land carry is valid under (tables.py mirror).
NOAHMP_SCOPE_OPTIONS = {
    "dveg": 4, "opt_crs": 1, "opt_btr": 1, "opt_run": 3, "opt_sfc": 1,
    "opt_frz": 1, "opt_inf": 1, "opt_rad": 3, "opt_alb": 2, "opt_snf": 1,
    "opt_tbot": 2, "opt_stc": 1,
}


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


def _land_field_order() -> tuple[str, ...]:
    if NoahMPLandState is None:
        raise ValueError("NoahMPLandState unavailable; cannot serialize land carry")
    return tuple(NoahMPLandState.__slots__)


def _land_to_host_fields(land_state) -> dict[str, np.ndarray]:
    return {field: np.asarray(getattr(land_state, field)) for field in _land_field_order()}


def write_checkpoint(
    state: State,
    namelist: Any,
    grid: Any,
    step_index: int,
    path: str | Path,
    *,
    runtime_state: Any | None = None,
    land_state: Any | None = None,
    scope_options: dict | None = None,
) -> Path:
    """Write a restart checkpoint containing State, namelist, grid, and step index.

    The State payload is stored as an explicit field dictionary so schema drift
    fails closed instead of silently relying on pickle internals. When
    ``land_state`` is given, the prognostic Noah-MP land carry is written under
    ``noahmp_land_state`` with the same fail-closed field-order discipline (v2);
    omitting it yields a payload a v1 reader can consume (the land keys are simply
    absent / None).
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
        # --- v2 Noah-MP land carry (ADR §5). None when not coupled. ---
        "noahmp_land_state": None,
        "noahmp_land_field_order": None,
        "noahmp_format": None,
    }
    if land_state is not None:
        order = _land_field_order()
        payload["noahmp_land_state"] = _land_to_host_fields(land_state)
        payload["noahmp_land_field_order"] = list(order)
        payload["noahmp_format"] = {
            "nsoil": 4, "nsnow": 3,
            "scope_options": dict(scope_options) if scope_options else dict(NOAHMP_SCOPE_OPTIONS),
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
    version = int(payload.get("format_version", -1))
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"unsupported checkpoint version {payload.get('format_version')} in {source}; "
            f"supported {SUPPORTED_FORMAT_VERSIONS}"
        )

    expected = tuple(State.__slots__)
    recorded = tuple(payload.get("state_field_order", ()))
    # Append-only schema rule (v0.6.0 S0): the recorded order must be either the
    # current full order, or a PREFIX of it whose only missing tail leaves are the
    # append-only additive leaves (Nc/Nn/rainc_acc). Any other divergence (a
    # reordered or unknown leaf, or a missing non-additive leaf) still fails closed.
    if recorded != expected:
        if recorded != expected[: len(recorded)]:
            raise ValueError("checkpoint State field order does not match current State schema")
        missing_tail = expected[len(recorded):]
        if any(leaf not in ADDITIVE_STATE_LEAVES_SINCE_V2 for leaf in missing_tail):
            raise ValueError(
                "checkpoint State field order is missing non-additive leaves: "
                f"{[leaf for leaf in missing_tail if leaf not in ADDITIVE_STATE_LEAVES_SINCE_V2]}"
            )
    fields = payload.get("state_fields")
    if not isinstance(fields, dict) or set(fields) != set(recorded):
        raise ValueError("checkpoint State fields do not match the recorded field order")

    # v2 land carry: exact-match the field order (fail-closed, like State). A v1
    # checkpoint or a v2 with no land carry simply has noahmp_land_state == None.
    land_fields = payload.get("noahmp_land_state")
    if land_fields is not None:
        land_order = tuple(payload.get("noahmp_land_field_order", ()))
        if NoahMPLandState is None:
            raise ValueError("checkpoint carries Noah-MP land state but NoahMPLandState is unavailable")
        if land_order != tuple(NoahMPLandState.__slots__):
            raise ValueError("checkpoint Noah-MP land field order does not match current schema")
        if set(land_fields) != set(NoahMPLandState.__slots__):
            raise ValueError("checkpoint Noah-MP land fields do not match current schema")
    return payload


def read_checkpoint(path: str | Path) -> tuple[State, Any, Any, int]:
    """Read a checkpoint and reconstruct JAX pytrees on the default device."""

    payload = _read_payload(path)
    return _restore_checkpoint_payload(payload)


def _restore_checkpoint_payload(payload: dict[str, Any]) -> tuple[State, Any, Any, int]:
    # Construct from the RECORDED leaves only. Append-only additive leaves absent
    # from an older (v1/v2) checkpoint default to None in ``State.__init__``, which
    # backfills them with zeros at the matrix dtype (cold-start the new physics
    # fields). A v3 checkpoint records all leaves, so nothing is defaulted.
    fields = payload["state_fields"]
    state = State(**{field: jax.device_put(value) for field, value in fields.items()})
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


def _restore_land_state(payload: dict[str, Any]):
    """Rebuild the NoahMPLandState carry (or None for a v1 / land-less payload)."""
    land_fields = payload.get("noahmp_land_state")
    if land_fields is None:
        return None
    order = tuple(NoahMPLandState.__slots__)
    return NoahMPLandState(**{f: jax.device_put(land_fields[f]) for f in order})


def read_checkpoint_with_land_state(
    path: str | Path,
) -> tuple[State, Any, Any, int, Any]:
    """Read a checkpoint and the prognostic Noah-MP land carry (ADR §5).

    Returns ``(state, namelist, grid, step_index, land_state)`` where
    ``land_state`` is the restored ``NoahMPLandState`` for a v2 checkpoint, or
    ``None`` for a v1 (or land-less v2) checkpoint — in which case the caller
    cold-initialises Noah-MP land state from ``wrfinput``.
    """
    payload = _read_payload(path)
    state, namelist, grid, step_index = _restore_checkpoint_payload(payload)
    land_state = _restore_land_state(payload)
    return state, namelist, grid, step_index, land_state


__all__ = [
    "read_checkpoint",
    "read_checkpoint_with_runtime_state",
    "read_checkpoint_with_land_state",
    "write_checkpoint",
]
