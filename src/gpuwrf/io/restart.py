"""``wrfrst``-equivalent restart for the FULL operational physics carry (P0-5b).

WRF's restart file (``wrfrst``) is a *bit-continuation* snapshot: reloading it and
continuing the integration reproduces the uninterrupted run exactly, because it
serializes not just the prognostic state but every piece of physics carry the
driver holds between steps (held radiation tendencies, accumulators, sub-step
scratch, land-surface memory, ...). ``runtime.checkpoint`` already round-trips the
prognostic :class:`State` (+ optional runtime/land sub-trees), but the operational
driver advances an :class:`~gpuwrf.runtime.operational_state.OperationalCarry`, and
the parts of that carry OUTSIDE ``State`` are exactly the WRF restart-relevant
physics memory:

* the WRF small-step scratch family ``t_2ave/ww/mudf/muave/muts/ph_tend`` and the
  RK/acoustic transition ``*_save`` arrays (``runtime.operational_state``);
* ``rthraten`` — the HELD radiative potential-temperature tendency (K/s) WRF
  refreshes once per ``radt`` and ADDS every dynamics step in between. Dropping it
  at a restart that lands mid-radiation-interval would silently lose up to one
  ``radt`` of radiative heating — a real (small) forecast discontinuity;
* ``noahmp_land`` — the prognostic Noah-MP land carry (soil/snow/canopy memory),
  and ``noahmp_rad`` — the held SOLDN/LWDN/COSZ surface-radiation forcing.

This module is the I/O-lane restart schema that serializes that COMPLETE carry so a
resumed forecast bit-continues. It does NOT touch ``runtime/checkpoint.py``
(runtime lane owns that); it is a strict superset built on the same fail-closed,
host-numpy, schema-versioned discipline.

Serialization discipline (mirrors ``runtime.checkpoint``):
* every array leaf is copied to a host ``numpy`` array (process-independent pickle,
  no device dependency — the save/load round-trip is CPU-only and GPU-free);
* each pytree is stored as an explicit ``{field_name: array}`` dict with the
  recorded ``__slots__``/field order, so a schema change FAILS CLOSED on read
  rather than silently mis-reconstructing;
* ``format``/``format_version`` gate the payload.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from dataclasses import fields as dataclass_fields, is_dataclass, replace
from pathlib import Path
import pickle
import os
import tempfile
from typing import Any

import jax
from jax import config
import numpy as np

from gpuwrf.contracts.state import CONDITIONAL_STATE_LEAVES, State
from gpuwrf.contracts.precision import GWDO_DIAGNOSTIC_LEAVES
from gpuwrf.runtime.operational_state import OperationalCarry

try:  # Noah-MP land/static live in the v0.2.0 land package; optional so a
    # pure-dycore checkout (Noah-MP off) can still read/write a land-less restart.
    from gpuwrf.contracts.noahmp_state import NoahMPLandState, NoahMPStatic
except Exception:  # pragma: no cover
    NoahMPLandState = None  # type: ignore
    NoahMPStatic = None  # type: ignore


configure_jax_x64()

FORMAT = "gpuwrf-operational-restart"
# v1: State + small-step scratch + Noah-MP land/radiation.
# v2: every current OperationalCarry field, including cadence-held physics,
# radiation diagnostics, optional land/base-state trees and census counters.
# v3: resident full-history accumulators. v4: held GWDO stress diagnostics;
# pre-GWDO-schema checkpoints fail closed (including old inactive domains).
# v5: held Noah-MP precipitation.
FORMAT_VERSION = 5
SUPPORTED_FORMAT_VERSIONS = (1, 2, 3, 4, 5)
# E78: a populated pre-cloud-fraction diagnostics tuple must not acquire None
# from the new NamedTuple default during unpickling and silently lose CLDFRA.
RADIATION_DIAGNOSTICS_SCHEMA_VERSION = 1
GWDO_DIAGNOSTICS_SCHEMA_VERSION = 2

# OperationalCarry leaves that are NOT themselves nested pytrees handled
# specially below (state / noahmp_land / noahmp_rad). These are the WRF
# small-step scratch + held radiation arrays carried between steps.
_CARRY_SCRATCH_FIELDS: tuple[str, ...] = (
    "t_2ave",
    "ww",
    "mudf",
    "muave",
    "muts",
    "ph_tend",
    "u_save",
    "v_save",
    "w_save",
    "t_save",
    "ph_save",
    "mu_save",
    "ww_save",
    "rthraten",
)

# Nested carry groups this format serializes (each as an explicit field dict).
_SUPPORTED_CARRY_GROUPS: tuple[str, ...] = ("noahmp_land", "noahmp_rad")

# Fields absent from the v1 payload. v2 stores each as an exact host pytree.
_V1_UNSUPPORTED_CARRY_FIELDS: tuple[str, ...] = tuple(
    field.name
    for field in dataclass_fields(OperationalCarry)
    if field.name != "state"
    and field.name not in _CARRY_SCRATCH_FIELDS
    and field.name not in _SUPPORTED_CARRY_GROUPS
)
UNSUPPORTED_CARRY_FIELDS: tuple[str, ...] = ()
_CARRY_FIELD_ORDER = tuple(field.name for field in dataclass_fields(OperationalCarry))


def _reject_unsupported_carry(carry: OperationalCarry) -> None:
    populated = [name for name in UNSUPPORTED_CARRY_FIELDS if getattr(carry, name) is not None]
    if populated:
        raise ValueError(
            f"restart format {FORMAT!r} v{FORMAT_VERSION} has no payload for populated "
            f"OperationalCarry fields {populated}; refusing to drop them"
        )


def _hostify(leaf: Any) -> np.ndarray:
    return np.asarray(leaf)


def _hostify_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(_hostify, tree)


def _device_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda leaf: jax.device_put(leaf), tree)


def _hostify_namelist(namelist: Any, grid: Any) -> Any:
    """Host-copy the namelist pytree, re-pinning ``grid`` by reference (matches
    ``runtime.checkpoint`` so a namelist carrying a grid handle round-trips)."""
    host = _hostify_tree(namelist)
    if is_dataclass(host) and hasattr(host, "grid"):
        return replace(host, grid=grid)
    return host


def _restore_namelist(namelist: Any, grid: Any) -> Any:
    device = _device_tree(namelist)
    if is_dataclass(device) and hasattr(device, "grid"):
        return replace(device, grid=grid)
    return device


def _state_fields(state: State) -> dict[str, np.ndarray]:
    return {name: _hostify(getattr(state, name)) for name in state.active_field_names()}


def _validate_state_field_order(recorded: tuple[str, ...]) -> None:
    expected = tuple(State.__slots__)
    if any(field not in expected for field in recorded):
        raise ValueError("restart State field order contains unknown leaves")
    if recorded != tuple(field for field in expected if field in recorded):
        raise ValueError("restart State field order does not match current State schema")
    missing = tuple(field for field in expected if field not in recorded)
    if any(field not in (*CONDITIONAL_STATE_LEAVES, *GWDO_DIAGNOSTIC_LEAVES) for field in missing):
        raise ValueError(
            "restart State fields are missing non-conditional leaves: "
            f"{[field for field in missing if field not in (*CONDITIONAL_STATE_LEAVES, *GWDO_DIAGNOSTIC_LEAVES)]}"
        )


def _land_fields(land: Any) -> dict[str, np.ndarray]:
    if NoahMPLandState is None:
        raise ValueError("NoahMPLandState unavailable; cannot serialize land carry")
    return {name: _hostify(getattr(land, name)) for name in NoahMPLandState.__slots__}


def _carry_to_payload(carry: OperationalCarry) -> dict[str, Any]:
    """Explicit field-named, host-numpy serialization of the FULL carry."""

    payload: dict[str, Any] = {
        "state_field_order": list(carry.state.active_field_names()),
        "state_fields": _state_fields(carry.state),
        "scratch_field_order": list(_CARRY_SCRATCH_FIELDS),
        "scratch_fields": {name: _hostify(getattr(carry, name)) for name in _CARRY_SCRATCH_FIELDS},
        # Noah-MP land carry (None when Noah-MP off).
        "noahmp_land_field_order": None,
        "noahmp_land_fields": None,
        # Held surface-radiation forcing (SOLDN, LWDN, COSZ) — a tuple of arrays.
        "noahmp_rad": None,
    }
    if carry.noahmp_land is not None:
        payload["noahmp_land_field_order"] = list(NoahMPLandState.__slots__)
        payload["noahmp_land_fields"] = _land_fields(carry.noahmp_land)
    if carry.noahmp_rad is not None:
        payload["noahmp_rad"] = [_hostify(component) for component in carry.noahmp_rad]
    payload["carry_field_order"] = list(_CARRY_FIELD_ORDER)
    payload["extra_fields"] = {
        name: _hostify_tree(getattr(carry, name)) for name in _V1_UNSUPPORTED_CARRY_FIELDS
    }
    payload["radiation_diagnostics_schema_version"] = RADIATION_DIAGNOSTICS_SCHEMA_VERSION
    payload["gwdo_diagnostics_schema_version"] = GWDO_DIAGNOSTICS_SCHEMA_VERSION
    return payload


def _validate_radiation_diagnostics_schema(payload: dict[str, Any]) -> None:
    if payload.get("extra_fields", {}).get("radiation_diagnostics") is None:
        return  # Legacy dycore-only restarts have no held radiation to lose.
    if payload.get("radiation_diagnostics_schema_version") != RADIATION_DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("restart radiation diagnostics schema predates held cloud_fraction (E78)")


def _validate_gwdo_diagnostics_schema(payload: dict[str, Any]) -> None:
    # An old checkpoint can omit these optional leaves even on an active GWDO
    # domain. They cannot be recomputed from its post-RK state: WRF used the
    # preceding pre-PBL inputs. Refuse rather than silently emit zeros (E78).
    if payload.get("gwdo_diagnostics_schema_version") != GWDO_DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError("restart GWDO diagnostics schema predates held DTAUX3D/DTAUY3D and DUSFCG/DVSFCG (E78)")


def write_restart(
    carry: OperationalCarry,
    namelist: Any,
    grid: Any,
    step_index: int,
    path: str | Path,
    *,
    noahmp_static: Any | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a full-carry ``wrfrst``-equivalent restart file.

    Serializes the COMPLETE :class:`OperationalCarry` (prognostic ``State`` + WRF
    small-step scratch + held ``rthraten`` + Noah-MP land carry + held
    ``noahmp_rad``) plus ``grid``, ``namelist`` and ``step_index``. ``noahmp_static``
    (the read-only per-run Noah-MP inputs / parameter tables) is optionally stored
    so a resumed run can rebuild the land driver without re-reading ``wrfinput``;
    it is host-copied like everything else. All arrays land as host numpy, so the
    write is GPU-free.
    """

    _reject_unsupported_carry(carry)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    host_grid = _hostify_tree(grid)
    payload: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "carry_type": "gpuwrf.runtime.operational_state.OperationalCarry",
        "carry": _carry_to_payload(carry),
        "namelist": _hostify_namelist(namelist, host_grid),
        "grid": host_grid,
        "step_index": int(step_index),
        "noahmp_static": None if noahmp_static is None else _hostify_static(noahmp_static),
        "metadata": dict(extra_metadata) if extra_metadata else {},
    }
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)
    directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target


def _hostify_static(static: Any) -> dict[str, Any]:
    """Host-copy NoahMPStatic field-by-field (arrays hostified; non-arrays kept).

    ``parameters`` (the table bundle) and scalars like ``dx_m`` are kept as-is; the
    field-named dict keeps the read fail-closed against a NoahMPStatic schema change.
    """
    if NoahMPStatic is None:
        raise ValueError("NoahMPStatic unavailable; cannot serialize Noah-MP static inputs")
    fields: dict[str, Any] = {}
    for name in NoahMPStatic.__slots__:
        value = getattr(static, name)
        fields[name] = _hostify_tree(value) if name != "parameters" else value
    return {"field_order": list(NoahMPStatic.__slots__), "fields": fields}


def _read_payload(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported restart format {payload.get('format')!r} in {source}")
    version = int(payload.get("format_version", -1))
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"unsupported restart version {payload.get('format_version')} in {source}; "
            f"supported {SUPPORTED_FORMAT_VERSIONS}"
        )

    carry = payload["carry"]
    _validate_radiation_diagnostics_schema(carry)
    if version < 5 and carry.get("noahmp_land_fields") is not None:
        raise ValueError("restart predates held Noah-MP precipitation (E78); cannot resume land physics")
    if version < 3:
        from gpuwrf.runtime.history_accumulators import full_history_enabled
        if full_history_enabled() and carry.get("noahmp_land_fields") is not None:
            raise ValueError("restart predates resident full-history accumulators; cannot resume full history")
    _validate_gwdo_diagnostics_schema(carry)
    if version >= 2:
        if tuple(carry.get("carry_field_order", ())) != _CARRY_FIELD_ORDER:
            raise ValueError("restart OperationalCarry field order does not match current schema")
        if set(carry.get("extra_fields", {})) != set(_V1_UNSUPPORTED_CARRY_FIELDS):
            raise ValueError("restart OperationalCarry extra fields do not match current schema")
    # Fail-closed schema checks (exact field-order + field-set match), so a State /
    # scratch / land schema drift raises instead of mis-reconstructing the carry.
    recorded_state_order = tuple(carry.get("state_field_order", ()))
    _validate_state_field_order(recorded_state_order)
    if set(carry.get("state_fields", {})) != set(recorded_state_order):
        raise ValueError("restart State fields do not match current State schema")
    if tuple(carry.get("scratch_field_order", ())) != _CARRY_SCRATCH_FIELDS:
        raise ValueError("restart carry-scratch field order does not match current schema")
    if set(carry.get("scratch_fields", {})) != set(_CARRY_SCRATCH_FIELDS):
        raise ValueError("restart carry-scratch fields do not match current schema")
    land_fields = carry.get("noahmp_land_fields")
    if land_fields is not None:
        if NoahMPLandState is None:
            raise ValueError("restart carries Noah-MP land state but NoahMPLandState is unavailable")
        if tuple(carry.get("noahmp_land_field_order", ())) != tuple(NoahMPLandState.__slots__):
            raise ValueError("restart Noah-MP land field order does not match current schema")
        if set(land_fields) != set(NoahMPLandState.__slots__):
            raise ValueError("restart Noah-MP land fields do not match current schema")
    return payload


def _restore_carry(carry_payload: dict[str, Any]) -> OperationalCarry:
    # Verbatim pytree rebuild (``tree_unflatten``), NOT ``State(**fields)``: the
    # constructor re-canonicalises leaves to the fp32-gated default matrix and
    # would silently DOWNCAST the fp64 boundary/number leaves a ``force_fp64``
    # operational run carries.  The fail-closed field-order check above already
    # guarantees every non-conditional leaf is present; conditional leaves absent
    # from the payload stay ``None`` exactly as they were on the written State.
    stored = {
        name: jax.device_put(carry_payload["state_fields"][name])
        for name in carry_payload["state_field_order"]
    }
    state = State.tree_unflatten(None, tuple(stored.get(name) for name in State.__slots__))
    scratch = {
        name: jax.device_put(carry_payload["scratch_fields"][name])
        for name in _CARRY_SCRATCH_FIELDS
    }
    noahmp_land = None
    if carry_payload.get("noahmp_land_fields") is not None:
        noahmp_land = NoahMPLandState(
            **{
                name: jax.device_put(carry_payload["noahmp_land_fields"][name])
                for name in NoahMPLandState.__slots__
            }
        )
    noahmp_rad = None
    if carry_payload.get("noahmp_rad") is not None:
        noahmp_rad = tuple(jax.device_put(component) for component in carry_payload["noahmp_rad"])
    extra = {name: _device_tree(value) for name, value in carry_payload.get("extra_fields", {}).items()}
    if extra.get("land_history") is not None or extra.get("energy_accumulators") is not None:
        # Checkpoints written before the packed history families hold per-name dicts.
        from gpuwrf.runtime.history_accumulators import ENERGY_ACCUMULATORS, LAND_FLUX_FIELDS, as_packed
        extra["land_history"] = as_packed(extra.get("land_history"), LAND_FLUX_FIELDS)
        extra["energy_accumulators"] = as_packed(extra.get("energy_accumulators"), ENERGY_ACCUMULATORS)
    return OperationalCarry(
        state=state,
        noahmp_land=noahmp_land,
        noahmp_rad=noahmp_rad,
        **scratch,
        **extra,
    )


def read_restart(path: str | Path) -> tuple[OperationalCarry, Any, Any, int]:
    """Read a full-carry restart and reconstruct the carry pytree on device.

    Returns ``(carry, namelist, grid, step_index)``. The carry is the COMPLETE
    :class:`OperationalCarry` (State + scratch + held rthraten + Noah-MP land/rad)
    so the operational driver can resume the scan with no re-seeding.
    """
    payload = _read_payload(path)
    carry = _restore_carry(payload["carry"])
    grid = _device_tree(payload["grid"])
    namelist = _restore_namelist(payload["namelist"], grid)
    return carry, namelist, grid, int(payload["step_index"])


def read_restart_metadata(path: str | Path) -> dict[str, Any]:
    """Read only the (host) provenance/metadata block + schema header.

    Lets a manager/inspector check a restart's step index, version and carry
    coverage WITHOUT materializing arrays on a device (GPU-free).
    """
    payload = _read_payload(path)
    carry = payload["carry"]
    return {
        "format": payload["format"],
        "format_version": int(payload["format_version"]),
        "step_index": int(payload["step_index"]),
        "has_noahmp_land": carry.get("noahmp_land_fields") is not None,
        "has_noahmp_rad": carry.get("noahmp_rad") is not None,
        "has_noahmp_static": payload.get("noahmp_static") is not None,
        "state_field_count": len(carry.get("state_fields", {})),
        "scratch_field_count": len(carry.get("scratch_fields", {})),
        "metadata": payload.get("metadata", {}),
    }


__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "SUPPORTED_FORMAT_VERSIONS",
    "UNSUPPORTED_CARRY_FIELDS",
    "read_restart",
    "read_restart_metadata",
    "write_restart",
]
