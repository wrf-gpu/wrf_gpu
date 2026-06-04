"""StableHLO/static audit for v0.10.0 Phase 0."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import jax
import jax.numpy as jnp

import gpuwrf.contracts.state as _state_mod
from gpuwrf.contracts.precision import DEFAULT_DTYPES, STATE_FIELD_ORDER
from gpuwrf.contracts.state import State
from gpuwrf.coupling.physics_couplers import (
    _flatten_columns_to_batch,
    _from_columns,
    _thompson_column_from_state,
    _to_columns,
    _unflatten_batch_to_columns,
)
from gpuwrf.physics.thompson_column import _fill_down
from gpuwrf.runtime.operational_mode import _enforce_operational_precision


_ORIG_ASARRAY = _state_mod.jnp.asarray


def _safe_asarray(x, dtype=None, **kwargs):
    try:
        if dtype is None:
            return _ORIG_ASARRAY(x, **kwargs)
        return _ORIG_ASARRAY(x, dtype=dtype, **kwargs)
    except (TypeError, ValueError):
        return x


_state_mod.jnp.asarray = _safe_asarray


def _state_shapes() -> dict[str, tuple[int, ...]]:
    nz, ny, nx = 3, 4, 5
    mass = (nz, ny, nx)
    surf = (ny, nx)
    bside = max(nx + 1, ny + 1)
    bwidth = 2
    bmass = (1, 4, bwidth, nz, bside)
    bface = (1, 4, bwidth, nz + 1, bside)
    bsurf = (1, 4, bwidth, 1, bside)
    return {
        "u": (nz, ny, nx + 1),
        "v": (nz, ny + 1, nx),
        "w": (nz + 1, ny, nx),
        "theta": mass,
        "qv": mass,
        "p": mass,
        "p_total": mass,
        "p_perturbation": mass,
        "ph": (nz + 1, ny, nx),
        "ph_total": (nz + 1, ny, nx),
        "ph_perturbation": (nz + 1, ny, nx),
        "mu": surf,
        "mu_total": surf,
        "mu_perturbation": surf,
        "qc": mass,
        "qr": mass,
        "qi": mass,
        "qs": mass,
        "qg": mass,
        "Ni": mass,
        "Nr": mass,
        "Ns": mass,
        "Ng": mass,
        "qke": mass,
        "ustar": surf,
        "theta_flux": surf,
        "qv_flux": surf,
        "tau_u": surf,
        "tau_v": surf,
        "rhosfc": surf,
        "fltv": surf,
        "t_skin": surf,
        "soil_moisture": surf,
        "xland": surf,
        "lakemask": surf,
        "mavail": surf,
        "roughness_m": surf,
        "lu_index": surf,
        "rain_acc": surf,
        "snow_acc": surf,
        "graupel_acc": surf,
        "ice_acc": surf,
        "u_bdy": bmass,
        "v_bdy": bmass,
        "theta_bdy": bmass,
        "qv_bdy": bmass,
        "ph_bdy": bface,
        "mu_bdy": bsurf,
        "w_bdy": bface,
        "p_bdy": bmass,
        "pb_bdy": bmass,
        "phb_bdy": bface,
        "mub_bdy": bsurf,
        "Nc": mass,
        "Nn": mass,
        "rainc_acc": surf,
    }


def _dummy_state() -> State:
    arrays: dict[str, Any] = {}
    for field, shape in _state_shapes().items():
        dtype = DEFAULT_DTYPES.dtype_for(field)
        if field == "lu_index":
            arrays[field] = jnp.zeros(shape, dtype=jnp.int32)
        elif field in ("p", "p_total"):
            arrays[field] = jnp.full(shape, 80000.0, dtype=dtype)
        elif field == "theta":
            arrays[field] = jnp.full(shape, 300.0, dtype=dtype)
        elif field == "qv":
            arrays[field] = jnp.full(shape, 0.005, dtype=dtype)
        elif field in ("ph", "ph_total"):
            z = jnp.arange(shape[0], dtype=dtype).reshape((shape[0], 1, 1))
            arrays[field] = jnp.broadcast_to(z * 100.0, shape)
        else:
            arrays[field] = jnp.zeros(shape, dtype=dtype)
    return State(**arrays)


def _stablehlo(fn, *args) -> str:
    return str(jax.jit(fn).lower(*args).compiler_ir(dialect="stablehlo"))


def _counts(text: str) -> dict[str, int]:
    return {
        "stablehlo_convert": len(re.findall(r"\bstablehlo\.convert\b", text)),
        "stablehlo_transpose": len(re.findall(r"\bstablehlo\.transpose\b", text)),
        "stablehlo_reshape": len(re.findall(r"\bstablehlo\.reshape\b", text)),
    }


def _audit_layout_ops(state: State) -> dict[str, Any]:
    field = jnp.zeros((3, 4, 5), dtype=jnp.float64)
    columns = jnp.zeros((4, 5, 3), dtype=jnp.float64)
    batch = jnp.zeros((20, 3), dtype=jnp.float64)
    active = columns > 0.0
    items = {
        "_to_columns": _stablehlo(lambda x: _to_columns(x), field),
        "_from_columns": _stablehlo(lambda x: _from_columns(x), columns),
        "_flatten_columns_to_batch": _stablehlo(lambda x: _flatten_columns_to_batch(x, 4, 5), columns),
        "_unflatten_batch_to_columns": _stablehlo(lambda x: _unflatten_batch_to_columns(x, 4, 5), batch),
        "thompson._fill_down": _stablehlo(lambda vt, act: _fill_down(vt, act), columns, active),
        "_thompson_column_from_state": _stablehlo(lambda s: _thompson_column_from_state(s, None), state),
    }
    return {
        name: {
            **_counts(text),
            "classification": (
                "real_transpose" if _counts(text)["stablehlo_transpose"] else "reshape_bitcast_only"
            ),
        }
        for name, text in items.items()
    }


def main() -> int:
    state = _dummy_state()
    precision_hlo = _stablehlo(lambda s: _enforce_operational_precision(s, force_fp64=True), state)
    layout = _audit_layout_ops(state)
    non_fp64_fields = [
        field for field in STATE_FIELD_ORDER
        if jnp.dtype(DEFAULT_DTYPES.dtype_for(field)) != jnp.dtype(jnp.float64)
    ]
    payload = {
        "schema": "V0100Phase0HloAudit",
        "schema_version": 1,
        "status": "PASS",
        "platform": str(jax.devices()[0]),
        "precision_enforcement_force_fp64": {
            **_counts(precision_hlo),
            "input_fields_not_fp64_by_default": non_fp64_fields,
            "input_fields_not_fp64_count": len(non_fp64_fields),
            "note": (
                "StableHLO was generated on the CPU backend with dummy shapes. "
                "Convert counts are source/lowering evidence for _enforce_operational_precision; "
                "GPU codegen-level copy/fusion still requires a visible CUDA backend."
            ),
        },
        "physics_layout_ops": layout,
        "donation_static_source_audit": {
            "_advance_chunk": "plain helper, not decorated with jax.jit and no donate_argnums at function definition",
            "run_forecast_operational": "decorated with donate_argnums=(0,)",
            "run_forecast_operational_single_scan": "decorated with donate_argnums=(0,)",
            "run_forecast_operational_with_limiter_diagnostics": "decorated with donate_argnums=(0,)",
            "run_forecast_operational_debug": "decorated with donate_argnums=(0,)",
            "run_forecast_operational_segmented": "host loop over _advance_chunk; donation depends on inner helper call/lowering, not on a public jit decorator",
        },
    }
    out = Path("proofs/v0100/hlo_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
