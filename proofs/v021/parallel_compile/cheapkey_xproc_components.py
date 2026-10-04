"""Dump every cheap_key SUBCOMPONENT for the canary call in a fresh process so we
can diff two GPU processes and find which determinant is process-UNSTABLE."""
import os, sys, json
os.environ["GPUWRF_NESTED_AOT"] = "1"

import numpy as np
import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist, _initial_carry_for_run, build_clock_base, _advance_chunk_fori,
)
from gpuwrf.runtime import aot_cheap_key as ck


def build_call():
    grid = GridSpec.canary_3km_template()
    shapes = _state_field_shapes(grid)
    fields = {f: jnp.asarray(np.zeros(s), dtype=DEFAULT_DTYPES.dtype_for(f)) for f, s in shapes.items()}
    state = State(**fields)
    sk = {"p": "p_total", "ph": "ph_total", "mu": "mu_total"}
    tend = Tendencies(**{k: jnp.zeros(shapes[sk.get(k, k)], dtype=DEFAULT_DTYPES.dtype_for(k))
                         for k in ("u", "v", "w", "theta", "qv", "p", "ph", "mu")})
    namelist = OperationalNamelist(grid=grid, tendencies=tend, metrics=grid.metrics,
                                   dt_s=10.0, acoustic_substeps=6, time_utc="2024-09-01_00:00:00")
    carry = _initial_carry_for_run(state, namelist)
    clock_base = build_clock_base(namelist)
    return carry, namelist, clock_base


carry, namelist, clock_base = build_call()
start = jnp.asarray(1, dtype=jnp.int32)
nsteps, cadence = 1, int(namelist.radiation_cadence_steps)
args = (carry, namelist, start, clock_base)
kwargs = {"n_steps": nsteps, "cadence": cadence}

comp = {
    "devices": str(jax.devices()),
    "program_config_hash": ck.program_config_hash(),
    "fn_identity_hash": ck.fn_identity_hash(_advance_chunk_fori),
    "source_fingerprint_hash": ck.source_fingerprint_hash(),
    "static_config_hash": ck.static_config_hash(namelist),
    "carry_aval_hash": ck.carry_aval_hash(args, kwargs),
    "global_trace_env_hash": ck.global_trace_env_hash(),
    "module_const_env_hash": ck.module_const_env_hash(),
    "exec_env_hash": ck.exec_env_hash(),
    "program_key": ck.program_key(_advance_chunk_fori, args, kwargs, namelist),
    "exec_key": ck.exec_key(_advance_chunk_fori, args, kwargs, namelist),
}

# Per-leaf metaty records (the suspected process-unstable source).
from gpuwrf.runtime.aot_executable import _flatten_call
flat, _ = _flatten_call(args, kwargs)
leaves = [ck._leaf_metaty_record(l) for l in flat]
comp["leaf0"] = leaves[0]
comp["leaf_committed_set"] = sorted({str(r.get("committed")) for r in leaves})
comp["leaf_sharding_set"] = sorted({str(r.get("sharding")) for r in leaves})[:3]
comp["leaf_isjax_set"] = sorted({str(r.get("is_jax_array")) for r in leaves})

print("COMPONENTS " + json.dumps(comp))
