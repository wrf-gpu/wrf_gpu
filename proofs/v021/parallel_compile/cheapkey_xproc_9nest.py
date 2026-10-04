"""CPU two-process repro at max_dom=9: build the REAL 9-domain nest (same path the
canary/nested_pipeline uses), then dump EVERY cheap_key subcomponent per domain.

NO GPU, NO forecast, NO lowering -- the cheap_key is metadata-only. Run this twice
(process A and B) with DIFFERENT PYTHONHASHSEED and diff the per-domain subcomponent
dumps to localize the process-nondeterministic determinant.

Usage:
    JAX_PLATFORMS=cpu PYTHONHASHSEED=0 python cheapkey_xproc_9nest.py A.json
    JAX_PLATFORMS=cpu PYTHONHASHSEED=1 python cheapkey_xproc_9nest.py B.json
"""
import os, sys, json

# CPU only -- the cheap_key derivation never touches the device.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("GPUWRF_NESTED_AOT", "1")
os.environ.setdefault("JAX_ENABLE_X64", "true")

from pathlib import Path
import jax
import jax.numpy as jnp

# METADATA-ONLY repro: the cheap_key derivation never touches the device, but the
# carry CONSTRUCTORS (State.zeros, _commit_to_operational_device) hard-require a GPU.
# Shim the centralized GPU-device check to the CPU device so we can build the real
# 9-domain carries on a GPU-less host. The placement of an uncommitted single-device
# leaf maps to the SAME "default" placement_class regardless of cpu-vs-gpu (see
# aot_cheap_key._placement_class), so the carry_aval_hash is representative.
import gpuwrf.contracts.state as _state_mod
_cpu_dev = jax.devices("cpu")[0]
_state_mod._gpu_device = lambda: _cpu_dev  # type: ignore[assignment]

from gpuwrf.integration.nested_pipeline import (
    _load_domains, NestedPipelineConfig, domain_names_for,
)
from gpuwrf.runtime.operational_mode import build_clock_base, _advance_chunk_fori
from gpuwrf.runtime import aot_cheap_key as ck
from gpuwrf.runtime.aot_executable import _flatten_call

OUT = sys.argv[1] if len(sys.argv) > 1 else "9nest_components.json"
INPUT = os.environ.get("CANARY_INPUT_DIR", "<DATA_ROOT>/wrf_downscale/runs/20240901/cpu")
MAXDOM = int(os.environ.get("CANARY_MAXDOM", "9"))

cfg = NestedPipelineConfig(
    input_dir=Path(INPUT),
    output_dir=Path("/tmp/ck9_out"),
    proof_dir=Path("/tmp/ck9_proof"),
    hours=1,
    max_dom=MAXDOM,
    scratch_dir=Path("/tmp/ck9_scratch"),
)
names = domain_names_for(MAXDOM)
hierarchy, bundles, meta, run_start, dt_by_domain, initial_carries = _load_domains(cfg, names)

result = {
    "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
    "devices": str(jax.devices()),
    # Process-GLOBAL components (same for every domain).
    "program_config_hash": ck.program_config_hash(),
    "fn_identity_hash": ck.fn_identity_hash(_advance_chunk_fori),
    "source_fingerprint_hash": ck.source_fingerprint_hash(),
    "global_trace_env_hash": ck.global_trace_env_hash(),
    "module_const_env_hash": ck.module_const_env_hash(),
    "exec_env_hash": ck.exec_env_hash(),
    "domains": {},
}

for name in names:
    bundle = bundles[name]
    carry = initial_carries[name]
    namelist = bundle.namelist
    start = jnp.asarray(1, dtype=jnp.int32)
    cadence = int(namelist.radiation_cadence_steps)
    clock_base = build_clock_base(namelist)
    n_steps = 1
    args = (carry, namelist, start, clock_base)
    kwargs = {"n_steps": n_steps, "cadence": cadence}

    flat, _ = _flatten_call(args, kwargs)
    leaves = [ck._leaf_metaty_record(l) for l in flat]

    d = {
        "static_config_hash": ck.static_config_hash(namelist),
        "carry_aval_hash": ck.carry_aval_hash(args, kwargs),
        "program_key": ck.program_key(_advance_chunk_fori, args, kwargs, namelist),
        "exec_key": ck.exec_key(_advance_chunk_fori, args, kwargs, namelist),
        "cheap_key": ck.cheap_key(_advance_chunk_fori, args, kwargs, namelist),
        "n_leaves": len(flat),
        # placement classes seen (the metaty-projected determinant carry_aval hashes)
        "placement_classes": sorted({str(r.get("placement_class")) for r in leaves}),
        "committed_set": sorted({str(r.get("committed")) for r in leaves}),
        "is_jax_array_set": sorted({str(r.get("is_jax_array")) for r in leaves}),
        "cadence": cadence,
    }
    result["domains"][name] = d
    print(f"{name}: cheap_key={d['cheap_key'][:16] if d['cheap_key'] else None} "
          f"static={d['static_config_hash'][:12]} aval={d['carry_aval_hash'][:12]}",
          flush=True)

Path(OUT).write_text(json.dumps(result, indent=2))
print(f"WROTE {OUT}", flush=True)
