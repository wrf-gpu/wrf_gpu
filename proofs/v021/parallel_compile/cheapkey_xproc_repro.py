"""Two-process cheap-key cross-process warm-load repro (REAL paths).

Usage:
    python xproc_repro.py serialize <cachedir>   # PROCESS A: compile+serialize
    python xproc_repro.py load <cachedir>         # PROCESS B: fresh load by cheap_key
"""
import os
import sys
import json

CACHE = sys.argv[2]
MODE = sys.argv[1]

# Ensure a deterministic, shared cache dir + AOT on, before importing gpuwrf.
os.environ["GPUWRF_JAX_CACHE_DIR"] = CACHE
os.environ["GPUWRF_NESTED_AOT"] = "1"
# Honour an externally-set JAX_PLATFORMS; default to whatever JAX picks (GPU on
# the workstation). Do NOT force cpu here -- the real bug is GPU-backend-specific.

import gpuwrf  # noqa: F401  configures x64 + cache
import numpy as np
import jax
import jax.numpy as jnp

from gpuwrf.contracts.grid import GridSpec
from gpuwrf.contracts.precision import DEFAULT_DTYPES
from gpuwrf.contracts.state import State, Tendencies, _state_field_shapes
from gpuwrf.runtime.operational_mode import (
    OperationalNamelist,
    _initial_carry_for_run,
    build_clock_base,
    _advance_chunk_fori,
)
from gpuwrf.runtime import aot_cheap_key as ck
from gpuwrf.runtime import aot_precompile as aotp
from gpuwrf.runtime import aot_executable as aotx
from gpuwrf.runtime.compile_cache import resolve_cache_dir, configure_compilation_cache


def build_call(time_utc="2024-09-01_00:00:00"):
    grid = GridSpec.canary_3km_template()
    shapes = _state_field_shapes(grid)
    fields = {
        f: jnp.asarray(np.zeros(s), dtype=DEFAULT_DTYPES.dtype_for(f))
        for f, s in shapes.items()
    }
    state = State(**fields)
    sk = {"p": "p_total", "ph": "ph_total", "mu": "mu_total"}
    tend = Tendencies(
        **{
            k: jnp.zeros(shapes[sk.get(k, k)], dtype=DEFAULT_DTYPES.dtype_for(k))
            for k in ("u", "v", "w", "theta", "qv", "p", "ph", "mu")
        }
    )
    namelist = OperationalNamelist(
        grid=grid,
        tendencies=tend,
        metrics=grid.metrics,
        dt_s=10.0,
        acoustic_substeps=6,
        time_utc=time_utc,
    )
    carry = _initial_carry_for_run(state, namelist)
    clock_base = build_clock_base(namelist)
    return carry, namelist, clock_base


def main():
    configure_compilation_cache()
    cache_dir_resolved = resolve_cache_dir()
    print(f"[{MODE}] devices = {jax.devices()}", file=sys.stderr)
    print(f"[{MODE}] resolved cache dir = {cache_dir_resolved}", file=sys.stderr)

    carry, namelist, clock_base = build_call()
    start = jnp.asarray(1, dtype=jnp.int32)
    n_steps, cadence = 1, int(namelist.radiation_cadence_steps)

    cheap = ck.cheap_key(
        _advance_chunk_fori,
        (carry, namelist, start, clock_base),
        {"n_steps": int(n_steps), "cadence": int(cadence)},
        namelist,
    )
    print(f"[{MODE}] cheap_key = {cheap}", file=sys.stderr)

    name = "d01"
    if MODE == "serialize":
        lowered = _advance_chunk_fori.lower(
            carry, namelist, start, clock_base, n_steps=n_steps, cadence=cadence
        )
        hlo = aotx.hlo_sha256_from_lowered(lowered)
        compiled = lowered.compile()
        ser = aotp._serialize_domain_blob(
            name,
            compiled,
            None,  # cache_dir=None -> resolve_cache_dir() (same as load)
            hlo_sha256=hlo,
            lowered=lowered,
            cheap_key=cheap,
            key_schema=ck.KEY_SCHEMA,
        )
        print("SERIALIZE_RESULT " + json.dumps({k: str(v) for k, v in ser.items()}))
        # Reference execution via the freshly compiled executable (the cold path).
        out = compiled(carry, namelist, start, clock_base, n_steps=n_steps, cadence=cadence)
        import hashlib as _h
        flat = jax.tree_util.tree_leaves(out)
        dig = _h.sha256()
        for leaf in flat:
            dig.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())
        print("EXEC_DIGEST " + dig.hexdigest())
        paths = aotp._aot_blob_paths(name, None, cheap_key=cheap)
        bp, mp = paths
        print(f"[serialize] blob exists={bp.is_file()} bytes={bp.stat().st_size if bp.is_file() else 0}", file=sys.stderr)
        print(f"[serialize] meta exists={mp.is_file()} bytes={mp.stat().st_size if mp.is_file() else 0}", file=sys.stderr)
        print(f"[serialize] blob_path={bp}", file=sys.stderr)
    elif MODE == "load":
        # Show what the loader WILL look for, and whether those files exist now.
        paths = aotp._aot_blob_paths(name, None, cheap_key=cheap)
        bp, mp = paths
        print(f"[load] will look for blob={bp}", file=sys.stderr)
        print(f"[load] blob is_file()={bp.is_file()}  meta is_file()={mp.is_file()}", file=sys.stderr)
        call, status = aotp.load_domain_blob(
            name, cheap_key=cheap, return_status=True
        )
        print("LOAD_STATUS " + json.dumps({k: str(v) for k, v in status.items()}))
        if call is not None:
            out = call(carry, namelist, start, clock_base, n_steps=n_steps, cadence=cadence)
            import hashlib as _h
            flat = jax.tree_util.tree_leaves(out)
            dig = _h.sha256()
            for leaf in flat:
                dig.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())
            print("EXEC_DIGEST " + dig.hexdigest())


if __name__ == "__main__":
    main()
