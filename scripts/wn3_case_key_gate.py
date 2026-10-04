"""CPU gate: per-domain AOT cheap key + lowered HLO hash for one WN3 case.

Usage (JAX_PLATFORMS=cpu, PYTHONPATH=<tree>/src): case_key_gate.py CASE_DIR OUT_JSON [--no-lower]
Builds the exact _advance_chunk_fori call the nested domain tree makes per domain
(runtime carry shapes, namelist, start, build_clock_base, n_steps, cadence) and
records cheap_key, its components and the lowered-HLO sha256. Compare the JSON of
several cases: identical keys/HLO => one compile serves every case.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

import jax
import jax.numpy as jnp


def main():
    case, out = Path(sys.argv[1]), Path(sys.argv[2])
    lower = "--no-lower" not in sys.argv
    from gpuwrf.integration import nested_pipeline as P
    from gpuwrf.runtime import aot_cheap_key as CK
    from gpuwrf.runtime import aot_executable as AX
    from gpuwrf.runtime.aot_precompile import _to_shape_dtype_tree
    from gpuwrf.runtime.operational_mode import _advance_chunk_fori, build_clock_base

    assert jax.devices()[0].platform == "cpu"
    import gpuwrf.contracts.state as state_contract
    state_contract._gpu_device = lambda: jax.devices("cpu")[0]  # CPU-only key/lowering audit
    tmp = Path(tempfile.mkdtemp(prefix="ckgate_", dir=out.parent))
    cfg = P.NestedPipelineConfig(input_dir=case, output_dir=tmp / "out", proof_dir=tmp / "proof",
                                 hours=1, max_dom=3, scratch_dir=tmp / "scratch", emit_initial_history=True)
    t0 = time.time()
    hierarchy, bundles, _meta, _start, _dt, carries = P._load_domains(cfg, ("d01", "d02", "d03"))
    rec = dict(case=str(case), load_s=round(time.time() - t0, 1), domains={})
    n_steps = {"d01": 1, "d02": 1, "d03": 3}
    for name, bundle in bundles.items():
        nml = bundle.namelist
        carry = _to_shape_dtype_tree(carries[name])
        clock = _to_shape_dtype_tree(build_clock_base(nml))
        start = jnp.asarray(1, dtype=jnp.int32)
        kw = {"n_steps": n_steps[name], "cadence": int(nml.radiation_cadence_steps)}
        args = (carry, nml, start, clock)
        d = dict(cheap_key=CK.cheap_key(_advance_chunk_fori, args, kw, nml),
                 static_config_hash=CK.static_config_hash(nml),
                 carry_aval_hash=CK.carry_aval_hash(args, kw),
                 n_leaves=len(jax.tree_util.tree_leaves(args)))
        if lower:
            t1 = time.time()
            lowered = _advance_chunk_fori.lower(carry, nml, start, clock, **kw)
            d["hlo_sha256"] = AX.hlo_sha256_from_lowered(lowered)
            d["lower_s"] = round(time.time() - t1, 1)
            del lowered
        rec["domains"][name] = d
        print(name, json.dumps(d), flush=True)
    out.write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    main()
