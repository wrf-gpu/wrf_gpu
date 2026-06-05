"""HLO launch-count PROXY for Wave-A (nsys-free, deterministic).

Compiles a ONE-step coupled ``_advance_chunk`` and counts instructions in the
OPTIMIZED (post-fusion) HLO module. The absolute number is not a kernel count,
but the DELTA across a fusion change tracks the launch-count drop (fewer fusion
roots / copies / converts / transposes => fewer launches).

We count instructions on the optimized HLO module via the compiled executable's
``hlo_modules()`` (instruction-level), which avoids text-regex fragility.

Run:
  PYTHONPATH=src GPUWRF_CANAIRY_ROOT=/mnt/data/canairy_meteo OMP_NUM_THREADS=4 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TF_GPU_ALLOCATOR=cuda_malloc_async taskset -c 0-3 \
    python proofs/v0100/wave_a_hlo_proxy.py --out wave_a_before_hlo.json
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import re
from pathlib import Path

import jax
import jax.numpy as jnp

from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import _advance_chunk, _enforce_operational_precision
from gpuwrf.runtime.operational_state import initial_operational_carry

PROOF = Path("proofs/v0100")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="wave_a_before_hlo.json")
    ap.add_argument("--cadence", type=int, default=180)
    args = ap.parse_args()
    cadence = int(args.cadence)
    acoustic_unroll = int(os.environ.get("GPUWRF_ACOUSTIC_UNROLL", "1"))

    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, _ = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist, run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=cadence, time_utc=case.run_start,
    )
    carry0 = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64))
    )

    # One-step compiled chunk (no radiation: step 1, cadence 180 -> no rad).
    def one_step(c, n):
        return _advance_chunk(c, n, jnp.asarray(1, dtype=jnp.int32), n_steps=1, cadence=cadence)

    jitted = jax.jit(one_step)
    lowered = jitted.lower(carry0, nl)
    compiled = lowered.compile()

    # Optimized HLO text (post-fusion) -- robust instruction taxonomy via regex on
    # the opcode token at the start of each instruction line.
    text = compiled.as_text()
    opcodes = collections.Counter()
    total = 0
    # HLO lines look like:  %name = type opcode(operands)  OR  ROOT %name = ...
    op_re = re.compile(r"=\s*\S+\s+([a-zA-Z0-9_\-]+)\(")
    for line in text.splitlines():
        m = op_re.search(line)
        if m:
            opcodes[m.group(1)] += 1
            total += 1
    # Buckets of interest for the launch proxy.
    buckets = {
        "fusion": opcodes.get("fusion", 0),
        "copy": opcodes.get("copy", 0) + opcodes.get("copy-start", 0) + opcodes.get("copy-done", 0),
        "transpose": opcodes.get("transpose", 0),
        "convert": opcodes.get("convert", 0),
        "bitcast": opcodes.get("bitcast", 0),
        "dynamic-update-slice": opcodes.get("dynamic-update-slice", 0),
        "pad": opcodes.get("pad", 0),
        "concatenate": opcodes.get("concatenate", 0),
        "custom-call": opcodes.get("custom-call", 0),
        "reduce": opcodes.get("reduce", 0),
        "while": opcodes.get("while", 0),
        "total_instructions": total,
    }
    out = {
        "scope": "Wave-A HLO launch-count proxy (optimized one-step coupled chunk)",
        "config": {"force_fp64": bool(nl.force_fp64), "acoustic_unroll": acoustic_unroll,
                   "radiation_cadence_steps": cadence},
        "buckets": buckets,
        "top_opcodes": dict(opcodes.most_common(25)),
    }
    PROOF.mkdir(parents=True, exist_ok=True)
    fn = PROOF / args.out
    fn.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    print(f"wrote {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
