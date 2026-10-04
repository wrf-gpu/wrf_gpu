#!/usr/bin/env python3
"""Companion check: the PRODUCTION adapter path cannot reproduce the sealed
sp2 baseline tendencies from the sealed capture state.

Runs the exact operational entry point ``mynn_adapter_with_source_leaves``
(the path production uses: ``step_mynn_pbl_column`` -> ``_tiled_mynn_step``,
``edmf=_MYNN_EDMF``, ``dx=_mynn_dx(grid)``, ``first_timestep=True``) on the
sealed ac671 capture state and compares its rublten/rvblten against

  (a) the sealed bundle's production sp2_rublten/sp2_rvblten, and
  (b) the tendency implied by the sealed (bit-exactly replayed) solve arrays,
      (solve_x - u_replay)/dt.

Expected result (sealing the authority-mismatch finding): (b) matches to
float64 identity while (a) differs by ~2.5e-4 RMS — i.e. replay==adapter,
and the sealed bundle's baseline tendencies belong to a different authority
than its coefficient arrays.

CPU backend only; no GPU discovery; no WRF/MPI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-kimi-pbl-solve-discriminator")
BRIEF_ROOT = Path("<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_ac6712170cbe5084")
CAPTURE_ROOT = BRIEF_ROOT / "capture/authentic-ac6712170cbe5084"
INPUT_DIR = Path("<DATA_ROOT>/wrf_gpu2/v0234_pristine_pbl_entry_closure_0b18530a1dc9cac2/capture-inputs")
ALLOWED_CPUS = {13, 14, 15, 29, 30, 31}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operands", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        print("REFUSE: output exists", file=sys.stderr)
        return 74
    if not set(os.sched_getaffinity(0)) <= ALLOWED_CPUS:
        print("REFUSE: affinity outside allowed set", file=sys.stderr)
        return 74

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path.insert(0, str(REPO / "src"))
    import jax
    from gpuwrf.contracts.state import State
    from gpuwrf.coupling import physics_couplers as couplers
    from gpuwrf.io.gen2_accessor import Gen2Run

    if jax.default_backend() != "cpu" or any(d.platform != "cpu" for d in jax.devices()):
        print("REFUSE: non-CPU backend", file=sys.stderr)
        return 74

    manifest = json.loads((CAPTURE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    values = [
        None if item["kind"] == "none"
        else jax.device_put(np.load(CAPTURE_ROOT / item["file"], allow_pickle=False))
        for item in manifest["state_leaves"]
    ]
    state = State.tree_unflatten(None, tuple(values))
    grid = Gen2Run(INPUT_DIR).grid("d03").as_grid_spec()
    leaves = couplers.mynn_adapter_with_source_leaves(state, 6.0, grid, first_timestep=True)

    sealed = np.load(BRIEF_ROOT / "cpu-reference/pbl-sp2-reference-28.npz", allow_pickle=False)
    ops = np.load(Path(args.operands), allow_pickle=False)

    proof: dict = {
        "schema": "wrfgpu2-v0234-fable-pbl-adapter-reproduction-v1",
        "backend": "cpu",
        "gpu_actions": 0,
        "adapter_entry": "mynn_adapter_with_source_leaves(first_timestep=True)",
        "edmf": bool(couplers._MYNN_EDMF),
        "dx": float(couplers._mynn_dx(grid)),
    }
    for comp, field in (("u", "rublten"), ("v", "rvblten")):
        mine = np.ascontiguousarray(np.asarray(getattr(leaves, field), np.float64))
        sealed_sp2 = np.asarray(sealed[f"sp2_r{comp}blten"])
        implied = (np.asarray(sealed[f"solve_{comp}_x"]) - np.asarray(ops[comp])) / 6.0
        proof[field] = {
            "adapter_vs_sealed_sp2_max_abs": float(np.max(np.abs(mine - sealed_sp2))),
            "adapter_vs_sealed_sp2_rms": float(np.sqrt(np.mean((mine - sealed_sp2) ** 2))),
            "adapter_vs_replay_implied_max_abs": float(np.max(np.abs(mine - implied))),
            "adapter_vs_replay_implied_bit_exact": bool(
                np.array_equal(mine.view(np.uint64), implied.view(np.uint64))
            ),
        }
    canonical = json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    proof["canonical_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.write_text(json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: proof[k] for k in ("rublten", "rvblten")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
