"""Bit-identity probe: run the idealized warm-bubble + Straka cases to completion
through the unified operational dycore and emit byte-exact field checksums.

f=0 for idealized cases means the new large_step_coriolis term must be identically
zero, so the final state must match the f-free base BYTE-FOR-BYTE.  Run this on the
coriolis branch and on the base commit; the checksums must be identical.
"""

from __future__ import annotations

import hashlib
import json
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "src")

from gpuwrf.ic_generators.idealized import (  # noqa: E402
    _build_setup,
    _initial_carry,
    _run_segment,
    build_density_current_numpy,
    build_warm_bubble_numpy,
)


def _hash(arr) -> str:
    host = np.asarray(arr, dtype=np.float64)
    return hashlib.sha256(host.tobytes()).hexdigest()


def _run(case):
    setup = _build_setup(case, require_gpu=True)
    carry = _initial_carry(setup.state)
    total = int(round(case.end_seconds / case.dt_s))
    carry = _run_segment(carry, setup.namelist, start_step=1, steps=total)
    st = carry.state
    out = {}
    for name in ("u", "v", "w", "theta", "mu_total", "ph_total", "p_total"):
        a = getattr(st, name)
        out[name] = {
            "sha256": _hash(a),
            "min": float(np.min(np.asarray(a))),
            "max": float(np.max(np.asarray(a))),
            "mean": float(np.mean(np.asarray(a))),
            "finite": bool(np.all(np.isfinite(np.asarray(a)))),
        }
    return out


def main() -> None:
    result = {
        "warm_bubble": _run(build_warm_bubble_numpy()),
        "density_current": _run(build_density_current_numpy()),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
