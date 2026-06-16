#!/usr/bin/env python3
"""v0.17 hail microphysics GPU operational smoke (WSM7 mp=24, WDM7 mp=26).

Runs the EXACT operational per-step physics block the GPU scan body executes
(``runtime.operational_mode._physics_step_forcing`` -> dispatcher-selected mp
adapter -> ...), on the GPU, for the two v0.17 hail schemes. Proves the
scan-wired ``wsm7_adapter`` / ``wdm7_adapter`` LOWER and RUN on the device and
produce finite, mutated output (truly ran, not a silent no-op), and that the
new ``hail_acc`` accumulator + ``qh`` leaf live on the GPU.

This is the GPU counterpart of tests/test_v013_operational_smoke.py
(test_microphysics_operational_runs_and_mutates), which is CPU-by-design. Run
under the shared GPU lock:

  bash scripts/with_gpu_lock.sh --label opus-hail -- \
    env JAX_PLATFORMS=cuda python proofs/v013_wdm7/gpu_hail_operational_smoke.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import jax  # noqa: E402
import numpy as np  # noqa: E402

jax.config.update("jax_enable_x64", True)

# Reuse the operational-smoke builders verbatim (same idealized columns the CPU
# functional gate uses) so the only difference is the device.
from test_v013_operational_smoke import (  # noqa: E402
    _all_finite,
    _base_state,
    _changed,
    _grid,
    _namelist,
)
from gpuwrf.runtime.operational_mode import _physics_step_forcing  # noqa: E402
from gpuwrf.runtime.operational_state import initial_operational_carry  # noqa: E402
from gpuwrf.runtime.operational_mode import _resolve_operational_suite  # noqa: E402


def _platform(state) -> str:
    return next(iter(state.theta.devices())).platform


def run_one(mp: int, name: str) -> bool:
    grid = _grid()
    state = _base_state(grid)
    nml = _namelist(grid, mp_physics=mp, bl_pbl_physics=0, sf_sfclay_physics=0, cu_physics=0)
    _resolve_operational_suite(nml)  # operational fail-closed authority must accept it
    carry = initial_operational_carry(state)
    forcing = _physics_step_forcing(carry, nml, 0.0, run_radiation=False)
    after = forcing.state
    plat = _platform(after)
    finite = _all_finite(after)
    qv_changed = _changed(after.qv, state.qv)
    qh_finite = bool(np.all(np.isfinite(np.asarray(after.qh))))
    hail_acc_finite = bool(np.all(np.isfinite(np.asarray(after.hail_acc))))
    ok = finite and qv_changed and qh_finite and hail_acc_finite and plat == "gpu"
    print(f"=== {name} (mp_physics={mp}) on {plat} -> {'PASS' if ok else 'FAIL'} ===")
    print(f"    all_finite={finite}  qv_mutated={qv_changed}  "
          f"qh_finite={qh_finite}  hail_acc_finite={hail_acc_finite}")
    print(f"    qh max={float(np.max(np.abs(np.asarray(after.qh)))):.3e}  "
          f"hail_acc max={float(np.max(np.abs(np.asarray(after.hail_acc)))):.3e}  "
          f"theta range=[{float(np.min(after.theta)):.2f},{float(np.max(after.theta)):.2f}]K")
    return ok


def main() -> int:
    devs = jax.devices()
    print("JAX devices:", devs)
    if not any(d.platform == "gpu" for d in devs):
        print("NO GPU BACKEND -- this smoke requires a GPU (run under with_gpu_lock.sh / JAX_PLATFORMS=cuda).")
        return 2
    ok24 = run_one(24, "WSM7")
    ok26 = run_one(26, "WDM7")
    allok = ok24 and ok26
    print("OVERALL:", "PASS" if allok else "FAIL")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
