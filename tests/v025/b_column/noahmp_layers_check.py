"""Noah-MP layer-select checks on real PROD inputs (b-noahmp NP06 columns), CPU.

mode ``hlo``: lower noah_mp_step (night_d01) with the environment's flags and
hash raw HLO text (+ module proto) -> compare base vs candidate (same path).
mode ``eager``: GPUWRF_NOAHMP_NATIVE_REAL=1, eager (no fusion) step with
GPUWRF_NOAHMP_LAYER_SELECT=0 vs 1 -> every output leaf must be bitwise equal.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

assert os.environ.get("JAX_PLATFORMS") == "cpu"
assert not Path("/tmp/wrf_gpu2_quiet").exists(), "quiet"
import jax  # noqa: E402
import numpy as np  # noqa: E402

INPUTS = Path("<USER_HOME>/wrf_gpu2_lanes/b-noahmp/NP06/real_columns/columns.pkl")


def _case(name):
    from gpuwrf.physics.noahmp.precision import real_tree
    land, forcing, static, ep, rp, dt = pickle.load(INPUTS.open("rb"))[name]
    land, forcing, static, ep, rp = jax.tree.map(  # as the product / b-noahmp component probe
        lambda x: jax.device_put(x) if hasattr(x, "dtype") else x, (land, forcing, static, ep, rp))
    land, forcing, static, ep, rp = real_tree((land, forcing, static, ep, rp))
    return land, forcing, static, ep, rp, dt


def mode_hlo(out):
    from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
    land, forcing, static, ep, rp, dt = _case("night_d01")
    lowered = jax.jit(lambda ls, ff: noah_mp_step(ls, ff, static, dt, energy_params=ep,
                                                  rad_params=rp)).lower(land, forcing)
    hlo = lowered.compiler_ir("hlo")
    text = hlo.as_hlo_text()
    rec = dict(flags={k: v for k, v in os.environ.items() if k.startswith("GPUWRF_")},
               text=hashlib.sha256(text.encode()).hexdigest(),
               proto=hashlib.sha256(hlo.as_serialized_hlo_module_proto()).hexdigest(),
               dus=text.count("dynamic-update-slice("), whiles=text.count(" while("),
               scatter=text.count(" scatter("), select=text.count(" select("))
    Path(out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


def mode_eager(out):
    os.environ["GPUWRF_NOAHMP_NATIVE_REAL"] = "1"
    from gpuwrf.physics.noahmp.noahmp_driver import noah_mp_step
    rec = {}
    for name in ("night_d01", "day_d02"):
        land, forcing, static, ep, rp, dt = _case(name)
        res = {}
        for flag in ("0", "1"):
            os.environ["GPUWRF_NOAHMP_LAYER_SELECT"] = flag
            with jax.disable_jit():
                res[flag] = jax.tree.leaves(noah_mp_step(land, forcing, static, dt,
                                                         energy_params=ep, rad_params=rp))
        bad = [i for i, (a, b) in enumerate(zip(res["0"], res["1"]))
               if not np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)]
        rec[name] = dict(leaves=len(res["0"]), differing=bad)
        print(name, rec[name], flush=True)
    Path(out).write_text(json.dumps(rec, indent=1) + "\n")


if __name__ == "__main__":
    {"hlo": mode_hlo, "eager": mode_eager}[sys.argv[1]](sys.argv[2])
