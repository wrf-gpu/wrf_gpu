"""OFF identity (E151) of the km_opt=3 wiring: location-stripped jaxprs of the production step loop.

usage: JAX_PLATFORMS=cpu GPUWRF_FAST_DEFAULTS=1 python off_identity.py <src_root> <out.json>
Real Swiss single-domain case (RD11, 42x42x44, Thompson/MYNN/Noah-MP/RRTMG, diff_opt=1/km_opt=4).  Arms per tree:
  release  : namelist as loaded (diff_opt=1, km_opt=4)
  km2      : diff_opt=2/km_opt=2 (untouched v022 scaffold path)
  km3      : diff_opt=2/km_opt=3 (expected to DIFFER: the new literal path)
Compare two trees' JSON: release/km2 sha must be identical.  tridiagonal_solve is replaced by a pure-JAX Thomas
scan in both trees (E105 CPU deadlock; not the product solver)."""
import dataclasses
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

SRC, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, SRC + "/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import cpu_safe  # noqa: E402

cpu_safe.install()
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(SRC), om.__file__
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
CASE = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")


def canon(jaxpr) -> str:
    text = str(jaxpr)
    text = re.sub(r"name_and_src_info=[^\]\n]*", "", text)
    text = re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", text)
    return text


out = {"src": SRC, "om": om.__file__}
# Release fast defaults select Triton Pallas kernels that cannot EXECUTE on CPU (E156); the carry is built with
# pallas_call forced to interpret mode, then the real pallas_call is restored for tracing (make_jaxpr never lowers).
from jax.experimental import pallas as pl  # noqa: E402

_real_pallas_call = pl.pallas_call


def _interpret_pallas_call(*a, **k):
    k["interpret"] = True
    k.pop("compiler_params", None)
    return _real_pallas_call(*a, **k)


pl.pallas_call = _interpret_pallas_call
with tempfile.TemporaryDirectory(dir="<USER_HOME>/wrf_gpu2_lanes/o1-smag3d") as tmp:
    cfg = pipeline.NestedPipelineConfig(CASE, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
    _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
pl.pallas_call = _real_pallas_call
ns, carry = bundles["d01"].namelist, carries["d01"]
out["release_namelist"] = {"diff_opt": int(ns.diff_opt), "km_opt": int(ns.km_opt), "bl_pbl_physics": int(ns.bl_pbl_physics)}
arms = {"release": ns, "km2": dataclasses.replace(ns, diff_opt=2, km_opt=2), "km3": dataclasses.replace(ns, diff_opt=2, km_opt=3)}
for name, nl in arms.items():
    if name == "km3" and not Path(SRC, "src/gpuwrf/runtime/les3d_km3.py").exists():
        continue
    jax.clear_caches()
    cb = om.build_clock_base(nl)
    try:
        # The production per-interval loop (physics + radiation cond + boundaries + RK/acoustics), one step.
        jx = jax.make_jaxpr(lambda c, nl=nl, cb=cb: om._advance_chunk_fori(
            c, nl, 1, cb, n_steps=1, cadence=int(nl.radiation_cadence_steps)))(carry)
    except Exception as exc:  # noqa: BLE001 - record untraceable arms (same in both trees)
        out[name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        Path(OUT).write_text(json.dumps(out, indent=1))
        print(name, out[name], flush=True)
        continue
    text = canon(jx)
    out[name] = {"sha": hashlib.sha256(text.encode()).hexdigest(), "eqns": len(jx.jaxpr.eqns), "chars": len(text)}
    Path(OUT).write_text(json.dumps(out, indent=1))
    print(name, out[name], flush=True)
