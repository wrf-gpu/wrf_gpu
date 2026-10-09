"""RE03 G4: OFF identity of the RRTMG driver refresh and the force-down — candidate vs base export (CPU).

usage: JAX_PLATFORMS=cpu GPUWRF_FAST_DEFAULTS=0 python off_identity.py <src_root> <out.json> <mode>
  mode nested_off : d01+d02 product carries, GPUWRF_NEST_O3_FROM_PARENT unset (released behaviour)
  mode single_on  : d01 only, flag=1 (single-domain runs must not change: no o3rad leaf is seeded)
refresh jaxpr equations are compared after stripping source locations (E151); the eager and fused force seams are
compared by executed outputs (fused with domain_tree._advance_chunk stubbed to identity: the only fused edit is the force).
"""
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

SRC, OUT, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
assert MODE in ("nested_off", "single_on")
if MODE == "single_on":
    os.environ["GPUWRF_NEST_O3_FROM_PARENT"] = "1"
else:
    os.environ.pop("GPUWRF_NEST_O3_FROM_PARENT", None)
os.environ["GPUWRF_NESTED_AOT"] = "0"
sys.path.insert(0, SRC + "/src")
import jax  # noqa: E402

from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import domain_tree as dt  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(SRC), om.__file__
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
CASE = Path("<USER_HOME>/wrf_gpu2_lanes/wn3/cases/20260227_18z_a1")


def canon(jaxpr) -> str:
    text = str(jaxpr)
    text = re.sub(r"name_and_src_info=[^\]\n]*", "", text)
    text = re.sub(r"/[^ \n:]*\.py:\d+(:\d+)?", "", text)
    return text


def digest(jx):
    return {"sha": hashlib.sha256(canon(jx).encode()).hexdigest(), "eqns": len(jx.jaxpr.eqns)}


def leaves_sha(tree):
    leaves = jax.tree.leaves(tree)
    return {"n": len(leaves), "sha": hashlib.sha256(b"".join(
        (str(x.dtype) + str(x.shape)).encode() + bytes(jax.device_get(x).tobytes()) for x in leaves)).hexdigest()}


names = ("d01",) if MODE == "single_on" else ("d01", "d02")
out = {"src": SRC, "mode": MODE, "gpuwrf": om.__file__}
with tempfile.TemporaryDirectory(dir="<USER_HOME>/wrf_gpu2_lanes/fid-q2/RE03") as tmp:
    cfg = pipeline.NestedPipelineConfig(CASE, Path(tmp) / "out", Path(tmp) / "proof", hours=3, max_dom=len(names))
    hierarchy, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, names)
out["o3rad_present"] = {k: getattr(c, "o3rad", None) is not None for k, c in carries.items()}
for name in names:
    ns, carry = bundles[name].namelist, carries[name]
    cb = om.build_clock_base(ns)
    out[f"refresh_{name}"] = digest(jax.make_jaxpr(
        lambda c, rr, ns=ns, cb=cb: om._refresh_rrtmg_driver(c, ns, 0.0, rr, cb))(carry, jax.numpy.asarray(True)))
    out[f"{name}_carry_leaves"] = leaves_sha(carry)
if MODE == "nested_off":
    tree = dt.DomainTree.from_domains(hierarchy, bundles)
    (edge,) = tree.children("d01")
    # build_child_boundary_package holds a nested jit with compiler_options -> not traceable by make_jaxpr;
    # the force seams are compared by their EXECUTED outputs (every leaf byte-hashed; CPU is deterministic).
    Path(OUT).write_text(json.dumps(out, indent=1))
    out["force_d01_d02_outputs"] = leaves_sha(dt._operational_force(edge, carries["d01"], carries["d02"]))
    Path(OUT).write_text(json.dumps(out, indent=1))
    ns, ns2 = bundles["d01"].namelist, bundles["d02"].namelist
    dt._advance_chunk = lambda carry, *_a, **_k: carry
    program = dt._build_fused_cascade_program(
        parent_name="d01", parent_namelist=ns, parent_cadence=int(ns.radiation_cadence_steps),
        child_names=("d02",), child_namelists=(ns2,), child_weights=(edge.weights,),
        child_bdy_widths=(int(tree.domains["d02"].state.u_bdy.shape[2]),),
        child_ratios=(int(edge.parent_grid_ratio),), child_cadences=(int(ns2.radiation_cadence_steps),),
    )
    out["fused_force_d01_d02_outputs"] = leaves_sha(program(carries["d01"], (carries["d02"],), 1, (1,)))
Path(OUT).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
