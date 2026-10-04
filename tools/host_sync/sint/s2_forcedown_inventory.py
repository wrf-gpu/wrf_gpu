"""S2 f64 inventory of the fused coupled force-down (GPU branches traced on CPU, jaxpr attribution).

Usage: (set -a; . flags.env; set +a; JAX_PLATFORMS=cpu python s2_forcedown_inventory.py <src tree> <out.json>)
"""
import collections, json, pickle, sys
from functools import partial
from pathlib import Path
src = Path(sys.argv[1]); out = Path(sys.argv[2])
sys.path.insert(0, str(src / "src"))
import jax, jax.numpy as jnp
import jax.extend
import numpy as np
jax.default_backend = lambda: "gpu"   # trace the GPU (.rn PTX) branches; lowering only, nothing executes
from jax._src import source_info_util
from gpuwrf.runtime.domain_tree import DomainTree
from gpuwrf.runtime.operational_mode import _initial_carry_for_run
import gpuwrf.nesting.boundary_construction as bc
with open("<USER_HOME>/wrf_gpu2_lanes/integrate/diagnose/common_inputs.pkl", "rb") as fh:
    h, b, _, _, _, _ = pickle.load(fh)
edge = DomainTree.from_domains(h, b).edges["d01"][0]
p = _initial_carry_for_run(b["d01"].state, b["d01"].namelist).state
c = bc.initialize_child_scalar_boundaries(_initial_carry_for_run(b["d02"].state, b["d02"].namelist).state)
fn = partial(bc._coupled_updates, bdy_width=int(c.u_bdy.shape[2]), parent_grid_ratio=3, sint_kernel=True, no_contract=True)
closed = jax.make_jaxpr(fn)(c, p, edge.weights, b["d01"].namelist.metrics, b["d02"].namelist.metrics)
by_site = collections.Counter(); by_prim = collections.Counter(); converts = collections.Counter(); n_pallas = 0
def walk(jaxpr):
    global n_pallas
    for e in jaxpr.eqns:
        if e.primitive.name == "pallas_call":
            n_pallas += 1
            continue
        for sub in jax.extend.core.jaxprs_in_params(e.params):
            walk(sub)
        f64_out = any(getattr(v.aval, "dtype", None) == np.float64 for v in e.outvars)
        if not f64_out:
            continue
        frame = source_info_util.user_frame(e.source_info.traceback)
        site = f"{Path(frame.file_name).name}:{frame.start_line}" if frame else "?"
        if e.primitive.name == "convert_element_type":
            converts[site] += 1
        elif e.primitive.name not in ("broadcast_in_dim", "reshape", "squeeze", "slice", "concatenate", "pad", "copy", "transpose", "select_n", "dynamic_slice", "gather", "rev", "expand_dims"):
            by_site[site] += 1; by_prim[e.primitive.name] += 1
walk(closed.jaxpr)
res = dict(f64_arith=sum(by_site.values()), by_prim=by_prim.most_common(), by_site=by_site.most_common(), f64_converts=sum(converts.values()),
           converts_by_site=converts.most_common(), pallas_calls=n_pallas)
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k in ("f64_arith", "by_prim", "f64_converts", "pallas_calls")}))
for s, n in res["by_site"]: print("arith", n, s)
for s, n in res["converts_by_site"][:12]: print("conv ", n, s)
