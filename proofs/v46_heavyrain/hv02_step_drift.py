"""HV02 (v46-heavyrain): N production steps of the port on CPU; per-step horizontal-mean profiles of theta/qv/qc
(+ full fields at the last step) to measure deterministic per-step drift vs precision arms. Harness copied from
o5-n333 qsqdef/step_dump_cp.py (tkeadvect oracle); only the recorded fields/paths and the mutant block changed.

Follows proofs/o1_nlbind/probe.py: real CLI loader nested_pipeline._load_domains on a private
case dir (symlinks + edited namelist.input), Pallas kept in interpret mode for load AND execution
(release Triton kernels cannot run on CPU; interpret results differ from Triton at round-off).
CPU only; no GPU claim.

usage: JAX_PLATFORMS=cpu python step_dump.py SRC_ROOT CASE_DIR OUT_NPZ NSTEPS [key=val,...]
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

SRC, CASE, OUT, NSTEPS = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
EDITS = dict(kv.split("=", 1) for kv in sys.argv[5].split(",")) if len(sys.argv) > 5 and sys.argv[5] else {}
sys.path.insert(0, SRC + "/src")
sys.path.insert(0, SRC + "/proofs/o1_nlbind")
import jax  # noqa: E402
import numpy as np  # noqa: E402

import cpu_safe  # noqa: E402

cpu_safe.install()
from gpuwrf.backends import pallas as facade  # noqa: E402
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402

assert jax.devices()[0].platform == "cpu"
assert om.__file__.startswith(SRC), om.__file__
state_contract._gpu_device = lambda: jax.devices("cpu")[0]
facade.cuda_native = lambda: False
_real = pl.pallas_call


def _interpret(*a, **k):
    k["interpret"] = True
    k.pop("compiler_params", None)
    return _real(*a, **k)


pl.pallas_call = _interpret
sys.path.insert(0, str(Path(SRC) / "proofs/o1_nlbind"))
probe_src = (Path(SRC) / "proofs/o1_nlbind/probe.py").read_text()
import re  # noqa: E402

edit_namelist = {"re": re}
exec(probe_src[probe_src.index("def edit_namelist"):probe_src.index("out = {")], edit_namelist)
FIELDS = ("theta", "qv", "qc", "qr", "mu", "u", "v")
lane = Path("<USER_HOME>/wrf_gpu2_lanes/v46-heavyrain/tmp")
lane.mkdir(parents=True, exist_ok=True)
t0 = time.time()
with tempfile.TemporaryDirectory(dir=lane) as tmp:
    case = Path(tmp) / "case"
    case.mkdir()
    for f in CASE.iterdir():
        if f.name != "namelist.input":
            (case / f.name).symlink_to(f)
    (case / "namelist.input").write_text(edit_namelist["edit_namelist"]((CASE / "namelist.input").read_text(), EDITS))
    cfg = pipeline.NestedPipelineConfig(case, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
    _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
ns, c = bundles["d01"].namelist, carries["d01"]
print("loaded", round(time.time() - t0, 1), "dt", getattr(ns, "dt_s", None), flush=True)
# Deletion-sensitivity mutants of the tkeadvect wiring (N333_MUTANT; unset = production code):
#   nodiff     qke_adv dropped from the RK1 horizontal/6th-order diffusion list
#   noflowdep  qke_adv dropped from the root flow_dep_bdy (RK1 split) list
MUTANT = os.environ.get("N333_MUTANT", "")
if MUTANT == "nodiff":
    _orig_diff = om._diffused_scalar_species
    om._diffused_scalar_species = lambda nl: tuple(n for n in _orig_diff(nl) if n != "qke_adv")
elif MUTANT == "noflowdep":
    _orig_split = om.root_scalar_rk1_split

    def _split(config, species, **kw):
        relaxed, flow = _orig_split(config, species, **kw)
        return relaxed, tuple(n for n in flow if n != "qke_adv")
    om.root_scalar_rk1_split = _split
elif MUTANT == "noseam":
    # cr-o5-n333 mutant: tsq/cov/sh3d dropped from the RK non-dry replace seam (E85 carry chain)
    om._PHYSICS_NON_DRY_REPLACE_FIELDS = tuple(n for n in om._PHYSICS_NON_DRY_REPLACE_FIELDS if n not in ("tsq", "cov", "sh3d"))
    assert "tsq" not in om._PHYSICS_NON_DRY_REPLACE_FIELDS
elif MUTANT:
    raise SystemExit(f"unknown N333_MUTANT {MUTANT!r}")
print("mutant", MUTANT or "<none>", flush=True)
om._resolve_operational_suite(ns)
cb = om.build_clock_base(ns)
dump = {}


def record(tag, carry, full=False):
    for f in FIELDS:
        v = getattr(carry.state, f, None)
        if v is None:
            continue
        a = np.asarray(v, np.float64)
        dump[f"{tag}_{f}_mean"] = a.reshape(a.shape[0], -1).mean(1) if a.ndim == 3 else np.array([a.mean()])
        if full:
            dump[f"{tag}_{f}"] = a
        if f == "theta":
            dump[f"{tag}_theta_dtype"] = np.array(str(v.dtype))


# XLA:CPU may alias host numpy memory: leaves built from one host array share a buffer and
# _advance_chunk_fori's carry donation then refuses ("donate the same buffer twice").  Report the
# alias groups (a GPU device_put copies, so this is CPU-harness only) and give every leaf its own buffer.
leaves, treedef = jax.tree_util.tree_flatten_with_path(c)
groups = {}
for path, x in leaves:
    if isinstance(x, jax.Array):
        groups.setdefault(x.unsafe_buffer_pointer(), []).append(jax.tree_util.keystr(path))
aliases = [g for g in groups.values() if len(g) > 1]
print("aliased carry leaves", aliases, flush=True)
c = jax.tree.map(lambda x: jax.numpy.array(x, copy=True) if isinstance(x, jax.Array) else x, c)
record("s0", c, full=True)
for i in range(NSTEPS):
    c = om._advance_chunk_fori(c, ns, i + 1, cb, n_steps=1, cadence=int(ns.radiation_cadence_steps))
    jax.block_until_ready(c)
    record(f"s{i + 1}", c, full=(i + 1 == NSTEPS or (i + 1) % 50 == 0))
    inject = [n for n in os.environ.get("N333_INJECT", "").split(",") if n]
    if i == 0 and inject:
        # cr-o5-n333 causal intervention: replace step-1 port leaves by the CPU-WRF step-1 restart values
        import netCDF4
        ds = netCDF4.Dataset(os.environ["N333_INJECT_RST"])
        upd = {n: jax.numpy.asarray(np.asarray(ds.variables[n.upper()][0], np.float32)[: getattr(c.state, n).shape[0]],
                                    dtype=getattr(c.state, n).dtype) for n in inject}
        c = c.replace(state=c.state.replace(**upd))
        print("injected", inject, flush=True)
    print("step", i + 1, round(time.time() - t0, 1), flush=True)
nonfinite = int(sum(int((~np.isfinite(v)).sum()) for v in dump.values() if v.dtype.kind == "f"))
np.savez_compressed(OUT, **dump)
Path(OUT).with_suffix(".json").write_text(json.dumps({
    "src": SRC, "case": str(CASE), "edits": EDITS, "nsteps": NSTEPS, "nonfinite": nonfinite,
    "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS", "<unset=1>"), "interpret_pallas": True,
    "dt_s": float(ns.dt_s), "env": {k: os.environ.get(k) for k in ("GPUWRF_CARRY_REAL_ALL", "GPUWRF_DYN_REAL_ALL", "GPUWRF_FAST_DEFAULTS")}, "aliased_leaves": aliases, "mutant": MUTANT, "wall_s": round(time.time() - t0, 1)}, indent=1))
print("done nonfinite", nonfinite, flush=True)
