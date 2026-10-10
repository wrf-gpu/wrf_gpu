"""CPU coupled smoke of diff_opt=2/km_opt=3 on the real Swiss single-domain case (RD11, 42x42x44, dt 18 s).

usage: JAX_PLATFORMS=cpu python smoke.py <out.json> [steps] [bl_pbl_physics] [arms=km3,km4_release]
Runs N production steps (om._advance_chunk -> _advance_chunk_fori) with km_opt=3 and with the released
diff_opt=1/km_opt=4 from the same initial carry; reports finiteness of every carry leaf, census guard events and the
km3-vs-km4 differences.  Two modes:
  * legacy:  GPUWRF_FAST_DEFAULTS=0 (fp64 legacy carry; everything executes natively on XLA:CPU);
  * release: GPUWRF_FAST_DEFAULTS=1 O1_PALLAS_INTERPRET=1 (REAL32 release carry/kernels; every pallas_call runs in
    interpret mode and the kernels' PTX "<op>.rn.f32" inline asm is emulated exactly-rounded in f64 -> REAL4).
    Same kernel bodies and REAL rounding points, CPU instruction semantics (no FMA guarantees) -- NOT the GPU binary.
tridiagonal_solve is a pure-JAX Thomas scan in every arm (E105 CPU deadlock; not the product solver)."""
import dataclasses
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import cpu_safe  # noqa: E402

cpu_safe.install()
import os  # noqa: E402

if os.environ.get("O1_PALLAS_INTERPRET") == "1":
    # Release fast defaults (GPUWRF_FAST_DEFAULTS=1, REAL32 carry) select Triton Pallas kernels that cannot
    # execute on CPU (E156): run EVERY pallas_call in interpret mode (same kernel bodies, CPU semantics).
    from jax.experimental import pallas as _pl

    _real_pallas_call = _pl.pallas_call

    def _interpret_pallas_call(*a, **k):
        k["interpret"] = True
        k.pop("compiler_params", None)
        return _real_pallas_call(*a, **k)

    _pl.pallas_call = _interpret_pallas_call
    # The release REAL kernels also pin REAL4 rounding with PTX inline asm ("<op>.rn.f32"/"<op>.<mode>.f64"),
    # which has no CPU lowering. CPU-execution shim (NOT the product instruction): exact op in f64, then
    # round to the result precision (the same emulation as boundary_apply's own CPU prescreen).
    import jax.experimental.pallas.triton as _pt

    def _cpu_inline_asm(asm, *, args, result_shape_dtypes, **_kw):
        op, mode, ptx = asm.split()[0].split(".")[:3]
        a, b = (jnp.asarray(x, jnp.float64) for x in args)
        exact = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}[op]
        out_dtype = result_shape_dtypes[0].dtype
        if jnp.dtype(out_dtype) == jnp.float32:
            exact = jax.lax.reduce_precision(exact, exponent_bits=8, mantissa_bits=23)
        return [exact.astype(out_dtype)]

    _pt.elementwise_inline_asm = _cpu_inline_asm
from gpuwrf.contracts import state as state_contract  # noqa: E402
from gpuwrf.integration import nested_pipeline as pipeline  # noqa: E402
from gpuwrf.runtime import operational_mode as om  # noqa: E402

state_contract._gpu_device = lambda: jax.devices("cpu")[0]
OUT = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
BL = int(sys.argv[3]) if len(sys.argv) > 3 else 5
CASE = Path("<USER_HOME>/wrf_gpu2_lanes/release-docs/RD11/cpu")
with tempfile.TemporaryDirectory(dir="<USER_HOME>/wrf_gpu2_lanes/o1-smag3d") as tmp:
    cfg = pipeline.NestedPipelineConfig(CASE, Path(tmp) / "out", Path(tmp) / "proof", hours=1, max_dom=1)
    _h, bundles, _x, _rs, _dt, carries = pipeline._load_domains(cfg, ("d01",))
ns, carry0 = bundles["d01"].namelist, carries["d01"]
report = {"steps": STEPS, "bl_pbl_physics": BL, "dt": float(ns.dt_s),
          "fast_defaults": os.environ.get("GPUWRF_FAST_DEFAULTS"), "pallas_interpret": os.environ.get("O1_PALLAS_INTERPRET"),
          "theta_dtype": str(carry0.state.theta.dtype), "u_dtype": str(carry0.state.u.dtype)}
ARMS = sys.argv[4].split(",") if len(sys.argv) > 4 else ["km3", "km4_release"]
arms = {k: v for k, v in {"km3": dataclasses.replace(ns, diff_opt=2, km_opt=3, bl_pbl_physics=BL),
                          "km4_release": ns}.items() if k in ARMS}
finals = {}
for name, nl in arms.items():
    cb = om.build_clock_base(nl)
    # Own copy per arm: release defaults DONATE the carry (GPUWRF_CARRY_DONATE=1).
    carry = jax.tree.map(lambda x: x.copy() if hasattr(x, "copy") else x, carry0)
    t0 = time.time()
    # Production entry _advance_chunk (unalias for donation + _advance_chunk_fori loop), one step per call.
    for i in range(STEPS):
        carry = om._advance_chunk(carry, nl, i + 1, cb, n_steps=1, cadence=int(nl.radiation_cadence_steps))
        jax.block_until_ready(carry)
    leaves = jax.tree.leaves(carry)
    nonfinite = int(sum(int((~np.isfinite(np.asarray(x))).sum()) for x in leaves
                        if np.issubdtype(np.asarray(x).dtype, np.floating)))
    finals[name] = carry
    report[name] = {"wall_s": round(time.time() - t0, 1), "nonfinite_values": nonfinite, "leaves": len(leaves)}
    if getattr(carry, "census", None) is not None:
        from gpuwrf.diagnostics.census import EVENTS, GUARDS

        g = np.asarray(carry.census.guards)
        report[name]["guards_fired"] = {GUARDS[r]: dict(zip(EVENTS, map(int, g[r]))) for r in range(g.shape[0]) if g[r].any()}
        report[name]["guard_total"] = int(g.sum())
    print(name, report[name], flush=True)
    Path(OUT).write_text(json.dumps(report, indent=1))
if "km4_release" not in finals:
    raise SystemExit(0)
a, b = finals["km3"].state, finals["km4_release"].state
for f in ("u", "v", "w", "theta", "qv", "qc", "mu_total"):
    x, y = np.asarray(getattr(a, f), np.float64), np.asarray(getattr(b, f), np.float64)
    report.setdefault("km3_minus_km4", {})[f] = {"max_abs": float(np.abs(x - y).max()), "rms": float(np.sqrt(np.mean((x - y) ** 2)))}
report["qv_min_km3"] = float(np.asarray(a.qv).min())
Path(OUT).write_text(json.dumps(report, indent=1))
print(json.dumps(report["km3_minus_km4"]), flush=True)
