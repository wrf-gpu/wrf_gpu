"""C05: native PD kernel (interpret, CPU) vs pristine WRF advect_scalar_pd REAL4 on real PROD d01 operands.
Bounds (b-core BC42 registration, unchanged): max|d| <= max(8*max E, 1e-5*max|ref|), rms <= max(8*rms E, 1e-6*max|ref|),
E = |pristine(fast) - pristine(off)|; region interior ring>=1; limiter active per species."""
import json, os, sys, time
from pathlib import Path
import numpy as np
assert os.environ["JAX_PLATFORMS"] == "cpu"
if Path("/tmp/wrf_gpu2_quiet").exists():
    raise SystemExit("QUIET")
SRC = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(SRC / "scripts/v025"), str(SRC / "tests/v025/b_core"), "<USER_HOME>/wrf_gpu2_lanes/b-carry/C05", str(SRC / "src")]
from real_state import load_real_snapshot, stage_advanced_state
from pd_scalar_oracle import evaluate
from prod_inputs import prod_domains
import scalar_oracle
import jax, jax.numpy as jnp
from gpuwrf.runtime import operational_mode as op
from gpuwrf.runtime.domain_tree import DomainTree
from gpuwrf.contracts.halo import apply_halo
from gpuwrf.dynamics.advection import halo_spec
from gpuwrf.kernels.dyn_pd_fp32 import advect_scalar_pd_fp32
B = Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase2b/BC42_root_numbers")
hierarchy, bundles, _, _, _, carries = prod_domains()
nml = DomainTree.from_domains(hierarchy, bundles, feedback_enabled=False).domains["d01"].namelist
m = nml.metrics; base = carries["d01"].state
case = Path("<DATA_ROOT>/alisios/runs/20260725_18z/wn2_wrf/alisios_operational_d01_121x71_d02_268x118_oldgrid_v1/attempt_001/cpu_case")
pair = sys.argv[3] if len(sys.argv) > 3 else "02:00:36,03:00:00"
a_name, b_name = pair.split(",")
h2 = load_real_snapshot(case, domain="d01", wrfout_name=f"wrfout_d01_2026-07-26_{a_name}")
h3 = load_real_snapshot(case, domain="d01", wrfout_name=f"wrfout_d01_2026-07-26_{b_name}")
adv, construction = stage_advanced_state(h2, h3, 54.0)
SPECIES = ("qv", "qc", "qr", "Nr", "Ni")
FIELDS = ("u", "v", "w", "mu_total", "mu_perturbation", "qv", "qc", "qr", "Nr")
def product_state(src):
    u = {n: jnp.asarray(np.asarray(getattr(src, n)), getattr(base, n).dtype) for n in FIELDS}
    u["Ni"] = jnp.asarray(1e8 * np.asarray(src.qc), base.Ni.dtype)  # b-core BC42: d01 has no ice; real cloud-edge structure
    return base.replace(**u)
origin = product_state(h3.state)
current = apply_halo(product_state(adv), halo_spec(nml.grid))
mub = np.asarray(h3.state.mu_total) - np.asarray(h3.state.mu_perturbation)
vel = op._stage_transport_velocities(current, nml)
assert vel.specified and vel.ru_full is not None
rdx = 1.0 / float(nml.grid.projection.dx_m); rdy = 1.0 / float(nml.grid.projection.dy_m)
f32 = lambda a: jnp.asarray(np.asarray(a), jnp.float32)
q = jnp.stack([f32(getattr(current, s)) for s in SPECIES]); qo = jnp.stack([f32(getattr(origin, s)) for s in SPECIES])
t0 = time.perf_counter()
mode = os.environ.get("PD_MUB", "split")
if mode == "split":   # WRF form: mub + old perturbation
    kw = dict(mub=f32(mub), muold=f32(np.asarray(origin.mu_total) - mub))
else:                 # product contract: full old mass
    kw = dict(muold=f32(origin.mu_total))
got = advect_scalar_pd_fp32(q, qo, f32(vel.ru_full), f32(vel.rv_full), f32(vel.rom), f32(current.mu_total),
        c1=f32(m.c1h), c2=f32(m.c2h), msftx=f32(m.msftx), msfty=f32(m.msfty), rdzw=f32(m.rdnw), fzm=f32(m.fnm), fzp=f32(m.fnp),
        rdx=rdx, rdy=rdy, dt=54.0, interpret=True, block=int(os.environ.get("PD_BLOCK", "16384")), **kw)
got = np.asarray(jax.block_until_ready(got), np.float64)
kernel_s = time.perf_counter() - t0
rows = []
for i, sp in enumerate(SPECIES):
    arrays = dict(q=np.asarray(q[i]), qold=np.asarray(qo[i]), ru=np.asarray(vel.ru_full), rv=np.asarray(vel.rv_full), rom=np.asarray(vel.rom),
        mut=np.asarray(current.mu_total), mub=mub, muold=np.asarray(origin.mu_total) - mub, mx=np.asarray(m.msftx), my=np.asarray(m.msfty),
        ux=np.asarray(m.msfux), uy=np.asarray(m.msfuy), vx=np.asarray(m.msfvx), vy=np.asarray(m.msfvy), fzm=np.asarray(m.fnm), fzp=np.asarray(m.fnp),
        rdzw=np.asarray(m.rdnw), c1=np.asarray(m.c1h), c2=np.asarray(m.c2h))
    off = evaluate(B / "oracle_off/pd.so", arrays, rdx, rdy, 54.0).astype(np.float64)
    fast = evaluate(B / "oracle_fast/pd.so", arrays, rdx, rdy, 54.0).astype(np.float64)
    plain = scalar_oracle.evaluate(Path("<USER_HOME>/wrf_gpu2_lanes/b-core/phase2b/BC30/oracles/off/scalar.so"),
        dict(q=arrays["q"], ru=arrays["ru"], rv=arrays["rv"], rom=arrays["rom"], mu=arrays["mut"], map=arrays["mx"],
             fzm=arrays["fzm"], fzp=arrays["fzp"], rdzw=arrays["rdzw"], c1=arrays["c1"], c2=arrays["c2"]), rdx).astype(np.float64)
    sl = (slice(None), slice(1, -1), slice(1, -1))
    d = (got[i] - off)[sl]; env = (fast - off)[sl]; refmax = float(np.max(np.abs(off[sl]))) or 1e-30
    max_bound = max(8 * float(np.max(np.abs(env))), 1e-5 * refmax); rms_bound = max(8 * float(np.sqrt(np.mean(env ** 2))), 1e-6 * refmax)
    limited = int(np.count_nonzero(np.abs(off - plain)[sl] > 1e-6 * refmax))
    row = dict(species=sp, ref_max=refmax, max_abs=float(np.max(np.abs(d))), rms=float(np.sqrt(np.mean(d * d))), env_max=float(np.max(np.abs(env))),
               max_bound=max_bound, rms_bound=rms_bound, limiter_cells=limited, full_max_abs=float(np.max(np.abs(got[i] - off))),
               full_ref_max=float(np.max(np.abs(off))), exact_frac=float(np.mean(got[i] == off)), finite=bool(np.isfinite(got[i]).all()))
    row["passed"] = bool(row["finite"] and row["max_abs"] <= max_bound and row["rms"] <= rms_bound and limited > 0)
    rows.append(row); print(json.dumps(row), flush=True)
res = dict(pair=pair, mub_mode=mode, construction=construction, kernel_interpret_s=kernel_s, rows=rows, passed=all(r["passed"] for r in rows))
(OUT / f"gate_{mode}_{a_name.replace(':','')}.json").write_text(json.dumps(res, indent=1) + "\n")
print("VERDICT", res["passed"], "kernel_s %.1f" % kernel_s)
