"""OFF identity of GPUWRF_THOMPSON_MIXED_PHASE_WRF (flag unset): location-stripped jaxprs (E151) of the Thompson
Python body (FULL_COLUMN=0) and of the C24 full-column pallas_call, for the tree on PYTHONPATH. Compare the outputs
of two trees with `diff`. usage: off_identity.py <fixture.npz> <out.txt>"""
import os, re, sys
os.environ.pop("GPUWRF_THOMPSON_MIXED_PHASE_WRF", None)
C24 = dict(GPUWRF_THOMPSON_NATIVE_REAL="1", GPUWRF_THOMPSON_COLUMN_SED="1", GPUWRF_THOMPSON_SED_FP32="1",
           GPUWRF_THOMPSON_COLUMN_LAYOUT="1", GPUWRF_THOMPSON_FULL_COLUMN="1",
           GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT="1", GPUWRF_THOMPSON_SED_PREP_FUSED="0", GPUWRF_THOMPSON_IMPLICIT_SED="0")
os.environ.update(C24)
import jax, jax.numpy as jnp, numpy as np
from gpuwrf.kernels import phys_thompson_full as full
from gpuwrf.physics import thompson_column as tc
z = dict(np.load(sys.argv[1])); idx = np.arange(6)
f = lambda v: jnp.asarray(z[f"in_{v}"][idx], jnp.float32)  # noqa: E731
T, p, qv = f("th") * f("pii"), f("p"), f("qv"); zero = jnp.zeros_like(qv)
state = tc.ThompsonColumnState(qv=qv, qc=f("qc"), qr=f("qr"), qi=f("qi"), qs=f("qs"), qg=f("qg"), Ni=f("ni"), Nr=f("nr"),
                               Ns=zero, Ng=zero, T=T, p=p, rho=tc.density_from_pressure_temperature(p, T, qv),
                               dz=f("dz8w"), w=f("w"))
strip = lambda s: re.sub(r"name_and_src_info=[^\n]*", "name_and_src_info=<>", re.sub(r"[\w./-]+\.py:\d+(:\d+)?", "<loc>", s))  # noqa: E731
out = []
for dt in (6.0, 18.0):
    jax.clear_caches()
    out.append(f"== full_column dt={dt}\n" + strip(str(jax.make_jaxpr(lambda s: full.full_column(s, dt))(state))))
    os.environ["GPUWRF_THOMPSON_FULL_COLUMN"] = "0"
    jax.clear_caches()
    out.append(f"== python body dt={dt}\n" + strip(str(jax.make_jaxpr(lambda s: tc._thompson_source_sink_body(s, dt, False, sediment=True))(state))))
    os.environ["GPUWRF_THOMPSON_FULL_COLUMN"] = "1"
open(sys.argv[2], "w").write("\n".join(out))
print(sys.argv[2], sum(len(s) for s in out), "chars")
