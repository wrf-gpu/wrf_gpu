"""MYNN surface-layer parity: pristine Fortran oracle vs production surface_layer.py
vs faithful fp64 NumPy ref.

Drives all three with IDENTICAL per-column inputs and reports per-field residuals,
on three regime families (daytime-unstable land [crux], stable-night land, neutral)
plus a daytime water column. Emits proofs/v090/mynnsl_parity.json.

Run: JAX_PLATFORMS=cpu JAX_ENABLE_X64=true taskset -c 0-3 python3 mynnsl_parity.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
# Cold-compile: the shared persistent XLA cache holds AOT objects built on a
# different AVX512 host and emits SIGILL-risk warnings on load. Forecast numerics
# are unaffected, but disable it here for a clean, reproducible parity run.
os.environ.setdefault("GPUWRF_JAX_CACHE", "0")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import jax.numpy as jnp  # noqa: E402

from mynn_faithful_ref import sfclay1d_mynn  # noqa: E402
from gpuwrf.physics.surface_constants import P0_PA, R_D_OVER_CP  # noqa: E402
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics  # noqa: E402

OUT_COLS = [
    "ust", "mol", "rmol", "zol", "regime", "psim", "psih", "br",
    "flhc", "flqc", "hfx", "qfx", "lh", "qsfc", "qgh",
    "chs", "chs2", "cqs2", "ch", "wspd", "gz1oz0",
    "u10", "v10", "th2", "t2", "q2", "cpm", "wstar", "qstar", "znt",
]
IN_COLS = [
    "u", "v", "t1d", "qv", "p1d", "dz8w", "rho", "u1d2", "v1d2", "dz2w",
    "mavail", "pblh", "xland", "tsk", "psfcpa", "qcg", "snowh",
    "znt", "ust", "mol", "qsfc", "hfx", "qfx",
]


def run_oracle(cases, exe="mynn_oracle", itimestep=2, isfflx=1, isftcflx=0, iz0tlnd=0, spp_pbl=0, dx=3000.0):
    n = len(cases)
    lines = [f"{n} {itimestep} {isfflx} {isftcflx} {iz0tlnd} {spp_pbl} {dx}"]
    for c in cases:
        lines.append(" ".join(repr(float(c[k])) for k in IN_COLS))
    proc = subprocess.run(
        ["taskset", "-c", "0-3", os.path.join(HERE, exe)],
        input="\n".join(lines) + "\n", capture_output=True, text=True, check=True,
    )
    rows = [ln for ln in proc.stdout.splitlines() if ln and not ln.startswith("#")]
    out = {k: np.zeros(n) for k in OUT_COLS}
    for r in rows:
        p = r.split()
        i = int(p[0]) - 1
        for j, k in enumerate(OUT_COLS):
            out[k][i] = float(p[1 + j])
    return out


class _State:
    pass


def run_production(cases, dx=3000.0):
    """Drive surface_layer.py with EXACTLY the oracle inputs.

    The production code derives t1d from t_air if present (we pass it), thx from
    t1d, and reads psfc/qsfc/mol/hfx/qfx/pblh/dx_m/roughness_m. We feed every field
    the oracle consumed so the only differences are algorithmic.
    """
    n = len(cases)
    g = lambda k: np.array([float(c[k]) for c in cases])  # noqa: E731
    shape = (n, 1)
    st = _State()
    # theta from t1d via the lowest-level pressure (production uses t_air directly
    # for t1d, so theta is only used as a fallback; pass both consistently).
    t1d = g("t1d"); p1d = g("p1d")
    theta = t1d * (P0_PA / p1d) ** R_D_OVER_CP
    st.u = jnp.asarray(g("u").reshape(shape + (1,)))
    st.v = jnp.asarray(g("v").reshape(shape + (1,)))
    st.theta = jnp.asarray(theta.reshape(shape + (1,)))
    st.qv = jnp.asarray(g("qv").reshape(shape + (1,)))
    st.p = jnp.asarray(p1d.reshape(shape + (1,)))
    st.dz = jnp.asarray(g("dz8w").reshape(shape + (1,)))
    st.t_air = jnp.asarray(t1d.reshape(shape))
    st.t_skin = jnp.asarray(g("tsk").reshape(shape))
    st.psfc = jnp.asarray(g("psfcpa").reshape(shape))
    st.xland = jnp.asarray(g("xland").reshape(shape))
    st.lakemask = jnp.zeros(shape)
    st.roughness_m = jnp.asarray(g("znt").reshape(shape))
    st.mavail = jnp.asarray(g("mavail").reshape(shape))
    st.ustar = jnp.asarray(g("ust").reshape(shape))
    st.mol = jnp.asarray(g("mol").reshape(shape))
    st.hfx = jnp.asarray(g("hfx").reshape(shape))
    st.qfx = jnp.asarray(g("qfx").reshape(shape))
    st.pblh = jnp.asarray(g("pblh").reshape(shape))
    st.dx_m = jnp.full(shape, dx)
    # qsfc: oracle gets -1 (recompute over water/<=0 land); production reads qsfc
    # field with same <=0 convention.
    st.qsfc = jnp.asarray(g("qsfc").reshape(shape))

    d = surface_layer_with_diagnostics(st)
    flat = lambda x: np.asarray(x, dtype=np.float64).reshape(n)  # noqa: E731
    return {
        "ust": flat(d.fluxes.ustar), "mol": flat(d.mol), "rmol": flat(d.rmol),
        "zol": flat(d.zol), "regime": flat(d.regime), "psim": flat(d.psim),
        "psih": flat(d.psih), "br": flat(d.br), "hfx": flat(d.hfx), "lh": flat(d.lh),
        "qsfc": flat(d.qsfc), "u10": flat(d.u10), "v10": flat(d.v10),
        "th2": flat(d.th2), "t2": flat(d.t2), "q2": flat(d.q2), "znt": flat(d.znt),
        # derived for cross-check
        "rhosfc": flat(d.fluxes.rhosfc),
    }


def ref_inputs(cases, dx=3000.0):
    inp = {k: np.array([float(c[k]) for c in cases]) for k in IN_COLS}
    inp["dx"] = np.full(len(cases), dx)
    return inp


# ---- representative regime cases (skin/air/wind chosen to span the regimes) ----
def make_cases():
    DZ = 51.4  # Canary d02 lowest full layer thickness -> za ~= 25.7 m
    base = dict(dz8w=DZ, u1d2=0.0, v1d2=0.0, dz2w=100.0, qcg=0.0, snowh=0.0, qsfc=-1.0)
    cases = []
    labels = []
    # --- DAYTIME UNSTABLE LAND sweep (the crux): warm skin over cooler air ---
    for dT, wind in [(8.0, 2.5), (12.0, 3.5), (16.0, 4.0), (6.0, 1.5)]:
        c = dict(base, u=wind, v=0.5 * wind, t1d=298.0, qv=0.010, p1d=95000.0, rho=1.10,
                 mavail=0.30, pblh=1200.0, xland=1.0, tsk=298.0 + dT, psfcpa=95200.0,
                 znt=0.10, ust=0.30, mol=-0.30, hfx=250.0, qfx=0.0001)
        cases.append(c); labels.append(f"unstable-land dT={dT:+.0f} U={wind}")
    # --- STABLE NIGHT LAND: cold skin under warmer air ---
    for dT, wind in [(-4.0, 1.5), (-8.0, 2.0), (-2.0, 3.0)]:
        c = dict(base, u=wind, v=0.5 * wind, t1d=288.0, qv=0.008, p1d=95000.0, rho=1.15,
                 mavail=0.30, pblh=150.0, xland=1.0, tsk=288.0 + dT, psfcpa=95200.0,
                 znt=0.10, ust=0.15, mol=0.10, hfx=-20.0, qfx=0.00001)
        cases.append(c); labels.append(f"stable-land dT={dT:+.0f} U={wind}")
    # --- NEUTRAL LAND ---
    c = dict(base, u=5.0, v=5.0, t1d=295.0, qv=0.009, p1d=95000.0, rho=1.12,
             mavail=0.30, pblh=800.0, xland=1.0, tsk=295.05, psfcpa=95200.0,
             znt=0.10, ust=0.25, mol=0.0, hfx=0.0, qfx=0.0)
    cases.append(c); labels.append("neutral-land")
    # --- DAYTIME WATER ---
    for dT, wind in [(1.0, 7.0), (2.0, 10.0)]:
        c = dict(base, u=wind, v=0.7 * wind, t1d=293.0, qv=0.013, p1d=101000.0, rho=1.18,
                 mavail=1.0, pblh=600.0, xland=2.0, tsk=293.0 + dT, psfcpa=101100.0,
                 znt=0.0028, ust=0.26, mol=-0.05, hfx=50.0, qfx=0.0001)
        cases.append(c); labels.append(f"water dT={dT:+.0f} U={wind}")
    return cases, labels


def residual(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    absd = np.abs(a - b)
    rel = absd / np.maximum(np.abs(b), 1e-12)
    return float(np.max(absd)), float(np.max(rel))


def main():
    cases, labels = make_cases()
    orc = run_oracle(cases)
    ref = sfclay1d_mynn(ref_inputs(cases))
    prod = run_production(cases)

    # fields the production code reports
    compare_fields = ["ust", "mol", "rmol", "zol", "regime", "psim", "psih", "br",
                      "hfx", "lh", "qsfc", "u10", "v10", "th2", "t2", "q2", "znt"]

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "oracle": "module_sf_mynn.F (byte-identical pristine WRF, fp32 build)",
        "oracle_sha256": "86395534a6c9bfc79dcad50094bce290eff05756777a95794b2673795f9761c3",
        "config": "ISFFLX=1 isftcflx=0 iz0tlnd=0 spp_pbl=0 COARE_OPT=3.0 psi_opt=0(CB05) itimestep=2 dx=3000",
        "n_cases": len(cases),
        "labels": labels,
        "faithful_ref_vs_oracle": {},
        "production_vs_oracle": {},
        "per_case": {},
    }

    print("\n================= FAITHFUL fp64 ref  vs  pristine Fortran oracle =================")
    for k in compare_fields:
        if k not in ref:
            continue
        absd, rel = residual(ref[k], orc[k])
        report["faithful_ref_vs_oracle"][k] = {"absmax": absd, "relmax": rel}
        print(f"  {k:8s} absmax={absd:12.4e} relmax={rel:12.4e}")

    print("\n================= PRODUCTION surface_layer.py  vs  pristine Fortran oracle =======")
    for k in compare_fields:
        if k not in prod:
            continue
        absd, rel = residual(prod[k], orc[k])
        report["production_vs_oracle"][k] = {"absmax": absd, "relmax": rel}
        print(f"  {k:8s} absmax={absd:12.4e} relmax={rel:12.4e}")

    print("\n================= PER-CASE HFX / T2 / MOL / ZOL (W vs production vs ref) =========")
    for i, lab in enumerate(labels):
        rec = {
            "label": lab,
            "hfx": {"oracle": float(orc["hfx"][i]), "prod": float(prod["hfx"][i]), "ref": float(ref["hfx"][i])},
            "t2": {"oracle": float(orc["t2"][i]), "prod": float(prod["t2"][i]), "ref": float(ref["t2"][i])},
            "mol": {"oracle": float(orc["mol"][i]), "prod": float(prod["mol"][i]), "ref": float(ref["mol"][i])},
            "zol": {"oracle": float(orc["zol"][i]), "prod": float(prod["zol"][i]), "ref": float(ref["zol"][i])},
            "br": {"oracle": float(orc["br"][i]), "prod": float(prod["br"][i]), "ref": float(ref["br"][i])},
            "ust": {"oracle": float(orc["ust"][i]), "prod": float(prod["ust"][i]), "ref": float(ref["ust"][i])},
        }
        report["per_case"][lab] = rec
        print(f"  {lab:26s} HFX  W={orc['hfx'][i]:8.2f}  prod={prod['hfx'][i]:8.2f}  ref={ref['hfx'][i]:8.2f}")
        print(f"  {'':26s} T2   W={orc['t2'][i]:8.3f}  prod={prod['t2'][i]:8.3f}  ref={ref['t2'][i]:8.3f}")

    out_path = os.path.join(HERE, "mynnsl_parity.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
