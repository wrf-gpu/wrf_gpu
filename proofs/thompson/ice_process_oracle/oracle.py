"""TH08 (CPU, generator of tests/v025/b_thompson/fixtures/ice_process_columns.npz): adversarial ice-process columns -> pristine WRF mp_gt_driver vs port (native REAL, C24, interpret).
usage: oracle.py build [ice|graupel] -> writes <set>_process_columns.npz (columns + WRF output; needs the TH08 driver)
       oracle.py compare -> port (fixed + mutants) vs WRF table"""
import inspect, json, os, subprocess, sys, textwrap
from pathlib import Path
import numpy as np
HERE = Path(os.environ.get("TH08_OUT_DIR", "<USER_HOME>/wrf_gpu2_lanes/b-thompson/TH08")); DRV = Path(os.environ.get("TH08_BUILD_DIR", str(HERE / "driver")))
NLEV, DT = 30, 54.0
P0, RD, CP = 1.0e5, 287.04, 1004.5

def qvsi_wrf(p, T):  # numpy copy of port RSIF via JAX is used in build for exactness (see below)
    raise NotImplementedError

def columns(sat_ice, sat_liq, which="ice"):
    """Return dict of (name -> dict of f32 [NLEV] arrays) for the "ice" (TH08/B52) or "graupel" (TH10/B52b) set."""
    k = np.arange(NLEV)
    p = (95000.0 * np.exp(-k / 12.0)).astype(np.float64)
    cols = {}
    def base(T_layer, k0, k1):
        T = np.empty(NLEV); z = 0.0; T0 = 288.0
        for kk in range(NLEV):
            T[kk] = T_layer if k0 <= kk <= k1 else max(T0 - 0.0065 * z, 205.0)
            dz = RD * T[kk] / 9.81 / 12.0; z += dz
        dzs = RD * T / 9.81 / 12.0
        qv = 0.5 * sat_liq(p, T)
        return T, dzs, qv
    def make(name, T_layer, k0, k1, ssati=None, ssatw=None, **q):
        T, dz, qv = base(T_layer, k0, k1)
        lay = (k >= k0) & (k <= k1)
        if ssati is not None:
            qv = np.where(lay, (1.0 + ssati) * sat_ice(p, T), qv)
        if ssatw is not None:
            qv = np.where(lay, (1.0 + ssatw) * sat_liq(p, T), qv)
        c = {n: np.zeros(NLEV) for n in ("qc", "qr", "qi", "qs", "qg", "ni", "nr")}
        for n, v in q.items():
            c[n] = np.where(lay, v, 0.0)
        pii = (p / P0) ** (RD / CP)
        c.update(qv=qv, th=T / pii, pii=pii, p=p, w=np.zeros(NLEV), dz8w=dz)
        cols[name] = {n: np.asarray(v, np.float32) for n, v in c.items()}
    if which == "graupel":
        # C: supercooled rain freezing onto graupel -> sedimentation with the re-diagnosed ng (B52b, :3287-3300).
        make("C1_rci_graupel", 272.0, 3, 7, ssatw=0.0, qr=4e-3, nr=3e3, qi=3e-7, ni=5e2, qg=1e-3)
        make("C2_rci_graupel_cold", 269.0, 3, 7, ssatw=0.0, qr=3e-3, nr=5e3, qi=5e-7, ni=1e3, qg=5e-4)
        # E: graupel sublimation (t2_qg_sd, cge(11)); F: graupel melt (t2_qg_me); G: riming (t1_qg_qc, cge(9), vtg);
        # H: graupel fall only (vtg = rhof*av_g*cgg(6)*ogg3*ilamg**bv_g, :3758).
        make("E1_graupel_subl", 258.0, 8, 12, ssati=-0.15, qg=2e-3)
        make("F1_graupel_melt", 277.0, 2, 5, ssatw=0.0, qg=2e-3)
        make("G1_graupel_rime", 266.0, 6, 10, ssatw=0.0, qc=6e-4, qg=2e-3)
        make("H1_graupel_fall", 250.0, 10, 14, ssati=0.0, qg=3e-3)
        return cols
    # A: sublimation clamps (fix 1): ice-subsaturated, snow term alone exceeds rate_max (per-term clamp binds).
    make("A1_subl_snowdom", 253.0, 8, 14, ssati=-0.05, qi=4e-4, ni=1e6, qs=1.2e-2)
    make("A2_subl_cold", 240.0, 8, 14, ssati=-0.05, qi=4e-4, ni=8e5, qs=1.2e-2)
    # B: pri_inu deposition ratio (fix 4): nucleation-active, deposition-heavy snow -> joint limiter binds.
    make("B1_inu_ssati", 238.0, 14, 18, ssati=0.30, qi=2e-3, ni=3e5, qs=1.2e-2)
    make("B2_inu_watersat", 240.0, 12, 16, ssatw=0.005, qi=2e-3, ni=3e5, qs=1.2e-2)
    # C: prg_rci formed once (fix 2): supercooled rain + little ice + entry graupel (so WRF sediments graupel too).
    make("C1_rci_graupel", 272.0, 3, 7, ssatw=0.0, qr=4e-3, nr=3e3, qi=3e-7, ni=5e2, qg=1e-3)
    make("C2_rci_graupel_cold", 269.0, 3, 7, ssatw=0.0, qr=3e-3, nr=5e3, qi=5e-7, ni=1e3, qg=5e-4)
    # D: control, no limiter should bind.
    make("D_control", 255.0, 8, 12, ssati=0.02, qi=1e-5, ni=5e4, qs=1e-4)
    return cols

ORDER_IN = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th", "pii", "p", "w", "dz8w")
ORDER_OUT = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th")

def run_wrf(cols, work):
    names = list(cols); ncol = len(names)
    fin, fout = work / "wrf_in.bin", work / "wrf_out.bin"
    with open(fin, "wb") as f:
        f.write(np.array([ncol, NLEV], ">i4").tobytes()); f.write(np.array([DT], ">f4").tobytes())
        for v in ORDER_IN:
            a = np.stack([cols[n][v] for n in names])            # (ncol, nlev) -> Fortran (i fastest)
            f.write(np.asarray(a.T, ">f4").tobytes(order="C"))   # (nlev, ncol) C == (ncol, nlev) F
    subprocess.run(["taskset", "-c", os.environ.get("TH08_CPUS", "12,13"), str(DRV / "thompson_column_driver.exe"), str(fin), str(fout)],
                   cwd=DRV, check=True, capture_output=True)
    raw = np.fromfile(fout, ">f4"); out = {}; o = 0
    for v in ORDER_OUT:
        out[v] = raw[o:o + ncol * NLEV].reshape(NLEV, ncol).T.astype(np.float32); o += ncol * NLEV
    for v in ("rainncv", "snowncv", "graupelncv", "sr"):
        out[v] = raw[o:o + ncol].astype(np.float32); o += ncol
    return names, out

def build():
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax, jax.numpy as jnp
    assert jax.devices()[0].platform == "cpu"
    from gpuwrf.physics import thompson_column as tc
    si = lambda p, T: np.asarray(tc.saturation_mixing_ratio_ice(jnp.asarray(p, jnp.float32), jnp.asarray(T, jnp.float32)), np.float64)
    sl = lambda p, T: np.asarray(tc.saturation_mixing_ratio_liquid(jnp.asarray(p, jnp.float32), jnp.asarray(T, jnp.float32)), np.float64)
    which = sys.argv[2] if len(sys.argv) > 2 else "ice"
    cols = columns(si, sl, which)
    names, out = run_wrf(cols, HERE)
    np.savez(HERE / f"{which}_process_columns.npz", names=np.array(names), dt=np.float32(DT),
             **{f"in_{v}": np.stack([cols[n][v] for n in names]) for v in ORDER_IN},
             **{f"wrf_{v}": out[v] for v in out})
    print("built", names)

if __name__ == "__main__":
    build() if sys.argv[1] == "build" else None


C24 = dict(GPUWRF_THOMPSON_NATIVE_REAL="1", GPUWRF_THOMPSON_COLUMN_SED="1", GPUWRF_THOMPSON_SED_FP32="1",
           GPUWRF_THOMPSON_COLUMN_LAYOUT="1", GPUWRF_THOMPSON_FULL_COLUMN="1",
           GPUWRF_THOMPSON_FULL_COLUMN_EARLY_EXIT="1", GPUWRF_THOMPSON_SED_PREP_FUSED="0")
FIX_LINES = {  # mutant name -> (fixed source snippet, snippet with the fix removed)
    "no_sublimation_clamp": ("sublimation_floor = ((state.qv - qvsi) * state.rho / float(dt) * 0.999).astype(jnp.float64)",
                             "sublimation_floor = None"),
    "no_inu_ratio": ("inu_mass = pri_inu * deposition_ratio * float(dt) / state.rho  # WRF scales pri_inu too (:2868)", "pass"),
    "prg_rci_rescaled": ("    if not _native_real_enabled():\n        prg_rci = pri_rci + prr_rci\n", "    prg_rci = pri_rci + prr_rci\n"),
}
FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th")

def port_state(z, tc, jnp):
    f = lambda v: jnp.asarray(z[f"in_{v}"], jnp.float32)
    T = f("th") * f("pii"); p, qv = f("p"), f("qv")
    zero = jnp.zeros_like(qv)
    return tc.ThompsonColumnState(qv=qv, qc=f("qc"), qr=f("qr"), qi=f("qi"), qs=f("qs"), qg=f("qg"), Ni=f("ni"), Nr=f("nr"),
                                  Ns=zero, Ng=zero, T=T, p=p, rho=tc.density_from_pressure_temperature(p, T, qv),
                                  dz=f("dz8w"), w=f("w"))

def mutate(tc, name):
    fixed, removed = FIX_LINES[name]
    src = textwrap.dedent(inspect.getsource(tc._ice_sources_with_process_flags))
    assert src.count(fixed) == 1, name
    ns = dict(vars(tc)); exec(compile(src.replace(fixed, removed), tc.__file__, "exec"), ns)
    return ns["_ice_sources_with_process_flags"]

def run_port(z, mutant=None):
    import jax, jax.numpy as jnp
    from gpuwrf.physics import thompson_column as tc
    from gpuwrf.kernels import phys_thompson_full as full
    os.environ.update(C24); jax.clear_caches()
    original = tc._ice_sources_with_process_flags
    try:
        if mutant:
            tc._ice_sources_with_process_flags = mutate(tc, mutant)
        out, _ppt = full.full_column(port_state(z, tc, jnp), float(z["dt"]), interpret=True)
    finally:
        tc._ice_sources_with_process_flags = original
    pii = np.asarray(z["in_pii"], np.float64)
    g = lambda a: np.asarray(a, np.float64)
    return {"qv": g(out.qv), "qc": g(out.qc), "qr": g(out.qr), "qi": g(out.qi), "qs": g(out.qs), "qg": g(out.qg),
            "ni": g(out.Ni), "nr": g(out.Nr), "th": g(out.T) / pii}

def errors(port, z):
    """Per column, per field: max |port - WRF| relative to the column's WRF field scale (max |WRF|, floor)."""
    res = {}
    for i, name in enumerate(z["names"]):
        row = {}
        for v in FIELDS:
            wrf = np.asarray(z[f"wrf_{v}"][i], np.float64); d = np.abs(port[v][i] - wrf)
            scale = max(np.abs(wrf).max(), {"ni": 1.0, "nr": 1.0, "th": 1.0}.get(v, 1e-9))
            row[v] = float(d.max() / scale)
        res[str(name)] = row
    return res

def compare():
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    assert jax.devices()[0].platform == "cpu"
    z = dict(np.load(HERE / "ice_process_columns.npz"))
    report = {"fixed": errors(run_port(z), z)}
    for m in FIX_LINES:
        report[m] = errors(run_port(z, m), z)
    json.dump(report, open(HERE / "compare.json", "w"), indent=1)
    for arm, res in report.items():
        print("==", arm)
        for col, row in res.items():
            print(f"   {col:18s} " + " ".join(f"{v}={row[v]:.1e}" for v in FIELDS))

if __name__ == "__main__" and sys.argv[1] == "compare":
    compare()
