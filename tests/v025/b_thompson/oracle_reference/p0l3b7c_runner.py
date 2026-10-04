# Provenance: source commit 83c7ff126; original SHA256 8902b77c5832ffa7fd20b7df63c473502d5c56b1ff419bce38ef562c72d6b65a
# Source: closure/k7/p0l3b7c_runner.py
"""P0 L3b r7 WRF/port column runner: one r7 column, one state-dtype arm (P0_L3B_R7_COLUMN_PACKAGE_CONTRACT.md).

Derived from the rev-6 runner l3/p0c_acol_runner.py (50b5479f; rejected with its manifest and NOT edited): the
P0C3BOUT parser, the WRF-input echo check, the 1x1 aerosol State build, the fixed and kernel-direct arms and the
in-memory mutants (source substitution of the product function's CURRENT source with exactly one registered line, or
a swap of thompson_aero_column's own gate/band helper; recompiled after jax.clear_caches(); no product edit) are
unchanged.  New:
  * exactly the committed r7 manifest 31f7e9d4, the Gate-2 tolerances 5bdc03da AND the mp28 tolerances 1601d68b and
    the frozen compare cb78b4f6 are hard-pinned here and full-sha checked BEFORE any comparison (p0l3b7c_verify);
  * the verdict is p0l3b7c_verify.decide (hash-pinned, next to this file) over the stored per-level arrays, which the
    launcher re-derives: every registered (quantity, level) of every DISCRIMINATING negative must be exceeded, K6b
    (entry_rebalance_nc) at the manifest k6b_target (A6a level 1, A6b level 2); a bit-identical mutant is recorded
    INEFFECTIVE (FAIL), never refused and never a pass; the frozen Gate-2 statistic must agree field by field;
  * raw_sed_n (K6c) is never run: the result carries the manifest's structural-only D* record; K6a is not a negative;
  * per-field, per-level abs / rel / fp32-ulp residuals; any crash exits 4, never the valid-FAIL code 1;
  * R2 (MANAGER_P0L3B7C_REAL_CRITIC_DECISION.md): the fp64 arm is the shipped force_fp64 carry -- State(**fields)
    casts Nc/nwfa/nifa to the ADR-007 fp32 gate, so the float64 inputs are re-applied with State.replace(_cast=False)
    and the product's _enforce_operational_precision(force_fp64=True) upcasts any remaining leaf; every
    STATE_FIELD_ORDER leaf except B39's WRF-REAL surface-layer carry must then be float64.  The production arm is the ADR-007 fp32-gated MEASUREMENT carry
    (DEFAULT_DTYPES), not the shipped force_fp64 default.  The dtype vector theta/qv/qc/Nc/nwfa/nifa/p_total entering
    the adapter must equal p0l3b7c_verify.STATE_DTYPES[arm] (else refusal 2) and is recorded for the verifier.
Usage: python3 p0l3b7c_runner.py --compare-module P --compare-sha256 S --manifest M --manifest-sha256 S --column C
         --npz port_column_C.npz --wrf-output OUTPUT_C.txt --tolerances MP28 --tolerances-sha256 S
         --gate2-tolerances G2 --gate2-tolerances-sha256 S --state-dtype production|fp64 --out RESULT.json --repo SNAP
       python3 p0l3b7c_runner.py --self-test --manifest M          (NumPy only; no JAX)
Exit: 0 PASS, 1 FAIL (valid, result written), 2 refusal (binding/scope; nothing written), 4 crash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import textwrap
import traceback
from pathlib import Path

import numpy as np

VERIFY_SHA = "8cc5f3837225717a9698c4be20379ca4cee1753831a8d9a4c9fdbfb5fed15bfb"   # p0l3b7c_verify.py
COMPARE_SHA = "ec317d0ee00e6814accbb6298a2d955e65fc376005fe2da2eb826d1e005f1a5d"  # frozen p0_gate2_compare.py
DRIVER_FIELDS = ("th", "pii", "p", "dz8w", "w", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa", "nifa")
OUT_FIELDS = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa", "nifa", "re_cloud", "re_ice", "re_snow")
R_DRY = 287.04                                                 # WRF rho = 0.622 p / (R T (qv + 0.622)), as the tool
# pre-K single-line substitutions (module key, function, anchor, replacement) -- rev-6 table minus raw_sed_n (K6c:
# structural only) and unqualified_limiter (K6a: structural gate)
SOURCE_MUTANTS = {
    "drop_cloud_sed_on": ("aero", "_sedimentation_aero", "_sed_cloud_water_aero(state, dt, aero, cloud_sed_on, cloud_rho)",
                          "_sed_cloud_water_aero(state, dt, aero, None, cloud_rho)"),
    "vtc_only_mask": ("aero", "_sed_cloud_water_aero",
                      "        vtnc = jnp.where(cloud_sed_on, vtnc, jnp.zeros_like(vtnc))\n", ""),
    "entry_rebalance_nc": ("aero", "_cloud_water_fall_speeds_aero",
                           "nc_m3 = jnp.maximum(2.0, jnp.minimum(state.Nc * rho, NT_C_MAX))",
                           "nc_m3 = _entry_cloud_number(state.qc, state.Nc, rho, aero)"),      # pre-K6b (b80a723cf)
    "mass_w": ("pc", "_thompson_aero_column_from_state", "w=_to_columns(state.w[:-1]),",
               "w=_to_columns(_w_mass(state)),"),                                              # pre-K2 (ba34d69cc)
}
SWAP_MUTANTS = ("gate_off", "all_lqc", "post_branch", "no_ksed")


class Refuse(Exception):
    pass


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path, sha, name):
    if sha256_file(path) != sha:
        raise Refuse(f"{name} sha256 mismatch")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_verify():
    return load_module(Path(__file__).resolve().with_name("p0l3b7c_verify.py"), VERIFY_SHA, "p0l3b7c_verify")


def parse_wrf_output(path):
    lines = Path(path).read_text().splitlines()
    if not lines or lines[0].strip() != "P0C3BOUT 1":
        raise Refuse("WRF output magic")
    head = lines[1].split()
    if len(head) != 5 or head[0] != "NZ":
        raise Refuse("WRF output NZ line")
    nz, dt, nwfa2d, nifa2d = int(head[1]), float(head[2]), float(head[3]), float(head[4])
    rec = {"IN": [], "OUT": [], "SFC": []}
    for ln in lines[2:]:
        parts = ln.split()
        if not parts or parts[0] not in rec:
            raise Refuse(f"WRF output unexpected record: {ln[:40]!r}")
        rec[parts[0]].append([float(x) for x in parts[1:]])
    if len(rec["IN"]) != nz or len(rec["OUT"]) != nz or len(rec["SFC"]) != 1:
        raise Refuse("WRF output record counts")
    for tag in ("IN", "OUT"):
        if [int(r[0]) for r in rec[tag]] != list(range(1, nz + 1)):
            raise Refuse(f"WRF output {tag} level order")
    a_in = np.array([r[1:] for r in rec["IN"]], np.float32)
    a_out = np.array([r[1:] for r in rec["OUT"]], np.float32)
    if a_in.shape != (nz, len(DRIVER_FIELDS)) or a_out.shape != (nz, len(OUT_FIELDS)):
        raise Refuse("WRF output column widths")
    return (nz, dt, np.float32(nwfa2d), np.float32(nifa2d), {n: a_in[:, i] for i, n in enumerate(DRIVER_FIELDS)},
            {n: a_out[:, i] for i, n in enumerate(OUT_FIELDS)}, np.array(rec["SFC"][0], np.float32))


def check_same_inputs(npz, nz, dt, nwfa2d, nifa2d, w_in):
    if int(npz["qv"].shape[0]) != nz or np.float32(npz["dt"]) != np.float32(dt):
        raise Refuse("nz/dt differ between npz and WRF output")
    if np.float32(npz["nwfa2d"]) != nwfa2d or np.float32(npz["nifa2d"]) != nifa2d:
        raise Refuse("nwfa2d/nifa2d differ between npz and WRF output")
    for name in DRIVER_FIELDS:
        if not np.array_equal(np.asarray(npz["driver_" + name], np.float32), w_in[name]):
            raise Refuse(f"WRF driver input echo differs from npz driver_{name}")


def source_mutant(module, fname, anchor, repl):
    """The product function's CURRENT source with exactly one registered substitution, executed in its module."""
    src = textwrap.dedent(inspect.getsource(getattr(module, fname)))
    if src.count(anchor) != 1:
        raise Refuse(f"mutant anchor for {fname} occurs {src.count(anchor)} times (product drift)")
    ns = dict(module.__dict__)
    exec(compile(src.replace(anchor, repl), f"<p0l3b7c mutant {fname}>", "exec"), ns)
    return ns[fname]


def patch_for(name, mods, jnp):
    if name == "raw_sed_n":
        raise Refuse("K6c scope misuse: raw_sed_n is structural D* evidence only, never a numeric mutant")
    if name in SOURCE_MUTANTS:
        key, fname, anchor, repl = SOURCE_MUTANTS[name]
        return mods[key], fname, source_mutant(mods[key], fname, anchor, repl)
    aero = mods["aero"]
    R1 = aero.R1
    lqc = {"gate_off": lambda qp, rp, qo, br: jnp.zeros(qp.shape[:-1] + (1,), bool),
           "all_lqc": lambda qp, rp, qo, br: jnp.all((qp > R1) & ~(br & (jnp.maximum(qo * rp, R1) == R1)), axis=-1,
                                                     keepdims=True),
           "post_branch": lambda qp, rp, qo, br: jnp.any(
               (qp > R1) & ~((qo > R1) & (jnp.maximum(qo * rp, R1) == R1)), axis=-1, keepdims=True)}
    if name in lqc:
        return aero, "_wrf_l_qc_any", lqc[name]
    if name == "no_ksed":
        return aero, "_wrf_cloud_sed_band", lambda rc, dz: (jnp.cumsum(dz, axis=-1) - dz) < 500.0
    raise Refuse(f"unknown mutant {name}")


def run_port(npz, repo, state_dtype, cmp, want_dtypes):
    """(fixed arm fields, kernel-direct fields, fixed_arm, mods, jax, jnp, meta) -- JAX imported here only."""
    sys.path.insert(0, str(Path(repo) / "src"))
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from gpuwrf.contracts.grid import BCMetadata, DycoreMetrics, GridSpec, Projection, TerrainProvenance, VerticalCoord
    from gpuwrf.contracts.precision import DEFAULT_DTYPES
    from gpuwrf.contracts.state import State, _state_field_shapes
    from gpuwrf.coupling import physics_couplers as pc
    from gpuwrf.physics import thompson_aero_column as aero
    from gpuwrf.contracts.precision import SURFACE_LAYER_CARRY_LEAVES, MYNN_DIAGNOSTIC_LEAVES
    from gpuwrf.runtime.operational_mode import STATE_FIELD_ORDER, _enforce_operational_precision

    nz = int(npz["qv"].shape[0])
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, 1, 1)
    terrain = TerrainProvenance(source_path="p0c3b", sha256="p0c3b", shape=(1, 1), units="m",
                                projection_transform="native-wrf-lambert", max_elevation_m=0.0,
                                coastline_sanity_check_passed=True)
    metrics = DycoreMetrics.flat(ny=1, nx=1, nz=nz, eta_levels=eta, top_pressure_pa=5000.0, provenance="p0c3b")
    grid = GridSpec(projection, terrain, VerticalCoord("hybrid_eta", nz, 5000.0, eta),
                    BCMetadata("ideal", (), 1, "linear", True), eta, jnp.zeros((1, 1), jnp.float64), metrics=metrics)
    shapes = _state_field_shapes(grid)

    def dt_of(name):
        if state_dtype == "fp64":
            return jnp.float64
        try:
            return DEFAULT_DTYPES.dtype_for(name)
        except KeyError:
            return jnp.float64

    col = lambda a: np.asarray(a, np.float64).reshape(-1, 1, 1)  # noqa: E731
    theta_dry, qv = col(npz["theta_dry"]), col(npz["qv"])
    values = {"theta": theta_dry * (1.0 + cmp.RV_OVER_RD * qv), "qv": qv, "qc": col(npz["qc"]), "qr": col(npz["qr"]),
              "qi": col(npz["qi"]), "qs": col(npz["qs"]), "qg": col(npz["qg"]), "Ni": col(npz["Ni"]),
              "Nr": col(npz["Nr"]), "Ns": col(npz["Ns"]), "Ng": col(npz["Ng"]), "Nc": col(npz["Nc"]),
              "nwfa": col(npz["nwfa"]), "nifa": col(npz["nifa"]), "p_total": col(npz["p"]), "ph_total": col(npz["ph"]),
              "w": col(npz["w"])}
    for name, arr in values.items():
        want = shapes.get(name, shapes["qc"] if name in ("nwfa", "nifa") else None)
        if want is None or tuple(want) != arr.shape:
            raise Refuse(f"State field {name}: shape {arr.shape} vs contract {want}")
    fields = {name: jnp.zeros(shape, dtype=dt_of(name)) for name, shape in shapes.items()}
    fields.update({name: jnp.asarray(arr, dtype=dt_of(name)) for name, arr in values.items()})
    state = State(**fields).ensure_conditional_leaves(mp_physics=28)
    if state_dtype == "fp64":             # R2: the shipped force_fp64 carry, no constructor fp32 gate on Nc/nwfa/nifa
        redo = {n: jnp.asarray(arr, dtype=jnp.float64) for n, arr in values.items()
                if getattr(state, n).dtype != jnp.float64}
        if redo:
            state = state.replace(_cast=False, **redo)
        state = _enforce_operational_precision(state, force_fp64=True)
        # WRF REAL surface and MYNN diagnostics stay REAL even in the fp64 arm.
        low = [f for f in STATE_FIELD_ORDER if f not in (*SURFACE_LAYER_CARRY_LEAVES, *MYNN_DIAGNOSTIC_LEAVES)
               and getattr(state, f, None) is not None and getattr(state, f).dtype != jnp.float64]
        if low:
            raise Refuse(f"fp64 arm: leaves not float64 after force_fp64: {low}")
    dtypes = {name: str(getattr(state, name).dtype) for name in ("theta", "qv", "qc", "Nc", "nwfa", "nifa", "p_total")}
    if dtypes != want_dtypes:
        raise Refuse(f"state dtype vector {dtypes} != the {state_dtype} arm's {want_dtypes}")
    dt = float(npz["dt"])
    nwfa2d_port = float(np.asarray(pc._aerosol_surface_emission_columns(state, grid)[0]).reshape(-1)[0])
    if np.float32(nwfa2d_port) != np.float32(npz["nwfa2d"]):
        raise Refuse(f"port nwfa2d {nwfa2d_port!r} does not round to the driver's {float(npz['nwfa2d'])!r}")

    kernel = pc.step_thompson_aero_column_with_precip
    if kernel is not aero.step_thompson_aero_column_with_precip:
        raise Refuse("physics_couplers kernel binding is not thompson_aero_column.step_thompson_aero_column_with_precip")
    flat = lambda a: np.asarray(a, np.float64).reshape(-1)  # noqa: E731
    p64 = flat(npz["p"])
    pi_port = (np.maximum(p64, 1.0) / float(pc.P0_PA)) ** float(pc.R_D_OVER_CP)

    def fixed_arm():
        seen = {}

        def spy(column, dt_, debug=False):
            out, precip = kernel(column, dt_, debug=debug)
            seen["T_in"], seen["T_out"] = np.asarray(column.T, np.float64), np.asarray(out.T, np.float64)
            return out, precip

        pc.step_thompson_aero_column_with_precip = spy
        try:
            nxt = pc.thompson_aero_adapter(state, dt, grid)
        finally:
            pc.step_thompson_aero_column_with_precip = kernel
        if "T_in" not in seen:
            raise Refuse("fixed arm did not reach the aero Thompson kernel")
        T_in, T_out = flat(seen["T_in"]), flat(seen["T_out"])
        return {"T": T_out, "qv": flat(nxt.qv), "qc": flat(nxt.qc), "dT": T_out - T_in, "dtheta_dry": (T_out - T_in) / pi_port,
                "theta_m": flat(nxt.theta), "nc": flat(nxt.Nc), "nwfa": flat(nxt.nwfa), "T_in": T_in}

    jax.clear_caches()
    fixed = fixed_arm()
    # kernel-direct attribution arm: WRF's own column inputs
    d = {k: np.asarray(npz["driver_" + k], np.float32).astype(np.float64) for k in DRIVER_FIELDS}
    Tw = d["th"] * d["pii"]
    rho_w = 0.622 * d["p"] / (R_DRY * Tw * (d["qv"] + 0.622))
    c1 = lambda a: jnp.asarray(np.asarray(a, np.float64).reshape(1, 1, -1))  # noqa: E731
    kcol = aero.ThompsonAeroColumnState(
        c1(d["qv"]), c1(d["qc"]), c1(d["qr"]), c1(d["qi"]), c1(d["qs"]), c1(d["qg"]), c1(d["ni"]), c1(d["nr"]),
        c1(d["nc"]), c1(d["nwfa"]), c1(d["nifa"]), c1(Tw), c1(d["p"]), c1(rho_w),
        Ns=c1(npz["Ns"]), Ng=c1(npz["Ng"]), dz=c1(d["dz8w"]), w=c1(d["w"]))
    kout, _ = kernel(kcol, dt, debug=False)
    kout = aero.apply_surface_aerosol_emission(kout, jnp.asarray([[float(npz["nwfa2d"])]]),
                                               jnp.asarray([[float(npz["nifa2d"])]]), dt)
    kd = {"T": flat(kout.T), "qv": flat(kout.qv), "qc": flat(kout.qc), "nc": flat(kout.Nc), "nwfa": flat(kout.nwfa)}
    mods = {"aero": aero, "pc": pc}
    code = {Path(m.__file__).name: sha256_file(m.__file__) for m in (pc, aero)}
    return fixed, kd, fixed_arm, mods, jax, jnp, {"state_dtypes": dtypes, "port_code_sha256": code,
                                                  "port_nwfa2d": nwfa2d_port}


def run(a):
    v = load_verify()
    if a.compare_sha256 != COMPARE_SHA:
        raise Refuse("compare sha is not the frozen cb78b4f6")
    m = v.load_manifest(a.manifest, a.manifest_sha256)                 # r7 31f7e9d4 only, fully re-checked
    tol = v.load_tolerances(a.tolerances, a.tolerances_sha256, a.gate2_tolerances, a.gate2_tolerances_sha256)
    if a.column not in v.COLUMNS or a.state_dtype not in ("production", "fp64"):
        raise Refuse("--column / --state-dtype")
    col = m["columns"][a.column]
    if Path(a.npz).name != f"port_column_{a.column}.npz" or \
            col["files"].get(Path(a.npz).name, {}).get("sha256") != sha256_file(a.npz):
        raise Refuse("constructed npz is not the manifest's")
    if Path(a.out).exists():
        raise Refuse("--out exists")
    cmp = load_module(a.compare_module, a.compare_sha256, "p0g2_compare_frozen")
    if cmp.T_WARM_K != v.T_WARM_K or tuple(cmp.GATED) != v.SHARED:
        raise Refuse("frozen compare warm mask / shared fields differ from the verifier's")
    npz = np.load(a.npz)
    nz, dt, nwfa2d, nifa2d, w_in, w_out, sfc = parse_wrf_output(a.wrf_output)
    check_same_inputs(npz, nz, dt, nwfa2d, nifa2d, w_in)
    if nz != m["nz"]:
        raise Refuse("nz differs from the manifest")
    wrf = cmp.wrf_fields(w_in, w_out, dt)
    wrf["nc"], wrf["nwfa"] = w_out["nc"].astype(np.float64), w_out["nwfa"].astype(np.float64)
    fixed, kd, fixed_arm, mods, jax, jnp, meta = run_port(npz, a.repo, a.state_dtype, cmp, v.STATE_DTYPES[a.state_dtype])
    muts = {}
    for name in v.numeric_discriminators(col):
        mod, attr, fn = patch_for(name, mods, jnp)
        orig = getattr(mod, attr)
        setattr(mod, attr, fn)
        try:
            jax.clear_caches()
            out = fixed_arm()
            muts[name] = {q: out[q] for q in v.DISC_Q}
        finally:
            setattr(mod, attr, orig)
    jax.clear_caches()
    e = lambda x: [v.enc(y) for y in np.asarray(x, np.float64).reshape(-1)]  # noqa: E731
    arrays = {"T_in": e(wrf["T_in"]), "wrf": {f: e(wrf[f]) for f in v.GATED}, "fixed": {f: e(fixed[f]) for f in v.GATED},
              "mutants": {n: {q: e(x) for q, x in qq.items()} for n, qq in muts.items()}}
    decision = v.decide(a.column, col, arrays, tol)
    mask = cmp.level_mask(wrf, v.LEVEL_MASK)
    for f in v.GATED:                                  # the frozen Gate-2 statistic must agree field by field
        if bool(cmp._field_stats(wrf[f], fixed[f], *tol[f], mask)["within"]) != decision["shared"][f]["within"]:
            raise Refuse(f"frozen Gate-2 statistic and the verifier disagree on {f}")
    res = {"schema": "p0l3b7c_result/1", "column": a.column, "state_dtype": a.state_dtype,
           "verdict": decision["verdict"], "decision": decision, "arrays": arrays, "k6c": v.k6c_record(m, a.column),
           "kernel_direct_max_abs_vs_wrf": {k: v.enc(np.max(np.abs(x - np.asarray(wrf[k], np.float64))))
                                            for k, x in kd.items()},
           "wrf_sfc": [v.enc(x) for x in sfc],
           "inputs": {"npz_sha256": sha256_file(a.npz), "wrf_output_sha256": sha256_file(a.wrf_output),
                      "tolerances_sha256": a.tolerances_sha256, "gate2_tolerances_sha256": a.gate2_tolerances_sha256,
                      "compare_sha256": a.compare_sha256, "manifest_sha256": a.manifest_sha256,
                      "verify_sha256": VERIFY_SHA}, **meta}
    with open(a.out, "x") as f:
        f.write(json.dumps(res, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"column": a.column, "dtype": a.state_dtype, "verdict": decision["verdict"],
                      "shared": decision["shared_ok"], "bounds": decision["bounds_ok"],
                      "mutants": {n: x["status"] for n, x in decision["mutants"].items()}}))
    return 0 if decision["verdict"] == "PASS" else 1


def self_test(manifest):
    import tempfile
    n = 0
    v = load_verify()
    n += 1; assert set(SOURCE_MUTANTS) | set(SWAP_MUTANTS) == set(v.NUMERIC) and "raw_sed_n" not in SOURCE_MUTANTS
    try:
        patch_for("raw_sed_n", {}, None)
        raise AssertionError("no refusal")
    except Refuse as e:
        n += 1; assert "K6c scope misuse" in str(e)
    # the source-substitution mechanism: exactly-once anchor, re-executed in the module namespace
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "p0l3b7c_st_mod.py").write_text("K = 2.0\n\ndef f(x):\n    y = x * K\n    return y + 1.0\n\n"
                                                    "def g(x):\n    return x - 1.0\n    return x - 1.0\n")
        sys.path.insert(0, td)
        import p0l3b7c_st_mod as sm
        fm = source_mutant(sm, "f", "    y = x * K\n", "    y = x * K * 10.0\n")
        n += 1; assert fm(1.0) == 21.0 and sm.f(1.0) == 3.0            # mutant uses the module global K; product intact
        for fn, anc in (("f", "not there"), ("g", "return x - 1.0")):
            try:
                source_mutant(sm, fn, anc, "")
                raise AssertionError("no refusal")
            except Refuse as e:
                n += 1; assert "occurs" in str(e)
    # parser: magic, widths, level order
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "o.txt"
        rows = ["P0C3BOUT 1", "NZ 2 1.8e+01 1.0e+00 0.0e+00"]
        rows += [f"IN {k} " + " ".join(["1.0"] * 16) for k in (1, 2)] + [f"OUT {k} " + " ".join(["2.0"] * 15) for k in (1, 2)]
        rows += ["SFC " + " ".join(["0.0"] * 7)]
        p.write_text("\n".join(rows) + "\n")
        nz, dt, a2, b2, wi, wo, _ = parse_wrf_output(p)
        n += 1; assert nz == 2 and wi["nifa"][1] == 1.0 and wo["re_snow"][0] == 2.0 and a2 == 1.0
        for bad in (rows[:2] + rows[3:2:-1] + rows[3:], ["P0C3BOUT 2"] + rows[1:], rows[:-1]):
            p.write_text("\n".join(bad) + "\n")
            try:
                parse_wrf_output(p)
                raise AssertionError("no refusal")
            except Refuse:
                n += 1
    k = v.self_test(manifest)                                          # the verdict rule on the real r7 manifest
    print(f"P0L3B7C_RUNNER SELF-TEST PASS checks={n + k}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    keys = ("compare-module", "compare-sha256", "manifest", "manifest-sha256", "column", "npz", "wrf-output",
            "tolerances", "tolerances-sha256", "gate2-tolerances", "gate2-tolerances-sha256", "state-dtype", "out", "repo")
    for k in keys:
        ap.add_argument(f"--{k}")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test(a.manifest)
    try:
        missing = [k for k in keys if not getattr(a, k.replace("-", "_"))]
        if missing:
            raise Refuse(f"missing arguments: {missing}")
        return run(a)
    except Exception as exc:                               # a refusal (ours or the verifier's) = 2; anything else = 4
        if type(exc).__name__ == "Refuse":
            print(f"P0L3B7C_RUNNER REFUSE: {exc}", file=sys.stderr)
            return 2
        traceback.print_exc()
        return 4


if __name__ == "__main__":
    sys.exit(main())
