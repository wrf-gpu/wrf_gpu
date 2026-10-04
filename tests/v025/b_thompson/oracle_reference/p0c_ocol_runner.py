# Provenance: read-only source commit 83c7ff126; original SHA256 e70819d97464e8c671c0f9524e14154c5de761e869930f81fb9e6443283d6a32
# Source: closure/l3/p0c_ocol_runner.py
"""P0 closure O-C runner: one constructed mp=8 WRF column, one state-dtype arm (contract rev 2 §3.3).

Uses the FROZEN Gate-2 compare (cb78b4f6) unchanged for the State build, the WRF field derivation and the frozen-tolerance
verdict (fixed arm + frozen wrong-convention mutant, frozen tolerances 5bdc03da).  Adds the pre-registered negatives of
the committed manifest p0c_column_discriminators.json (sha256 on the CLI): the fixed arm must stay within every
DISCRIMINATING bound, and every DISCRIMINATING mutant -- the port with ONE product function replaced in memory (no product
edit), recompiled after jax.clear_caches() -- must exceed its bound on at least one of its levels.  A mutant whose qc is
bit-identical to the fixed arm is refused as PATCH_INEFFECTIVE (exit 2), so a stale compiled kernel can never pass.
Mutants: gate_off / rev0_ungated / all_lqc / post_branch (thompson_column._wrf_l_qc_any), no_ksed (_wrf_cloud_sed_band ->
the old strict below-500 m mask), post_rho (_cloud_sed_rho_stages -> post-adjustment rho everywhere), mass_w / g_old
(physics_couplers._thompson_column_from_state -> mass-point w / dz from g = 9.80665).  UNDISCRIMINABLE mutants are
reported, never run as a pass.
Usage: python3 p0c_ocol_runner.py --compare-module PATH --compare-sha256 SHA --manifest M.json --manifest-sha256 SHA
         --column Ck --npz Ck.npz --wrf-output OUTPUT_Ck.txt --tolerances TOL --tolerances-sha256 SHA
         --state-dtype production|fp64 --out RESULT.json --repo SNAPSHOT
       python3 p0c_ocol_runner.py --self-test          (numpy only)
Exit: 0 PASS, 1 FAIL (valid), 2 refusal (incl. binding mismatch, PATCH_INEFFECTIVE; nothing written).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

MUTANTS = ("gate_off", "rev0_ungated", "all_lqc", "post_branch", "no_ksed", "post_rho", "mass_w", "g_old")


class Refuse(Exception):
    pass


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path, sha, column, npz_path):
    if sha256_file(path) != sha:
        raise Refuse("manifest sha256 mismatch")
    m = json.loads(Path(path).read_text())
    if m.get("schema") != "p0c_column_manifest/1" or column not in m.get("columns", {}):
        raise Refuse("manifest schema/column")
    c = m["columns"][column]
    want = c["files"].get(f"port_column_{column}.npz", {}).get("sha256")
    if want != sha256_file(npz_path):
        raise Refuse("constructed npz is not the pre-registered one")
    for name, d in c["discriminators"].items():
        if name not in MUTANTS or d.get("status") not in ("DISCRIMINATING", "UNDISCRIMINABLE"):
            raise Refuse(f"manifest mutant {name!r}")
    return m, c


def disc_check(qc_wrf, qc_fixed, mutants_qc, discs):
    """Pure numpy discriminator verdict.  discs: manifest discriminators; mutants_qc: name -> port qc (DISCRIMINATING)."""
    qc_wrf, qc_fixed = np.asarray(qc_wrf, np.float64), np.asarray(qc_fixed, np.float64)
    fixed, mut = {}, {}
    for name, d in discs.items():
        if d["status"] != "DISCRIMINATING":
            mut[name] = {"status": "UNDISCRIMINABLE", "reason": d.get("reason")}
            continue
        lv = {int(k) - 1: float(b) for k, b in d["bound"].items()}
        fixed[name] = {str(k + 1): {"err": float(abs(qc_fixed[k] - qc_wrf[k])), "bound": b,
                                    "within": bool(abs(qc_fixed[k] - qc_wrf[k]) <= b)} for k, b in lv.items()}
        q = np.asarray(mutants_qc[name], np.float64)
        if np.array_equal(q, qc_fixed):
            raise Refuse(f"PATCH_INEFFECTIVE: mutant {name} is bit-identical to the fixed arm")
        errs = {str(k + 1): float(abs(q[k] - qc_wrf[k])) for k in lv}
        mut[name] = {"status": "DISCRIMINATING", "err": errs, "exceeds": any(errs[str(k + 1)] > b for k, b in lv.items())}
    fixed_ok = all(v["within"] for f in fixed.values() for v in f.values())
    mut_ok = all(v["exceeds"] for v in mut.values() if v["status"] == "DISCRIMINATING")
    return fixed_ok, mut_ok, fixed, mut


def patch_for(name, pc, tc, jnp):
    R1 = tc.R1
    lqc = {"gate_off": lambda qp, rp, qo, br: jnp.zeros(qp.shape[:-1] + (1,), bool),
           "rev0_ungated": lambda qp, rp, qo, br: jnp.ones(qp.shape[:-1] + (1,), bool),
           "all_lqc": lambda qp, rp, qo, br: jnp.all((qp > R1) & ~(br & (jnp.maximum(qo * rp, R1) == R1)), axis=-1, keepdims=True),
           "post_branch": lambda qp, rp, qo, br: jnp.any(
               (qp > R1) & ~((qo > R1) & (jnp.maximum(qo * rp, R1) == R1)), axis=-1, keepdims=True)}
    if name in lqc:
        return tc, "_wrf_l_qc_any", lqc[name]
    if name == "no_ksed":
        return tc, "_wrf_cloud_sed_band", lambda rc, dz: (jnp.cumsum(dz, axis=-1) - dz) < 500.0
    if name == "post_rho":
        return tc, "_cloud_sed_rho_stages", lambda rp, qrp, st: (st.rho, st.rho, st.rho)
    builder = pc._thompson_column_from_state
    if name == "mass_w":
        return pc, "_thompson_column_from_state", lambda s, g=None: builder(s, g).replace(w=pc._to_columns(pc._w_mass(s)))
    if name == "g_old":
        return pc, "_thompson_column_from_state", lambda s, g=None: builder(s, g).replace(dz=pc._column_dz_from_state(s, g))
    raise Refuse(f"unknown mutant {name}")


def run(a):
    m, c = load_manifest(a.manifest, a.manifest_sha256, a.column, a.npz)
    if a.state_dtype not in ("production", "fp64"):
        raise Refuse("--state-dtype")
    if Path(a.out).exists():
        raise Refuse("--out exists")
    if sha256_file(a.compare_module) != a.compare_sha256:
        raise Refuse("compare sha256")
    spec = importlib.util.spec_from_file_location("p0g2_compare_frozen", a.compare_module)
    cmp = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cmp
    spec.loader.exec_module(cmp)
    tol, tol_sha = cmp.load_tolerances(a.tolerances, a.tolerances_sha256)
    npz = np.load(a.npz)
    nz, dt, w_in, w_out, _sfc = cmp.parse_wrf_output(a.wrf_output)
    cmp.check_same_inputs(npz, nz, dt, w_in)
    wrf = cmp.wrf_fields(w_in, w_out, dt)
    sys.path.insert(0, str(Path(a.repo) / "src"))
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from gpuwrf.coupling import physics_couplers as pc
    from gpuwrf.physics import thompson_column as tc

    jax.clear_caches()
    arms, meta = cmp.run_port_arms(npz, a.repo, a.state_dtype)
    frozen = cmp.gate(wrf, arms, tol)
    qc_fixed = np.asarray(arms["fixed"]["qc"], np.float64)
    mutants_qc = {}
    for name, d in c["discriminators"].items():
        if d["status"] != "DISCRIMINATING":
            continue
        mod, attr, fn = patch_for(name, pc, tc, jnp)
        orig = getattr(mod, attr)
        setattr(mod, attr, fn)
        try:
            jax.clear_caches()
            marms, _ = cmp.run_port_arms(npz, a.repo, a.state_dtype)
        finally:
            setattr(mod, attr, orig)
        mutants_qc[name] = np.asarray(marms["fixed"]["qc"], np.float64)
    jax.clear_caches()
    fixed_ok, mut_ok, fixed, mut = disc_check(wrf["qc"], qc_fixed, mutants_qc, c["discriminators"])
    verdict = "PASS" if (frozen["verdict"] == "PASS" and fixed_ok and mut_ok) else "FAIL"
    res = {"schema": "p0c_ocol_result/1", "column": a.column, "state_dtype": a.state_dtype, "verdict": verdict,
           "frozen_gate2": {k: frozen[k] for k in ("verdict", "fixed_all_within", "mutant_exceeds_T_or_qc", "arms")},
           "discriminators_fixed_within": fixed_ok, "discriminating_mutants_exceed": mut_ok,
           "fixed_vs_bounds": fixed, "mutants": mut,
           "inputs": {"npz_sha256": sha256_file(a.npz), "wrf_output_sha256": sha256_file(a.wrf_output),
                      "tolerances_sha256": tol_sha, "compare_sha256": a.compare_sha256,
                      "manifest_sha256": a.manifest_sha256}, **meta}
    with open(a.out, "x") as f:
        f.write(json.dumps(res, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"column": a.column, "dtype": a.state_dtype, "verdict": verdict, "frozen": frozen["verdict"],
                      "fixed_within": fixed_ok, "mutants_exceed": mut_ok}))
    return 0 if verdict == "PASS" else 1


def self_test():
    n = 0
    wrf = np.array([1e-3, 0.0, 5e-4])
    discs = {"gate_off": {"status": "DISCRIMINATING", "bound": {"1": 1e-6, "2": 1e-12}},
             "g_old": {"status": "UNDISCRIMINABLE", "reason": "x"}}
    ok_f, ok_m, _, mut = disc_check(wrf, wrf + [1e-9, 0, 0], {"gate_off": wrf + [1e-5, 0, 0]}, discs); n += 1
    assert ok_f and ok_m and mut["g_old"]["status"] == "UNDISCRIMINABLE"
    ok_f, ok_m, _, _ = disc_check(wrf, wrf + [2e-6, 0, 0], {"gate_off": wrf + [1e-5, 0, 0]}, discs); n += 1
    assert not ok_f                                                    # fixed outside its bound -> FAIL
    ok_f, ok_m, _, _ = disc_check(wrf, wrf, {"gate_off": wrf + [5e-7, 0, 0]}, discs); n += 1
    assert ok_f and not ok_m                                           # mutant inside the bound -> FAIL
    ok_f, ok_m, _, _ = disc_check(wrf, wrf, {"gate_off": wrf + [0, 3e-11, 0]}, discs); n += 1
    assert ok_m                                                        # the exact-zero receiving level separates it
    try:
        disc_check(wrf, wrf + [1e-9, 0, 0], {"gate_off": wrf + [1e-9, 0, 0]}, discs)
        raise AssertionError("no refusal")
    except Refuse as e:
        assert "PATCH_INEFFECTIVE" in str(e); n += 1
    print(f"P0C_OCOL_RUNNER SELF-TEST PASS checks={n}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    keys = ("compare-module", "compare-sha256", "manifest", "manifest-sha256", "column", "npz", "wrf-output",
            "tolerances", "tolerances-sha256", "state-dtype", "out", "repo")
    for k in keys:
        ap.add_argument(f"--{k}")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    try:
        missing = [k for k in keys if not getattr(a, k.replace("-", "_"))]
        if missing:
            raise Refuse(f"missing arguments: {missing}")
        return run(a)
    except Refuse as exc:
        print(f"P0C_OCOL REFUSE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
