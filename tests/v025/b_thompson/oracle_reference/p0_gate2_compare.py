# Provenance: read-only source commit 83c7ff126; original SHA256 cb78b4f6eb65e507bd7c56f47fa7f3a2e8db6e8dc875f74cffa34df917d85e96
# Source: gate2/p0_gate2_compare.py
"""P0 gate 2: WRF Thompson column (production objects) vs the port coupler, plus a wrong-convention mutant.

Contract: P0_MOIST_THETA_CONTRACT.md gate 2; p0/P0_SUBTASK_G2.md deliverable 4.

Inputs
  --npz          port_column.npz from p0_gate2_extract_column.py (same column the WRF driver ran)
  --wrf-output   OUTPUT.txt of p0_thompson_column_driver (IN echo, OUT per level, SFC)
  --tolerances   frozen gate2_tolerances.json (default: next to this script)
  --tolerances-sha256  REQUIRED full sha256 of that file; missing file or mismatch -> refuse (exit 2)
Arms (one Thompson step at the driver's dt)
  (a) fixed : production ``physics_couplers.thompson_adapter`` on a 1x1 State with
              theta = theta_m = theta_dry * (1 + Rv/Rd qv)  (Rv/Rd = 461.6/287, physics_couplers.WRF_RV_OVER_RD).
  (b) mutant: the pre-P0 convention rebuilt HERE (no product edit): the kernel column gets T = theta_m * pi and
              rho from that T, and theta is written back as T'/pi without moist recoupling.
  Both arms call the same production kernel ``thompson_column.step_thompson_column_with_precip``.
Compared fields (every level k = 1..nz), WRF side from the driver output (float32 values, fp64 arithmetic):
  T          = th_out * pii                              port: kernel output T (dry, as the kernel produced it)
  qv, qc     = driver outputs                            port: next State qv, qc
  dT         = (th_out - th_in) * pii                    port: kernel T_out - kernel T_in
  dtheta_dry = th_out - th_in                            port: (T_out - T_in) / pi_port        [fixed]
                                                                theta_m'/(1+R qv') - theta_m/(1+R qv) [mutant]
  theta_m    = WRF moist_physics_finish_em (module_big_step_utilities_em.F:5728-5741, use_theta_m=1):
               th_in(1+R qv_in) + mpten(1+R qv_out) + R (qv_out-qv_in) th_out,  mpten = th_out-th_in clamped to
               +-mp_tend_lim*dt (mp_tend_lim = 10 K/s Registry default)      port: next State theta (theta_m)
  (The full product th_out*(1+R qv_out) named in P0_SUBTASK_G2.md equals WRF's form minus
   R*(th_out-th_in)*(qv_out-qv_in); it is reported as ``theta_m_product`` but is not gated.)
Tolerance test per field: |port - wrf| <= atol + rtol*|wrf| at every GATED level.  The frozen tolerance file
chooses the gated levels, pre-registered: "level_mask": "all" or "warm_T_in_gt_275K" (WRF input th*pii > 275 K,
the criterion's warm threshold).  All-level statistics are always reported (ungated) under "all_levels".
PASS rule (pre-registered): fixed arm within tolerance on EVERY field AND mutant exceeds tolerance on T or qc.
Exit: 0 PASS, 1 FAIL, 2 refusal.  --out JSON must not exist.

Run (later, under a GO; JAX CPU, 1 column, expect < 2 min dominated by the Thompson kernel trace/compile):
  taskset -c <pair> nice -n 19 env LC_ALL=C JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python3 p0_gate2_compare.py --npz COL/port_column.npz --wrf-output RUN/OUTPUT.txt \
    --tolerances-sha256 <frozen sha> --out RUN/gate2_result.json
Self-test (no WRF, no real data, no JAX): python3 p0_gate2_compare.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_TOL = HERE / "gate2_tolerances.json"
GATED = ("T", "qv", "qc", "dT", "dtheta_dry", "theta_m")
LEVEL_MASKS = ("all", "warm_T_in_gt_275K")
T_WARM_K = 275.0
DRIVER_FIELDS = ("th", "pii", "p", "dz8w", "w", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")
OUT_FIELDS = ("th", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "re_cloud", "re_ice", "re_snow")
SFC_FIELDS = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv", "sr")
RV_OVER_RD = 461.6 / 287.0     # WRF R_v/R_d
MP_TEND_LIM = 10.0             # K/s, Registry.EM_COMMON:2646 default; Canary namelist does not set it


class Refuse(Exception):
    """Fail-closed refusal (exit 2); raised before any gate verdict can be produced."""

    def __init__(self, msg):
        super().__init__(f"P0G2_COMPARE REFUSE: {msg}")


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ tolerances
def load_tolerances(path, expected_sha):
    if not expected_sha or len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
        raise Refuse("--tolerances-sha256 must be a full lowercase sha256")
    path = Path(path)
    if not path.is_file():
        raise Refuse(f"tolerance file missing: {path}")
    got = sha256_file(path)
    if got != expected_sha:
        raise Refuse(f"tolerance file sha256 {got} != {expected_sha}")
    tol = json.loads(path.read_text())
    fields = tol.get("fields")
    if tol.get("schema") != "p0_gate2_tolerances/1" or not isinstance(fields, dict) or set(fields) != set(GATED):
        raise Refuse(f"tolerance schema must be p0_gate2_tolerances/1 with fields exactly {GATED}")
    if tol.get("level_mask") not in LEVEL_MASKS:
        raise Refuse(f"tolerance level_mask must be one of {LEVEL_MASKS}")
    for name, spec in fields.items():
        for key in ("atol", "rtol"):
            val = spec.get(key)
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise Refuse(f"tolerance {name}.{key} must be a finite number >= 0")
    out = {name: (float(spec["atol"]), float(spec["rtol"])) for name, spec in fields.items()}
    out["level_mask"] = tol["level_mask"]
    return out, got


# ------------------------------------------------------------------ WRF driver output
def parse_wrf_output(path):
    lines = Path(path).read_text().splitlines()
    if not lines or lines[0].strip() != "P0G2OUT 1":
        raise Refuse("WRF output magic")
    head = lines[1].split()
    if head[0] != "NZ":
        raise Refuse("WRF output NZ line")
    nz, dt = int(head[1]), float(head[2])
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
    arr_in = np.array([r[1:] for r in rec["IN"]], np.float32)
    arr_out = np.array([r[1:] for r in rec["OUT"]], np.float32)
    if arr_in.shape != (nz, len(DRIVER_FIELDS)) or arr_out.shape != (nz, len(OUT_FIELDS)):
        raise Refuse("WRF output column widths")
    w_in = {n: arr_in[:, c] for c, n in enumerate(DRIVER_FIELDS)}
    w_out = {n: arr_out[:, c] for c, n in enumerate(OUT_FIELDS)}
    sfc = dict(zip(SFC_FIELDS, np.array(rec["SFC"][0], np.float32)))
    return nz, dt, w_in, w_out, sfc


def check_same_inputs(npz, nz, dt, w_in):
    """The WRF driver's echoed inputs must be float32-identical to the extractor's driver_* arrays."""
    if int(npz["qv"].shape[0]) != nz:
        raise Refuse("nz differs between npz and WRF output")
    if np.float32(npz["dt"]) != np.float32(dt):
        raise Refuse("dt differs between npz and WRF output")
    for name in DRIVER_FIELDS:
        if not np.array_equal(np.asarray(npz["driver_" + name], np.float32), w_in[name]):
            raise Refuse(f"WRF driver input echo differs from npz driver_{name}")


def wrf_fields(w_in, w_out, dt):
    d = lambda a: np.asarray(a, np.float64)  # noqa: E731
    th0, th1, pii = d(w_in["th"]), d(w_out["th"]), d(w_in["pii"])
    qv0, qv1 = d(w_in["qv"]), d(w_out["qv"])
    mpten = np.clip(th1 - th0, -MP_TEND_LIM * dt, MP_TEND_LIM * dt)
    return {
        "T": th1 * pii,
        "qv": qv1,
        "qc": d(w_out["qc"]),
        "dT": (th1 - th0) * pii,
        "dtheta_dry": th1 - th0,
        "theta_m": th0 * (1.0 + RV_OVER_RD * qv0) + mpten * (1.0 + RV_OVER_RD * qv1) + RV_OVER_RD * (qv1 - qv0) * th1,
        "theta_m_product": th1 * (1.0 + RV_OVER_RD * qv1),
        "T_in": th0 * pii,
    }


# ------------------------------------------------------------------ gate
def level_mask(wrf, kind):
    mask = np.ones_like(wrf["T_in"], dtype=bool) if kind == "all" else np.asarray(wrf["T_in"]) > T_WARM_K
    if not mask.any():
        raise Refuse(f"level mask {kind} selects no level")
    return mask


def _field_stats(ref, val, atol, rtol, mask):
    ref = np.asarray(ref, np.float64)
    err = np.abs(np.asarray(val, np.float64) - ref)
    excess = np.where(mask, err - (atol + rtol * np.abs(ref)), -np.inf)
    excess = np.where(mask & ~np.isfinite(err), np.inf, excess)          # NaN/inf on a gated level never passes
    k = int(np.argmax(excess))
    return {"max_abs_err": float(np.max(np.where(mask, err, 0.0))), "worst_level_1based": k + 1,
            "max_excess": float(excess[k]), "within": bool(np.all(excess <= 0.0))}


def gate(wrf, arms, tol):
    mask = level_mask(wrf, tol["level_mask"])
    everywhere = np.ones_like(mask)
    report, all_levels = {}, {}
    for arm, fields in arms.items():
        report[arm] = {n: _field_stats(wrf[n], fields[n], *tol[n], mask) for n in GATED}
        all_levels[arm] = {n: _field_stats(wrf[n], fields[n], *tol[n], everywhere) for n in GATED}
    fixed_ok = all(r["within"] for r in report["fixed"].values())
    mutant_fails = (not report["mutant"]["T"]["within"]) or (not report["mutant"]["qc"]["within"])
    return {"verdict": "PASS" if (fixed_ok and mutant_fails) else "FAIL",
            "fixed_all_within": fixed_ok, "mutant_exceeds_T_or_qc": mutant_fails,
            "level_mask": tol["level_mask"], "gated_levels_1based": [int(k) + 1 for k in np.flatnonzero(mask)],
            "arms": report, "all_levels": all_levels}


# ------------------------------------------------------------------ port arms (JAX; imported lazily)
def run_port_arms(npz, repo, state_dtype):
    sys.path.insert(0, str(Path(repo) / "src"))
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from gpuwrf.contracts.grid import BCMetadata, DycoreMetrics, GridSpec, Projection, TerrainProvenance, VerticalCoord
    from gpuwrf.contracts.precision import DEFAULT_DTYPES
    from gpuwrf.contracts.state import State, _state_field_shapes
    from gpuwrf.coupling import physics_couplers as pc
    from gpuwrf.physics import thompson_column as tcol

    if abs(pc.WRF_RV_OVER_RD - RV_OVER_RD) > 0.0:
        raise Refuse("port WRF_RV_OVER_RD differs from 461.6/287")
    nz = int(npz["qv"].shape[0])
    eta = jnp.linspace(1.0, 0.0, nz + 1, dtype=jnp.float64)
    projection = Projection("lambert", 28.3, -16.4, 3000.0, 3000.0, 1, 1)
    terrain = TerrainProvenance(source_path="p0-gate2", sha256="p0-gate2", shape=(1, 1), units="m",
                                projection_transform="native-wrf-lambert", max_elevation_m=0.0,
                                coastline_sanity_check_passed=True)
    metrics = DycoreMetrics.flat(ny=1, nx=1, nz=nz, eta_levels=eta, top_pressure_pa=5000.0, provenance="p0-gate2")
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
    theta_dry = col(npz["theta_dry"])
    qv = col(npz["qv"])
    values = {
        "theta": theta_dry * (1.0 + RV_OVER_RD * qv),
        "qv": qv, "qc": col(npz["qc"]), "qr": col(npz["qr"]), "qi": col(npz["qi"]), "qs": col(npz["qs"]),
        "qg": col(npz["qg"]), "Ni": col(npz["Ni"]), "Nr": col(npz["Nr"]), "Ns": col(npz["Ns"]), "Ng": col(npz["Ng"]),
        "p_total": col(npz["p"]), "ph_total": col(npz["ph"]), "w": col(npz["w"]),
    }
    for name, arr in values.items():
        if name not in shapes or tuple(shapes[name]) != arr.shape:
            raise Refuse(f"State field {name}: shape {arr.shape} vs contract {shapes.get(name)}")
    fields = {name: jnp.zeros(shape, dtype=dt_of(name)) for name, shape in shapes.items()}
    fields.update({name: jnp.asarray(arr, dtype=dt_of(name)) for name, arr in values.items()})
    state = State(**fields)
    dt = float(npz["dt"])

    kernel = pc.step_thompson_column_with_precip
    if kernel is not tcol.step_thompson_column_with_precip:
        raise Refuse("physics_couplers kernel binding is not thompson_column.step_thompson_column_with_precip")
    seen = {}

    def spy(column, dt_, debug=False):
        out, precip = kernel(column, dt_, debug=debug)
        seen["T_in"] = np.asarray(column.T, np.float64)
        seen["T_out"] = np.asarray(out.T, np.float64)
        return out, precip

    pc.step_thompson_column_with_precip = spy
    try:
        nxt = pc.thompson_adapter(state, dt, grid)
    finally:
        pc.step_thompson_column_with_precip = kernel
    if "T_in" not in seen:
        raise Refuse("fixed arm did not reach the Thompson kernel")

    flat = lambda a: np.asarray(a, np.float64).reshape(-1)  # noqa: E731
    p64 = flat(npz["p"])
    pi_port = (np.maximum(p64, 1.0) / float(pc.P0_PA)) ** float(pc.R_D_OVER_CP)
    T_in, T_out = flat(seen["T_in"]), flat(seen["T_out"])
    fixed = {"T": T_out, "qv": flat(nxt.qv), "qc": flat(nxt.qc), "dT": T_out - T_in,
             "dtheta_dry": (T_out - T_in) / pi_port, "theta_m": flat(nxt.theta), "T_in": T_in}

    # (b) mutant: pre-P0 column view, rebuilt from the same State (old physics_couplers.py:1396/1433 convention)
    T_m = pc._temperature_from_theta(state.theta, state.p)          # theta_m * pi (the defect)
    rho_m = tcol.density_from_pressure_temperature(state.p, T_m, state.qv)
    column = tcol.ThompsonColumnState(
        pc._to_columns(state.qv), pc._to_columns(state.qc), pc._to_columns(state.qr), pc._to_columns(state.qi),
        pc._to_columns(state.qs), pc._to_columns(state.qg), pc._to_columns(state.Ni), pc._to_columns(state.Nr),
        pc._to_columns(T_m), pc._to_columns(state.p), pc._to_columns(rho_m),
        Ns=pc._to_columns(state.Ns), Ng=pc._to_columns(state.Ng),
        dz=pc._column_dz_from_state(state, grid), w=pc._to_columns(pc._w_mass(state)))
    out_m, _ = kernel(column, dt, debug=False)
    Tm_in, Tm_out = flat(pc._from_columns(column.T)), flat(pc._from_columns(out_m.T))
    qv_m1 = flat(pc._from_columns(out_m.qv))
    theta_m1 = Tm_out / pi_port                                          # written back as theta, no recoupling
    theta_m0 = flat(state.theta)
    mutant = {"T": Tm_out, "qv": qv_m1, "qc": flat(pc._from_columns(out_m.qc)), "dT": Tm_out - Tm_in,
              "dtheta_dry": theta_m1 / (1.0 + RV_OVER_RD * qv_m1) - theta_m0 / (1.0 + RV_OVER_RD * flat(state.qv)),
              "theta_m": theta_m1, "T_in": Tm_in}
    dtypes = {name: str(getattr(state, name).dtype) for name in ("theta", "qv", "qc", "p_total")}
    code = {Path(m.__file__).name: sha256_file(m.__file__) for m in (pc, tcol)}   # exact product code exercised
    return {"fixed": fixed, "mutant": mutant}, {"state_dtypes": dtypes, "port_code_sha256": code}


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="P0 gate 2 WRF-vs-port Thompson column compare")
    ap.add_argument("--npz")
    ap.add_argument("--wrf-output")
    ap.add_argument("--tolerances", default=str(DEFAULT_TOL))
    ap.add_argument("--tolerances-sha256")
    ap.add_argument("--out")
    ap.add_argument("--state-dtype", choices=("production", "fp64"), default="production")
    ap.add_argument("--repo", default=str(HERE.parents[4]))
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    tol, tol_sha = load_tolerances(a.tolerances, a.tolerances_sha256)   # refuses before anything else runs
    if not (a.npz and a.wrf_output and a.out):
        raise Refuse("--npz, --wrf-output and --out are required")
    if os.path.exists(a.out):
        raise Refuse("--out exists")
    npz = np.load(a.npz)
    nz, dt, w_in, w_out, sfc = parse_wrf_output(a.wrf_output)
    check_same_inputs(npz, nz, dt, w_in)
    wrf = wrf_fields(w_in, w_out, dt)
    arms, dtypes = run_port_arms(npz, a.repo, a.state_dtype)
    result = gate(wrf, arms, tol)
    result.update({
        "schema": "p0_gate2_result/1",
        "inputs": {"npz": a.npz, "npz_sha256": sha256_file(a.npz), "wrf_output": a.wrf_output,
                   "wrf_output_sha256": sha256_file(a.wrf_output), "tolerances": a.tolerances,
                   "tolerances_sha256": tol_sha},
        "nz": nz, "dt_s": dt, "state_dtype": a.state_dtype, **dtypes,
        "tolerances": {k: {"atol": tol[k][0], "rtol": tol[k][1]} for k in GATED},
        "diagnostics": {
            "input_T_port_minus_wrf_max_abs_K": float(np.max(np.abs(arms["fixed"]["T_in"] - wrf["T_in"]))),
            "wrf_theta_m_product_minus_finish_max_abs_K": float(np.max(np.abs(wrf["theta_m_product"] - wrf["theta_m"]))),
            "wrf_dT_max_K": float(np.max(wrf["dT"])), "wrf_qc_out_max": float(np.max(wrf["qc"])),
            "wrf_sfc": {k: float(v) for k, v in sfc.items()},
        },
    })
    with open(a.out, "x", encoding="ascii") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({"verdict": result["verdict"], "fixed_all_within": result["fixed_all_within"],
                      "mutant_exceeds_T_or_qc": result["mutant_exceeds_T_or_qc"]}))
    return 0 if result["verdict"] == "PASS" else 1


# ------------------------------------------------------------------ self-test (synthetic; no WRF, no real data)
def self_test():
    nz, dt = 6, 18.0
    rng = np.random.default_rng(0)
    w_in = {n: np.zeros(nz, np.float32) for n in DRIVER_FIELDS}
    w_in.update(th=np.float32(290.0 + np.arange(nz)), pii=np.float32(0.98 - 0.01 * np.arange(nz)),
                p=np.float32(93000.0 - 3000.0 * np.arange(nz)), dz8w=np.full(nz, 200.0, np.float32),
                qv=np.float32(0.014 - 0.001 * np.arange(nz)), nr=np.full(nz, 1.0e3, np.float32))
    w_out = {n: np.zeros(nz, np.float32) for n in OUT_FIELDS}
    w_out.update(th=w_in["th"] + np.float32(4.0), qv=w_in["qv"] - np.float32(1.5e-3), qc=np.full(nz, 1.5e-3, np.float32))
    tol_spec = {"schema": "p0_gate2_tolerances/1", "level_mask": "all",
                "fields": {n: {"atol": a, "rtol": 0.0} for n, a in
                           zip(GATED, (0.05, 2.0e-5, 2.0e-5, 0.05, 0.06, 0.06))}}
    checks = 0
    with tempfile.TemporaryDirectory() as d:
        tpath = Path(d) / "gate2_tolerances.json"
        tpath.write_text(json.dumps(tol_spec))
        tsha = sha256_file(tpath)
        # refusals: missing file, wrong sha, malformed sha, wrong schema, CLI without sha
        for args, msg in (((Path(d) / "absent.json", tsha), "missing"), ((tpath, "0" * 64), "sha256"),
                          ((tpath, tsha[:40]), "full lowercase")):
            try:
                load_tolerances(*args)
                raise AssertionError("no refusal: " + msg)
            except Refuse as e:
                assert msg in str(e), e
                checks += 1
        bad = Path(d) / "bad.json"
        bad.write_text(json.dumps({"schema": "p0_gate2_tolerances/1", "fields": {"T": {"atol": 1, "rtol": 0}}}))
        try:
            load_tolerances(bad, sha256_file(bad))
            raise AssertionError("no refusal: schema")
        except Refuse as e:
            assert "fields exactly" in str(e), e
            checks += 1
        try:
            main(["--tolerances", str(tpath), "--npz", "x", "--wrf-output", "y", "--out", str(Path(d) / "o.json")])
            raise AssertionError("no refusal: CLI without sha")
        except Refuse as e:
            assert "full lowercase" in str(e), e
            checks += 1
        tol, _ = load_tolerances(tpath, tsha)

        # fake WRF driver output file -> parser + input-echo check
        opath = Path(d) / "OUTPUT.txt"
        with open(opath, "w") as f:
            f.write("P0G2OUT 1\n")
            f.write(f"NZ {nz} {dt:.10e}\n")
            for k in range(nz):
                f.write(f"IN {k + 1} " + " ".join(f"{w_in[n][k]:.10e}" for n in DRIVER_FIELDS) + "\n")
            for k in range(nz):
                f.write(f"OUT {k + 1} " + " ".join(f"{w_out[n][k]:.10e}" for n in OUT_FIELDS) + "\n")
            f.write("SFC " + " ".join("0.0" for _ in SFC_FIELDS) + "\n")
        pnz, pdt, pin, pout, _ = parse_wrf_output(opath)
        assert pnz == nz and pdt == dt and all(np.array_equal(pin[n], w_in[n]) for n in DRIVER_FIELDS)
        assert all(np.array_equal(pout[n], w_out[n]) for n in OUT_FIELDS)
        npz = {"qv": np.zeros(nz), "dt": np.float64(dt), **{"driver_" + n: w_in[n] for n in DRIVER_FIELDS}}
        check_same_inputs(npz, nz, dt, pin)
        npz_bad = dict(npz, driver_th=w_in["th"] + np.float32(0.01))
        try:
            check_same_inputs(npz_bad, nz, dt, pin)
            raise AssertionError("no refusal: input echo")
        except Refuse as e:
            assert "input echo" in str(e), e
        checks += 2

        wrf = wrf_fields(pin, pout, dt)
        # finish form minus full product = +R*dth*dqv (product - finish = -R*dth*dqv)
        np.testing.assert_allclose(wrf["theta_m_product"] - wrf["theta_m"],
                                   -RV_OVER_RD * (np.float64(pout["th"]) - np.float64(pin["th"]))
                                   * (np.float64(pout["qv"]) - np.float64(pin["qv"])), atol=1e-9)
        near = {n: np.asarray(wrf[n]) + rng.uniform(-0.3, 0.3, nz) * tol[n][0] for n in GATED}
        cold = dict(near, T=np.asarray(wrf["T"]) + 5.0, qc=np.zeros(nz))                # wrong-convention fingerprint
        r = gate(wrf, {"fixed": near, "mutant": cold}, tol)
        assert r["verdict"] == "PASS", r
        r = gate(wrf, {"fixed": dict(near, qv=np.asarray(wrf["qv"]) + 3.0e-5), "mutant": cold}, tol)
        assert r["verdict"] == "FAIL" and not r["fixed_all_within"], r                   # fixed arm off in qv
        r = gate(wrf, {"fixed": near, "mutant": dict(cold, T=near["T"], qc=near["qc"])}, tol)
        assert r["verdict"] == "FAIL" and not r["mutant_exceeds_T_or_qc"], r             # mutant not separable
        r = gate(wrf, {"fixed": near, "mutant": dict(cold, T=near["T"])}, tol)
        assert r["verdict"] == "PASS" and r["arms"]["mutant"]["T"]["within"], r          # qc alone suffices
        nan = dict(near, theta_m=np.full(nz, np.nan))
        r = gate(wrf, {"fixed": nan, "mutant": cold}, tol)
        assert r["verdict"] == "FAIL" and not r["arms"]["fixed"]["theta_m"]["within"], r  # NaN never passes
        # warm mask: WRF T_in = th*pii; make the top two levels cold and put a fixed-arm miss only there
        pin_c = dict(pin, pii=np.float32(np.r_[w_in["pii"][:4], 0.9, 0.9]), th=np.float32(np.r_[w_in["th"][:4], 280.0, 282.0]))
        wrf_c = wrf_fields(pin_c, pout, dt)
        assert list(np.flatnonzero(level_mask(wrf_c, "warm_T_in_gt_275K"))) == [0, 1, 2, 3]
        near_c = {n: np.asarray(wrf_c[n]) for n in GATED}
        miss_aloft = dict(near_c, qv=np.asarray(wrf_c["qv"]) + np.r_[0, 0, 0, 0, 1.0e-4, 0])
        cold_c = dict(near_c, T=np.asarray(wrf_c["T"]) + 5.0)
        tol_w = dict(tol, level_mask="warm_T_in_gt_275K")
        r = gate(wrf_c, {"fixed": miss_aloft, "mutant": cold_c}, tol_w)
        assert r["verdict"] == "PASS" and not r["all_levels"]["fixed"]["qv"]["within"], r  # reported, not gated
        r = gate(wrf_c, {"fixed": miss_aloft, "mutant": cold_c}, tol)
        assert r["verdict"] == "FAIL", r                                                   # gated with "all"
        checks += 8
    print(f"P0G2_COMPARE SELF-TEST PASS checks={checks}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refuse as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
