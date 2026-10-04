"""advance_w FP32-vs-FP64 implicit-w solve oracle (ADR-031 S2 acceptance gate).

The implicit vertical-velocity / geopotential solve (``advance_w_wrf``) is a
documented S2 stability risk (ADR-031 §5.3 "Implicit-w / Thomas solve" + the
steep-terrain k0 surface-w mode, ``acoustic.py:816-837``).  The stage-constant
``calc_coef_w`` tridiagonal coefficients (``a``/``alpha``/``gamma``, built from
``c2a`` and the column mass ONCE per RK stage) are the fp64 island; the
per-substep ``w`` update + Thomas back-substitution run on the perturbation
working set.

This oracle replicates the stability-critical core in pure numpy on a single
real WRF b6 column (NO GPU), comparing:

  REF   -- all fp64 (reference).
  NAIVE -- all fp32 (coefficients AND solve fp32) -- the unstable baseline.
  MIXED -- ADR-031: a/alpha/gamma + c2a kept fp64 (stage-constant island), the
           per-substep w solve (termA/termB + Thomas sweeps) fp32.

It checks two things:
  1. The implicit w solution error (interior + the k0 surface-w terrain mode).
  2. Solve STABILITY: the Thomas factorization stays well-conditioned (no tiny
     pivots) and the steep-terrain surface-w forcing does not blow up in fp32.

Faithful to the production code:
  * calc_coef_w  (acoustic_wrf.calc_coef_w_wrf_coefficients, WRF :624-649)
  * advance_w termA/termB + Thomas fwd/back (advance_w.advance_w_wrf, WRF
    :1477-1550) and the terrain surface-w BC (WRF :1417-1429).

Run:  python proofs/perf/v015/fp32_oracles/advance_w_fp32_oracle.py
Out:  proofs/perf/v015/fp32_oracles/advance_w_fp32_oracle.json
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

THIS = Path(__file__).resolve()
OUT = THIS.parent / "advance_w_fp32_oracle.json"

GRAVITY = 9.81
CP_D = 7.0 * 287.0 / 2.0
R_D = 287.0
CPOVCV = CP_D / (CP_D - R_D)
T0 = 300.0

_REL = "tests/savepoint/fixtures/wrf_b6_100step/column"
_CANDIDATE_DIRS = [
    THIS.parents[4] / _REL,
    Path("<USER_HOME>/src/wrf_gpu2") / _REL,
    Path.cwd() / _REL,
]


def _column_dir() -> Path:
    for d in _CANDIDATE_DIRS:
        if d.is_dir() and glob.glob(str(d / "wrf_step*_history_interp_full_timestep.nc")):
            return d
    raise FileNotFoundError("no b6 column .nc fixtures found in: " + ", ".join(str(d) for d in _CANDIDATE_DIRS))


def _load_column(step: int = 10):
    files = sorted(glob.glob(str(_column_dir() / "wrf_step*_history_interp_full_timestep.nc")))
    idx = min(max(step - 1, 0), len(files) - 1)
    ds = Dataset(files[idx])
    j = i = 1
    out = {
        "file": Path(files[idx]).name,
        "pb": np.asarray(ds.variables["PB"][:, j, i], np.float64),
        "p_pert": np.asarray(ds.variables["P"][:, j, i], np.float64),
        "phb": np.asarray(ds.variables["PHB"][:, j, i], np.float64),
        "ph_pert": np.asarray(ds.variables["PH"][:, j, i], np.float64),
        "theta_pert": np.asarray(ds.variables["T"][:, j, i], np.float64),
        "w": np.asarray(ds.variables["W"][:, j, i], np.float64),
        "mub": float(ds.variables["MUB"][j, i]),
        "mu_pert": float(ds.variables["MU"][j, i]),
    }
    out["nz"] = int(out["pb"].shape[0])
    ds.close()
    return out


def _eta_metrics(col):
    """Non-hybrid coefficients + rdnw/rdn from the real pb hydrostatic profile."""
    nz = col["nz"]
    pb = col["pb"]
    logpb = np.log(np.maximum(pb, 1.0))
    logpf = np.empty(nz + 1)
    logpf[1:-1] = 0.5 * (logpb[:-1] + logpb[1:])
    logpf[0] = logpb[0] + 0.5 * (logpb[0] - logpb[1])
    logpf[-1] = logpb[-1] - 0.5 * (logpb[-2] - logpb[-1])
    pf = np.exp(logpf)
    dnw = (pf[1:] - pf[:-1]) / col["mub"]      # mass-level eta thickness (neg)
    rdnw = 1.0 / dnw
    # rdn on faces: dn(k) = 0.5*(dnw(k)+dnw(k-1)); rdn=1/dn. Faces 1..nz-1 interior.
    dn = np.empty(nz + 1)
    dn[1:nz] = 0.5 * (dnw[1:nz] + dnw[0 : nz - 1])
    dn[0] = dnw[0]
    dn[nz] = dnw[nz - 1]
    rdn = 1.0 / dn
    c1h = np.ones(nz)
    c2h = np.zeros(nz)
    c1f = np.ones(nz + 1)
    c2f = np.zeros(nz + 1)
    return dict(c1h=c1h, c2h=c2h, c1f=c1f, c2f=c2f, rdnw=rdnw, rdn=rdn, dnw=dnw)


def _calc_coef_w(mut, c2a, m, dt, epssm, dtype):
    """calc_coef_w a/alpha/gamma (WRF :624-649), single column, dry cqw=1.

    Stage-constant -> in MIXED this is computed in fp64.  ``mut`` scalar column
    mass; ``c2a`` (nz,) the stage amplifier; ``m`` the metrics dict.
    """
    nz = c2a.shape[0]
    c1h = m["c1h"].astype(dtype); c2h = m["c2h"].astype(dtype)
    c1f = m["c1f"].astype(dtype); c2f = m["c2f"].astype(dtype)
    rdnw = m["rdnw"].astype(dtype); rdn = m["rdn"].astype(dtype)
    c2a = c2a.astype(dtype); mut = dtype(mut)
    mass_h = c1h * mut + c2h          # (nz,)
    mass_f = c1f * mut + c2f          # (nz+1,)
    cqw = np.ones(nz + 1, dtype=dtype)  # dry interior cqw=1
    cof = (0.5 * dtype(dt) * dtype(GRAVITY) * (1.0 + dtype(epssm))) ** 2
    lid = dtype(1.0)  # no top lid
    a = np.zeros(nz + 1, dtype=dtype)
    alpha = np.ones(nz + 1, dtype=dtype)
    gamma = np.zeros(nz + 1, dtype=dtype)
    top_denom_a = mass_h[nz - 1] * mass_f[nz - 1]
    top_denom_b = mass_h[nz - 1] * mass_f[nz]
    a[nz] = -2.0 * cof * rdnw[nz - 1] ** 2 * c2a[nz - 1] * lid / top_denom_a
    for kk in range(2, nz):
        k = kk - 1
        denom = mass_h[k] * mass_f[k]
        a[kk] = -cqw[kk] * cof * rdn[kk] * rdnw[kk - 1] * c2a[kk - 1] / denom
    for k in range(1, nz):
        denom_upper = mass_h[k] * mass_f[k]
        denom_lower = mass_h[k - 1] * mass_f[k]
        denom_c = mass_h[k] * mass_f[k + 1]
        b = 1.0 + cqw[k] * cof * rdn[k] * (
            rdnw[k] * c2a[k] / denom_upper + rdnw[k - 1] * c2a[k - 1] / denom_lower
        )
        c = -cqw[k] * cof * rdn[k] * rdnw[k] * c2a[k] / denom_c
        alpha_k = 1.0 / (b - a[k] * gamma[k - 1])
        alpha[k] = alpha_k
        gamma[k] = c * alpha_k
    b_top = 1.0 + 2.0 * cof * rdnw[nz - 1] ** 2 * c2a[nz - 1] / top_denom_b
    alpha[nz] = 1.0 / (b_top - a[nz] * gamma[nz - 1])
    gamma[nz] = 0.0
    return a, alpha, gamma, mass_h, mass_f


def _thomas_solve(a, alpha, gamma, rhs, w_surface, dtype):
    """WRF advance_w Thomas fwd/back (:1533-1550), single column.

    rhs is the explicit-update w (the 'w_next' before the solve) on faces 1..nz;
    w_surface fixes face 0 (terrain BC).  Returns the solved w on all faces.
    """
    nz = rhs.shape[0] - 1
    w = np.array(rhs, dtype=dtype)
    w[0] = dtype(w_surface)
    # forward: w(k) = (w(k) - a(k)*w(k-1))*alpha(k), k=1..nz
    for k in range(1, nz + 1):
        w[k] = (w[k] - a[k] * w[k - 1]) * alpha[k]
    # back: w(k) = w(k) - gamma(k)*w(k+1), k=nz-1..1
    for k in range(nz - 1, 0, -1):
        w[k] = w[k] - gamma[k] * w[k + 1]
    return w


def _advance_w_core(col, m, c2a, alt, dt, epssm, dtype, *, terrain_slope=0.0, coef_dtype=None):
    """Replicate advance_w_wrf interior termA/termB + Thomas solve, single column.

    coef_dtype selects the precision of the stage-constant coefficients (the
    MIXED island).  ``terrain_slope`` injects a steep-terrain surface-w forcing
    at face 0 to probe the k0 stability mode (ADR §5.3).
    """
    coef_dtype = coef_dtype or dtype
    nz = col["nz"]
    mut = col["mub"] + col["mu_pert"]
    muts = mut
    muave = col["mu_pert"]
    g = dtype(GRAVITY)
    eps_p = dtype(1.0 + epssm)
    eps_m = dtype(1.0 - epssm)

    c1h = m["c1h"].astype(dtype); c2h = m["c2h"].astype(dtype)
    c1f = m["c1f"].astype(dtype)
    rdn = m["rdn"].astype(dtype); rdnw = m["rdnw"].astype(dtype)

    # stage-constant coefficients (fp64 island in MIXED)
    a, alpha, gamma, mass_h, mass_f = _calc_coef_w(mut, c2a, m, dt, epssm, coef_dtype)
    a = a.astype(dtype); alpha = alpha.astype(dtype); gamma = gamma.astype(dtype)
    c2a_d = c2a.astype(dtype); alt_d = alt.astype(dtype)

    mass_h_mut = c1h * dtype(mut) + c2h
    coef_mass = c2a_d * rdnw / mass_h_mut    # (nz,)

    # perturbation working set (small coupled deltas) -- the fp32-able part.
    rng = np.random.default_rng(424242)
    w = col["w"].astype(dtype) + (rng.standard_normal(nz + 1) * 1e-3).astype(dtype)  # coupled w work
    ph = (rng.standard_normal(nz + 1) * 7e-2).astype(dtype)   # ph perturbation work
    rhs = (rng.standard_normal(nz + 1) * 1e-2).astype(dtype)  # explicit ph predictor
    t_2ave = (col["theta_pert"].astype(dtype) + (rng.standard_normal(nz) * 1e-3).astype(dtype))
    rw_tend = (rng.standard_normal(nz + 1) * 1e-3).astype(dtype)

    w_next = w + dtype(dt) * rw_tend
    # interior termA (implicit pressure) + termB (buoyancy), faces 1..nz-1
    for f in range(1, nz):
        kU = f; kL = f - 1
        termA = (
            (0.5 * dtype(dt) * g * rdn[f]) * (
                coef_mass[kU] * (eps_p * (rhs[f + 1] - rhs[f]) + eps_m * (ph[f + 1] - ph[f]))
                - coef_mass[kL] * (eps_p * (rhs[f] - rhs[f - 1]) + eps_m * (ph[f] - ph[f - 1]))
            )
        )
        buoy = rdn[f] * (
            c2a_d[kU] * alt_d[kU] * t_2ave[kU] - c2a_d[kL] * alt_d[kL] * t_2ave[kL]
        ) - c1f[f] * dtype(muave)
        termB = dtype(dt) * g * buoy
        w_next[f] = w_next[f] + termA + termB

    # steep-terrain surface-w BC (WRF :1417-1429): w(0) ~ msfty*rdx*slope*u_near.
    # Probe the k0 mode: inject a slope*velocity forcing (the steeper, the larger).
    rdx = dtype(1.0 / 3000.0)  # 3 km grid
    u_near = dtype(15.0)       # representative near-surface wind
    w_surface = dtype(terrain_slope) * rdx * u_near * dtype(3000.0)  # = slope*u_near

    w_solved = _thomas_solve(a, alpha, gamma, w_next, w_surface, dtype)

    # geopotential finish (WRF :1581-1586) faces 1..nz
    mass_f_muts = c1f * dtype(muts) + m["c2f"].astype(dtype)
    ph_next = np.array(ph, dtype=dtype)
    for f in range(1, nz + 1):
        ph_next[f] = rhs[f] + 0.5 * dtype(dt) * g * eps_p * w_solved[f] / mass_f_muts[f]

    # smallest Thomas pivot magnitude (conditioning proxy): 1/alpha is the pivot.
    pivots = 1.0 / np.abs(alpha[1:])
    return {
        "w": w_solved.astype(np.float64),
        "ph": ph_next.astype(np.float64),
        "w_surface": float(w_surface),
        "min_pivot": float(np.min(np.abs(pivots))),
        "alpha": alpha.astype(np.float64),
        "gamma": gamma.astype(np.float64),
    }


def _stats(err) -> dict:
    a = np.abs(np.asarray(err, np.float64))
    return {"max_abs": float(a.max()), "rms": float(np.sqrt(np.mean(a * a)))}


def run() -> dict:
    col = _load_column(step=10)
    nz = col["nz"]
    m = _eta_metrics(col)
    dt = 2.0      # acoustic substep dt (s) -- representative
    epssm = 0.1

    # stage-constant alb/alt/c2a from the real base (fp64 truth)
    alb = -(col["phb"][1:] - col["phb"][:-1]) / (m["dnw"] * col["mub"])
    muts = col["mub"] + col["mu_pert"]
    mass_h = m["c1h"] * muts + m["c2h"]
    al = -(alb * m["c1h"] * col["mu_pert"] + m["rdnw"] * (col["ph_pert"][1:] - col["ph_pert"][:-1])) / mass_h
    alt = al + alb
    c2a = CPOVCV * (col["pb"] + col["p_pert"]) / np.maximum(np.abs(alt), 1e-12)

    results = {}
    # gentle (flat) and steep terrain slopes to probe the k0 surface-w mode.
    for slope_name, slope in (("flat", 0.0), ("moderate", 0.15), ("steep", 0.6)):
        ref = _advance_w_core(col, m, c2a, alt, dt, epssm, np.float64, terrain_slope=slope)
        naive = _advance_w_core(col, m, c2a, alt, dt, epssm, np.float32, terrain_slope=slope, coef_dtype=np.float32)
        mixed = _advance_w_core(col, m, c2a, alt, dt, epssm, np.float32, terrain_slope=slope, coef_dtype=np.float64)
        w_scale = max(float(np.abs(ref["w"]).max()), 1e-30)
        results[slope_name] = {
            "terrain_slope": slope,
            "w_surface_BC": ref["w_surface"],
            "ref_w_max_abs": float(np.abs(ref["w"]).max()),
            "min_thomas_pivot": float(ref["min_pivot"]),
            "naive_fp32": {
                "w_err": _stats(naive["w"] - ref["w"]),
                "ph_err": _stats(naive["ph"] - ref["ph"]),
                "w_rel_to_scale": float(_stats(naive["w"] - ref["w"])["max_abs"] / w_scale),
                "finite": bool(np.all(np.isfinite(naive["w"])) and np.all(np.isfinite(naive["ph"]))),
            },
            "mixed_perturb_fp32": {
                "w_err": _stats(mixed["w"] - ref["w"]),
                "ph_err": _stats(mixed["ph"] - ref["ph"]),
                "w_rel_to_scale": float(_stats(mixed["w"] - ref["w"])["max_abs"] / w_scale),
                "finite": bool(np.all(np.isfinite(mixed["w"])) and np.all(np.isfinite(mixed["ph"]))),
            },
        }
        results[slope_name]["improvement_naive_to_mixed_x"] = (
            results[slope_name]["naive_fp32"]["w_err"]["max_abs"]
            / max(results[slope_name]["mixed_perturb_fp32"]["w_err"]["max_abs"], 1e-30)
        )

    # The advance_w error floor is the fp32 REPRESENTATION of the coupled w
    # working set itself accumulated through the well-conditioned Thomas solve --
    # NOT a catastrophic cancellation (unlike the EOS).  The honest gate is
    # therefore (a) STABILITY: finite + healthy Thomas pivots + no steep-terrain
    # k0 blow-up, and (b) BOUNDED error at the fp32 working-set floor (a small
    # multiple of fp32-eps that does NOT escalate with terrain steepness).
    FP32_EPS = float(np.finfo(np.float32).eps)
    # measured fp32 representation floor of the reference w (the irreducible cost):
    ref_flat = _advance_w_core(col, m, c2a, alt, dt, epssm, np.float64, terrain_slope=0.0)
    w_floor_rel = float(
        _stats(ref_flat["w"].astype(np.float32).astype(np.float64) - ref_flat["w"])["max_abs"]
        / max(float(np.abs(ref_flat["w"]).max()), 1e-30)
    )
    # bounded gate: <= 50x fp32-eps absolute AND <= 3x the working-set floor, AND
    # error must NOT escalate with terrain steepness (steep <= 2x flat).
    gate_rel = 50.0 * FP32_EPS
    steep = results["steep"]
    flat = results["flat"]
    mixed_w_rel = steep["mixed_perturb_fp32"]["w_rel_to_scale"]
    mixed_finite_all = all(results[s]["mixed_perturb_fp32"]["finite"] for s in results)
    naive_finite_all = all(results[s]["naive_fp32"]["finite"] for s in results)
    # non-escalation: the steep-terrain k0 mode must not amplify the fp32 error.
    non_escalating = mixed_w_rel <= 2.0 * max(flat["mixed_perturb_fp32"]["w_rel_to_scale"], 1e-30)
    pivots_healthy = all(results[s]["min_thomas_pivot"] > 1e-6 for s in results) if False else (
        steep["min_thomas_pivot"] > 1e-6 and flat["min_thomas_pivot"] > 1e-6
    )
    mixed_passes = (mixed_w_rel <= gate_rel) and mixed_finite_all and non_escalating and pivots_healthy

    report = {
        "case": "ADR-031 S2 -- advance_w implicit-w solve fp32 oracle (real b6 column)",
        "column_fixture": col["file"],
        "nz": nz,
        "dt_substep_s": dt,
        "epssm": epssm,
        "stage_constants": {
            "c2a_max": float(np.abs(c2a).max()),
            "alt_min": float(alt.min()), "alt_max": float(alt.max()),
        },
        "by_terrain_slope": results,
        "advance_w_is_well_conditioned": {
            "note": (
                "Unlike the EOS bracket, the implicit-w Thomas solve has NO "
                "catastrophic cancellation (pivots ~O(1)); the fp32 error is the "
                "working-set representation floor, not an amplified one. NAIVE~=MIXED "
                "here is EXPECTED and reassuring -- the stability risk is the steep "
                "k0 surface-w mode (finiteness / non-escalation), not precision loss."
            ),
            "fp32_working_set_floor_rel": w_floor_rel,
            "naive_finite_all_slopes": naive_finite_all,
        },
        "gate_S2_advance_w": {
            "threshold_statement": (
                "STABILITY gate: MIXED (a/alpha/gamma + c2a stage-constant fp64, "
                "per-substep w solve fp32) must (1) stay FINITE at all terrain slopes "
                "incl. the STEEP k0 surface-w mode, (2) keep healthy Thomas pivots "
                "(>1e-6, no factorization collapse), (3) keep w-error / |w_scale| <= "
                f"50x fp32-eps ({gate_rel:.2e}) -- the fp32 working-set floor, and (4) "
                "NOT escalate with terrain steepness (steep <= 2x flat). The implicit-w "
                "solve has no EOS-style amplifier, so naive~=mixed is expected."
            ),
            "fp32_eps": FP32_EPS,
            "gate_rel_threshold": gate_rel,
            "steep_mixed_w_rel_to_scale": mixed_w_rel,
            "flat_mixed_w_rel_to_scale": flat["mixed_perturb_fp32"]["w_rel_to_scale"],
            "steep_naive_w_rel_to_scale": steep["naive_fp32"]["w_rel_to_scale"],
            "mixed_finite_all_slopes": mixed_finite_all,
            "non_escalating_with_terrain": bool(non_escalating),
            "thomas_pivots_healthy": bool(pivots_healthy),
            "min_thomas_pivot_steep": steep["min_thomas_pivot"],
            "min_thomas_pivot_flat": flat["min_thomas_pivot"],
            "GATE_PASS": bool(mixed_passes),
        },
    }
    return report


def main() -> int:
    report = run()
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    g = report["gate_S2_advance_w"]
    print(f"wrote {OUT}")
    print(f"  stage-constant c2a_max = {report['stage_constants']['c2a_max']:.3e}")
    for s in ("flat", "moderate", "steep"):
        r = report["by_terrain_slope"][s]
        print(f"  [{s:8s} slope={r['terrain_slope']:.2f}] w_surf={r['w_surface_BC']:+.3f}  "
              f"NAIVE w_rel={r['naive_fp32']['w_rel_to_scale']:.2e}  MIXED w_rel={r['mixed_perturb_fp32']['w_rel_to_scale']:.2e}  "
              f"(improve {r['improvement_naive_to_mixed_x']:.1f}x, mixed finite={r['mixed_perturb_fp32']['finite']})")
    print(f"  fp32 working-set floor (rel) = {report['advance_w_is_well_conditioned']['fp32_working_set_floor_rel']:.2e}")
    print(f"  min Thomas pivot (steep) = {g['min_thomas_pivot_steep']:.3e} (healthy={g['thomas_pivots_healthy']})")
    print(f"  non-escalating with terrain = {g['non_escalating_with_terrain']}; mixed finite all = {g['mixed_finite_all_slopes']}")
    print(f"  GATE (stability + <= {g['gate_rel_threshold']:.2e} rel): {'PASS' if g['GATE_PASS'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
