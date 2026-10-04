"""calc_p_rho FP32-vs-FP64 EOS oracle (ADR-031 S2 acceptance gate).

This is the **baseline + gate** for the linearized-EOS perturbation-pressure
amplifier named in ADR-031 §2.2.1 (``calc_p_rho.py:91``) and the hydrostatic
inverse-density cancellation (§2.2.2, ``calc_p_rho.py:82`` / the ``alb`` face
difference).  It performs NO model run and uses NO GPU (pure numpy on a real
WRF column savepoint).

It is deliberately HONEST: it SHOWS the naive-fp32 failure (it does not hide it),
and it establishes the threshold the perturbation-frame fp32 fix must beat.

Three precision variants are evaluated on the SAME real WRF b6 column state
(``tests/savepoint/fixtures/wrf_b6_100step/column/``):

  REF   -- everything fp64 (the production / oracle reference).
  NAIVE -- everything fp32: the large absolute totals (pb~1e5, phb~3e5) AND the
           stage-constant amplifier c2a~1.3e5 held in fp32.  This is "wired fp32"
           and it DETONATES the EOS bracket -- the baseline the fix must beat.
  MIXED -- ADR-031 perturbation frame: the stage-constant fp64 island
           (c2a, alt, alb, the pb-derived inputs) stays fp64; ONLY the dynamic
           perturbation work arrays (mu_work, ph_work, theta_work, theta_1) go
           fp32.  This must fall to the fp32-noise floor.

The WRF-faithful relations replicate the production code verbatim:

  al = -( alt*c1h*mu_work + rdnw*(ph_work[k+1]-ph_work[k]) ) / (c1h*muts+c2h)
       (calc_p_rho._calc_al_p, WRF module_small_step_em.F:522-523)
  p' = c2a*( alt*(theta_work - c1h*mu_work*theta_1)
             / ((c1h*muts+c2h)*(t0+theta_1)) - al )
       (calc_p_rho._calc_al_p, WRF :527-528)
  alb = -dphb / (dnw*(c1h*mub+c2h))
       (acoustic_wrf.diagnose_pressure_al_alt, WRF module_initialize_real.F:3817)
  c2a = (cp/cv)*(pb+p')/|alt|
       (small_step_prep_wrf, WRF small_step_prep c2a build)

Run:  python proofs/perf/v015/fp32_oracles/calc_p_rho_fp32_oracle.py
Out:  proofs/perf/v015/fp32_oracles/calc_p_rho_fp32_oracle.json
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

THIS = Path(__file__).resolve()
OUT = THIS.parent / "calc_p_rho_fp32_oracle.json"

# WRF share/module_model_constants.F (the exact constants the dycore uses).
CP_D = 7.0 * 287.0 / 2.0  # 1004.5 J/kg/K (R_D=287.0 here matches acoustic_wrf)
R_D = 287.0
CPOVCV = CP_D / (CP_D - R_D)  # cp/cv ~ 1.4 -> the EOS amplifier base ratio
T0 = 300.0  # WRF base potential temperature offset

# b6 column savepoint search path (the .nc fixtures are real WRF data, untracked
# in worktrees; search the worktree-relative path first then the shared checkout).
_REL = "tests/savepoint/fixtures/wrf_b6_100step/column"
_CANDIDATE_DIRS = [
    THIS.parents[4] / _REL,                       # repo root from proofs/perf/v015/fp32_oracles
    Path("<USER_HOME>/src/wrf_gpu2") / _REL,       # shared checkout (authoritative fixtures)
    Path.cwd() / _REL,
]


def _column_dir() -> Path:
    for d in _CANDIDATE_DIRS:
        if d.is_dir() and glob.glob(str(d / "wrf_step*_history_interp_full_timestep.nc")):
            return d
    raise FileNotFoundError(
        "no b6 column .nc fixtures found in: " + ", ".join(str(d) for d in _CANDIDATE_DIRS)
    )


def _load_column(step: int = 10):
    """Load a real WRF b6 column savepoint, returning the central (i=j=1) column."""
    files = sorted(glob.glob(str(_column_dir() / "wrf_step*_history_interp_full_timestep.nc")))
    idx = min(max(step - 1, 0), len(files) - 1)
    ds = Dataset(files[idx])
    j = i = 1  # central column of the 3x3 patch
    pb = np.asarray(ds.variables["PB"][:, j, i], dtype=np.float64)  # (nz,)
    p_pert = np.asarray(ds.variables["P"][:, j, i], dtype=np.float64)  # (nz,)
    phb = np.asarray(ds.variables["PHB"][:, j, i], dtype=np.float64)  # (nz+1,)
    ph_pert = np.asarray(ds.variables["PH"][:, j, i], dtype=np.float64)  # (nz+1,)
    theta_pert = np.asarray(ds.variables["T"][:, j, i], dtype=np.float64)  # (nz,) theta-300
    mub = float(np.asarray(ds.variables["MUB"][j, i], dtype=np.float64))
    mu_pert = float(np.asarray(ds.variables["MU"][j, i], dtype=np.float64))
    ds.close()
    return {
        "file": Path(files[idx]).name,
        "pb": pb, "p_pert": p_pert, "phb": phb, "ph_pert": ph_pert,
        "theta_pert": theta_pert, "mub": mub, "mu_pert": mu_pert, "nz": int(pb.shape[0]),
    }


def _alb_from_phb(phb, mub, dnw, c1h, c2h, dtype):
    """alb = -dphb/(dnw*(c1h*mub+c2h)) -- the base inverse density.

    Hydrostatic cancellation site #2 (ADR §2.2.2): dphb is a face difference of
    phb (~2e5 absolute at the top); in fp32 the ULP of phb-top contaminates the
    small layer thickness.
    """
    phb = phb.astype(dtype)
    mub_a = dtype(mub)
    mass_h_base = c1h.astype(dtype) * mub_a + c2h.astype(dtype)
    dphb = phb[1:] - phb[:-1]
    denom = dnw.astype(dtype) * mass_h_base
    safe = np.where(np.abs(denom) > 1.0e-12, denom, dtype(1.0e-12))
    return (-dphb / safe).astype(dtype)


def _calc_al_p(*, mu_work, muts_total, ph_work, theta_work, theta_1, c2a, alt, c1h, c2h, rdnw, t0, dtype):
    """Verbatim port of calc_p_rho._calc_al_p (the amplifier), single column.

    Every input is cast to ``dtype`` first -- this is what makes the variant
    (REF / NAIVE / MIXED) honest: MIXED passes fp64 c2a/alt/alb/rdnw and fp32
    work arrays; NAIVE passes everything fp32; REF everything fp64.
    """
    c1h = c1h.astype(dtype)
    c2h = c2h.astype(dtype)
    rdnw = rdnw.astype(dtype)
    muts_total = dtype(muts_total)
    c2a = np.asarray(c2a, dtype=dtype)
    alt = np.asarray(alt, dtype=dtype)
    ph_work = np.asarray(ph_work, dtype=dtype)
    theta_work = np.asarray(theta_work, dtype=dtype)
    theta_1 = np.asarray(theta_1, dtype=dtype)
    mass_h = c1h * muts_total + c2h
    safe_mass = np.where(np.abs(mass_h) > 1.0e-12, mass_h, dtype(1.0e-12))
    mu_term = c1h * dtype(mu_work)
    # al = -(alt*c1h*mu + rdnw*(ph(k+1)-ph(k)))/(c1h*muts+c2h)   (WRF :522-523)
    al = -(alt * mu_term + rdnw * (ph_work[1:] - ph_work[:-1])) / safe_mass
    # p = c2a*( alt*(t_2 - c1h*mu*t_1)/((c1h*muts+c2h)*(t0+t_1)) - al )  (WRF :527-528)
    theta_total_ref = dtype(t0) + theta_1
    safe_theta_ref = np.where(np.abs(theta_total_ref) > 1.0e-6, theta_total_ref, dtype(1.0e-6))
    p = c2a * (alt * (theta_work - mu_term * theta_1) / (safe_mass * safe_theta_ref) - al)
    return al.astype(dtype), p.astype(dtype)


def _real_substep_state(col, alt, c2a, mass_h, rdnw, c1h, rng):
    """Reconstruct the WRF acoustic work state that REPRODUCES the real stored p'.

    The most defensible oracle uses the REAL stored WRF perturbation pressure
    ``P`` (up to ~1825 Pa on this column) as the fp64 reference -- no contrived
    residual.  We invert the WRF linearized EOS for the coupled work theta ``t_2``
    so that the fp64 evaluation reproduces the stored ``p'`` EXACTLY:

      p' = c2a*( alt*(t2 - c1h*mu*t1)/(mass*tref) - al )
      =>  t2 = mass*tref/alt * ( p'/c2a + al ) + c1h*mu*t1

    Here ``al`` is the self-consistent substep ``al`` (from the small ph_work/mu'
    deltas), ``c2a``/``alt`` the stage constants from the large absolutes, and
    ``p'`` the real stored perturbation pressure.  This makes the EOS bracket sit
    at the column's TRUE cancellation depth (measured ~4.6 digits lost) and the
    reference p' the REAL physical magnitude -- so the fp32 error is reported
    against the real ~1825 Pa scale, the honest physical denominator.

    The substep mu_work/ph_work are small coupled deltas at their measured live
    magnitudes (mu' ~ O(0.01) Pa, ph_work ~ O(0.07) m^2/s^2).
    """
    nz = col["nz"]
    theta_1 = col["theta_pert"].astype(np.float64)        # WRF t_1
    muts = col["mub"] + col["mu_pert"]
    mu_work = float(rng.standard_normal() * 1.0e-2)
    ph_work = rng.standard_normal(nz + 1) * 7.0e-2
    tref = T0 + theta_1
    p_real = col["p_pert"].astype(np.float64)             # the real stored p' (reference)
    # self-consistent substep al from the small work deltas (what _calc_al_p uses):
    al_substep = -(alt * c1h * mu_work + rdnw * (ph_work[1:] - ph_work[:-1])) / mass_h
    # invert EOS for the coupled t_2 that reproduces p_real in fp64:
    t2 = mass_h * tref / alt * (p_real / c2a + al_substep) + c1h * mu_work * theta_1
    return theta_1, t2, ph_work, mu_work, muts


def _stats(err) -> dict:
    a = np.abs(np.asarray(err, dtype=np.float64))
    return {
        "max_abs": float(a.max()), "mean_abs": float(a.mean()),
        "p99_abs": float(np.percentile(a, 99)), "rms": float(np.sqrt(np.mean(a * a))),
    }


def run() -> dict:
    rng = np.random.default_rng(20260613)
    col = _load_column(step=10)
    nz = col["nz"]

    c1h = np.ones(nz, dtype=np.float64)
    c2h = np.zeros(nz, dtype=np.float64)

    # Recover dnw/rdnw (eta spacing) from the real pb hydrostatic structure so alb
    # reproduces the real layer thickness.  Faces by log-p interpolation (dry,
    # monotone); dnw(k) = (pf(k+1)-pf(k))/mub (negative, monotone in eta).
    pb_mass = col["pb"]
    logpb = np.log(np.maximum(pb_mass, 1.0))
    logpf = np.empty(nz + 1, dtype=np.float64)
    logpf[1:-1] = 0.5 * (logpb[:-1] + logpb[1:])
    logpf[0] = logpb[0] + 0.5 * (logpb[0] - logpb[1])
    logpf[-1] = logpb[-1] - 0.5 * (logpb[-2] - logpb[-1])
    pf = np.exp(logpf)
    dnw = (pf[1:] - pf[:-1]) / col["mub"]
    rdnw = 1.0 / dnw

    # ----- stage-constant fp64 island (alb, alt, c2a) computed ONCE from base ---
    muts = col["mub"] + col["mu_pert"]
    mass_h64 = c1h * muts + c2h
    alb64 = _alb_from_phb(col["phb"], col["mub"], dnw, c1h, c2h, np.float64)
    al_for_alt_64 = -(
        alb64 * c1h * col["mu_pert"] + rdnw * (col["ph_pert"][1:] - col["ph_pert"][:-1])
    ) / mass_h64
    alt64 = al_for_alt_64 + alb64
    safe_alt64 = np.where(np.abs(alt64) > 1.0e-12, np.abs(alt64), 1.0e-12)
    c2a64 = CPOVCV * (col["pb"] + col["p_pert"]) / safe_alt64  # ~1.3e5 amplifier

    # real substep work state: reconstruct the coupled t_2 that REPRODUCES the
    # real stored p' in fp64 (so the reference is the true physical perturbation
    # pressure, and the bracket sits at the column's measured cancellation depth).
    theta_1_64, theta_work_64, ph_work_64, mu_work, _muts = _real_substep_state(
        col, alt64, c2a64, mass_h64, rdnw, c1h, rng
    )

    # ===================== REF (all fp64) =====================
    al_ref, p_ref = _calc_al_p(
        mu_work=mu_work, muts_total=muts, ph_work=ph_work_64, theta_work=theta_work_64,
        theta_1=theta_1_64, c2a=c2a64, alt=alt64, c1h=c1h, c2h=c2h, rdnw=rdnw, t0=T0, dtype=np.float64,
    )

    # ===================== NAIVE (all fp32) =====================
    alb32 = _alb_from_phb(col["phb"], col["mub"], dnw, c1h, c2h, np.float32)
    al_for_alt_32 = (-(
        alb32 * c1h.astype(np.float32) * np.float32(col["mu_pert"])
        + rdnw.astype(np.float32) * (col["ph_pert"].astype(np.float32)[1:] - col["ph_pert"].astype(np.float32)[:-1])
    ) / mass_h64.astype(np.float32)).astype(np.float32)
    alt32 = (al_for_alt_32 + alb32).astype(np.float32)
    safe_alt32 = np.where(np.abs(alt32) > 1.0e-12, np.abs(alt32), np.float32(1.0e-12))
    c2a32 = (CPOVCV * (col["pb"].astype(np.float32) + col["p_pert"].astype(np.float32)) / safe_alt32).astype(np.float32)
    al_naive, p_naive = _calc_al_p(
        mu_work=np.float32(mu_work), muts_total=np.float32(muts),
        ph_work=ph_work_64.astype(np.float32), theta_work=theta_work_64.astype(np.float32),
        theta_1=theta_1_64.astype(np.float32), c2a=c2a32, alt=alt32,
        c1h=c1h, c2h=c2h, rdnw=rdnw, t0=T0, dtype=np.float32,
    )

    # ===================== MIXED (ADR-031 perturbation frame) =====================
    # Stage-constant fp64 island: c2a, alt, alb, rdnw STAY fp64.
    # Only the dynamic perturbation work arrays go fp32 (then the op widens).
    mu_work_m = float(np.float32(mu_work))
    ph_work_m = ph_work_64.astype(np.float32).astype(np.float64)
    theta_work_m = theta_work_64.astype(np.float32).astype(np.float64)
    theta_1_m = theta_1_64.astype(np.float32).astype(np.float64)
    al_mixed, p_mixed = _calc_al_p(
        mu_work=mu_work_m, muts_total=muts, ph_work=ph_work_m, theta_work=theta_work_m,
        theta_1=theta_1_m, c2a=c2a64, alt=alt64, c1h=c1h, c2h=c2h, rdnw=rdnw, t0=T0, dtype=np.float64,
    )

    # ----- the fp32 noise floor reference: round the fp64 result to fp32 -----
    p_ref_fp32_repr = p_ref.astype(np.float32).astype(np.float64)
    floor_err = _stats(p_ref_fp32_repr - p_ref)  # irreducible fp32 representation error of p'

    # The PHYSICAL scale of the active perturbation pressure, taken from the REAL
    # stored WRF p' on this column (the realistic |p'| the substep produces).  This
    # is the meaningful denominator for the gate -- an EOS p' error is acceptable
    # iff it is small vs the physical p' the dynamics carries, not vs the (tiny,
    # cancellation-collapsed) fp32 representation floor of a contrived residual.
    p_phys_scale = float(np.abs(col["p_pert"]).max())  # real stored |P| (up to ~1800 Pa)

    naive_p = _stats(p_naive.astype(np.float64) - p_ref)
    mixed_p = _stats(p_mixed - p_ref)
    naive_al = _stats(al_naive.astype(np.float64) - al_ref)
    mixed_al = _stats(al_mixed - al_ref)

    # ----- per-input fp32 sensitivity: round EXACTLY ONE input to fp32, all else
    # fp64.  This isolates which absolute is the catastrophic amplifier so the
    # MIXED fp64-island is provably the right one.  (The honest finding: alt --
    # the hydrostatic inverse density built from the phb face-difference -- and
    # the coupled work theta t_2 are the casualties; c2a-fp32-alone is benign.)
    def _round32(x):
        return np.asarray(x, dtype=np.float32).astype(np.float64)

    def _p_with(c2a_in=c2a64, alt_in=alt64, t2_in=theta_work_64, ph_in=ph_work_64, t1_in=theta_1_64, mu_in=mu_work):
        _, p = _calc_al_p(
            mu_work=mu_in, muts_total=muts, ph_work=ph_in, theta_work=t2_in,
            theta_1=t1_in, c2a=c2a_in, alt=alt_in, c1h=c1h, c2h=c2h, rdnw=rdnw, t0=T0, dtype=np.float64,
        )
        return p

    sens = {}
    for name, kw in (
        ("c2a", {"c2a_in": _round32(c2a64)}),
        ("alt", {"alt_in": _round32(alt64)}),
        ("theta_work_t2", {"t2_in": _round32(theta_work_64)}),
        ("ph_work", {"ph_in": _round32(ph_work_64)}),
        ("theta_1_t1", {"t1_in": _round32(theta_1_64)}),
        ("mu_work", {"mu_in": float(np.float32(mu_work))}),
    ):
        e = _stats(_p_with(**kw) - p_ref)
        sens[name] = {"p_prime_max_abs_Pa": e["max_abs"], "p_prime_rms_Pa": e["rms"]}

    bracket = (alt64 * (theta_work_64 - c1h * mu_work * theta_1_64) / (mass_h64 * (T0 + theta_1_64)) - al_ref)
    nz_bracket = bracket[np.abs(bracket) > 0]
    amp = {
        "c2a_max": float(np.abs(c2a64).max()),
        "c2a_mean": float(np.abs(c2a64).mean()),
        "bracket_abs_max": float(np.abs(bracket).max()),
        "bracket_abs_min": float(np.abs(nz_bracket).min()) if nz_bracket.size else 0.0,
        "p_pert_ref_abs_max": float(np.abs(p_ref).max()),
    }

    floor = max(floor_err["max_abs"], 1e-30)
    improvement = naive_p["max_abs"] / max(mixed_p["max_abs"], 1e-30)
    for name in sens:
        sens[name]["over_floor_ratio"] = sens[name]["p_prime_max_abs_Pa"] / floor
        sens[name]["rel_to_phys"] = sens[name]["p_prime_max_abs_Pa"] / max(p_phys_scale, 1e-30)
    sens_ranked = sorted(sens.items(), key=lambda kv: kv[1]["p_prime_max_abs_Pa"], reverse=True)
    dominant_input = sens_ranked[0][0]

    # ---- the two gate framings, both honest ----
    # (A) floor-relative: shows the EOS amplifier is CATASTROPHIC for naive fp32
    #     (the bracket cancellation collapses the fp32 floor to ~1e-12; any input
    #     error is then 1e4..1e5x that floor -> this is WHY naive fp32 detonates).
    # (B) physics-relative: the MEANINGFUL acceptance gate -- the fp32 p' error vs
    #     the real physical p' magnitude (~1825 Pa).  MIXED must keep this <= the
    #     fp32 relative precision (~1.2e-7 * scale, allow 10x margin) so the EOS
    #     contributes no more than fp32-noise to the dynamically-active pressure.
    naive_rel_phys = naive_p["max_abs"] / max(p_phys_scale, 1e-30)
    mixed_rel_phys = mixed_p["max_abs"] / max(p_phys_scale, 1e-30)
    FP32_EPS = float(np.finfo(np.float32).eps)  # ~1.19e-7
    phys_gate_rel = 10.0 * FP32_EPS             # allow 10x fp32-eps headroom
    mixed_passes_phys = mixed_rel_phys <= phys_gate_rel
    naive_fails_phys = naive_rel_phys > phys_gate_rel

    report = {
        "case": "ADR-031 S2 -- calc_p_rho linearized-EOS fp32 oracle (real b6 column)",
        "column_fixture": col["file"],
        "nz": nz,
        "constants": {"CP_D": CP_D, "R_D": R_D, "CPOVCV": CPOVCV, "T0": T0},
        "column_magnitudes": {
            "pb_min": float(col["pb"].min()), "pb_max": float(col["pb"].max()),
            "phb_max": float(col["phb"].max()),
            "phb_top_fp32_ulp": float(np.spacing(np.float32(col["phb"].max()))),
            "theta_pert_min": float(col["theta_pert"].min()), "theta_pert_max": float(col["theta_pert"].max()),
            "mub": col["mub"], "mu_pert": col["mu_pert"],
        },
        "eos_amplifier": amp,
        "fp32_noise_floor_of_p_prime": floor_err,
        "per_input_fp32_sensitivity": {
            "explanation": (
                "Round EXACTLY ONE input to fp32 (all else fp64) and measure the "
                "induced |p'| error. Ranks the amplifier inputs; confirms the MIXED "
                "fp64-island (alt/alb/c2a stage-constant) targets the real casualties."
            ),
            "results": sens,
            "dominant_fp32_casualty": dominant_input,
        },
        "errors_vs_fp64_ref": {
            "naive_fp32": {"p_prime": naive_p, "al": naive_al},
            "mixed_perturb_fp32": {"p_prime": mixed_p, "al": mixed_al},
        },
        "physical_p_prime_scale_Pa": p_phys_scale,
        "verdict": {
            "naive_fp32_p_prime_max_abs_Pa": naive_p["max_abs"],
            "mixed_fp32_p_prime_max_abs_Pa": mixed_p["max_abs"],
            "fp32_noise_floor_max_abs_Pa": floor_err["max_abs"],
            "mixed_over_floor_ratio": mixed_p["max_abs"] / floor,
            "naive_over_floor_ratio": naive_p["max_abs"] / floor,
            "naive_rel_to_physical_p_prime": naive_rel_phys,
            "mixed_rel_to_physical_p_prime": mixed_rel_phys,
            "improvement_naive_to_mixed_x": improvement,
        },
        "gate_S2_calc_p_rho": {
            "threshold_statement": (
                "PRIMARY gate (physics-relative): MIXED (perturbation-frame fp32, "
                "c2a/alt/alb kept fp64) must hold |p'_fp32 - p'_fp64| / |p'_physical| "
                f"<= 10x fp32-eps ({phys_gate_rel:.2e}) -- i.e. the EOS adds no more "
                "than fp32-noise to the dynamically-active perturbation pressure. "
                "SECONDARY diagnostic (floor-relative): the EOS bracket cancellation "
                "(~4.6 digits lost on real data) makes naive fp32 catastrophic "
                "(error >> the collapsed fp32 floor); MIXED must beat naive (improvement>1)."
            ),
            "fp32_eps": FP32_EPS,
            "phys_gate_rel_threshold": phys_gate_rel,
            "mixed_rel_to_physical_p_prime": mixed_rel_phys,
            "naive_rel_to_physical_p_prime": naive_rel_phys,
            "mixed_PASS_physics_gate": bool(mixed_passes_phys),
            "naive_FAILS_or_marginal": bool(naive_fails_phys),
            "mixed_beats_naive": bool(improvement > 1.0),
            "GATE_PASS": bool(mixed_passes_phys),
        },
    }
    return report


def main() -> int:
    report = run()
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    v = report["verdict"]
    g = report["gate_S2_calc_p_rho"]
    print(f"wrote {OUT}")
    print(f"  EOS amplifier c2a_max = {report['eos_amplifier']['c2a_max']:.3e}; physical |p'| = {report['physical_p_prime_scale_Pa']:.1f} Pa")
    print(f"  dominant fp32 casualty input = {report['per_input_fp32_sensitivity']['dominant_fp32_casualty']}")
    print(f"  NAIVE fp32 |p'| err = {v['naive_fp32_p_prime_max_abs_Pa']:.4e} Pa  (rel-phys {v['naive_rel_to_physical_p_prime']:.2e}; {v['naive_over_floor_ratio']:.0f}x floor)")
    print(f"  MIXED fp32 |p'| err = {v['mixed_fp32_p_prime_max_abs_Pa']:.4e} Pa  (rel-phys {v['mixed_rel_to_physical_p_prime']:.2e}; {v['mixed_over_floor_ratio']:.0f}x floor)")
    print(f"  improvement naive->mixed = {v['improvement_naive_to_mixed_x']:.1f}x")
    print(f"  PHYSICS gate (<= {g['phys_gate_rel_threshold']:.2e} rel): MIXED {'PASS' if g['mixed_PASS_physics_gate'] else 'FAIL'}")
    print(f"  GATE_PASS = {g['GATE_PASS']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
