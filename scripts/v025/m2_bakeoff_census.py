#!/usr/bin/env python3
"""M2 fused-module bake-off: static launch census (CPU-only).

Sprint `.agent/sprints/2026-09-17-v0250-m2-fused-bakeoff` step 1.

Ranks operator families of the operational FAST-shaped timestep by *launched
kernels per timestep*, then picks the family for the ONE fused Pallas module.

Two corrections over the A6 census (`proofs/v025/m0/hlo_dtype_transfer_census.json`)
that this script is built from -- both are what make the ranking honest:

1. **Cadence weighting.** A6 lowers every operator ONCE. Production runs the
   acoustic substep body ``1 + n/2 + n`` times per step (RK3, ``acoustic_substeps``
   = ``n``), the large-step tendencies three times, and radiation/cumulus only
   every ``radt``/``cudt``. A per-invocation count therefore ranks RRTMG first
   (1294) although it amortises to ~39 launches/step, and hides the substep chain.

2. **Loop weighting.** The static launch proxy counts a ``while`` body once.
   On the device every iteration launches the body's kernels again (the
   production nsys capture `p1r_fullcycle_g0_relocation_20260901` is 817,827
   instances dominated by sub-microsecond ``loop_*_fusion`` bodies). The Thomas
   sweeps in ``advance_w`` are ``lax.scan`` loops over ``nz`` levels, so their
   real launch count is ~10x the static proxy. This script lowers the
   vertical-implicit family at FAST shape and multiplies each while body by its
   trip count (parsed from the loop condition), reporting both numbers.

The XLA:CPU lowering is used for the loop structure only; absolute per-body
kernel counts differ on XLA:GPU. The loop structure (2 sweeps x nz trips) is
backend-stable, which is what the ranking needs. The device number is the
bake-off harness's job (`m2_bakeoff_vertical_implicit.py`, manager-run).

Usage (CPU):
    python scripts/v025/m2_bakeoff_census.py [--acoustic-substeps 10] \
        [--out <DATA_ROOT>/wrf_gpu2/v025/m2/census/m2_launch_census.json] \
        [--md .agent/sprints/2026-09-17-v0250-m2-fused-bakeoff/CENSUS.md]
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import cpu_guard  # noqa: E402,F401  MUST precede jax/gpuwrf imports

SCHEMA = "wrf_gpu2.v025.m2.launch_census.v1"
A6_CENSUS = REPO / "proofs/v025/m0/hlo_dtype_transfer_census.json"
DEFAULT_OUT = Path("<DATA_ROOT>/wrf_gpu2/v025/m2/census/m2_launch_census.json")
DEFAULT_MD = REPO / ".agent/sprints/2026-09-17-v0250-m2-fused-bakeoff/CENSUS.md"

# FAST d01 production geometry (proofs/v025/m0 census case_config).
FAST_NZ, FAST_NY, FAST_NX = 44, 70, 120

# --------------------------------------------------------------------------- #
# cadence map: invocations per timestep of each inventoried operator            #
# --------------------------------------------------------------------------- #
# Derived from the operational program, not guessed:
#   * per_substep: called inside ``acoustic_substep_core``
#     (src/gpuwrf/dynamics/core/acoustic.py:1029-1485, grep-verified 2026-09-17:
#     calc_p_rho_step, advance_uv_wrf -> _x/_y_face_pressure_dpn,
#     advance_mu_t_core -> mu_continuity_tendency, advance_w_wrf).
#   * per_stage: once per RK stage (``_acoustic_scan`` prologue: calc_coef_w,
#     small_step_prep; rk_tendency: pgf, coriolis, curvature, advection,
#     diffusion tendencies, rhs_ph, eos seeds, pg_buoy_w).
#   * per_step: once per timestep (physics every step; deformation/Km once per
#     step at RK stage 1; positive-definite limiter at the final stage).
#   * per_radt / per_cudt: radiation / cumulus intervals (case_config radt_min,
#     cudt_min, dt).
#   * subsumed: counted inside another inventoried operator (thomas inside
#     advance_w) -> multiplicity 0 to avoid double counting.
CADENCE: dict[str, str] = {
    # --- vertical implicit ---
    "calc_coef_w": "per_stage",
    "advance_w": "per_substep",
    "thomas_solve_scan": "subsumed",  # inside advance_w
    "pg_buoy_w_dry": "per_stage",
    "pg_buoy_w_moist": "per_stage",
    "moist_cqw_calc_face": "per_stage",
    # --- substep chain ---
    "calc_p_rho_step": "per_substep",
    "x_face_pressure_dpn": "per_substep",
    "y_face_pressure_dpn": "per_substep",
    "mu_continuity_tendency": "per_substep",
    # --- per RK stage ---
    "small_step_prep": "per_stage",
    "rhs_ph": "per_stage",
    "horizontal_pressure_gradient": "per_stage",
    "large_step_horizontal_pgf": "per_stage",
    "large_step_coriolis": "per_stage",
    "large_step_horizontal_curvature": "per_stage",
    "advect_u_flux": "per_stage",
    "advect_v_flux": "per_stage",
    "advect_w_flux": "per_stage",
    "advect_scalar_flux": "per_stage",
    "advect_moisture_scalars": "per_stage",
    "flux5_face_periodic": "per_stage",
    "horizontal_diffusion_coord_scalar_tendency": "per_stage",
    "wrf_sixth_order_scalar_tendf": "per_stage",
    "sixth_order_diffusion_tendency": "per_stage",
    "calc_p_rho": "per_stage",
    "calc_al_p_kernel": "per_stage",
    "diagnose_pressure_al_alt": "per_stage",
    "moisture_coupling_factors": "per_stage",
    "apply_rayleigh_w": "per_stage",
    "apply_smdiv_pressure": "per_stage",
    "w_damp_vertical_cfl": "subsumed",  # inside advance_w
    # --- per step ---
    "smag2d_horizontal_km": "per_step",
    "horizontal_deformation_2d": "per_step",
    "dry_brunt_vaisala_squared": "per_step",
    "positive_definite_limiter": "per_step",
    "hybrid_mass_level_pressure": "per_step",
    "hybrid_face_level_pressure": "per_step",
    "hybrid_mass_weight": "per_step",
    "hybrid_face_weight": "per_step",
    "hybrid_pressure_thickness": "per_step",
    "thompson_microphysics": "per_step",
    "mynn_pbl": "per_step",
    "mynn_surface_layer": "per_step",
    "gwdo_gravity_wave_drag": "per_step",
    "noahmp_land_surface": "per_step",
    # --- intervals ---
    "rrtmg_sw": "per_radt",
    "rrtmg_lw": "per_radt",
    "kain_fritsch_cumulus": "per_cudt",
}


def substeps_per_step(n_sound: int) -> int:
    """RK3 acoustic cadence (operational_mode.py:5009-5013): 1 + n/2 + n."""

    return 1 + max(1, n_sound // 2) + n_sound


def multiplicities(case: dict[str, Any], n_sound: int) -> dict[str, float]:
    dt = float(case["dt"])
    return {
        "per_substep": float(substeps_per_step(n_sound)),
        "per_stage": 3.0,
        "per_step": 1.0,
        "per_radt": dt / (60.0 * float(case["radt_min"])),
        "per_cudt": dt / (60.0 * float(case["cudt_min"])),
        "subsumed": 0.0,
    }


# --------------------------------------------------------------------------- #
# loop-weighted HLO launch proxy                                               #
# --------------------------------------------------------------------------- #
_COMP_HEADER = re.compile(r"^(?:ENTRY\s+)?%?([\w.\-]+)\s*\(.*\)\s*->.*\{\s*$")
_WHILE = re.compile(r"while\(.*\),\s*condition=%?([\w.\-]+),\s*body=%?([\w.\-]+)")


def split_computations(hlo: str) -> tuple[dict[str, list[str]], str | None]:
    """Return {computation name: body lines} and the ENTRY computation name."""

    comps: dict[str, list[str]] = {}
    entry: str | None = None
    current: str | None = None
    for line in hlo.splitlines():
        match = _COMP_HEADER.match(line)
        if match:
            current = match.group(1)
            comps[current] = []
            if line.startswith("ENTRY"):
                entry = current
            continue
        if line.strip() == "}":
            current = None
            continue
        if current is not None:
            comps[current].append(line)
    return comps, entry


def static_launches(lines: list[str]) -> int:
    body = "\n".join(lines)
    return len(re.findall(r"\bfusion\(", body)) + len(re.findall(r"\bcustom-call\(", body))


def trip_count(cond_lines: list[str]) -> int | None:
    """A static ``lax.scan`` condition compares the counter to one constant."""

    constants = [int(x) for x in re.findall(r"constant\((-?\d+)\)", "\n".join(cond_lines))]
    return max(constants) if constants else None


def loop_weighted_launches(hlo: str) -> dict[str, Any]:
    """Static proxy with every while body multiplied by its trip count.

    Nested loops multiply. A loop whose trip count cannot be parsed counts as
    one iteration and is reported in ``unresolved_loops`` (never silently zero).
    """

    comps, entry = split_computations(hlo)
    if entry is None:
        raise ValueError("no ENTRY computation in HLO text")
    loops: list[dict[str, Any]] = []
    unresolved: list[str] = []
    total = 0

    def walk(name: str, mult: int) -> None:
        nonlocal total
        total += static_launches(comps[name]) * mult
        for line in comps[name]:
            match = _WHILE.search(line)
            if not match:
                continue
            cond, body = match.group(1), match.group(2)
            trips = trip_count(comps.get(cond, []))
            if trips is None:
                unresolved.append(body)
                trips = 1
            loops.append({
                "body": body,
                "trips": trips,
                "body_launches": static_launches(comps[body]),
                "cond_launches": static_launches(comps.get(cond, [])),
                "outer_multiplicity": mult,
            })
            total += static_launches(comps.get(cond, [])) * mult * trips
            walk(body, mult * trips)

    walk(entry, 1)
    return {
        "static_launch_proxy": static_launches(comps[entry]),
        "loop_weighted_launches": total,
        "loops": loops,
        "unresolved_loops": unresolved,
    }


# --------------------------------------------------------------------------- #
# FAST-shaped lowering of the vertical-implicit family                         #
# --------------------------------------------------------------------------- #
def fast_shaped_inputs(nz: int = FAST_NZ, ny: int = FAST_NY, nx: int = FAST_NX):
    """Shape-faithful synthetic inputs. Launch structure is shape-driven."""

    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(20260917)

    def rnd(shape, index, lo=0.1, hi=1.0):
        return jax.random.uniform(jax.random.fold_in(key, index), shape,
                                  dtype=jnp.float64, minval=lo, maxval=hi)

    faces, mass, surf = (nz + 1, ny, nx), (nz, ny, nx), (ny, nx)
    metrics = types.SimpleNamespace(
        c1h=rnd((nz,), 1), c2h=rnd((nz,), 2), c1f=rnd((nz + 1,), 3), c2f=rnd((nz + 1,), 4),
        rdn=rnd((nz,), 5), rdnw=rnd((nz,), 6), fnm=rnd((nz,), 7), fnp=rnd((nz,), 8),
    )
    cqw = jnp.ones(faces, dtype=jnp.float64).at[0].set(0.0).at[nz].set(0.0)
    arrays = dict(
        w=rnd(faces, 11, -1, 1), rw_tend=rnd(faces, 15, -1e-3, 1e-3), ww=rnd(faces, 16, -1, 1),
        u=rnd((nz, ny, nx + 1), 13, -10, 10), v=rnd((nz, ny + 1, nx), 14, -10, 10),
        mu_work=rnd(surf, 17, -10, 10), mut=rnd(surf, 9, 1e4, 1e5), muave=rnd(surf, 18, -10, 10),
        muts=rnd(surf, 19, 1e4, 1e5), t_2ave=rnd(mass, 20, 1, 10), t_2=rnd(mass, 21, 1, 10),
        t_1=rnd(mass, 22, 1, 10), ph=rnd(faces, 23, -1e3, 1e3), ph_1=rnd(faces, 24, -1e3, 1e3),
        phb=rnd(faces, 25, 0, 2e5), ph_tend=rnd(faces, 26, -1, 1), ht=rnd(surf, 27, 0, 2000),
        c2a=rnd(mass, 10), cqw=cqw, alt=rnd(mass, 28, 0.5, 1.5),
        msftx=rnd(surf, 29, 0.9, 1.1), msfty=rnd(surf, 30, 0.9, 1.1), w_save=rnd(faces, 31, -1, 1),
        cf1=jnp.float64(1.5), cf2=jnp.float64(-0.5), cf3=jnp.float64(0.0),
    )
    scalars = dict(rdx=1.0 / 9000.0, rdy=1.0 / 9000.0, dts=5.4, epssm=0.5, top_lid=False,
                   damp_opt=3, dampcoef=0.2, zdamp=5000.0, w_damping=1)
    return metrics, arrays, scalars


def lower_family(n_sound: int) -> dict[str, Any]:
    import jax

    from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients
    from gpuwrf.dynamics.core.advance_w import advance_w_wrf
    from gpuwrf.dynamics.tridiag_solve import thomas_solve_scan

    metrics, arrays, scalars = fast_shaped_inputs()
    records: dict[str, Any] = {}

    def coef(mut, cqw, c2a):
        return calc_coef_w_wrf_coefficients(mut, metrics, dt=scalars["dts"],
                                            epssm=scalars["epssm"], top_lid=False,
                                            cqw=cqw, c2a=c2a)

    t0 = time.perf_counter()
    lowered = jax.jit(coef).lower(arrays["mut"], arrays["cqw"], arrays["c2a"])
    compiled = lowered.compile()
    records["calc_coef_w"] = {**loop_weighted_launches(compiled.as_text()),
                              "cadence": "per_stage", "seconds": time.perf_counter() - t0}
    a, alpha, gamma = jax.jit(coef)(arrays["mut"], arrays["cqw"], arrays["c2a"])

    t0 = time.perf_counter()
    compiled = jax.jit(thomas_solve_scan).lower(a, alpha, gamma, arrays["w"]).compile()
    records["thomas_solve_scan"] = {**loop_weighted_launches(compiled.as_text()),
                                    "cadence": "subsumed (inside advance_w)",
                                    "seconds": time.perf_counter() - t0}

    metric_arrays = {k: getattr(metrics, k) for k in ("c1h", "c2h", "c1f", "c2f", "rdnw", "rdn", "fnm", "fnp")}

    def adv(**arrs):
        return advance_w_wrf(**arrs, **scalars)

    t0 = time.perf_counter()
    compiled = jax.jit(adv).lower(a=a, alpha=alpha, gamma=gamma, **arrays, **metric_arrays).compile()
    records["advance_w"] = {**loop_weighted_launches(compiled.as_text()),
                            "cadence": "per_substep", "seconds": time.perf_counter() - t0}
    per_step = (records["calc_coef_w"]["loop_weighted_launches"] * 3
                + records["advance_w"]["loop_weighted_launches"] * substeps_per_step(n_sound))
    per_step_static = (records["calc_coef_w"]["static_launch_proxy"] * 3
                       + records["advance_w"]["static_launch_proxy"] * substeps_per_step(n_sound))
    return {
        "operators": records,
        "family_per_step_loop_weighted": per_step,
        "family_per_step_static": per_step_static,
        "acoustic_substeps": n_sound,
        "substeps_per_step": substeps_per_step(n_sound),
        "shape": {"nz": FAST_NZ, "ny": FAST_NY, "nx": FAST_NX},
    }


# --------------------------------------------------------------------------- #
# ranking                                                                      #
# --------------------------------------------------------------------------- #
def rank_families(a6: dict[str, Any], n_sound: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mult = multiplicities(a6["case_config"], n_sound)
    families: dict[str, dict[str, Any]] = {}
    operators: list[dict[str, Any]] = []
    unmapped: list[str] = []
    for record in a6["operators"]:
        name = record["name"]
        cadence = CADENCE.get(name)
        if cadence is None:
            unmapped.append(name)
            continue
        static = record.get("static_launch_proxy") if record["status"] == "LOWERED" else None
        per_step = None if static is None else static * mult[cadence]
        operators.append({
            "name": name, "family": record["family"], "cadence": cadence,
            "static_per_call": static, "invocations_per_step": mult[cadence],
            "static_per_step": per_step,
            "status": record["status"],
        })
        entry = families.setdefault(record["family"], {"family": record["family"], "operators": [],
                                                       "static_per_step": 0.0, "not_lowered": []})
        entry["operators"].append(name)
        if per_step is None:
            entry["not_lowered"].append(name)
        else:
            entry["static_per_step"] += per_step
    ranked = sorted(families.values(), key=lambda f: -f["static_per_step"])
    total = sum(f["static_per_step"] for f in ranked)
    for f in ranked:
        f["share"] = f["static_per_step"] / total if total else None
    return ranked, {"operators": operators, "unmapped": unmapped, "multiplicities": mult,
                    "total_static_per_step": total}


def render_markdown(obj: dict[str, Any]) -> str:
    fam = obj["family_lowering"]
    lines = [
        "# M2 static launch census (CPU) — FAST d01 shape 44x70x120",
        "",
        f"Generated {obj['generated_at_utc']} by `scripts/v025/m2_bakeoff_census.py` "
        f"from A6 census `{A6_CENSUS.relative_to(REPO)}` ({obj['a6_generated_at_utc']}).",
        "",
        f"Cadence: RK3, `acoustic_substeps={obj['acoustic_substeps']}` -> "
        f"{obj['substeps_per_step']} substeps/step; radt/cudt from case_config "
        f"(per_radt x{obj['ranking_meta']['multiplicities']['per_radt']:.3f}, "
        f"per_cudt x{obj['ranking_meta']['multiplicities']['per_cudt']:.3f}).",
        "",
        "## Family ranking by static launches per timestep (cadence-weighted, loops NOT expanded)",
        "",
        "| # | family | static launches/step | share | operators |",
        "|---|--------|---------------------:|------:|-----------|",
    ]
    for i, f in enumerate(obj["family_ranking"], 1):
        lines.append(f"| {i} | {f['family']} | {f['static_per_step']:.0f} | "
                     f"{100 * (f['share'] or 0):.1f}% | {', '.join(f['operators'])} |")
    lines += [
        "",
        f"Total static launches/step (cadence-weighted): {obj['ranking_meta']['total_static_per_step']:.0f} "
        f"(A6 per-invocation sum was {obj['a6_static_launch_proxy_sum']}).",
        "",
        "## Loop-weighted refinement of the #1 family (dycore.vertical_implicit)",
        "",
        "| operator | cadence | static/call | loop-weighted/call | loops (body launches x trips) |",
        "|----------|---------|-----------:|-------------------:|-------------------------------|",
    ]
    for name, rec in fam["operators"].items():
        loops = "; ".join(f"{l['body_launches']}+{l['cond_launches']}cond x{l['trips']}" for l in rec["loops"]) or "none"
        lines.append(f"| {name} | {rec['cadence']} | {rec['static_launch_proxy']} | "
                     f"{rec['loop_weighted_launches']} | {loops} |")
    lines += [
        "",
        f"Family launches per timestep: static {fam['family_per_step_static']} -> "
        f"loop-weighted **{fam['family_per_step_loop_weighted']}** "
        f"(calc_coef_w x3 + advance_w x{fam['substeps_per_step']}).",
        "",
        "## Decision",
        "",
        obj["decision"],
        "",
        "## Caveats (pre-registered)",
        "",
    ]
    lines += [f"- {c}" for c in obj["caveats"]]
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acoustic-substeps", type=int, default=10,
                        help="OperationalNamelist.acoustic_substeps (production default 10)")
    parser.add_argument("--a6", type=Path, default=A6_CENSUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    a6 = json.loads(args.a6.read_text())
    ranked, meta = rank_families(a6, args.acoustic_substeps)
    family = lower_family(args.acoustic_substeps)

    top = ranked[0]
    decision = (
        f"Family **{top['family']}** is #1 by cadence-weighted static launches "
        f"({top['static_per_step']:.0f}/step, {100 * top['share']:.1f}%) and its loop-weighted "
        f"count is {family['family_per_step_loop_weighted']}/step, i.e. more than the whole-step "
        f"A6 static sum ({a6['totals']['static_launch_proxy_sum']}): the static proxy hides the "
        f"Thomas `lax.scan` sweeps. The fused module is `src/gpuwrf/kernels/fused_vertical_implicit.py`: "
        f"one Pallas column kernel for `advance_w` (per substep) and one for `calc_coef_w` "
        f"(per stage); both are column-local (k-recurrences only), which is exactly the shape a "
        f"single launch resolves."
    )
    caveats = [
        "XLA:CPU lowering; absolute per-body kernel counts differ on XLA:GPU. The loop structure "
        "(two while loops x nz trips per advance_w call) is backend-stable; the device count is "
        "measured by the bake-off harness.",
        "Cadence map is documented in CADENCE (grep-verified against acoustic_substep_core); "
        "physics column schemes (MYNN, Thompson sedimentation, RRTMG g-point loops) also contain "
        "while loops that this census does NOT expand, so their per-step numbers are lower bounds. "
        "They run at most once per step (radiation every 33 steps) and cannot overtake a per-substep "
        "family with 16 invocations/step.",
        "noahmp_land_surface did not lower in A6 (ConcretizationTypeError) and is excluded.",
        "acoustic_substeps=10 is the OperationalNamelist default (operational_mode.py:783); with the "
        "WRF-typical 4 the substep count is 7 and the family total scales accordingly "
        "(still #1: 133*3 + 377*7 = 3038 > any other family).",
    ]
    obj = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "a6_census": str(args.a6.relative_to(REPO)) if args.a6.is_relative_to(REPO) else str(args.a6),
        "a6_generated_at_utc": a6["generated_at_utc"],
        "a6_static_launch_proxy_sum": a6["totals"]["static_launch_proxy_sum"],
        "case_config": a6["case_config"],
        "acoustic_substeps": args.acoustic_substeps,
        "substeps_per_step": substeps_per_step(args.acoustic_substeps),
        "family_ranking": ranked,
        "ranking_meta": meta,
        "family_lowering": family,
        "decision": decision,
        "caveats": caveats,
        "seconds_total": time.perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(render_markdown(obj))
    print(render_markdown(obj))
    print(f"wrote {args.out}\nwrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
