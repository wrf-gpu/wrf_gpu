#!/usr/bin/env python3
"""Phase 3 of the Fable PBL solve discriminator: audit invalidation + term
attribution of the authentic MYNN SP2 momentum baseline.

Inputs (all sealed / hash-bound):
  - WRF dump tree (462 files, tree SHA 88e94f6a...)
  - 28-array reference bundle (SHA a6416b72...)
  - gpu-operands.npz from the bit-exact phase-2 replay (this sprint), whose
    coefficient rebuild reproduced the sealed solve_* arrays bit-exactly.

Establishes, with committed numbers:

  I.   REPLAY-FLOOR ARTIFACT — the sealed surface/mixing "decisive
       falsifications" (SSE ratios ~200/~22) equal the replay-vs-baseline
       floor; the substitution effects are 1-2 orders below the floor.
  II.  TAUTOLOGY — the sealed residual_after_lower_bc equals
       (u_wrf - u_replay)/dt: a statement about the shared step-1 momentum
       initial condition (5e-7 RMS), not about coefficients.
  III. AUTHORITY MISMATCH — the tendencies implied by the sealed solve arrays
       (x - u)/dt differ from the sealed baseline sp2 tendencies by RMS
       ~2.5e-4, 14x the baseline-vs-WRF RMS. (The production-adapter path
       reproduces the replay bit-for-bit; see the companion adapter check.)
  IV.  STRUCTURE — the authentic baseline error is >99.99% in k0-4 (~74% k0).
  V.   TERM PARTITION — hybrid f64 Thomas solves on the bit-exact WRF system
       with port operand groups substituted row-by-row and term-by-term
       (kdz interface diffusion, mass-flux s_aw/s_awx, drag rhosfc/ust/wspd),
       including a full-depth sweep that reproduces the replay floor as its
       own sanity anchor.

No GPU, no JAX. Read-only on sealed roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v0234_fable_pbl_solve_discriminator import (  # noqa: E402
    BRIEF_ROOT, DUMP_ROOT, WrfDump, thomas, sha256_file,
)

DELT = 6.0
SPRINT = Path("<USER_HOME>/src/wrf_gpu2_wt/v0234-kimi-pbl-solve-discriminator/"
              ".agent/sprints/2026-07-19-v0234-fable-pbl-solve-discriminator")
OPERANDS_SHA256_FROM_PHASE2_PROOF = True


def S(x) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sum(x * x))


def rms(x) -> float:
    x = np.asarray(x, np.float64)
    return float(np.sqrt(np.mean(x * x)))


def corr(a, b) -> float:
    return float(np.sum(np.asarray(a, np.float64) * np.asarray(b, np.float64))
                 / max(math.sqrt(S(a) * S(b)), 1e-300))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operands", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        print("REFUSE: output exists", file=sys.stderr)
        return 74

    proof: dict = {"schema": "wrfgpu2-v0234-fable-pbl-attribution-audit-v1"}
    operands_path = Path(args.operands)
    proof["inputs"] = {
        "reference_archive_sha256": sha256_file(BRIEF_ROOT / "cpu-reference/pbl-sp2-reference-28.npz"),
        "operands_sha256": sha256_file(operands_path),
    }

    dump = WrfDump(DUMP_ROOT)
    tags = ("bc_u", "bc_v", "bc_dtz", "bc_rho", "bc_lower_operands", "bc_kmdz",
            "bc_s_aw", "bc_s_awu", "bc_s_awv",
            "solve_u_a", "solve_u_b", "solve_u_c", "solve_u_d", "solve_v_d")
    w = {t: dump.columns(t).astype(np.float64) for t in tags}
    wrf_t = {"u": dump.outer("rublten_exit", 3), "v": dump.outer("rvblten_exit", 3)}
    sealed = np.load(BRIEF_ROOT / "cpu-reference/pbl-sp2-reference-28.npz", allow_pickle=False)
    ops = np.load(operands_path, allow_pickle=False)

    base = {c: np.asarray(sealed[f"sp2_r{c}blten"]) - wrf_t[c] for c in ("u", "v")}
    u_rep = {c: np.asarray(ops[c]) for c in ("u", "v")}
    x_rep = {c: np.asarray(sealed[f"solve_{c}_x"]) for c in ("u", "v")}

    # --- I. replay floor vs sealed surface/mixing residuals -----------------
    sealed_cmp = json.loads((BRIEF_ROOT / "comparator/scientific-comparison.json").read_text())
    floor_stage: dict = {}
    for c in ("u", "v"):
        floor = (x_rep[c] - u_rep[c]) / DELT - wrf_t[c]
        entry = {
            "replay_floor_rms": rms(floor),
            "replay_floor_sse_ratio_vs_baseline": S(floor) / S(base[c]),
            "sealed_surface_sse_ratio": sealed_cmp["quantitative_attribution"]["spans"]["surface"][f"r{c}blten"]["sse_ratio"],
            "sealed_mixing_sse_ratio": sealed_cmp["quantitative_attribution"]["spans"]["mixing"][f"r{c}blten"]["sse_ratio"],
        }
        for span in ("surface", "mixing"):
            res = np.asarray(sealed[f"residual_after_{span}_r{c}blten"])
            entry[f"{span}_residual_minus_floor_rms"] = rms(res - floor)
        floor_stage[c] = entry
    proof["I_replay_floor_artifact"] = floor_stage

    # --- II. tautology ------------------------------------------------------
    taut: dict = {}
    for c in ("u", "v"):
        res_l = np.asarray(sealed[f"residual_after_lower_bc_r{c}blten"])
        ident = (w[f"bc_{c}"] - u_rep[c]) / DELT
        taut[c] = {
            "sealed_lower_bc_residual_rms": rms(res_l),
            "state_identity_rms": rms(ident),
            "difference_rms": rms(res_l - ident),
            "u_replay_vs_wrf_state_rms": rms(u_rep[c] - w[f"bc_{c}"]),
        }
    proof["II_lower_bc_residual_is_state_tautology"] = taut

    # --- III. authority mismatch -------------------------------------------
    auth: dict = {}
    for c in ("u", "v"):
        implied = (x_rep[c] - u_rep[c]) / DELT
        delta = implied - np.asarray(sealed[f"sp2_r{c}blten"])
        auth[c] = {
            "sealed_solve_implied_tendency_vs_sealed_sp2_rms": rms(delta),
            "sealed_solve_implied_tendency_vs_sealed_sp2_max": float(np.max(np.abs(delta))),
            "baseline_vs_wrf_rms": rms(base[c]),
            "ratio": rms(delta) / rms(base[c]),
        }
    proof["III_bundle_authority_mismatch"] = auth

    # --- IV. baseline structure --------------------------------------------
    struct: dict = {}
    for c in ("u", "v"):
        b = base[c]
        struct[c] = {
            "per_k_rms_k0_to_k7": [rms(b[k]) for k in range(8)],
            "sse_fraction_k0": S(b[0]) / S(b),
            "sse_fraction_k0_4": S(b[:5]) / S(b),
        }
    proof["IV_baseline_structure"] = struct

    # --- V. term partition --------------------------------------------------
    low = w["bc_lower_operands"]
    drag_w = low[10] * low[6] * low[6] / low[7]
    drag_p = np.asarray(ops["rhosfc"]) * np.asarray(ops["ustar"]) ** 2 / np.asarray(ops["wind"])
    dtz = w["bc_dtz"]
    rhoinv = np.empty_like(w["bc_rho"])
    rhoinv[0] = 1.0 / w["bc_rho"][0]
    rhoinv[1:] = 1.0 / np.maximum(w["bc_rho"][1:], 1e-4)
    P = dtz * rhoinv
    dk = np.asarray(ops["kdz_floored"]) - w["bc_kmdz"]
    ds = np.asarray(ops["s_aw"]) - w["bc_s_aw"]
    dsx = {"u": np.asarray(ops["s_awu"]) - w["bc_s_awu"],
           "v": np.asarray(ops["s_awv"]) - w["bc_s_awv"]}
    nz = w["bc_u"].shape[0]

    proof["V_operand_deltas"] = {
        "ust_rms": rms(np.asarray(ops["ustar"]) - low[6]),
        "ust_rel_rms": rms((np.asarray(ops["ustar"]) - low[6]) / np.maximum(low[6], 1e-3)),
        "wspd_rms": rms(np.asarray(ops["wind"]) - low[7]),
        "rhosfc_rms": rms(np.asarray(ops["rhosfc"]) - low[10]),
        "rhosfc_rel_mean": float(np.mean(np.asarray(ops["rhosfc"]) / low[10])),
        "kdz_interface_rms_1_2_3": [rms(dk[k]) for k in (1, 2, 3)],
        "s_aw_interface_rms_1_2_3": [rms(ds[k]) for k in (1, 2, 3)],
        "s_awu_interface_rms_1_2_3": [rms(dsx["u"][k]) for k in (1, 2, 3)],
    }

    def build(K: int, kdz: bool, saw: bool, drag: bool):
        A = w["solve_u_a"].copy()
        B = w["solve_u_b"].copy()
        C = w["solve_u_c"].copy()
        D = {"u": w["solve_u_d"].copy(), "v": w["solve_v_d"].copy()}
        if drag:
            B[0] += P[0] * (drag_p - drag_w)
        if kdz:
            B[0] += P[0] * dk[1]
            C[0] += -P[0] * dk[1]
        if saw:
            B[0] += -0.5 * P[0] * ds[1]
            C[0] += -0.5 * P[0] * ds[1]
            for c in ("u", "v"):
                D[c][0] += -P[0] * dsx[c][1]
        for k in range(1, min(K + 1, nz - 1)):
            if kdz:
                A[k] += -P[k] * dk[k]
                B[k] += P[k] * (dk[k] + dk[k + 1])
                C[k] += -P[k] * dk[k + 1]
            if saw:
                A[k] += 0.5 * P[k] * ds[k]
                B[k] += 0.5 * P[k] * (ds[k] - ds[k + 1])
                C[k] += -0.5 * P[k] * ds[k + 1]
                for c in ("u", "v"):
                    D[c][k] += P[k] * (dsx[c][k] - dsx[c][k + 1])
        return A, B, C, D

    def evaluate(A, B, C, D) -> dict:
        out = {}
        for c in ("u", "v"):
            x = thomas(A, B, C, D[c])
            err = (x - w[f"bc_{c}"]) / DELT - wrf_t[c]
            out[c] = {
                "predicted_sse_over_baseline": S(err) / S(base[c]),
                "unexplained_sse_fraction": S(base[c] - err) / S(base[c]),
                "correlation_with_baseline": corr(err, base[c]),
            }
        return out

    partition: dict = {"depth_sweep_all_groups": {}, "row0_terms": {}, "rows0_4_groups": {}}
    for K in (0, 1, 2, 4, 6, 10, 20, nz - 2):
        partition["depth_sweep_all_groups"][f"rows0_{K}"] = evaluate(*build(K, True, True, True))
    for name, kw in (
        ("drag_only", dict(kdz=False, saw=False, drag=True)),
        ("kdz1_only", dict(kdz=True, saw=False, drag=False)),
        ("saw_only", dict(kdz=False, saw=True, drag=False)),
        ("all", dict(kdz=True, saw=True, drag=True)),
    ):
        partition["row0_terms"][name] = evaluate(*build(0, **kw))
    for name, kw in (
        ("kdz_only", dict(kdz=True, saw=False, drag=False)),
        ("saw_only", dict(kdz=False, saw=True, drag=False)),
        ("kdz_drag", dict(kdz=True, saw=False, drag=True)),
        ("all", dict(kdz=True, saw=True, drag=True)),
    ):
        partition["rows0_4_groups"][name] = evaluate(*build(4, **kw))

    # drag sub-partition (rhosfc vs ust) at row 0 only
    for name, rhosfc_h, ust_h, wspd_h in (
        ("drag_rhosfc_only", np.asarray(ops["rhosfc"]), low[6], low[7]),
        ("drag_ust_only", low[10], np.asarray(ops["ustar"]), low[7]),
    ):
        drag_h = rhosfc_h * ust_h * ust_h / wspd_h
        A = w["solve_u_a"].copy()
        B = w["solve_u_b"].copy()
        C = w["solve_u_c"].copy()
        B[0] = B[0] + P[0] * (drag_h - drag_w)
        D = {"u": w["solve_u_d"].copy(), "v": w["solve_v_d"].copy()}
        partition["row0_terms"][name] = evaluate(A, B, C, D)
    proof["V_term_partition"] = partition

    canonical = json.dumps(proof, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    proof["canonical_payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.write_text(json.dumps(proof, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "floor_ratio_u": floor_stage["u"]["replay_floor_sse_ratio_vs_baseline"],
        "sealed_surface_ratio_u": floor_stage["u"]["sealed_surface_sse_ratio"],
        "tautology_diff_rms_u": taut["u"]["difference_rms"],
        "authority_mismatch_ratio_u": auth["u"]["ratio"],
        "baseline_k0_4_fraction_u": struct["u"]["sse_fraction_k0_4"],
        "best_row01_corr_u": partition["depth_sweep_all_groups"]["rows0_1"]["u"]["correlation_with_baseline"],
        "proof": str(output),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
