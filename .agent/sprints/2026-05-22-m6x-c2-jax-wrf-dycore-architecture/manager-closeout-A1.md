# c2-A1 Manager Closeout — Architecture Skeleton LANDED; ADR pending spike absorption

**Sprint**: c2-A1 JAX/WRF Dycore Architecture
**Status**: **SKELETON-COMPLETE — 7/8 ACs PASS, AC7 PARTIAL, ADR-002 amendment patch deferred for spike absorption**
**Date**: 2026-05-22 ~15:30
**Worker**: codex gpt-5.5 xhigh (~2.5 hours actual vs 3-5 day budget)
**Reviewer**: not yet dispatched (manager waits for c2-A1' or first Opus review)

## What c2-A1 produced

| AC | Status | Evidence |
|---|---|---|
| AC1 ADR/architecture | PASS | `architecture.md` + `ADR-020-c2-dycore-architecture.md` + ADR-002 amendment patch (deferred ACK pending spike) |
| AC2 metrics (msf*) | PASS | `proofs/metrics.json` — WRF wrfinput map-factor shapes loaded |
| AC3 hybrid-eta (c1h/c2h/c3h/c4h) | PASS | `proofs/hybrid_eta.json` — analytic oracle max error 0.0 |
| AC4 damping/diffusion/limiter skeletons | PASS | `tests/test_m6x_c2_stabilizers.py` |
| AC5 scan carry | PASS (static JAXPR audit + executed GPU scan) | `proofs/scan_transfer_audit.md` |
| AC6 limiter conservation | PASS (relative mass error 0.0) | `proofs/limiter_conservation.json` |
| AC7 integration warm-bubble | PARTIAL (analytic smoke only; warm-bubble script not in worktree) | `proofs/integration_warm_bubble.json` |
| AC8 decision gate | PASS | `manager-closeout.md` (worker draft) — recommends continue |

7 new files under `src/gpuwrf/dynamics/`: metrics.py, hybrid_eta.py, damping.py, hyperdiffusion.py, limiters.py, acoustic_wrf.py, orchestrator.py. 4 new test files (12 tests pass).

Worker correctly DEFERRED final ADR commits per manager's heads-up about parallel spike. Branch contains commit `124b351 Defer c2 ADR pending stability spike`.

## Parallel spike result (DEFINITIVE)

| Test | Result | Verdict |
|---|---|---|
| Test 1A flat warm-bubble | First nonfinite step 76 (150s) | FORMULATION ERROR |
| Test 1B Schär mountain | First nonfinite step 36 (70s) | Worsened (metric terms also needed but not sole cause) |
| Test 2 brute-force smdiv + Rayleigh | First nonfinite step 76 (150s) — UNCHANGED | Damping alone doesn't save it |

**Net spike verdict**: c1 has a fundamental FORMULATION ERROR (not just missing damping, not solely missing metric terms). Most likely WRF-style state-decomposition gap:
- Need native `p_total/p'/pb`, `ph_total/ph'/phb`, `mu/mub` decomposition from day 1
- Need well-balanced horizontal PGF with hydrostatic terrain-following slope cancellation (WRF's `ph/php, p/dpn, pb, al/alt`)
- Damping (smdiv, Rayleigh) is STABILIZER not architectural fix

## c2-A1' continuation needed

c2-A1 deferred the ADR-002 amendment patch acknowledgement. Now spike findings inform what the ADR must include:

1. **Variable-level base-state-vs-perturbation decomposition** in State pytree (not just `state.p - state.pb` inferred late)
2. **DycoreMetrics** as first-class pytree carrying `msf*`, `c*h/c*f`, terrain slopes `∂z/∂x`, `∂z/∂y`
3. **Well-balanced PGF formulation** specified in ADR (cite WRF `module_small_step_em.F:828-862, 902-936`)
4. **Damping as secondary stabilizer**, not architectural

Manager will dispatch:
1. c2-A1' (codex) — update ADR with spike-informed decomposition + state additions
2. Opus reviewer of c2-A1 + spike + c2-A1' for final acceptance

## What's preserved on main

- c2-A1 branch merged (architecture skeleton)
- Spike worker-report + role-prompt + test artifacts copied (NOT merged — temp acoustic.py changes excluded)
- Spike scripts: `m6_spike_test1_flat_vs_mountain.py`, `m6_spike_test2_brute_force_damping.py`
- Spike artifacts: 3 JSONs in `artifacts/m6/spike/`

## Gemini orthogonal angle PAID OFF

Without Gemini's spike recommendation (per user's "ask gemini for stuck after 2 iterations" rule), c2-A1 would have committed to wrong ADR. The 2-day spike (actual: 14 min) saved 3 weeks of c2 implementation building on wrong foundation.

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 ~15:30
