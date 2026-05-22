# Sprint Contract — c2-A2 WRF Small-Step PGF + Acoustic Implementation

**Sprint ID**: `2026-05-22-m6x-c2-A2-pgf-acoustic-implementation`
**Created**: 2026-05-22
**Status**: ACTIVE
**Trigger**: c2-A1 architecture skeleton + c2-A1' spike absorption + c2-A1'' PGF fix all ACCEPTED by Opus re-review. ADR-020 + state taxonomy + DycoreMetrics finalized with full WRF PGF spec including the 4th non-hydrostatic term, cf*/fnm/fnp coefficients, and implicit terrain cancellation.
**Branch**: `worker/codex/m6x-c2-A2-pgf-acoustic-implementation` (from main; main has c2-A1''-finalized architecture)
**Worktree**: `/tmp/wrf_gpu2_c2_a2`

## Objective

Implement the WRF small-step PGF + acoustic substep + mu-continuity per ADR-020 specification. Validate against warm-bubble (must survive >600s) and Skär mountain wave (must not blow up at 300s). Then run 1h coupled probe targeting sanitize <5% firing rate.

## Acceptance criteria

- **AC1 PGF implementation in `acoustic_wrf.py`**: 4 terms from ADR-020 (ph gradient + alt·dp + al·dpb + 4th non-hydrostatic via cf*/fnm/fnp). Cite WRF `module_small_step_em.F:828-862` (x) and `:902-936` (y) for every term.
- **AC2 al/alt as scan-carried intermediates**: per c2-A1'' R3 policy. Computed from `(p_perturbation, theta, ph_perturbation, mu_perturbation)` via WRF `module_em.F:1326,1340` `calc_p_rho_phi` pattern.
- **AC3 Acoustic substep scan**: `forward_backward_acoustic` with proper scan carry (state + previous-pressure for smdiv + al/alt + cqu/cqv). Diagnostic pressure from EoS each substep.
- **AC4 mu continuity inside acoustic substep**: WRF `module_small_step_em.F:1102-1108` pattern (not c1's post-loop update).
- **AC5 M1/M2 docs fix**: address minor docs from c2-A1'' re-review (M1 fnm/fnp average specification, M2 php/dpn substep-local doc).
- **AC6 Warm-bubble pass at 600s**: w_max in [5,10] m/s at 300s + 600s; bubble centroid >2500m at 300s, >3000m at 600s; ALL state leaves finite; matches Skamarock-Klemp 1994 reference.
- **AC7 Schär mountain wave**: stable for 600s+; no blow-up; mountain wave propagation visible per Schär 2002 reference.
- **AC8 1h coupled probe (sanitize ON)**: sanitize firing rate <5%; theta in [200,350]K (NOT clipped); mu in [5000,110000]Pa; nonfinite_count=0; matches Gen2 d02 wrfout reference.
- **AC9 24h coupled probe**: same bounds as AC8 at 24h.
- **AC10 Speedup re-measurement**: ≥4× (target preserved from c1-A7's 44× baseline; expect 20-30× per Gemini estimate after full architecture).
- **AC11 Tier-4 RMSE retest** (Option B harness): max_ratio < 1.5× (vs c1's 21.26×).
- **AC12 Mandatory Opus reviewer** at closeout.

## File ownership

- `src/gpuwrf/dynamics/acoustic_wrf.py` — IMPLEMENT (was skeleton in c2-A1)
- `src/gpuwrf/dynamics/orchestrator.py` — IMPLEMENT
- `src/gpuwrf/dynamics/damping.py` — wire smdiv into acoustic substep
- `src/gpuwrf/dynamics/hyperdiffusion.py` — wire if needed for stability
- `src/gpuwrf/dynamics/limiters.py` — wire if needed
- `src/gpuwrf/dynamics/metrics.py` — only if M1 fnm/fnp fix needed
- `src/gpuwrf/contracts/grid.py` — only docs (M1/M2)
- `src/gpuwrf/contracts/state.py` — only docs (M2)
- `scripts/m6_warm_bubble_test.py` — extend if needed
- `scripts/m6_full_domain_batching.py` — extend if needed
- `tests/test_m6x_c2_acoustic.py` (NEW), `test_m6x_c2_pgf.py` (NEW)
- `.agent/decisions/ADR-007-precision-policy.md` — Status → PASS after AC8-AC10 green
- `.agent/decisions/ADR-020-c2-dycore-architecture.md` — Status amendment after AC8+ green

## Hard rules

1. ADR-020 is the spec. NO deviations without manager-approved patch.
2. Cite WRF `module_small_step_em.F:lineno` for every numerical line
3. Sequential AC1-AC8 with probe verification between AC6, AC8, AC9 (no bundling)
4. NO sanitize masking of broken dynamics — if AC6 warm-bubble doesn't pass, debug FIRST
5. Empirical bisection methodology if you get stuck (per `[[feedback_bisection_before_theory]]`)
6. NO physics-kernel changes
7. NO M3/M4 test changes (existing baseline frozen)
8. BEFORE `/exit`: `git add . && git commit && git push`
9. `/exit` slash-command

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (MANDATORY at closeout)
- Wall-time: **5-7 working days** aggressive / 1-2 weeks conservative
- Worktree: `/tmp/wrf_gpu2_c2_a2`
- Branch: `worker/codex/m6x-c2-A2-pgf-acoustic-implementation`

## End-goal

If c2-A2 lands GREEN (warm-bubble 600s + 1h coupled + 24h coupled + Tier-4 RMSE all pass):
- M6.x closes
- M6-S8 model-consistency closeout dispatches
- M7-S0 Tier-4 RMSE harness dispatches
- M7 milestone work begins
- Path to operational Canary 3km daily forecast is clear

If c2-A2 FAILS specific AC → bisection per warm-bubble/1h-probe shows which term, manager dispatches focused fix sprint (c2-A2.x).

If c2-A2 fails STRUCTURALLY → escalate to user (rare but possible — Gemini already warned about XLA stencil compilation pressure risk).
