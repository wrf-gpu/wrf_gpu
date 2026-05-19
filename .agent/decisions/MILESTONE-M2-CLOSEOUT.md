# Milestone M2 Closeout — Backend Bakeoff

Date: 2026-05-19
Status: **CLOSED on manager side; M3 dispatch gated on explicit user approval of ADR-001.**

## Summary

M2 establishes the project's GPU backend through evidence-driven comparison of six candidate families. **JAX is selected as v0 primary** with a per-scheme gated Triton fallback for M5 physics. The decision rests on M2 micro-fixture profile evidence + cross-AI verification + Codex critical review; final ratification awaits explicit human approval at this closeout.

## Closed Sprints

| ID | Outcome | Cycles |
|---|---|---|
| `2026-05-19-m2-scout-blackwell-toolchain` (S1) | Accept | 1w + 1t (Claude) + 1r — 0 fix cycles |
| `2026-05-19-m2-cuda-tile-stencil-column` (S2) | Accept-with-fixes | 1w + 1t (Claude) + 1r — 3 reviewer hygiene fixes applied at closeout |
| `2026-05-19-m2-cupy-stencil-column` (S3) | Accept | 1w + 1t (Claude) + 1r — 0 fix cycles |
| `2026-05-19-m2-kokkos-stencil-column` (S4) | Accept | 1w + 1t (Claude) + 1r — 0 fix cycles |
| `2026-05-19-m2-jax-stencil-column` (S5) | Accept | 1w + 1t (Claude, hard-verified HLO+thunk+cubin) + 1r — 0 fix cycles |
| `2026-05-19-m2-triton-stencil-column` (S6) | Accept (attempt 2) | 2w + 1t (Claude — found cubin-cache bug) + 2r — 1 fix cycle |
| `2026-05-19-m2-adr-001-backend-selection` (S8) | Accept (this closeout) | manager + Codex critical-review + binding reviewer — 11 findings total, all applied |

GT4Py (S7) was not implemented as a separate sprint; the M2-S1 scout established `blocked` (DaCe 0.10.0 / Python 3.13 SymPy break) and S8 produced the candidate-failure artifacts in the required format.

## Bakeoff Results (5-way + 1 excluded)

### Stencil (3D advection-diffusion, 32×16×8, fp64)

| Candidate | regs | local | occ% | launches | wall (ms) |
|---|---|---|---|---|---|
| cuda_tile | 58 | 0 | 66.7 | 1 | 0.92 |
| cupy | 58 | 64 | 66.7 | 1 | 0.06 |
| kokkos | 64 | 0 | 66.7 | 1 | 0.09 |
| **jax** | **48** | 0 | **83.3** | 1 | 0.05 |
| triton | 60 | 0 | 66.7 | 1 | 0.03 |
| gt4py | EXCLUDED (toolchain) | | | | |

### Column (40-level moist thermo, fp64)

| Candidate | regs | local | occ% | launches | wall (ms) |
|---|---|---|---|---|---|
| cuda_tile | 24 | 0 | 100 | 1 | 1.00 |
| cupy | 24 | 0 | 100 | 1 | 0.03 |
| kokkos | 40 | 0 | 100 | 1 | 0.10 |
| **jax** | **22** | 0 | 83.3 | 1 | 0.05 |
| triton | 34 | 0 | 100 | 1 | 0.03 |
| gt4py | EXCLUDED (toolchain) | | | | |

**Key signal:** NO candidate spills local memory on column. The architectural failure mode that capped the previous wrf_gpu attempt (register spilling → 8–12% occupancy) does not materialize on the M1 analytic surrogate for any candidate. Real M5 physics is the still-open question.

## ADR-001 (in this milestone)

Decision: **Selected backend: jax** (pending explicit user approval).
File: `.agent/decisions/ADR-001-backend-selection.md` (~15 KB; all required tokens; structural test passes).
Cross-model challenge: Codex `gpt-5.5 xhigh` critical-review at `REVIEW-codex-ADR-001/critical-review.md` + manager response pointer at `REVIEW-codex-ADR-001.md`. 11 findings across critical-review + binding-reviewer all applied; no manager counter-dissent.
Fallback: per-scheme gated Triton mini-ADR if M5 stop/go gate trips.

## Residual Risks

- **M5 first real physics is the decisive open test.** M2 evidence is necessary, not sufficient. Thompson microphysics / MYNN PBL have shapes the analytic surrogate did not exercise. M5 stop/go gate is binding.
- **Profiler counters blocked** (`ERR_NVGPUCTRPERM`). All M2 metrics are fallback-derived. M3 follow-up action: obtain `nvidia-driver-perfmon-allow=1` via system admin before any M4+ performance claim publishes.
- **GT4Py remains "blocked, not disproven."** If JAX+Triton inadequate at M5, a Python-3.12-venv remediation scout could resurrect it. Deferred.
- **Wall-time data is noise-dominated** at M1 fixture sizes. Throughput discrimination only meaningful at M4+ kernel sizes.

## Top Three Lessons

1. **Cross-AI testing pattern is paying for itself.** Claude Opus testers caught real defects across M2 that codex same-AI testing would have missed: M2-S2 file-size proof command malformation, M2-S6 cubin-cache contamination (would have polluted ADR-001 with wrong Triton numbers), M2-S5 JAX HLO+thunk+cubin honesty verification (corroborated the eye-popping numbers were real). Cross-AI testing is now established as a project pattern.
2. **`git worktree add` is the proper isolation pattern.** M2-S6 reviewer (codex) discovered this independently — solved the manager-during-worker contamination problem. Dispatch_role.sh should adopt going forward (deferred refactor).
3. **Cross-model challenge of architecture decisions catches genuine overreach.** Codex critical-review of ADR-001 caught 5 substantive issues in the manager's first draft (irreversibility framing, GT4Py oracle gap, fallback scope too broad, profile-fidelity overclaiming, missing M5 stop/go gate). Without the gate, ADR-001 would have shipped with subtle architectural overreaches.

## Required Next Action — Explicit User Approval

Per the constitution (`PROJECT_CONSTITUTION.md:16`) and `.agent/rules/architecture-decision-policy.md:13`, **ADR-001 cannot be treated as locked until the user explicitly approves it.** The manager-autonomy directive of 2026-05-19 delegates operational and design decisions but does not silently amend the constitution.

The status report to the user accompanying this closeout solicits that approval. M3 implementation is dispatch-blocked until the user replies with **"approved"** (or pushes back with revisions).

## Recommended Next Milestone

**M3 — GPU State & Grid Skeleton** opens immediately on user approval. First milestone where real model-shape code lands:
- `GridSpec` with named, machine-readable fields (map projection, terrain provenance, vertical coords, halo width, BC metadata — AIFS-driven per `PROJECT_PLAN.md §11.6`)
- `State` object device-resident on GPU
- Dummy 1000-step timestep loop with **zero** host/device transfers (audited)
- ADR-002 state layout + halo abstraction stub
- Real `ncu` profiler artifacts (after `nvidia-driver-perfmon-allow=1` is set, M3 follow-up from ADR-001)

In JAX (per ADR-001).
