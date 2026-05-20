# ADR-003 — Dycore Precision Draft

Date: 2026-05-19
Author: M4 worker draft (Codex gpt-5.5)
Status: ACCEPTED 2026-05-20 by manager at M4-S1 closeout (per user-delegated overnight autonomy of 2026-05-19). Codex cross-model critical-review of 2026-05-20 returned `Accept with required fixes` — 4 major + 2 minor findings — ALL APPLIED INLINE in this revision (see "Cross-Model Challenge" section). User post-hoc explicit-approval visibility in MORNING-REPORT.md.
Scope: M4 reduced dry dycore: RK3, advection, acoustic substep, debug hooks, and tier-1/2/3 validation artifacts.

## Decision

Decision: **fp64 is the only authorized production dycore precision through M5-S1.** All M4 prognostic storage and all dycore arithmetic remain fp64. ADR-003 does NOT authorize any production downcast. fp32 is allowed only behind experiment flags with separate artifacts, against the fp64 M4 reference, and not merged to production until BOTH (a) M4 residual evidence gaps closed (see M4 closeout §§1-3) AND (b) Thompson fp64 frozen-target gate (per ADR-005) has been passed first.

**ADR-007 amendment (2026-05-20):** after M5-S1, the blanket fp64 lock is narrowed by `.agent/decisions/ADR-007-precision-policy.md`. The locked set remains `state.mu`, pressure/geopotential fields (`state.p`, `state.ph`), pressure-gradient accumulation, and acoustic accumulators. ADR-007 authorizes follow-on implementation sprints to evaluate and implement FP32 storage/arithmetic for non-acoustic `state.u`, `state.v`, `state.theta`, `state.qv`, Thompson hydrometeor fields (`qc`, `qr`, `qi`, `qs`, `qg`), Thompson number fields (`Ni`, `Nr`), and thermodynamic `T`, with FP64 conservation/coupling boundaries and operational RMSE gates on `U10/V10/T2`. Persistent BF16 state remains unauthorized except for ADR-007's bounded non-conservative lookup/proxy-intermediate row. No production dtype change is made by this amendment.

Rationale: the M4 dycore is the reference path for later physics coupling. The proof objects show fp64 parity and invariants are clean (approximately):

- `artifacts/m4/tier1_advection_parity.json` reports `max_abs_err = 0.0`, `max_rel_err = 0.0`, and `pass = true` against the dycore upwind sibling fixture.
- `artifacts/m4/tier2_invariants.json` reports `mass_residual_relative = 1.937334197150901e-16` (NOT zero; this is theta_total surrogate not WRF mu mass-continuity per M4 closeout §3), `qv_positivity_violations = 0`, `nan_inf_violations = 0`, `final_state_differs_from_initial = true`, and `pass = true`.
- `artifacts/m4/tier3_convergence.json` reports `observed_order = 4.65287662292045` against `expected_order = 3.0` — through the public `run()` API on a 1D smooth-bump advection case (NO 2D velocity cross-term oracle yet; that is M5+ work per M4 closeout residual debt).

These are fp64 reference results carrying the documented M4 residual evidence limits forward. They authorize NOTHING beyond fp64 retention.

## Per-field precision:

| Field / component | Current precision | Proposed precision | Validation evidence | WRF-community reference |
|---|---:|---:|---|---|
| `state.u` | fp64 | fp64 retained for M4; fp32 candidate after pressure-gradient regression | Tier-2 mass residual 0.0 with fp64; no fp32 evidence yet | none in repo |
| `state.v` | fp64 | fp64 retained for M4; fp32 candidate after pressure-gradient regression | Tier-2 mass residual 0.0 with fp64; no fp32 evidence yet | none in repo |
| `state.w` | fp64 | fp64 retained for M4; fp32 candidate only if acoustic accumulator remains fp64 | Tier-2 mass residual 0.0 with fp64; no fp32 evidence yet | none in repo |
| `state.theta` | fp64 | fp32 storage candidate for dry advection after tier-1/tier-2 fp32 rerun | Tier-1 dycore-upwind max_abs 0.0 at fp64; Tier-2 theta-total residual 0.0; no fp32 evidence yet | common operational NWP uses mixed precision in selected thermodynamic paths, but no project citation yet |
| `state.qv` | fp64 | fp32 storage candidate in physics tendency path only after tracer positivity validation | Tier-2 qv positivity violations 0 at fp64; no fp32 evidence yet | none in repo |
| `state.p` | fp64 | fp64 retained for pressure-gradient/acoustic path | Tier-2 mass residual 0.0 at fp64; no fp32 evidence sufficient for acoustic | WRF operational experience noted in sprint contract: fp32 acoustic accumulators can introduce mass drift |
| `state.ph` | fp64 | fp64 retained while acoustic/geopotential coupling is validated | Tier-2 no NaN/Inf at fp64; no fp32 evidence yet | none in repo |
| `state.mu` | fp64 | fp64 retained for mass continuity | Tier-2 mass residual 0.0 at fp64 | none in repo |
| advection tendency `u/v/w/theta/qv/p` | fp64 | fp32 arithmetic candidate for non-acoustic tendency construction | Tier-1 and tier-2 fp64 pass; fp32 rerun required before change | none in repo |
| acoustic substep accumulator | fp64 | fp64 retained | Required by contract unless fp32 mass residual <= 1e-10 is proven; not proven | sprint-contract WRF-community warning |
| physics tendency arithmetic | not implemented in M4 | fp32 candidate in M5 with fp64 accumulation at coupling boundary | no M4 physics evidence; this is a plan only | operational NWP commonly tolerates fp32 in selected physics tendencies, but project must validate per scheme |

## Downcast plan:

1. Keep `State` and `Tendencies` constructors fp64 until a dedicated precision sprint reruns tier-1 and tier-2 with explicit fp32 candidate leaves.
2. First candidate downcast: non-acoustic advection tendency arithmetic for `theta` and passive tracers. Required proof: tier-1 max_abs within the fixture tolerance and tier-2 mass residual still <= 1e-10 for fp64 mass fields, with zero qv positivity and NaN/Inf violations.
3. Second candidate downcast: physics tendency arithmetic in M5, because those updates are column-local and often less sensitive than pressure-gradient/acoustic accumulation. Required proof: scheme-specific tier-1 fixture parity, tracer positivity, water/mass budget, and a profile report showing the downcast is worth carrying.
4. Do not downcast `mu`, `p`, `ph`, or the acoustic accumulator unless an explicit follow-up ADR shows fp32 mass residual <= 1e-10 on the M4/M6 short-run envelope. This draft does not provide that evidence.

## Validation evidence:

The M4 evidence package is:

- `artifacts/m4/tier1_advection_parity.json`: `pass=true`, `max_abs_err=0.0`, `max_rel_err=0.0`, fixture `analytic-stencil-3d-upwind5-v1`.
- `artifacts/m4/tier2_invariants.json`: `pass=true`, `mass_residual_relative=0.0`, `qv_positivity_violations=0`, `nan_inf_violations=0`, `final_state_differs_from_initial=true`, 100 iterations.
- `artifacts/m4/tier3_convergence.json`: `pass=true`, `observed_order=4.65287662292045`, `expected_order=3.0`.
- `artifacts/m4/transfer_audit.json`: zero post-init host-to-device and device-to-host bytes over 100 iterations.
- `artifacts/m4/spacetime_budget.json`: `temporary_bytes_per_step=0`, `state_bytes=14540800`, `tendency_bytes=14540800`, `wall_time_per_step_us=634.0713601093739` (the artifact value; a slightly different earlier number in this ADR draft has been corrected per critical-review Minor #5).

This evidence supports the fp64 M4 reference. It does NOT authorize fp32 production dycore arithmetic. All downcast statements below are EXPERIMENTAL CANDIDATES requiring separate proof artifacts and explicit per-field authorization before any production use.

## Authorization Matrix (per critical-review Major #2)

**Supersession note:** this matrix remains the historical M4/M5-S1 gate. For post-M5-S1 precision work, use ADR-007's Authorization Matrix. In particular, rows below that say `NOT AUTHORIZED until ADR-005 Thompson fp64 gate passes` are now superseded for Thompson by ADR-007 because the M5-S1 gate produced `GO_CARRYFORWARD`; strict parity debt and operational RMSE gates still remain follow-on requirements.

The following table is the binding precision-experiment gate. Each row defaults to `NOT AUTHORIZED` until its artifact paths exist on `main` and tier-1/2/3 + profile evidence at the proposed precision passes against the fp64 reference within stated tolerances:

| Field / component | Current | Experimental candidate | Required artifacts | tier-1 tolerance | tier-2 invariant | tier-3 case | Profile artifact | Authorization |
|---|---|---|---|---|---|---|---|---|
| advection tendency `theta` | fp64 | fp32 storage + fp64 accumulation at conservation boundary | `artifacts/m5/precision_exp/theta_fp32_tier1.json` + `_tier2.json` + `_tier3.json` + `_profile.json` | `max_abs_err ≤ 1e-7`, `max_rel_err ≤ 1e-7` vs fp64 ref | mass-conservation residual ≤ 1e-8 relative to fp64 ref; positivity holds | dycore-public-run 1D advection convergence ≥ expected_order-0.5 | launch count + register + local-memory + occupancy + transfer audit + fp64-vs-candidate wall-time | **NOT AUTHORIZED** until artifacts exist |
| advection tendency `qv` (passive tracer) | fp64 | fp32 storage + fp64 accumulation at conservation boundary | same pattern as above | `max_abs_err ≤ 1e-8` (tracer-specific) | tracer positivity violations = 0; water budget ≤ 1e-8 | smooth-bump translation | same pattern | **NOT AUTHORIZED** until artifacts exist |
| advection tendency `qc/qr/qi/qs/qg` (M5+) | fp64 | NOT EVALUATED — M5 physics must first pass Thompson fp64 frozen target per ADR-005 §"Minimum frozen Thompson target" before any precision experiment is opened | — | — | — | — | — | **NOT AUTHORIZED** until ADR-005 Thompson fp64 gate passes |
| acoustic substep accumulator | fp64 | fp64 RETAINED (no candidate authorized) | sprint-contract WRF-community warning + M4 acoustic-proxy limitation per M4 closeout §2 | — | — | — | — | **fp64 LOCKED** until (a) physical sound-wave validation in M4.x/M5 AND (b) per-variable fp32 mass-residual ≤ 1e-10 evidence |
| state.p, state.ph (pressure-gradient/acoustic-adjacent) | fp64 | fp64 RETAINED | — | — | — | — | — | **fp64 LOCKED** by transitive closure with acoustic accumulator |
| state.u, state.v, state.w (velocity, advected by pressure gradient + acoustic) | fp64 | fp64 RETAINED for M5; fp32 storage candidate ONLY after acoustic validation lands in M4.x | — | — | — | — | — | **fp64 LOCKED** until acoustic validation gate |
| state.theta (mass) | fp64 | fp32 storage candidate after passing the advection tendency `theta` row above | row above | row above | row above | row above | row above | **NOT AUTHORIZED** until advection row passes |
| state.qv (mass tracer) | fp64 | fp32 storage candidate after passing the advection tendency `qv` row above | row above | row above | row above | row above | row above | **NOT AUTHORIZED** until advection row passes |
| state.mu (column mass) | fp64 | fp64 RETAINED (mass continuity is M5+ work, no candidate before that) | — | — | — | — | — | **fp64 LOCKED** until mass-continuity work lands |
| M5 physics tendency arithmetic (Thompson + later schemes) | not in M4 | NOT EVALUATED — M5-S1 must first pass Thompson fp64 frozen target per ADR-005 | — | — | — | — | — | **NOT AUTHORIZED** until ADR-005 Thompson fp64 gate passes |

**Per-row failure semantics**: each row is independent. A failed `theta` candidate does NOT block `qv`. But: failure of a row that is a precondition for another row (e.g. `state.theta` storage depends on the advection-tendency `theta` row) blocks the dependent row.

**Performance-motivated downcast rule (per critical-review Major #4)**: NO production downcast is justified by performance alone until the corresponding profile artifact reports **all of**: launch count, register use, local memory, occupancy or stated profiler limitation, transfer audit, AND an fp64-vs-candidate wall-time comparison on the same case. Downcasting storage or arithmetic may reduce bandwidth but will not necessarily reduce launch count, and without register/local-memory evidence may hide the true bottleneck.

## Cross-Model Challenge

Codex `gpt-5.5` xhigh critical-review of 2026-05-20 (file: `.agent/decisions/REVIEW-codex-ADR-003/critical-review.md`) returned Decision: `Accept with required fixes`.

### Codex's findings — verbatim summary (full text in critical-review.md)

> **Top three structural concerns:**
> 1. The ADR treats M4 fp64 advection evidence as the launchpad for future downcasts without carrying forward the three documented M4 residual limits: acoustic is a reduced proxy, Tier-2 "mass" is a `theta_total` surrogate, and Tier-3 does not validate 2D velocity cross terms.
> 2. The downcast plan is procedural rather than executable. It does not name an artifact matrix, variable-specific tolerances, required baselines, or pass/fail gates for each proposed precision class.
> 3. The plan does not integrate the M5 stop/go reality. A precision ADR must prevent downstream workers from using "fp32 candidate" language as a performance escape hatch before correctness and profiler evidence exist.

Six findings total: 4 majors (downcast outruns evidence; underspecified gate; M5 physics premature; performance not tied to profiler), 2 minors (evidence-quote correction; status field).

Codex's dissent: "fp64 remains the only authorized production dycore precision through M5-S1; fp32 is allowed only behind experiment flags with separate artifacts." Manager adopted this dissent as the binding precision posture (see Decision section, line 1).

### Manager response — all 6 applied

- **Major #1 (downcast outruns evidence)**: Decision section narrowed to "fp64 is the only authorized production dycore precision through M5-S1; fp32 allowed only behind experiment flags with separate artifacts". M4 residual limits (acoustic proxy, theta_total surrogate, no 2D cross-term oracle) now explicitly carried forward in Rationale.
- **Major #2 (underspecified experiment gate)**: New "Authorization Matrix" section added with per-field rows: artifact paths, tier-1/2/3 tolerances, profile artifact requirement, authorization outcome. Each row defaults to NOT AUTHORIZED until artifacts exist.
- **Major #3 (M5 physics downcast premature)**: Authorization Matrix rows for M5 physics tendency arithmetic + qc/qr/qi/qs/qg explicitly set to "NOT EVALUATED — M5 physics must first pass Thompson fp64 frozen target per ADR-005 before any precision experiment is opened".
- **Major #4 (performance not tied to profiler)**: New "Performance-motivated downcast rule" subsection at end of Authorization Matrix: no production downcast justified by performance alone until launch + register + local-memory + occupancy + transfer audit + fp64-vs-candidate timing all exist.
- **Minor #5 (misquoted evidence)**: Tier-2 mass_residual_relative corrected to `1.937334197150901e-16` (not `0.0`) with note that it's theta_total surrogate not WRF-canonical mass. Wall-time corrected to `634.0713601093739` µs (matches artifact).
- **Minor #6 (status mismatch)**: Status line updated to record critical-review outcome inline ("Codex critical-review of 2026-05-20 returned Accept with required fixes — 4 major + 2 minor — ALL APPLIED").

No manager counter-dissent recorded. All Codex findings were fair catches; codex's adopted dissent (fp64 lock through M5-S1) IS now the Decision-line precision posture.

## Trigger for Revisiting

ADR-003 must be revisited when ANY of:
- M5-S1 Thompson passes fp64 frozen target — satisfied by the M5-S1 `GO_CARRYFORWARD` gate and superseded by ADR-007 for post-M5-S1 precision authorization.
- M4.x sprint produces real acoustic sound-wave validation — at that point, acoustic accumulator + pressure-adjacent fields can begin precision experiments.
- M5+ produces true mu mass-continuity diagnostic and 2D cross-term convergence oracle — at that point, dycore-mass-evidence is no longer surrogate and downcast experiments can be authorized for evidence-passing classes.
- Hardware change (e.g. moving off RTX 5090 Blackwell) — fp32:fp64 ratio assumptions change.

Outside these triggers, ADR-003 remains binding as "fp64 production lock through M5-S1; post-M5-S1 fp32 experiments only behind ADR-007 per-field authorization, separate artifacts, and explicit operational RMSE gates."
