# V0.14 FP32 Acoustic Roadmap

Date: 2026-06-08
Owner: manager, from GPT-5.5 xhigh feasibility refresh
Primary evidence: `.agent/reviews/2026-06-08-gpt-fp32-acoustic-refresh.md`

Update 2026-06-08 22:45 WEST: the three de-risk workers are complete and mirrored into
this branch:

- `.agent/reviews/2026-06-08-gpt-fp32-roi-and-v013-decision.md` (`a945107a`) says ship
  v0.13 after fp64 TOST; do not hold the tag for FP32 acoustic.
- `.agent/reviews/2026-06-08-gpt-fp32-probes.md` + `proofs/v014/fp32_acoustic_probes.*`
  (`a1357aee`) show the absolute-total fp32 cancellation mechanism and perturbation-form
  rescue on CPU-only probes.
- `.agent/reviews/2026-06-08-gpt-fp32-r0r1.md` +
  `.agent/decisions/ADR-031-mixed-perturb-fp32-acoustic-DRAFT.md` (`014fb7aa`) define the
  R0/R1 contract and static audit. Its default-inert source scaffold is **not merged into
  v0.13**.

Update 2026-06-09 10:40 WEST: FP32 acoustic remains v0.14 P1, but it must not
preempt the active fp64 grid-divergence root cause. The latest direct grid proof
(`proofs/v014/grid_after_live_nest_base.json`) is
`GRID_SYMPTOM_NOT_CLOSED`, and same-state localization
(`proofs/v014/same_state_momentum_mass.json`) shows `U` already mismatching at
`post_after_all_rk_steps_pre_halo`. The next action is dynamic root-cause
localization/fix on fp64; mixed FP32 resumes after the failing operator is named
or bounded.

Update 2026-06-09 ~24:00 WEST (Mythos memory/FP32 lane, branch
`worker/mythos/v014-memory-fp32`, manager to review/merge):

- **R0 LANDED, default-inert**: `AcousticPrecisionMode` labels in
  `src/gpuwrf/contracts/precision.py`, `OperationalNamelist.
  acoustic_precision_mode` (default `fp64_default`) riding in static aux for a
  separate JIT/cache variant, fail-closed on unknown mode strings, and the
  5-test cache-key suite (`tests/test_operational_namelist_cache_key.py`)
  green. The static audit regenerated on this exact branch
  (`proofs/v014/fp32_acoustic_static_audit.json`): 25 base-reconstruction-from-
  totals lines, 60 hard fp64 casts in scope, **0 timestep consumers** of the
  mode — the default fp64 program is untouched.
- **R1/R2 EXACT BLOCKER (do not start yet)**: the strict Step-1 one-RK-step
  comparison still shows REAL fp64 dynamics divergence concentrated in the
  P/PH/MU (acoustic/mass/vertical) lane — WRF-EOS pressure rederived from the
  JAX post-step state sits at p95 ≈ 770 Pa vs ≈ 0.02 Pa for WRF's own state
  (`proofs/v014/mythos_kernel_fix_260609.json`), with one-step namelist parity
  (acoustic substep count, epssm, damping) not yet frozen. R1's explicit
  base-state plumbing touches exactly those acoustic prep/finish/staging
  files: editing them now would unfreeze the active root-cause lane's fault
  surface, and any mixed-mode validation delta would be confounded by the
  open fp64 defect (no WRF-anchored acceptance gate can pass). This is a
  correctness-gate blocker, not a feasibility refutation — the CPU probes
  (`proofs/v014/fp32_acoustic_probes.json`, perturbation-form rescue ratio
  ~1.4e6) still support the mixed-perturbation design.
- **Minimal remaining roadmap** (unchanged in substance, now precisely gated):
  (1) freeze one-step namelist parity and close the RK1 substage divergence on
  fp64 (the active grid-parity lane); (2) then R1 explicit base-state plumbing
  with fp64-default bit-identity tests over acoustic prep/finish plus a
  one-step operational carry test; (3) R2 perturbation-authoritative loop
  behind `mixed_perturb_fp32`; (4) R3-R8 as written below. ADR-031 stays
  DRAFT pending manager review.

## Priority

Current manager decision, updated 2026-06-09: this is the highest-priority
v0.14 memory/performance lane after the direct grid-cell CPU-WRF-vs-GPU
divergence root cause is named and fixed or formally bounded.

It is no longer a v0.13 pull-in candidate. The old v0.13 tag plan was
superseded by the grid-parity-first directive: do not land FP32 acoustic source
changes while the fp64 production path still has unresolved broad grid-field
divergence.

## Decision

Stable fp32 acoustics are not mathematically impossible on GPU/JAX. The current global/naive
fp32 mode remains unsafe, but an opt-in mixed precision acoustic path is feasible in principle
if the acoustic state is made perturbation-authoritative and the fp64 islands are deliberately
kept where the numerics need them.

This is not part of the current fp64 production line. The de-risk evidence
upgrades FP32 acoustic to v0.14 P1, but the immediate gate is same-state
grid-cell/root-operator localization. TOST is paused until the grid-field
envelope is credible; mixed precision gets its own validation lane after that.

## Scope Boundary

Allowed in v0.14:

- an explicit experimental mode, for example `acoustic_precision_mode = "mixed_perturb_fp32"`
- fp64-default bit identity for every existing v0.13 production path
- measured VRAM/performance claims only after profiler proof objects exist

Rejected:

- a global fp32 dtype flip
- widening tolerances after the fact
- calling mixed precision equivalent by JAX-vs-JAX self-comparison only
- changing v0.13 release docs into an fp32 claim

## Roadmap

1. R0 ADR and precision-mode contract:
   define the opt-in mode, cache-key behavior, default-off policy, report labels, and kill gates.
2. R1 explicit base-state plumbing:
   thread `BaseState` through acoustic prep/finish, pressure diagnostics, boundary staging, and
   restart/init assembly without recovering base fields from fp32 total-minus-perturbation in loop.
3. R2 perturbation-authoritative acoustic state:
   make `p'`, `ph'`, `mu'`, WRF work arrays, and pressure memory authoritative in the acoustic
   loop; reconstruct absolute totals only at controlled interfaces.
4. R3 CPU oracle and analytic gates:
   scalar cancellation probes, one-column acoustic recurrence, WRF savepoint parity for the
   small-step core, flat-rest and terrain-rest budgets.
5. R4 idealized and boundary-coupled dry gates:
   warm bubble, Straka, terrain rest, lateral-boundary dry case, restart roundtrip, and
   fake-mesh partition checks.
6. R5 current-module integration:
   GWD, nesting/two-way feedback, RRTMG SW/LW tiling and clear-sky diagnostics, MYNN/MYJ/Janjic,
   moisture advection, operational physics smoke, restart, and wrfout writer.
7. R6 staged real-GPU campaign:
   1-step dry smoke, 1h L2 dry/limited physics, 6h L2 full physics, 6h L3, then bounded
   nested/GWD/feedback gates with transfer audit and VRAM records.
8. R7 fp64 island demotion:
   demote implicit-w coefficients, implicit-w solve, `calc_p_rho` bracket, terrain PGF, and EOS
   refresh one at a time, each with its own before/after proof.
9. R8 validation and documentation:
   run mixed-mode TOST/AEMET/CPU-WRF validation with predeclared single-precision margins and
   publish mode-labeled docs only after gates pass.

## Compatibility Notes

- Dycore acoustic is directly touched and needs the strongest gates.
- Nesting/two-way feedback and `ph'` boundary forcing are high-risk; keep boundary/reference
  leaves fp64 initially.
- RRTMG memory remains orthogonal: mixed acoustic does not replace column tiling.
- PBL/surface and `qke` have fp64-sensitive history; do not globally demote turbulence.
- TOST for v0.13 remains the fp64 production campaign. Mixed mode needs a separate v0.14 lane.

## First Sprint Recommendation

Start with R0+R1 only after the same-state/grid-divergence sprint has named the
first failing operator, or after it proves acoustic `p/ph/mu/w` coupling is not
on the critical fault surface. The first source sprint should be ADR +
explicit base-state plumbing with fp64-default bit identity over focused
acoustic prep/finish plus a one-step operational carry test. The worker R0
scaffold may be reviewed as a starting patch, but it should not be merged into
the active fp64 grid-debug line until the root-cause surface is clear.
