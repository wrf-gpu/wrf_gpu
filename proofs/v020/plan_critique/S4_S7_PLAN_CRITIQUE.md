# S4/S7 Production Contract — Adversarial Cross-Model Critique (Claude-MAX)

**Critic:** Claude (max effort), independent of codex who authored the contract.
**Date:** 2026-06-21. **Worktree:** `worker/claude/v020-plancrit` (off main / v0.19.x baseline).
**Mandate:** try to BREAK the S4/S7 contract before a scarce GPU window is committed.

## Documents reviewed (in full)
- Contract under review: `…/v020-fp32-proto/proofs/v020/fp32_proto/S4_S7_PRODUCTION_PLAN.md`
- Prototype: `…/fp32_proto/REPORT.md`, `fp32_column_proto.py`, `acceptance_bands.py`, `proof_results.json`
- `.agent/decisions/V0200-ROADMAP.md` (§8 binding), `…/FINAL_FP32_SPRINT_PLAN.md` (S1–S8/R1–R8/G0),
  `V0200-RUNNABILITY-CONSTRAINT.md`, `V0200-HOST-BUBBLE-DIAGNOSTIC.md`
- **Ground-truth read of the actual v0.19.x source** (`src/gpuwrf/…`) to test claims against code,
  not just against the synthetic prototype.

## Bottom line (verdict at the end, in full)

The strategy is **sound and unusually well-gated** — I could not break the central thesis
(perturbation-authoritative fp32 + register-resident fp64 islands is the right and only credible
route to DRAM-scale speed + the 1 km unlock). But two issues must be fixed **before** the GPU
window is committed, because they touch (B1) the BINDING `fp64_default`-bit-identical / runnability
constraint and (B2) the very G0 decision that authorizes the GPU spend. **Verdict: REVISE (fix
B1 + B2 first), then GO.** Both fixes are narrow and cheap.

---

# BLOCKERS (fix before committing the GPU window / before execution)

## B1 — "Remove the hot-carry total aliases" is asserted bit-identical; the code proves it is *not* free

**Where:** Contract Prereq #2 ("S1 … total-alias removal lands first … hot path must no longer
require `p_total/ph_total/mu_total`"); `FINAL_FP32_SPRINT_PLAN.md` S1 ("remove the hot-carry total
aliases … **pure liveness reduction**"). This S1 is a hard prerequisite that lands **first**, on the
runnable path, and the runnability constraint (`V0200-RUNNABILITY-CONSTRAINT.md` §2) makes
`fp64_default` **BIT-IDENTICAL** binding.

**Why it breaks:** the current hot code does not just *carry* totals — it **reads stored totals
directly** and reconstructs base by differencing:
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py:166` `mub = (state.mu_total - state.mu_perturbation)`
- `:168` `alt = _inverse_density_from_theta_pressure(state.theta, state.p_total …)` ← reads `state.p_total`
- `:176` `phb = (state.ph_total - state.ph_perturbation)`
- `:195` `muts = state.mu_total`, `:209` `muts = mu_total + mu_pert`

If S1 removes `p_total` from the carry, line 168 must instead form `p_total = pb + p_perturbation`.
In IEEE FP, `pb + (p_total − pb)` is **not bitwise equal** to the stored `p_total` unless the stored
total was, byte-for-byte, produced by that exact addition with that exact rounding. The same applies
to every `muts`/`phb` site above. So "remove + reconstruct" silently perturbs `fp64_default` in the
last ULP → it **propagates** through alt → PGF → w-solve and is no longer bit-identical. Calling S1
"pure liveness reduction" is the trap: liveness is invariant, the *bits* are not.

**Concrete fix (cheap — the base leaves already exist):**
1. The base state is already carried as `BaseState{pb, phb, mub}` (`src/gpuwrf/contracts/state.py`),
   so hot functions should read `base.mub`/`base.phb`/`base.pb` **directly** instead of
   `mu_total − mu_perturbation`. That removes the *differencing*, which is the actual goal, without
   removing a stored total.
2. **Scope the carry-alias *removal* to mixed mode only.** In `fp64_default`, keep the stored totals
   exactly as today (so the bytes are untouched and bit-identity is trivially preserved); only the
   opt-in fp32 path drops the totals and reconstructs from explicit base + perturbation.
3. Make S1's gate an **empirical** bit-identity diff (`fp64_default` wrfout byte-equal pre/post S1 on
   the nightly config), not the assertion "pure liveness reduction." The contract's Gate-1 precision
   matrix test checks dtypes, not bytes — add the byte diff.

Until this is restructured, S1 cannot be declared a precondition that "lands first" on the runnable
path without risking the binding constraint.

---

## B2 — The G0 GO/KILL criteria can mis-commit the scarce GPU window (three independent gaps)

The brief's core question is "does the plan smuggle funding … or re-derive the old 1.1× trap?" The
gating against *smuggling* is excellent (Non-Goals "No source edit before G0-GO"; Gate 0; Prereq #1).
But the **content** of the G0 criterion has three gaps that can produce a *wrong* GO **or** a wrong
KILL — either of which wastes the window the brief is trying to protect.

**B2a — G0 measures a faster precision config than production will ship.**
`FINAL_FP32_SPRINT_PLAN.md` §2 G0 probe explicitly "downcasts p′/ph′/mu′/w storage **AND drops the
fp64 PCR solve to fp32**." But the production contract keeps the solve fp64: "Keep … Thomas/PCR solve
fp64-local first" and "the <=5 cancellation brackets **plus the tridiagonal solve stay fp64** in
registers." So G0's speed includes an fp32 tridiagonal that **production will not use**. A Speed-GO
read off the G0 number therefore **overstates** the speed production can achieve. *Fix:* G0 must report
the A/B **both** with fp32 solve (aggressive ceiling) **and** with fp64-in-register solve (the
production config); the GO decision uses the production-config number, the fp32-solve number is
context only.

**B2b — G0 measures fp32 storage on the UNFUSED topology, so its speed is a lower bound that excludes
the S7 multiplier → risk of a FALSE KILL.** G0 is "~1 GPU session" and cannot build the fused
megakernel. The whole roadmap thesis is that the speed comes from S4 (storage) **∧** S7 (register-
resident islands, launch collapse) — neither alone. Yet the KILL criterion is "ms/step ≤ ~1.2× even
with p′/ph′/mu′/w fp32" measured on today's *unfused* kernels with the fp64 islands still
**HBM-resident** (see SF1: `alt` is a full-grid fp64 field today). That ≤1.2× can be *exactly the
unfused-island cap that S7 is designed to remove* — killing the speed path for the wrong reason.
*Fix:* the Speed-KILL must distinguish two cases via the profiler, not the wall ratio alone:
(i) **real kill** — fp32 didn't shrink the transient / dominant kernels' fp32 and fp64 wall are
identical (the GPT FP32-S2 "1.106× with identical kernel wall" signature, `FINAL` §0); vs
(ii) **not a kill** — fp32 *did* shrink storage/transient but wall is capped by launch count +
HBM-resident fp64 islands, i.e. the cost is in precisely what S7 attacks. Case (ii) is a GO-to-S7,
not a KILL.

**B2c — the "bottleneck moved to a named component" GO-escape can re-enter the 1.1× trap.**
Roadmap §8.4 G0-Speed-GO allows GO if "profiler shows the bottleneck moved to a named non-dycore
component." If that component (qke/MYNN, microphysics, radiation, output) is **not** in S4/S7 scope,
then funding S4/S7 leaves the new bottleneck untouched and end-to-end lands back at ~1.1–1.2× — the
exact trap, re-derived. *Fix:* the "bottleneck-moved" escape is a valid GO **only if** the named new
bottleneck is itself addressed by the funded workstream (S4/S7) or by a *named, scheduled* follow-on;
otherwise it is a **redirect** (to P3/P4/multi-GPU per §8.6), not a Speed-GO.

These three are individually small wording fixes, but together they are the difference between
spending the GPU window on a decision that is actually falsifiable and spending it on one that can
rationalize either answer. Fix before the probe runs.

---

# SHOULD-FIX (material; fix during S4/S7, not necessarily before G0)

## SF1 — S4's standalone HBM/speed win is capped by the EOS island until S7; say so, and protect `alt`/`al′`

The EOS output `alt` (full inverse density) is computed nonlinearly and returned as a **full-grid
fp64 3-D array today** (`rk_addtend_dry.py:159,167`, consumed by PGF + buoyancy + w-solve). The EOS
is a genuine power law (`acoustic_wrf.py:172` `(p/P0)**CVPM`, `:185` `argument**CPOVCV`) and the
default hypsometric_opt=2 `al` is a **log-difference** (`rk_addtend_dry.py:198-207`,
`al = logα(total) − logα(base)`), already delicate at ~1.7e-6 rel in fp64 (the comment at :191-194).

Consequences the contract under-states:
- The "grad(base)_fp64 + grad(perturbation)_fp32" framing is **exactly right for the linear PGF split
  (p = PB + p′)** but **does not apply to the EOS** — you cannot split `(PB+p′)^γ` into base + pert
  gradients. The contract's actual mitigation (keep the EOS/log-α bracket fp64-local, reconstruct
  `p_total = PB+p′` inside the bracket in fp64) is correct; just make explicit that the EOS is handled
  by **bracket-localization, not by gradient-splitting**, so no implementer tries to fp32 it.
- Because `alt`/`al′` is full-grid and consumed across the (currently separate) operators, it stays
  **HBM-resident fp64 until S7 fuses the operators**. Therefore S4 *alone* reduces **VRAM** (p/ph/mu/w
  storage halves) but the **HBM-bandwidth / wall** win is capped until S7 register-izes the island.
  The contract should state that S4's standalone Gate-5 island matrix will show `alt`/`al′`/`php`
  resident, and that this is expected (removed only by S7), so S4 is not mis-scored as "fp32 didn't
  help." This also reinforces B2b.
- **Add `alt` and `al′` (and `php`) to an explicit "never fp32-stored" derived-field list.** The
  contract protects p′/ph′/mu′/w storage and the brackets, but the most cancellation-prone *derived*
  field (inverse-density perturbation `al′ = alt − alb`, two O(1) quantities differenced to O(1e-3))
  is not named as storage-protected. Naming it prevents a later "store alt fp32" optimization from
  silently reintroducing the cancellation the whole plan exists to avoid.

## SF2 — The acceptance bands are *proven insufficient by your own proof* to catch the cancellation pathology; keep the savepoint cancellation gate as the hard veto

The proof artifacts contain a direct demonstration that the skill bands cannot tell a numerically
broken path from a good one: the **naive fp32-total** form fails the local-cancellation gate 4/4
(34×–9292× worse, `proof_results.json` summary) **yet passes every acceptance band** —
`"naive_cases_passing_bands": 4` (and `REPORT.md` "Honest Limits": "the naive total-field path fails
the local cancellation gate strongly, but … still stays inside the relaxed forecast-style W/P/MU
bands"). It was the **27× local-cancellation ratio gate**, not the bands, that caught the broken form.

Implication: the contract makes `acceptance_bands.py` the imported authoritative gate (Prereq #5,
Gate 4) — but bands alone would have **green-lit the broken naive path**. The contract *does* include
the per-operator savepoint cancellation gate (Gate 2: "naive fp32-total baseline reproduces local
cancellation failure … perturbation path does not"). *Fix:* state explicitly that the **Gate-2
savepoint local-cancellation test is the primary numerical veto and is NOT redundant with the skill
bands** — so no future cleanup drops it as "covered by Gate 4." Without it, a damped run can carry a
cancellation-corrupt field inside bands for a long time before it surfaces.

## SF3 — Acceptance-band table conflicts with the binding §8.5 "relative-to-existing-gap" rule

`acceptance_bands.py` encodes **absolute** bands (e.g. wind `rmse_increase` 0.25/0.50/0.80). The
canonical, binding acceptance contract (roadmap §8.5) requires bands "expressed RELATIVE to the
fp64-GPU-vs-CPU-WRF gap where available (fp32 must not add a second port-sized error)" and a "no
material ACC/neighborhood skill drop" test. The contract imports the **absolute** table as
authoritative (Prereq #5), which conflicts with the canonical spec. *Fix:* reconcile — either
re-express the bands relative to the measured fp64-GPU-vs-CPU gap, or have the manager's frozen
`proofs/v020/ACCEPTANCE_TABLE.md` (promised in §8.5) supersede `acceptance_bands.py` and demote the
latter to "synthetic-proof default only." As written, two "authoritative" tables disagree.

## SF4 — XLA-first fusion of the *tridiagonal solve* is unlikely; the contract's own pass-criterion kernel may not even be the default hot kernel

Two code facts complicate S7b's "XLA fusion first":
- The acoustic substep loop is a **Python `for` loop** that unrolls (`acoustic.py:1015-1018`), and the
  w-solve is **inside** each substep — good, XLA sees one big graph.
- But the tridiagonal solve is a **hand-written `lax.scan` Thomas sweep** (`advance_w.py:533-567`,
  forward/back), default `unroll=1`. XLA does **not** fuse arbitrary elementwise work *across a
  `lax.scan` boundary* into a single kernel — a scan lowers to a sequential loop region. So "fuse the
  acoustic substep + vertical solve into one kernel with the tridiagonal in registers" is essentially
  a **Pallas column-kernel** description, not something XLA will synthesize from a scan. The realistic
  XLA-first win is launch-count/convert reduction on the **non-solve** parts; the solve→registers step
  almost certainly needs Pallas.
- Separately, the contract's XLA-pass criterion targets `pcrGtsvBatchSharedMemKernel*`
  (a cuSPARSE-style **batched PCR** kernel), but the code default is the `lax.scan` Thomas. *Either*
  the profiled build uses a different solve path (e.g. `jax.lax.linalg.tridiagonal_solve` →
  cuSPARSE) *or* the named kernel isn't the actual default hot kernel. **Reconcile in the S7 nsys
  baseline before targeting a kernel by name.**

*Fix:* time-box the XLA-first attempt with an explicit prediction ("the `lax.scan` Thomas will not
collapse into the acoustic kernel under XLA; expect launch-count reduction on the elementwise graph
only") so the team doesn't burn a sprint discovering it, and so the Pallas ADR is triggered on the
*predicted* failure, not treated as a surprise. The contract already has the Pallas governance path —
just set the expectation correctly.

## SF5 — No explicit register-spill / occupancy probe for the named risk

The brief asks directly about register pressure for "≤5 in-register fp64 brackets + the tridiagonal —
spills?" A full column (nz≈45–60) of fp64 in registers/shared for the Thomas solve plus 5 fp64
brackets is genuinely tight on Blackwell (≤255 regs/thread; fp64 = 2 regs/value). The contract's
gates measure launch count and "no compile-cache/RSS blowup" — but **a register spill shows up as
local-memory traffic and low achieved occupancy, not as RSS**. *Fix:* add an explicit
`ncu`/`nsys` probe to the S7 gates: registers-per-thread, achieved occupancy, and local-memory
(spill) load/store bytes for the fused/Pallas kernel. Without it, S7 can "pass" launch-count while
silently spilling and losing the win.

## SF6 — Restart/output must persist perturbations natively, not round-trip through fp32 totals

The contract reconstructs totals at named exits (wrfout/restart) — additive `PB+p′`, which is safe.
The dangerous direction is the **return trip**: standard WRF restart/wrfout stores **fp32 totals**;
if a mixed-mode restart writes `p_total` (fp32) and a later continue recovers `p′ = p_total(fp32) −
PB(fp64)`, you reintroduce **exactly** the cancellation the scheme avoids, at restart. Gate 2 lists
"restart/write/read reconstruction consistency for totals" but frames it as *total* consistency, not
*perturbation-recovery precision*. *Fix:* require mixed-mode restart to **persist fp32 `p′/ph′/mu′`
+ fp64 base directly** (or fp64 totals), and add a restart-continue test that checks the *recovered
perturbation* against pre-restart, not just the total. This is also implied by roadmap §8.5
"restart-write→restart-continue equivalence" — make it explicit for the perturbation leaves.

## SF7 — Water-budget conservation is in §8.5 but missing from the contract's gates

Roadmap §8.5 requires "dry-mass + **water budgets** + cumulative monotonicity." The contract gates
dry-mass drift (bands) and cumulative monotonicity (compensated sums), and carves QVAPOR out of
*parity* — but a parity carve-out is **not** a conservation exemption. There is no explicit total-water
budget check in Gates 2–5. *Fix:* add a column/domain total-water conservation check to Gate 4/5
(QVAPOR may drift from the fp64 path in *distribution* but the *budget* must still close to a stated
tolerance).

---

# NICE (low priority / hygiene)

- **N1 — Stability ladder is cadence-dependent.** The S4/S7 ladder (Gate 3) assumes the current
  dt/n_sound. The roadmap pairs S7 with P4 (dt/n_sound changes) which alter step count *and* fp32
  drift. If P4 lands on the integration branch first, the S4/S7 ladder must be re-run under the new
  cadence. Add a one-line dependency note.
- **N2 — S7a bit-identity must be proven on the nightly config.** S7a (async output / double-buffer)
  is "likely bit-identical." The proof should be on the **exact** nightly all-7 config the corpus
  runs, not a reduced test, since that's the path the runnability constraint protects.
- **N3 — Name the host-bubble baseline freshness risk.** S7a's 1.2–1.3× is derived from the
  2026-06-20 diagnostic (79% util). The v0.19.2 compile-cache fix and any allocator change shift that
  baseline; S7a's gate should replay the diagnostic on the *current* commit, which the contract's
  Gate-5 "live host-bubble replay" does cover — just tie S7a's target number to the replay, not the
  stale 21%.
- **N4 — `_safe_pressure`/`_safe_alt`/`jnp.maximum(...,1e-12)` clamps already exist** in the EOS
  (`acoustic_wrf.py:158,172,184-185`). The contract's "no new clamps/masks/finite guards" Non-Goal is
  good, but the **existing** limiter hit-counts must be reported (per §8.5 "report existing limiter
  hit-counts") because fp32 inputs will push these clamps more often; a rising clamp-hit count is a
  hidden fp32 failure mode that no skill band will show.

---

# What I tried to break and could not (the plan's genuine strengths)

I attacked the core thesis and the gate structure; these held up:

1. **The perturbation form is mathematically sound for the linear operators**, and the code already
   carries `p_perturbation/ph_perturbation/mu_perturbation` + a separate `BaseState{pb,phb,mub}`
   (`contracts/state.py`), so S4 is a *re-plumbing to explicit base leaves*, not a greenfield rewrite.
   The PGF split `p = PB + p′` is exactly linear ⇒ `∂p = ∂PB + ∂p′` splits with no approximation; the
   100×–9000× cancellation-error reduction (`proof_results.json`) is real and physically grounded
   (perturbations are 10²–10⁴× smaller than totals).
2. **The EOS nonlinearity does *not* sink the plan** — the contract handles it by bracket-localization
   (keep EOS/log-α fp64-local, reconstruct totals inside the bracket), which is the correct technique.
   My SF1 only asks that this be stated explicitly and that `alt`/`al′` be storage-protected.
3. **No funding is smuggled before G0.** Non-Goals + Gate 0 + Prereq #1 are airtight; the CPU
   prototype is correctly CPU-only and touches no `src/`.
4. **Reversibility is genuine and per-field** — the precision matrix is already field-granular
   (`contracts/precision.py`), the `force_fp64` enforcement path exists
   (`operational_mode.py`), and the Rollback section is thorough (every demotion independently
   revertible). A failing field reverts to fp64-in-bracket without unwinding the others.
5. **The measure-first / three-outcome G0 (speed vs capability vs kill)** is the right shape and
   directly answers the user's "or find the provable reason" — *provided* B2 sharpens the criteria.
6. **Runnability/additivity is well-designed** — worktree isolation, additive opt-in modes,
   `fp64_default` bit-identical (modulo B1), nested all-7 as the always-green benchmark. The one real
   crack is B1 (alias removal), which is narrow and cheap to fix.
7. **The honest-negative discipline holds** — every gate has a "STOP and write the exact cause"
   branch (Gate B corollary, XLA-first stop-before-Pallas), so a failure produces a falsifiable
   proof, not a vague "fp32 didn't help."

---

# VERDICT

**REVISE — fix B1 and B2 first; then GO (execution-ready post-G0-GO).**

The contract is strategically correct, well-gated, and honest about its own limits; the prototype is
a legitimate (if narrow) de-risk of the *linear* cancellation mechanism only. I could not break the
core thesis or the anti-smuggling structure. But:

- **B1** (alias-removal bit-identity) directly tensions the BINDING `fp64_default`-bit-identical /
  v0.19.x-runnability constraint and lands *first* on the runnable path — fix the S1 framing
  (read existing base leaves; scope removal to mixed mode; prove byte-identity) before S1 ships.
- **B2** (G0 criterion gaps a/b/c) governs whether the scarce GPU window is spent on a *falsifiable*
  decision — sharpen the three points before the G0 probe runs.

Both are narrow wording/scoping fixes, not redesigns. With them addressed, and the SHOULD-FIX items
folded into the S4/S7 sprint gates, this contract is ready to execute the moment G0 records a GO.

The single most important non-blocking insight, because it recurs in B2b/SF1/SF2: **S4 and S7 deliver
the speed *jointly* — S4 halves storage/VRAM, but the wall-clock win needs S7 to pull the fp64 islands
(`alt`, log-α, tridiagonal) off HBM into registers.** Do not score S4 alone as "fp32 didn't help," and
do not let G0 KILL the speed path on an unfused, island-resident measurement.
