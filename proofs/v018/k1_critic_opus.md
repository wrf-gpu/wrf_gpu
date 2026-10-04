# v0.18 K1 fp32-stability plan — adversarial critique (Opus-MAX kernel critic)

Date: 2026-06-16
Role: Opus-MAX kernel critic (analysis only — NO GPU, NO code changes)
Critiques: `proofs/v018/k1_fp32_stability_scoping.md` (GPT, K1 frontrunner)
For: principal's morning go/no-go on launching the **K1 implementation sprint**

---

## VERDICT: APPROVE-WITH-CONDITIONS (near-REJECT — gate on one cheap probe, expect STOP)

The K1 *methodology* is sound and honest; the K1 *sprint as scoped* is not yet
launchable. GPT's plan correctly demolishes the "~4× proven" premise and proposes the
only valid technique (perturbation-authoritative mixed mode, fp64 cancellation islands).
But as written it is an **implement-first, measure-last** program: Phases 0–5 build a
multi-sprint perturbation-native *rewrite* of the acoustic core before Phase 7 finally
measures the large-grid warm speedup — **a number we have already measured three times
at ~1.0–1.1×.** Approve only a **~1-GPU-session decision probe** with a hard kill;
do **not** authorize the Phase 0–7 rewrite until that probe clears a >2×-trajectory bar.

**My prior, stated honestly: I expect the probe to FAIL and K1 to convert to an honest
negative** ("valid fp32 ceiling ~1.1× single-card; GPU value = large/DRAM-bound grids +
cluster"). I recommend APPROVE-WITH-CONDITIONS rather than flat REJECT only because the
*exact* clean config K1 proposes (alias-free, convert-free, totals reconstructed only at
interfaces, at a DRAM-bound grid) was never isolated — prior attempts carried the
`State.replace` p↔p_total alias-sync bug and XLA convert-scatter — and the probe that
closes that seam is cheaper to run than to argue about.

### The single cheapest decisive experiment (run this BEFORE any K1 code)
**A warm fp32-vs-fp64 A/B + HLO transient-memory decomposition at the largest DRAM-bound
grid that BOTH precisions fit (~130–147k cols, BouLac-O(nz) on), using the EXISTING
`worker/opus/v016-fp32-compact` / `worker/perf/v017-fp32-dycore` branches — no rewrite.**
Measure: (a) warm ms/step ratio; (b) peak VRAM incl. transient; (c) `jax` HLO
`memory_analysis` split of the transient peak into *fp32-able* (State leaves + RK
tendencies + non-cancellation work) vs *fp64-pinned* (EOS/PGF/advance_w cancellation
islands + dense stencil temps + qke). Cost: ~1 GPU session.
- **KILL (expected):** transient still fp64-island/qke-dominated **and** ms/step ≤ ~1.2× ⇒
  STOP. The clean rewrite cannot beat walls that survive at the graph level. Ship the
  honest negative.
- **CONDITIONAL GO:** transient is *majority* fp32-able but unrealized (alias/convert
  artifacts) **and** ms/step trends toward the bar ⇒ the Phase 0–7 rewrite is justified.

### Binding kill-gates (all must hold to keep K1 alive past each point)
1. **G0 (probe, above) is the FIRST binding gate — not HLO-liveness.** HLO liveness is
   necessary-not-sufficient (see Q2). No new K1 code before G0.
2. **Speed bar = the principal's keep-bar, not ~1.1×.** Optional fast-mode must trend to
   **>2× @ 24h-tolerance** (principal, memory L41/L264). 1.1–1.6× is an honest negative,
   reported as a stability experiment — *not* shipped as a speed lane (this is exactly how
   fp32-physics ~1.5× was already handled).
3. **24h stability must be proven at the LARGE/DRAM-bound grid where speed is claimed**,
   not only on tiny-nest Canary d02/d03 (see Q3).
4. **ADR-031 kill gates bind verbatim** (ADR-031 L76–89): default `fp64_default` stays
   bit-identical; no global fp32 flip; no in-loop fp32 total−perturbation; no
   tolerance-widening; no JAX-vs-JAX-only proof; no in-loop host transfer; no clamp/mask.
5. **Forbid the "~4× proven" framing** in all K1 reporting — it is stale and contradicted
   by the repo's own double-confirmed evidence (see Evidence base).

---

## Evidence base (what is ALREADY proven — K1 is re-entering closed ground)

| # | Finding | Source |
|---|---|---|
| E1 | Valid mixed/perturb fp32 = **~1.04–1.11×**, **0% VRAM relief**, OOM at the identical 18.80 GiB @147k — two independent agents, two strategies converge | make-or-break L210, L227–231 |
| E2 | Full-working-set fp32 = **IMPOSSIBILITY PROVEN, double-confirmed** (Opus + GPT blind reproduce 1.105/1.111×, vram 1.000) | make-or-break L223–231 |
| E3 | **Transient peak is precision-INSENSITIVE** — `memory_analysis` temp_size 5305→5379 MiB *unchanged* while persistent shrank; dominated by fp64 cancellation islands (EOS/PGF/advance_w) + qke-MYNN | make-or-break L228, L230 |
| E4 | `p_total/ph_total` **cannot** store fp32 — corrupts geopotential/PGF **27×–127×**, *at storage* (fp64 islands powerless) | make-or-break L229; hostgap §7.2 L426–430 |
| E5 | fp32-dycore **CLOSED: VALID-but-NO-SPEED-NO-VRAM**, double-confirmed. 16k 1.04× / 65k **0.996×** / **147k fp32 19.41 > fp64 18.80 GiB (VRAM WORSE — islands ADD buffers)** | make-or-break L331, L337–338 |
| E6 | nsys (all-7): top kernel 11%, **every kernel ~1.5 µs**, hundreds–thousands of instances → **occupancy/launch-bound; NO saturating kernel for fp32 to halve** | hostgap §7.1 L416–423 |
| E7 | **deep-why:** small-grid working set 11–74 MB is **L2-resident (96 MiB L2)** → DRAM-BW edge unused → fp32 can't help; GPU only wins when working set **>96 MiB (≥147k cols)** = DRAM-bound regime | make-or-break L324–327 |
| E8 | **bigswiss 211k fp64 = OOM** (single transient **21–22 GiB = State ~30 leaves + dycore work + RK tendencies**); resident only ~9 GiB. fp64 ceiling ~167–209k cols | bigswiss_gpu_benchmark.md L9–12, L19, L49–54 |
| E9 | bigswiss **core** factor **PROJECTED 4.35× vs CPU IF it fit** (fp64, core-only, *projection not measurement*) — the GPU's large-grid win is **fp64-real but VRAM-blocked** | bigswiss_gpu_benchmark.md L74–85 |

The chain E5+E8 is decisive against K1's headline upside: at 147k the *as-built* fp32 path
already OOMs **worse** than fp64, and 211k OOMs in fp64 on a transient (State + tendencies)
that fp32 was measured (E3) not to shrink. The one residual seam K1 targets is "did a
*clean* alias/convert-free graph fail to realize fp32 the prior branches left on the
table?" — that is exactly what the G0 probe settles without a rewrite.

---

## Critique by question

### Q1 — Is perturbation-authoritative mixed mode the right approach? Flaws / simpler alternative / missed risk?
**Technique: yes. Expectation and novelty: no.** Perturbation-native state + fp64
islands + total-alias removal + one-at-a-time demotion is the correct and *only* valid
approach; it matches ADR-031 and the v016 perturbation identity brief. GPT's "Premise
correction" (scoping L74–86) honestly relabels ~4.29× as an "upper-bound warning label" —
credit due.

The flaw is that **Rank-1/Rank-2 keep fp64 exactly where the bytes and the cancellation
are, and demote fp32 exactly where the evidence says it doesn't matter.** The transient
VRAM peak is the cancellation islands + qke (E3), which K1 explicitly keeps fp64 ("keep
final bracket fp64 until oracle proves otherwise", scoping L105–110); and the speed-limited
regime is tiny/L2-resident/occupancy-bound (E6/E7), where halving non-island bytes changes
nothing. So the plan's own structure caps it at the already-measured ~1.1×.

- **Missed risk (VRAM):** at 147k the island/buffer overhead already made fp32 *worse*
  than fp64 (E5). A perturbation-native rewrite adds base+perturbation leaves and
  reconstruction buffers; unless it provably *nets* fewer resident+transient bytes than
  fp64 at the OOM grid, it cannot unlock a bigger grid — the entire VRAM rationale.
- **Simpler alternative for the actual goal:** if the goal is the large-grid GPU win (E9
  says fp64 already projects 4.35× core at scale), the right lever is **transient
  column-tiling of the MP-adapter + dycore work arrays** (the honestly-deferred ~4–5 GiB
  task, bigswiss L71–72, L94) and/or **multi-GPU** — *not* fp32. fp32 is aimed at the
  wrong bottleneck for that win (see Redirect).

### Q2 — Is the EARLY-KILL gate right (first gate = compile/HLO/liveness, kill ~1.1×)? Will it predict the 24h outcome? Can a real win hide behind a slow first compile?
**The threshold floor is right; the gate ORDER is wrong; HLO-first can give a false GO.**
- HLO liveness is **necessary-not-sufficient.** The prior compact-island already achieved
  **−37% f64_3d in the acoustic graph** yet **~1.1× full-step**, because the acoustic scan
  is only ~8% of the step (Amdahl) and peak VRAM didn't move (make-or-break L213). A 20–30%
  temp-liveness drop at a "65k proxy" (scoping L245, L395) can therefore PASS the HLO gate
  and still deliver ~1.1× end-to-end. HLO-first authorizes the expensive 24h campaign on a
  proxy that does not predict the outcome.
- **Wrong size:** speed only exists in the DRAM-bound regime (E7), so a 65k HLO proxy is
  measured below the regime that matters.
- **~1.1× is the right "definitely dead" floor but the wrong KEEP bar** — the principal's
  keep-bar is >2× @ 24h (memory L41/L264). Frame the kill as "STOP unless the *large-grid
  warm* number is on a credible >2× trajectory," not "stop if near 1.1×" (which would let a
  dead-on-arrival 1.4× limp forward).
- **Slow-compile hiding a win:** handled *for compile* — the plan correctly excludes
  compile from warm timing (scoping L330). The real "hidden win" risk is **grid size, not
  compile time**: a win that only appears at DRAM-bound scale is hidden by testing too
  small. G0 (large-grid probe) is the fix.
- **Fix:** make **G0 (large-grid warm A/B + transient decomposition) the first binding
  gate**; keep HLO liveness as a structural sub-check that cannot, alone, authorize 24h.

### Q3 — Is the 24h stability design sound? Gap that lets an unstable mode pass or a stable mode fail?
**Design is genuinely strong** — CPU-WRF/savepoint references (not JAX-self-compare,
scoping L259), predeclared tolerance/grid-delta atlas with no post-hoc widening, the
1-substep→24h early-divergence ladder, d03 1km as the qke detector. This satisfies the
project's validation rules and the B6 self-compare lesson. Three real gaps:
1. **Geometry mismatch (the important one):** stability is gated on tiny-nest Canary
   (d02/d03), but speed only exists on large grids; the large-grid fixture gets only
   "warm perf + selected stability horizon" (scoping L269), **not** the full 24h. A mode
   can pass tiny-nest 24h + large-grid warm-perf yet never be proven 24h-stable **where it
   is actually used.** Bind: 24h at the large/DRAM-bound grid (kill-gate 3).
2. **Chaotic decorrelation vs instability:** the fused cascade already shows P diverging
   **1.3→20 over 2h, chaotically, while staying tolerance-PASS** (hostgap §5.4 L342).
   fp32's per-step perturbation is larger. "RMSE vs fp64-GPU" (listed as a metric, scoping
   L308) is a *decorrelation* measure for a chaotic system, not an error — it will grow by
   construction and could false-fail a physical mode, while the "monotonic+superlinear"
   abort (scoping L351) could false-trip on early chaotic growth. **Acceptance must be vs
   the CPU-WRF / grid-delta-atlas absolute envelope only**; drop vs-fp64-GPU RMSE as a
   pass/fail criterion (keep it as diagnostic).
3. **qke/d03 coupling (GPT flags this, L549):** keeping qke fp64 did **not** fix the d03
   1km nonfinite signature — the instability is in the steep-terrain dynamics K1 is
   demoting. The d03 early gate is the right detector; acceptable, but it raises the
   probability of an early STOP.

### Q4 — REALISM: given the nsys occupancy finding, WHERE is K1's payoff real? Worth it, or "stable ~1.1×, not a speed win"?
**Honest expectation = stable mixed mode at ~1.1× on the tiny nests; no single-card speed
win.** Quantified:
- **Tiny nests (all-7):** occupancy/launch-bound, ~1.5 µs kernels, no saturating kernel
  (E6) → **fp32 ≈ 1.0×.** Zero K1 payoff. (This is the v0.17 STOP, already shipped.)
- **Mid grids (~65k):** fp32-dycore measured **0.996×** (E5), L2-resident (E7) → no payoff.
- **Large/DRAM-bound (≥147k):** the *only* regime where fp32 *could* help (E7) — **but**
  fp32-as-built OOMs *worse* than fp64 at 147k (E5) and 211k OOMs in fp64 on a transient
  fp32 didn't shrink (E3/E8). So today fp32 neither speeds nor fits the large grid.
- **The large-grid win that IS real is fp64** (projected 4.35× core, E9) — gated by VRAM,
  which fp32 was measured not to relieve. So K1's nominal target (large-grid supremacy) is
  reachable *without* fp32, via transient tiling + multi-GPU.

Net: **K1's realistic deliverable is "stable mixed mode, ~1.1×," i.e. a stability/correctness
artifact, not a speed lane** — exactly what GPT concedes in its own gate (scoping L247,
L508). That is a real but small prize and does **not** meet the principal's >2× keep-bar.

### Q5 — RISK/COST: worst-case wasted effort; is early-kill enough; cheapest decisive experiment?
- **Worst case:** Phases 0–5 are a multi-sprint architectural rewrite of perturbation-native
  state across `small_step_prep/finish`, `acoustic`, `advance_w`, `calc_p_rho`, boundary,
  and restart — then Phase 7 re-derives the known ~1.1×. Dozens of engineer-sprints to
  reconfirm a double-confirmed negative.
- **Is the plan's early-kill enough?** Partially. The HLO gate fires early but can false-GO
  (Q2). The gate that actually bounds cost — the large-grid warm A/B — is deferred to
  gate-3/Phase-7. So as ordered, the early-kill does **not** bound the worst case.
- **Cheapest decisive experiment:** **G0** above (existing branches, ~1 session, mostly
  static HLO + one warm A/B at the largest grid both fit). It converts K1 from
  "commit-then-measure" to "measure-then-commit" and makes the worst case ~1 session, not
  ~dozens of sprints.

---

## Constructive redirect (if the goal is the large-grid GPU win, fp32 is the wrong lever)
E9 shows the GPU **already** projects ~4.35× core at DRAM-bound scale **in fp64** — the
large-grid win is real and precision-independent; it is **VRAM-blocked** at ~167k cols
(E8). fp32 was measured **not** to relieve that transient peak (E3) and to make it **worse**
as-built (E5). Therefore:
- **Higher-value than K1:** (a) **transient column-tiling** of the MP-adapter + dycore work
  + per-RK tendency arrays (the deferred ~4–5 GiB task, bigswiss L71/L94) to lift the fp64
  grid ceiling; (b) **multi-GPU / cluster weak-scaling** (the standing project value).
  This aligns with the principal's K2 ("other kernel-complex problem," tip: multi-GPU,
  memory L12/L19).
- **Recommendation:** if G0 fails (expected), **redirect the v0.18 perf effort from K1
  (fp32) to K2 (multi-GPU + transient tiling)** — that is where a defensible >2× lives.

---

## Handoff
- **Objective:** adversarially critique the K1 fp32-stability plan; give a launch verdict
  for the principal's morning decision. (Analysis only — no GPU, no code.)
- **Verdict:** **APPROVE-WITH-CONDITIONS (near-REJECT).** Approve only a ~1-session decision
  probe (G0) with a hard kill; do not authorize the Phase 0–7 rewrite until G0 clears a
  >2×-trajectory bar. Expected outcome: STOP → honest negative + redirect to K2.
- **Files read:** k1 scoping; hostgap §7; ADR-031; V0140-FP32-ACOUSTIC-ROADMAP; ADR-007;
  fp32 make-or-break auto-memory; bigswiss_gpu_benchmark.md. Verified the referenced
  branches/flags/source files still exist.
- **Files changed:** `proofs/v018/k1_critic_opus.md` (this file).
- **Unresolved risks:** (1) the *clean* alias/convert-free perturbation-native graph at a
  DRAM-bound grid was never isolated — the one narrow seam G0 closes; (2) if the principal
  values the stability/correctness artifact (perturbation-native acoustic core) for its own
  sake independent of speed, that is a separate, smaller-scope decision than a "speed sprint."
- **Next decision needed:** principal authorizes **G0 only** (not Phase 0–7), with kill-gate
  2 (>2×-trajectory) and kill-gate 3 (large-grid 24h) binding; and pre-agrees the
  K1→K2 redirect if G0 fails.
