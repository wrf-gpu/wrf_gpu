# V0.14 Debug-Method Critic — Grid Divergence & Live-Nest Base Port

Reviewer: Claude Opus xhigh (independent critic)
Date: 2026-06-09
Branch: `worker/gpt/v013-close-manager`
Scope: critique the v0.14 grid-divergence debug process and the conclusion that
native live-nest base initialization is the next source-fix target.
Mode: read-only. No source/proof edits. No GPU/TOST/Switzerland/FP32.

**Bottom line:** The native live-nest base port is a *legitimate correctness
fix* but it is **not justified as the next or symptom-closing target.** The
manager's own evidence — including the 2026-06-08 divergence probe — ranks the
base-state mismatch as a bounded, boundary-localized *contributor that "cannot by
itself explain"* the V10 divergence, while the evidence-favored, interior-wide
dynamics thread was abandoned mid-localization. Run one cheap falsifier before
dispatching the port.

---

## Findings (ordered by severity)

### F1 — CRITICAL: The next-target conclusion contradicts the manager's own probe
`proofs/v014/wind_mass_divergence_probe.md` (the manager's own 2026-06-08 22:25
anatomy of the actual V10 symptom) ranks root causes:
- **#1 `favored_by_this_probe`**: "Prognostic wind-column divergence … peaking
  around h10–h14" (a dynamics hypothesis).
- **#2 `plausible_contributor_not_full_explanation`**: "Static base-state …
  mismatch", with the explicit limit: *"The static base-state signal cannot by
  itself explain the h10–h14 V10 peak or the old case-to-case V10 bias sign
  changes."*
- Its recommended **"Next Fix Probe"** was a CPU-only same-state **momentum/mass
  tendency localization** (PGF/Coriolis/advection/diffusion/spec-relax/physics
  folding) on h8–h14 interior ocean cells.

The next sprint (`.agent/sprints/.../v014-live-nest-base-source/`) instead commits
a full native source port to hypothesis **#2**, and the probe's recommended #1
follow-up was not run. This is chasing the cleanly-attributable signal (a crisp,
constant `MUB ≈ 1050 Pa`) over the dominant-but-messier one (interior-wide,
lead-dependent wind divergence). *Answer to required-work item 1: the conclusion
is not justified as the primary/closing target.*

### F2 — CRITICAL: Spatial mismatch between symptom and proposed cause
The V10 error is **interior-wide, not boundary-localized**:
`wind_mass_divergence_probe.md` §Splits → boundary `frame_5cells` RMSE 2.541 vs
`interior_excluding_5cell_frame` RMSE 2.519 (**ratio 1.009**), and hypothesis #4
(boundary-frame defect) is `disfavored_by_this_probe`. But the proposed fix,
`blend_terrain` (`dyn_em/nest_init_utils.F:712`, cited in
`live_nest_base_hook.json`), **only modifies the outer
`spec_bdy_width + blend_width ≈ 10` frame rows/columns**; the interior is 100%
child `wrfinput` terrain. The worst base/terrain deltas sit at the domain corner
`[54,155]` (HGT 228 m, MUB 2636 Pa; `live_nest_base_hook.json`
`native_wrfinput_vs_cpu_h0_whole_domain`). A boundary-frame terrain correction
is structurally incapable of closing an interior-wide V10 error.

### F3 — HIGH: "worst field = MUB 1050.3" is a static-field reporting artifact
Every snapshot in `earlier_source_bisect.md` and `previous_step_handoff_bisect.md`
reports `worst field = MUB, max abs 1050.3046875` — *identical at d02 steps 0,
600 … 5999*. MUB is the **time-invariant base mass column**; in WRF's
base/perturbation split it never evolves. A bisect that selects "worst field by
max-abs" will therefore finger the static corner error at **every** snapshot
regardless of the dynamics, masking the dynamic signal the same-day theta thread
*did* localize (`jax_t_history_source_attribution.md`
`T_EVOLUTION_MISMATCH_CONFIRMED`; `previous_step_handoff_bisect.md`
`BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE`; `prestep_carry_source_trace.md`
`PRODUCER_WRITES_BAD_FINAL_CARRY`). A static base error cannot be "bad *before
the final partial subcycle* but fine before" — that signature is dynamic. The
bisect headline metric, not the physics, drove the pivot to base-state.

### F4 — HIGH: The decisive, cheap falsifier was skipped
No proof injects the CPU-WRF h0 base/terrain as a one-shot **init override**
(validation-oracle, not production) and re-measures V10 (grep of `proofs/v014`
finds none). This is the single experiment that discriminates: if injecting
oracle HGT/MUB/PB/PHB at GPU init collapses V10 below the 1.5 m/s envelope, the
port is justified; if V10 stays red (the probe's prediction), the port is
misdirected. It is one CPU/GPU forecast, no new production source, and uses h0
strictly as an oracle — fully inside the rules. *The source-sprint's item 6
("show the init split no longer explains the h10 pre-RK mismatch") frames this as
post-fix verification; it must be a pre-dispatch go/no-go instead.*

### F5 — MEDIUM: `NATIVE_PORT_PLAN_READY` over-weights the easy half
The port's strongest evidence — `start_domain_em` formula residuals < 0.1
(`live_nest_base_hook.json` `start_domain_formula_residuals`, tol 0.2) —
validates the **downstream algebra given correct terrain**. The genuinely
uncertain stage is the **parent→child SINT interpolation + `blend_terrain`
reproducing CPU h0 HGT**, which is explicitly untested (`unresolved_risks[0]`:
"Native SINT parity still needs implementation-level tests") and whose naive
substitute is already rejected (bilinear+blend → PB patch max 796 Pa,
`base_state_split_fix.md`). The readiness label is calibrated on the proven half;
the risk lives in the unproven half.

### F6 — MEDIUM: Declining the WRF savepoint leaves two ported fields with no oracle
The decision (`live_nest_base_hook.json` `decision.why_not_new_wrf_savepoint…`)
skips a disposable-tree savepoint on cost grounds ("patch/relink 1.6G tree"), but
`real.exe`/`wrf.exe` are **already built** in that tree
(`wrf_run_feasibility.built_*_exe`) and marker runs exist. Consequence: CPU h0
has no `T_INIT`/`ALB` (`netcdf_inventory.cpu_wrfout_h0`), so 2 of the 5 recomputed
fields will be validated only by formula residual, and post-blend HGT only
against h0. If the port proceeds, a minimal post-blend/post-`start_domain`
savepoint would give a *complete* oracle (incl. the untested interp+blend stage)
at low marginal cost. The stated cost rationale is weak given the prebuilt exes.

### F7 — PROCESS: Sprint fragmentation
~24 `2026-06-09-v014-*` sprints in a single day target one divergence bug, with a
long localization ladder (pre-RK → post-RK refresh → theta evolution → t-history
→ prestep-carry → handoff bisect → base-state split → earlier-source bisect →
live-nest hook → live-nest source). This is the exact anti-pattern AGENTS.md
warns against ("prefer large coherent sprints / whole milestones with strong
end-to-end falsifiable gates over many small chained sprints") and the
`feedback_sprint_sizing` memory ("coupling bugs hide in chained mini-sprints").
An early end-to-end override experiment (F4) would have pruned most of the ladder.

### What the process did well (keep)
- Grid-fields-first ordering matches principal priority #1; station TOST is
  correctly held behind the field envelope (`V0140-VALIDATION-PLAN.md` B4b-before-B4a).
- Performance-protection is well-encoded in the source-sprint contract: no h0
  production input, no timestep-loop host/device transfers, init host-side before
  stepping, explicit init API instead of global-state smuggling, standalone/restart
  regression check (item 7).
- Tolerances are pre-registered ("no tolerance widening after seeing results").
- Cross-model debug rule appears *satisfied*: `d02_replay.py`/`nesting/*` are
  Opus-era authored (git: `[v0.12.0 A-standalone]`, `[P0-1a nesting]`); GPT-5.5 is
  debugging/fixing them (opposite model). *(Inference from commit tags.)*

---

## Concrete accelerators / falsifiers

1. **Init-override falsifier (do first, gates the port).** Inject CPU-WRF h0
   `HGT/MUB/PB/PHB` (oracle only) into the GPU d02 init, run the existing
   20260501 case, re-measure the V10 grid envelope. *Falsifier:* if V10 RMSE does
   not drop materially below 1.5 m/s, the live-nest base port cannot be the
   symptom fix → do not dispatch it as such. Cheap, rule-clean.
2. **Resume the probe's own #1 follow-up.** Run the CPU-only same-state
   momentum/mass tendency localization on h8–h14 interior ocean cells
   (PGF/Coriolis/advection/diffusion/spec-relax/physics folding → `ru`/`rv`/`mu`),
   as `wind_mass_divergence_probe.md §Next Fix Probe` already specifies. This
   targets favored hypothesis #1 directly and continues the abandoned
   `BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE` lead.
3. **Fix the bisect metric.** Exclude static base fields (MUB/PB/PHB/HGT) from the
   "worst field" selector, or subtract the base and bisect on perturbation/dynamic
   fields (theta, u, v, w, P′). Otherwise every dynamics bisect keeps fingering
   MUB (F3).
4. **Spatially co-locate cause and symptom.** Overlay the per-cell V10 error map
   on the per-cell HGT/MUB blend-zone delta map for the 20260501 case. If the V10
   error does not concentrate where the blend delta lives, F2 is confirmed
   quantitatively.
5. **If the port proceeds, validate the uncertain stage.** Take the prebuilt
   disposable WRF (F6) to a minimal post-`blend_terrain` / post-`start_domain_em`
   savepoint, giving a complete `HGT/MUB/PHB/T_INIT/ALB` oracle. Gate the port on
   GPU-SINT reproducing CPU h0 HGT to a pre-declared tolerance, not just on the
   downstream formula residual.

---

## Process critique (short)
- **Hypothesis discipline:** the team produced a well-ranked probe and then acted
  against its ranking. Re-anchor each pivot to the latest ranked evidence; a
  pivot away from the favored hypothesis should require an explicit falsifier of
  it, not a crisp number from a lower-ranked one.
- **Metric hygiene:** a static field's max-abs captured a chain of dynamics
  bisects (F3). Headline selectors must be physics-aware.
- **Sprint boundaries:** collapse the localization ladder into fewer sprints with
  end-to-end falsifiable gates (F4, F7); each micro-sprint re-pays context cost.
- **Evidence quality:** the V10 symptom rests on essentially **one** fresh spatial
  case (`v10_grid_diagnostics.md`: 2/3 cases are `case_json_only`, GPU wrfout
  absent). Confirm the symptom on ≥2 fresh spatial cases before a large port, so
  the fix target isn't tuned to a single realization.
- **Validation-oracle vs production dependency:** correctly maintained — every
  proof flags h0 as oracle-only and the contracts forbid h0 as production input;
  keep this bright line for the override experiment in accelerator #1.

---

## Final recommendation
**Do not dispatch the native live-nest base port as the next or symptom-closing
fix yet.** It is a real correctness gap (WRF blends; the GPU path should too) but
the manager's own evidence bounds it as a boundary-localized,
contributor-not-explanation defect that cannot close the interior-wide V10
divergence.

Re-sequence:
1. **Immediate (before any source port):** run accelerator #1 (init-override
   falsifier) and accelerator #2 (momentum/mass tendency localization) in
   parallel; apply accelerator #3 to the bisect metric. Confirm the symptom on a
   second fresh spatial case.
2. **Next-after-falsifier:** fix whichever defect the falsifier implicates.
   Expectation from current evidence: the dynamic acoustic/RK theta-carry path
   (`BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE` / `PRODUCER_WRITES_BAD_FINAL_CARRY`) is the
   primary symptom owner; the base port is a secondary correctness fix to land
   after, validated with a complete savepoint oracle (accelerator #5).
3. **Stop / avoid:** do not let the static `MUB 1050` number drive further bisects
   (F3); do not bill the base port as the grid-divergence closer until #1 shows
   V10 actually moves; do not skip the V10-after-override check; preserve the
   GPU-native rules already encoded (no h0 production input, no timestep-loop
   transfers).

Sequencing priorities #2 (FP32/mixed acoustic) and #3 (memory) remain correctly
gated behind credible grid fields (`V0140-FP32-ACOUSTIC-ROADMAP.md`,
`V0140-MEMORY-FIX-ROADMAP.md`); no change recommended there.
