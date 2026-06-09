# V0.14 Dynamic Root-Cause — Opus xhigh Critic / Debugger

Reviewer: Claude Opus xhigh (independent critic/debugger)
Date: 2026-06-09
Branch: `worker/gpt/v013-close-manager`
Mode: read-only. No `src/` edits, no GPU, no TOST, no Switzerland, no FP32, no
memory work, no Hermes.

## Objective

Challenge the manager's conclusion that the next-best target is **final-RK
dynamic coupling** (PGF / mass-wind / theta-pressure around PSFC, MU, P, PH, U/V,
V10). Decide whether the evidence still points to another bug class, produce a
ranked hypothesis table, recommend one highest-yield sprint plus a backup, and
state whether production edits are justified now.

---

## Bottom line

**The manager's "final-RK coupling" target is NOT justified by the current
evidence, because the *input* to that final RK step is already grossly diverged
from WRF.** `proofs/v014/pre_rk_input_boundary.json`
(`PRE_RK_INPUT_JAX_PRESTEP_MISMATCH_CONFIRMED`) shows the step-5999 carry that
feeds step 6000 already differs from WRF's own pre-RK input savepoint by
**MU′ rmse 195.9 / worst 267 Pa, P′(k1) rmse 526 / worst 590 Pa, T_OLD rmse 4.64
/ worst 6.2 K** — *before any final-RK term executes*. You cannot localize a
single-step coupling defect on the **output** side when the **input** is off by
hundreds of Pa and several K; every output field is guaranteed to read "DIFF"
regardless of whether the dycore coupling is correct. That is exactly what
`same_state_momentum_mass.json` produced (all 10 fields DIFF, "first mismatch U"
purely by sprint field order).

The class the evidence favors is **upstream**: accumulated per-step trajectory
drift and/or a carry-producer/handoff defect (corroborated by the existing
`PRODUCER_WRITES_BAD_FINAL_CARRY` and `BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE`
threads, and the theta-side `T_EVOLUTION_MISMATCH_CONFIRMED`). The final-RK
coupling hypothesis is *possible but unproven* and is currently **untestable**
under a diverged input.

Two of the three trigger proofs are also **contaminated/non-probative**:

1. `same_state_momentum_mass.json` loads `d02_step5999_full_carry.pkl`, which
   **predates** the live-nest base-source fix. Its `MUB` worst = **1050.3046875
   Pa** and `PB` worst = **1047.015625 Pa** are *bit-identical* to the pre-fix
   base bug (`live_nest_base_source_fix.json` `original_target_patch_max_abs`),
   now closed to ≤0.05 Pa. So its mass/pressure "DIFF" is a **stale, already-fixed
   bug**, and `strict_same_input_wrf_savepoint: false` — it is not a same-input
   single-step localizer at all (it runs one step from a 5999-step-drifted,
   stale-base carry). Its own `unresolved_risks[0]` admits this.
2. `grid_after_live_nest_base.json` (`GRID_SYMPTOM_NOT_CLOSED`, post-fix GPU
   h1–h12, V10 rmse 2.55, worst h11 4.28) is **clean** and is the *only*
   probative trigger: it proves the base fix did **not** close V10 → the residual
   really is dynamic/upstream. This confirms the prior debug-method critic's
   F2/F4 prediction.

So: the manager pivoted to the right **class** name ("dynamic"), justified by the
clean `grid_after` proof — but proposes to instrument the **wrong boundary** (the
final RK step's output), using a **contaminated instrument** (`same_state`).
Proof must precede edits, and the correct proof boundary is a **strict same-input
single-RK-step parity**, not final-RK output instrumentation.

---

## Ranked hypothesis table

| # | Hypothesis | Evidence FOR | Evidence AGAINST | Cheapest falsifier | Instrument (files/functions) | Expected proof object |
|---|---|---|---|---|---|---|
| H1 | **Upstream accumulated per-step drift** — the 5999-step carry is the end of slow JAX↔WRF divergence; final-RK coupling is a red herring | pre-RK *input* already off (MU′ 267 Pa, P′ 590 Pa, T 6.2 K) before final RK; `grid_after` V10 grows with lead (h7–h14 worst); base fix did not move V10 | A single per-step term could also be wrong (overlaps H3/H5) | **Strict same-input single-RK-step parity at step 6000**: WRF `pre_rk_input` → JAX `step.step` → compare to WRF `post_after_all_rk_steps`. If it MATCHES, the dycore is clean and divergence is pure accumulation | `dynamics/step.py::step`, `dynamics/rk3.py::rk3_step`, `runtime/checkpoint` loader; new proof harness | `proofs/v014/same_input_single_rk_parity.json` verdict `DYNAMICS_CLEAN` vs `…DEFECT_LOCALIZED` |
| H2 | **Carry-producer / previous-step handoff writes a bad final carry** | existing `prestep_carry_source_trace` = `PRODUCER_WRITES_BAD_FINAL_CARRY`; `previous_step_handoff_bisect` = `BAD_BEFORE_FINAL_PARTIAL_SUBCYCLE` (a static base error cannot be "bad before the final partial subcycle") | producer trace not yet tied to a specific operator/line | Compare carry **producer** output for one step against WRF same-step post-RK with **WRF input** (removes drift): does the producer reproduce WRF for one controlled step? | `integration/d02_replay.py` (carry write path), `dynamics/coupled_step.py`, `dynamics/mu_t_advance.py`, `small_step_scratch.py` | per-step producer-parity JSON, first divergent operator named |
| H5 | **Theta / physics-tendency source folded into RK** (theta-pressure, not PGF/mass-wind) | `T_OLD` off **6.2 K** and `T_HIST_SRC` off 3.36 K at input; same-day `jax_theta_evolution_localization` = `T_EVOLUTION_MISMATCH_CONFIRMED`; T drives P via EOS → P′ 590 Pa is downstream of T | MU′ also off (267 Pa) → not purely thermal | In the same-input single step, decompose residual by field: T-dominated ⇒ H5; U/V/PGF-dominated ⇒ H3 | `dynamics/tendencies.py`, `dynamics/mu_t_advance.py`, physics→dycore tendency fold in `coupled_step.py` | field-decomposed residual table inside the H1 proof object |
| H3 | **Final-RK PGF / mass-wind coupling defect** (manager's target) | output U/V/P "DIFF" in `same_state`; `grid_after` PSFC/P/MU/PH large dynamic rmse | **Input already diverged → output DIFF is non-probative**; `same_state` base "DIFF" is the stale fixed bug; symptom is interior-wide, not a localized coupling signature | Same-input single step (H1): if it MATCHES, H3 is **falsified** | `dynamics/acoustic_wrf.py`, `acoustic_loop.py`, `tendencies.py` (PGF), `mu_t_advance.py` | only pursue **after** H1 shows a real same-input single-step mismatch |
| H4 | **Stale-base / non-same-input contamination of the trigger proofs** (measurement artifact, not a live bug) | `same_state` MUB/PB worst = **bit-identical** to the pre-fix base bug (1050.30 / 1047.02 Pa), already fixed to ≤0.05 Pa; `strict_same_input_wrf_savepoint:false` | n/a — this is confirmed | Regenerate the step-5999 carry post-fix and rerun, **or** switch to WRF-input single-step (H1) which sidesteps it entirely | `proofs/v014/same_state_momentum_mass.py` input carry | superseded by H1 proof (no separate sprint needed) |

Ranking: **H4 is certain** (clear it by construction). **H1/H2 (upstream) are
strongly favored.** **H5 (theta/physics) outranks H3** on the input signature
(T 6.2 K, theta-evolution already confirmed). **H3 (manager's PGF/mass-wind) is
least supported and currently untestable.** Crucially, **one experiment — the
strict same-input single-RK-step parity at step 6000 — discriminates all of
them at once.**

---

## Recommended next sprint (highest-yield)

**Strict same-input single-RK-step parity at d02 step 6000.**

- Input: WRF's own `pre_rk_input` savepoint (already on disk:
  `/tmp/wrf_gpu2_v014_pre_rk_input_boundary/pre_rk_output/pre_rk_input_d2_step_6000_*.txt`).
- Action: build a JAX `State`/`Tendencies` from that WRF pre-RK patch, run **one**
  `gpuwrf.dynamics.step.step` (or `rk3.rk3_step`), compare to WRF
  `post_after_all_rk_steps_pre_halo` step-6000 savepoint (already on disk:
  `/mnt/data/wrf_gpu2/v014_post_rk_refresh/refresh_output/…`), on the
  halo-valid interior subset of the patch.
- Why it beats the manager's plan and the alternatives:
  1. It removes **both** confounds that make every prior trigger non-probative —
     5999-step accumulated drift (H1) **and** stale-base contamination (H4) — by
     using WRF's correct input.
  2. It is a clean **go/no-go on the manager's own hypothesis (H3)**: MATCH ⇒
     dycore coupling is correct, divergence is upstream (redirect to H1/H2);
     MISMATCH ⇒ a real single-step defect, now **term-localizable** (T-dominated
     ⇒ H5; U/V/PGF-dominated ⇒ H3).
  3. It reuses **existing WRF savepoints** (both surfaces already emitted at step
     6000) — no new WRF build/run; cheap, CPU-only, rule-clean.
- Proof object: `proofs/v014/same_input_single_rk_parity.json`, schema
  `wrfgpu2.v014.same_input_single_rk_parity.v1`, with per-field max_abs/rmse on
  the valid interior, a field-decomposed residual ranking, and verdict
  `DYNAMICS_CLEAN_SINGLE_STEP` vs `FINAL_RK_<term>_DEFECT_LOCALIZED`.

**Design caveats the sprint must handle (do not skip):**
- *Tendency control.* WRF's post-RK includes physics tendencies. Feed WRF's
  `rk_tendency`/physics tendency from a savepoint (or compare against a
  dynamics-only sub-boundary such as the already-emitted
  `post_final_calc_p_rho_phi`) so the comparison isolates the **dycore**, not
  JAX-vs-WRF physics. Otherwise a physics mismatch re-contaminates the result.
- *Patch width.* `pre_rk_input` is a patch (i 1..23) with an 8-cell halo ⇒ only
  ~i 9..15 is stencil-valid for a single step. That is enough to test an
  interior-wide coupling, but if the vertical-implicit/acoustic stencil needs a
  wider footprint, re-emit a wider `pre_rk_input` patch via the existing
  `pre_rk_input_boundary_wrf_patch.diff` hook (cheap; exes already built).

## Backup sprint (if same-input single step MATCHES → dynamics clean)

**Per-step drift-onset bisection + carry-producer trace (H1/H2).** Sweep the
strict same-input single-step parity across earlier steps (binary search
0→5999) to find the first step whose single-step output diverges. If *every*
single step matches but the multi-step carry drifts, the divergence is pure
accumulation/feedback or a producer/handoff write — resume the
`PRODUCER_WRITES_BAD_FINAL_CARRY` thread on `integration/d02_replay.py` and
`dynamics/coupled_step.py`/`mu_t_advance.py`, comparing the producer's
single-step write against WRF with WRF input. Proof:
`proofs/v014/per_step_drift_onset.json` naming the first divergent step/operator.

---

## Production-edit decision

**No production edits are justified now. Proof must precede edits.** The only
edit with a proof to date is the live-nest base-source fix, already partially
landed; it is confirmed *not* the V10 closer (`grid_after` V10 rmse 2.55). No
current proof isolates a specific final-RK code defect — the would-be evidence
(`same_state`) is contaminated and non-same-input. The exact proof boundary that
must turn green-or-red before any `dynamics/` edit is the **strict same-input
single-RK-step parity at step 6000** above. It beats editing-then-checking
because it is the first instrument whose result is *interpretable*: under the
current diverged input, any final-RK source change would be tuned against noise.

---

## Files changed

- None in `src/` (verified `git diff -- src` empty).
- Added: this review; optional `proofs/v014/dynamic_root_cause_opus_critic.json`.

## Commands run

- `python3` JSON inspections of `same_state_momentum_mass.json`,
  `grid_after_live_nest_base.json`, `live_nest_base_source_fix.json`,
  `pre_rk_input_boundary.json` (read-only).
- `ls` of WRF savepoint dirs (`pre_rk_input` + `post_after_all_rk_steps` +
  `post_final_calc_p_rho_phi`, all step 6000, confirmed present).
- `grep`/`sed` of `dynamics/` entry points (`step.py`, `rk3.py`) and the
  `same_state`/`pre_rk_input_boundary` generators (read-only).
- `python -m json.tool proofs/v014/dynamic_root_cause_opus_critic.json` (validate).
- `git diff -- src` (confirmed empty).

## Proof objects produced

- `.agent/reviews/2026-06-09-v014-dynamic-root-cause-opus-critic.md` (this file).
- `proofs/v014/dynamic_root_cause_opus_critic.json` (compact machine summary).

## Unresolved risks

- The same-input single-step proof depends on **controlling the tendency input**;
  if the sprint lets JAX compute its own physics, the result is reconfounded.
- `pre_rk_input` patch width may be too narrow for the full single-step stencil;
  may need a one-line wider-emit re-run of the existing WRF hook.
- The V10 symptom rests on essentially one fresh spatial case (prior critic's
  evidence-quality note); confirm on ≥2 cases before any large fix lands.
- H5 (theta/physics) vs H3 (PGF) cannot be separated until the same-input single
  step runs; do not pre-commit to either operator family.

## Next decision needed

Approve dispatching the **strict same-input single-RK-step parity at step 6000**
(GPT-5.5 xhigh, codex — cross-model vs the Opus-authored dycore) as the next
sprint, with the tendency-control and patch-width caveats baked into its
contract, **before** any `dynamics/` source edit. If it returns
`DYNAMICS_CLEAN_SINGLE_STEP`, redirect to the upstream drift/producer backup; if
it localizes a term, edit that term only.
