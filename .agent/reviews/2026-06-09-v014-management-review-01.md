# V0.14 Management Review 01 — Roadmap Drift / Fastest-Rigorous-Path

Reviewer: Opus 4.8 xhigh (independent management reviewer)
Date: 2026-06-09
Branch: `worker/gpt/v013-close-manager`
Mode: read-only except this file. No `src/` edits, no GPU, no Hermes.

## 1. Verdict

The roadmap is correctly framed: grid-cell parity first, TOST held ("no slob"),
priority order (grid → fp32 → memory → TOST) sound, base-fix honestly *not*
billed as the V10 closer. Proof discipline is real — honest `BLOCKED` verdicts,
roundoff-validated WRF truth, cross-model debug, pre-registered tolerances. But
the manager is **not** on the *fastest* rigorous path. The dynamic-root-cause
critic's "cheap, reuse-on-disk" same-input single-RK experiment has become a
4-sprint instrumentation ladder (same-input → full-pre-rk → source-save →
full-domain-wrapper), all `BLOCKED`, with the discriminating comparison still
unrun — re-triggering the F7 fragmentation the first critic flagged. It is also
the hardest instance (step 6000) of an experiment that is cheapest from the clean
end. Re-sequence, do not change the goal.

## 2. Ranked findings

| Sev | Issue | Evidence | Fix |
|---|---|---|---|
| CRIT | "Cheap discriminator" turned into a 4-sprint blocked ladder; no green/red same-input result yet | 4 consecutive `BLOCKED` verdicts (`same_input…`, `full_pre_rk…`, `source_save…`, full-domain-wrapper open); critic said surfaces were "already on disk, no new WRF run" | Consolidate into ONE larger sprint: build full WRF emit + JAX pre-halo wrapper + run the comparison; if blocked, name ALL remaining blockers in one pass, not one/sprint |
| CRIT | Wrong *instance* of the right experiment: step 6000/h10 is the most instrumentation-heavy place to test single-step parity, yet divergence is trajectory drift already huge by h10 | needs 5999-step carry, deep source/save hooks, `scalar_old`; `pre_rk_input` already off T6.2K/MU267Pa/P590Pa | Run same-input single-step parity from the SHARED `wrfinput` at an early step + coarse 0→5999 drift-onset bisection (critic's own backup); bisect from the clean end where instrumentation is ~free |
| HIGH | F7 sprint fragmentation recurring after being flagged | ~30 `2026-06-09-v014-*` micro-sprints; each `BLOCKED→next blocker` re-pays context cost | Larger coherent sprints with end-to-end falsifiable gates per AGENTS.md / `feedback_sprint_sizing` |
| HIGH | V10 symptom still rests on ~1 fresh spatial case | first critic F-note; only Case 3 had retained wrfout; `grid_after` = 1 fresh h12 case | Confirm dominant-field/lead signature on ≥2 fresh cases BEFORE any large dycore fix lands |
| MED | Risk the step-6000 wrapper, even if CLEAN, won't locate drift onset → pivot to bisection anyway | critic: CLEAN ⇒ "redirect to upstream drift/producer backup" | Make drift-onset bisection the first decisive move, not the fallback |
| LOW | Bisect "worst-field" metric hygiene (static MUB artifact) | first critic F3; later proofs moved to T/MU′/P′ | Keep perturbation/dynamic-field selectors; exclude static base from headline selectors |
| KEEP | Honesty + GPU-native rules intact | no fake greens; no h0 production input; no timestep-loop transfers; base-fix not over-claimed | Preserve these bright lines through the re-sequence |

## 3. Next 3 sprints

- **Decisive discriminator (consolidated).** In ONE sprint: (a) finish the
  step-6000 same-input single-RK wrapper AND (b) run same-input single-step
  parity from shared `wrfinput_d02` at an early step (≈1, 60, 600, 3000, 5999),
  tendency-controlled. *Gate:* at least one strict same-input comparison executes
  with `DYNAMICS_CLEAN` / `FIRST_DIVERGENT_STEP_N_<field>` — no more `BLOCKED`-only.
- **Symptom confirmation on a 2nd case.** Cheap GPU h12 on a different fresh L2
  date + grid envelope (CPU-scored, parallel). *Gate:* ≥2 fresh cases agree on the
  dominant diverging field(s) and lead window before any large fix.
- **Fix or trace (branch on discriminator).** If MISMATCH→edit only the named
  operator; if CLEAN→per-step producer/handoff trace. *Gate:* same-input parity
  goes green at the fix AND V10 drops below the 1.5 m/s envelope on both confirmed cases.

## 4. Goal-change gate

**NO_GOAL_CHANGE.** The end goal (WRF-faithful-enough, GPU-optimized, scalable
GPU rewrite; Canary 3/1 km first) remains technically achievable and is the right
target. The current grid-divergence investigation is squarely on the critical
path — the constitution's "physics correctness precedes speed claims" mandates
closing the ~2.5 m/s V10 / hundreds-of-Pa residual before TOST or fp32/memory
work. The problem is path *efficiency* (instance choice + fragmentation), not goal
validity. No evidence shows the goal impossible or superseded.

## 5. Context-sparing handoff (manager remembers)

- Grid-first ordering + TOST hold = correct; keep them.
- The base-state split fix is real but NOT the V10 closer (`grid_after` V10 2.55) — settled; stop re-litigating base fields.
- `same_state_momentum_mass` is contaminated (pre-fix stale carry, `strict_same_input:false`) — not a localizer; do not cite as evidence.
- The single-step same-input parity is the right experiment; you have the right CLASS and BOUNDARY — only the INSTANCE and CADENCE are wrong.
- Bisect from the CLEAN end (early step, shared wrfinput) — instrumentation is near-free there vs deep hooks at step 6000.
- Stop emitting one BLOCKED micro-sprint at a time; one larger sprint that names all blockers at once.
- A CLEAN step-6000 result still won't locate drift onset → run bisection first regardless.
- Confirm the V10 signature on ≥2 fresh cases before landing any large dycore edit.
- Keep tendency-control + patch-width caveats (physics mismatch re-contaminates).
- No production edit until a strict same-input proof names the first wrong operator/step; preserve no-h0-input / no-loop-transfer rules.
