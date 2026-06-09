# Reviewer Report: V0.14 Step-1 First-RK Part1 P-State Split

Date: 2026-06-09

## Decision

Accept the sprint as a localization proof, not a source fix.

Decision: accepted with next sprint required for live-nest perturbation-state
initialization.

## Evidence

- `proofs/v014/step1_first_rk_part1_p_state_split.json`
- `proofs/v014/step1_first_rk_part1_p_state_split.md`
- `.agent/reviews/2026-06-09-v014-step1-first-rk-part1-p-state-split.md`

## Review

The proof answers the sprint question with the fastest rigorous method available:
reuse accepted CPU-WRF savepoints and add a current CPU-JAX stage capture. It
shows WRF `first_rk_step_part1` and `phy_prep` are not the first material
`P/MU/W` fault surface. It also rules out boundary package, carry construction,
and halo as the source because those transitions have zero deltas.

The exact next surface is the live-nest child perturbation-state initialization
contract from raw child to pre-part1/live child state for `P_STATE/MU_STATE` and
`W_STATE`.

## Risks

- The proof does not provide the WRF formula/source implementation for the
  missing perturbation-state initialization.
- The next sprint must avoid a CPU-WRF runtime dependency and preserve GPU
  residency.
