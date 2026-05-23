# Tester Report — M6.x Warm-Bubble Failure Diagnostic

**Role:** tester (Opus 4.7 acting as sonnet-test-engineer)
**Sprint:** `2026-05-23-m6x-warm-bubble-failure-diagnostic`
**Substantive deliverable:** `diagnostic-report.md` (this folder) — the 9-section diagnostic verdict required by the sprint contract.

This report exists to satisfy the tester role's procedural requirement (>=400 bytes + `Decision:` token). All evidence is captured in `diagnostic-report.md` and `probe_warm_bubble_vs_slice.json`.

## Tests Added Or Run

I did NOT add tests under `tests/` because:
- The sprint contract is explicitly **diagnostic-only**: "No new code or refactoring. READ-ONLY except for adding diagnostic instrumentation in a single file" (sprint-contract.md §"Non-Goals").
- The contract authorizes exactly ONE code addition: `scripts/diagnostic_warm_bubble_vs_slice.py`. Adding tests would exceed the diagnostic scope.
- The user's explicit instruction promoted this session to the diagnostic-analyst role (producing `diagnostic-report.md` with §1–§9, including a §7 verdict from the allowed set).

Instead I ran the diagnostic probes the contract requires and reproduced the failure independently. The new file `scripts/diagnostic_warm_bubble_vs_slice.py` defines 10 probes that act as one-shot test runs:

1. `a_3d_acoustic_substep_carry_baseline` — reproduces the harness failure (`w_signed_max=0.0409` at 600 s, matches worker-reported `0.0409710985`).
2. `b_3d_direct_recurrence` — same setup driven only through `_mpas_recurrence_vertical_update` (bypasses `acoustic_substep_carry`); also fails to sustain bubble lift.
3. `c_1d_shrunk_acoustic_substep_carry` — single mid-bubble column through the full carry path.
4. `d_1d_shrunk_direct_recurrence` — single column through pure recurrence; isolates architectural limit from carry-path wiring.
5. `e_first_substep_rhs_decomposition` — buoyancy / `cofwz·θ-restoring` / `cofwr·ρ-coupling` / `cofwt·pressure` magnitudes at substep 1.
6. `f_buoyancy_scale_sweep` — `MPAS_COLUMN_BUOYANCY_TENDENCY_SCALE ∈ {0.38, 1.0, 2.6, 2.63}` 60-s sweep.
7. `g_overwrite_check` — direct confirmation that `diagnose_pressure_al_alt` erases the recurrence-derived `p_perturbation` (12.7 Pa → 4×10⁻¹¹ Pa).
8. `h_slice_oracle_vs_unified` — production-grade single-column setup; unified path matches slice oracle within ~38% at 40 substeps.
9. `i_epssm_sweep_300s` — `epssm ∈ {0.0, 0.1, 0.3}` 300-s sweep.
10. `j_mu_continuity_ablation_60s` — `mu_continuity={True,False}` ablation.

## Results

Captured in `probe_warm_bubble_vs_slice.json` (full instrument output, ~30 KB) and `diagnostic-report.md` (interpretation).

Headline:
- Worker's reported failure (`w_max ≈ 0.04 at 600 s`) reproduced exactly (`probe a`).
- Bypassing `acoustic_substep_carry` (`probe d`) ALSO fails — the architectural recurrence cannot sustain bubble lift on its own.
- `diagnose_pressure_al_alt` confirmed to overwrite recurrence's `p_perturbation` (`probe g`): a real wiring bug independent of the architectural gap.
- Magic-number sweep (`probe f`) shows no scale value lifts sustained `w_signed_max` toward the [5,10] m/s target.
- Sign convention is correct (positive θ′ → positive `buoyancy_face` → positive `w` at step 1).
- The harness's `jnp.max(w) = 0.04` at 2400 substeps hides a co-occurring **non-physical blowup** in θ′ (1.98 K → 144 K) and p′ (0 → 450 kPa), bounded only by the temporary `_mu_continuity_increment` `tanh` limiter.

## Fixtures Used

- `scripts/m6_warm_bubble_test.py` setup (mirrored, not invoked directly): nx=ny=64, nz=40, dx=400 m, dz=100 m, dt=2 s, n_acoustic=8, duration=600 s, Gaussian θ′ peak=2 K, bubble center z=2000 m, radius=2000 m.
- `mpas_column_slice` warm_bubble_2km: N_LEVELS=16, COLUMN_HEIGHT=10000 m, DT_ACOUSTIC=1.0 s, N_SUBSTEPS=40, EPSSM=0.1, cos²·3 K bubble.
- Single mid-bubble column extracted from the 3-D harness state (1×1×40, dz=100 m) for the apples-to-apples probes `c` and `d`.

No binary fixtures committed; the slice oracle's `data/fixtures/mpas_column_slice/warm_bubble_2km.npz` already exists in the repo (gitignored).

## Gaps

1. I did NOT do a line-by-line MPAS-A 5.3 `mpas_atm_time_integration.F:2146-2208` vs `_mpas_recurrence_vertical_update` audit — a deviation there (off-by-one, missing `zz[k]` zeta-factor, wrong `cofwt` index) could in principle reverse the verdict from MIXED to WIRING-BUG. The parallel ADR-021 prototype sprint dispatched in `181f544` will surface this if so.
2. I did NOT test the wiring fix proposed in §7. A 1-line `if not config.non_hydrostatic` guard on the post-vertical-update overwrite is the smallest possible fix; the next worker sprint should land + measure it.
3. The slice oracle's `tend_rw = 0.38 · buoyancy_face` magic number was traced to slice oracle line 108 but NOT to a MPAS source line — both the slice and the recurrence may be co-tuned rather than physics-grounded.
4. I did not extend the slice-oracle test to 600 s to confirm it would also decay — a follow-on safety test that would prevent future "slice passes ⇒ recurrence is physically correct" misreadings.

## Decision

**Decision:** `MIXED` (§7 verdict from `diagnostic-report.md`). The diagnostic exposes one real wiring bug (`diagnose_pressure_al_alt` erases the recurrence's density-derived `p_perturbation` and uses `theta_base` instead of `state.theta` against its own cited WRF source anchor) AND a genuine architectural insufficiency (the conservative MPAS-recurrence as scoped cannot sustain bubble lifting; the missing WRF small-step `t_2ave` / `ww` / `muave` / `_save` scratch matches the ADR-021 fallback hypothesis). The wiring fix is ~2-4 worker hours and should land regardless; the architectural answer is ADR-021 prototype (already dispatched in parallel as Plan B).

Recommendation to manager: dispatch the wiring-bug fix sprint (small) and let the ADR-021 prototype continue in parallel as the primary path to the warm-bubble target. Confidence HIGH; one explicit limitation noted in §8 of the diagnostic report.
