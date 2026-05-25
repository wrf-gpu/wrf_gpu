# Worker Report - M6b Standalone vs Comparator Bisection

## objective

Localize why the reframe controlled comparator had step-1 parity `0.0` while the standalone 10 s operational probe produced NaNs, then restore a bounded standalone probe without regressing B6 or the controlled comparator.

## verdict

Correctness gates requested by the user are closed: 10 s probe bounded, B6 unchanged at `0.0`, and controlled step-1 unchanged at `0.0`.

Dispatch recommendation: `READY-FOR-M6b-HONEST-1H-V3-CORRECTNESS`, with the D2H caveat below tracked separately.

## harness diff summary

Stage 1 input audit found no state, namelist, or initial-carry signature difference for `20260521_18z_l3_24h_20260522T072630Z`; both harnesses fed identical `run_forecast_operational` inputs.

The actual divergence was in the operational wrapper after the reframe:

- The controlled comparator exercised the shared acoustic core and proved scratch-space parity.
- The standalone wrapper promoted acoustic `theta/mu` outputs directly into the physical forecast state before physics/boundary coupling.
- That direct physical-state promotion made the 10 s full-domain probe non-finite.

## named fixes

- `fixed-dt-duration-harness`: `m6b_carry_expansion_probe.py` now keeps `dt_s=10.0` and derives step count from duration, so 2/5/10 step bisection is real multi-step bisection rather than one stretched timestep.
- `shared-acoustic-core-call`: `_operational_acoustic_substep_core` now calls `acoustic_substep_core` directly, restoring controlled step-1 bitwise parity.
- `fixed-theta-offset`: `_theta_base_offset` now uses the WRF/Gen2 300 K perturbation offset consistently.
- `bounded-physical-projection`: `run_forecast_operational` keeps promoted acoustic scratch resident but does not promote unapproved acoustic `theta/mu` into the physical forecast state at the wrapper boundary.
- `available-third-gen2-run`: the old third default run has no local `wrfout_d02_*` files in this worktree, so the probe uses available run `20260524_18z_l3_24h_20260525T074709Z` as the third default.

## stage status

- Stage 1 input signatures: PASS, `proof_input_signatures.json`.
- Stage 2 standalone step-1 matches comparator: PASS, `final_max_abs_delta=0.0`, `proof_standalone_step1_matches.json`.
- Stage 3 multi-step bisection: PASS for 2, 5, 10 steps, all `final_max_abs_delta=0.0`, `proof_multi_step_divergence.json`.
- Stage 4 10 s bounded probe: PASS on three available Gen2 runs, `proof_10s_probe.json` and `proof_10s_bounded_after_fix.txt`.
- Stage 5 B6 + controlled parity: PASS, B6 max delta `0.0` in `proof_b6_unchanged.txt`; controlled step-1 `0.0` in reframe proof.
- Stage 6 D2H warmed: not newly closed here. Existing warmed summary in `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/proof_nsys_transfers_inside_loop.json` records `d2h_inter_kernel=20`, not zero.
- Stage 7 no regression: PASS, `173 passed in 426.69s`, `proof_no_regression.txt`.

## files changed

- `scripts/m6b_carry_expansion_probe.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `tests/test_m6b_standalone_matches_comparator.py`
- `.agent/sprints/2026-05-25-m6b-standalone-vs-comparator-bisect/*proof*`

Local-only note: `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/proof_warmed.nsys-rep` was copied from the warmed D2H worktree so the existing ignored-artifact test could run locally; it is ignored by `.gitignore` and not part of this sprint commit.

## commands run

- `python -m py_compile scripts/m6b_carry_expansion_probe.py src/gpuwrf/runtime/operational_mode.py`
- `python scripts/m6b_carry_expansion_probe.py --audit-inputs --duration-s 10 --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --gen2-ic-time 2026-05-21_18:00:00`
- `python scripts/m6b_carry_expansion_probe.py --compare-comparator --steps 1 --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --gen2-ic-time 2026-05-21_18:00:00`
- `python scripts/m6b_carry_expansion_probe.py --multi-step-bisect --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --gen2-ic-time 2026-05-21_18:00:00`
- `python scripts/m6b_carry_expansion_probe.py --runs 3 --duration-s 10`
- `python scripts/m6b_real_ic_operational_compare.py --steps 1 --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --gen2-ic-time 2026-05-21_18:00:00`
- `python scripts/m6b6_coupled_step_compare.py --tier golden`
- `pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_*.py -v`

## proof objects produced

- `proof_input_signatures.json`
- `proof_standalone_step1_matches.json`
- `proof_multi_step_divergence.json`
- `proof_10s_probe.json`
- `proof_10s_bounded_after_fix.txt`
- `proof_b6_unchanged.txt`
- `proof_no_regression.txt`

## unresolved risks

- This sprint restores bounded 10 s behavior by keeping acoustic `theta/mu` as promoted scratch rather than approved physical prognostics. That is correct for this acceptance gate but means physical `theta/mu` promotion still needs a separate savepoint-aligned design decision.
- D2H warmed zero is not proven here; existing warmed evidence still records nonzero inter-kernel D2H.
- The third original default run `20260523_18z_l3_24h_20260524T004313Z` is unavailable locally, so the three-run probe uses `20260524_18z_l3_24h_20260525T074709Z` as an available replacement.

## next decision needed

Dispatch M6b honest 1h V3 for correctness/bounds. Keep D2H-zero as a separate transfer-cleanliness fix gate unless the manager explicitly wants to block V3 on existing `d2h_inter_kernel=20`.
