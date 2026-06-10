# Manager Closeout

Date: 2026-06-10 03:39 WEST
Manager: Codex
Worker: GPT-5.5 xhigh in tmux `0:3`

## Outcome

Closed as a valid local correctness fix and a successful narrowing sprint, not
as full Step-1 parity closure.

Verdict:

`TSK_ZNT_SOURCE_FIXED_NEXT_BLOCKER_THERMODYNAMIC_COLUMN_INPUTS`

The sprint fixed missing WRF cold-start `ZNT/MAVAIL` sourcing from
`LANDUSE.TBL` by `LU_INDEX`, proved `TSK/ZNT/MAVAIL` parity at the exact
`sfclay_mynn` hook, and identified the next boundary as the non-surface
thermodynamic column inputs.

## Proof Objects

- `proofs/v014/step1_tsk_znt_sourcing_fix.py`
- `proofs/v014/step1_tsk_znt_sourcing_fix.json`
- `proofs/v014/step1_tsk_znt_sourcing_fix.md`
- `proofs/v014/step1_tsk_znt_sourcing_fix_wrf_patch.diff`
- `.agent/reviews/2026-06-10-v014-step1-tsk-znt-sourcing.md`

## Manager-Rerun Evidence

- Focused tests: `4 passed, 1 skipped`.
- `step1_tsk_znt_sourcing_fix.py` reran successfully.
- `step1_source_fidelity_closure.py` reran with the thermodynamic-column blocker
  label.
- `mynn_driver_source_output_fix.py` reran and kept the MYNN kernel/init proof.
- JSON validation passed for the primary and refreshed proof artifacts.
- `git diff --check` passed.

## Merge Decision:

Merge. This is a WRF-sourced, performance-compatible lower-boundary fix. It
does not justify TOST, Switzerland, broad FP32, or long GPU validation yet.

## Key Numbers

- `TSK` input max_abs `0.0 K`.
- `ZNT` input max_abs `1.1920928910669204e-08 m`.
- `MAVAIL` input max_abs `1.1920928966180355e-08`.
- strict after-conv `T_TENDF` max_abs `1497.6112467075195`, RMSE
  `13.252694871222973`.
- next blocker: `th_phy(kts)` max_abs `5.490148027499686 K`, derived
  `t_phy(kts)` max_abs `5.521345498302992 K`, `p_phy(kts)` max_abs
  `292.8203125 Pa`.

## Next Sprint

Open a GPT-5.5 xhigh thermodynamic-column sprint:

- compare exact WRF `sfclay_mynn` hook inputs `th_phy/t_phy/p_phy/dz8w`;
- compare against JAX `_surface_column_view`;
- fix Step-1 temperature/pressure sourcing if local;
- rerun strict Step-1 and source-fidelity proofs.
