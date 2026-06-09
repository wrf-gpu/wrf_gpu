# v0.14 Empirical/Static Memory Map

- Verdict: `NO_REMAINING_NON_RADIATION_MEMORY_FIX_SHOULD_BLOCK_LONG_VALIDATION_AFTER_GRID_PARITY`
- Branch: `worker/gpt/v013-close-manager`
- HEAD: `cdbfb05c4f922365ca0617fb68e3757a66f235cf`
- Dirty worktree: `True`
- GPU/TOST/Switzerland/FP32 source work: not run

## Decision

RRTMG column/band/optics tiling remains prior fixed evidence. On the exact current branch, the remaining non-radiation memory items are not blockers for the first long validation after grid-cell parity closes. Run the short exact-branch memory preflight again for the selected long-run configuration, but do not hold validation for new broad memory rewrites.

Smallest safe memory-only source sprint: `WDM6 slmsk shape-only cleanup, preserving current values and proving exact WDM6 output equality`.
Only material bit-identical cleanup: `moisture transport velocity reuse when active moisture advection matters`.

## Ranked Map

| Rank | Candidate | Recommendation | Target fp64 estimate | Blocks long validation? |
|---:|---|---|---:|---|
| 1 | `wdm6_slmsk_full_column_broadcast` | `FIX_NOW_BIT_IDENTICAL` | 0.075119 GiB recoverable | `False` |
| 2 | `moisture_advection_duplicate_transport_velocity` | `FIX_NOW_BIT_IDENTICAL` | 0.237621-0.620881 GiB | `False` |
| 3 | `mynn_boulac_dense_column_pair_matrices` | `MEASURE_FIRST` | 3.832597 GiB per dense matrix | `False` |
| 4 | `non_radiation_column_physics_tiling` | `MEASURE_FIRST` | see JSON | `False` |
| 5 | `post_physics_non_dry_sparse_donated_merge` | `MEASURE_FIRST` | 1.303084-2.606168 GiB | `False` |
| 6 | `moisture_limiter_and_species_workspace` | `MEASURE_FIRST` | 0.459912 GiB outputs; limiter higher | `False` |
| 7 | `pbl_surface_bottom_only_prep_and_duplicate_diagnostics` | `DEFER_SEMANTIC_OR_DYCORE` | see JSON | `False` |
| 8 | `acoustic_scan_carry_split_evolving_only` | `DO_NOT_DO_BEFORE_GRID_PARITY` | see JSON | `False` |
| 9 | `state_total_perturbation_base_alias_reduction` | `DEFER_SEMANTIC_OR_DYCORE` | see JSON | `False` |
| 10 | `small_dycore_masks_and_pad_helpers` | `NOT_WORTH_STANDALONE` | see JSON | `False` |

## Proof Notes

- `FIX_NOW_BIT_IDENTICAL` here means safe as a post-grid-parity memory-only sprint with exact-output proof, not a reason to interrupt grid parity.
- `MEASURE_FIRST` items have plausible GiB-scale upside but need HLO/RSS or short GPU peak evidence before source work.
- `DEFER_SEMANTIC_OR_DYCORE` and `DO_NOT_DO_BEFORE_GRID_PARITY` items touch PBL/dycore semantics or previous reverted acoustic work.
- Detailed array formulas, source-pattern checks, proof references, and proof gates are in `proofs/v014/empirical_memory_map.json`.
