# V0.14 Step-1 Current-MUB/Base-Input Split

Verdict: `STEP1_CURRENT_MUB_BASE_SPLIT_WRF_BLEND_UNIMPLEMENTED_OR_MISMATCHED`.

## Result

- CPU-only proof; GPU used: `False`.
- Required ancestor `9a7016d9` present: `True`.
- Fresh scratch WRF hook not run here: `WRITABLE_NOT_USED`.
- Recovered WRF adjust hook: `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth/adjust_tempqv_d2_i18_j10_k2.txt`.
- Target zero `{'k': 1, 'y': 9, 'x': 17}`, Fortran `{'i': 18, 'j': 10, 'k': 2}`.

## Explanation

- WRF copies `nest%mub` to `nest%mub_save`, blends `nest%mub_fine` into current `nest%mub`, then calls `adjust_tempqv`.
- Later, WRF calls `start_domain(nest,.TRUE.)`, which recomputes the final base fields used by the pre-part1 truth.
- The JAX theta proof used that final post-`start_domain` base MUB for `adjust_tempqv`; WRF uses the transient post-blend/pre-`start_domain` MUB.

## Target Values

| Surface | MUB | PB/current base | p_new |
|---|---:|---:|---:|
| WRF `adjust_tempqv` hook | 86812.250000000000 | 90936.031250000000 | 92686.187500000000 |
| JAX theta proof final base | 86794.574960128695 | 90918.537242976337 | 92668.693492976337 |
| JAX direct WRF MUB blend | 86812.250452109511 | 90936.029408214526 | 92686.185658214526 |

## Comparisons

| Comparison | Field | Delta |
|---|---|---:|
| WRF adjust_tempqv current MUB minus JAX theta-proof final MUB | `mub` | 1.7675039871304762e+01 |
| WRF adjust_tempqv current MUB minus JAX direct WRF blend MUB | `mub` | -4.5210951066110283e-04 |
| WRF pre-part1 final MUB minus JAX theta-proof final MUB | `mub` | -4.6476286952383816e-03 |
| WRF adjust_tempqv current MUB minus WRF pre-part1 final MUB | `mub` | 1.7679687500000000e+01 |
| WRF p_new minus JAX theta-proof p_new | `p_new` | 1.7494007023662562e+01 |
| WRF pb_new_equiv minus JAX theta-proof live_pb | `pb_new_equiv` | 1.7494007023662562e+01 |

## Formula Check

- WRF source formula: `p_new = p + c4h + c3h*mub + p_top`.
- WRF source formula minus hook `p_new`: `-0.00228920578956604` Pa.
- Requested grouped formula: `p_new = p + c3h*(mub+p_top) + c4h`.
- Requested grouped formula minus hook `p_new`: `-51.86131039261818` Pa.
- Therefore the grouped `c3h*(mub+p_top)` form is not the WRF `adjust_tempqv` formula for this hook.

## Handoff

objective: explain the current-MUB/base-input mismatch driving the Step-1 live-nest theta residual.

files changed:
- `proofs/v014/step1_current_mub_base_input_split.py`
- `proofs/v014/step1_current_mub_base_input_split.json`
- `proofs/v014/step1_current_mub_base_input_split.md`
- `proofs/v014/step1_current_mub_base_input_split_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-current-mub-base-input-split.md`

commands run:
- `python -m py_compile proofs/v014/step1_current_mub_base_input_split.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_current_mub_base_input_split.py`
- `python -m json.tool proofs/v014/step1_current_mub_base_input_split.json >/tmp/step1_current_mub_base_input_split.validated.json`
- `git diff -- src/gpuwrf`

proof objects produced:
- `/home/user/src/wrf_gpu2/proofs/v014/step1_current_mub_base_input_split.json`
- `/home/user/src/wrf_gpu2/proofs/v014/step1_current_mub_base_input_split.md`
- `/home/user/src/wrf_gpu2/proofs/v014/step1_current_mub_base_input_split_wrf_patch.diff`
- `/home/user/src/wrf_gpu2/.agent/reviews/2026-06-09-v014-step1-current-mub-base-input-split.md`

unresolved risks:
- Fresh WRF terrain/PHB target emission could not be run because /mnt/data scratch writes are read-only in this sandbox.
- The source-changing sprint should validate the transient MUB blend over the full domain before patching production initialization.

next decision needed: Open the smallest source-changing sprint to add a transient live-nest adjust base path: compute WRF post-blend/pre-start_domain MUB for adjust_tempqv, use it only for theta/QV adjustment, keep final BaseState from start_domain, and rerun the Step-1 theta proof.
