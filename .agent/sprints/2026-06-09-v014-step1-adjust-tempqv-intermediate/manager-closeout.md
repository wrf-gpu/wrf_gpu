# Manager Closeout: V0.14 Step-1 Adjust-TempQV Intermediate Truth

Date: 2026-06-09 17:16 WEST

## Outcome

The sprint is closed as a successful proof sprint with verdict
`STEP1_ADJUST_TEMPQV_INTERMEDIATE_PRESSURE_INPUT_MISMATCH`.

The earlier Codex-sandbox OpenMPI/PMIx blocker was resolved by a manager
unsandboxed rerun. WRF emitted the requested one-cell `adjust_tempqv`
intermediate file, and the proof now compares WRF internals against the prior
JAX theta/QVAPOR proof values.

## Proof Objects

- `proofs/v014/step1_adjust_tempqv_intermediate.py`
- `proofs/v014/step1_adjust_tempqv_intermediate.json`
- `proofs/v014/step1_adjust_tempqv_intermediate.md`
- `proofs/v014/step1_adjust_tempqv_intermediate_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-step1-adjust-tempqv-intermediate.md`
- `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/wrf_truth/adjust_tempqv_d2_i18_j10_k2.txt`
- `/mnt/data/wrf_gpu2/v014_step1_adjust_tempqv_intermediate/logs/wrf_run_mpirun_np28_manager.log`

Key comparison facts:

- `p`, `mub_save`, `c3h`, `c4h`, and `p_top` match exactly or within
  negligible precision.
- `mub` differs by `17.67503987130476 Pa`.
- `pb_new_equiv` and `p_new` differ by `17.49400702366256 Pa`.
- `t_2_post` remains different by `0.00541785382188209 K`.

## Merge Decision:

Commit and push proof/review/sprint documentation only. Do not patch production
model source from this sprint. The evidence points to a current live-nest
pressure/base-input mismatch, so the next sprint must split `mub`/`pb_new` at
the WRF/JAX live-nest base boundary before any source-changing fix.

## Scope Changes

The worker initially closed fail-blocked on a PMIx sandbox error. The manager
reran the exact WRF truth capture outside the sandbox and patched the proof
script to read `manager_log=` from the status file so the final artifact points
at the successful run.

## Lessons

The thermodynamic formula lane should not be pursued further until the current
`mub`/`pb_new` input mismatch is split. Saved-state inputs match, so the likely
surface is the current live-nest terrain/base blend or the JAX reconstruction
used by the candidate proof.

## Next Sprint

Open `v014-step1-current-mub-base-input-split`: CPU-only, no GPU, no TOST, no
Switzerland, no FP32 source work. It must identify whether the `17.5 Pa`
current-`mub`/`pb_new` mismatch is a WRF hook boundary issue, a JAX
live-nest-base reconstruction bug, or a production source patch candidate.
