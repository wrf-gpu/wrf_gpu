# GATE-2 Fix Report

## Verdict

PASS for the release gate. F1 default path was not inert. With the fix, v0.23 default
with `GPUWRF_BATCH_ENSEMBLE` unset is field-identical to the fresh v0.22.2
reference when using the v0.22.2 warm AOT cache:

- Before fix: `max_abs_global = 0.43773239459551405`, worst `d01/0005/5`, `50/178` leaves differ.
- After fix: `max_abs_global = 0.0`, `0/178` leaves differ.

## Localization

Captured F1-only commit `80287687` with:

- `PYTHONPATH=<USER_HOME>/src/wrf_gpu2_wt/v023-f1/src`
- `CANARY_DUMP_STATE=1`
- `CANARY_FIXED_REPEATS=2`
- `CANARY_MAXDOM=3`
- `CANARY_FIXED_ROOT_STEPS=2`
- `GPUWRF_BATCH_ENSEMBLE` unset
- GPU lock: `scripts/with_gpu_lock.sh --label v023-gate2 -- ...`

Field compare vs fresh v0.22.2 `53bd20bf`:

- Proof: `proofs/v023/gate2_fix/F1_VS_V0222_FIELD_COMPARE.json`
- Result: `max_abs_global = 0.43773239459551405`
- Worst field: `d01/0005/5`
- Mismatched leaves: `50/178`

This confirmed F1 `80287687` as the first real default-path divergence. P-bundle
bisect was not needed.

## Fix

The fixed release default bypasses F1 batched/vmap orchestration when
`GPUWRF_BATCH_ENSEMBLE` is unset or `1`:

- Removed F1 batched/vmap execution from the production `DomainTree` runtime path.
- Removed the `DomainTree.batch_namelists` default-path hook.
- Restored unbatched `build_clock_base`, `_operational_force`, and fused cascade construction.
- Moved the temporary `GPUWRF_BATCH_ENSEMBLE` gate to `nested_pipeline.py` so `B>1` fails closed without changing the fused runtime source fingerprint.
- Reverted traced helper changes that shifted the source/AOT fingerprint without being part of the release default.

No tolerances were widened. No masking, clamps, or `nan_to_num` were added.

## Verification

Primary release gate:

- Proof: `proofs/v023/gate2_fix/FIXED2_REFCACHE_V023_VS_V0222_FIELD_COMPARE.json`
- Candidate capture: `proofs/v023/gate2_fix/fixed_capture2_refcache/v023_fixed2_refcache_gate2_summary.json`
- Reference: `<USER_HOME>/src/wrf_gpu2_wt/v023-refcheck/proofs/v023/gpu_gates/canary_state/v0222_cap_state.npz`
- Result: `exact = true`, `max_abs_global = 0.0`, `mismatched_leaf_count = 0/178`
- Candidate summary: `all_finite = true`, warm `s_per_step = 13.7422495`

Reference stability check:

- Proof: `proofs/v023/gate2_fix/ref_recap/v0222_recap_gate2_summary.json`
- Result: recaptured v0.22.2 digest equals baseline digest `8b5b4405699d0f67ac46a4c8977d201482cdbce3635417cbee944e67387345ac`.

Fresh-cache sanity:

- Proof: `proofs/v023/gate2_fix/FIXED3_FRESH_V023_VS_V0222_FIELD_COMPARE.json`
- Result vs baseline: `max_abs_global = 1.1976226232945919e-08`, worst `d03/0005/5`.
- Same-cache control proof: `proofs/v023/gate2_fix/FIXED3_VS_V0222_SAMECACHE_FIELD_COMPARE.json`
- Same-cache control result: `exact = true`, `max_abs_global = 0.0`, `mismatched_leaf_count = 0/178`.

Interpretation: the release default code path is byte-identical to v0.22.2. The
fresh-cache-only `1.2e-08` delta is reproduced by v0.22.2 itself when compared
against the older baseline cache and disappears when v0.22.2 and fixed v0.23 use
the same freshly compiled AOT blobs.

## Commands Run

- `scripts/with_gpu_lock.sh --label v023-gate2 -- ... CANARY_TAG=f1_80287687_gate2 python -u proofs/v023/gpu_gates/canary_state_capture.py`
- `python proofs/v023/gpu_gates/canary_state_compare.py --output proofs/v023/gate2_fix/F1_VS_V0222_FIELD_COMPARE.json`
- `scripts/with_gpu_lock.sh --label v023-gate2 -- ... CANARY_TAG=v023_fixed2_refcache_gate2 python -u proofs/v023/gpu_gates/canary_state_capture.py`
- `python proofs/v023/gpu_gates/canary_state_compare.py --output proofs/v023/gate2_fix/FIXED2_REFCACHE_V023_VS_V0222_FIELD_COMPARE.json`
- `scripts/with_gpu_lock.sh --label v023-gate2 -- ... CANARY_TAG=v023_fixed3_fresh_gate2 python -u proofs/v023/gpu_gates/canary_state_capture.py`
- `scripts/with_gpu_lock.sh --label v023-gate2 -- ... CANARY_TAG=v0222_fixed3cache_gate2 python -u proofs/v023/gpu_gates/canary_state_capture.py`
- `python proofs/v023/gpu_gates/canary_state_compare.py --output proofs/v023/gate2_fix/FIXED3_VS_V0222_SAMECACHE_FIELD_COMPARE.json`
- `python -m py_compile src/gpuwrf/runtime/domain_tree.py src/gpuwrf/integration/nested_pipeline.py src/gpuwrf/runtime/finite_state_guard.py proofs/v023/batched_minigrid/f1_cpu_b2_gate.py`
- `PYTHONPATH=src python - <<'PY' ... _batch_ensemble_size_from_env fail-closed check ... PY`

## Unresolved Risks

- F1 `B>1` remains disabled on the production nested pipeline until it is re-landed
  behind a proven default-inert implementation.
- Large NPZ/cache proof artifacts remain in the local worktree for audit but are
  not intended for the git commit.
