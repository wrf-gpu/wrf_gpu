# Sprint Contract: V0.14 Dry Source-Leaf Fix

Date: 2026-06-10 00:20 WEST
Owner: GPT-5.5 xhigh worker in tmux
Manager: `worker/gpt/v013-close-manager`

## Objective

Implement or conclusively block the next narrow v0.14 grid-parity fix:
true WRF dry physics source leaves for active `RTHRATEN` and `RTHBLTEN` before
`_augment_large_step_tendencies`.

Accepted prior proof:

- `proofs/v014/step1_part2_source_leaves_split.{py,json,md}`
- verdict:
  `STEP1_PART2_SOURCE_LEAVES_LOCALIZED_UPDATE_PHY_TEN_RAW_RTH_TO_T_TENDF_MISSING_IN_JAX_DRY_BUNDLE`

Current facts:

- WRF `update_phy_ten` explains the Step-1 `T_TENDF` surface exactly as
  `pre + active RTH`, nested-interior max_abs `0.0`.
- WRF `conv_t_tendf_to_moist` closes to roundoff, max_abs
  `0.00016236981809925055`.
- Current patched-init JAX dry `T_TENDF` remains divergent: max_abs
  `2457.5830078125`, RMSE `21.674279301376934`.
- Active WRF leaves in the proof case are `RTHRATEN` and `RTHBLTEN`;
  `RTHBLTEN` is the dominant active raw leaf.
- Aggregate post-physics state delta is rejected as a narrow replacement.
- Inactive WRF `RTH*TEN` leaves can contain uninitialized junk and must not be
  used causally.

## Required Work

Use the fastest rigorous CPU-only path. This is an implementation sprint, but
the endpoint is proof, not a guessed patch.

1. Read the prior proof and relevant source:
   - `proofs/v014/step1_part2_source_leaves_split.{py,json,md}`;
   - `src/gpuwrf/runtime/operational_mode.py`;
   - `src/gpuwrf/dynamics/core/rk_addtend_dry.py`;
   - PBL/radiation adapters that can expose `RTHBLTEN`/`RTHRATEN`.
2. Implement a production path that feeds WRF-compatible raw source leaves into
   `DryPhysicsTendencies.t_tendf` before `_augment_large_step_tendencies`.
3. Do not solve this by reusing aggregate state deltas unless you first produce
   a scheme-level raw-leaf proof that makes that mathematically equivalent for
   this gate.
4. If exact implementation is blocked, return a precise blocker with the next
   source boundary and the shortest proof/fix route. Do not silently weaken the
   v0.14 goal.
5. Preserve GPU performance architecture: no timestep-loop host/device
   transfer, no CPU-WRF dependency in production, no broad dycore rewrite, no
   extra materialized full-domain temporaries unless measured/justified.

## File Ownership

Allowed production files:

- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/scan_adapters.py`
- `src/gpuwrf/physics/**` only if a scheme adapter must expose raw
  `RTHBLTEN`/related leaves.

Avoid editing `src/gpuwrf/dynamics/core/**` unless you prove the contract in
`rk_addtend_dry` is wrong. Do not edit unrelated memory/FP32/GPU-run code.

Allowed proof/test/docs files:

- `proofs/v014/step1_dry_source_leaf_fix.py`
- `proofs/v014/step1_dry_source_leaf_fix.json`
- `proofs/v014/step1_dry_source_leaf_fix.md`
- updates to `proofs/v014/step1_part2_source_leaves_split.{py,json,md}` if the
  same proof is reused post-fix;
- `.agent/reviews/2026-06-10-v014-dry-source-leaf-fix.md`;
- focused tests under `tests/`.

## Acceptance Gates

Required, manager-rerunnable:

```bash
python -m py_compile proofs/v014/step1_dry_source_leaf_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_dry_source_leaf_fix.py
python -m json.tool proofs/v014/step1_dry_source_leaf_fix.json >/tmp/step1_dry_source_leaf_fix.validated.json
python -m py_compile proofs/v014/step1_part2_source_leaves_split.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/step1_part2_source_leaves_split.py
python -m json.tool proofs/v014/step1_part2_source_leaves_split.json >/tmp/step1_part2_source_leaves_split.validated.json
git diff --check
```

If production code is changed, add at least one focused unit/regression test for
the new source-leaf plumbing and run it CPU-only.

Pass target:

- Step-1 `after_conv_t_tendf_to_moist` versus current JAX dry `T_TENDF` should
  collapse from max_abs `2457.5830078125` to near-zero. Use the prior WRF
  formula tolerance as guidance: target nested-interior max_abs `<= 1.0e-3` and
  RMSE `<= 1.0e-5`, or provide a WRF-anchored reason for any larger residual.
- No production host/device transfer in timestep loops.
- No GPU/TOST/Switzerland required for this sprint.

## Completion Marker

Print exactly:

`GPT DRY_SOURCE_LEAF_FIX DONE - see proofs/v014/step1_dry_source_leaf_fix.md`
