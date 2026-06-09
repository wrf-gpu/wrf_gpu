# Manager Closeout

Date: 2026-06-10 00:41 WEST
Manager: Codex
Worker: GPT-5.5 xhigh in tmux `0:3`

## Outcome

Closed as a valid blocked implementation sprint, not as a fix.

The worker implemented narrow WRF-style source-leaf plumbing for MYNN and
`rad_rk_tendf=1`, and the path is active under proof. The strict Step-1
`T_TENDF` residual did not collapse, so the grid-parity blocker remains open.

Accepted verdict:

`DRY_SOURCE_LEAF_PLUMBING_ACTIVE_BUT_STEP1_T_TENDF_NOT_CLOSED`

## Proof Objects

- `proofs/v014/step1_dry_source_leaf_fix.py`
- `proofs/v014/step1_dry_source_leaf_fix.json`
- `proofs/v014/step1_dry_source_leaf_fix.md`
- `.agent/reviews/2026-06-10-v014-dry-source-leaf-fix.md`
- refreshed `proofs/v014/step1_part2_source_leaves_split.json`

Key proof numbers:

- Patched JAX dry `T_TENDF` is nonzero: max_abs `260.83156991819124`.
- WRF top active leaf remains much larger: `RTHBLTEN` max_abs
  `2522.90576171875`.
- Step-1 after-conv residual remains max_abs `2457.575215120763`, RMSE
  `21.445918959761645`.
- Forcing radiation only moves max_abs to `2454.161554535577`, so radiation
  cadence is secondary to MYNN source fidelity.
- WRF `conv_t_tendf_to_moist` contributes a further max_abs
  `224.50967407226562`, RMSE `4.572429855170764`.

## Commands Run

Worker and manager both ran the required gates:

- `python -m py_compile proofs/v014/step1_dry_source_leaf_fix.py proofs/v014/step1_part2_source_leaves_split.py tests/test_v014_dry_source_leaf_wiring.py src/gpuwrf/coupling/physics_couplers.py src/gpuwrf/runtime/operational_mode.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_dry_source_leaf_wiring.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_dry_source_leaf_fix.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_part2_source_leaves_split.py`
- `python -m json.tool proofs/v014/step1_dry_source_leaf_fix.json`
- `python -m json.tool proofs/v014/step1_part2_source_leaves_split.json`
- `git diff --check`

## Merge Decision:

Merge the proof artifacts and the narrow source-leaf plumbing as an intermediate
boundary, not as a release fix. The default `rad_rk_tendf=0` path remains
intended to compile to the prior operational behavior; the changed
`rad_rk_tendf=1` path is explicitly still not WRF-equivalent.

The next sprint must not claim closure from this commit. It must close or
further split the three ranked blockers below.

## Unresolved Risks

Ranked blockers from the proof:

1. JAX MYNN `RTHBLTEN` source is not WRF-compatible at this Step-1 boundary.
2. Held JAX `rthraten` is zero or stale at Step 1 while WRF has active
   `RTHRATEN`; forced radiation shows this is secondary.
3. WRF `conv_t_tendf_to_moist` and its `QV_TEND` term are missing from the JAX
   dry source bundle.

## Next Sprint

Open one coherent GPT-5.5 xhigh sprint, not Fable/Mythos yet:

Split MYNN PBL adapter/kernel inputs and outputs against WRF
`RTHBLTEN/RQVBLTEN`, seed or refresh held `RTHRATEN` at the same Step-1
boundary, implement WRF `conv_t_tendf_to_moist` before
`DryPhysicsTendencies.t_tendf`, and rerun the strict Step-1 proof. The sprint
should be allowed to fix all three source-fidelity gaps if the evidence stays
local and performance-compatible.
