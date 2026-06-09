# Reviewer Report

Decision: ACCEPT_PARTIAL_PROOF_BLOCK_PATCH.

## Review

The proof is useful and correctly fail-closed. It identifies the dominant WRF
semantic path: for `USE_THETA_M=1`, WRF solve-time `grid%t_2` is moist
perturbation theta, and the live-nest path requires dry-to-moist theta
semantics plus `adjust_tempqv`. This explains nearly all of the prior
`5.49 K` residual.

The proof does not justify a production patch because the best candidate leaves
max_abs `0.00541785382188209 K`, above the prior `1e-3 K` material gate, and
because accepted same-boundary pre-call `QVAPOR` truth is missing.

## Evidence

- WRF `module_initialize_real.F:4918-4928` converts `grid%t_2` to moist theta
  when `use_theta_m=1`.
- WRF `mediation_integrate.F:726-762` calls `adjust_tempqv` after live-nest
  terrain/base blending.
- WRF `nest_init_utils.F:812-890` preserves RH and updates both `th` and `qv`.
- The proof's best candidate reduces `T_STATE` max_abs from
  `5.490173101425171` to `0.00541785382188209`.

## Required Follow-Up

Do not patch `src/gpuwrf` from this sprint alone. First add or reuse an accepted
same-boundary WRF pre-call `QVAPOR` savepoint, rerun the theta proof, and isolate
the remaining millikelvin residual.
