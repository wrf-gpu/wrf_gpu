# Reviewer Report

Decision: ACCEPT_MISSING_SAVEPOINT_SPEC.

## Review

The proof correctly blocks a premature production theta patch. It establishes
that existing QVAPOR truth artifacts are not at the required
`before_first_rk_step_part1_call` boundary. The generated savepoint spec is
minimal: extend the existing pre-call hook to emit `QVAPOR` from
`moist(i,k,j,P_QV)` while preserving the existing field schema.

## Evidence

- Accepted pre-call text schema has `T_STATE/P_STATE/PB/MU_STATE/MUB/MUT` but
  no `QVAPOR`.
- Existing QVAPOR-bearing Step-1 truth is classified as
  `post_after_all_rk_steps_pre_halo` / RK4 or as the promoted NPZ from that
  same boundary.
- The proposed savepoint retains existing pre-call fields and adds one mass-grid
  `QVAPOR` field with shape `[44,66,159]` after assembly.

## Required Follow-Up

Run the savepoint sprint, then rerun the theta semantics proof with
same-boundary `T_STATE` and `QVAPOR`. Do not reuse post-RK QVAPOR for the
pre-call proof.
