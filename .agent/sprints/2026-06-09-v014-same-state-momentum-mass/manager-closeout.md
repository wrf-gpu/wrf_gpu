# Manager Closeout

## Outcome

The sprint is accepted as a CPU-only localization proof. It does not fix the
model, but it narrows the current v0.14 grid-divergence search: the selected h10
state already mismatches WRF at `post_after_all_rk_steps_pre_halo`.

## Proof Objects

- `proofs/v014/same_state_momentum_mass.json`
- `proofs/v014/same_state_momentum_mass.md`
- `.agent/reviews/2026-06-09-v014-same-state-momentum-mass.md`

Key result:

- Verdict: `JAX_MISMATCH_U_post_after_all_rk_steps_pre_halo`
- First failing field: `U`
- Max abs: `6.292358893898424`
- RMSE: `2.032497018496295`
- Worst native key: `[4, 13]`
- JAX vs WRF: `-4.735481996086533` vs `1.55687689781189`

## Merge Decision:

Merge the proof artifacts and closeout. Do not resume TOST on this basis. The
next priority remains grid-field/root-cause debug, not release packaging.

## Scope Changes

No production code was changed. The worker added one proof-only environment
guard, `JAX_ENABLE_COMPILATION_CACHE=false`, to keep CPU proof reruns free of
stale persistent-cache warning floods.

## Lessons

The failure is now closer to the source than station TOST or wrfout comparison:
post-RK/pre-halo dynamic state itself fails. This supports the principal's
grid-first priority and shifts the next debug step inward to final RK
momentum/mass/theta-pressure assembly.

The live-nest base-source fix remains important but partial. Because this carry
predates that fix, base-field residuals should not be over-interpreted until a
fresh h10 carry is regenerated on current code.

## Next Sprint

After the current direct grid-after-base GPU run finishes, use the two GPT
debug sprint outputs together. If the grid proof does not show closure and the
same-state result remains inconclusive after the fresh carry, follow the manager
cadence and dispatch one Opus xhigh critic/debugger before committing to the
next root-cause conclusion.
