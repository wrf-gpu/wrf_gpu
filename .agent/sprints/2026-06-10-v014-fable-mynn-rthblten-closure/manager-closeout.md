# Manager Closeout

Merge Decision: ACCEPT_AND_COMMIT_AS_FORMAL_BOUND.

The sprint achieved the fallback endpoint: it did not make strict Step-1 green,
but it narrowed the blocker beyond "MYNN-EDMF RTHBLTEN" and corrected the
roadmap direction. The field-dominant lane is now RRTMG `RTHRATEN`; MYNN is a
bounded worst-cell/max floor under the current non-bitwise JAX implementation.

Accepted evidence:

- `proofs/v014/mynn_rthblten_step1_closure.*`
- `.agent/reviews/2026-06-10-v014-fable-mynn-rthblten-closure.md`
- refreshed `proofs/v014/noahmp_step1_closure.*`

Manager decision:

- Do not relax the strict gate silently.
- Do not start TOST or Switzerland-GPU yet.
- Next sprint should close or formally bound RRTMG clear-sky `RTHRATEN`.
- After RRTMG, record an explicit reviewed tolerance-policy decision for the
  non-bitwise MYNN/RRTMG mass-coupled Step-1 gate.

Residual risk: the current strict gate remains red at max_abs `53.52301833555157`
and RMSE `2.5444971494115354`; v0.14 release validation remains blocked.
