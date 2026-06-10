# Memory Patch

Reviewer Status: ACCEPT.

Durable memory:

- MYNN-EDMF `RTHBLTEN` is no longer the field-dominant strict Step-1 blocker.
  The accepted proof is `proofs/v014/mynn_rthblten_step1_closure.*`.
- The strict operational dry `T_TENDF` source leaf is exactly reassembled from
  mass/theta_m-coupled `RTHRATEN + RTHBLTEN` plus QV coupling; reassembly-vs-runtime
  max_abs is `4.55e-13`.
- RRTMG `RTHRATEN` is field-dominant: substituting WRF `RTHRATEN` collapses RMSE
  `2.5378 -> 0.5433` and p99 `16.63 -> 0.84`.
- MYNN remains a bounded worst-cell/max floor. Do not spend another broad MYNN
  sprint unless new evidence contradicts this proof.
- Do not start TOST/Switzerland-GPU or silently change tolerances before RRTMG
  is closed/bounded and a reviewed gate-policy decision is recorded.
