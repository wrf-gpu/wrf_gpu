# Worker Report

Summary: Fable/Mythos reconciled the apparent MYNN-EDMF `RTHBLTEN` blocker with
the earlier same-input MYNN green proof. The sprint produced a formal bound, not
a production fix and not a green strict Step-1 gate.

Files changed:

- `proofs/v014/mynn_rthblten_step1_closure.{py,json,md}` (new)
- `proofs/v014/noahmp_step1_closure.{py,json,md}` (refreshed cross-reference)
- `.agent/reviews/2026-06-10-v014-fable-mynn-rthblten-closure.md`

Proof summary:

- Operational dry `T_TENDF` reassembly matches runtime to max_abs
  `4.547473508864641e-13`.
- Current strict runtime residual remains max_abs `53.52301833555157`, RMSE
  `2.5444971494115354`, p99 `16.631650419560028`.
- Replacing JAX `RTHRATEN` with WRF `RTHRATEN` collapses RMSE to
  `0.5433421347190945` and p99 to `0.8430227515735698`.
- MYNN remains relevant to worst-cell max, but RRTMG `RTHRATEN` is the
  field-dominant lane.

No production code changed. TOST/Switzerland-GPU remain blocked.
