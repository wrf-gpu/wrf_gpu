# Pending Memory: V0.14 Same-Input Single-RK Parity

Status: pending promotion after the full pre-RK native-state/tendency hook and
actual same-input comparison complete.

Lesson:

- The requested WRF pre-RK input -> one JAX RK step -> WRF post-RK/pre-halo
  proof is currently blocked by missing instrumentation, not by a proven model
  source defect.
- The available WRF pre-RK savepoint emits only `MASS_K1` fields:
  `T_THM`, `T_OLD`, `T_HIST_SRC`, `P`, `PB`, `MU_NEW`, `MU_OLD`, and `MUB`.
- Do not infer upstream drift, final-RK PGF/mass-wind, or theta/source causality
  from this blocked proof.
- The next exact task is to add a full WRF pre-RK native-state plus RK-fixed
  tendency/source hook and a proof-only JAX `OperationalCarry` loader, then rerun
  the same-input single-RK boundary on halo-valid cells.

Evidence:

- `proofs/v014/same_input_single_rk_parity.json`
- `proofs/v014/same_input_single_rk_parity.md`
- `.agent/reviews/2026-06-09-v014-same-input-single-rk-parity.md`
- `.agent/sprints/2026-06-09-v014-same-input-single-rk-parity/manager-closeout.md`
