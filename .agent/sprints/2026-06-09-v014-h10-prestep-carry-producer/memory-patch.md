# Memory Patch

Scope:

Project-memory update for v0.14 same-state JAX-vs-WRF debugging after the
missing h10 pre-step JAX carry checkpoint was produced.

Evidence:

- `proofs/v014/jax_h10_prestep_carry_producer.json` proves a CPU-loadable
  checkpoint now exists at completed `d02` step 5999:
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`.
- The checkpoint contains `OperationalCarry`, paired `OperationalNamelist`,
  grid shape `159 x 66 x 44`, and SHA256
  `0896e4a272cbeaa85d1bb969ecae82b047e75a028df45a87ddab4f4572af8dde`.
- `proofs/v014/jax_h10_prestep_carry.json` now runs the pre-halo comparison and
  reports `JAX_MISMATCH_T`; first mismatch is `T` with max_abs
  `3.3545763228707983` and RMSE `1.0296598586362888`.
- `proofs/v014/wrf_same_state_marker_savepoint.json` and
  `proofs/v014/wrf_post_rk_refresh_localization.json` establish that the WRF
  target surface is green and that WRF history `T` is sourced from
  `grid%th_phy_m_t0`.

Proposed destination:

Create `.agent/memory/pending/2026-06-09-v014-h10-prestep-carry-produced.md`.
After independent review and the next T history/source-attribution sprint,
condense the durable lesson into `.agent/memory/stable/recurring-gotchas.md`:
same-surface dynamic debugging must distinguish JAX theta/history source mapping
from true dycore theta evolution before editing acoustic/RK operators.

Reviewer Status:

Pending. Do not apply to stable memory until the T history/source-attribution
sprint confirms whether `JAX_MISMATCH_T` is a source/cadence mapping issue or a
real numerical theta-update issue.
