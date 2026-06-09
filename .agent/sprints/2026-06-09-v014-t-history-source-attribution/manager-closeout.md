# Manager Closeout

Merge Decision: accept and land the attribution proof.

Objective:

Determine whether `JAX_MISMATCH_T` is a JAX history/source/cadence mapping
issue or a true theta-evolution mismatch.

Accepted verdict:

`T_EVOLUTION_MISMATCH_CONFIRMED`.

Accepted evidence:

- `proofs/v014/jax_t_history_source_attribution.py`
- `proofs/v014/jax_t_history_source_attribution.json`
- `proofs/v014/jax_t_history_source_attribution.md`
- `.agent/reviews/2026-06-09-v014-t-history-source-attribution.md`

Manager validation:

- Python compilation.
- CPU-only proof rerun against the produced h10 carry checkpoint.
- JSON validation.
- `git diff --check`.

Roadmap effect:

The source-mapping branch is closed for the first h10 `T` mismatch. No inspected
JAX theta/history candidate matches WRF history `T_HIST_SRC` or WRF `T_THM`.
The next sprint is theta-evolution localization: stage/cadence/component
attribution from prestep carry through physics forcing, RK stage inputs,
theta tendency/advection/diffusion/radiation folding, acoustic finish, and
post-RK refresh.

Next decision:

Open a read-only theta-evolution localization sprint. Production dycore edits
remain forbidden until that sprint names a narrower failing stage/operator or
state-update boundary.
