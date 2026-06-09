# Manager Closeout

Merge Decision: accept and land the localization proof.

Objective:

Localize the confirmed h10 `T`/theta evolution mismatch to the narrowest
reachable JAX stage/cadence/component boundary before any production source
fix.

Accepted verdict:

`THETA_MISMATCH_PRESTEP_OR_INPUT`.

Accepted evidence:

- `proofs/v014/jax_theta_evolution_localization.py`
- `proofs/v014/jax_theta_evolution_localization.json`
- `proofs/v014/jax_theta_evolution_localization.md`
- `.agent/reviews/2026-06-09-v014-theta-evolution-localization.md`

Manager validation:

- Python compilation.
- CPU-only proof rerun against the produced h10 step-5999 carry checkpoint.
- JSON validation.
- `git diff --check`.

Key finding:

The mismatch is already present before current-step physics/RK at the earliest
available input/reference theta surface. `T_OLD` versus JAX prestep theta has
max_abs `6.218735851548047` and RMSE `4.638818160588427`. The proof-local RK
mirror agrees with the existing pre-halo helper (`max_abs=0.0`), so this is not
a proof-wrapper artifact. `MU_OLD` context is also divergent (`267.01919069732367`
max_abs), while explicit WRF input-boundary `P/PB/MUB` are not available in the
current artifacts.

Roadmap effect:

Do not start with final `small_step_finish`, post-RK refresh, or history-source
mapping as the first production fix. The next grid-parity sprint must expose
the explicit step-6000 pre-RK input boundary for both WRF and JAX over
`T/P/PB/MU/MUB`.

Next decision:

Open a WRF/JAX input-boundary emitter or hook sprint. Production dycore edits
remain blocked until that sprint distinguishes bad JAX checkpoint/prestep carry
generation from a specific prior-step update or boundary/tendency packaging
fault.
