# Memory Patch

Reviewer Status: pending memory updated, stable memory untouched.

This sprint refined the prior dynamic-layer pending memory. The durable lesson
is not merely that `post_small_step_finish` is not history-aligned; the usable
green WRF compare target is now known:

- immediately after `dyn_em/solve_em.F::after_all_rk_steps`
- before RK halo exchanges
- with final `calc_p_rho_phi` closing `P`
- with `after_all_rk_steps` closing `V/W`

Updated pending memory:

- `.agent/memory/pending/2026-06-09-v014-dynamic-layer-boundary.md`

Stable memory must wait for independent review and for the JAX wrapper sprint to
confirm this surface is practical as a repeated compare target.
