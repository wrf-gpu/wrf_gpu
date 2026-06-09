# V0.14 Early-Step Discriminator Blocked

Date: 2026-06-09

`proofs/v014/early_step_discriminator.*` closed fail-closed with verdict
`EARLY_STEP_DISCRIMINATOR_BLOCKED_CPU_REALCASE_LOADER_GPU_ONLY_NO_CANDIDATE_WRF_PREHALO_TRUTH_NO_SAME_INPUT_CARRY_CONTRACT`.

The sprint covered steps `1`, `60`, `600`, `3000`, and `5999` and did not run a
weak comparison. Common blockers:

- CPU-only proof cannot load the real d02 case through production loaders
  because `State.zeros` requires a visible GPU.
- No candidate-step WRF post-RK/pre-halo full-field truth surface exists.
- No WRF-controlled same-input `OperationalCarry` sequence exists for the
  candidate starts.
- A frozen WRF/JAX field/staggering schema is still missing.

Next: build same-input comparison tooling first, then rerun the discriminator.
Do not start a source-changing dycore/runtime fix from this proof alone.
