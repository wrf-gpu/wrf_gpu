# V0.14 Early-Step Same-Input Discriminator

Verdict: `EARLY_STEP_DISCRIMINATOR_BLOCKED_CPU_REALCASE_LOADER_GPU_ONLY_NO_CANDIDATE_WRF_PREHALO_TRUTH_NO_SAME_INPUT_CARRY_CONTRACT`.

No strict same-input comparison ran. The proof avoids weak WRF-output, JAX-vs-JAX, and mixed-source comparisons.

## Result

- Candidate steps covered: `[1, 60, 600, 3000, 5999]`.
- CPU real-case loader probe: `BLOCKED_GPU_ONLY_STATE_ZERO`.
- Candidate WRF pre-halo surfaces found: `0`.
- Existing step-6000 surfaces are non-candidate and patch-only: `True`.

## Consolidated Blockers

- `CPU_REALCASE_REPLAY_LOADER_GPU_ONLY`: A CPU-compatible real-case wrfinput loader or checkpoint reader that constructs State, Tendencies, BaseState/metrics, OperationalNamelist, and initial OperationalCarry without GPU allocation.
- `NO_CANDIDATE_WRF_POST_RK_PRE_HALO_TRUTH`: Disposable CPU-WRF candidate-step surface at post_after_all_rk_steps_pre_halo, or an exactly named equivalent boundary, covering the required dynamic fields and active moisture.
- `NO_SAME_INPUT_CARRY_SEQUENCE`: For each candidate start, a WRF-controlled or exactly reproduced same-input carry, including promoted RK/acoustic scratch leaves and live d01->d02 boundary forcing.
- `MISSING_REQUIRED_FIELD_SURFACE_SCHEMA`: A WRF/JAX field map with units, staggering, counts, first/worst index semantics, and static/base field exclusion from headline dynamic selectors.

Next decision: build one CPU-compatible same-input contract, not another single-blocker ladder step.
